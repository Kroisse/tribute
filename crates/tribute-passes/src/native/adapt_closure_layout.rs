//! Adapt semantic closure allocations to the native closure layout.
//!
//! A semantic closure layout, the base or a capture layout, starts with a
//! convention table reference. The native backend stores the address of the
//! table there instead and keeps the captures. This step rewrites
//! `adt.struct_new` and `adt.struct_get` on the semantic layouts, and the
//! `tribute_rtti.layout` declarations naming them, to the native layouts.
//!
//! It also lowers the convention table operations: each distinct entry list
//! becomes one read-only `clif.data` of pointer-sized entry addresses written
//! by `clif.func_reloc`, `tribute_rt.convention_table` its address, and
//! `tribute_rt.convention_entry` a load of one entry.
//!
//! It runs after native ownership planning and RC materialization, which
//! read the semantic layout, and before `func_to_clif`.

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use tribute_ir::dialect::adt;
use tribute_ir::dialect::tribute_rt;
use tribute_ir::dialect::tribute_rtti;
use trunk_ir::OpName;
use trunk_ir::context::{BlockData, IrContext, RegionData};
use trunk_ir::dialect::{clif, core};
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{OpRef, TypeRef};
use trunk_ir::rewrite::{
    Module, PatternApplicator, PatternRewriter, RewritePattern, TypeConverter,
};
use trunk_ir::smallvec::smallvec;
use trunk_ir::types::{Attribute, TypeDataBuilder};
use trunk_ir::walk::{WalkAction, walk_op};
use trunk_ir::{Symbol, SymbolPath};
use trunk_ir_cranelift_backend::passes::cf_to_clif::rebuild_op_as;

use crate::closure_lower::{
    closure_base_layout, closure_captures_layout, is_closure_layout, is_closure_struct_type_ref,
};

/// Rewrite semantic closure allocations and their RTTI declarations to the
/// native closure layouts.
pub(crate) fn lower(ctx: &mut IrContext, module: Module) {
    let tables = declare_convention_tables(ctx, module);
    PatternApplicator::new(TypeConverter::new())
        .add_pattern(ClosureStructAdaptPattern)
        .add_pattern(ConventionTablePattern { tables })
        .add_pattern(ConventionEntryPattern)
        .apply_partial(ctx, module);

    let layouts = module
        .ops(ctx)
        .iter()
        .copied()
        .filter_map(|op| tribute_rtti::Layout::from_op(ctx, op).ok())
        .collect::<Vec<_>>();
    for layout in layouts {
        let ty = layout.r#type(ctx);
        if let Some(native) = native_closure_layout(ctx, ty) {
            layout.set_type(ctx, native);
        }
    }
}

/// The native layout of the semantic closure layout `ty`: the same layout
/// with the unmanaged address of the convention table in place of the
/// semantic reference. `None` when `ty` is not a semantic closure layout.
fn native_closure_layout(ctx: &mut IrContext, ty: TypeRef) -> Option<TypeRef> {
    if !is_closure_layout(ctx, ty) {
        return None;
    }
    let ptr_ty = core::Ptr::params().intern(ctx).as_type_ref();
    let fields: Vec<TypeRef> = adt::Struct::from_type_ref(ctx, ty)?
        .fields(ctx)
        .map(|(_, field)| field)
        .collect();
    let (&entries, captures) = fields.split_first()?;
    if entries == ptr_ty {
        return None;
    }
    let base = closure_base_layout(ctx, ptr_ty);
    Some(if is_closure_struct_type_ref(ctx, ty) {
        base
    } else {
        closure_captures_layout(ctx, base, captures)
    })
}

/// Prefix of the convention table data symbols.
const CONVENTION_TABLE_PREFIX: &str = "__tribute_convention_table_";
const POINTER_SIZE: u32 = 8;

/// Declare one read-only `clif.data` per distinct entry list of the module's
/// `tribute_rt.convention_table` operations, in their order of appearance,
/// under symbols no module-level operation already uses.
fn declare_convention_tables(
    ctx: &mut IrContext,
    module: Module,
) -> HashMap<Vec<SymbolPath>, Symbol> {
    let mut tables = HashMap::default();
    let Some(module_block) = module.first_block(ctx) else {
        return tables;
    };
    let mut lists = Vec::new();
    let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
        if let Ok(table) = tribute_rt::ConventionTable::from_op(ctx, op) {
            lists.push(table.entries(ctx).cloned().collect::<Vec<_>>());
        }
        std::ops::ControlFlow::Continue(WalkAction::Advance)
    });
    let taken: HashSet<Symbol> = ctx
        .block(module_block)
        .ops
        .iter()
        .filter_map(|&op| ctx.op_attrs(op).get_str("sym_name").map(Symbol::new))
        .collect();
    let location = ctx.op(module.op()).location;
    let mut next = 0u32;
    for entries in lists {
        if tables.contains_key(&entries) {
            continue;
        }
        let symbol = loop {
            let candidate = Symbol::new(&format!("{CONVENTION_TABLE_PREFIX}{next}"));
            next += 1;
            if !taken.contains(&candidate) {
                break candidate;
            }
        };
        let block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        for (index, entry) in entries.iter().enumerate() {
            let reloc = clif::FuncReloc::operands()
                .offset(index as u32 * POINTER_SIZE)
                .func(entry.clone())
                .build(ctx, location);
            ctx.push_op(block, reloc.op_ref());
        }
        let relocs = ctx.create_region(RegionData {
            location,
            blocks: smallvec![block],
            parent_op: None,
        });
        let bytes = vec![0u8; entries.len() * POINTER_SIZE as usize];
        let data = clif::Data::operands()
            .sym_name(ctx.intern_symbol_text(&symbol))
            .bytes(bytes.as_slice().into())
            .align(POINTER_SIZE)
            .regions(Some(relocs))
            .build(ctx, location);
        ctx.push_op(module_block, data.op_ref());
        tables.insert(entries, symbol);
    }
    tables
}

/// Pattern: `tribute_rt.convention_table` → the address of its table.
struct ConventionTablePattern {
    tables: HashMap<Vec<SymbolPath>, Symbol>,
}

impl RewritePattern for ConventionTablePattern {
    fn roots(&self) -> Option<&[OpName]> {
        Some(&[tribute_rt::ConventionTable::NAME])
    }

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(table) = tribute_rt::ConventionTable::from_op(ctx, op) else {
            return false;
        };
        let entries: Vec<_> = table.entries(ctx).cloned().collect();
        let symbol = self.tables[&entries].clone();
        let ptr_ty = core::Ptr::params().intern(ctx).as_type_ref();
        let address = clif::SymbolAddr::operands()
            .sym(SymbolPath::new([symbol]))
            .results(ptr_ty)
            .build(ctx, ctx.op(op).location);
        rewriter.replace_op(address.op_ref());
        true
    }
}

/// Pattern: `tribute_rt.convention_entry` → a load of the entry's address.
struct ConventionEntryPattern;

impl RewritePattern for ConventionEntryPattern {
    fn roots(&self) -> Option<&[OpName]> {
        Some(&[tribute_rt::ConventionEntry::NAME])
    }

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(entry) = tribute_rt::ConventionEntry::from_op(ctx, op) else {
            return false;
        };
        let offset = entry.index(ctx) * POINTER_SIZE;
        let table = entry.table(ctx);
        let i64_ty = TypeDataBuilder::new("core.i64").intern(ctx);
        let load = clif::Load::operands(table)
            .offset(i32::try_from(offset).expect("convention table offset fits i32"))
            .results(i64_ty)
            .build(ctx, ctx.op(op).location);
        rewriter.replace_op(load.op_ref());
        true
    }
}

/// Pattern: adapt closure layout operations to the native layouts.
struct ClosureStructAdaptPattern;

impl RewritePattern for ClosureStructAdaptPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if let Ok(struct_new) = adt::StructNew::from_op(ctx, op) {
            let Some(native_ty) = native_closure_layout(ctx, struct_new.r#type(ctx)) else {
                return false;
            };
            let new_op = rebuild_op_as(ctx, op, "adt.struct_new");
            ctx.op_attrs_mut(new_op)
                .insert("type", Attribute::Type(native_ty));
            let Some(&original_ty) = ctx.op_result_types(new_op).first() else {
                rewriter.replace_op(new_op);
                return true;
            };
            ctx.set_op_result_type(new_op, 0, native_ty);
            // Users still declare the semantic closure type, so the
            // allocation is cast back to it; type conversion reconciles the cast.
            let location = ctx.op(op).location;
            let allocated = ctx.op_result(new_op, 0);
            let restored = core::UnrealizedConversionCast::operands(allocated)
                .results(original_ty)
                .build(ctx, location);
            rewriter.insert_op(new_op);
            rewriter.insert_op(restored.op_ref());
            rewriter.erase_op(vec![restored.result(ctx)]);
            return true;
        }

        if let Ok(struct_get) = adt::StructGet::from_op(ctx, op) {
            let Some(native_ty) = native_closure_layout(ctx, struct_get.r#type(ctx)) else {
                return false;
            };
            let new_op = rebuild_op_as(ctx, op, "adt.struct_get");
            ctx.op_attrs_mut(new_op)
                .insert("type", Attribute::Type(native_ty));
            // The convention table address is read by `convention_entry`;
            // captures keep their semantic types.
            if struct_get.field(ctx) == 0 {
                let ptr_ty = core::Ptr::params().intern(ctx).as_type_ref();
                ctx.set_op_result_type(new_op, 0, ptr_ty);
            }
            rewriter.replace_op(new_op);
            return true;
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;

    #[test]
    fn adapts_closure_allocations_and_their_rtti_declaration() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !Base = adt.struct<_closure(entries: tribute_rt.convention_table_ref), {layout = "closure"}>
  !Captures = adt.struct<_closure_captures(entries: tribute_rt.convention_table_ref, _0: tribute_rt.anyref), {closure_base = !Base, layout = "closure_captures"}>
  !Other = adt.struct<Other(value: tribute_rt.anyref)>
  tribute_rtti.layout {type = !Captures, index = 32, fields = ["raw", "dynamic"]}
  tribute_rtti.layout {type = !Base, index = 33, fields = ["raw"]}
  tribute_rtti.layout {type = !Other, index = 34, fields = ["dynamic"]}
  func.func @make(%table: core.i32, %env: tribute_rt.anyref) -> !Captures {
    %closure = adt.struct_new %table, %env {type = !Captures} : !Captures
    %loaded = adt.struct_get %closure {field = 1, type = !Captures} : tribute_rt.anyref
    func.return %closure
  }
}"#,
        );
        let alias = |ctx: &IrContext, name: &str| {
            ctx.type_aliases()
                .iter()
                .find_map(|(alias, ty)| (*alias == name).then_some(*ty))
                .expect("fixture alias")
        };
        let [captures, other] = ["Captures", "Other"].map(|name| alias(&ctx, name));

        lower(&mut ctx, module);

        let ptr_ty = core::Ptr::params().intern(&mut ctx).as_type_ref();
        let anyref = tribute_ir::dialect::tribute_rt::Anyref::params()
            .intern(&mut ctx)
            .as_type_ref();
        let native_base = closure_base_layout(&mut ctx, ptr_ty);
        let native_captures = closure_captures_layout(&mut ctx, native_base, &[anyref]);
        assert_ne!(native_captures, captures);
        let layouts = module
            .ops(&ctx)
            .iter()
            .copied()
            .filter_map(|op| tribute_rtti::Layout::from_op(&ctx, op).ok())
            .collect::<Vec<_>>();
        assert_eq!(layouts[0].r#type(&ctx), native_captures);
        assert_eq!(
            layouts[0].field_kinds(&ctx),
            [
                tribute_rtti::FieldKind::Raw,
                tribute_rtti::FieldKind::Dynamic
            ]
        );
        assert_eq!(layouts[1].r#type(&ctx), native_base);
        assert_eq!(layouts[2].r#type(&ctx), other);
        let output = print_module(&ctx, module.op());
        assert!(!output.contains("type = !Captures}"), "{output}");
    }

    #[test]
    fn convention_tables_are_shared_per_entry_list_under_fresh_symbols() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  clif.data {align = 1, bytes = b"\0", sym_name = "__tribute_convention_table_0"}
  func.func @make() -> core.nil {
    %a = tribute_rt.convention_table {entries = [@first]} : tribute_rt.convention_table_ref
    %b = tribute_rt.convention_table {entries = [@second, @first]} : tribute_rt.convention_table_ref
    %c = tribute_rt.convention_table {entries = [@first]} : tribute_rt.convention_table_ref
    %d = tribute_rt.convention_entry %b {index = 1} : core.i32
    func.return
  }
}"#,
        );
        lower(&mut ctx, module);
        let output = print_module(&ctx, module.op());
        assert_eq!(output.matches("clif.data").count(), 3, "{output}");
        assert_eq!(
            output
                .matches("clif.symbol_addr {sym = @__tribute_convention_table_1}")
                .count(),
            2,
            "{output}"
        );
        assert!(
            output.contains("clif.symbol_addr {sym = @__tribute_convention_table_2}"),
            "{output}"
        );
        assert!(
            output.contains("clif.func_reloc {func = @first, offset = 8}"),
            "{output}"
        );
        assert!(output.contains("clif.load %"), "{output}");
        assert!(output.contains("{offset = 8} : core.i64"), "{output}");
        assert!(!output.contains("tribute_rt."), "{output}");
    }

    /// Adapt closures, then lower functions as the native pipeline does.
    fn adapt_then_lower_func(ir: &str) -> String {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, ir);
        lower(&mut ctx, module);
        let (type_converter, _) = crate::native::type_converter::native_type_converter(&mut ctx);
        trunk_ir_cranelift_backend::passes::func_to_clif::lower(&mut ctx, module, type_converter)
            .expect("func_to_clif");
        let (type_converter, _) = crate::native::type_converter::native_type_converter(&mut ctx);
        PatternApplicator::new(type_converter)
            .add_pattern(trunk_ir::conversion::UnrealizedCastConversionPattern)
            .apply_partial(&mut ctx, module);
        let core_module = core::Module::from_op(&ctx, module.op()).unwrap();
        trunk_ir::pass::Pass::run(
            &mut trunk_ir::conversion::ReconcileUnrealizedCasts,
            &mut ctx,
            core_module,
            &mut trunk_ir::analysis::AnalysisCache::new(),
        )
        .unwrap();
        print_module(&ctx, module.op())
    }

    #[test]
    fn test_closure_struct_adaptation() {
        let result = adapt_then_lower_func(
            r#"core.module @test {
  !Base = adt.struct<_closure(entries: tribute_rt.convention_table_ref), {layout = "closure"}>
  !Captures = adt.struct<_closure_captures(entries: tribute_rt.convention_table_ref, _0: core.ptr), {closure_base = !Base, layout = "closure_captures"}>
  func.func @test_fn() -> core.i32 {
    %0 = tribute_rt.convention_table {entries = [@lifted_fn]} : tribute_rt.convention_table_ref
    %1 = mem.null : core.ptr
    %2 = adt.struct_new %0, %1 {type = !Captures} : !Captures
    %3 = adt.struct_get %2 {field = 0, type = !Captures} : tribute_rt.convention_table_ref
    %4 = adt.struct_get %2 {field = 1, type = !Captures} : core.ptr
    %6 = tribute_rt.convention_entry %3 {index = 0} : core.i32
    %5 = func.call_indirect %6, %4 {signature = func.func_sig<(core.ptr) -> core.i32>} : core.i32
    func.return %5
  }
}"#,
        );
        insta::assert_snapshot!(result);
    }

    #[test]
    fn test_closure_struct_anyref_adaptation() {
        let result = adapt_then_lower_func(
            r#"core.module @test {
  !Base = adt.struct<_closure(entries: tribute_rt.convention_table_ref), {layout = "closure"}>
  !Captures = adt.struct<_closure_captures(entries: tribute_rt.convention_table_ref, _0: tribute_rt.anyref), {closure_base = !Base, layout = "closure_captures"}>
  func.func @test_fn(%1: tribute_rt.anyref) -> core.i32 {
    %0 = tribute_rt.convention_table {entries = [@lifted_fn]} : tribute_rt.convention_table_ref
    %2 = adt.struct_new %0, %1 {type = !Captures} : !Captures
    %3 = adt.struct_get %2 {field = 0, type = !Captures} : tribute_rt.convention_table_ref
    %4 = adt.struct_get %2 {field = 1, type = !Captures} : tribute_rt.anyref
    %6 = tribute_rt.convention_entry %3 {index = 0} : core.i32
    %5 = func.call_indirect %6, %4 {signature = func.func_sig<(core.ptr) -> core.i32>} : core.i32
    func.return %5
  }
}"#,
        );
        insta::assert_snapshot!(result);
    }
}
