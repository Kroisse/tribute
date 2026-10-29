//! Lower tribute_rt dialect operations to wasm dialect.
//!
//! This pass converts boxing/unboxing operations to their wasm equivalents:
//! - `tribute_rt.box_int` -> `wasm.ref_i31` (i32 -> i31ref)
//! - `tribute_rt.unbox_int` -> `wasm.ref_cast` (i31ref) + `wasm.i31_get_s`
//! - `tribute_rt.box_nat` -> `wasm.ref_i31` (i32 -> i31ref)
//! - `tribute_rt.unbox_nat` -> `wasm.ref_cast` (i31ref) + `wasm.i31_get_u`
//! - `tribute_rt.box_float` -> `adt.struct_new` (f64 -> BoxedF64 struct)
//! - `tribute_rt.unbox_float` -> `adt.ref_cast` (BoxedF64) + `adt.struct_get`
//! - `tribute_rt.box_bool` -> `wasm.ref_i31` (i32 -> i31ref)
//! - `tribute_rt.unbox_bool` -> `wasm.ref_cast` (i31ref) + `wasm.i31_get_u`
//! - `tribute_rt.convention_table` -> `wasm.i32_const` of the table's first
//!   slot in the function table
//! - `tribute_rt.convention_entry` -> that index plus the entry's position
//!
//! Each distinct entry list takes consecutive slots appended to the element
//! segment of function table 0, which `func_to_wasm` declares for
//! `func.constant`; the table and segment are declared here when it did not.
//!
//! ## Type Mappings
//!
//! - `tribute_rt.int` -> `core.i32`
//! - `tribute_rt.nat` -> `core.i32`
//! - `tribute_rt.float` -> `core.f64`
//! - `tribute_rt.bool` -> `core.i32`
//! - `tribute_rt.intref` -> `wasm.i31ref`
//! - `tribute_rt.anyref` -> `wasm.anyref`

use rustc_hash::FxHashMap as HashMap;
use tribute_ir::dialect::adt;
use tribute_ir::dialect::tribute_rt;
use trunk_ir::SymbolPath;
use trunk_ir::context::{BlockData, IrContext, RegionData};
use trunk_ir::dialect::wasm as wasm_dialect;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{BlockRef, OpRef, TypeRef, ValueRef};
use trunk_ir::rewrite::{
    Module, OpRewritePattern, PatternApplicator, PatternRewriter, TypeConverter,
};
use trunk_ir::smallvec::smallvec;
use trunk_ir::types::{Attribute, AttributeMap, Location, TypeDataBuilder};
use trunk_ir::walk::{WalkAction, walk_op};

/// Helper to create arena type refs for common types.
fn i32_type(ctx: &mut IrContext) -> TypeRef {
    TypeDataBuilder::new("core.i32").intern(ctx)
}

fn f64_type(ctx: &mut IrContext) -> TypeRef {
    TypeDataBuilder::new("core.f64").intern(ctx)
}

/// Get the BoxedF64 struct type:
/// `adt.struct<_BoxedF64(value: core.f64), {layout = "boxed_f64"}>`
fn boxed_f64_type(ctx: &mut IrContext) -> TypeRef {
    let f64_ty = f64_type(ctx);
    let mut attrs = AttributeMap::new();
    let layout = ctx.string_attr(tribute_ir::runtime_layout::BOXED_F64);
    attrs
        .view_mut(ctx)
        .insert(tribute_ir::runtime_layout::LAYOUT_ATTR, layout);
    adt::Struct::from_fields(ctx, "_BoxedF64", [("value", f64_ty)], attrs).as_type_ref()
}

/// Create i31 unbox operations (ref_cast to i31ref + i31_get_s/u).
///
/// Returns `(prefix_ops, final_op)` where prefix_ops should be inserted before
/// and final_op is the replacement.
fn create_i31_unbox(
    ctx: &mut IrContext,
    location: Location,
    value: ValueRef,
    is_signed: bool,
) -> (Vec<OpRef>, OpRef) {
    let i31ref_ty = wasm_dialect::I31ref::params().intern(ctx).as_type_ref();
    let i32_ty = i32_type(ctx);

    // Cast anyref to i31ref first (abstract type, no type_idx needed)
    let cast_op = wasm_dialect::RefCast::operands(value)
        .target_type(i31ref_ty)
        .type_idx(None)
        .results(i31ref_ty)
        .build(ctx, location);
    let cast_result = cast_op.result(ctx);

    // Extract value: signed or unsigned
    let get_op_ref = if is_signed {
        wasm_dialect::I31GetS::operands(cast_result)
            .results(i32_ty)
            .build(ctx, location)
            .op_ref()
    } else {
        wasm_dialect::I31GetU::operands(cast_result)
            .results(i32_ty)
            .build(ctx, location)
            .op_ref()
    };

    (vec![cast_op.op_ref()], get_op_ref)
}

/// Lower tribute_rt dialect to wasm dialect.
pub(crate) fn lower(ctx: &mut IrContext, module: Module) {
    let tables = place_convention_tables(ctx, module);
    let applicator = PatternApplicator::new(TypeConverter::new())
        .add_pattern(ConventionTablePattern { tables })
        .add_pattern(ConventionEntryPattern)
        .add_pattern(BoxIntPattern)
        .add_pattern(UnboxIntPattern)
        .add_pattern(BoxNatPattern)
        .add_pattern(UnboxNatPattern)
        .add_pattern(BoxFloatPattern)
        .add_pattern(UnboxFloatPattern)
        .add_pattern(BoxBoolPattern)
        .add_pattern(UnboxBoolPattern);
    applicator.apply_partial(ctx, module);
}

/// Give each distinct entry list of the module's `tribute_rt.convention_table`
/// operations consecutive slots at the end of function table 0, in their
/// order of appearance, and return the first slot of each list.
fn place_convention_tables(ctx: &mut IrContext, module: Module) -> HashMap<Vec<SymbolPath>, u32> {
    let mut tables = HashMap::default();
    let mut lists = Vec::new();
    let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
        if let Ok(table) = tribute_rt::ConventionTable::from_op(ctx, op) {
            let entries: Vec<_> = table.entries(ctx).cloned().collect();
            if !lists.contains(&entries) {
                lists.push(entries);
            }
        }
        std::ops::ControlFlow::Continue(WalkAction::Advance)
    });
    if lists.is_empty() {
        return tables;
    }
    let Some((table, funcs)) = function_table(ctx, module) else {
        return tables;
    };
    let location = ctx.op(module.op()).location;
    let funcref_ty = wasm_dialect::Funcref::params().intern(ctx).as_type_ref();
    let mut next = u32::try_from(ctx.block(funcs).ops.len()).expect("function table fits u32");
    for entries in lists {
        let start = next;
        for entry in &entries {
            let reference = wasm_dialect::RefFunc::operands()
                .func_name(entry.clone())
                .results(funcref_ty)
                .build(ctx, location);
            ctx.push_op(funcs, reference.op_ref());
            next += 1;
        }
        tables.insert(entries, start);
    }
    let mut attrs = ctx.op_attrs_mut(table.op_ref());
    attrs.insert("min", Attribute::Int(next.into()));
    attrs.insert("max", Attribute::Int(next.into()));
    tables
}

/// Function table 0 and the block of its element segment's functions,
/// declared at the start of the module when absent.
fn function_table(ctx: &mut IrContext, module: Module) -> Option<(wasm_dialect::Table, BlockRef)> {
    let module_block = module.first_block(ctx)?;
    let ops = ctx.block(module_block).ops.clone();
    let table = ops
        .iter()
        .find_map(|&op| wasm_dialect::Table::from_op(ctx, op).ok());
    let elem = ops
        .iter()
        .filter_map(|&op| wasm_dialect::Elem::from_op(ctx, op).ok())
        .find(|elem| elem.table(ctx).unwrap_or(0) == 0 && elem.offset(ctx).unwrap_or(0) == 0);
    if let (Some(table), Some(elem)) = (table, elem) {
        let funcs = ctx.region(elem.funcs(ctx)).blocks[0];
        return Some((table, funcs));
    }
    let location = ctx.op(module.op()).location;
    let table = wasm_dialect::Table::operands()
        .reftype("funcref")
        .min(0)
        .max(Some(0))
        .build(ctx, location);
    let funcs = ctx.create_block(BlockData {
        location,
        args: vec![],
        ops: smallvec![],
        parent_region: None,
    });
    let region = ctx.create_region(RegionData {
        location,
        blocks: smallvec![funcs],
        parent_op: None,
    });
    let elem = wasm_dialect::Elem::operands()
        .table(Some(0))
        .offset(Some(0))
        .regions(region)
        .build(ctx, location);
    if let Some(&first) = ops.first() {
        ctx.insert_op_before(module_block, first, table.op_ref());
        ctx.insert_op_before(module_block, first, elem.op_ref());
    } else {
        ctx.push_op(module_block, table.op_ref());
        ctx.push_op(module_block, elem.op_ref());
    }
    Some((table, funcs))
}

/// Pattern for `tribute_rt.convention_table` -> `wasm.i32_const` of the
/// table's first slot.
struct ConventionTablePattern {
    tables: HashMap<Vec<SymbolPath>, u32>,
}

impl OpRewritePattern for ConventionTablePattern {
    type Op = tribute_rt::ConventionTable;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        table: tribute_rt::ConventionTable,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let op = table.op_ref();
        let entries: Vec<_> = table.entries(ctx).cloned().collect();
        let start = self.tables[&entries];
        let i32_ty = i32_type(ctx);
        let index = wasm_dialect::I32Const::operands()
            .value(i32::try_from(start).expect("function table index fits i32"))
            .results(i32_ty)
            .build(ctx, ctx.op(op).location);
        rewriter.replace_op(index.op_ref());
        true
    }
}

/// Pattern for `tribute_rt.convention_entry` -> the function table index of
/// the entry: the table's first slot plus the entry's position.
struct ConventionEntryPattern;

impl OpRewritePattern for ConventionEntryPattern {
    type Op = tribute_rt::ConventionEntry;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        entry: tribute_rt::ConventionEntry,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let op = entry.op_ref();
        let table = entry.table(ctx);
        let position = entry.index(ctx);
        if position == 0 {
            rewriter.erase_op(vec![table]);
            return true;
        }
        let location = ctx.op(op).location;
        let i32_ty = i32_type(ctx);
        let offset = wasm_dialect::I32Const::operands()
            .value(i32::try_from(position).expect("convention entry index fits i32"))
            .results(i32_ty)
            .build(ctx, location);
        let index = wasm_dialect::I32Add::operands(table, offset.result(ctx)).build(ctx, location);
        rewriter.insert_op(offset.op_ref());
        rewriter.replace_op(index.op_ref());
        true
    }
}

/// Pattern for `tribute_rt.box_int` -> `wasm.ref_i31`
///
/// Boxing converts an unboxed i32 to an i31ref via `wasm.ref_i31`.
struct BoxIntPattern;

impl OpRewritePattern for BoxIntPattern {
    type Op = tribute_rt::BoxInt;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        box_op: tribute_rt::BoxInt,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let op = box_op.op_ref();
        let location = ctx.op(op).location;
        let value = box_op.value(ctx);

        let i31ref_ty = wasm_dialect::I31ref::params().intern(ctx).as_type_ref();

        // wasm.ref_i31: i32 -> i31ref
        let new_op = wasm_dialect::RefI31::operands(value)
            .results(i31ref_ty)
            .build(ctx, location);

        rewriter.replace_op(new_op.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "BoxIntPattern"
    }
}

/// Pattern for `tribute_rt.unbox_int` -> `wasm.ref_cast` + `wasm.i31_get_s`
///
/// Unboxing extracts the signed i32 from an i31ref.
struct UnboxIntPattern;

impl OpRewritePattern for UnboxIntPattern {
    type Op = tribute_rt::UnboxInt;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        unbox_op: tribute_rt::UnboxInt,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let op = unbox_op.op_ref();
        let location = ctx.op(op).location;
        let (prefix_ops, final_op) = create_i31_unbox(ctx, location, unbox_op.value(ctx), true);
        for prefix in prefix_ops {
            rewriter.insert_op(prefix);
        }
        rewriter.replace_op(final_op);
        true
    }

    fn name(&self) -> &'static str {
        "UnboxIntPattern"
    }
}

/// Pattern for `tribute_rt.box_nat` -> `wasm.ref_i31`
///
/// Boxing converts an unboxed nat (u32) to an i31ref.
struct BoxNatPattern;

impl OpRewritePattern for BoxNatPattern {
    type Op = tribute_rt::BoxNat;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        box_op: tribute_rt::BoxNat,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let op = box_op.op_ref();
        let location = ctx.op(op).location;
        let value = box_op.value(ctx);
        let i31ref_ty = wasm_dialect::I31ref::params().intern(ctx).as_type_ref();

        let new_op = wasm_dialect::RefI31::operands(value)
            .results(i31ref_ty)
            .build(ctx, location);
        rewriter.replace_op(new_op.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "BoxNatPattern"
    }
}

/// Pattern for `tribute_rt.unbox_nat` -> `wasm.ref_cast` + `wasm.i31_get_u`
///
/// Unboxing extracts the unsigned i32 from an i31ref.
struct UnboxNatPattern;

impl OpRewritePattern for UnboxNatPattern {
    type Op = tribute_rt::UnboxNat;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        unbox_op: tribute_rt::UnboxNat,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let op = unbox_op.op_ref();
        let location = ctx.op(op).location;
        let (prefix_ops, final_op) = create_i31_unbox(ctx, location, unbox_op.value(ctx), false);
        for prefix in prefix_ops {
            rewriter.insert_op(prefix);
        }
        rewriter.replace_op(final_op);
        true
    }

    fn name(&self) -> &'static str {
        "UnboxNatPattern"
    }
}

/// Pattern for `tribute_rt.box_float` -> `adt.struct_new(BoxedF64, anyref)`
///
/// Boxing converts an f64 to a BoxedF64 struct typed as anyref via
/// `adt.struct_new` with `anyref_ty` as the result type.
struct BoxFloatPattern;

impl OpRewritePattern for BoxFloatPattern {
    type Op = tribute_rt::BoxFloat;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        box_op: tribute_rt::BoxFloat,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let op = box_op.op_ref();
        let location = ctx.op(op).location;
        let value = box_op.value(ctx);
        let anyref_ty = wasm_dialect::Anyref::params().intern(ctx).as_type_ref();
        let boxed_f64_ty = boxed_f64_type(ctx);

        // adt.struct_new creates BoxedF64 struct with the f64 value
        let struct_op = adt::StructNew::operands(vec![value])
            .r#type(boxed_f64_ty)
            .results(anyref_ty)
            .build(ctx, location);

        rewriter.replace_op(struct_op.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "BoxFloatPattern"
    }
}

/// Pattern for `tribute_rt.unbox_float` -> `adt.ref_cast(BoxedF64)` + `adt.struct_get`
///
/// Unboxing extracts the f64 from a BoxedF64 struct.
struct UnboxFloatPattern;

impl OpRewritePattern for UnboxFloatPattern {
    type Op = tribute_rt::UnboxFloat;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        unbox_op: tribute_rt::UnboxFloat,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let op = unbox_op.op_ref();
        let location = ctx.op(op).location;
        let value = unbox_op.value(ctx);

        let boxed_f64_ty = boxed_f64_type(ctx);
        let f64_ty = f64_type(ctx);

        // Cast anyref to BoxedF64 struct first
        let cast_op = adt::RefCast::operands(value)
            .r#type(boxed_f64_ty)
            .results(boxed_f64_ty)
            .build(ctx, location);
        let cast_result = cast_op.result(ctx);

        // adt.struct_get extracts field 0 (the f64 value) from BoxedF64
        let get_op = adt::StructGet::operands(cast_result)
            .r#type(boxed_f64_ty)
            .field(0)
            .results(f64_ty)
            .build(ctx, location);

        rewriter.insert_op(cast_op.op_ref());
        rewriter.replace_op(get_op.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "UnboxFloatPattern"
    }
}

/// Pattern for `tribute_rt.box_bool` -> `wasm.ref_i31`
///
/// Boxing converts an i32 boolean to an i31ref.
struct BoxBoolPattern;

impl OpRewritePattern for BoxBoolPattern {
    type Op = tribute_rt::BoxBool;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        box_op: tribute_rt::BoxBool,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let op = box_op.op_ref();
        let location = ctx.op(op).location;
        let value = box_op.value(ctx);
        let i31ref_ty = wasm_dialect::I31ref::params().intern(ctx).as_type_ref();

        let new_op = wasm_dialect::RefI31::operands(value)
            .results(i31ref_ty)
            .build(ctx, location);
        rewriter.replace_op(new_op.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "BoxBoolPattern"
    }
}

/// Pattern for `tribute_rt.unbox_bool` -> `wasm.ref_cast` + `wasm.i31_get_u`
///
/// Unboxing extracts the boolean (0 or 1) from an i31ref.
struct UnboxBoolPattern;

impl OpRewritePattern for UnboxBoolPattern {
    type Op = tribute_rt::UnboxBool;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        unbox_op: tribute_rt::UnboxBool,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let op = unbox_op.op_ref();
        let location = ctx.op(op).location;
        // Use unsigned extraction since bool is 0 or 1
        let (prefix_ops, final_op) = create_i31_unbox(ctx, location, unbox_op.value(ctx), false);
        for prefix in prefix_ops {
            rewriter.insert_op(prefix);
        }
        rewriter.replace_op(final_op);
        true
    }

    fn name(&self) -> &'static str {
        "UnboxBoolPattern"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::walk::{WalkAction, walk_region};

    #[test]
    fn convention_tables_take_consecutive_slots_after_the_function_table() {
        let mut ctx = IrContext::new();
        let module = trunk_ir::parser::parse_test_module(
            &mut ctx,
            r#"core.module @test {
  wasm.table {reftype = "funcref", min = 1, max = 1}
  wasm.elem {table = 0, offset = 0} {
    %r = wasm.ref_func {func_name = @constant} : wasm.funcref
  }
  func.func @f() -> core.nil {
    %a = tribute_rt.convention_table {entries = [@first, @second]} : tribute_rt.convention_table_ref
    %b = tribute_rt.convention_table {entries = [@third]} : tribute_rt.convention_table_ref
    %c = tribute_rt.convention_table {entries = [@first, @second]} : tribute_rt.convention_table_ref
    %d = tribute_rt.convention_entry %a {index = 0} : core.i32
    %e = tribute_rt.convention_entry %c {index = 1} : core.i32
    func.return
  }
}"#,
        );
        lower(&mut ctx, module);
        let ir = trunk_ir::printer::print_module(&ctx, module.op());
        assert!(
            ir.contains(r#"wasm.table {max = 4, min = 4, reftype = "funcref"}"#),
            "{ir}"
        );
        let slots: Vec<_> = ["@constant", "@first", "@second", "@third"]
            .into_iter()
            .map(|name| ir.find(&format!("func_name = {name}")))
            .collect();
        assert!(slots.iter().all(Option::is_some), "{ir}");
        assert!(slots.is_sorted(), "{ir}");
        assert_eq!(ir.matches("wasm.i32_const {value = 1}").count(), 3, "{ir}");
        assert!(ir.contains("wasm.i32_const {value = 3}"), "{ir}");
        assert_eq!(ir.matches("wasm.i32_add").count(), 1, "{ir}");
        assert!(!ir.contains("tribute_rt."), "{ir}");
    }

    #[test]
    fn convention_tables_declare_the_function_table_when_absent() {
        let mut ctx = IrContext::new();
        let module = trunk_ir::parser::parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @f() -> core.nil {
    %a = tribute_rt.convention_table {entries = [@only]} : tribute_rt.convention_table_ref
    func.return
  }
}"#,
        );
        lower(&mut ctx, module);
        let ir = trunk_ir::printer::print_module(&ctx, module.op());
        assert!(
            ir.contains(r#"wasm.table {max = 1, min = 1, reftype = "funcref"}"#),
            "{ir}"
        );
        assert!(ir.contains("wasm.elem {offset = 0, table = 0}"), "{ir}");
        assert!(ir.contains("func_name = @only"), "{ir}");
        assert!(ir.contains("wasm.i32_const {value = 0}"), "{ir}");
    }

    #[test]
    fn boxed_floats_carry_their_layout_identifier() {
        let mut ctx = IrContext::new();
        let module = trunk_ir::parser::parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @f(%0: core.f64) -> core.nil {
    %1 = tribute_rt.box_float %0 : tribute_rt.anyref
    %2 = tribute_rt.unbox_float %1 : core.f64
    func.return
  }
}"#,
        );
        lower(&mut ctx, module);

        let mut boxed = Vec::new();
        let body = module.body(&ctx).expect("module body");
        let _ = walk_region::<()>(&ctx, body, &mut |op| {
            if let Ok(op) = adt::StructNew::from_op(&ctx, op) {
                boxed.push(op.r#type(&ctx));
            } else if let Ok(op) = adt::StructGet::from_op(&ctx, op) {
                boxed.push(op.r#type(&ctx));
            }
            std::ops::ControlFlow::Continue(WalkAction::Advance)
        });

        assert_eq!(boxed.len(), 2);
        for ty in boxed {
            assert!(tribute_ir::runtime_layout::has_runtime_layout(
                &ctx,
                ty,
                tribute_ir::runtime_layout::BOXED_F64,
            ));
        }
    }
}
