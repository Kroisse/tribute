//! Lower `adt.string_const` and `adt.bytes_const` to native (clif) operations.
//!
//! This pass uses a two-phase approach:
//! 1. **Analysis**: Collect the distinct string/bytes payloads of the input IR
//! 2. **Lowering**: Declare a `clif.data` object for each payload and replace
//!    const operations with clif operations that:
//!    - Reference the data object via `clif.symbol_addr`
//!    - Copy it into a new byte array the runtime allocates
//!    - Wrap the array in `String::Leaf(Bytes { .. })` for string constants
//!
//! The Cranelift backend emits the declared `clif.data` objects from the IR.
//!
//! ## Pipeline Position
//!
//! Runs before `adt_rc_header` (Phase 1.95) so that `adt.variant_new` operations
//! produced here are handled by the existing variant lowering.

use hashbrown::hash_map::EntryRef;

use crate::collections::{HashMap, HashSet};

use tribute_ir::dialect::adt;
use tribute_ir::dialect::tribute_rt;
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::clif;
use trunk_ir::dialect::core;
use trunk_ir::ops::{DialectOp, VariadicResults};
use trunk_ir::refs::{OpRef, RegionRef, TypeRef, ValueRef};
use trunk_ir::rewrite::{
    ConversionError, ConversionTarget, Module, OpRewritePattern, PatternApplicator,
    PatternRewriter, TypeConverter,
};
use trunk_ir::types::{Attribute, TypeDataBuilder};

use crate::string_literal::{ExpandBytesConst, ExpandStringConst, LiteralLayouts};
use trunk_ir::SymbolPath;
use trunk_ir::dialect::func;

/// Name of the runtime function that copies bytes into a new byte array.
const BYTE_ARRAY_FROM_FN: &str = "__tribute_byte_array_from";

/// Prefix of the data objects this pass declares.
const RODATA_PREFIX: &str = "__tribute_rodata_";

/// Result of const analysis.
pub struct NativeConstAnalysis {
    /// Distinct string/bytes payloads, in first-occurrence order.
    contents: Vec<Vec<u8>>,
    /// The layouts the byte and string constants build.
    layouts: LiteralLayouts,
}

impl NativeConstAnalysis {
    pub fn is_empty(&self) -> bool {
        self.contents.is_empty()
    }
}

/// Context for collecting const payloads during analysis.
struct ConstCollector {
    contents: Vec<Vec<u8>>,
    seen: HashSet<Vec<u8>>,
    has_string_consts: bool,
}

impl ConstCollector {
    fn new() -> Self {
        Self {
            contents: Vec::new(),
            seen: HashSet::default(),
            has_string_consts: false,
        }
    }

    fn intern(&mut self, content: &[u8]) {
        self.seen.get_or_insert_with(content, |content| {
            self.contents.push(content.to_vec());
            content.to_vec()
        });
    }

    fn visit_op(&mut self, ctx: &IrContext, op: OpRef) {
        let data = ctx.op(op);

        if data.name.dialect(ctx) == adt::DIALECT_NAME {
            if adt::StringConst::matches(ctx, op) {
                if let Some(s) = ctx.op_attrs(op).get_str("value") {
                    self.intern(s.as_bytes());
                    self.has_string_consts = true;
                }
            } else if (adt::BytesConst::matches(ctx, op) || adt::ArrayConst::matches(ctx, op))
                && let Some(Attribute::Bytes(b)) = ctx.op_attrs(op).get("value")
            {
                self.intern(b);
            }
        }
    }
}

/// Walk all operations in a region recursively.
fn walk_ops_in_region(
    ctx: &IrContext,
    region: RegionRef,
    callback: &mut impl FnMut(&IrContext, OpRef),
) {
    for &block in ctx.region(region).blocks.iter() {
        for &op in ctx.block(block).ops.iter() {
            callback(ctx, op);
            for nested in ctx.op_regions(op) {
                walk_ops_in_region(ctx, nested, callback);
            }
        }
    }
}

/// Analyze a module to collect all string/bytes constants.
fn analyze_consts(ctx: &IrContext, module: Module) -> NativeConstAnalysis {
    let mut collector = ConstCollector::new();

    if let Some(body) = module.body(ctx) {
        walk_ops_in_region(ctx, body, &mut |ctx, op| {
            collector.visit_op(ctx, op);
        });
    }

    let layouts = LiteralLayouts::collect(ctx, module, collector.has_string_consts);

    NativeConstAnalysis {
        contents: collector.contents,
        layouts,
    }
}

/// Lower `adt.string_const`, `adt.bytes_const`, and `adt.array_const`
/// operations to native clif ops.
///
/// `adt.array_const dense<"0x68656C6C6F">` becomes:
/// ```text
/// %data_ptr = clif.symbol_addr @__tribute_rodata_0
/// %len = clif.iconst 5
/// %payload = clif.call @__tribute_byte_array_from(%data_ptr, %len)
/// %result = tribute_rt.from_raw %payload : adt.array<core.i8>
/// ```
///
/// `adt.bytes_const` and `adt.string_const` first expand to the `Bytes`
/// representation and `String::Leaf` over an `adt.array_const` of the same
/// bytes, which is then lowered as above.
fn lower(
    ctx: &mut IrContext,
    module: Module,
    analysis: &NativeConstAnalysis,
) -> Result<(), ConversionError> {
    if analysis.is_empty() {
        return Ok(());
    }

    let ptr_ty = core::Ptr::params().intern(ctx).as_type_ref();
    let i64_ty = TypeDataBuilder::new("core.i64").intern(ctx);

    let content_to_symbol = declare_rodata(ctx, module, &analysis.contents);
    declare_byte_array_from(ctx, module, ptr_ty, i64_ty);

    let nat = |ctx: &mut IrContext, ty: TypeRef, value: i64, loc| {
        let constant = trunk_ir::dialect::arith::Constant::operands()
            .value(Attribute::Int(i128::from(value)))
            .results(ty)
            .build(ctx, loc);
        (constant.op_ref(), constant.result(ctx))
    };
    let target = ConversionTarget::new()
        .illegal_op("adt.array_const")
        .illegal_op("adt.bytes_const")
        .illegal_op("adt.string_const");
    PatternApplicator::new(TypeConverter::new())
        .add_pattern(ArrayConstNativePattern {
            content_to_symbol,
            ptr_ty,
            i64_ty,
        })
        .add_pattern(ExpandBytesConst {
            layouts: &analysis.layouts,
            nat: &nat,
        })
        .add_pattern(ExpandStringConst {
            layouts: &analysis.layouts,
            nat: &nat,
            cast_to_result: true,
        })
        .with_target(target)
        .apply_partial_conversion(ctx, module, "const-to-native")?;
    Ok(())
}

/// Map each payload to a byte-aligned `clif.data` object of the module.
///
/// Reuses an existing byte-aligned data object with the same bytes, and
/// appends a new one at the end of the module for every other payload, under
/// a symbol no module-level operation already uses.
fn declare_rodata(
    ctx: &mut IrContext,
    module: Module,
    contents: &[Vec<u8>],
) -> HashMap<Vec<u8>, Symbol> {
    let mut content_to_symbol = HashMap::default();
    let Some(module_block) = module.first_block(ctx) else {
        return content_to_symbol;
    };

    let mut taken = HashSet::default();
    for &op in ctx.block(module_block).ops.iter() {
        if let Some(name) = ctx.op_attrs(op).get_str("sym_name").map(Symbol::new) {
            taken.insert(name);
        }
        if let Ok(data) = clif::Data::from_op(ctx, op)
            && data.align(ctx) == 1
        {
            content_to_symbol
                .entry_ref(data.bytes(ctx).as_slice())
                .or_insert_with(|| Symbol::new(data.sym_name(ctx)));
        }
    }

    let location = ctx.op(module.op()).location;
    let mut next_idx = 0u32;
    for content in contents {
        let EntryRef::Vacant(entry) = content_to_symbol.entry_ref(content.as_slice()) else {
            continue;
        };
        let sym = loop {
            let candidate = Symbol::new(&format!("{RODATA_PREFIX}{next_idx}"));
            next_idx += 1;
            if !taken.contains(&candidate) {
                break candidate;
            }
        };
        let data = clif::Data::operands()
            .sym_name(ctx.intern_symbol_text(&sym))
            .bytes(content.as_slice().into())
            .align(1)
            .regions(None)
            .build(ctx, location);
        ctx.push_op(module_block, data.op_ref());
        entry.insert(sym);
    }
    content_to_symbol
}

/// Emit clif ops that copy the bytes of a rodata symbol into a new byte
/// array, which the runtime allocates with the byte array's reserved
/// descriptor (`design/rc.md` § "Bytes").
///
/// Returns the ops to insert, in order, and the payload pointer they produce.
fn emit_byte_array(
    ctx: &mut IrContext,
    loc: trunk_ir::types::Location,
    data_sym: Symbol,
    content_len: u64,
    ptr_ty: TypeRef,
    i64_ty: TypeRef,
) -> (Vec<OpRef>, ValueRef) {
    let data_ptr = clif::SymbolAddr::operands()
        .sym(SymbolPath::new([data_sym]))
        .results(ptr_ty)
        .build(ctx, loc);
    let len = clif::Iconst::operands()
        .value(content_len as i64)
        .results(i64_ty)
        .build(ctx, loc);
    let call = clif::Call::operands([data_ptr.result(ctx), len.result(ctx)])
        .callee(SymbolPath::new([BYTE_ARRAY_FROM_FN]))
        .results([ptr_ty])
        .build(ctx, loc);
    let payload = call.result(ctx);
    (
        vec![data_ptr.op_ref(), len.op_ref(), call.op_ref()],
        payload,
    )
}

/// Declare the runtime function that copies bytes into a new byte array,
/// unless the module already declares it.
fn declare_byte_array_from(ctx: &mut IrContext, module: Module, ptr_ty: TypeRef, i64_ty: TypeRef) {
    let Some(block) = module.first_block(ctx) else {
        return;
    };
    let declared = ctx.block(block).ops.iter().any(|&op| {
        func::Func::from_op(ctx, op).is_ok_and(|func| func.sym_name(ctx) == BYTE_ARRAY_FROM_FN)
    });
    if declared {
        return;
    }
    let loc = ctx.op(module.op()).location;
    let declaration =
        super::build_extern_func(ctx, loc, BYTE_ARRAY_FROM_FN, &[ptr_ty, i64_ty], ptr_ty);
    ctx.push_op(block, declaration);
}

/// Pattern for `adt.array_const` → clif ops (rodata + a copying allocation).
struct ArrayConstNativePattern {
    content_to_symbol: HashMap<Vec<u8>, Symbol>,
    ptr_ty: TypeRef,
    i64_ty: TypeRef,
}

impl OpRewritePattern for ArrayConstNativePattern {
    type Op = adt::ArrayConst;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        array_const: Self::Op,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let op = array_const.op_ref();
        let content: Vec<u8> = array_const.value(ctx).to_vec();

        let data_sym = self.content_to_symbol.get(&content).cloned()?;

        let loc = ctx.op(op).location;
        let (insert_ops, payload) = emit_byte_array(
            ctx,
            loc,
            data_sym,
            content.len() as u64,
            self.ptr_ty,
            self.i64_ty,
        );

        // The allocation is a `core.ptr` to a new object; the constant is the
        // owned managed value it holds.
        let result_ty = ctx.op_result_types(op)[0];
        let typed = tribute_rt::FromRaw::operands(payload)
            .results(result_ty)
            .build(ctx, loc);

        for o in insert_ops {
            rewriter.insert_op(o);
        }
        rewriter.replace_op(typed.op_ref());
        Some(())
    }

    fn name(&self) -> &'static str {
        "ArrayConstNativePattern"
    }
}

/// Lowers string and bytes constants to `clif.data` objects and RC-managed
/// allocations.
pub struct ConstToNative;

impl trunk_ir::pass::Pass for ConstToNative {
    type Target = trunk_ir::dialect::core::Module;

    fn name(&self) -> &'static str {
        "const-to-native"
    }

    fn run(
        &mut self,
        ctx: &mut trunk_ir::context::IrContext,
        target: trunk_ir::dialect::core::Module,
        _analyses: &mut trunk_ir::analysis::AnalysisCache,
    ) -> trunk_ir::pass::PassRunResult {
        let analysis = analyze_consts(ctx, target.into());
        lower(ctx, target.into(), &analysis)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;

    fn type_alias(ctx: &IrContext, name: &str) -> TypeRef {
        ctx.type_aliases()
            .iter()
            .find_map(|(alias, ty)| (*alias == name).then_some(*ty))
            .unwrap_or_else(|| panic!("missing type alias !{name}"))
    }

    fn string_module(ctx: &mut IrContext) -> Module {
        parse_test_module(
            ctx,
            r#"core.module @test {
  !Bytes = adt.struct<"std::Bytes", (core.i32, adt.array<core.i8>, core.i32)>
  !PreludeString = adt.enum<"String", [(!Bytes), (tribute_rt.anyref, tribute_rt.anyref, core.i32)]>
  !UserString = adt.enum<"user::String", [(!Bytes), (tribute_rt.anyref, tribute_rt.anyref, core.i32)]>
  adt.struct_decl "std::Bytes" !Bytes ["len", "data", "start"]
  adt.enum_decl "String" !PreludeString ["Leaf", "Branch"]
  adt.enum_decl "user::String" !UserString ["Leaf", "Branch"]
  func.func @main() -> core.nil {
    %string = adt.string_const "hello" : tribute_rt.anyref
    func.return
  }
}"#,
        )
    }

    #[test]
    fn analysis_uses_exact_string_metadata_not_user_lookalike() {
        let mut ctx = IrContext::new();
        let module = string_module(&mut ctx);
        let prelude_string = type_alias(&ctx, "PreludeString");
        tribute_ir::metadata::WellKnownTypes {
            string: Some(prelude_string),
        }
        .attach(&mut ctx, module.op());

        let analysis = analyze_consts(&ctx, module);

        let leaf = analysis
            .layouts
            .string_leaf()
            .expect("the prelude String's leaf");
        assert_eq!(
            (leaf.string, leaf.leaf, leaf.bytes),
            (prelude_string, 0, type_alias(&ctx, "Bytes"))
        );
        // Fields are found by their declared names, not their positions.
        let repr = analysis
            .layouts
            .bytes_repr(leaf.bytes)
            .expect("the representation of Bytes");
        assert_eq!((repr.data, repr.start, repr.len), (1, 2, 0));
    }

    #[test]
    fn string_constant_builds_a_leaf_of_bytes_over_a_byte_array() {
        let mut ctx = IrContext::new();
        let module = string_module(&mut ctx);
        let prelude_string = type_alias(&ctx, "PreludeString");
        tribute_ir::metadata::WellKnownTypes {
            string: Some(prelude_string),
        }
        .attach(&mut ctx, module.op());
        let analysis = analyze_consts(&ctx, module);

        lower(&mut ctx, module, &analysis).expect("string constants should lower");

        let output = module.op().display(&ctx).to_string();
        assert!(!output.contains("adt.string_const"), "{output}");
        assert!(!output.contains("adt.bytes_const"), "{output}");
        assert!(
            output.contains("func.call @__tribute_byte_array_from")
                || output.contains("clif.call @__tribute_byte_array_from"),
            "{output}"
        );
        assert!(output.contains("adt.struct_new"), "{output}");
        assert!(output.contains("adt.variant_new 0"), "{output}");
    }

    #[test]
    fn lowering_rejects_missing_string_metadata() {
        let mut ctx = IrContext::new();
        let module = string_module(&mut ctx);
        let analysis = analyze_consts(&ctx, module);

        let error = lower(&mut ctx, module, &analysis).expect_err("metadata is required");

        assert_eq!(error.boundary(), "const-to-native");
        assert!(
            error
                .operations()
                .iter()
                .any(|illegal| { illegal.name == "adt.string_const" })
        );
    }

    fn data_objects(ctx: &IrContext, module: Module) -> Vec<(String, Vec<u8>, u32)> {
        module
            .ops(ctx)
            .iter()
            .copied()
            .filter_map(|op| clif::Data::from_op(ctx, op).ok())
            .map(|data| {
                (
                    data.sym_name(ctx).to_string(),
                    data.bytes(ctx).to_vec(),
                    data.align(ctx),
                )
            })
            .collect()
    }

    fn symbol_addrs(ctx: &IrContext, module: Module) -> Vec<String> {
        let mut symbols = Vec::new();
        if let Some(body) = module.body(ctx) {
            walk_ops_in_region(ctx, body, &mut |ctx, op| {
                if let Ok(addr) = clif::SymbolAddr::from_op(ctx, op) {
                    symbols.push(addr.sym(ctx).to_string());
                }
            });
        }
        symbols
    }

    #[test]
    fn lowering_declares_one_data_object_per_payload() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @main() -> core.nil {
    %first = adt.array_const dense<"0x6669727374"> : adt.array<core.i8>
    %second = adt.array_const dense<"0x7365636F6E64"> : adt.array<core.i8>
    %again = adt.array_const dense<"0x6669727374"> : adt.array<core.i8>
    func.return
  }
}"#,
        );
        let analysis = analyze_consts(&ctx, module);

        lower(&mut ctx, module, &analysis).expect("bytes constants should lower");

        assert_eq!(
            data_objects(&ctx, module),
            [
                ("__tribute_rodata_0".to_owned(), b"first".to_vec(), 1),
                ("__tribute_rodata_1".to_owned(), b"second".to_vec(), 1),
            ]
        );
        assert_eq!(
            symbol_addrs(&ctx, module),
            [
                "__tribute_rodata_0",
                "__tribute_rodata_1",
                "__tribute_rodata_0"
            ]
        );
    }

    #[test]
    fn lowering_reuses_data_objects_and_avoids_taken_symbols() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  clif.data @__tribute_rodata_0(b"kept") align 1
  clif.data @aligned(b"wide") align 8
  func.func @__tribute_rodata_1() -> core.nil {
    func.return
  }
  func.func @main() -> core.nil {
    %kept = adt.array_const dense<"0x6B657074"> : adt.array<core.i8>
    %wide = adt.array_const dense<"0x77696465"> : adt.array<core.i8>
    func.return
  }
}"#,
        );
        let analysis = analyze_consts(&ctx, module);

        lower(&mut ctx, module, &analysis).expect("bytes constants should lower");

        assert_eq!(
            data_objects(&ctx, module),
            [
                ("__tribute_rodata_0".to_owned(), b"kept".to_vec(), 1),
                ("aligned".to_owned(), b"wide".to_vec(), 8),
                ("__tribute_rodata_2".to_owned(), b"wide".to_vec(), 1),
            ]
        );
        assert_eq!(
            symbol_addrs(&ctx, module),
            ["__tribute_rodata_0", "__tribute_rodata_2"]
        );
    }

    #[test]
    fn lowered_bytes_constant_keeps_its_declared_type_at_block_arguments() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @pick(%x: adt.array<core.i8>, %flag: core.i1) -> adt.array<core.i8> {
    ^entry:
      cf.cond_br %flag [^param, ^literal]
    ^param:
      cf.br %x [^merge]
    ^literal:
      %no = adt.array_const dense<"0x6E6F"> : adt.array<core.i8>
      cf.br %no [^merge]
    ^merge(%picked: adt.array<core.i8>):
      func.return %picked
  }
}"#,
        );
        let analysis = analyze_consts(&ctx, module);

        lower(&mut ctx, module, &analysis).expect("bytes constants should lower");

        let bytes_ty = tribute_ir::runtime_layout::byte_array(&mut ctx);
        let mut forwarded = Vec::new();
        let body = module.body(&ctx).expect("module body");
        walk_ops_in_region(&ctx, body, &mut |ctx, op| {
            if trunk_ir::dialect::cf::Br::matches(ctx, op) {
                forwarded.extend(ctx.op_operands(op).iter().map(|&value| ctx.value_ty(value)));
            }
        });
        assert_eq!(forwarded, [bytes_ty, bytes_ty]);
    }
}
