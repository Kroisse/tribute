//! Lower `adt.string_const` and `adt.bytes_const` to native (clif) operations.
//!
//! This pass uses a two-phase approach:
//! 1. **Analysis**: Collect the distinct string/bytes payloads of the input IR
//! 2. **Lowering**: Declare a `clif.data` object for each payload and replace
//!    const operations with clif operations that:
//!    - Reference the data object via `clif.symbol_addr`
//!    - Allocate RC-managed `TributeBytes` structs
//!    - Wrap bytes in `adt.variant_new(String, Leaf, bytes)` for string constants
//!
//! The Cranelift backend emits the declared `clif.data` objects from the IR.
//!
//! ## Pipeline Position
//!
//! Runs before `adt_rc_header` (Phase 1.95) so that `adt.variant_new` operations
//! produced here are handled by the existing variant lowering.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};

use tribute_ir::dialect::adt;
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::clif;
use trunk_ir::dialect::core;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{OpRef, RegionRef, TypeRef, ValueRef};
use trunk_ir::rewrite::{
    ConversionError, ConversionTarget, Module, PatternApplicator, PatternRewriter, RewritePattern,
    TypeConverter,
};
use trunk_ir::types::{Attribute, TypeDataBuilder};

use tribute_ir::dialect::tribute_rt::{RC_HEADER_SIZE, REFCOUNT_OFFSET, RTTI_IDX_OFFSET};
use trunk_ir::SymbolPath;

/// Name of the runtime allocation function.
const ALLOC_FN: &str = "__tribute_alloc";

/// Prefix of the data objects this pass declares.
const RODATA_PREFIX: &str = "__tribute_rodata_";

/// Result of const analysis.
pub struct NativeConstAnalysis {
    /// Distinct string/bytes payloads, in first-occurrence order.
    contents: Vec<Vec<u8>>,
    /// Whether the module contains any `adt.string_const` ops.
    has_string_consts: bool,
    /// The exact prelude String enum type from module metadata.
    string_enum_ty: Option<TypeRef>,
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
            seen: HashSet::new(),
            has_string_consts: false,
        }
    }

    fn intern(&mut self, content: Vec<u8>) {
        if !self.seen.contains(&content) {
            self.seen.insert(content.clone());
            self.contents.push(content);
        }
    }

    fn visit_op(&mut self, ctx: &IrContext, op: OpRef) {
        let data = ctx.op(op);

        if data.dialect == adt::DIALECT_NAME() {
            if data.name == Symbol::new("string_const") {
                if let Some(s) = data.attributes.get_str(ctx, "value") {
                    let bytes = s.as_bytes().to_vec();
                    self.intern(bytes);
                    self.has_string_consts = true;
                }
            } else if data.name == Symbol::new("bytes_const")
                && let Some(Attribute::Bytes(b)) = data.attributes.get("value")
            {
                let bytes: Vec<u8> = b.to_vec();
                self.intern(bytes);
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
pub fn analyze_consts(ctx: &IrContext, module: Module) -> NativeConstAnalysis {
    let mut collector = ConstCollector::new();

    if let Some(body) = module.body(ctx) {
        walk_ops_in_region(ctx, body, &mut |ctx, op| {
            collector.visit_op(ctx, op);
        });
    }

    let string_enum_ty = collector
        .has_string_consts
        .then(|| tribute_ir::metadata::WellKnownTypes::from_module(ctx, module.op()).string)
        .flatten();

    NativeConstAnalysis {
        contents: collector.contents,
        has_string_consts: collector.has_string_consts,
        string_enum_ty,
    }
}

/// Lower `adt.string_const` and `adt.bytes_const` operations to native clif ops.
///
/// `adt.bytes_const(b"hello")` becomes:
/// ```text
/// %data_ptr = clif.symbol_addr @__tribute_rodata_0
/// %len = clif.iconst 5
/// %raw = clif.call @__tribute_alloc(24)  // RC(8) + ptr(8) + len(8)
/// // Store RC header
/// clif.store 1, %raw, offset=0          // refcount
/// clif.store 0, %raw, offset=4          // rtti_idx
/// // Compute payload pointer
/// %payload = clif.iadd %raw, 8
/// // Store TributeBytes fields
/// clif.store %data_ptr, %payload, offset=0
/// clif.store %len, %payload, offset=8
/// → result = %payload
/// ```
///
/// `adt.string_const("hello")` becomes the above bytes lowering +
/// `adt.variant_new(type=String, tag=Leaf, %bytes_payload)`
pub fn lower(
    ctx: &mut IrContext,
    module: Module,
    analysis: &NativeConstAnalysis,
) -> Result<(), ConversionError> {
    if analysis.is_empty() {
        return Ok(());
    }

    let ptr_ty = core::ptr(ctx).as_type_ref();
    let i64_ty = ctx.intern_type(TypeDataBuilder::new("core", "i64").build());
    let i32_ty = ctx.intern_type(TypeDataBuilder::new("core", "i32").build());

    let content_to_symbol = declare_rodata(ctx, module, &analysis.contents);
    let string_enum_ty = analysis.string_enum_ty;

    let mut applicator =
        PatternApplicator::new(TypeConverter::new()).add_pattern(BytesConstNativePattern {
            content_to_symbol: content_to_symbol.clone(),
            ptr_ty,
            i64_ty,
            i32_ty,
        });

    if analysis.has_string_consts {
        applicator = applicator.add_pattern(StringConstNativePattern {
            content_to_symbol,
            ptr_ty,
            i64_ty,
            i32_ty,
            string_enum_ty,
        });
    }

    let target = ConversionTarget::new()
        .illegal_op("adt", "bytes_const")
        .illegal_op("adt", "string_const");
    applicator
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
    let mut content_to_symbol = HashMap::new();
    let Some(module_block) = module.first_block(ctx) else {
        return content_to_symbol;
    };

    let mut taken = HashSet::new();
    for &op in ctx.block(module_block).ops.iter() {
        if let Some(name) = ctx
            .op(op)
            .attributes
            .get_str(ctx, "sym_name")
            .map(Symbol::from_dynamic)
        {
            taken.insert(name);
        }
        if let Ok(data) = clif::Data::from_op(ctx, op)
            && data.align(ctx) == 1
        {
            content_to_symbol
                .entry(data.bytes(ctx).to_vec())
                .or_insert(Symbol::from_dynamic(data.sym_name(ctx)));
        }
    }

    let location = ctx.op(module.op()).location;
    let mut next_idx = 0u32;
    for content in contents {
        let Entry::Vacant(entry) = content_to_symbol.entry(content.clone()) else {
            continue;
        };
        let sym = loop {
            let candidate = Symbol::from_dynamic(&format!("{RODATA_PREFIX}{next_idx}"));
            next_idx += 1;
            if !taken.contains(&candidate) {
                break candidate;
            }
        };
        let data = clif::Data::operands()
            .sym_name(sym.clone())
            .bytes(content.as_slice().into())
            .align(1)
            .regions(None)
            .build(ctx, location);
        ctx.push_op(module_block, data.op_ref());
        entry.insert(sym);
    }
    content_to_symbol
}

/// Emit clif ops to allocate an RC-managed TributeBytes from a rodata symbol.
///
/// Returns the ops to insert, in order, and the payload pointer they produce.
fn emit_bytes_alloc(
    ctx: &mut IrContext,
    loc: trunk_ir::types::Location,
    data_sym: Symbol,
    content_len: u64,
    ptr_ty: TypeRef,
    i64_ty: TypeRef,
    i32_ty: TypeRef,
) -> (Vec<OpRef>, ValueRef) {
    let mut ops: Vec<OpRef> = Vec::new();

    // 1. Get rodata address
    let data_ptr_op = clif::SymbolAddr::operands()
        .sym(data_sym.into())
        .results(ptr_ty)
        .build(ctx, loc);
    ops.push(data_ptr_op.op_ref());
    let data_ptr = data_ptr_op.result(ctx);

    // 2. Length constant
    let len_op = clif::Iconst::operands()
        .value(content_len as i64)
        .results(i64_ty)
        .build(ctx, loc);
    ops.push(len_op.op_ref());
    let len_val = len_op.result(ctx);

    // 3. Allocate RC header (8) + TributeBytes payload (ptr=8 + len=8 = 16) = 24 bytes
    let alloc_size = RC_HEADER_SIZE + 16; // ptr(8) + len(8)
    let size_op = clif::Iconst::operands()
        .value(alloc_size as i64)
        .results(i64_ty)
        .build(ctx, loc);
    ops.push(size_op.op_ref());

    let call_op = clif::Call::operands([size_op.result(ctx)])
        .callee(SymbolPath::from(ALLOC_FN))
        .results([ptr_ty])
        .build(ctx, loc);
    ops.push(call_op.op_ref());
    let raw_ptr = call_op.results(ctx)[0];

    // 4. Store RC header: refcount=1, rtti_idx=0
    let rc_one = clif::Iconst::operands()
        .value(1)
        .results(i32_ty)
        .build(ctx, loc);
    ops.push(rc_one.op_ref());
    let store_rc = clif::Store::operands(rc_one.result(ctx), raw_ptr)
        .offset(REFCOUNT_OFFSET as i32)
        .build(ctx, loc);
    ops.push(store_rc.op_ref());

    let rtti_zero = clif::Iconst::operands()
        .value(0)
        .results(i32_ty)
        .build(ctx, loc);
    ops.push(rtti_zero.op_ref());
    let store_rtti = clif::Store::operands(rtti_zero.result(ctx), raw_ptr)
        .offset(RTTI_IDX_OFFSET as i32)
        .build(ctx, loc);
    ops.push(store_rtti.op_ref());

    // 5. Compute payload pointer = raw + 8
    let hdr_size = clif::Iconst::operands()
        .value(RC_HEADER_SIZE as i64)
        .results(i64_ty)
        .build(ctx, loc);
    ops.push(hdr_size.op_ref());
    let payload_op = clif::Iadd::operands(raw_ptr, hdr_size.result(ctx))
        .results(ptr_ty)
        .build(ctx, loc);
    ops.push(payload_op.op_ref());
    let payload = payload_op.result(ctx);

    // 6. Store TributeBytes fields: ptr at payload+0, len at payload+8
    let store_ptr = clif::Store::operands(data_ptr, payload)
        .offset(0)
        .build(ctx, loc);
    ops.push(store_ptr.op_ref());

    let store_len = clif::Store::operands(len_val, payload)
        .offset(8)
        .build(ctx, loc);
    ops.push(store_len.op_ref());

    (ops, payload)
}

/// Pattern for `adt.bytes_const` → clif ops (rodata + alloc).
struct BytesConstNativePattern {
    content_to_symbol: HashMap<Vec<u8>, Symbol>,
    ptr_ty: TypeRef,
    i64_ty: TypeRef,
    i32_ty: TypeRef,
}

impl RewritePattern for BytesConstNativePattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(bytes_const) = adt::BytesConst::from_op(ctx, op) else {
            return false;
        };

        let content: Vec<u8> = bytes_const.value(ctx).to_vec();

        let Some(data_sym) = self.content_to_symbol.get(&content).cloned() else {
            return false;
        };

        let loc = ctx.op(op).location;
        let (insert_ops, payload) = emit_bytes_alloc(
            ctx,
            loc,
            data_sym,
            content.len() as u64,
            self.ptr_ty,
            self.i64_ty,
            self.i32_ty,
        );

        // The allocation is a `core.ptr`; uses still declare the constant's
        // type until native type conversion runs.
        let result_ty = ctx.op_result_types(op)[0];
        let typed = core::UnrealizedConversionCast::operands(payload)
            .results(result_ty)
            .build(ctx, loc);

        for o in insert_ops {
            rewriter.insert_op(o);
        }
        rewriter.replace_op(typed.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "BytesConstNativePattern"
    }
}

/// Pattern for `adt.string_const` → bytes alloc + `adt.variant_new(String, Leaf, bytes)`.
struct StringConstNativePattern {
    content_to_symbol: HashMap<Vec<u8>, Symbol>,
    ptr_ty: TypeRef,
    i64_ty: TypeRef,
    i32_ty: TypeRef,
    string_enum_ty: Option<TypeRef>,
}

impl RewritePattern for StringConstNativePattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(string_const) = adt::StringConst::from_op(ctx, op) else {
            return false;
        };

        let content = string_const.value(ctx).as_bytes();
        let content_len = content.len() as u64;
        let Some(data_sym) = self.content_to_symbol.get(content).cloned() else {
            return false;
        };

        let Some(string_enum_ty) = self.string_enum_ty else {
            tracing::warn!("const_to_native: String enum type not found, skipping string_const");
            return false;
        };

        let loc = ctx.op(op).location;

        // Emit bytes allocation
        let (insert_ops, bytes_payload) = emit_bytes_alloc(
            ctx,
            loc,
            data_sym,
            content_len,
            self.ptr_ty,
            self.i64_ty,
            self.i32_ty,
        );

        // Get the result type of the original string_const
        let result_ty = ctx.op_result_types(op)[0];

        // Create adt.variant_new(type=String, tag=Leaf, bytes_payload)
        // Use the actual String enum type for the type attribute so that
        // adt_rc_header can compute the correct enum layout.
        let variant_new = adt::VariantNew::operands([bytes_payload])
            .r#type(string_enum_ty)
            .tag("Leaf")
            .results(result_ty)
            .build(ctx, loc);

        for o in insert_ops {
            rewriter.insert_op(o);
        }
        rewriter.replace_op(variant_new.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "StringConstNativePattern"
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
  !PreludeString = adt.enum<{name = "String", variants = [["Leaf", [core.bytes]], ["Branch", [tribute_rt.anyref, tribute_rt.anyref, core.i32]]]}>
  !UserString = adt.enum<{name = "user::String", variants = [["Leaf", [core.bytes]], ["Branch", [tribute_rt.anyref, tribute_rt.anyref, core.i32]]]}>
  func.func @main() -> core.nil {
    %string = adt.string_const {value = "hello"} : tribute_rt.anyref
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
        let user_string = type_alias(&ctx, "UserString");
        tribute_ir::metadata::WellKnownTypes {
            string: Some(prelude_string),
        }
        .attach(&mut ctx, module.op());

        let analysis = analyze_consts(&ctx, module);

        assert_eq!(analysis.string_enum_ty, Some(prelude_string));
        assert_ne!(analysis.string_enum_ty, Some(user_string));
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
                .any(|illegal| { illegal.dialect == "adt" && illegal.name == "string_const" })
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
    %first = adt.bytes_const {value = b"first"} : core.bytes
    %second = adt.bytes_const {value = b"second"} : core.bytes
    %again = adt.bytes_const {value = b"first"} : core.bytes
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
  clif.data {sym_name = "__tribute_rodata_0", bytes = b"kept", align = 1}
  clif.data {sym_name = "aligned", bytes = b"wide", align = 8}
  func.func @__tribute_rodata_1() -> core.nil {
    func.return
  }
  func.func @main() -> core.nil {
    %kept = adt.bytes_const {value = b"kept"} : core.bytes
    %wide = adt.bytes_const {value = b"wide"} : core.bytes
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
  func.func @pick(%x: core.bytes, %flag: core.i1) -> core.bytes {
    ^entry:
      cf.cond_br %flag [^param, ^literal]
    ^param:
      cf.br %x [^merge]
    ^literal:
      %no = adt.bytes_const {value = b"no"} : core.bytes
      cf.br %no [^merge]
    ^merge(%picked: core.bytes):
      func.return %picked
  }
}"#,
        );
        let analysis = analyze_consts(&ctx, module);

        lower(&mut ctx, module, &analysis).expect("bytes constants should lower");

        let bytes_ty = core::bytes(&mut ctx).as_type_ref();
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
