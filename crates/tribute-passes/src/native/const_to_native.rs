//! Lower `adt.string_const` and `adt.bytes_const` to native (clif) operations.
//!
//! This pass uses a two-phase approach:
//! 1. **Analysis**: Collect all string/bytes constants and assign data symbol names
//! 2. **Lowering**: Replace const operations with clif operations that:
//!    - Reference rodata via `clif.symbol_addr`
//!    - Allocate RC-managed `TributeBytes` structs
//!    - Wrap bytes in `adt.variant_new(String, Leaf, bytes)` for string constants
//!
//! The analysis produces `NativeConstAnalysis` which is passed to the Cranelift
//! backend for data section emission.
//!
//! ## Pipeline Position
//!
//! Runs before `adt_rc_header` (Phase 1.95) so that `adt.variant_new` operations
//! produced here are handled by the existing variant lowering.

use std::collections::HashMap;

use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::adt as arena_adt;
use trunk_ir::dialect::clif as arena_clif;
use trunk_ir::dialect::core as arena_core;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{OpRef, RegionRef, TypeRef};
use trunk_ir::rewrite::{
    Module, PatternApplicator, PatternRewriter, RewritePattern, TypeConverter,
};
use trunk_ir::types::{Attribute, TypeDataBuilder};

use tribute_ir::dialect::tribute_rt::RC_HEADER_SIZE;

/// Name of the runtime allocation function.
const ALLOC_FN: &str = "__tribute_alloc";

/// Result of const analysis — maps content to data symbol names.
///
/// Passed to Cranelift backend for data section emission.
pub struct NativeConstAnalysis {
    /// (symbol_name, byte_content) pairs for rodata sections.
    pub rodata: Vec<(Symbol, Vec<u8>)>,
    /// Map from byte content to its assigned symbol name.
    content_to_symbol: HashMap<Vec<u8>, Symbol>,
    /// Whether the module contains any `adt.string_const` ops.
    has_string_consts: bool,
    /// The String enum type (found by scanning IR types).
    string_enum_ty: Option<TypeRef>,
}

impl NativeConstAnalysis {
    pub fn is_empty(&self) -> bool {
        self.rodata.is_empty()
    }
}

/// Find the String enum type by scanning interned types.
///
/// Looks for an `adt.enum` type named "String" with variants "Leaf" and "Branch".
fn find_string_enum_type(ctx: &IrContext) -> Option<TypeRef> {
    for (ty_ref, td) in ctx.types.iter() {
        if td.dialect != Symbol::new("adt") {
            continue;
        }
        if let Some(Attribute::Symbol(name)) = td.attrs.get(&Symbol::new("name")) {
            if *name != Symbol::new("String") {
                continue;
            }
        } else {
            continue;
        }
        // Check it has "variants" attribute with Leaf and Branch
        if let Some(Attribute::List(variants)) = td.attrs.get(&Symbol::new("variants")) {
            let has_leaf = variants.iter().any(|v| {
                if let Attribute::List(items) = v {
                    items.first().is_some_and(
                        |n| matches!(n, Attribute::Symbol(s) if *s == Symbol::new("Leaf")),
                    )
                } else {
                    false
                }
            });
            let has_branch = variants.iter().any(|v| {
                if let Attribute::List(items) = v {
                    items.first().is_some_and(
                        |n| matches!(n, Attribute::Symbol(s) if *s == Symbol::new("Branch")),
                    )
                } else {
                    false
                }
            });
            if has_leaf && has_branch {
                return Some(ty_ref);
            }
        }
    }
    None
}

/// Context for collecting const allocations during analysis.
struct ConstCollector {
    rodata: Vec<(Symbol, Vec<u8>)>,
    seen: HashMap<Vec<u8>, Symbol>,
    next_idx: u32,
    has_string_consts: bool,
}

impl ConstCollector {
    fn new() -> Self {
        Self {
            rodata: Vec::new(),
            seen: HashMap::new(),
            next_idx: 0,
            has_string_consts: false,
        }
    }

    fn intern(&mut self, content: Vec<u8>) -> Symbol {
        if let Some(&sym) = self.seen.get(&content) {
            return sym;
        }
        let sym = Symbol::from_dynamic(&format!("__tribute_rodata_{}", self.next_idx));
        self.next_idx += 1;
        self.seen.insert(content.clone(), sym);
        self.rodata.push((sym, content));
        sym
    }

    fn visit_op(&mut self, ctx: &IrContext, op: OpRef) {
        let data = ctx.op(op);

        if data.dialect == arena_adt::DIALECT_NAME() {
            if data.name == Symbol::new("string_const") {
                if let Some(Attribute::String(s)) = data.attributes.get(&Symbol::new("value")) {
                    let bytes = s.clone().into_bytes();
                    self.intern(bytes);
                    self.has_string_consts = true;
                }
            } else if data.name == Symbol::new("bytes_const")
                && let Some(Attribute::Bytes(b)) = data.attributes.get(&Symbol::new("value"))
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
            for &nested in ctx.op(op).regions.iter() {
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

    let string_enum_ty = if collector.has_string_consts {
        find_string_enum_type(ctx)
    } else {
        None
    };

    NativeConstAnalysis {
        content_to_symbol: collector.seen,
        rodata: collector.rodata,
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
pub fn lower(ctx: &mut IrContext, module: Module, analysis: &NativeConstAnalysis) {
    if analysis.is_empty() {
        return;
    }

    let ptr_ty = arena_core::ptr(ctx).as_type_ref();
    let i64_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i64")).build());
    let i32_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());

    let content_to_symbol = analysis.content_to_symbol.clone();
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

    applicator.apply_partial(ctx, module);
}

/// Emit clif ops to allocate an RC-managed TributeBytes from a rodata symbol.
///
/// Returns (ops_to_insert, last_op_to_replace_with).
/// The result value of the last op is the payload pointer.
fn emit_bytes_alloc(
    ctx: &mut IrContext,
    loc: trunk_ir::types::Location,
    data_sym: Symbol,
    content_len: u64,
    ptr_ty: TypeRef,
    i64_ty: TypeRef,
    i32_ty: TypeRef,
) -> (Vec<OpRef>, OpRef) {
    let mut ops: Vec<OpRef> = Vec::new();

    // 1. Get rodata address
    let data_ptr_op = arena_clif::symbol_addr(ctx, loc, ptr_ty, data_sym);
    ops.push(data_ptr_op.op_ref());
    let data_ptr = data_ptr_op.result(ctx);

    // 2. Length constant
    let len_op = arena_clif::iconst(ctx, loc, i64_ty, content_len as i64);
    ops.push(len_op.op_ref());
    let len_val = len_op.result(ctx);

    // 3. Allocate RC header (8) + TributeBytes payload (ptr=8 + len=8 = 16) = 24 bytes
    let alloc_size = RC_HEADER_SIZE + 16; // ptr(8) + len(8)
    let size_op = arena_clif::iconst(ctx, loc, i64_ty, alloc_size as i64);
    ops.push(size_op.op_ref());

    let call_op = arena_clif::call(
        ctx,
        loc,
        [size_op.result(ctx)],
        ptr_ty,
        Symbol::new(ALLOC_FN),
    );
    ops.push(call_op.op_ref());
    let raw_ptr = call_op.result(ctx);

    // 4. Store RC header: refcount=1 at raw+0, rtti_idx=0 at raw+4
    let rc_one = arena_clif::iconst(ctx, loc, i32_ty, 1);
    ops.push(rc_one.op_ref());
    let store_rc = arena_clif::store(ctx, loc, rc_one.result(ctx), raw_ptr, 0);
    ops.push(store_rc.op_ref());

    let rtti_zero = arena_clif::iconst(ctx, loc, i32_ty, 0);
    ops.push(rtti_zero.op_ref());
    let store_rtti = arena_clif::store(ctx, loc, rtti_zero.result(ctx), raw_ptr, 4);
    ops.push(store_rtti.op_ref());

    // 5. Compute payload pointer = raw + 8
    let hdr_size = arena_clif::iconst(ctx, loc, i64_ty, RC_HEADER_SIZE as i64);
    ops.push(hdr_size.op_ref());
    let payload_op = arena_clif::iadd(ctx, loc, raw_ptr, hdr_size.result(ctx), ptr_ty);
    ops.push(payload_op.op_ref());
    let payload = payload_op.result(ctx);

    // 6. Store TributeBytes fields: ptr at payload+0, len at payload+8
    let store_ptr = arena_clif::store(ctx, loc, data_ptr, payload, 0);
    ops.push(store_ptr.op_ref());

    let store_len = arena_clif::store(ctx, loc, len_val, payload, 8);
    ops.push(store_len.op_ref());

    // 7. Identity iadd(payload, 0) to produce a fresh SSA value that the
    //    rewrite pattern can use as the replacement result.
    let zero_op = arena_clif::iconst(ctx, loc, i64_ty, 0);
    ops.push(zero_op.op_ref());
    let identity_op = arena_clif::iadd(ctx, loc, payload, zero_op.result(ctx), ptr_ty);
    ops.push(identity_op.op_ref());

    let last = ops.pop().unwrap();
    (ops, last)
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
        let Ok(bytes_const) = arena_adt::BytesConst::from_op(ctx, op) else {
            return false;
        };

        let content: Vec<u8> = bytes_const.value(ctx).to_vec();

        let Some(data_sym) = self.content_to_symbol.get(&content).copied() else {
            return false;
        };

        let loc = ctx.op(op).location;
        let (insert_ops, last_op) = emit_bytes_alloc(
            ctx,
            loc,
            data_sym,
            content.len() as u64,
            self.ptr_ty,
            self.i64_ty,
            self.i32_ty,
        );

        for o in insert_ops {
            rewriter.insert_op(o);
        }
        rewriter.replace_op(last_op);
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
        let Ok(string_const) = arena_adt::StringConst::from_op(ctx, op) else {
            return false;
        };

        let value_str = string_const.value(ctx);
        let content = value_str.into_bytes();

        let Some(data_sym) = self.content_to_symbol.get(&content).copied() else {
            return false;
        };

        let Some(string_enum_ty) = self.string_enum_ty else {
            tracing::warn!("const_to_native: String enum type not found, skipping string_const");
            return false;
        };

        let loc = ctx.op(op).location;

        // Emit bytes allocation
        let (insert_ops, bytes_last_op) = emit_bytes_alloc(
            ctx,
            loc,
            data_sym,
            content.len() as u64,
            self.ptr_ty,
            self.i64_ty,
            self.i32_ty,
        );
        let bytes_payload = arena_clif::Iadd::from_op(ctx, bytes_last_op)
            .expect("last op is iadd")
            .result(ctx);

        // Get the result type of the original string_const
        let result_ty = ctx.op_result_types(op)[0];

        // Create adt.variant_new(type=String, tag=Leaf, bytes_payload)
        // Use the actual String enum type for the type attribute so that
        // adt_rc_header can compute the correct enum layout.
        let variant_new = arena_adt::variant_new(
            ctx,
            loc,
            [bytes_payload],
            result_ty,
            string_enum_ty,
            Symbol::new("Leaf"),
        );

        for o in insert_ops {
            rewriter.insert_op(o);
        }
        rewriter.insert_op(bytes_last_op);
        rewriter.replace_op(variant_new.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "StringConstNativePattern"
    }
}
