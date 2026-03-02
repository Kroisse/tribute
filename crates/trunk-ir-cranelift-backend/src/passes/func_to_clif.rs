//! Lower func dialect operations to clif dialect.
//!
//! This pass converts function-level operations to Cranelift equivalents:
//! - `func.func` -> `clif.func`
//! - `func.call` -> `clif.call`
//! - `func.call_indirect` -> `clif.call_indirect`
//! - `func.return` -> `clif.return`
//! - `func.tail_call` -> `clif.return_call`
//! - `func.unreachable` -> `clif.trap`
//! - `func.constant` -> `clif.symbol_addr`

use trunk_ir::arena::context::IrContext;
use trunk_ir::arena::dialect::clif as arena_clif;
use trunk_ir::arena::dialect::func as arena_func;
use trunk_ir::arena::ops::ArenaDialectOp;
use trunk_ir::arena::refs::{OpRef, TypeRef};
use trunk_ir::arena::rewrite::{
    ArenaModule, ArenaRewritePattern, ArenaTypeConverter,
    PatternApplicator as ArenaPatternApplicator, PatternRewriter as ArenaPatternRewriter,
};
use trunk_ir::arena::types::Attribute as ArenaAttribute;
use trunk_ir::dialect::core::Module;
use trunk_ir::dialect::{adt, clif, core, func};
use trunk_ir::rewrite::{
    ConversionError, ConversionTarget, PatternApplicator, PatternRewriter, RewritePattern,
    TypeConverter,
};
use trunk_ir::{Attribute, DialectOp, DialectType, Operation, Symbol, Type};

/// Lower func dialect to clif dialect.
///
/// Returns an error if any `func.*` operations remain after conversion.
///
/// The `type_converter` parameter allows language-specific backends to provide
/// their own type conversion rules.
pub fn lower<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
    type_converter: TypeConverter,
) -> Result<Module<'db>, ConversionError> {
    // Phase 1: Adapt closure structs for native backend.
    // This runs separately because adt.* ops are "legal" in the func->clif target
    // and would be skipped by the main pattern applicator.
    let module = adapt_closure_structs(db, module);

    // Phase 2: Lower func dialect to clif dialect.
    let target = ConversionTarget::new()
        .legal_dialect("clif")
        .illegal_dialect("func");

    Ok(PatternApplicator::new(type_converter)
        .add_pattern(FuncFuncPattern)
        .add_pattern(FuncCallPattern)
        .add_pattern(FuncCallIndirectPattern)
        .add_pattern(FuncReturnPattern)
        .add_pattern(FuncTailCallPattern)
        .add_pattern(FuncUnreachablePattern)
        .add_pattern(FuncConstantPattern)
        .apply(db, module, target)?
        .module)
}

// =============================================================================
// Arena IR version
// =============================================================================

/// Lower func dialect to clif dialect (arena IR).
pub fn lower_arena(ctx: &mut IrContext, module: ArenaModule, type_converter: ArenaTypeConverter) {
    use trunk_ir::arena::rewrite::ArenaConversionTarget;

    // Phase 1: Adapt closure structs for native backend
    adapt_closure_structs_arena(ctx, module);

    // Phase 2: Lower func dialect to clif dialect
    let mut target = ArenaConversionTarget::new();
    target.add_legal_dialect("clif");
    target.add_illegal_dialect("func");

    let applicator = ArenaPatternApplicator::new(type_converter)
        .with_target(target)
        .add_pattern(ArenaFuncFuncPattern)
        .add_pattern(ArenaFuncCallPattern)
        .add_pattern(ArenaFuncCallIndirectPattern)
        .add_pattern(ArenaFuncReturnPattern)
        .add_pattern(ArenaFuncTailCallPattern)
        .add_pattern(ArenaFuncUnreachablePattern)
        .add_pattern(ArenaFuncConstantPattern);
    applicator.apply_partial(ctx, module);
}

fn adapt_closure_structs_arena(ctx: &mut IrContext, module: ArenaModule) {
    let applicator = ArenaPatternApplicator::new(ArenaTypeConverter::new())
        .add_pattern(ArenaClosureStructAdaptPattern);
    applicator.apply_partial(ctx, module);
}

const CLOSURE_STRUCT_NAME_STR: &str = "_closure";

fn is_closure_struct_arena(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.types.get(ty);
    data.attrs
        .get(&Symbol::new("name"))
        .and_then(|a| match a {
            ArenaAttribute::Symbol(s) => Some(*s),
            _ => None,
        })
        .is_some_and(|name| name == Symbol::new(CLOSURE_STRUCT_NAME_STR))
}

fn native_closure_struct_type_arena(ctx: &mut IrContext) -> TypeRef {
    use trunk_ir::arena::types::TypeDataBuilder;
    let i64_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i64")).build());
    let ptr_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ptr")).build());
    let mut builder = TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("struct"));
    builder = builder.param(i64_ty).param(ptr_ty);
    builder = builder.attr(
        "name",
        ArenaAttribute::Symbol(Symbol::new(CLOSURE_STRUCT_NAME_STR)),
    );
    builder = builder.attr(
        "fields",
        ArenaAttribute::List(vec![
            ArenaAttribute::List(vec![
                ArenaAttribute::Symbol(Symbol::new("func_ptr")),
                ArenaAttribute::Type(i64_ty),
            ]),
            ArenaAttribute::List(vec![
                ArenaAttribute::Symbol(Symbol::new("env")),
                ArenaAttribute::Type(ptr_ty),
            ]),
        ]),
    );
    ctx.types.intern(builder.build())
}

fn intern_ptr_type(ctx: &mut IrContext) -> TypeRef {
    use trunk_ir::arena::types::TypeDataBuilder;
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ptr")).build())
}

fn intern_i64_type(ctx: &mut IrContext) -> TypeRef {
    use trunk_ir::arena::types::TypeDataBuilder;
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i64")).build())
}

/// Arena pattern: `func.func` -> `clif.func`
struct ArenaFuncFuncPattern;

impl ArenaRewritePattern for ArenaFuncFuncPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut ArenaPatternRewriter<'_>,
    ) -> bool {
        if arena_func::Func::from_op(ctx, op).is_err() {
            return false;
        }

        let tc = rewriter.type_converter();

        // Convert parameter and return types in the function signature
        let data = ctx.op(op);
        let func_type_attr = data.attributes.get(&Symbol::new("type")).and_then(|a| {
            if let ArenaAttribute::Type(t) = a {
                Some(*t)
            } else {
                None
            }
        });

        let mut new_attrs = data.attributes.clone();
        if let Some(func_ty) = func_type_attr {
            let type_data = ctx.types.get(func_ty);
            if type_data.dialect == Symbol::new("core") && type_data.name == Symbol::new("func") {
                // The arena core.func type may use two layouts:
                // - Layout A: params = [ret, arg1, arg2, ...] (translate_signature_arena format)
                // - Layout B: params = [arg1, arg2, ...], attrs.result = ret
                // We read both and output in Layout A for translate_signature_arena.
                let (arg_params, ret_ty) = if let Some(ArenaAttribute::Type(r)) =
                    type_data.attrs.get(&Symbol::new("result"))
                {
                    // Layout B: return type in attrs
                    (&type_data.params[..], Some(*r))
                } else if !type_data.params.is_empty() {
                    // Layout A: params[0] = return type
                    (&type_data.params[1..], Some(type_data.params[0]))
                } else {
                    (&type_data.params[..], None)
                };

                // Convert params and return type
                let new_params: Vec<TypeRef> = arg_params
                    .iter()
                    .map(|&p| tc.convert_type_or_identity(ctx, p))
                    .collect();
                let new_ret = ret_ty.map(|r| tc.convert_type_or_identity(ctx, r));

                // Build new func type in Layout A: params[0] = return type
                use trunk_ir::arena::types::TypeDataBuilder;
                let mut ft_builder = TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func"));
                if let Some(r) = new_ret {
                    ft_builder = ft_builder.param(r);
                } else {
                    let nil_ty = ctx.types.intern(
                        TypeDataBuilder::new(Symbol::new("core"), Symbol::new("nil")).build(),
                    );
                    ft_builder = ft_builder.param(nil_ty);
                }
                for &p in &new_params {
                    ft_builder = ft_builder.param(p);
                }
                let new_func_ty = ctx.types.intern(ft_builder.build());
                new_attrs.insert(Symbol::new("type"), ArenaAttribute::Type(new_func_ty));
            }
        }

        let new_op = crate::passes::cf_to_clif::rebuild_op_as(
            ctx,
            op,
            Symbol::new("clif"),
            Symbol::new("func"),
        );
        // Patch attributes on the new op
        ctx.op_mut(new_op).attributes = new_attrs;
        // Update dialect/name (already done by rebuild_op_as)
        rewriter.replace_op(new_op);
        true
    }
}

/// Arena pattern: `func.call` -> `clif.call`
struct ArenaFuncCallPattern;

impl ArenaRewritePattern for ArenaFuncCallPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut ArenaPatternRewriter<'_>,
    ) -> bool {
        let Ok(call_op) = arena_func::Call::from_op(ctx, op) else {
            return false;
        };

        let callee = call_op.callee(ctx);
        let new_op = crate::passes::cf_to_clif::rebuild_op_as(
            ctx,
            op,
            Symbol::new("clif"),
            Symbol::new("call"),
        );
        ctx.op_mut(new_op)
            .attributes
            .insert(Symbol::new("callee"), ArenaAttribute::Symbol(callee));
        rewriter.replace_op(new_op);
        true
    }
}

/// Arena pattern: `func.call_indirect` -> `clif.call_indirect`
struct ArenaFuncCallIndirectPattern;

impl ArenaRewritePattern for ArenaFuncCallIndirectPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut ArenaPatternRewriter<'_>,
    ) -> bool {
        if arena_func::CallIndirect::from_op(ctx, op).is_err() {
            return false;
        }

        let operands = ctx.op_operands(op).to_vec();

        // Collect arg types (skip operand 0 = callee).
        // Operand types are already converted by the applicator's cast insertion.
        let param_types: Vec<TypeRef> = operands[1..].iter().map(|&v| ctx.value_ty(v)).collect();

        // Result type with conversion applied
        let result_ty = rewriter.result_type(ctx, op, 0);

        // Build sig type matching translate_signature_arena layout:
        // params[0] = return type, params[1..] = parameter types
        use trunk_ir::arena::types::TypeDataBuilder;
        let mut sig_builder = TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func"));
        if let Some(r) = result_ty {
            sig_builder = sig_builder.param(r);
        } else {
            // nil return type
            let nil_ty = ctx
                .types
                .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("nil")).build());
            sig_builder = sig_builder.param(nil_ty);
        }
        for &p in &param_types {
            sig_builder = sig_builder.param(p);
        }
        let sig_ty = ctx.types.intern(sig_builder.build());

        let new_op = crate::passes::cf_to_clif::rebuild_op_as(
            ctx,
            op,
            Symbol::new("clif"),
            Symbol::new("call_indirect"),
        );
        ctx.op_mut(new_op)
            .attributes
            .insert(Symbol::new("sig"), ArenaAttribute::Type(sig_ty));
        rewriter.replace_op(new_op);
        true
    }
}

/// Arena pattern: `func.return` -> `clif.return`
struct ArenaFuncReturnPattern;

impl ArenaRewritePattern for ArenaFuncReturnPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut ArenaPatternRewriter<'_>,
    ) -> bool {
        if arena_func::Return::from_op(ctx, op).is_err() {
            return false;
        }
        let new_op = crate::passes::cf_to_clif::rebuild_op_as(
            ctx,
            op,
            Symbol::new("clif"),
            Symbol::new("return"),
        );
        rewriter.replace_op(new_op);
        true
    }
}

/// Arena pattern: `func.tail_call` -> `clif.return_call`
struct ArenaFuncTailCallPattern;

impl ArenaRewritePattern for ArenaFuncTailCallPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut ArenaPatternRewriter<'_>,
    ) -> bool {
        let Ok(tail_call) = arena_func::TailCall::from_op(ctx, op) else {
            return false;
        };

        let callee = tail_call.callee(ctx);
        let new_op = crate::passes::cf_to_clif::rebuild_op_as(
            ctx,
            op,
            Symbol::new("clif"),
            Symbol::new("return_call"),
        );
        ctx.op_mut(new_op)
            .attributes
            .insert(Symbol::new("callee"), ArenaAttribute::Symbol(callee));
        rewriter.replace_op(new_op);
        true
    }
}

/// Arena pattern: `func.unreachable` -> `clif.trap`
struct ArenaFuncUnreachablePattern;

impl ArenaRewritePattern for ArenaFuncUnreachablePattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut ArenaPatternRewriter<'_>,
    ) -> bool {
        if arena_func::Unreachable::from_op(ctx, op).is_err() {
            return false;
        }
        let loc = ctx.op(op).location;
        let new_op = arena_clif::trap(ctx, loc, Symbol::new("unreachable"));
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// Arena pattern: `func.constant` -> `clif.symbol_addr`
struct ArenaFuncConstantPattern;

impl ArenaRewritePattern for ArenaFuncConstantPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut ArenaPatternRewriter<'_>,
    ) -> bool {
        let Ok(const_op) = arena_func::Constant::from_op(ctx, op) else {
            return false;
        };

        let func_ref = const_op.func_ref(ctx);
        let loc = ctx.op(op).location;
        let ptr_ty = intern_ptr_type(ctx);
        let new_op = arena_clif::symbol_addr(ctx, loc, ptr_ty, func_ref);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// Arena pattern: Adapt `_closure` struct ops for native backend
struct ArenaClosureStructAdaptPattern;

impl ArenaRewritePattern for ArenaClosureStructAdaptPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut ArenaPatternRewriter<'_>,
    ) -> bool {
        use trunk_ir::arena::dialect::adt as arena_adt;

        let native_ty = native_closure_struct_type_arena(ctx);

        // Handle adt.struct_new on _closure
        if let Ok(struct_new) = arena_adt::StructNew::from_op(ctx, op) {
            let ty = struct_new.r#type(ctx);
            if !is_closure_struct_arena(ctx, ty) {
                return false;
            }
            let new_op = crate::passes::cf_to_clif::rebuild_op_as(
                ctx,
                op,
                Symbol::new("adt"),
                Symbol::new("struct_new"),
            );
            ctx.op_mut(new_op)
                .attributes
                .insert(Symbol::new("type"), ArenaAttribute::Type(native_ty));
            // Update result type to native_ty
            let result_types = ctx.op_result_types(new_op).to_vec();
            if !result_types.is_empty() {
                ctx.set_op_result_type(new_op, 0, native_ty);
            }
            rewriter.replace_op(new_op);
            return true;
        }

        // Handle adt.struct_get on _closure
        if let Ok(struct_get) = arena_adt::StructGet::from_op(ctx, op) {
            let ty = struct_get.r#type(ctx);
            if !is_closure_struct_arena(ctx, ty) {
                return false;
            }
            let field_idx = struct_get.field(ctx);
            let new_op = crate::passes::cf_to_clif::rebuild_op_as(
                ctx,
                op,
                Symbol::new("adt"),
                Symbol::new("struct_get"),
            );
            ctx.op_mut(new_op)
                .attributes
                .insert(Symbol::new("type"), ArenaAttribute::Type(native_ty));
            if field_idx == 0 {
                let i64_ty = intern_i64_type(ctx);
                ctx.set_op_result_type(new_op, 0, i64_ty);
            } else if field_idx == 1 {
                let ptr_ty = intern_ptr_type(ctx);
                ctx.set_op_result_type(new_op, 0, ptr_ty);
            }
            rewriter.replace_op(new_op);
            return true;
        }

        false
    }
}

// =============================================================================
// Salsa IR version
// =============================================================================

/// Adapt closure struct operations for the native backend.
///
/// Runs as a pre-processing step with no legality constraints so that
/// `adt.*` operations on `_closure` structs are visited by the pattern.
fn adapt_closure_structs<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    let target = ConversionTarget::new();
    PatternApplicator::new(TypeConverter::new())
        .add_pattern(ClosureStructAdaptPattern)
        .apply_partial(db, module, target)
        .module
}

/// Pattern for `func.func` -> `clif.func`
///
/// Converts the function type attribute's parameter and return types using
/// the type converter, ensuring high-level types (e.g., `core.array`) are
/// mapped to their native representations before Cranelift emission.
struct FuncFuncPattern;

impl<'db> RewritePattern<'db> for FuncFuncPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        let Ok(func_op) = func::Func::from_operation(db, *op) else {
            return false;
        };

        let type_converter = rewriter.type_converter();
        let func_type_attr = func_op.r#type(db);

        // Convert parameter and return types in the function signature
        let mut builder = op.modify(db).dialect_str("clif").name_str("func");
        if let Some(func_ty) = core::Func::from_type(db, func_type_attr) {
            let new_params = type_converter.convert_types(db, &func_ty.params(db));
            let new_ret = type_converter
                .convert_type(db, func_ty.result(db))
                .unwrap_or(func_ty.result(db));
            let new_func_ty = core::Func::new(db, new_params, new_ret).as_type();
            builder = builder.attr("type", Attribute::Type(new_func_ty));
        }

        rewriter.replace_op(builder.build());
        true
    }
}

/// Pattern for `func.call` -> `clif.call`
struct FuncCallPattern;

impl<'db> RewritePattern<'db> for FuncCallPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        let Ok(call_op) = func::Call::from_operation(db, *op) else {
            return false;
        };

        let new_op = op
            .modify(db)
            .dialect_str("clif")
            .name_str("call")
            .attr("callee", Attribute::Symbol(call_op.callee(db)))
            .build();

        rewriter.replace_op(new_op);
        true
    }
}

/// Pattern for `func.call_indirect` -> `clif.call_indirect`
///
/// Constructs the required `sig` attribute by collecting operand/result types.
/// `func.call_indirect` has operands: [callee, args...] and one result.
/// The `sig` is a `core.func` type with param types from args and the result type.
struct FuncCallIndirectPattern;

impl<'db> RewritePattern<'db> for FuncCallIndirectPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        let Ok(_call_indirect) = func::CallIndirect::from_operation(db, *op) else {
            return false;
        };

        // Collect argument types (skip operand 0 which is the callee).
        // Bail out if any type is unavailable so the ConversionTarget can report the unconverted op.
        let mut param_types = Vec::new();
        for i in 1..rewriter.num_operands() {
            let Some(ty) = rewriter.operand_type(i) else {
                return false;
            };
            param_types.push(ty);
        }

        let Some(result_ty) = rewriter.result_type(db, op, 0) else {
            return false;
        };
        let sig_ty = core::Func::new(db, param_types.into(), result_ty).as_type();

        let new_op = op
            .modify(db)
            .dialect_str("clif")
            .name_str("call_indirect")
            .attr("sig", Attribute::Type(sig_ty))
            .build();

        rewriter.replace_op(new_op);
        true
    }
}

/// Pattern for `func.return` -> `clif.return`
struct FuncReturnPattern;

impl<'db> RewritePattern<'db> for FuncReturnPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        let Ok(_return_op) = func::Return::from_operation(db, *op) else {
            return false;
        };

        let new_op = op.modify(db).dialect_str("clif").name_str("return").build();

        rewriter.replace_op(new_op);
        true
    }
}

/// Pattern for `func.tail_call` -> `clif.return_call`
struct FuncTailCallPattern;

impl<'db> RewritePattern<'db> for FuncTailCallPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        let Ok(tail_call_op) = func::TailCall::from_operation(db, *op) else {
            return false;
        };

        let new_op = op
            .modify(db)
            .dialect_str("clif")
            .name_str("return_call")
            .attr("callee", Attribute::Symbol(tail_call_op.callee(db)))
            .build();

        rewriter.replace_op(new_op);
        true
    }
}

/// Pattern for `func.unreachable` -> `clif.trap`
struct FuncUnreachablePattern;

impl<'db> RewritePattern<'db> for FuncUnreachablePattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        let Ok(_unreachable_op) = func::Unreachable::from_operation(db, *op) else {
            return false;
        };

        let new_op = clif::trap(db, op.location(db), Symbol::new("unreachable"));
        rewriter.replace_op(new_op.as_operation());
        true
    }
}

/// Pattern for `func.constant` -> `clif.symbol_addr`
///
/// In the Cranelift backend, function references are symbol addresses (pointers)
/// rather than table indices (unlike WASM). The result type is `core.ptr`.
struct FuncConstantPattern;

impl<'db> RewritePattern<'db> for FuncConstantPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        let Ok(const_op) = func::Constant::from_operation(db, *op) else {
            return false;
        };

        let ptr_ty = core::Ptr::new(db).as_type();
        let new_op = clif::symbol_addr(db, op.location(db), ptr_ty, const_op.func_ref(db));
        rewriter.replace_op(new_op.as_operation());
        true
    }
}

// =============================================================================
// Closure struct adaptation for native backend
// =============================================================================

const CLOSURE_STRUCT_NAME: &str = "_closure";

/// Check if a type is the closure struct type (name == "_closure").
fn is_closure_struct(db: &dyn salsa::Database, ty: Type<'_>) -> bool {
    adt::get_type_name(db, ty)
        .map(|name| name == Symbol::new(CLOSURE_STRUCT_NAME))
        .unwrap_or(false)
}

/// Create the native closure struct type: `{ func_ptr: i64, env: ptr }`.
///
/// In the native backend, both fields are 64-bit:
/// - field 0: function pointer (from `clif.symbol_addr`) — typed `i64` rather
///   than `ptr` so the RC insertion pass does NOT retain/release it (function
///   pointers are code addresses, not heap-allocated objects)
/// - field 1: environment pointer (boxed captures)
fn native_closure_struct_type(db: &dyn salsa::Database) -> Type<'_> {
    let i64_ty = core::I64::new(db).as_type();
    let ptr_ty = core::Ptr::new(db).as_type();
    adt::struct_type(
        db,
        Symbol::new(CLOSURE_STRUCT_NAME),
        vec![
            (Symbol::new("func_ptr"), i64_ty),
            (Symbol::new("env"), ptr_ty),
        ],
    )
}

/// Adapt closure struct operations for the native backend.
///
/// The shared `closure_lower` pass produces `_closure { i32, anyref }` structs
/// (designed for WASM). For native codegen, we need `_closure { i64, ptr }`:
/// - `adt.struct_new` on `_closure`: change type attribute to `{ i64, ptr }`
/// - `adt.struct_get` on `_closure` field 0: change result type to `i64`
///   (NOT `ptr`, so the RC pass doesn't retain/release function pointers)
/// - `adt.struct_get` on `_closure` field 1: change result type to `ptr`
struct ClosureStructAdaptPattern;

impl<'db> RewritePattern<'db> for ClosureStructAdaptPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        let native_ty = native_closure_struct_type(db);

        // Handle adt.struct_new on _closure
        if let Ok(struct_new) = adt::StructNew::from_operation(db, *op) {
            let ty = struct_new.r#type(db);
            if !is_closure_struct(db, ty) {
                return false;
            }
            let new_op = op
                .modify(db)
                .attr("type", Attribute::Type(native_ty))
                .results(vec![native_ty].into())
                .build();
            rewriter.replace_op(new_op);
            return true;
        }

        // Handle adt.struct_get on _closure
        if let Ok(struct_get) = adt::StructGet::from_operation(db, *op) {
            let ty = struct_get.r#type(db);
            if !is_closure_struct(db, ty) {
                return false;
            }
            let field_idx = struct_get.field(db);
            let mut builder = op.modify(db).attr("type", Attribute::Type(native_ty));
            if field_idx == 0 {
                // func_ptr: i64 (not ptr — avoids RC tracking of function pointers)
                let i64_ty = core::I64::new(db).as_type();
                builder = builder.results(vec![i64_ty].into());
            } else if field_idx == 1 {
                // env: ptr (heap-allocated captures — needs RC tracking)
                let ptr_ty = core::Ptr::new(db).as_type();
                builder = builder.results(vec![ptr_ty].into());
            }
            let new_op = builder.build();
            rewriter.replace_op(new_op);
            return true;
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use insta::assert_snapshot;
    use salsa_test_macros::salsa_test;
    use trunk_ir::dialect::{arith, core, wasm};
    use trunk_ir::{Attribute, Block, BlockId, DialectType, Location, PathId, Region, Span, idvec};

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    fn test_converter() -> TypeConverter {
        TypeConverter::new()
    }

    /// Format module operations for snapshot testing
    fn format_module_ops(db: &dyn salsa::Database, module: &Module<'_>) -> String {
        let body = module.body(db);
        let ops = &body.blocks(db)[0].operations(db);
        ops.iter()
            .map(|op| format_op(db, op, 0))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn format_op(db: &dyn salsa::Database, op: &Operation<'_>, indent: usize) -> String {
        let prefix = "  ".repeat(indent);
        let name = op.full_name(db);
        let operands = op.operands(db);
        let results = op.results(db);
        let attrs = op.attributes(db);

        let mut parts = vec![name];

        for (key, attr) in attrs.iter() {
            match attr {
                Attribute::Symbol(s)
                    if *key == "callee"
                        || *key == "sym_name"
                        || *key == "code"
                        || *key == "sym"
                        || *key == "func_ref" =>
                {
                    parts.push(format!("{}={}", key, s));
                }
                Attribute::Type(ty) if *key == "sig" || *key == "type" => {
                    parts.push(format!("{}={}", key, ty.name(db)));
                }
                Attribute::IntBits(n) if *key == "field" => {
                    parts.push(format!("field={}", n));
                }
                _ => {}
            }
        }

        if !operands.is_empty() {
            parts.push(format!("operands={}", operands.len()));
        }

        if !results.is_empty() {
            let result_types: Vec<_> = results.iter().map(|t| t.name(db).to_string()).collect();
            parts.push(format!("-> {}", result_types.join(", ")));
        }

        let mut result = format!("{}{}", prefix, parts.join(" "));

        // Recurse into regions
        for region in op.regions(db).iter() {
            for block in region.blocks(db).iter() {
                for nested_op in block.operations(db).iter() {
                    result.push('\n');
                    result.push_str(&format_op(db, nested_op, indent + 1));
                }
            }
        }

        result
    }

    #[salsa::tracked]
    fn make_func_call_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        let func_call = func::call(db, location, vec![], i32_ty, Symbol::new("foo"));

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![func_call.as_operation()],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa::tracked]
    fn make_func_func_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let nil_ty = core::Nil::new(db).as_type();
        let func_ty = core::Func::new(db, idvec![], nil_ty).as_type();

        let func_return = func::r#return(db, location, vec![]);

        let body_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![func_return.as_operation()],
        );
        let body_region = Region::new(db, location, idvec![body_block]);

        let func_func =
            func::func(db, location, Symbol::new("test_fn"), func_ty, body_region).as_operation();

        let block = Block::new(db, BlockId::fresh(), location, idvec![], idvec![func_func]);
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa::tracked]
    fn format_lowered_module<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> String {
        let lowered = lower(db, module, test_converter()).expect("conversion should succeed");
        format_module_ops(db, &lowered)
    }

    #[salsa_test]
    fn test_func_call_to_clif(db: &salsa::DatabaseImpl) {
        let module = make_func_call_module(db);
        let formatted = format_lowered_module(db, module);

        assert_snapshot!(formatted, @"clif.call callee=foo -> i32");
    }

    #[salsa_test]
    fn test_func_func_to_clif(db: &salsa::DatabaseImpl) {
        let module = make_func_func_module(db);
        let formatted = format_lowered_module(db, module);

        assert_snapshot!(formatted);
    }

    #[salsa::tracked]
    fn make_func_constant_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let func_ty = core::Func::new(db, idvec![], core::Nil::new(db).as_type()).as_type();

        let func_constant = func::constant(db, location, func_ty, Symbol::new("test_func"));

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![func_constant.as_operation()],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_func_constant_to_clif(db: &salsa::DatabaseImpl) {
        let module = make_func_constant_module(db);
        let formatted = format_lowered_module(db, module);

        assert_snapshot!(formatted, @"clif.symbol_addr sym=test_func -> ptr");
    }

    #[salsa::tracked]
    fn make_func_unreachable_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);

        let unreachable_op = func::unreachable(db, location);

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![unreachable_op.as_operation()],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_func_unreachable_to_clif(db: &salsa::DatabaseImpl) {
        let module = make_func_unreachable_module(db);
        let formatted = format_lowered_module(db, module);

        assert_snapshot!(formatted, @"clif.trap code=unreachable");
    }

    #[salsa::tracked]
    fn make_call_indirect_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        let callee_op = arith::r#const(db, location, i32_ty, Attribute::IntBits(0));
        let callee_val = callee_op.result(db);

        let arg_op = arith::r#const(db, location, i32_ty, Attribute::IntBits(42));
        let arg_val = arg_op.result(db);

        let call_indirect = func::call_indirect(db, location, callee_val, vec![arg_val], i32_ty);

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![
                callee_op.as_operation(),
                arg_op.as_operation(),
                call_indirect.as_operation()
            ],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_call_indirect_to_clif(db: &salsa::DatabaseImpl) {
        let module = make_call_indirect_module(db);
        let formatted = format_lowered_module(db, module);

        // func.call_indirect should become clif.call_indirect
        // arith.const ops should remain unchanged (different dialect)
        assert_snapshot!(formatted);
    }

    #[salsa::tracked]
    fn make_func_tail_call_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);

        let tail_call = func::tail_call(db, location, vec![], Symbol::new("target_fn"));

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![tail_call.as_operation()],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_func_tail_call_to_clif(db: &salsa::DatabaseImpl) {
        let module = make_func_tail_call_module(db);
        let formatted = format_lowered_module(db, module);

        assert_snapshot!(formatted, @"clif.return_call callee=target_fn");
    }

    // =========================================================================
    // Closure struct adaptation tests
    // =========================================================================

    /// Create a WASM-style closure struct type: `{ table_idx: i32, env: anyref }`
    fn wasm_closure_struct_type(db: &dyn salsa::Database) -> trunk_ir::Type<'_> {
        let i32_ty = core::I32::new(db).as_type();
        // Use core.ptr as a stand-in for anyref in tests
        let anyref_ty = core::Ptr::new(db).as_type();
        adt::struct_type(
            db,
            Symbol::new("_closure"),
            vec![
                (Symbol::new("table_idx"), i32_ty),
                (Symbol::new("env"), anyref_ty),
            ],
        )
    }

    #[salsa::tracked]
    fn make_closure_struct_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();
        let ptr_ty = core::Ptr::new(db).as_type();
        let closure_ty = wasm_closure_struct_type(db);

        // Simulate closure creation: func.constant + adt.struct_new
        let func_const = func::constant(db, location, i32_ty, Symbol::new("lifted_fn"));
        let func_ptr_val = func_const.result(db);

        let env_op = arith::r#const(db, location, ptr_ty, Attribute::IntBits(0));
        let env_val = env_op.result(db);

        let struct_new = adt::struct_new(
            db,
            location,
            vec![func_ptr_val, env_val],
            closure_ty,
            closure_ty,
        );
        let closure_val = struct_new.result(db);

        // Extract func ptr (field 0) and env (field 1)
        let get_func = adt::struct_get(db, location, closure_val, i32_ty, closure_ty, 0);
        let func_ptr = get_func.result(db);

        let get_env = adt::struct_get(db, location, closure_val, ptr_ty, closure_ty, 1);
        let env = get_env.result(db);

        // Indirect call through the closure
        let call_result_ty = i32_ty;
        let call_indirect = func::call_indirect(db, location, func_ptr, vec![env], call_result_ty);

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![
                func_const.as_operation(),
                env_op.as_operation(),
                struct_new.as_operation(),
                get_func.as_operation(),
                get_env.as_operation(),
                call_indirect.as_operation()
            ],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_closure_struct_adaptation(db: &salsa::DatabaseImpl) {
        let module = make_closure_struct_module(db);
        let formatted = format_lowered_module(db, module);

        // func.constant -> clif.symbol_addr with ptr result
        // adt.struct_new on _closure -> type adapted to { ptr, ptr }
        // adt.struct_get field 0 -> result type becomes ptr
        // func.call_indirect -> clif.call_indirect with sig attribute
        assert_snapshot!(formatted);
    }

    // Note: The previous test `test_call_indirect_unchanged_on_missing_types` was
    // removed during the OpAdaptor→PatternRewriter migration. It tested internal
    // behavior by manually constructing an OpAdaptor with None operand types.
    // With the new PatternRewriter API (whose constructor is pub(crate) to trunk_ir),
    // this white-box test cannot be replicated from outside the crate. The behavior
    // is still upheld by the pattern implementation (bail on missing types).

    /// Test closure struct adaptation with actual `wasm.anyref` env type.
    ///
    /// Verifies that `adt.struct_get` on field 1 (env) produces `ptr` result
    /// even when the original type is `wasm.anyref`.
    #[salsa::tracked]
    fn make_closure_struct_anyref_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();
        let anyref_ty = wasm::Anyref::new(db).as_type();

        // WASM-style closure struct: { table_idx: i32, env: anyref }
        let closure_ty = adt::struct_type(
            db,
            Symbol::new("_closure"),
            vec![
                (Symbol::new("table_idx"), i32_ty),
                (Symbol::new("env"), anyref_ty),
            ],
        );

        let func_const = func::constant(db, location, i32_ty, Symbol::new("lifted_fn"));
        let func_ptr_val = func_const.result(db);

        let env_op = arith::r#const(db, location, anyref_ty, Attribute::IntBits(0));
        let env_val = env_op.result(db);

        let struct_new = adt::struct_new(
            db,
            location,
            vec![func_ptr_val, env_val],
            closure_ty,
            closure_ty,
        );
        let closure_val = struct_new.result(db);

        // Extract func ptr (field 0) and env (field 1)
        let get_func = adt::struct_get(db, location, closure_val, i32_ty, closure_ty, 0);
        let func_ptr = get_func.result(db);

        let get_env = adt::struct_get(db, location, closure_val, anyref_ty, closure_ty, 1);
        let env = get_env.result(db);

        let call_indirect = func::call_indirect(db, location, func_ptr, vec![env], i32_ty);

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![
                func_const.as_operation(),
                env_op.as_operation(),
                struct_new.as_operation(),
                get_func.as_operation(),
                get_env.as_operation(),
                call_indirect.as_operation()
            ],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_closure_struct_anyref_adaptation(db: &salsa::DatabaseImpl) {
        let module = make_closure_struct_anyref_module(db);
        let formatted = format_lowered_module(db, module);

        // Both struct_get field 0 and field 1 should produce ptr results
        assert_snapshot!(formatted);
    }
}
