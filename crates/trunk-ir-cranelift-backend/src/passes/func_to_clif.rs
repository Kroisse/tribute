//! Lower func dialect operations to clif dialect.
//!
//! This pass converts function-level operations to Cranelift equivalents:
//! - `func.func` -> `clif.func`
//! - `func.call` -> `clif.call`
//! - `func.call_indirect` -> `clif.call_indirect`
//! - `func.tail_call` -> `clif.return_call`
//! - `func.tail_call_indirect` -> `clif.return_call_indirect`
//! - `func.return` -> `clif.return`
//! - `func.unreachable` -> `clif.trap`
//! - `func.constant` -> `clif.symbol_addr`

use std::collections::HashSet;
use std::ops::ControlFlow;

use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::clif;
use trunk_ir::dialect::core;
use trunk_ir::dialect::func::{self, CallLike, TailCallLike};
use trunk_ir::op_interface::IndirectCallLikeModel;
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{OpRef, TypeRef};
use trunk_ir::rewrite::{
    ConversionError, ConversionTarget, Module, PatternApplicator, PatternRewriter, RewritePattern,
    TypeConverter,
};
use trunk_ir::types::Attribute;
use trunk_ir::walk::{WalkAction, walk_region};

use crate::function::{CPS_CALLING_CONVENTION, TRIBUTE_CALLING_CONVENTION_ATTR};

/// An exact type-identity rewrite performed by `func_to_clif`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TypeRewrite {
    pub source: TypeRef,
    pub target: TypeRef,
}

/// Stable identities rewritten while lowering function-level representation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LoweringResult {
    rtti_layout_rewrites: Vec<TypeRewrite>,
}

impl LoweringResult {
    pub fn rtti_layout_rewrites(&self) -> &[TypeRewrite] {
        &self.rtti_layout_rewrites
    }
}

/// Lower func dialect to clif dialect.
pub fn lower(
    ctx: &mut IrContext,
    module: Module,
    type_converter: TypeConverter,
) -> Result<LoweringResult, ConversionError> {
    // Phase 1: Adapt closure structs for native backend
    let rtti_layout_rewrites = adapt_closure_structs(ctx, module);

    // Phase 2: Lower func dialect to clif dialect
    let applicator = PatternApplicator::new(type_converter)
        .with_auto_type_conversion(true)
        .add_pattern(FuncFuncPattern)
        .add_pattern(FuncCallPattern)
        .add_pattern(FuncCallIndirectPattern)
        .add_pattern(FuncReturnPattern)
        .add_pattern(FuncTailCallPattern)
        .add_pattern(FuncTailCallIndirectPattern)
        .add_pattern(FuncUnreachablePattern)
        .add_pattern(FuncConstantPattern)
        .with_target(func_to_clif_target());
    applicator.apply_partial_conversion(ctx, module, "func-to-clif")?;
    Ok(LoweringResult {
        rtti_layout_rewrites,
    })
}

fn func_to_clif_target() -> ConversionTarget {
    ConversionTarget::new()
        .legal_dialect("clif")
        .illegal_dialect("func")
}

fn adapt_closure_structs(ctx: &mut IrContext, module: Module) -> Vec<TypeRewrite> {
    let native_ty = native_closure_struct_type(ctx);
    let mut sources = HashSet::new();
    if let Some(body) = module.body(ctx) {
        let _ = walk_region::<()>(ctx, body, &mut |op| {
            if let Ok(struct_new) = trunk_ir::dialect::adt::StructNew::from_op(ctx, op) {
                let source = struct_new.r#type(ctx);
                if source != native_ty && is_closure_struct(ctx, source) {
                    sources.insert(source);
                }
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
    }
    let mut rtti_layout_rewrites = sources
        .into_iter()
        .map(|source| TypeRewrite {
            source,
            target: native_ty,
        })
        .collect::<Vec<_>>();
    rtti_layout_rewrites.sort_by_key(|rewrite| rewrite.source);

    let applicator =
        PatternApplicator::new(TypeConverter::new()).add_pattern(ClosureStructAdaptPattern);
    applicator.apply_partial(ctx, module);
    rtti_layout_rewrites
}

const CLOSURE_STRUCT_NAME_STR: &str = "_closure";

fn is_closure_struct(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.types.get(ty);
    data.attrs
        .get_symbol("name")
        .is_some_and(|name| name == Symbol::new(CLOSURE_STRUCT_NAME_STR))
}

fn native_closure_struct_type(ctx: &mut IrContext) -> TypeRef {
    use trunk_ir::types::TypeDataBuilder;
    let i64_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i64")).build());
    let ptr_ty = core::ptr(ctx).as_type_ref();
    let mut builder = TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("struct"));
    builder = builder.param(i64_ty).param(ptr_ty);
    builder = builder.attr(
        "name",
        Attribute::Symbol(Symbol::new(CLOSURE_STRUCT_NAME_STR)),
    );
    builder = builder.attr(
        "fields",
        Attribute::List(vec![
            Attribute::List(vec![
                Attribute::Symbol(Symbol::new("func_ptr")),
                Attribute::Type(i64_ty),
            ]),
            Attribute::List(vec![
                Attribute::Symbol(Symbol::new("env")),
                Attribute::Type(ptr_ty),
            ]),
        ]),
    );
    ctx.types.intern(builder.build())
}

fn intern_ptr_type(ctx: &mut IrContext) -> TypeRef {
    core::ptr(ctx).as_type_ref()
}

fn intern_i64_type(ctx: &mut IrContext) -> TypeRef {
    use trunk_ir::types::TypeDataBuilder;
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i64")).build())
}

/// Pattern: `func.func` -> `clif.func`
struct FuncFuncPattern;

impl RewritePattern for FuncFuncPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if func::Func::from_op(ctx, op).is_err() {
            return false;
        }

        let tc = rewriter.type_converter();

        // Convert parameter and return types in the function signature
        let data = ctx.op(op);
        let func_type_attr = data.attributes.get_type("type");

        let mut new_attrs = data.attributes.clone();
        if let Some(func_ty) = func_type_attr
            && let Some(new_func_ty) = trunk_ir::rewrite::convert_function_type(ctx, func_ty, tc)
        {
            new_attrs.insert(Symbol::new("type"), Attribute::Type(new_func_ty));
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

/// Pattern: `func.call` -> `clif.call`
struct FuncCallPattern;

impl RewritePattern for FuncCallPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(call_op) = func::Call::from_op(ctx, op) else {
            return false;
        };

        let callee = call_op.callee(ctx);
        let new_op = crate::passes::cf_to_clif::rebuild_op_as(
            ctx,
            op,
            Symbol::new("clif"),
            Symbol::new("call"),
        );
        for (index, result_ty) in rewriter.result_types(ctx, op).into_iter().enumerate() {
            ctx.set_op_result_type(new_op, index as u32, result_ty);
        }
        ctx.op_mut(new_op)
            .attributes
            .insert(Symbol::new("callee"), Attribute::Symbol(callee));
        rewriter.replace_op(new_op);
        true
    }
}

/// Pattern: `func.call_indirect` -> `clif.call_indirect`
struct FuncCallIndirectPattern;

impl RewritePattern for FuncCallIndirectPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(call) = func::CallIndirect::from_op(ctx, op) else {
            return false;
        };

        let result_types = rewriter.result_types(ctx, op);
        let sig_ty = if let Some(signature) = call.exact_signature(ctx) {
            let Some(signature) =
                trunk_ir::rewrite::convert_function_type(ctx, signature, rewriter.type_converter())
            else {
                return false;
            };
            let Some(callable) = core::Func::from_type_ref(ctx, signature) else {
                return false;
            };
            let Some(callable_result) = callable.single_result(ctx) else {
                return false;
            };
            let results_match = if crate::function::is_nil_type(ctx, callable_result) {
                result_types.is_empty()
            } else {
                result_types == [callable_result]
            };
            if callable.inputs(ctx).len() != CallLike::call_args(&call, ctx).len()
                || callable
                    .inputs(ctx)
                    .iter()
                    .zip(CallLike::call_args(&call, ctx))
                    .any(|(&param, &arg)| param != ctx.value_ty(arg))
                || !results_match
            {
                return false;
            }
            signature
        } else {
            let param_types: Vec<TypeRef> = CallLike::call_args(&call, ctx)
                .iter()
                .map(|&value| ctx.value_ty(value))
                .collect();
            let result = result_types
                .first()
                .copied()
                .unwrap_or_else(|| core::nil(ctx).as_type_ref());
            core::func(ctx, param_types, [result]).as_type_ref()
        };

        let new_op = crate::passes::cf_to_clif::rebuild_op_as(
            ctx,
            op,
            Symbol::new("clif"),
            Symbol::new("call_indirect"),
        );
        func::remove_indirect_call_signature(&mut ctx.op_mut(new_op).attributes);
        if !clif::set_indirect_call_signature(ctx, new_op, sig_ty) {
            return false;
        }
        for (index, result_ty) in result_types.into_iter().enumerate() {
            ctx.set_op_result_type(new_op, index as u32, result_ty);
        }
        rewriter.replace_op(new_op);
        true
    }
}

/// Pattern: `func.return` -> `clif.return`
struct FuncReturnPattern;

impl RewritePattern for FuncReturnPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if func::Return::from_op(ctx, op).is_err() {
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

/// Pattern: `func.tail_call` -> `clif.return_call`
struct FuncTailCallPattern;

impl RewritePattern for FuncTailCallPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(tail_call) = func::TailCall::from_op(ctx, op) else {
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
            .insert(Symbol::new("callee"), Attribute::Symbol(callee));
        rewriter.replace_op(new_op);
        true
    }
}

/// Pattern: `func.tail_call_indirect` -> `clif.return_call_indirect`.
///
/// The physical closure lowering boundary records the exact callable ABI on
/// the transfer.  Do not infer a signature from the function pointer: after
/// closure lowering it is untyped at the TrunkIR level.
struct FuncTailCallIndirectPattern;

impl RewritePattern for FuncTailCallIndirectPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(tail) = func::TailCallIndirect::from_op(ctx, op) else {
            return false;
        };
        if ctx
            .op(op)
            .attributes
            .get_u8(TRIBUTE_CALLING_CONVENTION_ATTR)
            != Ok(Some(CPS_CALLING_CONVENTION))
        {
            return false;
        }

        let Some(signature) = tail.exact_signature(ctx) else {
            return false;
        };
        let Some(signature) =
            trunk_ir::rewrite::convert_function_type(ctx, signature, rewriter.type_converter())
        else {
            return false;
        };
        let Some(callable) = core::Func::from_type_ref(ctx, signature) else {
            return false;
        };
        if callable.single_result(ctx) != Some(core::nil(ctx).as_type_ref())
            || !TailCallLike::is_resultless(&tail, ctx)
            || callable.inputs(ctx).len() != CallLike::call_args(&tail, ctx).len()
            || callable
                .inputs(ctx)
                .iter()
                .zip(CallLike::call_args(&tail, ctx))
                .any(|(&param, &arg)| param != ctx.value_ty(arg))
        {
            return false;
        }

        let new_op = crate::passes::cf_to_clif::rebuild_op_as(
            ctx,
            op,
            Symbol::new("clif"),
            Symbol::new("return_call_indirect"),
        );
        func::remove_indirect_call_signature(&mut ctx.op_mut(new_op).attributes);
        if !clif::set_indirect_call_signature(ctx, new_op, signature) {
            return false;
        }
        rewriter.replace_op(new_op);
        true
    }
}

/// Pattern: `func.unreachable` -> `clif.trap`
struct FuncUnreachablePattern;

impl RewritePattern for FuncUnreachablePattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if func::Unreachable::from_op(ctx, op).is_err() {
            return false;
        }
        let loc = ctx.op(op).location;
        let new_op = clif::trap(ctx, loc, Symbol::new("unreachable"));
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// Pattern: `func.constant` -> `clif.symbol_addr`
struct FuncConstantPattern;

impl RewritePattern for FuncConstantPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(const_op) = func::Constant::from_op(ctx, op) else {
            return false;
        };

        let func_ref = const_op.func_ref(ctx);
        let loc = ctx.op(op).location;
        let ptr_ty = intern_ptr_type(ctx);
        let new_op = clif::symbol_addr(ctx, loc, ptr_ty, func_ref);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// Pattern: Adapt `_closure` struct ops for native backend
struct ClosureStructAdaptPattern;

impl RewritePattern for ClosureStructAdaptPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        use trunk_ir::dialect::adt;

        let native_ty = native_closure_struct_type(ctx);

        // Handle adt.struct_new on _closure
        if let Ok(struct_new) = adt::StructNew::from_op(ctx, op) {
            let ty = struct_new.r#type(ctx);
            if !is_closure_struct(ctx, ty) || ty == native_ty {
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
                .insert(Symbol::new("type"), Attribute::Type(native_ty));
            // Update result type to native_ty
            let result_types = ctx.op_result_types(new_op).to_vec();
            if !result_types.is_empty() {
                ctx.set_op_result_type(new_op, 0, native_ty);
            }
            rewriter.replace_op(new_op);
            return true;
        }

        // Handle adt.struct_get on _closure
        if let Ok(struct_get) = adt::StructGet::from_op(ctx, op) {
            let ty = struct_get.r#type(ctx);
            if !is_closure_struct(ctx, ty) || ty == native_ty {
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
                .insert(Symbol::new("type"), Attribute::Type(native_ty));
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

#[cfg(test)]
mod tests {
    use trunk_ir::Symbol;
    use trunk_ir::context::IrContext;
    use trunk_ir::dialect::core;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;
    use trunk_ir::rewrite::TypeConverter;
    use trunk_ir::types::TypeDataBuilder;

    const TAIL_TRANSFERS: &str = r#"core.module @test {
  func.func @direct_target(%value: core.i32) -> core.nil attributes {tribute.calling_convention = 2} {
    func.return
  }
  func.func @direct_caller(%value: core.i32) -> core.nil attributes {tribute.calling_convention = 2} {
    func.tail_call %value {callee = @direct_target, tribute.calling_convention = 2}
  }
  func.func @indirect_caller(%callee: core.ptr, %value: core.i32) -> core.nil attributes {tribute.calling_convention = 2} {
    func.tail_call_indirect %callee, %value {signature = core.func(core.nil, core.i32), tribute.calling_convention = 2}
  }
}"#;

    fn run_pass(ir: &str) -> String {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, ir);
        let type_converter = TypeConverter::new();
        super::lower(&mut ctx, module, type_converter).unwrap();
        print_module(&ctx, module.op())
    }

    #[test]
    fn test_func_func_to_clif() {
        let result = run_pass(
            r#"core.module @test {
  func.func @test_fn() -> core.nil {
    func.return
  }
}"#,
        );
        insta::assert_snapshot!(result);
    }

    #[test]
    fn direct_call_result_uses_the_converted_type() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @callee() -> tribute_rt.anyref
  func.func @caller() -> tribute_rt.anyref {
    %result = func.call {callee = @callee} : tribute_rt.anyref
    func.return %result
  }
}"#,
        );
        let anyref_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("tribute_rt"), Symbol::new("anyref")).build());
        let ptr_ty = core::ptr(&mut ctx).as_type_ref();
        let mut type_converter = TypeConverter::new();
        type_converter.add_conversion(move |_, ty| (ty == anyref_ty).then_some(ptr_ty));

        super::lower(&mut ctx, module, type_converter).expect("func-to-clif lowering");

        let printed = print_module(&ctx, module.op());
        let call = printed
            .lines()
            .find(|line| line.contains("clif.call"))
            .expect("lowered direct call");
        assert!(call.contains(": core.ptr"), "{printed}");
        assert!(!call.contains("tribute_rt.anyref"), "{printed}");
    }

    #[test]
    fn test_call_indirect_to_clif() {
        let result = run_pass(
            r#"core.module @test {
  func.func @test_fn() -> core.i32 {
    %0 = arith.const {value = 0} : core.i32
    %1 = arith.const {value = 42} : core.i32
    %2 = func.call_indirect %0, %1 : core.i32
    func.return %2
  }
}"#,
        );
        insta::assert_snapshot!(result);
    }

    #[test]
    fn exact_unit_call_indirect_has_no_ssa_result() {
        let result = run_pass(
            r#"core.module @test {
  func.func @test_fn(%callee: core.ptr) -> core.nil {
    func.call_indirect %callee {signature = core.func(core.nil)}
    func.return
  }
}"#,
        );

        assert!(
            result.contains("clif.call_indirect %0 {sig = core.func<() -> core.nil>}"),
            "{result}"
        );
        assert!(!result.contains("func.call_indirect"), "{result}");
    }

    #[test]
    fn test_tail_transfers_to_clif() {
        let result = run_pass(TAIL_TRANSFERS);
        insta::assert_snapshot!(result);
    }

    #[test]
    fn tail_call_indirect_signature_converts_array_to_ptr() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !evidence = core.array(core.i32)
  func.func @caller(%callee: core.ptr, %evidence: !evidence) -> core.nil attributes {tribute.calling_convention = 2} {
    func.tail_call_indirect %callee, %evidence {signature = core.func(core.nil, !evidence), tribute.calling_convention = 2}
  }
}"#,
        );
        let caller = module.ops(&ctx)[0];
        let entry = ctx.region(ctx.op(caller).regions[0]).blocks[0];
        let evidence_ty = ctx.value_ty(ctx.block_args(entry)[1]);
        let ptr_ty = core::ptr(&mut ctx).as_type_ref();
        let mut type_converter = TypeConverter::new();
        type_converter.add_conversion(move |_, ty| (ty == evidence_ty).then_some(ptr_ty));

        super::lower(&mut ctx, module, type_converter).expect("func-to-clif lowering");

        let printed = print_module(&ctx, module.op());
        assert!(
            printed.contains("clif.return_call_indirect")
                && printed.contains("sig = core.func<(core.ptr) -> core.nil>"),
            "{printed}"
        );
        assert!(
            !printed.contains("sig = core.func<(!evidence) -> core.nil>"),
            "{printed}"
        );
    }

    #[test]
    fn indirect_call_signature_converts_before_physical_validation() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !evidence = core.array(core.i32)
  func.func @physical(%callee: core.ptr, %evidence: core.ptr) -> core.i32 {
    %result = func.call_indirect %callee, %evidence {signature = core.func(core.i32, !evidence)} : core.i32
    func.return %result
  }
  func.func @mismatch(%callee: core.ptr, %value: core.i32) -> core.i32 {
    %result = func.call_indirect %callee, %value {signature = core.func(core.i32, !evidence)} : core.i32
    func.return %result
  }
}"#,
        );
        let ptr_ty = core::ptr(&mut ctx).as_type_ref();
        let mut type_converter = TypeConverter::new();
        type_converter.add_conversion(move |ctx, ty| {
            ctx.types
                .is_dialect(ty, Symbol::new("core"), Symbol::new("array"))
                .then_some(ptr_ty)
        });

        let error = super::lower(&mut ctx, module, type_converter).unwrap_err();
        assert!(error.to_string().contains("func.call_indirect"), "{error}");

        let printed = print_module(&ctx, module.op());
        assert!(
            printed.contains("clif.call_indirect")
                && printed.contains("sig = core.func<(core.ptr) -> core.i32>"),
            "{printed}"
        );
        assert!(
            printed.contains(
                "func.call_indirect %0, %1 {signature = core.func<(!evidence) -> core.i32>} : core.i32"
            ),
            "the mismatched call must remain unchanged: {printed}"
        );
    }

    #[test]
    fn tail_call_indirect_with_mismatched_signature_params_is_rejected_before_mutation() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @caller(%callee: core.ptr, %value: core.i32) -> core.nil {
    func.tail_call_indirect %callee, %value {signature = core.func(core.nil, core.i64), tribute.calling_convention = 2}
  }
}"#,
        );

        let error = super::lower(&mut ctx, module, TypeConverter::new()).unwrap_err();
        assert!(
            error.to_string().contains("func.tail_call_indirect"),
            "{error}"
        );
        let after = print_module(&ctx, module.op());
        assert!(after.contains("func.tail_call_indirect"), "{after}");
        assert!(!after.contains("clif.return_call_indirect"), "{after}");
    }

    #[test]
    fn tail_transfers_emit_native_object() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, TAIL_TRANSFERS);
        super::lower(&mut ctx, module, TypeConverter::new()).unwrap();

        let object = crate::emit_module_to_native(&ctx, module, &[]).unwrap();
        assert!(!object.is_empty());
    }

    #[test]
    fn tail_call_indirect_without_exact_signature_is_rejected() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @caller(%callee: core.ptr, %value: core.i32) -> core.nil {
    func.tail_call_indirect %callee, %value {tribute.calling_convention = 2}
  }
}"#,
        );

        let error = super::lower(&mut ctx, module, TypeConverter::new()).unwrap_err();
        assert!(
            error.to_string().contains("func.tail_call_indirect"),
            "{error}"
        );
    }

    #[test]
    fn tail_call_indirect_with_nonempty_signature_is_rejected() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @caller(%callee: core.ptr, %value: core.i32) -> core.nil {
    func.tail_call_indirect %callee, %value {signature = core.func(core.i32, core.i32), tribute.calling_convention = 2}
  }
}"#,
        );

        let error = super::lower(&mut ctx, module, TypeConverter::new()).unwrap_err();
        assert!(
            error.to_string().contains("func.tail_call_indirect"),
            "{error}"
        );
    }

    #[test]
    fn tail_call_indirect_without_cps_metadata_is_rejected_before_mutation() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @caller(%callee: core.ptr, %value: core.i32) -> core.nil {
    func.tail_call_indirect %callee, %value {signature = core.func(core.nil, core.i32)}
  }
}"#,
        );
        let error = super::lower(&mut ctx, module, TypeConverter::new()).unwrap_err();
        assert!(
            error.to_string().contains("func.tail_call_indirect"),
            "{error}"
        );
        let after = print_module(&ctx, module.op());
        assert!(after.contains("func.tail_call_indirect"), "{after}");
        assert!(!after.contains("clif.return_call_indirect"), "{after}");
    }

    #[test]
    fn tail_call_indirect_with_non_cps_metadata_is_rejected_before_mutation() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @caller(%callee: core.ptr, %value: core.i32) -> core.nil {
    func.tail_call_indirect %callee, %value {signature = core.func(core.nil, core.i32), tribute.calling_convention = 0}
  }
}"#,
        );
        let error = super::lower(&mut ctx, module, TypeConverter::new()).unwrap_err();
        assert!(
            error.to_string().contains("func.tail_call_indirect"),
            "{error}"
        );
        let after = print_module(&ctx, module.op());
        assert!(after.contains("func.tail_call_indirect"), "{after}");
        assert!(!after.contains("clif.return_call_indirect"), "{after}");
    }

    #[test]
    fn test_closure_struct_adaptation() {
        let result = run_pass(
            r#"core.module @test {
  func.func @test_fn() -> core.i32 {
    %0 = func.constant {func_ref = @lifted_fn} : core.i32
    %1 = arith.const {value = 0} : core.ptr
    %2 = adt.struct_new %0, %1 {type = adt.struct(core.i32, core.ptr) {name = @_closure, fields = [@table_idx, @env]}} : adt.struct(core.i32, core.ptr) {name = @_closure, fields = [@table_idx, @env]}
    %3 = adt.struct_get %2 {field = 0, type = adt.struct(core.i32, core.ptr) {name = @_closure, fields = [@table_idx, @env]}} : core.i32
    %4 = adt.struct_get %2 {field = 1, type = adt.struct(core.i32, core.ptr) {name = @_closure, fields = [@table_idx, @env]}} : core.ptr
    %5 = func.call_indirect %3, %4 : core.i32
    func.return %5
  }
}"#,
        );
        insta::assert_snapshot!(result);
    }

    #[test]
    fn test_closure_struct_anyref_adaptation() {
        let result = run_pass(
            r#"core.module @test {
  func.func @test_fn() -> core.i32 {
    %0 = func.constant {func_ref = @lifted_fn} : core.i32
    %1 = arith.const {value = 0} : wasm.anyref
    %2 = adt.struct_new %0, %1 {type = adt.struct(core.i32, wasm.anyref) {name = @_closure, fields = [@table_idx, @env]}} : adt.struct(core.i32, wasm.anyref) {name = @_closure, fields = [@table_idx, @env]}
    %3 = adt.struct_get %2 {field = 0, type = adt.struct(core.i32, wasm.anyref) {name = @_closure, fields = [@table_idx, @env]}} : core.i32
    %4 = adt.struct_get %2 {field = 1, type = adt.struct(core.i32, wasm.anyref) {name = @_closure, fields = [@table_idx, @env]}} : wasm.anyref
    %5 = func.call_indirect %3, %4 : core.i32
    func.return %5
  }
}"#,
        );
        insta::assert_snapshot!(result);
    }
}
