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
use trunk_ir::ir::Symbol;

/// Lower func dialect to clif dialect.
pub fn lower(ctx: &mut IrContext, module: ArenaModule, type_converter: ArenaTypeConverter) {
    use trunk_ir::arena::rewrite::ArenaConversionTarget;

    // Phase 1: Adapt closure structs for native backend
    adapt_closure_structs(ctx, module);

    // Phase 2: Lower func dialect to clif dialect
    let mut target = ArenaConversionTarget::new();
    target.add_legal_dialect("clif");
    target.add_illegal_dialect("func");

    let applicator = ArenaPatternApplicator::new(type_converter)
        .with_target(target)
        .add_pattern(FuncFuncPattern)
        .add_pattern(FuncCallPattern)
        .add_pattern(FuncCallIndirectPattern)
        .add_pattern(FuncReturnPattern)
        .add_pattern(FuncTailCallPattern)
        .add_pattern(FuncUnreachablePattern)
        .add_pattern(FuncConstantPattern);
    applicator.apply_partial(ctx, module);
}

fn adapt_closure_structs(ctx: &mut IrContext, module: ArenaModule) {
    let applicator = ArenaPatternApplicator::new(ArenaTypeConverter::new())
        .add_pattern(ClosureStructAdaptPattern);
    applicator.apply_partial(ctx, module);
}

const CLOSURE_STRUCT_NAME_STR: &str = "_closure";

fn is_closure_struct(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.types.get(ty);
    data.attrs
        .get(&Symbol::new("name"))
        .and_then(|a| match a {
            ArenaAttribute::Symbol(s) => Some(*s),
            _ => None,
        })
        .is_some_and(|name| name == Symbol::new(CLOSURE_STRUCT_NAME_STR))
}

fn native_closure_struct_type(ctx: &mut IrContext) -> TypeRef {
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

/// Pattern: `func.func` -> `clif.func`
struct FuncFuncPattern;

impl ArenaRewritePattern for FuncFuncPattern {
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
                // - Layout A: params = [ret, arg1, arg2, ...] (translate_signature format)
                // - Layout B: params = [arg1, arg2, ...], attrs.result = ret
                // We read both and output in Layout A for translate_signature.
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

/// Pattern: `func.call` -> `clif.call`
struct FuncCallPattern;

impl ArenaRewritePattern for FuncCallPattern {
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

/// Pattern: `func.call_indirect` -> `clif.call_indirect`
struct FuncCallIndirectPattern;

impl ArenaRewritePattern for FuncCallIndirectPattern {
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
        if operands.is_empty() {
            return false;
        }

        // Collect arg types (skip operand 0 = callee).
        // Operand types are already converted by the applicator's cast insertion.
        let param_types: Vec<TypeRef> = operands[1..].iter().map(|&v| ctx.value_ty(v)).collect();

        // Result type with conversion applied
        let result_ty = rewriter.result_type(ctx, op, 0);

        // Build sig type matching translate_signature layout:
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

/// Pattern: `func.return` -> `clif.return`
struct FuncReturnPattern;

impl ArenaRewritePattern for FuncReturnPattern {
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

/// Pattern: `func.tail_call` -> `clif.return_call`
struct FuncTailCallPattern;

impl ArenaRewritePattern for FuncTailCallPattern {
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

/// Pattern: `func.unreachable` -> `clif.trap`
struct FuncUnreachablePattern;

impl ArenaRewritePattern for FuncUnreachablePattern {
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

/// Pattern: `func.constant` -> `clif.symbol_addr`
struct FuncConstantPattern;

impl ArenaRewritePattern for FuncConstantPattern {
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

/// Pattern: Adapt `_closure` struct ops for native backend
struct ClosureStructAdaptPattern;

impl ArenaRewritePattern for ClosureStructAdaptPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut ArenaPatternRewriter<'_>,
    ) -> bool {
        use trunk_ir::arena::dialect::adt as arena_adt;

        let native_ty = native_closure_struct_type(ctx);

        // Handle adt.struct_new on _closure
        if let Ok(struct_new) = arena_adt::StructNew::from_op(ctx, op) {
            let ty = struct_new.r#type(ctx);
            if !is_closure_struct(ctx, ty) {
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
            if !is_closure_struct(ctx, ty) {
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

#[cfg(test)]
mod tests {
    use trunk_ir::arena::context::IrContext;
    use trunk_ir::arena::parser::parse_test_module;
    use trunk_ir::arena::printer::print_module;
    use trunk_ir::arena::rewrite::ArenaTypeConverter;

    fn run_pass(ir: &str) -> String {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, ir);
        let type_converter = ArenaTypeConverter::new();
        super::lower(&mut ctx, module, type_converter);
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
