//! Lower func dialect operations to clif dialect.
//!
//! This pass converts function-level operations to Cranelift equivalents:
//! - `func.func` -> `clif.func`
//! - `func.call` -> `clif.call`
//! - `func.call_indirect` -> `clif.call_indirect`
//! - `func.return` -> `clif.return`
//! - `func.unreachable` -> `clif.trap`
//! - `func.constant` -> `clif.symbol_addr`

use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::clif;
use trunk_ir::dialect::core;
use trunk_ir::dialect::func;
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{OpRef, TypeRef};
use trunk_ir::rewrite::{
    ConversionError, ConversionTarget, Module, PatternApplicator, PatternRewriter, RewritePattern,
    TypeConverter,
};
use trunk_ir::types::Attribute;

/// Lower func dialect to clif dialect.
pub fn lower(
    ctx: &mut IrContext,
    module: Module,
    type_converter: TypeConverter,
) -> Result<(), ConversionError> {
    // Phase 1: Adapt closure structs for native backend
    adapt_closure_structs(ctx, module);

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
    Ok(())
}

fn func_to_clif_target() -> ConversionTarget {
    ConversionTarget::new()
        .legal_dialect("clif")
        .illegal_dialect("func")
}

fn adapt_closure_structs(ctx: &mut IrContext, module: Module) {
    let applicator =
        PatternApplicator::new(TypeConverter::new()).add_pattern(ClosureStructAdaptPattern);
    applicator.apply_partial(ctx, module);
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
        if let Some(func_ty) = func_type_attr {
            let type_data = ctx.types.get(func_ty);
            if type_data.dialect == Symbol::new("core") && type_data.name == Symbol::new("func") {
                // The arena core.func type may use two layouts:
                // - Layout A: params = [ret, arg1, arg2, ...] (translate_signature format)
                // - Layout B: params = [arg1, arg2, ...], attrs.result = ret
                // We read both and output in Layout A for translate_signature.
                let (arg_params, ret_ty) = if let Some(r) = type_data.attrs.get_type("result") {
                    // Layout B: return type in attrs
                    (&type_data.params[..], Some(r))
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
                let ret_ty = new_ret.unwrap_or_else(|| core::nil(ctx).as_type_ref());
                let new_func_ty = core::func(ctx, ret_ty, new_params.iter().copied()).as_type_ref();
                new_attrs.insert(Symbol::new("type"), Attribute::Type(new_func_ty));
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
        let location = ctx.op(op).location;
        let args = ctx.op_operands(op).to_vec();
        let result = rewriter
            .result_type(ctx, op, 0)
            .unwrap_or_else(|| core::nil(ctx).as_type_ref());
        let new_op = clif::call(ctx, location, args, result, callee);
        let mut attrs = ctx.op(op).attributes.clone();
        attrs.remove(Symbol::new("callee"));
        ctx.op_mut(new_op.op_ref()).attributes.extend(attrs);
        rewriter.replace_op(new_op.op_ref());
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
        if func::CallIndirect::from_op(ctx, op).is_err() {
            return false;
        }

        let operands = ctx.op_operands(op).to_vec();
        if operands.is_empty() {
            return false;
        }

        // Closure lowering records the exact callable ABI before replacing a
        // typed closure with an untyped function/table reference. Prefer that
        // contract over representation operands, whose pointer types cannot
        // recover source-data positions such as an erased resume input.
        let sig_ty = ctx
            .op(op)
            .attributes
            .get_type(func::INDIRECT_CALL_SIGNATURE_ATTR)
            .unwrap_or_else(|| {
                let param_types: Vec<TypeRef> = operands[1..]
                    .iter()
                    .map(|&value| ctx.value_ty(value))
                    .collect();
                let result_ty = rewriter.result_type(ctx, op, 0);
                let ret_ty = result_ty.unwrap_or_else(|| core::nil(ctx).as_type_ref());
                core::func(ctx, ret_ty, param_types.iter().copied()).as_type_ref()
            });
        let Some(sig_ty) = lower_indirect_signature(ctx, rewriter.type_converter(), sig_ty) else {
            return false;
        };

        let new_op = crate::passes::cf_to_clif::rebuild_op_as(
            ctx,
            op,
            Symbol::new("clif"),
            Symbol::new("call_indirect"),
        );
        ctx.op_mut(new_op)
            .attributes
            .insert(Symbol::new("sig"), Attribute::Type(sig_ty));
        ctx.op_mut(new_op)
            .attributes
            .remove(Symbol::new(func::INDIRECT_CALL_SIGNATURE_ATTR));
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
        let operands = ctx.op_operands(op);
        if operands.len() == 1 {
            let value_ty = ctx.value_ty(operands[0]);
            let data = ctx.types.get(value_ty);
            if data.dialect == Symbol::new("core") && data.name == Symbol::new("nil") {
                let new_op = clif::r#return(ctx, ctx.op(op).location, []);
                rewriter.replace_op(new_op.op_ref());
                return true;
            }
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

/// Pattern: `func.tail_call_indirect` -> `clif.return_call_indirect`
struct FuncTailCallIndirectPattern;

impl RewritePattern for FuncTailCallIndirectPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if func::TailCallIndirect::from_op(ctx, op).is_err() {
            return false;
        }

        let operands = ctx.op_operands(op).to_vec();
        if operands.is_empty() || !ctx.op_result_types(op).is_empty() {
            return false;
        }

        let sig_ty = ctx
            .op(op)
            .attributes
            .get_type(func::INDIRECT_CALL_SIGNATURE_ATTR)
            .unwrap_or_else(|| {
                let param_types: Vec<TypeRef> = operands[1..]
                    .iter()
                    .map(|&value| ctx.value_ty(value))
                    .collect();
                let nil_ty = core::nil(ctx).as_type_ref();
                core::func(ctx, nil_ty, param_types.iter().copied()).as_type_ref()
            });
        let Some(sig_ty) = lower_indirect_signature(ctx, rewriter.type_converter(), sig_ty) else {
            return false;
        };

        let new_op = crate::passes::cf_to_clif::rebuild_op_as(
            ctx,
            op,
            Symbol::new("clif"),
            Symbol::new("return_call_indirect"),
        );
        ctx.op_mut(new_op)
            .attributes
            .insert(Symbol::new("sig"), Attribute::Type(sig_ty));
        ctx.op_mut(new_op)
            .attributes
            .remove(Symbol::new(func::INDIRECT_CALL_SIGNATURE_ATTR));
        rewriter.replace_op(new_op);
        true
    }
}

/// Project the exact indirect ABI through the target type converter.  The
/// provenance records logical closure values, while the native call boundary
/// receives their lowered pointer representation; source scalar parameters
/// such as a resumed `core.i32` stay scalar.
fn lower_indirect_signature(
    ctx: &mut IrContext,
    converter: &TypeConverter,
    signature: TypeRef,
) -> Option<TypeRef> {
    let callable = core::Func::from_type_ref(ctx, signature)?;
    let result_ty = callable.r#return(ctx);
    let params = callable.params(ctx).to_vec();
    let result = converter.convert_type_or_identity(ctx, result_ty);
    let params: Vec<_> = params
        .iter()
        .map(|&param| converter.convert_type_or_identity(ctx, param))
        .collect();
    Some(core::func(ctx, result, params).as_type_ref())
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
    use trunk_ir::context::IrContext;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;
    use trunk_ir::rewrite::TypeConverter;

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
    fn nil_return_has_no_clif_value_operand() {
        let result = run_pass(
            r#"core.module @test {
  func.func @test_fn() -> core.nil {
    %0 = arith.const {value = unit} : core.nil
    func.return %0
  }
}"#,
        );
        assert!(result.contains("clif.return\n"), "{result}");
        assert!(!result.contains("clif.return %"), "{result}");
    }

    #[test]
    fn bodyless_func_declaration_lowers_without_fabricating_a_body() {
        let result = run_pass(
            r#"core.module @test {
  func.func @imported(%value: core.i32) -> core.i32 attributes {abi = "C"}
  func.func @defined() -> core.nil {
    func.return
  }
}"#,
        );

        assert!(result.contains(
            "clif.func {abi = \"C\", sym_name = @imported, type = core.func(core.i32, core.i32)}"
        ));
        assert!(result.contains("clif.func {sym_name = @defined, type = core.func(core.nil)} {"));
        assert!(!result.contains("sym_name = @imported, type = core.func(core.i32, core.i32)} {"));
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
    fn preserves_per_transfer_calling_convention_for_tail_and_ordinary_indirect_calls() {
        let result = run_pass(
            r#"core.module @test {
  func.func @done() -> core.nil attributes {clif.calling_convention = @tail} {
    func.return
  }
  func.func @tail_direct() -> core.nil attributes {clif.calling_convention = @tail} {
    func.tail_call {callee = @done, clif.calling_convention = @tail}
  }
  func.func @tail_indirect(%callee: core.i32) -> core.nil attributes {clif.calling_convention = @tail} {
    func.tail_call_indirect %callee {clif.calling_convention = @tail}
  }
  func.func @ordinary_indirect(%callee: core.i32) -> core.i32 attributes {clif.calling_convention = @platform} {
    %result = func.call_indirect %callee {clif.calling_convention = @platform} : core.i32
    func.return %result
  }
}"#,
        );

        assert!(
            result.contains("clif.return_call {callee = @done, clif.calling_convention = @tail}")
        );
        assert!(result.contains("clif.return_call_indirect %0 {clif.calling_convention = @tail"));
        assert!(result.contains("clif.call_indirect %0 {clif.calling_convention = @platform"));
        assert!(
            !result.contains("clif.return_call_indirect %0 {clif.calling_convention = @platform")
        );
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
