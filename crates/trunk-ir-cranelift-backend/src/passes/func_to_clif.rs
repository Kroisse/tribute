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

use std::collections::HashMap;

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
use trunk_ir::symbol_table::SymbolTable;
use trunk_ir::types::Attribute;
use trunk_ir::{Symbol, SymbolPath};

/// Lower func dialect to clif dialect.
pub fn lower(
    ctx: &mut IrContext,
    module: Module,
    type_converter: TypeConverter,
) -> Result<(), ConversionError> {
    let functions = function_signatures(ctx, module);

    let applicator = PatternApplicator::new(type_converter)
        .with_auto_type_conversion(true)
        .add_pattern(FuncFuncPattern)
        .add_pattern(FuncCallPattern)
        .add_pattern(FuncCallIndirectPattern)
        .add_pattern(FuncReturnPattern)
        .add_pattern(FuncTailCallPattern)
        .add_pattern(FuncTailCallIndirectPattern)
        .add_pattern(FuncUnreachablePattern)
        .add_pattern(FuncConstantPattern { functions })
        .with_target(func_to_clif_target());
    applicator.apply_partial_conversion(ctx, module, "func-to-clif")?;
    Ok(())
}

fn convert_attribute_to_clif(
    ctx: &mut IrContext,
    attribute: &Attribute,
    converter: &TypeConverter,
) -> Option<Attribute> {
    attribute
        .try_map_types(&mut |ty| convert_type_to_clif(ctx, ty, converter).ok_or(()))
        .ok()
}

fn convert_nested_callable_type(
    ctx: &mut IrContext,
    ty: TypeRef,
    converter: &TypeConverter,
) -> Option<TypeRef> {
    if func::FuncSig::from_type_ref(ctx, ty).is_some() {
        return convert_type_to_clif(ctx, ty, converter);
    }
    let data = ctx.get_type(ty).clone();
    let params = data
        .params
        .iter()
        .map(|parameter| convert_nested_callable_type(ctx, *parameter, converter))
        .collect::<Option<Vec<_>>>()?;
    let attrs = data
        .attrs
        .iter()
        .map(|(key, value)| {
            Some((
                key.clone(),
                convert_nested_callable_attribute(ctx, value, converter)?,
            ))
        })
        .collect::<Option<_>>()?;
    if params.as_slice() == data.params.as_slice() && attrs == data.attrs {
        return Some(ty);
    }
    let mut converted_data = data;
    converted_data.params = params.into();
    converted_data.attrs = attrs;
    Some(ctx.intern_type(converted_data))
}

fn convert_nested_callable_attribute(
    ctx: &mut IrContext,
    attribute: &Attribute,
    converter: &TypeConverter,
) -> Option<Attribute> {
    attribute
        .try_map_types(&mut |ty| convert_nested_callable_type(ctx, ty, converter).ok_or(()))
        .ok()
}

fn convert_type_to_clif(
    ctx: &mut IrContext,
    ty: TypeRef,
    converter: &TypeConverter,
) -> Option<TypeRef> {
    if let Some(shared) = func::FuncSig::from_type_ref(ctx, ty) {
        let inputs = shared.inputs(ctx).to_vec();
        let results = shared.results(ctx).to_vec();
        let type_attrs = shared
            .non_reserved_attrs(ctx)
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect::<Vec<_>>();
        let attrs = type_attrs
            .into_iter()
            .map(|(key, value)| Some((key, convert_attribute_to_clif(ctx, &value, converter)?)))
            .collect::<Option<_>>()?;
        let inputs = inputs
            .into_iter()
            .map(|ty| convert_type_to_clif(ctx, ty, converter))
            .collect::<Option<Vec<_>>>()?;
        let results = results
            .into_iter()
            .map(|ty| convert_type_to_clif(ctx, ty, converter))
            .collect::<Option<Vec<_>>>()?;
        return Some(clif::func_sig_with_attrs(ctx, inputs, results, attrs).as_type_ref());
    }
    let converted = converter.convert_type_or_identity(ctx, ty);
    if converted != ty {
        return convert_type_to_clif(ctx, converted, converter);
    }
    let data = ctx.get_type(ty).clone();
    let params = data
        .params
        .iter()
        .map(|parameter| convert_nested_callable_type(ctx, *parameter, converter))
        .collect::<Option<Vec<_>>>()?;
    let attrs = data
        .attrs
        .iter()
        .map(|(key, value)| {
            Some((
                key.clone(),
                convert_nested_callable_attribute(ctx, value, converter)?,
            ))
        })
        .collect::<Option<_>>()?;
    if params.as_slice() == data.params.as_slice() && attrs == data.attrs {
        return Some(ty);
    }
    let mut converted_data = data;
    converted_data.params = params.into();
    converted_data.attrs = attrs;
    Some(ctx.intern_type(converted_data))
}

fn convert_to_clif_func_type(
    ctx: &mut IrContext,
    signature: TypeRef,
    converter: &TypeConverter,
) -> Option<TypeRef> {
    func::FuncSig::from_type_ref(ctx, signature)?;
    convert_type_to_clif(ctx, signature, converter)
}

fn func_to_clif_target() -> ConversionTarget {
    ConversionTarget::new()
        .legal_dialect("clif")
        .illegal_dialect("func")
}

fn intern_ptr_type(ctx: &mut IrContext) -> TypeRef {
    core::ptr(ctx).as_type_ref()
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
        let Some(func_ty) = func_type_attr else {
            return false;
        };
        let Some(new_func_ty) = convert_to_clif_func_type(ctx, func_ty, tc) else {
            return false;
        };
        new_attrs.insert(Symbol::new("type"), Attribute::Type(new_func_ty));

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

        let callee = call_op.callee(ctx).clone();
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
            .insert(Symbol::new("callee"), Attribute::SymbolRef(callee));
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
        let Some(signature) = call.exact_signature(ctx) else {
            return false;
        };
        let Some(sig_ty) = convert_to_clif_func_type(ctx, signature, rewriter.type_converter())
        else {
            return false;
        };
        let Some(callable) = clif::FuncSig::from_type_ref(ctx, sig_ty) else {
            return false;
        };
        let runtime_result_types = callable
            .results(ctx)
            .iter()
            .copied()
            .filter(|ty| !crate::function::is_nil_type(ctx, *ty))
            .collect::<Vec<_>>();
        if callable.inputs(ctx).len() != CallLike::call_args(&call, ctx).len()
            || callable
                .inputs(ctx)
                .iter()
                .zip(CallLike::call_args(&call, ctx))
                .any(|(&param, &arg)| param != ctx.value_ty(arg))
            || (result_types.as_slice() != callable.results(ctx)
                && result_types != runtime_result_types)
        {
            return false;
        }

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

        let callee = tail_call.callee(ctx).clone();
        let new_op = crate::passes::cf_to_clif::rebuild_op_as(
            ctx,
            op,
            Symbol::new("clif"),
            Symbol::new("return_call"),
        );
        ctx.op_mut(new_op)
            .attributes
            .insert(Symbol::new("callee"), Attribute::SymbolRef(callee));
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
        let Some(signature) = tail.exact_signature(ctx) else {
            return false;
        };
        let Some(signature) = convert_to_clif_func_type(ctx, signature, rewriter.type_converter())
        else {
            return false;
        };
        let Some(callable) = clif::FuncSig::from_type_ref(ctx, signature) else {
            return false;
        };
        if callable.call_conv(ctx) != Some(func::CallConv::Tail)
            || !callable.results(ctx).is_empty()
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
        let new_op = clif::Trap::operands().code("unreachable").build(ctx, loc);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// Each uniquely defined `func.func` by root-qualified name, with its exact
/// signature captured before lowering converts it.
fn function_signatures(ctx: &IrContext, module: Module) -> HashMap<SymbolPath, TypeRef> {
    let table = SymbolTable::collect(ctx, module);
    table
        .iter()
        .filter_map(|(name, ops)| match ops {
            &[op] => Some((name.clone(), func::Func::from_op(ctx, op).ok()?.r#type(ctx))),
            _ => None,
        })
        .collect()
}

/// Pattern: `func.constant` -> `clif.symbol_addr`
///
/// The resulting pointer no longer names a callable contract, so a typed
/// reference must carry exactly its target's signature, calling convention
/// included, before it is erased.
struct FuncConstantPattern {
    functions: HashMap<SymbolPath, TypeRef>,
}

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

        let func_ref = const_op.func_ref(ctx).clone();
        if let &[result] = ctx.op_result_types(op)
            && func::FuncSig::matches(ctx, result)
            && self.functions.get(&func_ref).copied() != Some(result)
        {
            return false;
        }
        let loc = ctx.op(op).location;
        let ptr_ty = intern_ptr_type(ctx);
        let new_op = clif::SymbolAddr::operands()
            .sym(func_ref)
            .results(ptr_ty)
            .build(ctx, loc);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

#[cfg(test)]
mod tests {
    use trunk_ir::context::IrContext;
    use trunk_ir::dialect::{clif, core, func};
    use trunk_ir::ops::DialectType;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;
    use trunk_ir::rewrite::TypeConverter;
    use trunk_ir::types::TypeDataBuilder;
    use trunk_ir::{Attribute, AttributeMap, Symbol};

    const TAIL_TRANSFERS: &str = r#"core.module @test {
  func.func @direct_target(%value: core.i32) attributes {type = func.func_sig<(core.i32) -> (), {call_conv = "tail"}>} {
    func.return
  }
  func.func @direct_caller(%value: core.i32) attributes {type = func.func_sig<(core.i32) -> (), {call_conv = "tail"}>} {
    func.tail_call %value {callee = @direct_target}
  }
  func.func @indirect_caller(%callee: core.ptr, %value: core.i32) attributes {type = func.func_sig<(core.ptr, core.i32) -> (), {call_conv = "tail"}>} {
    func.tail_call_indirect %callee, %value {signature = func.func_sig<(core.i32) -> (), {call_conv = "tail"}>}
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
    fn review_invalid_function_signature_does_not_move_body() {
        use trunk_ir::rewrite::PatternApplicator;
        for malformed in [false, true] {
            let mut ctx = IrContext::new();
            let module = parse_test_module(
                &mut ctx,
                "core.module @m { func.func @f(%x: core.i32) -> core.i32 { func.return %x } }",
            );
            let op = module.ops(&ctx)[0];
            if malformed {
                let ty = ctx.intern_type(TypeDataBuilder::new("func", "func_sig").build());
                ctx.op_mut(op)
                    .attributes
                    .insert(Symbol::new("type"), trunk_ir::Attribute::Type(ty));
            } else {
                ctx.op_mut(op).attributes.remove("type");
            }
            let before = print_module(&ctx, module.op());
            let regions = ctx.op_regions(op).collect::<trunk_ir::RegionList>();
            PatternApplicator::new(TypeConverter::new())
                .add_pattern(super::FuncFuncPattern)
                .apply_partial(&mut ctx, module);
            assert_eq!(print_module(&ctx, module.op()), before);
            assert_eq!(module.ops(&ctx)[0], op);
            assert!(ctx.op_regions(op).eq(regions));
        }
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
  func.func @callee() -> test.anyref
  func.func @caller() -> test.anyref {
    %result = func.call {callee = @callee} : test.anyref
    func.return %result
  }
}"#,
        );
        let anyref_ty = ctx.intern_type(TypeDataBuilder::new("test", "anyref").build());
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
        assert!(!call.contains("test.anyref"), "{printed}");
    }

    #[test]
    fn nested_callable_metadata_uses_the_native_owned_signature() {
        let mut ctx = IrContext::new();
        let i32_ty = ctx.intern_type(TypeDataBuilder::new("core", "i32").build());
        let i64_ty = ctx.intern_type(TypeDataBuilder::new("core", "i64").build());
        let mut type_converter = TypeConverter::new();
        type_converter.add_conversion(move |_, ty| (ty == i32_ty).then_some(i64_ty));

        let mut inner_attrs = AttributeMap::new();
        inner_attrs.insert(
            Symbol::new("tag"),
            Attribute::SymbolRef(trunk_ir::SymbolPath::from("preserved")),
        );
        inner_attrs.insert(
            Symbol::new("nested"),
            Attribute::List(vec![Attribute::Type(i32_ty)]),
        );
        let nested =
            func::func_sig_with_attrs(&mut ctx, [i32_ty], [i32_ty], inner_attrs).as_type_ref();
        let mut attrs = AttributeMap::new();
        attrs.insert(
            Symbol::new("evidence"),
            Attribute::List(vec![Attribute::Type(nested)]),
        );
        let source = func::func_sig_with_attrs(&mut ctx, [i32_ty], [i32_ty], attrs).as_type_ref();
        let converted = super::convert_to_clif_func_type(&mut ctx, source, &type_converter)
            .expect("target callable contract");
        let signature = clif::FuncSig::from_type_ref(&ctx, converted).expect("clif.func_sig");
        assert!(signature.inputs(&ctx) == [i64_ty] && signature.results(&ctx) == [i64_ty]);
        let Some(Attribute::List(evidence)) = signature
            .non_reserved_attrs(&ctx)
            .find_map(|(key, value)| (*key == Symbol::new("evidence")).then_some(value))
        else {
            panic!("missing nested target contract");
        };
        let [Attribute::Type(nested)] = evidence.as_slice() else {
            panic!("nested target contract must be a one-element type list");
        };
        let nested = clif::FuncSig::from_type_ref(&ctx, *nested).expect("nested clif.func_sig");
        assert_eq!(nested.inputs(&ctx), [i64_ty]);
        assert_eq!(nested.results(&ctx), [i64_ty]);
        assert_eq!(nested.non_reserved_attrs(&ctx).count(), 2);
    }

    #[test]
    fn parameter_attributes_are_preserved_and_their_types_converted() {
        let mut ctx = IrContext::new();
        let i32_ty = ctx.intern_type(TypeDataBuilder::new("core", "i32").build());
        let i64_ty = ctx.intern_type(TypeDataBuilder::new("core", "i64").build());
        let mut type_converter = TypeConverter::new();
        type_converter.add_conversion(move |_, ty| (ty == i32_ty).then_some(i64_ty));
        let dict = |ty| -> AttributeMap {
            [(Symbol::new("witness"), Attribute::Type(ty))]
                .into_iter()
                .collect()
        };
        let source = func::func_sig_with_param_attrs(
            &mut ctx,
            [(i32_ty, dict(i32_ty)), (i32_ty, AttributeMap::new())],
            [(i32_ty, AttributeMap::new())],
            AttributeMap::new(),
        )
        .as_type_ref();
        let converted = super::convert_to_clif_func_type(&mut ctx, source, &type_converter)
            .expect("target callable contract");
        let data = ctx.get_type(converted);
        assert_eq!(data.param_attrs(0), &dict(i64_ty));
        assert!(data.param_attrs(1).is_empty() && data.param_attrs(2).is_empty());
    }

    #[test]
    fn test_call_indirect_to_clif() {
        let result = run_pass(
            r#"core.module @test {
  func.func @test_fn() -> core.i32 {
    %0 = arith.const {value = 0} : core.i32
    %1 = arith.const {value = 42} : core.i32
    %2 = func.call_indirect %0, %1 {signature = func.func_sig<(core.i32) -> core.i32>} : core.i32
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
    func.call_indirect %callee {signature = func.func_sig<() -> core.nil>}
    func.return
  }
}"#,
        );

        assert!(
            result.contains("clif.call_indirect %0 {sig = clif.func_sig<() -> core.nil>}"),
            "{result}"
        );
        assert!(!result.contains("func.call_indirect"), "{result}");
    }

    #[test]
    fn indirect_call_preserves_live_logical_nil_result() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @caller(%callee: core.ptr) -> core.nil {
    %unit = func.call_indirect %callee {signature = func.func_sig<() -> core.nil>} : core.nil
    func.return %unit
  }
}"#,
        );

        super::lower(&mut ctx, module, TypeConverter::new()).expect("func-to-clif lowering");
        crate::validate_clif_ir(&ctx, module).expect("logical nil call result is valid native IR");
        let printed = print_module(&ctx, module.op());
        assert!(
            printed
                .contains("clif.call_indirect %0 {sig = clif.func_sig<() -> core.nil>} : core.nil"),
            "{printed}"
        );
        assert!(printed.contains("clif.return %1"), "{printed}");
        assert!(
            !crate::emit_module_to_native(&ctx, module)
                .expect("native emitter projects nil to zero-width")
                .is_empty()
        );
    }

    #[test]
    fn indirect_call_without_exact_signature_is_rejected_before_mutation() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @caller(%callee: core.ptr) -> core.nil {
    %unit = func.call_indirect %callee : core.nil
    func.return %unit
  }
}"#,
        );

        let error = super::lower(&mut ctx, module, TypeConverter::new()).unwrap_err();
        assert!(error.to_string().contains("func.call_indirect"), "{error}");
        let printed = print_module(&ctx, module.op());
        assert!(printed.contains("func.call_indirect"), "{printed}");
        assert!(!printed.contains("clif.call_indirect"), "{printed}");
    }

    #[test]
    fn indirect_call_does_not_equate_nil_contract_with_no_result() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @caller(%callee: core.ptr) -> core.nil {
    %unit = func.call_indirect %callee {signature = func.func_sig<() -> ()>} : core.nil
    func.return %unit
  }
}"#,
        );

        let error = super::lower(&mut ctx, module, TypeConverter::new()).unwrap_err();
        assert!(error.to_string().contains("func.call_indirect"), "{error}");
        let printed = print_module(&ctx, module.op());
        assert!(printed.contains("func.call_indirect"), "{printed}");
        assert!(!printed.contains("clif.call_indirect"), "{printed}");
    }

    #[test]
    fn callable_value_is_erased_while_its_exact_contract_stays_target_owned() {
        let result = run_pass(
            r#"core.module @test {
  func.func @target() -> core.i32 {
    %value = arith.const {value = 7} : core.i32
    func.return %value
  }
  func.func @caller() -> core.i32 {
    %callee = func.constant {func_ref = @target} : func.func_sig<() -> core.i32>
    %value = func.call_indirect %callee {signature = func.func_sig<() -> core.i32>} : core.i32
    func.return %value
  }
}"#,
        );
        assert!(
            result.contains("clif.symbol_addr {sym = @target} : core.ptr"),
            "{result}"
        );
        assert!(
            result.contains("clif.call_indirect %0 {sig = !t0} : core.i32"),
            "{result}"
        );
        assert!(
            result.contains("!t0 = clif.func_sig<() -> core.i32>"),
            "{result}"
        );
        assert!(!result.contains("func.func_sig"), "{result}");
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
  !evidence = core.array<core.i32>
  func.func @caller(%callee: core.ptr, %evidence: !evidence) attributes {type = func.func_sig<(core.ptr, !evidence) -> (), {call_conv = "tail"}>} {
    func.tail_call_indirect %callee, %evidence {signature = func.func_sig<(!evidence) -> (), {call_conv = "tail"}>}
  }
}"#,
        );
        let caller = module.ops(&ctx)[0];
        let entry = ctx.region(ctx.op_region(caller, 0).unwrap()).blocks[0];
        let evidence_ty = ctx.value_ty(ctx.block_args(entry)[1]);
        let ptr_ty = core::ptr(&mut ctx).as_type_ref();
        let mut type_converter = TypeConverter::new();
        type_converter.add_conversion(move |_, ty| (ty == evidence_ty).then_some(ptr_ty));

        super::lower(&mut ctx, module, type_converter).expect("func-to-clif lowering");

        let printed = print_module(&ctx, module.op());
        assert!(
            printed.contains("clif.return_call_indirect")
                && printed
                    .contains("sig = clif.func_sig<(core.ptr) -> (), {call_conv = \"tail\"}>"),
            "{printed}"
        );
        assert!(
            !printed.contains("sig = clif.func_sig<(!evidence) -> ()>"),
            "{printed}"
        );
    }

    #[test]
    fn indirect_call_signature_converts_before_physical_validation() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !evidence = core.array<core.i32>
  func.func @physical(%callee: core.ptr, %evidence: core.ptr) -> core.i32 {
    %result = func.call_indirect %callee, %evidence {signature = func.func_sig<(!evidence) -> core.i32>} : core.i32
    func.return %result
  }
  func.func @mismatch(%callee: core.ptr, %value: core.i32) -> core.i32 {
    %result = func.call_indirect %callee, %value {signature = func.func_sig<(!evidence) -> core.i32>} : core.i32
    func.return %result
  }
}"#,
        );
        let ptr_ty = core::ptr(&mut ctx).as_type_ref();
        let mut type_converter = TypeConverter::new();
        type_converter.add_conversion(move |ctx, ty| {
            ctx.types()
                .is_dialect(ty, "core", "array")
                .then_some(ptr_ty)
        });

        let error = super::lower(&mut ctx, module, type_converter).unwrap_err();
        assert!(error.to_string().contains("func.call_indirect"), "{error}");

        let printed = print_module(&ctx, module.op());
        assert!(
            printed.contains("clif.call_indirect")
                && printed.contains("sig = clif.func_sig<(core.ptr) -> core.i32>"),
            "{printed}"
        );
        assert!(
            printed.contains(
                "func.call_indirect %0, %1 {signature = func.func_sig<(!evidence) -> core.i32>} : core.i32"
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
    func.tail_call_indirect %callee, %value {signature = func.func_sig<(core.i64) -> core.nil>}
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

        let object = crate::emit_module_to_native(&ctx, module).unwrap();
        assert!(!object.is_empty());
    }

    #[test]
    fn tail_call_indirect_without_exact_signature_is_rejected() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @caller(%callee: core.ptr, %value: core.i32) -> core.nil {
    func.tail_call_indirect %callee, %value
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
    func.tail_call_indirect %callee, %value {signature = func.func_sig<(core.i32) -> core.i32>}
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
    func.tail_call_indirect %callee, %value {signature = func.func_sig<(core.i32) -> core.nil>}
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
    fn function_references_must_keep_their_target_call_conv() {
        let lower_reference = |reference: &str| {
            let mut ctx = IrContext::new();
            let module = parse_test_module(
                &mut ctx,
                &format!(
                    r#"core.module @test {{
  func.func @target(%value: core.i32) attributes {{type = func.func_sig<(core.i32) -> (), {{call_conv = "tail"}}>}} {{
    func.return
  }}
  func.func @take() {{
    %reference = func.constant {{func_ref = @target}} : {reference}
    func.return
  }}
}}"#
                ),
            );
            super::lower(&mut ctx, module, TypeConverter::new()).map(|_| ())
        };

        lower_reference("func.func_sig<(core.i32) -> (), {call_conv = \"tail\"}>")
            .expect("a reference with the target convention lowers");
        let error = lower_reference("func.func_sig<(core.i32) -> ()>")
            .expect_err("a platform reference to a tail function must not be erased");
        assert!(error.to_string().contains("func.constant"), "{error}");
    }

    #[test]
    fn nested_function_references_resolve_by_root_qualified_path() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  core.module @inner {
    func.func @helper(%value: core.i32) attributes {type = func.func_sig<(core.i32) -> (), {call_conv = "tail"}>} {
      func.return
    }
    func.func @take() {
      %reference = func.constant {func_ref = @"inner::helper"} : func.func_sig<(core.i32) -> (), {call_conv = "tail"}>
      func.return
    }
  }
}"#,
        );
        super::lower(&mut ctx, module, TypeConverter::new())
            .expect("a nested reference resolves by its qualified path");
        let printed = print_module(&ctx, module.op());
        assert!(printed.contains("clif.symbol_addr"), "{printed}");
        assert!(!printed.contains("func.constant"), "{printed}");
    }

    #[test]
    fn function_reference_conventions_fail_closed() {
        let lower = |input: &str| {
            let mut ctx = IrContext::new();
            let module = parse_test_module(&mut ctx, input);
            super::lower(&mut ctx, module, TypeConverter::new())
                .map(|_| ())
                .unwrap_err()
                .to_string()
        };

        let unknown = lower(
            r#"core.module @test {
  func.func @take() {
    %reference = func.constant {func_ref = @missing} : func.func_sig<(core.i32) -> (), {call_conv = "tail"}>
    func.return
  }
}"#,
        );
        assert!(unknown.contains("func.constant"), "{unknown}");

        let duplicated = lower(
            r#"core.module @test {
  core.module @left {
    func.func @helper(%value: core.i32) attributes {type = func.func_sig<(core.i32) -> (), {call_conv = "tail"}>} {
      func.return
    }
    func.func @helper(%value: core.i32) attributes {type = func.func_sig<(core.i32) -> (), {call_conv = "tail"}>} {
      func.return
    }
    func.func @take() {
      %reference = func.constant {func_ref = @"left::helper"} : func.func_sig<(core.i32) -> (), {call_conv = "tail"}>
      func.return
    }
  }
}"#,
        );
        assert!(duplicated.contains("func.constant"), "{duplicated}");

        let different_inputs = lower(
            r#"core.module @test {
  func.func @target(%env: core.ptr, %value: core.i32) attributes {type = func.func_sig<(core.ptr, core.i32) -> (), {call_conv = "tail"}>} {
    func.return
  }
  func.func @take() {
    %reference = func.constant {func_ref = @target} : func.func_sig<(core.i32) -> (), {call_conv = "tail"}>
    func.return
  }
}"#,
        );
        assert!(
            different_inputs.contains("func.constant"),
            "{different_inputs}"
        );
    }

    #[test]
    fn tail_call_indirect_without_tail_call_conv_is_rejected_before_mutation() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @caller(%callee: core.ptr, %value: core.i32) {
    func.tail_call_indirect %callee, %value {signature = func.func_sig<(core.i32) -> ()>}
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
}
