//! Wasm-owned callable signature conversion.

use trunk_ir::context::{IrContext, OperationDataBuilder};
use trunk_ir::dialect::wasm;
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{OpRef, TypeRef};
use trunk_ir::rewrite::{
    PatternRewriter, RewritePattern, convert_signature_components, rewrite_function_signature,
};
use trunk_ir::types::Attribute;

/// Converts validated `wasm.func` signatures through a `TypeConverter`.
pub struct WasmFuncSignatureConversionPattern;

impl RewritePattern for WasmFuncSignatureConversionPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(wasm_func) = wasm::Func::from_op(ctx, op) else {
            return false;
        };
        let func_type = wasm_func.r#type(ctx);
        let Some(signature) = wasm::FuncSig::from_type_ref(ctx, func_type) else {
            return false;
        };
        let converted = convert_signature_components(
            ctx,
            rewriter.type_converter(),
            signature.inputs(ctx),
            signature.results(ctx),
            signature.non_reserved_attrs(ctx),
        );
        if !converted.changed {
            return false;
        }
        let new_func_type = wasm::func_sig_with_attrs(
            ctx,
            converted.inputs.iter().copied(),
            converted.results.iter().copied(),
            converted.attrs,
        )
        .as_type_ref();
        let body = ctx.op(op).regions.first().copied();
        let sym_name = wasm_func.sym_name(ctx);
        let loc = ctx.op(op).location;

        rewrite_function_signature(
            ctx,
            op,
            rewriter,
            body,
            new_func_type,
            &converted.inputs,
            |ctx, ty, body| match body {
                Some(body) => wasm::func(ctx, loc, sym_name, ty, body).op_ref(),
                None => make_bodyless_wasm_func(ctx, loc, sym_name, ty),
            },
        )
    }

    fn name(&self) -> &'static str {
        "WasmFuncSignatureConversionPattern"
    }
}

fn make_bodyless_wasm_func(
    ctx: &mut IrContext,
    loc: trunk_ir::types::Location,
    sym_name: trunk_ir::Symbol,
    func_type: TypeRef,
) -> OpRef {
    let data = OperationDataBuilder::new(
        loc,
        trunk_ir::Symbol::new("wasm"),
        trunk_ir::Symbol::new("func"),
    )
    .attr("sym_name", Attribute::Symbol(sym_name))
    .attr("type", Attribute::Type(func_type))
    .build(ctx);
    ctx.create_op(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::Symbol;
    use trunk_ir::context::{BlockArgData, BlockData, RegionData};
    use trunk_ir::location::Span;
    use trunk_ir::rewrite::{ConversionTarget, Module, PatternApplicator, TypeConverter};
    use trunk_ir::smallvec::smallvec;
    use trunk_ir::types::TypeDataBuilder;

    fn test_ctx() -> (IrContext, trunk_ir::types::Location) {
        let mut ctx = IrContext::new();
        let path = ctx.paths.intern("test.trb".to_owned());
        let loc = trunk_ir::types::Location::new(path, Span::new(0, 0));
        (ctx, loc)
    }

    fn type_ref(ctx: &mut IrContext, name: &'static str) -> TypeRef {
        ctx.types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new(name)).build())
    }

    fn make_module(ctx: &mut IrContext, loc: trunk_ir::types::Location, ops: Vec<OpRef>) -> Module {
        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        for op in ops {
            ctx.push_op(block, op);
        }
        let region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block],
            parent_op: None,
        });
        let module = OperationDataBuilder::new(loc, Symbol::new("core"), Symbol::new("module"))
            .attr("sym_name", Attribute::Symbol(Symbol::new("test")))
            .region(region)
            .build(ctx);
        let module = ctx.create_op(module);
        Module::new(ctx, module).unwrap()
    }

    fn make_wasm_func(
        ctx: &mut IrContext,
        loc: trunk_ir::types::Location,
        name: &'static str,
        signature: TypeRef,
        inputs: &[TypeRef],
    ) -> OpRef {
        let entry = ctx.create_block(BlockData {
            location: loc,
            args: inputs
                .iter()
                .map(|&ty| BlockArgData {
                    ty,
                    attrs: Default::default(),
                })
                .collect(),
            ops: smallvec![],
            parent_region: None,
        });
        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        wasm::func(ctx, loc, Symbol::new(name), signature, body).op_ref()
    }

    fn i32_to_i64(i32: TypeRef, i64: TypeRef) -> TypeConverter {
        let mut converter = TypeConverter::new();
        converter.add_conversion(move |_, ty| (ty == i32).then_some(i64));
        converter
    }

    #[test]
    fn converts_wasm_zero_and_multiple_results_with_nested_metadata() {
        for results in [0, 2] {
            let (mut ctx, loc) = test_ctx();
            let i32 = type_ref(&mut ctx, "i32");
            let i64 = type_ref(&mut ctx, "i64");
            let ptr = type_ref(&mut ctx, "ptr");
            let mut attrs = trunk_ir::AttributeMap::new();
            attrs.insert(
                Symbol::new("nested"),
                Attribute::List(vec![
                    Attribute::Type(i32),
                    Attribute::List(vec![Attribute::Type(i32)]),
                ]),
            );
            attrs.insert(Symbol::new("tag"), Attribute::Symbol(Symbol::new("keep")));
            let signature = wasm::func_sig_with_attrs(
                &mut ctx,
                [i32],
                (results == 2).then_some(vec![i32, ptr]).unwrap_or_default(),
                attrs,
            )
            .as_type_ref();
            let func = make_wasm_func(&mut ctx, loc, "f", signature, &[i32]);
            let module = make_module(&mut ctx, loc, vec![func]);

            let result = PatternApplicator::new(i32_to_i64(i32, i64))
                .add_pattern(WasmFuncSignatureConversionPattern)
                .with_target(ConversionTarget::new())
                .apply_partial_conversion(&mut ctx, module, "wasm-test")
                .unwrap();
            assert!(result.reached_fixpoint);
            let function = wasm::Func::from_op(&ctx, module.ops(&ctx)[0]).unwrap();
            let signature = wasm::FuncSig::from_type_ref(&ctx, function.r#type(&ctx)).unwrap();
            assert_eq!(signature.inputs(&ctx), [i64]);
            assert_eq!(
                signature.results(&ctx),
                (results == 2).then_some(vec![i64, ptr]).unwrap_or_default()
            );
            let attrs = signature
                .non_reserved_attrs(&ctx)
                .map(|(key, value)| (*key, value.clone()))
                .collect::<trunk_ir::AttributeMap>();
            assert_eq!(
                attrs.get("nested"),
                Some(&Attribute::List(vec![
                    Attribute::Type(i64),
                    Attribute::List(vec![Attribute::Type(i64)]),
                ])),
            );
            assert_eq!(attrs.get_symbol("tag"), Some(Symbol::new("keep")),);
            assert_eq!(ctx.types.get(function.r#type(&ctx)).attrs.len(), 4);
        }
    }

    #[test]
    fn preserves_bodyless_declarations_and_avoids_noop_rewrites() {
        let (mut ctx, loc) = test_ctx();
        let i32 = type_ref(&mut ctx, "i32");
        let i64 = type_ref(&mut ctx, "i64");
        let changed = wasm::func_sig(&mut ctx, [i32], [i32, i32]).as_type_ref();
        let unchanged = wasm::func_sig(&mut ctx, [i64], []).as_type_ref();
        let declaration = make_bodyless_wasm_func(&mut ctx, loc, Symbol::new("extern"), changed);
        let unchanged_func = make_wasm_func(&mut ctx, loc, "unchanged", unchanged, &[i64]);
        let module = make_module(&mut ctx, loc, vec![declaration, unchanged_func]);

        let result = PatternApplicator::new(i32_to_i64(i32, i64))
            .add_pattern(WasmFuncSignatureConversionPattern)
            .with_target(ConversionTarget::new())
            .apply_partial_conversion(&mut ctx, module, "wasm-test")
            .unwrap();
        assert!(result.reached_fixpoint);
        let ops = module.ops(&ctx);
        assert!(ctx.op(ops[0]).regions.is_empty());
        let converted =
            wasm::FuncSig::from_type_ref(&ctx, ctx.op(ops[0]).attributes.get_type("type").unwrap())
                .unwrap();
        assert_eq!(converted.inputs(&ctx), [i64]);
        assert_eq!(converted.results(&ctx), [i64, i64]);
        assert_eq!(
            wasm::Func::from_op(&ctx, ops[1]).unwrap().r#type(&ctx),
            unchanged
        );
        assert_eq!(result.total_changes, 1);
    }

    #[test]
    fn rejects_entry_arity_mismatch_without_mutating() {
        let (mut ctx, loc) = test_ctx();
        let i32 = type_ref(&mut ctx, "i32");
        let i64 = type_ref(&mut ctx, "i64");
        let signature = wasm::func_sig(&mut ctx, [i32, i32], []).as_type_ref();
        let func = make_wasm_func(&mut ctx, loc, "bad", signature, &[i32]);
        let module = make_module(&mut ctx, loc, vec![func]);
        let before = trunk_ir::printer::print_module(&ctx, module.op());

        let result = PatternApplicator::new(i32_to_i64(i32, i64))
            .add_pattern(WasmFuncSignatureConversionPattern)
            .with_target(ConversionTarget::new())
            .apply_partial_conversion(&mut ctx, module, "wasm-test")
            .unwrap();
        assert!(result.reached_fixpoint);
        let function = wasm::Func::from_op(&ctx, module.ops(&ctx)[0]).unwrap();
        assert_eq!(function.r#type(&ctx), signature);
        let entry = ctx.region(function.body(&ctx)).blocks[0];
        assert_eq!(ctx.value_ty(ctx.block_arg(entry, 0)), i32);
        assert_eq!(trunk_ir::printer::print_module(&ctx, module.op()), before);
    }

    #[test]
    fn rejects_malformed_wasm_signature_without_mutating() {
        let (mut ctx, loc) = test_ctx();
        let i32 = type_ref(&mut ctx, "i32");
        let i64 = type_ref(&mut ctx, "i64");
        let malformed = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("wasm"), Symbol::new("func_sig")).build());
        let func = make_bodyless_wasm_func(&mut ctx, loc, Symbol::new("bad"), malformed);
        let module = make_module(&mut ctx, loc, vec![func]);
        let before = trunk_ir::printer::print_module(&ctx, module.op());

        let result = PatternApplicator::new(i32_to_i64(i32, i64))
            .add_pattern(WasmFuncSignatureConversionPattern)
            .with_target(ConversionTarget::new())
            .apply_partial_conversion(&mut ctx, module, "wasm-test")
            .unwrap();
        assert!(result.reached_fixpoint);
        assert_eq!(
            ctx.op(module.ops(&ctx)[0]).attributes.get_type("type"),
            Some(malformed)
        );
        assert_eq!(trunk_ir::printer::print_module(&ctx, module.op()), before);
    }
}
