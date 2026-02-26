//! Arena-based function signature conversion patterns.
//!
//! Provides MLIR-style signature conversion that automatically converts
//! function parameter and return types using an `ArenaTypeConverter`.
//!
//! - [`FuncSignatureConversionPattern`]: Converts `func.func` signatures
//! - [`WasmFuncSignatureConversionPattern`]: Converts `wasm.func` signatures

use crate::arena::context::IrContext;
use crate::arena::dialect::{func, wasm};
use crate::arena::ops::ArenaDialectOp;
use crate::arena::refs::{OpRef, TypeRef};
use crate::arena::rewrite::pattern::ArenaRewritePattern;
use crate::arena::rewrite::rewriter::PatternRewriter;
use crate::arena::rewrite::type_converter::ArenaTypeConverter;
use crate::arena::types::{Attribute, TypeDataBuilder};
use crate::ir::Symbol;

/// Result of converting a `func.fn` type's params and result.
struct ConvertedSignature {
    new_params: Vec<TypeRef>,
    new_result: TypeRef,
    effect: Option<TypeRef>,
    params_changed: bool,
}

/// Analyze a `func.fn` TypeRef and convert params/result via the type converter.
///
/// `func.fn` type layout: `params[0]` = return type, `params[1..]` = param types,
/// `attrs["effect"]` = optional effect type.
///
/// Returns `None` if no types changed.
fn convert_func_signature(
    ctx: &IrContext,
    func_type: TypeRef,
    converter: &ArenaTypeConverter,
) -> Option<ConvertedSignature> {
    let type_data = ctx.types.get(func_type);

    // Verify this is a func.fn type
    if type_data.dialect != Symbol::new("func") || type_data.name != Symbol::new("fn") {
        return None;
    }

    if type_data.params.is_empty() {
        return None;
    }

    let old_result = type_data.params[0];
    let old_params = &type_data.params[1..];

    let new_result = converter.convert_type_or_identity(ctx, old_result);
    let new_params: Vec<TypeRef> = old_params
        .iter()
        .map(|&ty| converter.convert_type_or_identity(ctx, ty))
        .collect();

    let params_changed = new_params
        .iter()
        .zip(old_params.iter())
        .any(|(new, old)| new != old);
    let result_changed = new_result != old_result;

    if !params_changed && !result_changed {
        return None;
    }

    let effect = type_data
        .attrs
        .get(&Symbol::new("effect"))
        .and_then(|a| match a {
            Attribute::Type(ty) => Some(*ty),
            _ => None,
        });

    Some(ConvertedSignature {
        new_params,
        new_result,
        effect,
        params_changed,
    })
}

/// Build a new `func.fn` TypeRef from converted params/result/effect.
fn rebuild_func_type(ctx: &mut IrContext, sig: &ConvertedSignature) -> TypeRef {
    let mut builder = TypeDataBuilder::new(Symbol::new("func"), Symbol::new("fn"))
        .param(sig.new_result)
        .params(sig.new_params.iter().copied());

    if let Some(eff) = sig.effect {
        builder = builder.attr("effect", Attribute::Type(eff));
    }

    ctx.types.intern(builder.build())
}

/// Update entry block argument types in-place to match converted params.
fn update_entry_block_args(ctx: &mut IrContext, op: OpRef, new_params: &[TypeRef]) {
    let regions = &ctx.op(op).regions;
    if regions.is_empty() {
        return;
    }
    let body = regions[0];
    let blocks = &ctx.region(body).blocks;
    if blocks.is_empty() {
        return;
    }
    let entry_block = blocks[0];

    let num_args = ctx.block(entry_block).args.len();
    for (i, &new_ty) in new_params.iter().enumerate() {
        if i < num_args {
            ctx.set_block_arg_type(entry_block, i as u32, new_ty);
        }
    }
}

/// Pattern that converts `func.func` operation signatures using an `ArenaTypeConverter`.
///
/// This pattern:
/// 1. Matches `func.func` operations
/// 2. Converts parameter and result types using the type converter
/// 3. Updates entry block argument types to match
/// 4. Rebuilds the function with the converted signature
pub struct FuncSignatureConversionPattern;

impl ArenaRewritePattern for FuncSignatureConversionPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(func_op) = func::Func::from_op(ctx, op) else {
            return false;
        };

        let func_type = func_op.r#type(ctx);
        let converter = rewriter.type_converter();

        let Some(sig) = convert_func_signature(ctx, func_type, converter) else {
            return false;
        };

        // Update entry block args in-place
        if sig.params_changed {
            update_entry_block_args(ctx, op, &sig.new_params);
        }

        // Build new func type
        let new_func_type = rebuild_func_type(ctx, &sig);

        // Detach body region so it can be reused in the new op
        let body = func_op.body(ctx);
        ctx.detach_region(body);

        // Create replacement op with new type
        let loc = ctx.op(op).location;
        let sym_name = func_op.sym_name(ctx);
        let new_op = func::func(ctx, loc, sym_name, new_func_type, body);

        rewriter.replace_op(new_op.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "FuncSignatureConversionPattern"
    }
}

/// Pattern that converts `wasm.func` operation signatures using an `ArenaTypeConverter`.
///
/// Identical to [`FuncSignatureConversionPattern`] but targets `wasm.func` operations.
pub struct WasmFuncSignatureConversionPattern;

impl ArenaRewritePattern for WasmFuncSignatureConversionPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(wasm_func_op) = wasm::Func::from_op(ctx, op) else {
            return false;
        };

        let func_type = wasm_func_op.r#type(ctx);
        let converter = rewriter.type_converter();

        let Some(sig) = convert_func_signature(ctx, func_type, converter) else {
            return false;
        };

        // Update entry block args in-place
        if sig.params_changed {
            update_entry_block_args(ctx, op, &sig.new_params);
        }

        // Build new func type
        let new_func_type = rebuild_func_type(ctx, &sig);

        // Detach body region so it can be reused in the new op
        let body = wasm_func_op.body(ctx);
        ctx.detach_region(body);

        // Create replacement op with new type
        let loc = ctx.op(op).location;
        let sym_name = wasm_func_op.sym_name(ctx);
        let new_op = wasm::func(ctx, loc, sym_name, new_func_type, body);

        rewriter.replace_op(new_op.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "WasmFuncSignatureConversionPattern"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::context::{
        BlockArgData, BlockData, IrContext, OperationDataBuilder, RegionData,
    };
    use crate::arena::rewrite::{
        ArenaConversionTarget, ArenaModule, ArenaTypeConverter, PatternApplicator,
    };
    use crate::location::Span;
    use smallvec::smallvec;
    use std::collections::BTreeMap;

    fn test_ctx() -> (IrContext, crate::arena::types::Location) {
        let mut ctx = IrContext::new();
        let path = ctx.paths.intern("test.trb".to_owned());
        let loc = crate::arena::types::Location::new(path, Span::new(0, 0));
        (ctx, loc)
    }

    fn i32_type(ctx: &mut IrContext) -> TypeRef {
        ctx.types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build())
    }

    fn i64_type(ctx: &mut IrContext) -> TypeRef {
        ctx.types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i64")).build())
    }

    fn make_func_type(ctx: &mut IrContext, params: &[TypeRef], ret: TypeRef) -> TypeRef {
        ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("func"), Symbol::new("fn"))
                .param(ret)
                .params(params.iter().copied())
                .build(),
        )
    }

    fn make_func_type_with_effect(
        ctx: &mut IrContext,
        params: &[TypeRef],
        ret: TypeRef,
        effect: TypeRef,
    ) -> TypeRef {
        ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("func"), Symbol::new("fn"))
                .param(ret)
                .params(params.iter().copied())
                .attr("effect", Attribute::Type(effect))
                .build(),
        )
    }

    fn make_module(
        ctx: &mut IrContext,
        loc: crate::arena::types::Location,
        ops: Vec<OpRef>,
    ) -> ArenaModule {
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
        let module_data =
            OperationDataBuilder::new(loc, Symbol::new("core"), Symbol::new("module"))
                .attr("sym_name", Attribute::Symbol(Symbol::new("test")))
                .region(region)
                .build(ctx);
        let module_op = ctx.create_op(module_data);
        ArenaModule::new(ctx, module_op).expect("test module should be valid")
    }

    /// Create a func.func op with a body region containing an entry block with args.
    fn make_func_op(
        ctx: &mut IrContext,
        loc: crate::arena::types::Location,
        name: &'static str,
        func_type: TypeRef,
        param_types: &[TypeRef],
    ) -> OpRef {
        let entry_block = ctx.create_block(BlockData {
            location: loc,
            args: param_types
                .iter()
                .map(|&ty| BlockArgData {
                    ty,
                    attrs: BTreeMap::new(),
                })
                .collect(),
            ops: smallvec![],
            parent_region: None,
        });
        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry_block],
            parent_op: None,
        });
        let f = func::func(ctx, loc, Symbol::new(name), func_type, body);
        f.op_ref()
    }

    /// Create a wasm.func op with a body region containing an entry block with args.
    fn make_wasm_func_op(
        ctx: &mut IrContext,
        loc: crate::arena::types::Location,
        name: &'static str,
        func_type: TypeRef,
        param_types: &[TypeRef],
    ) -> OpRef {
        let entry_block = ctx.create_block(BlockData {
            location: loc,
            args: param_types
                .iter()
                .map(|&ty| BlockArgData {
                    ty,
                    attrs: BTreeMap::new(),
                })
                .collect(),
            ops: smallvec![],
            parent_region: None,
        });
        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry_block],
            parent_op: None,
        });
        let f = wasm::func(ctx, loc, Symbol::new(name), func_type, body);
        f.op_ref()
    }

    /// i32 → i64 converter
    fn i32_to_i64_converter(i32_ty: TypeRef, i64_ty: TypeRef) -> ArenaTypeConverter {
        let mut tc = ArenaTypeConverter::new();
        tc.add_conversion(move |ctx, ty| {
            if ctx
                .types
                .is_dialect(ty, Symbol::new("core"), Symbol::new("i32"))
            {
                Some(i64_ty)
            } else {
                None
            }
        });
        let _ = i32_ty; // used for clarity
        tc
    }

    #[test]
    fn func_signature_i32_to_i64() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);
        let i64_ty = i64_type(&mut ctx);

        let func_ty = make_func_type(&mut ctx, &[i32_ty], i32_ty);
        let func_op = make_func_op(&mut ctx, loc, "test_fn", func_ty, &[i32_ty]);
        let module = make_module(&mut ctx, loc, vec![func_op]);

        let tc = i32_to_i64_converter(i32_ty, i64_ty);
        let applicator = PatternApplicator::new(tc).add_pattern(FuncSignatureConversionPattern);
        let target = ArenaConversionTarget::new();

        let result = applicator.apply(&mut ctx, module, &target).unwrap();
        assert!(result.reached_fixpoint);
        assert_eq!(result.total_changes, 1);

        // Verify converted type
        let ops = module.ops(&ctx);
        assert_eq!(ops.len(), 1);
        let new_func = func::Func::from_op(&ctx, ops[0]).unwrap();
        let new_type = new_func.r#type(&ctx);
        let td = ctx.types.get(new_type);
        assert_eq!(td.dialect, Symbol::new("func"));
        assert_eq!(td.name, Symbol::new("fn"));
        // params[0] = return type, params[1..] = param types
        assert_eq!(td.params[0], i64_ty, "return type should be i64");
        assert_eq!(td.params[1], i64_ty, "param type should be i64");

        // Verify entry block args are updated
        let body = new_func.body(&ctx);
        let entry = ctx.region(body).blocks[0];
        assert_eq!(ctx.block(entry).args[0].ty, i64_ty);
        assert_eq!(ctx.value_ty(ctx.block_arg(entry, 0)), i64_ty);
    }

    #[test]
    fn no_change_when_types_not_matched() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);
        let i64_ty = i64_type(&mut ctx);

        // Function with i64 params — i32→i64 converter won't match
        let func_ty = make_func_type(&mut ctx, &[i64_ty], i64_ty);
        let func_op = make_func_op(&mut ctx, loc, "already_i64", func_ty, &[i64_ty]);
        let module = make_module(&mut ctx, loc, vec![func_op]);

        let tc = i32_to_i64_converter(i32_ty, i64_ty);
        let applicator = PatternApplicator::new(tc).add_pattern(FuncSignatureConversionPattern);
        let target = ArenaConversionTarget::new();

        let result = applicator.apply(&mut ctx, module, &target).unwrap();
        assert!(result.reached_fixpoint);
        assert_eq!(result.total_changes, 0);
    }

    #[test]
    fn wasm_func_signature_conversion() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);
        let i64_ty = i64_type(&mut ctx);

        let func_ty = make_func_type(&mut ctx, &[i32_ty, i32_ty], i32_ty);
        let func_op = make_wasm_func_op(&mut ctx, loc, "wasm_fn", func_ty, &[i32_ty, i32_ty]);
        let module = make_module(&mut ctx, loc, vec![func_op]);

        let tc = i32_to_i64_converter(i32_ty, i64_ty);
        let applicator = PatternApplicator::new(tc).add_pattern(WasmFuncSignatureConversionPattern);
        let target = ArenaConversionTarget::new();

        let result = applicator.apply(&mut ctx, module, &target).unwrap();
        assert!(result.reached_fixpoint);
        assert_eq!(result.total_changes, 1);

        // Verify converted wasm.func
        let ops = module.ops(&ctx);
        let new_func = wasm::Func::from_op(&ctx, ops[0]).unwrap();
        let new_type = new_func.r#type(&ctx);
        let td = ctx.types.get(new_type);
        assert_eq!(td.params[0], i64_ty, "return type should be i64");
        assert_eq!(td.params[1], i64_ty, "first param should be i64");
        assert_eq!(td.params[2], i64_ty, "second param should be i64");

        // Verify entry block args
        let body = new_func.body(&ctx);
        let entry = ctx.region(body).blocks[0];
        assert_eq!(ctx.block(entry).args.len(), 2);
        assert_eq!(ctx.value_ty(ctx.block_arg(entry, 0)), i64_ty);
        assert_eq!(ctx.value_ty(ctx.block_arg(entry, 1)), i64_ty);
    }

    #[test]
    fn effect_attribute_preserved() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);
        let i64_ty = i64_type(&mut ctx);

        let effect_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("effect_row")).build());
        let func_ty = make_func_type_with_effect(&mut ctx, &[i32_ty], i32_ty, effect_ty);
        let func_op = make_func_op(&mut ctx, loc, "effectful", func_ty, &[i32_ty]);
        let module = make_module(&mut ctx, loc, vec![func_op]);

        let tc = i32_to_i64_converter(i32_ty, i64_ty);
        let applicator = PatternApplicator::new(tc).add_pattern(FuncSignatureConversionPattern);
        let target = ArenaConversionTarget::new();

        let result = applicator.apply(&mut ctx, module, &target).unwrap();
        assert_eq!(result.total_changes, 1);

        // Verify effect is preserved in the new type
        let ops = module.ops(&ctx);
        let new_func = func::Func::from_op(&ctx, ops[0]).unwrap();
        let new_type = new_func.r#type(&ctx);
        let td = ctx.types.get(new_type);
        let effect_attr = td.attrs.get(&Symbol::new("effect"));
        assert_eq!(effect_attr, Some(&Attribute::Type(effect_ty)));
    }

    #[test]
    fn partial_conversion_only_params() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);
        let i64_ty = i64_type(&mut ctx);

        // Return i64, params i32 — only params should change
        let func_ty = make_func_type(&mut ctx, &[i32_ty], i64_ty);
        let func_op = make_func_op(&mut ctx, loc, "partial", func_ty, &[i32_ty]);
        let module = make_module(&mut ctx, loc, vec![func_op]);

        let tc = i32_to_i64_converter(i32_ty, i64_ty);
        let applicator = PatternApplicator::new(tc).add_pattern(FuncSignatureConversionPattern);
        let target = ArenaConversionTarget::new();

        let result = applicator.apply(&mut ctx, module, &target).unwrap();
        assert_eq!(result.total_changes, 1);

        let ops = module.ops(&ctx);
        let new_func = func::Func::from_op(&ctx, ops[0]).unwrap();
        let td = ctx.types.get(new_func.r#type(&ctx));
        assert_eq!(td.params[0], i64_ty, "return stays i64");
        assert_eq!(td.params[1], i64_ty, "param converted to i64");
    }
}
