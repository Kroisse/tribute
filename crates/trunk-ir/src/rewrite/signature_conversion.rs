//! Arena-based function signature conversion patterns.
//!
//! Provides MLIR-style signature conversion that automatically converts
//! function parameter and return types using a `TypeConverter`.
//!
//! - [`FuncSignatureConversionPattern`]: Converts `func.func` signatures
//! - [`WasmFuncSignatureConversionPattern`]: Converts `wasm.func` signatures

use crate::context::{IrContext, OperationDataBuilder};
use crate::dialect::{core, func, wasm};
use crate::ops::{DialectOp, DialectType};
use crate::refs::{OpRef, RegionRef, TypeRef};
use crate::rewrite::clone_attrs_except;
use crate::rewrite::pattern::RewritePattern;
use crate::rewrite::rewriter::PatternRewriter;
use crate::rewrite::type_converter::TypeConverter;
use crate::types::Attribute;

/// Result of converting a `core.func` type's params and result.
struct ConvertedSignature {
    new_params: Vec<TypeRef>,
    new_results: Vec<TypeRef>,
    type_attrs: crate::AttributeMap,
    changed: bool,
}

/// Analyze a `core.func` TypeRef and convert params/result via the type converter.
///
/// Returns `None` when `func_type` is not a well-formed `core.func` type.
fn convert_func_signature(
    ctx: &IrContext,
    func_type: TypeRef,
    converter: &TypeConverter,
) -> Option<ConvertedSignature> {
    let func = core::Func::from_type_ref(ctx, func_type)?;
    let data = ctx.types.get(func_type);
    let type_attrs = func
        .non_reserved_attrs(ctx)
        .map(|(key, value)| (*key, convert_attribute_types(ctx, converter, value)))
        .collect::<crate::AttributeMap>();
    let attrs_changed = type_attrs
        .iter()
        .any(|(key, value)| data.attrs.get(key) != Some(value));

    let old_results = func.results(ctx);
    let old_params = func.inputs(ctx);

    let new_results: Vec<_> = old_results
        .iter()
        .map(|&ty| converter.convert_type_or_identity(ctx, ty))
        .collect();
    let new_params: Vec<TypeRef> = old_params
        .iter()
        .map(|&ty| converter.convert_type_or_identity(ctx, ty))
        .collect();

    let params_changed = new_params
        .iter()
        .zip(old_params.iter())
        .any(|(new, old)| new != old);
    let result_changed = new_results != old_results;

    Some(ConvertedSignature {
        new_params,
        new_results,
        type_attrs,
        changed: params_changed || result_changed || attrs_changed,
    })
}

fn convert_attribute_types(
    ctx: &IrContext,
    converter: &TypeConverter,
    attribute: &Attribute,
) -> Attribute {
    match attribute {
        Attribute::Type(ty) => Attribute::Type(converter.convert_type_or_identity(ctx, *ty)),
        Attribute::List(values) => Attribute::List(
            values
                .iter()
                .map(|value| convert_attribute_types(ctx, converter, value))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// Build a new `core.func` TypeRef from converted params/result.
fn rebuild_func_type(ctx: &mut IrContext, sig: &ConvertedSignature) -> TypeRef {
    crate::dialect::core::func_with_attrs(
        ctx,
        sig.new_params.iter().copied(),
        sig.new_results.iter().copied(),
        sig.type_attrs.clone(),
    )
    .as_type_ref()
}

/// Convert the parameter and result types of a `core.func` type.
///
/// This is the type-only counterpart to the function-signature rewrite
/// patterns. It is for exact callable attributes whose source operation does
/// not own a function body or block arguments to rewrite.
pub fn convert_function_type(
    ctx: &mut IrContext,
    func_type: TypeRef,
    converter: &TypeConverter,
) -> Option<TypeRef> {
    let signature = convert_func_signature(ctx, func_type, converter)?;
    Some(rebuild_func_type(ctx, &signature))
}

/// Update entry block argument types in-place to match converted params.
///
/// Returns `false` if there is an arity mismatch between the entry block args
/// and the new params, leaving the IR unchanged to avoid partial updates.
fn update_entry_block_args(ctx: &mut IrContext, op: OpRef, new_params: &[TypeRef]) -> bool {
    let regions = &ctx.op(op).regions;
    if regions.is_empty() {
        // Declarations have no entry block whose arguments need updating.
        return true;
    }
    let body = regions[0];
    let blocks = &ctx.region(body).blocks;
    if blocks.is_empty() {
        return new_params.is_empty();
    }
    let entry_block = blocks[0];

    let num_args = ctx.block(entry_block).args.len();
    if num_args != new_params.len() {
        return false;
    }
    for (i, &new_ty) in new_params.iter().enumerate() {
        ctx.set_block_arg_type(entry_block, i as u32, new_ty);
    }
    true
}

/// Create a bodyless function declaration for a dialect whose generated
/// constructor requires a body region.
fn make_bodyless_function_op(
    ctx: &mut IrContext,
    loc: crate::types::Location,
    dialect: crate::Symbol,
    sym_name: crate::Symbol,
    func_type: TypeRef,
) -> OpRef {
    let data = OperationDataBuilder::new(loc, dialect, crate::Symbol::new("func"))
        .attr("sym_name", Attribute::Symbol(sym_name))
        .attr("type", Attribute::Type(func_type))
        .build(ctx);
    ctx.create_op(data)
}

/// Shared implementation for function signature conversion.
///
/// Converts parameter and result types using the type converter, updates
/// entry block argument types, rebuilds the function type, and replaces
/// the operation. Accepts a constructor closure to create the replacement op,
/// allowing reuse across `func.func` and `wasm.func` patterns.
fn rewrite_function_signature(
    ctx: &mut IrContext,
    op: OpRef,
    rewriter: &mut PatternRewriter<'_>,
    func_type: TypeRef,
    body: Option<RegionRef>,
    make_op: impl FnOnce(&mut IrContext, TypeRef, Option<RegionRef>) -> OpRef,
) -> bool {
    let converter = rewriter.type_converter();
    let attrs_to_preserve = clone_attrs_except(ctx, op, &["sym_name", "type"]);

    let Some(sig) = convert_func_signature(ctx, func_type, converter) else {
        return false;
    };
    if !sig.changed {
        return false;
    }

    // Update entry block args in-place
    if !update_entry_block_args(ctx, op, &sig.new_params) {
        return false;
    }

    // Build new func type
    let new_func_type = rebuild_func_type(ctx, &sig);

    // Detach body region so it can be reused in the new op
    if let Some(body) = body {
        ctx.detach_region(body);
    }

    // Create replacement op with new type
    let new_op = make_op(ctx, new_func_type, body);
    ctx.op_mut(new_op).attributes.extend(attrs_to_preserve);

    rewriter.replace_op(new_op);
    true
}

/// Pattern that converts `func.func` operation signatures using a `TypeConverter`.
///
/// This pattern:
/// 1. Matches `func.func` operations
/// 2. Converts parameter and result types using the type converter
/// 3. Updates entry block argument types to match
/// 4. Rebuilds the function with the converted signature
pub struct FuncSignatureConversionPattern;

impl RewritePattern for FuncSignatureConversionPattern {
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
        let body = ctx.op(op).regions.first().copied();
        let sym_name = func_op.sym_name(ctx);
        let loc = ctx.op(op).location;

        rewrite_function_signature(
            ctx,
            op,
            rewriter,
            func_type,
            body,
            |ctx, ty, body| match body {
                Some(body) => func::func(ctx, loc, sym_name, ty, body).op_ref(),
                None => {
                    make_bodyless_function_op(ctx, loc, crate::Symbol::new("func"), sym_name, ty)
                }
            },
        )
    }

    fn name(&self) -> &'static str {
        "FuncSignatureConversionPattern"
    }
}

/// Pattern that converts `wasm.func` operation signatures using a `TypeConverter`.
///
/// Identical to [`FuncSignatureConversionPattern`] but targets `wasm.func` operations.
pub struct WasmFuncSignatureConversionPattern;

impl RewritePattern for WasmFuncSignatureConversionPattern {
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
        let body = ctx.op(op).regions.first().copied();
        let sym_name = wasm_func_op.sym_name(ctx);
        let loc = ctx.op(op).location;

        rewrite_function_signature(
            ctx,
            op,
            rewriter,
            func_type,
            body,
            |ctx, ty, body| match body {
                Some(body) => wasm::func(ctx, loc, sym_name, ty, body).op_ref(),
                None => {
                    make_bodyless_function_op(ctx, loc, crate::Symbol::new("wasm"), sym_name, ty)
                }
            },
        )
    }

    fn name(&self) -> &'static str {
        "WasmFuncSignatureConversionPattern"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Symbol;
    use crate::context::{BlockArgData, BlockData, IrContext, OperationDataBuilder, RegionData};
    use crate::location::Span;
    use crate::printer::print_module;
    use crate::rewrite::{ConversionTarget, Module, PatternApplicator, TypeConverter};
    use crate::types::{Attribute, TypeDataBuilder};
    use smallvec::smallvec;
    fn test_ctx() -> (IrContext, crate::types::Location) {
        let mut ctx = IrContext::new();
        let path = ctx.paths.intern("test.trb".to_owned());
        let loc = crate::types::Location::new(path, Span::new(0, 0));
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
        crate::dialect::core::func(ctx, params.iter().copied(), [ret]).as_type_ref()
    }

    fn make_module(ctx: &mut IrContext, loc: crate::types::Location, ops: Vec<OpRef>) -> Module {
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
        Module::new(ctx, module_op).expect("test module should be valid")
    }

    /// Create a func.func op with a body region containing an entry block with args.
    fn make_func_op(
        ctx: &mut IrContext,
        loc: crate::types::Location,
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
                    attrs: Default::default(),
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
        loc: crate::types::Location,
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
                    attrs: Default::default(),
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
    fn i32_to_i64_converter(i32_ty: TypeRef, i64_ty: TypeRef) -> TypeConverter {
        let mut tc = TypeConverter::new();
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

        let mut type_attrs = crate::AttributeMap::new();
        type_attrs.insert(Symbol::new("metadata_type"), Attribute::Type(i32_ty));
        let func_ty = core::func_with_attrs(&mut ctx, [i32_ty], [i32_ty], type_attrs).as_type_ref();
        let func_op = make_func_op(&mut ctx, loc, "test_fn", func_ty, &[i32_ty]);
        let module = make_module(&mut ctx, loc, vec![func_op]);

        let tc = i32_to_i64_converter(i32_ty, i64_ty);
        let applicator = PatternApplicator::new(tc).add_pattern(FuncSignatureConversionPattern);
        let target = ConversionTarget::new();

        let result = applicator
            .with_target(target)
            .apply_partial_conversion(&mut ctx, module, "test-boundary")
            .unwrap();
        assert!(result.reached_fixpoint);
        // 1 block arg converted by applicator + 1 pattern match
        assert!(result.total_changes >= 1);

        // Verify converted type
        let ops = module.ops(&ctx);
        assert_eq!(ops.len(), 1);
        let new_func = func::Func::from_op(&ctx, ops[0]).unwrap();
        let new_type = new_func.r#type(&ctx);
        let function = core::Func::from_type_ref(&ctx, new_type).unwrap();
        assert_eq!(function.inputs(&ctx), &[i64_ty]);
        assert_eq!(function.single_result(&ctx), Some(i64_ty));
        assert_eq!(
            ctx.types.get(new_type).attrs.get_type("metadata_type"),
            Some(i64_ty),
            "nested type attributes should be converted and preserved"
        );

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
        let target = ConversionTarget::new();

        let result = applicator
            .with_target(target)
            .apply_partial_conversion(&mut ctx, module, "test-boundary")
            .unwrap();
        assert!(result.reached_fixpoint);
        assert_eq!(result.total_changes, 0);
    }

    #[test]
    fn resultless_function_type_conversion_preserves_cardinality() {
        let (mut ctx, _) = test_ctx();
        let i32_ty = i32_type(&mut ctx);
        let i64_ty = i64_type(&mut ctx);
        let resultless = core::func(&mut ctx, [i32_ty, i32_ty], []).as_type_ref();
        let before = ctx.types.get(resultless).clone();
        let converter = i32_to_i64_converter(i32_ty, i64_ty);

        let converted = convert_function_type(&mut ctx, resultless, &converter).unwrap();
        assert_eq!(ctx.types.get(resultless), &before);
        let function = core::Func::from_type_ref(&ctx, converted).unwrap();
        assert_eq!(function.inputs(&ctx), [i64_ty, i64_ty]);
        assert!(function.is_resultless(&ctx));
    }

    #[test]
    fn wasm_func_signature_conversion() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);
        let i64_ty = i64_type(&mut ctx);

        let func_ty = make_func_type(&mut ctx, &[i32_ty, i32_ty], i32_ty);
        let func_op = make_wasm_func_op(&mut ctx, loc, "wasm_fn", func_ty, &[i32_ty, i32_ty]);
        ctx.op_mut(func_op)
            .attributes
            .insert(Symbol::new("custom"), Attribute::Int(7));
        let module = make_module(&mut ctx, loc, vec![func_op]);

        let tc = i32_to_i64_converter(i32_ty, i64_ty);
        let applicator = PatternApplicator::new(tc).add_pattern(WasmFuncSignatureConversionPattern);
        let target = ConversionTarget::new();

        let result = applicator
            .with_target(target)
            .apply_partial_conversion(&mut ctx, module, "test-boundary")
            .unwrap();
        assert!(result.reached_fixpoint);
        // 2 block args converted by applicator + 1 pattern match
        assert!(result.total_changes >= 1);

        // Verify converted wasm.func
        let ops = module.ops(&ctx);
        let new_func = wasm::Func::from_op(&ctx, ops[0]).unwrap();
        let new_type = new_func.r#type(&ctx);
        let function = core::Func::from_type_ref(&ctx, new_type).unwrap();
        assert_eq!(function.inputs(&ctx), &[i64_ty, i64_ty]);
        assert_eq!(function.single_result(&ctx), Some(i64_ty));
        assert_eq!(
            ctx.op(ops[0]).attributes.get("custom"),
            Some(&Attribute::Int(7)),
            "signature conversion should preserve custom metadata"
        );

        // Verify entry block args
        let body = new_func.body(&ctx);
        let entry = ctx.region(body).blocks[0];
        assert_eq!(ctx.block(entry).args.len(), 2);
        assert_eq!(ctx.value_ty(ctx.block_arg(entry, 0)), i64_ty);
        assert_eq!(ctx.value_ty(ctx.block_arg(entry, 1)), i64_ty);
    }

    #[test]
    fn bodyless_signatures_convert_without_inventing_bodies() {
        for result_count in [0, 1] {
            let (mut ctx, loc) = test_ctx();
            let i32_ty = i32_type(&mut ctx);
            let i64_ty = i64_type(&mut ctx);
            let results = if result_count == 0 {
                vec![]
            } else {
                vec![i32_ty]
            };
            let func_ty = core::func(&mut ctx, [i32_ty], results).as_type_ref();

            let func_decl = make_bodyless_function_op(
                &mut ctx,
                loc,
                Symbol::new("func"),
                Symbol::new("external"),
                func_ty,
            );
            let wasm_decl = make_bodyless_function_op(
                &mut ctx,
                loc,
                Symbol::new("wasm"),
                Symbol::new("wasm_external"),
                func_ty,
            );
            let func_def = make_func_op(&mut ctx, loc, "defined", func_ty, &[i32_ty]);
            let wasm_def = make_wasm_func_op(&mut ctx, loc, "wasm_defined", func_ty, &[i32_ty]);
            let module = make_module(
                &mut ctx,
                loc,
                vec![func_decl, wasm_decl, func_def, wasm_def],
            );

            let tc = i32_to_i64_converter(i32_ty, i64_ty);
            let applicator = PatternApplicator::new(tc)
                .add_pattern(FuncSignatureConversionPattern)
                .add_pattern(WasmFuncSignatureConversionPattern);
            let result = applicator
                .with_target(ConversionTarget::new())
                .apply_partial_conversion(&mut ctx, module, "test-boundary")
                .unwrap();

            assert!(result.reached_fixpoint);
            assert!(result.total_changes >= 4);

            let ops = module.ops(&ctx);
            for (index, expected_regions) in [(0, 0), (1, 0), (2, 1), (3, 1)] {
                let data = ctx.op(ops[index]);
                assert_eq!(data.regions.len(), expected_regions);
                let func_ty = data.attributes.get_type("type").unwrap();
                let function = core::Func::from_type_ref(&ctx, func_ty).unwrap();
                assert_eq!(function.inputs(&ctx), [i64_ty]);
                assert_eq!(function.results(&ctx), vec![i64_ty; result_count]);
            }

            let text = print_module(&ctx, module.op());
            let arrow = if result_count == 0 {
                ""
            } else {
                " -> core.i64"
            };
            let result = if result_count == 0 { "()" } else { "core.i64" };
            assert!(text.contains(&format!("func.func @external(%arg0: core.i64){arrow}\n")));
            assert!(
                text.contains(&format!("!t0 = core.func<(core.i64) -> {result}>")),
                "{text}"
            );
            assert!(
                text.contains("wasm.func {sym_name = @wasm_external, type = !t0}\n"),
                "{text}"
            );
            assert!(text.contains(&format!("func.func @defined(%0: core.i64){arrow} {{")));
        }
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
        let target = ConversionTarget::new();

        let result = applicator
            .with_target(target)
            .apply_partial_conversion(&mut ctx, module, "test-boundary")
            .unwrap();
        // 1 block arg converted + 1 pattern match
        assert!(result.total_changes >= 1);

        let ops = module.ops(&ctx);
        let new_func = func::Func::from_op(&ctx, ops[0]).unwrap();
        let function = core::Func::from_type_ref(&ctx, new_func.r#type(&ctx)).unwrap();
        assert_eq!(function.inputs(&ctx), &[i64_ty]);
        assert_eq!(function.single_result(&ctx), Some(i64_ty));
    }

    #[test]
    fn arity_mismatch_returns_unchanged_for_wasm() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);
        let i64_ty = i64_type(&mut ctx);

        // Signature has 2 params, but entry block has only 1 arg (arity mismatch)
        let func_ty = make_func_type(&mut ctx, &[i32_ty, i32_ty], i32_ty);
        let func_op = make_wasm_func_op(&mut ctx, loc, "mismatched_wasm", func_ty, &[i32_ty]);
        let module = make_module(&mut ctx, loc, vec![func_op]);

        let tc = i32_to_i64_converter(i32_ty, i64_ty);
        let applicator = PatternApplicator::new(tc).add_pattern(WasmFuncSignatureConversionPattern);
        let target = ConversionTarget::new();

        let result = applicator
            .with_target(target)
            .apply_partial_conversion(&mut ctx, module, "test-boundary")
            .unwrap();
        // Pattern should not match due to arity mismatch.
        // Default shared signature conversion leaves entry arguments unchanged.
        assert!(result.reached_fixpoint);

        // Verify original func type attribute is preserved (pattern didn't match)
        let ops = module.ops(&ctx);
        let original_func = wasm::Func::from_op(&ctx, ops[0]).unwrap();
        assert_eq!(original_func.r#type(&ctx), func_ty);
        let entry = ctx.region(original_func.body(&ctx)).blocks[0];
        assert_eq!(ctx.value_ty(ctx.block_arg(entry, 0)), i32_ty);
    }

    #[test]
    fn arity_mismatch_returns_unchanged() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);
        let i64_ty = i64_type(&mut ctx);

        // Signature has 2 params, but entry block has only 1 arg (arity mismatch)
        let func_ty = make_func_type(&mut ctx, &[i32_ty, i32_ty], i32_ty);
        let func_op = make_func_op(&mut ctx, loc, "mismatched", func_ty, &[i32_ty]);
        let module = make_module(&mut ctx, loc, vec![func_op]);

        let tc = i32_to_i64_converter(i32_ty, i64_ty);
        let applicator = PatternApplicator::new(tc).add_pattern(FuncSignatureConversionPattern);
        let target = ConversionTarget::new();

        let result = applicator
            .with_target(target)
            .apply_partial_conversion(&mut ctx, module, "test-boundary")
            .unwrap();
        // Pattern should not match due to arity mismatch.
        // Default shared signature conversion leaves entry arguments unchanged.
        assert!(result.reached_fixpoint);

        // Verify original func type attribute is preserved (pattern didn't match)
        let ops = module.ops(&ctx);
        let original_func = func::Func::from_op(&ctx, ops[0]).unwrap();
        assert_eq!(original_func.r#type(&ctx), func_ty);
        let entry = ctx.region(original_func.body(&ctx)).blocks[0];
        assert_eq!(ctx.value_ty(ctx.block_arg(entry, 0)), i32_ty);
    }
}

#[cfg(test)]
mod result_list_tests {
    use super::*;
    use crate::{Symbol, ops::DialectOp, parser::parse_test_module};

    #[test]
    fn conversion_preserves_all_cardinalities_and_nested_attributes() {
        for inputs in [0, 2] {
            for results in [0, 1] {
                let mut ctx = IrContext::new();
                let nil = core::nil(&mut ctx).as_type_ref();
                let ptr = core::ptr(&mut ctx).as_type_ref();
                let mut attrs = crate::AttributeMap::new();
                attrs.insert(
                    Symbol::new("nested"),
                    Attribute::List(vec![Attribute::List(vec![Attribute::Type(nil)])]),
                );
                attrs.insert(Symbol::new("tag"), Attribute::Symbol(Symbol::new("keep")));
                let signature =
                    core::func_with_attrs(&mut ctx, vec![nil; inputs], vec![nil; results], attrs)
                        .as_type_ref();
                let mut converter = TypeConverter::new();
                converter.add_conversion(move |_, ty| (ty == nil).then_some(ptr));
                let converted = convert_function_type(&mut ctx, signature, &converter).unwrap();
                let function = core::Func::from_type_ref(&ctx, converted).unwrap();
                assert_eq!(function.inputs(&ctx), vec![ptr; inputs]);
                assert_eq!(function.results(&ctx), vec![ptr; results]);
                assert_eq!(
                    ctx.types.get(converted).attrs.get("nested"),
                    Some(&Attribute::List(vec![Attribute::List(vec![
                        Attribute::Type(ptr)
                    ])]))
                );
                assert_eq!(
                    ctx.types.get(converted).attrs.get_symbol("tag"),
                    Some(Symbol::new("keep"))
                );
            }
        }
    }

    #[test]
    fn signature_rewrite_rejects_entry_mismatch_atomically() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            "core.module @m { func.func @f(%x: core.nil) { func.return } }",
        );
        let op = module.ops(&ctx)[0];
        let function = func::Func::from_op(&ctx, op).unwrap();
        let nil = core::nil(&mut ctx).as_type_ref();
        let ptr = core::ptr(&mut ctx).as_type_ref();
        let bad_signature = core::func(&mut ctx, [nil, nil], []).as_type_ref();
        ctx.op_mut(op)
            .attributes
            .insert(Symbol::new("type"), Attribute::Type(bad_signature));
        let before = crate::printer::print_module(&ctx, module.op());
        let mut converter = TypeConverter::new();
        converter.add_conversion(move |_, ty| (ty == nil).then_some(ptr));
        let mut rewriter = PatternRewriter::new(&converter);
        assert!(!FuncSignatureConversionPattern.match_and_rewrite(&mut ctx, op, &mut rewriter));
        assert_eq!(ctx.op(op).attributes.get_type("type"), Some(bad_signature));
        let entry = ctx.region(function.body(&ctx)).blocks[0];
        assert_eq!(ctx.value_ty(ctx.block_arg(entry, 0)), nil);
        assert_eq!(crate::printer::print_module(&ctx, module.op()), before);
    }
}
