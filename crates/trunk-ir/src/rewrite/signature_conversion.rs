//! Arena-based function signature conversion patterns.
//!
//! Provides MLIR-style signature conversion that automatically converts
//! function parameter and return types using a `TypeConverter`.
//!
//! - [`FuncSignatureConversionPattern`]: Converts `func.func` signatures

use crate::context::{IrContext, OperationDataBuilder};
use crate::dialect::func;
use crate::ops::{DialectOp, DialectType};
use crate::refs::{OpRef, RegionRef, TypeRef};
use crate::rewrite::clone_attrs_except;
use crate::rewrite::pattern::RewritePattern;
use crate::rewrite::rewriter::PatternRewriter;
use crate::rewrite::type_converter::TypeConverter;
use crate::types::Attribute;

/// Converted components of a validated callable signature.
///
/// The caller owns dialect-specific validation and reconstruction. This helper
/// only maps the supplied components and non-reserved attributes.
pub struct ConvertedSignatureComponents {
    pub inputs: Vec<TypeRef>,
    pub results: Vec<TypeRef>,
    pub attrs: crate::AttributeMap,
    pub changed: bool,
}

/// Convert validated signature components through a type converter.
///
/// Type-bearing attributes are traversed recursively. The callback remains
/// immutable and is applied at most once to each type occurrence.
pub fn convert_signature_components<'a>(
    ctx: &IrContext,
    converter: &TypeConverter,
    old_inputs: &[TypeRef],
    old_results: &[TypeRef],
    attrs: impl IntoIterator<Item = (&'a crate::Symbol, &'a Attribute)>,
) -> ConvertedSignatureComponents {
    let inputs: Vec<_> = old_inputs
        .iter()
        .map(|&ty| converter.convert_type_or_identity(ctx, ty))
        .collect();
    let results: Vec<_> = old_results
        .iter()
        .map(|&ty| converter.convert_type_or_identity(ctx, ty))
        .collect();
    let mut attrs_changed = false;
    let attrs = attrs
        .into_iter()
        .map(|(key, value)| {
            let converted = convert_attribute_types(ctx, converter, value);
            attrs_changed |= converted != *value;
            (*key, converted)
        })
        .collect();
    let changed = inputs != old_inputs || results != old_results;

    ConvertedSignatureComponents {
        changed: changed || attrs_changed,
        inputs,
        results,
        attrs,
    }
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

/// Convert the parameter and result types of a `func.func_sig` type.
///
/// This is the type-only counterpart to the function-signature rewrite
/// patterns. It is for exact callable attributes whose source operation does
/// not own a function body or block arguments to rewrite.
pub fn convert_function_type(
    ctx: &mut IrContext,
    func_type: TypeRef,
    converter: &TypeConverter,
) -> Option<TypeRef> {
    let func = func::FuncSig::from_type_ref(ctx, func_type)?;
    let signature = convert_signature_components(
        ctx,
        converter,
        func.inputs(ctx),
        func.results(ctx),
        func.non_reserved_attrs(ctx),
    );
    Some(
        func::func_sig_with_attrs(ctx, signature.inputs, signature.results, signature.attrs)
            .as_type_ref(),
    )
}

/// Validate the entry block arity before changing callable signature state.
fn entry_block_for_signature_update(
    ctx: &IrContext,
    op: OpRef,
    new_inputs: &[TypeRef],
) -> Option<Option<crate::BlockRef>> {
    let regions = &ctx.op(op).regions;
    if regions.is_empty() {
        // Declarations have no entry block whose arguments need updating.
        return Some(None);
    }
    let body = regions[0];
    let blocks = &ctx.region(body).blocks;
    if blocks.is_empty() {
        return new_inputs.is_empty().then_some(None);
    }
    let entry_block = blocks[0];

    let num_args = ctx.block(entry_block).args.len();
    (num_args == new_inputs.len()).then_some(Some(entry_block))
}

/// Update a previously validated entry block to match converted inputs.
fn update_entry_block_args(
    ctx: &mut IrContext,
    entry_block: Option<crate::BlockRef>,
    new_inputs: &[TypeRef],
) {
    let Some(entry_block) = entry_block else {
        return;
    };
    for (i, &new_ty) in new_inputs.iter().enumerate() {
        ctx.set_block_arg_type(entry_block, i as u32, new_ty);
    }
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
/// allowing dialect adapters to reuse the operation update sequence.
pub fn rewrite_function_signature(
    ctx: &mut IrContext,
    op: OpRef,
    rewriter: &mut PatternRewriter<'_>,
    body: Option<RegionRef>,
    new_func_type: TypeRef,
    new_inputs: &[TypeRef],
    make_op: impl FnOnce(&mut IrContext, TypeRef, Option<RegionRef>) -> OpRef,
) -> bool {
    let attrs_to_preserve = clone_attrs_except(ctx, op, &["sym_name", "type"]);
    let Some(entry_block) = entry_block_for_signature_update(ctx, op, new_inputs) else {
        return false;
    };

    update_entry_block_args(ctx, entry_block, new_inputs);

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
        let Some(signature) = func::FuncSig::from_type_ref(ctx, func_type) else {
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
        let new_func_type = func::func_sig_with_attrs(
            ctx,
            converted.inputs.iter().copied(),
            converted.results.iter().copied(),
            converted.attrs,
        )
        .as_type_ref();
        let body = ctx.op(op).regions.first().copied();
        let sym_name = func_op.sym_name(ctx);
        let loc = ctx.op(op).location;

        rewrite_function_signature(
            ctx,
            op,
            rewriter,
            body,
            new_func_type,
            &converted.inputs,
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
        crate::dialect::func::func_sig(ctx, params.iter().copied(), [ret]).as_type_ref()
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
        let func_ty =
            func::func_sig_with_attrs(&mut ctx, [i32_ty], [i32_ty], type_attrs).as_type_ref();
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
        let function = func::FuncSig::from_type_ref(&ctx, new_type).unwrap();
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
        let resultless = func::func_sig(&mut ctx, [i32_ty, i32_ty], []).as_type_ref();
        let before = ctx.types.get(resultless).clone();
        let converter = i32_to_i64_converter(i32_ty, i64_ty);

        let converted = convert_function_type(&mut ctx, resultless, &converter).unwrap();
        assert_eq!(ctx.types.get(resultless), &before);
        let function = func::FuncSig::from_type_ref(&ctx, converted).unwrap();
        assert_eq!(function.inputs(&ctx), [i64_ty, i64_ty]);
        assert!(function.is_resultless(&ctx));
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
            let func_ty = func::func_sig(&mut ctx, [i32_ty], results.clone()).as_type_ref();

            let func_decl = make_bodyless_function_op(
                &mut ctx,
                loc,
                Symbol::new("func"),
                Symbol::new("external"),
                func_ty,
            );
            let func_def = make_func_op(&mut ctx, loc, "defined", func_ty, &[i32_ty]);
            let module = make_module(&mut ctx, loc, vec![func_decl, func_def]);

            let tc = i32_to_i64_converter(i32_ty, i64_ty);
            let applicator = PatternApplicator::new(tc).add_pattern(FuncSignatureConversionPattern);
            let result = applicator
                .with_target(ConversionTarget::new())
                .apply_partial_conversion(&mut ctx, module, "test-boundary")
                .unwrap();

            assert!(result.reached_fixpoint);
            assert!(result.total_changes >= 2);

            let ops = module.ops(&ctx);
            for (index, expected_regions) in [(0, 0), (1, 1)] {
                let data = ctx.op(ops[index]);
                assert_eq!(data.regions.len(), expected_regions);
                let func_ty = data.attributes.get_type("type").unwrap();
                let function = func::FuncSig::from_type_ref(&ctx, func_ty).unwrap();
                assert_eq!(function.inputs(&ctx), [i64_ty]);
                assert_eq!(function.results(&ctx), vec![i64_ty; result_count]);
            }

            let text = print_module(&ctx, module.op());
            let arrow = if result_count == 0 {
                ""
            } else {
                " -> core.i64"
            };
            assert!(text.contains(&format!("func.func @external(%arg0: core.i64){arrow}\n")));
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
        let function = func::FuncSig::from_type_ref(&ctx, new_func.r#type(&ctx)).unwrap();
        assert_eq!(function.inputs(&ctx), &[i64_ty]);
        assert_eq!(function.single_result(&ctx), Some(i64_ty));
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
    use crate::dialect::core;
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
                let signature = func::func_sig_with_attrs(
                    &mut ctx,
                    vec![nil; inputs],
                    vec![nil; results],
                    attrs,
                )
                .as_type_ref();
                let mut converter = TypeConverter::new();
                converter.add_conversion(move |_, ty| (ty == nil).then_some(ptr));
                let converted = convert_function_type(&mut ctx, signature, &converter).unwrap();
                let function = func::FuncSig::from_type_ref(&ctx, converted).unwrap();
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
        let bad_signature = func::func_sig(&mut ctx, [nil, nil], []).as_type_ref();
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
