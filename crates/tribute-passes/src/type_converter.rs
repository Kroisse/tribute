//! Generic type converter for target-agnostic IR transformations.
//!
//! This module provides an arena-based `TypeConverter` configuration for
//! converting high-level Tribute types to their core representations. This
//! converter handles target-agnostic transformations that apply to all backends.
//!
//! ## Type Conversion Rules
//!
//! | Source Type         | Target Type     | Notes                              |
//! |---------------------|-----------------|-------------------------------------|
//! | `tribute_rt.int`    | `core.i32`      | Arbitrary precision → i32 (Phase 1) |
//! | `tribute_rt.nat`    | `core.i32`      | Arbitrary precision → i32 (Phase 1) |
//! | `tribute_rt.bool`   | `core.i32`      | Boolean as i32                      |
//! | `tribute_rt.float`  | `core.f64`      | Float as f64                        |
//!
//! Backend-specific type conversions (e.g., `core.i1 → core.i32`,
//! `tribute_rt.anyref` → `wasm.anyref`) are handled by backend-specific
//! type converters.

use tribute_ir::dialect::tribute_rt;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::core;
use trunk_ir::ops::DialectType;
use trunk_ir::refs::TypeRef;
use trunk_ir::rewrite::type_converter::{MaterializeResult, TypeConverter};

fn is_adt_struct_type(ctx: &IrContext, ty: TypeRef) -> bool {
    tribute_ir::dialect::adt::Struct::matches(ctx, ty)
}

fn is_adt_typeref(ctx: &IrContext, ty: TypeRef) -> bool {
    tribute_ir::dialect::adt::Typeref::matches(ctx, ty)
}

/// Create an TypeConverter configured for target-agnostic type conversions.
///
/// This converter handles the IR-level type transformations that are common
/// across all backends. Backend-specific converters can extend this with
/// additional conversions.
pub fn generic_type_converter(ctx: &mut IrContext) -> TypeConverter {
    // Pre-intern commonly used types (TypeRef is Copy)
    let tribute_rt_int = tribute_rt::Int::params().intern(ctx).as_type_ref();
    let tribute_rt_nat = tribute_rt::Nat::params().intern(ctx).as_type_ref();
    let tribute_rt_bool = tribute_rt::Bool::params().intern(ctx).as_type_ref();
    let tribute_rt_float = tribute_rt::Float::params().intern(ctx).as_type_ref();
    let tribute_rt_anyref = tribute_rt::Anyref::params().intern(ctx).as_type_ref();
    let core_i32 = core::I32::type_ref(ctx);
    let core_f64 = core::F64::type_ref(ctx);

    let canonical_closure = crate::closure_lower::closure_struct_type_ref(ctx);

    let mut tc = TypeConverter::new();

    // Convert tribute_rt.int → core.i32 (Phase 1: arbitrary precision as i32)
    tc.add_conversion(move |_ctx, ty| {
        if ty == tribute_rt_int {
            Some(core_i32)
        } else {
            None
        }
    });

    // Convert tribute_rt.nat → core.i32 (Phase 1: arbitrary precision as i32)
    tc.add_conversion(move |_ctx, ty| {
        if ty == tribute_rt_nat {
            Some(core_i32)
        } else {
            None
        }
    });

    // Convert tribute_rt.bool → core.i32 (boolean as i32)
    tc.add_conversion(move |_ctx, ty| {
        if ty == tribute_rt_bool {
            Some(core_i32)
        } else {
            None
        }
    });

    // Convert tribute_rt.float → core.f64 (float as f64)
    tc.add_conversion(move |_ctx, ty| {
        if ty == tribute_rt_float {
            Some(core_f64)
        } else {
            None
        }
    });

    // Single materializer combining all materialization rules
    tc.set_materializer(move |ctx, location, value, from_ty, to_ty| {
        // Same type - no materialization needed
        if from_ty == to_ty {
            return Some(MaterializeResult { value, ops: vec![] });
        }

        // -----------------------------------------------------------------
        // Primitive type equivalence materializations (NoOp)
        // -----------------------------------------------------------------

        // tribute_rt.int → core.i32 (same representation)
        if from_ty == tribute_rt_int && to_ty == core_i32 {
            return Some(MaterializeResult { value, ops: vec![] });
        }
        // tribute_rt.nat → core.i32 (same representation)
        if from_ty == tribute_rt_nat && to_ty == core_i32 {
            return Some(MaterializeResult { value, ops: vec![] });
        }
        // tribute_rt.bool → core.i32 (same representation)
        if from_ty == tribute_rt_bool && to_ty == core_i32 {
            return Some(MaterializeResult { value, ops: vec![] });
        }
        // tribute_rt.float → core.f64 (same representation)
        if from_ty == tribute_rt_float && to_ty == core_f64 {
            return Some(MaterializeResult { value, ops: vec![] });
        }

        // -----------------------------------------------------------------
        // Boxing and unboxing: scalars ↔ tribute_rt.anyref
        // -----------------------------------------------------------------
        if let Some(op) = tribute_rt::box_or_unbox(ctx, location, value, to_ty) {
            return Some(MaterializeResult {
                value: ctx.op_result(op, 0),
                ops: vec![op],
            });
        }

        // adt.struct/adt.typeref → any: no-op (already a reference type)
        if to_ty == tribute_rt_anyref
            && (is_adt_struct_type(ctx, from_ty) || is_adt_typeref(ctx, from_ty))
        {
            return Some(MaterializeResult { value, ops: vec![] });
        }

        if from_ty == tribute_rt_anyref {
            // Keep exact canonical closure recovery until target materialization.
            // Erasing it here would pass the original anyref into an exact tail
            // slot, losing the explicit narrowing required by Wasm.
            if to_ty == canonical_closure {
                return None;
            }

            // any → adt.struct: no-op (already a reference type)
            if is_adt_struct_type(ctx, to_ty) {
                return Some(MaterializeResult { value, ops: vec![] });
            }

            // Keep dynamic nominal recovery for the target converter. Native
            // lowering can erase it to a pointer, while Wasm needs a ref.cast.
            if is_adt_typeref(ctx, to_ty) {
                return None;
            }

            // Note: any → adt.array conversions
            // are handled by wasm_type_converter, not here, because they require
            // wasm.ref_cast operations that are only available after WASM lowering.
        }

        None
    });

    tc
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::OperationDataBuilder;
    use trunk_ir::analysis::AnalysisCache;
    use trunk_ir::conversion::{ReconcileUnrealizedCasts, UnrealizedCastConversionPattern};
    use trunk_ir::dialect::{func, wasm};
    use trunk_ir::location::Span;
    use trunk_ir::ops::DialectOp;
    use trunk_ir::pass::Pass;
    use trunk_ir::refs::ValueRef;
    use trunk_ir::types::{Attribute, Location};
    use trunk_ir::walk::{WalkAction, walk_op};

    #[test]
    fn canonical_closure_recovery_reaches_exact_wasm_tail() {
        for recover in [true, false] {
            let mut ctx = IrContext::new();
            let cast = if recover {
                "%closure = core.unrealized_conversion_cast %erased : !Closure"
            } else {
                ""
            };
            let arg = if recover { "%closure" } else { "%erased" };
            let module = trunk_ir::parser::parse_test_module(
                &mut ctx,
                &format!(
                    r#"core.module @test {{
                !Closure = adt.struct<_closure(entries: tribute_rt.convention_table_ref), {{layout = "closure"}}>
                func.func @transfer(%index: core.i32, %erased: tribute_rt.anyref) {{
                    {cast}
                    func.tail_call_indirect %index, {arg} {{signature = func.func_sig<(!Closure) -> ()>, tribute.calling_convention = "cps"}}
                }}
            }}"#
                ),
            );
            let closure = crate::closure_lower::closure_struct_type_ref(&mut ctx);
            assert_eq!(ctx.type_alias_by_text("Closure"), Some(closure));

            let lowered =
                crate::wasm::lower::lower_to_wasm(&mut ctx, module, &mut Default::default());
            if !recover && cfg!(debug_assertions) {
                // The erased argument breaks the exact closure slot of the tail
                // call's signature, so the debug verifier rejects the input
                // before any lowering step runs.
                let error = lowered.expect_err("erased argument must not fill the closure slot");
                assert!(
                    matches!(error.kind(), trunk_ir::pass::PassErrorKind::InvalidInput(_)),
                    "{error}"
                );
                assert!(
                    error.to_string().contains("func.tail_call_indirect"),
                    "{error}"
                );
                continue;
            }
            lowered.unwrap();
            let tc = crate::wasm::type_converter::wasm_type_converter(&mut ctx);
            trunk_ir::rewrite::PatternApplicator::new(tc)
                .add_pattern(UnrealizedCastConversionPattern)
                .apply_partial(&mut ctx, module);
            let core_module = trunk_ir::dialect::core::Module::from_op(&ctx, module.op()).unwrap();
            ReconcileUnrealizedCasts
                .run(&mut ctx, core_module, &mut AnalysisCache::new())
                .unwrap();
            crate::wasm::lower::finalize_wasm_gc_types(&mut ctx, module).unwrap();
            let mut tails = Vec::new();
            let _: std::ops::ControlFlow<()> = walk_op(&ctx, module.op(), &mut |op| {
                if func::TailCallIndirect::matches(&ctx, op)
                    || wasm::ReturnCallIndirect::matches(&ctx, op)
                {
                    tails.push(op);
                }
                std::ops::ControlFlow::Continue(WalkAction::Advance)
            });
            assert_eq!(tails.len(), 1);
            let tail = tails[0];
            if recover {
                assert!(wasm::ReturnCallIndirect::matches(&ctx, tail));
                let arg = ctx.op_operands(tail)[1];
                let trunk_ir::refs::ValueDef::OpResult(producer, _) = ctx.value_def(arg) else {
                    panic!("cast result expected")
                };
                assert!(wasm::RefCast::matches(&ctx, producer));
                let target = crate::wasm::type_converter::closure_adt_type(&mut ctx);
                assert_eq!(ctx.value_ty(arg), target);
                let signature = ctx.op_attrs(tail).get_type("signature").unwrap();
                assert_eq!(ctx.get_type(signature).params[0], target);
            } else {
                assert!(
                    func::TailCallIndirect::matches(&ctx, tail),
                    "erased argument must not satisfy the exact closure slot"
                );
            }
        }
    }

    #[test]
    fn generic_materializer_boxes_scalars_and_forwards_references() {
        let mut ctx = IrContext::new();
        let module = trunk_ir::parser::parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !S = adt.struct<S(x: core.i32)>
  func.func @f(%x: core.i32, %s: !S, %any: tribute_rt.anyref) { func.return }
}"#,
        );
        let function = func::Func::from_op(&ctx, module.ops(&ctx)[0]).unwrap();
        let block = ctx.region(function.body(&ctx)).blocks[0];
        let [x, s, any]: [ValueRef; 3] = ctx.block_args(block).try_into().unwrap();
        let (i32_ty, struct_ty, anyref) = (ctx.value_ty(x), ctx.value_ty(s), ctx.value_ty(any));
        let location: Location = ctx.op(function.op_ref()).location;
        let tc = generic_type_converter(&mut ctx);

        let boxed = tc
            .materialize(&mut ctx, location, x, i32_ty, anyref)
            .expect("an i32 boxes into anyref");
        assert!(tribute_rt::BoxInt::matches(&ctx, boxed.ops[0]));
        assert_eq!(boxed.value, ctx.op_result(boxed.ops[0], 0));

        let erased = tc
            .materialize(&mut ctx, location, s, struct_ty, anyref)
            .expect("a struct reference erases to anyref");
        assert!(erased.ops.is_empty());
        assert_eq!(erased.value, s);
    }

    #[test]
    fn recovery_exception_uses_complete_canonical_storage_identity() {
        let mut ctx = IrContext::new();
        let module = trunk_ir::parser::parse_test_module(
            &mut ctx,
            "core.module @test { func.func @f(%value: tribute_rt.anyref) { func.return } }",
        );
        let function = func::Func::from_op(&ctx, module.ops(&ctx)[0]).unwrap();
        let block = ctx.region(function.body(&ctx)).blocks[0];
        let value = ctx.block_args(block)[0];
        let anyref = ctx.value_ty(value);
        let canonical = crate::closure_lower::closure_struct_type_ref(&mut ctx);
        let mut near = ctx.get_type(canonical).clone();
        near.attrs
            .view_mut(&mut ctx)
            .insert("unrelated", Attribute::Bool(true));
        let near = ctx.intern_type(near);
        let tc = generic_type_converter(&mut ctx);
        let location: Location = ctx.op(function.op_ref()).location;
        assert!(
            tc.materialize(&mut ctx, location, value, anyref, canonical)
                .is_none()
        );
        let (native, native_types) = crate::native::type_converter::native_type_converter(&mut ctx);
        // Native represents both sides as a pointer, so the recovery becomes an
        // identity cast for reconciliation instead of a forwarded value.
        let target = native.convert_type(&ctx, canonical).unwrap();
        assert_eq!(target, native_types.core_ptr);
        assert_eq!(native.convert_type(&ctx, anyref), Some(target));

        // Preserve the existing general struct policy, including same-named
        // types that are not the compiler's exact canonical storage.
        let result = tc
            .materialize(&mut ctx, location, value, anyref, near)
            .unwrap();
        assert_eq!(result.value, value);
        assert!(result.ops.is_empty());
    }

    fn test_location(ctx: &mut IrContext) -> Location {
        let path = ctx.intern_path("test.trb");
        Location::new(path, Span::new(0, 0))
    }

    fn anyref_type(ctx: &mut IrContext) -> TypeRef {
        tribute_rt::Anyref::params().intern(ctx).as_type_ref()
    }

    fn nominal_reference_type(ctx: &mut IrContext) -> TypeRef {
        tribute_ir::dialect::adt::Typeref::params()
            .name("String")
            .intern(ctx)
            .as_type_ref()
    }

    fn struct_type(ctx: &mut IrContext) -> TypeRef {
        tribute_ir::dialect::adt::Struct::params([])
            .name("Payload")
            .intern(ctx)
            .expect("a struct without fields is valid")
            .as_type_ref()
    }

    fn dynamic_value(ctx: &mut IrContext, location: Location) -> ValueRef {
        let anyref_ty = anyref_type(ctx);
        let source_data = OperationDataBuilder::new(location, "test.source")
            .result(anyref_ty)
            .build(ctx);
        let source = ctx.create_op(source_data);
        ctx.op_result(source, 0)
    }

    #[test]
    fn generic_cleanup_preserves_dynamic_nominal_recovery_for_target_materialization() {
        let mut ctx = IrContext::new();
        let location = test_location(&mut ctx);
        let anyref_ty = anyref_type(&mut ctx);
        let nominal_ty = nominal_reference_type(&mut ctx);
        let struct_ty = struct_type(&mut ctx);
        let value = dynamic_value(&mut ctx, location);
        let generic_converter = generic_type_converter(&mut ctx);
        assert!(
            generic_converter
                .materialize(&mut ctx, location, value, anyref_ty, nominal_ty)
                .is_none()
        );
        let struct_result = generic_converter
            .materialize(&mut ctx, location, value, anyref_ty, struct_ty)
            .expect("dynamic struct recovery stays a no-op");
        assert!(struct_result.ops.is_empty());

        let (native_converter, native_refs) =
            crate::native::type_converter::native_type_converter(&mut ctx);
        // Native represents both sides as a pointer, so the recovery becomes an
        // identity cast for reconciliation instead of a forwarded value.
        let native_target = native_converter
            .convert_type(&ctx, nominal_ty)
            .expect("native nominal representation");
        assert_eq!(native_target, native_refs.core_ptr);
        assert_eq!(
            native_converter.convert_type(&ctx, anyref_ty),
            Some(native_target)
        );

        let wasm_converter = crate::wasm::type_converter::wasm_type_converter(&mut ctx);
        let wasm_target = wasm_converter
            .convert_type(&ctx, nominal_ty)
            .expect("Wasm nominal representation");
        let wasm_result = wasm_converter
            .materialize(&mut ctx, location, value, anyref_ty, wasm_target)
            .expect("Wasm target consumes nominal recovery");
        assert!(ctx.op(wasm_result.ops[0]).name.is(&ctx, "wasm_gc.ref_cast"));
    }
}
