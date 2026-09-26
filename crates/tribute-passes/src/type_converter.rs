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
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::refs::TypeRef;
use trunk_ir::rewrite::type_converter::{MaterializeResult, TypeConverter};
use trunk_ir::types::TypeDataBuilder;

fn intern_type(ctx: &mut IrContext, dialect: Symbol, name: Symbol) -> TypeRef {
    ctx.intern_type(TypeDataBuilder::new(dialect, name).build())
}

fn is_type(ctx: &IrContext, ty: TypeRef, dialect: Symbol, name: Symbol) -> bool {
    ctx.types().is_dialect(ty, dialect, name)
}

fn is_adt_struct_type(ctx: &IrContext, ty: TypeRef) -> bool {
    is_type(ctx, ty, Symbol::new("adt"), Symbol::new("struct"))
}

fn is_adt_typeref(ctx: &IrContext, ty: TypeRef) -> bool {
    is_type(ctx, ty, Symbol::new("adt"), Symbol::new("typeref"))
}

/// Create an TypeConverter configured for target-agnostic type conversions.
///
/// This converter handles the IR-level type transformations that are common
/// across all backends. Backend-specific converters can extend this with
/// additional conversions.
pub fn generic_type_converter(ctx: &mut IrContext) -> TypeConverter {
    // Pre-intern commonly used types (TypeRef is Copy)
    let tribute_rt_int = intern_type(ctx, Symbol::new("tribute_rt"), Symbol::new("int"));
    let tribute_rt_nat = intern_type(ctx, Symbol::new("tribute_rt"), Symbol::new("nat"));
    let tribute_rt_bool = intern_type(ctx, Symbol::new("tribute_rt"), Symbol::new("bool"));
    let tribute_rt_float = intern_type(ctx, Symbol::new("tribute_rt"), Symbol::new("float"));
    let tribute_rt_anyref = intern_type(ctx, Symbol::new("tribute_rt"), Symbol::new("anyref"));
    let core_i32 = intern_type(ctx, Symbol::new("core"), Symbol::new("i32"));
    let core_i1 = intern_type(ctx, Symbol::new("core"), Symbol::new("i1"));
    let core_f64 = intern_type(ctx, Symbol::new("core"), Symbol::new("f64"));

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
        // Boxing: primitive types → tribute_rt.anyref
        // -----------------------------------------------------------------
        if to_ty == tribute_rt_anyref {
            // Int/Nat/I32 → any: use tribute_rt.box_int
            if from_ty == tribute_rt_int || from_ty == tribute_rt_nat || from_ty == core_i32 {
                let box_op = tribute_rt::BoxInt::operands(value)
                    .results(tribute_rt_anyref)
                    .build(ctx, location);
                return Some(MaterializeResult {
                    value: box_op.result(ctx),
                    ops: vec![box_op.op_ref()],
                });
            }

            // Bool/I1 → any: use tribute_rt.box_bool
            if from_ty == tribute_rt_bool || from_ty == core_i1 {
                let box_op = tribute_rt::BoxBool::operands(value)
                    .results(tribute_rt_anyref)
                    .build(ctx, location);
                return Some(MaterializeResult {
                    value: box_op.result(ctx),
                    ops: vec![box_op.op_ref()],
                });
            }

            // Float/F64 → any: use tribute_rt.box_float
            if from_ty == tribute_rt_float || from_ty == core_f64 {
                let box_op = tribute_rt::BoxFloat::operands(value)
                    .results(tribute_rt_anyref)
                    .build(ctx, location);
                return Some(MaterializeResult {
                    value: box_op.result(ctx),
                    ops: vec![box_op.op_ref()],
                });
            }

            // adt.struct/adt.typeref → any: no-op (already a reference type)
            if is_adt_struct_type(ctx, from_ty) || is_adt_typeref(ctx, from_ty) {
                return Some(MaterializeResult { value, ops: vec![] });
            }
        }

        // -----------------------------------------------------------------
        // Unboxing: tribute_rt.anyref → primitive types
        // -----------------------------------------------------------------
        if from_ty == tribute_rt_anyref {
            // any → Int/I32: use tribute_rt.unbox_int
            if to_ty == tribute_rt_int || to_ty == core_i32 {
                let unbox_op = tribute_rt::UnboxInt::operands(value)
                    .results(to_ty)
                    .build(ctx, location);
                return Some(MaterializeResult {
                    value: unbox_op.result(ctx),
                    ops: vec![unbox_op.op_ref()],
                });
            }

            // any → Nat: use tribute_rt.unbox_nat
            if to_ty == tribute_rt_nat {
                let unbox_op = tribute_rt::UnboxNat::operands(value)
                    .results(to_ty)
                    .build(ctx, location);
                return Some(MaterializeResult {
                    value: unbox_op.result(ctx),
                    ops: vec![unbox_op.op_ref()],
                });
            }

            // any → Bool/I1: use tribute_rt.unbox_bool
            if to_ty == tribute_rt_bool || to_ty == core_i1 {
                let unbox_op = tribute_rt::UnboxBool::operands(value)
                    .results(to_ty)
                    .build(ctx, location);
                return Some(MaterializeResult {
                    value: unbox_op.result(ctx),
                    ops: vec![unbox_op.op_ref()],
                });
            }

            // any → Float/F64: use tribute_rt.unbox_float
            if to_ty == tribute_rt_float || to_ty == core_f64 {
                let unbox_op = tribute_rt::UnboxFloat::operands(value)
                    .results(to_ty)
                    .build(ctx, location);
                return Some(MaterializeResult {
                    value: unbox_op.result(ctx),
                    ops: vec![unbox_op.op_ref()],
                });
            }

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

            // Note: any → core.array conversions
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
    use trunk_ir::conversion::{
        UnrealizedCastConversionPattern, materialize_unrealized_casts, reconcile_unrealized_casts,
    };
    use trunk_ir::dialect::{func, wasm};
    use trunk_ir::location::Span;
    use trunk_ir::ops::DialectOp;
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
                !Closure = adt.struct() {{name = @_closure, fields = [[@func_ptr, core.i32], [@env, tribute_rt.anyref]]}}
                func.func @transfer(%index: core.i32, %erased: tribute_rt.anyref) {{
                    {cast}
                    func.tail_call_indirect %index, {arg} {{signature = func.func_sig<(!Closure) -> ()>, tribute.calling_convention = 2}}
                }}
            }}"#
                ),
            );
            let closure = crate::closure_lower::closure_struct_type_ref(&mut ctx);
            assert_eq!(
                ctx.type_alias_by_name(Symbol::new("Closure")),
                Some(closure)
            );
            let before = trunk_ir::printer::print_module(&ctx, module.op());
            let tc = generic_type_converter(&mut ctx);
            materialize_unrealized_casts(&mut ctx, module, &tc);
            assert_eq!(trunk_ir::printer::print_module(&ctx, module.op()), before);

            crate::wasm::lower::lower_to_wasm(&mut ctx, module).unwrap();
            let tc = crate::wasm::type_converter::wasm_type_converter(&mut ctx);
            trunk_ir::rewrite::PatternApplicator::new(tc)
                .add_pattern(UnrealizedCastConversionPattern)
                .apply_partial(&mut ctx, module);
            reconcile_unrealized_casts(&mut ctx, module);
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
                let signature = ctx.op(tail).attributes.get_type("signature").unwrap();
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
            .insert(Symbol::new("unrelated"), Attribute::Bool(true));
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
        let path = ctx.intern_path("test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    fn anyref_type(ctx: &mut IrContext) -> TypeRef {
        intern_type(ctx, Symbol::new("tribute_rt"), Symbol::new("anyref"))
    }

    fn nominal_reference_type(ctx: &mut IrContext) -> TypeRef {
        ctx.intern_type(
            TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("typeref"))
                .attr("name", Attribute::Symbol(Symbol::new("String")))
                .build(),
        )
    }

    fn struct_type(ctx: &mut IrContext) -> TypeRef {
        ctx.intern_type(
            TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("struct"))
                .attr("name", Attribute::Symbol(Symbol::new("Payload")))
                .build(),
        )
    }

    fn dynamic_value(ctx: &mut IrContext, location: Location) -> ValueRef {
        let anyref_ty = anyref_type(ctx);
        let source_data =
            OperationDataBuilder::new(location, Symbol::new("test"), Symbol::new("source"))
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
        assert_eq!(
            (
                ctx.op(wasm_result.ops[0]).dialect,
                ctx.op(wasm_result.ops[0]).name
            ),
            (Symbol::new("wasm_gc"), Symbol::new("ref_cast"))
        );
    }
}
