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
//! `tribute_rt.any` → `wasm.anyref`) are handled by backend-specific
//! type converters.

use tribute_ir::arena::dialect::tribute_rt as arena_tribute_rt;
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::refs::TypeRef;
use trunk_ir::rewrite::type_converter::{MaterializeResult, TypeConverter};
use trunk_ir::types::TypeDataBuilder;

fn intern_type(ctx: &mut IrContext, dialect: Symbol, name: Symbol) -> TypeRef {
    ctx.types
        .intern(TypeDataBuilder::new(dialect, name).build())
}

fn is_type(ctx: &IrContext, ty: TypeRef, dialect: Symbol, name: Symbol) -> bool {
    ctx.types.is_dialect(ty, dialect, name)
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
    let tribute_rt_any = intern_type(ctx, Symbol::new("tribute_rt"), Symbol::new("any"));
    let core_i32 = intern_type(ctx, Symbol::new("core"), Symbol::new("i32"));
    let core_i1 = intern_type(ctx, Symbol::new("core"), Symbol::new("i1"));
    let core_f64 = intern_type(ctx, Symbol::new("core"), Symbol::new("f64"));

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
        // Boxing: primitive types → tribute_rt.any
        // -----------------------------------------------------------------
        if to_ty == tribute_rt_any {
            // Int/Nat/I32 → any: use tribute_rt.box_int
            if from_ty == tribute_rt_int || from_ty == tribute_rt_nat || from_ty == core_i32 {
                let box_op = arena_tribute_rt::box_int(ctx, location, value, tribute_rt_any);
                return Some(MaterializeResult {
                    value: box_op.result(ctx),
                    ops: vec![box_op.op_ref()],
                });
            }

            // Bool/I1 → any: use tribute_rt.box_bool
            if from_ty == tribute_rt_bool || from_ty == core_i1 {
                let box_op = arena_tribute_rt::box_bool(ctx, location, value, tribute_rt_any);
                return Some(MaterializeResult {
                    value: box_op.result(ctx),
                    ops: vec![box_op.op_ref()],
                });
            }

            // Float/F64 → any: use tribute_rt.box_float
            if from_ty == tribute_rt_float || from_ty == core_f64 {
                let box_op = arena_tribute_rt::box_float(ctx, location, value, tribute_rt_any);
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
        // Unboxing: tribute_rt.any → primitive types
        // -----------------------------------------------------------------
        if from_ty == tribute_rt_any {
            // any → Int/I32: use tribute_rt.unbox_int
            if to_ty == tribute_rt_int || to_ty == core_i32 {
                let unbox_op = arena_tribute_rt::unbox_int(ctx, location, value, to_ty);
                return Some(MaterializeResult {
                    value: unbox_op.result(ctx),
                    ops: vec![unbox_op.op_ref()],
                });
            }

            // any → Nat: use tribute_rt.unbox_nat
            if to_ty == tribute_rt_nat {
                let unbox_op = arena_tribute_rt::unbox_nat(ctx, location, value, to_ty);
                return Some(MaterializeResult {
                    value: unbox_op.result(ctx),
                    ops: vec![unbox_op.op_ref()],
                });
            }

            // any → Bool/I1: use tribute_rt.unbox_bool
            if to_ty == tribute_rt_bool || to_ty == core_i1 {
                let unbox_op = arena_tribute_rt::unbox_bool(ctx, location, value, to_ty);
                return Some(MaterializeResult {
                    value: unbox_op.result(ctx),
                    ops: vec![unbox_op.op_ref()],
                });
            }

            // any → Float/F64: use tribute_rt.unbox_float
            if to_ty == tribute_rt_float || to_ty == core_f64 {
                let unbox_op = arena_tribute_rt::unbox_float(ctx, location, value, to_ty);
                return Some(MaterializeResult {
                    value: unbox_op.result(ctx),
                    ops: vec![unbox_op.op_ref()],
                });
            }

            // any → adt.struct/adt.typeref: no-op (already a reference type)
            if is_adt_struct_type(ctx, to_ty) || is_adt_typeref(ctx, to_ty) {
                return Some(MaterializeResult { value, ops: vec![] });
            }

            // Note: any → trampoline.resume_wrapper and any → core.array conversions
            // are handled by wasm_type_converter, not here, because they require
            // wasm.ref_cast operations that are only available after WASM lowering.
        }

        None
    });

    tc
}
