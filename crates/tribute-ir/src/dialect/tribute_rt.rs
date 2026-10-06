//! Tribute runtime dialect — boxing, unboxing, and reference counting.

// === Managed reference registrations ===
inventory::submit!(crate::dialect::tribute_rtti::ManagedRefType::new::<Intref>());
inventory::submit!(crate::dialect::tribute_rtti::ManagedRefType::new::<Anyref>());

#[trunk_ir::dialect]
mod tribute_rt {
    // Types
    struct Int;
    struct Nat;
    struct Float;
    struct Bool;
    struct Intref;
    struct Anyref;

    fn box_int(value: Value<_>) -> Value<_> {}
    fn unbox_int(value: Value<_>) -> Value<_> {}
    fn box_nat(value: Value<_>) -> Value<_> {}
    fn unbox_nat(value: Value<_>) -> Value<_> {}
    fn box_float(value: Value<_>) -> Value<_> {}
    fn unbox_float(value: Value<_>) -> Value<_> {}
    fn box_bool(value: Value<_>) -> Value<_> {}
    fn unbox_bool(value: Value<_>) -> Value<_> {}

    fn retain(ptr: Value<_>) -> Value<_> {}

    fn release(alloc_size: Attr<u64>, ptr: Value<_>) {}

    /// Consume one managed ownership unit while crossing a native raw-pointer
    /// representation boundary. This is deliberately not a pure operation.
    fn into_raw(value: Value<_>) -> Value<_> {}
}

// === RC Header Layout ===
// Re-exported from `tribute-rc`. See `tribute_rc::RcBox` for the layout definition.

pub use tribute_rc::HEADER_SIZE as RC_HEADER_SIZE;
pub use tribute_rc::REFCOUNT_OFFSET;
pub use tribute_rc::RTTI_IDX_OFFSET;

// === Pure operation registrations ===
// Boxing and unboxing operations are pure (no side effects)

inventory::submit! { trunk_ir::op_interface::PureOps::register::<BoxInt>() }
inventory::submit! { trunk_ir::op_interface::PureOps::register::<UnboxInt>() }
inventory::submit! { trunk_ir::op_interface::PureOps::register::<BoxNat>() }
inventory::submit! { trunk_ir::op_interface::PureOps::register::<UnboxNat>() }
inventory::submit! { trunk_ir::op_interface::PureOps::register::<BoxFloat>() }
inventory::submit! { trunk_ir::op_interface::PureOps::register::<UnboxFloat>() }
inventory::submit! { trunk_ir::op_interface::PureOps::register::<BoxBool>() }
inventory::submit! { trunk_ir::op_interface::PureOps::register::<UnboxBool>() }

#[cfg(test)]
mod tests {
    use trunk_ir::Span;
    use trunk_ir::Symbol;
    use trunk_ir::ops::DialectOp;
    use trunk_ir::refs::PathRef;
    use trunk_ir::types::Location;
    use trunk_ir::{Attribute, IrContext, TypeDataBuilder};

    fn dummy_location() -> Location {
        Location::new(PathRef::from_u32(0), Span::default())
    }

    fn make_i32_type(ctx: &mut IrContext) -> trunk_ir::TypeRef {
        ctx.intern_type(TypeDataBuilder::new("core", "i32").build())
    }

    fn make_ptr_type(ctx: &mut IrContext) -> trunk_ir::TypeRef {
        ctx.intern_type(TypeDataBuilder::new("core", "ptr").build())
    }

    #[test]
    fn test_single_value_ops_round_trip() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let i32_ty = make_i32_type(&mut ctx);
        let ptr_ty = make_ptr_type(&mut ctx);
        let f64_ty = ctx.intern_type(TypeDataBuilder::new("core", "f64").build());
        let bool_ty = ctx.intern_type(TypeDataBuilder::new("core", "i1").build());
        let name_attr = ctx.string_attr("Box");
        let managed_ty = ctx.intern_type(
            TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("typeref"))
                .attr("name", name_attr)
                .build(),
        );

        // Build `op` over a fresh operand of `operand_ty`, then check that
        // `from_op` matches it and that its operand accessor and result type
        // round-trip.
        macro_rules! check {
            ($op:ident, $accessor:ident, $operand_ty:expr, $result_ty:expr) => {{
                let name = stringify!($op);
                let value = trunk_ir::dialect::arith::Const::operands()
                    .value(Attribute::Int(0))
                    .results($operand_ty)
                    .build(&mut ctx, loc)
                    .result(&ctx);
                let op = super::$op::operands(value)
                    .results($result_ty)
                    .build(&mut ctx, loc);
                let round_trip = super::$op::from_op(&ctx, op.op_ref())
                    .unwrap_or_else(|_| panic!("{name}: from_op must match"));
                assert_eq!(round_trip.op_ref(), op.op_ref(), "{name}");
                assert_eq!(op.$accessor(&ctx), value, "{name}");
                assert_eq!(ctx.value_ty(op.result(&ctx)), $result_ty, "{name}");
            }};
        }

        check!(BoxInt, value, i32_ty, ptr_ty);
        check!(UnboxInt, value, ptr_ty, i32_ty);
        check!(BoxFloat, value, f64_ty, ptr_ty);
        check!(BoxBool, value, bool_ty, ptr_ty);
        check!(Retain, ptr, ptr_ty, ptr_ty);
        check!(IntoRaw, value, managed_ty, ptr_ty);
    }

    #[test]
    fn test_release_round_trip() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();

        // Create a ptr value
        let c = trunk_ir::dialect::mem::Null::operands().build(&mut ctx, loc);
        let ptr_val = c.result(&ctx);

        // Create tribute_rt.release (no result, has alloc_size attr)
        let op = super::Release::operands(ptr_val)
            .alloc_size(16u64)
            .build(&mut ctx, loc);

        // Verify from_op round-trip
        let op2 =
            super::Release::from_op(&ctx, op.op_ref()).expect("should match tribute_rt.release");
        assert_eq!(op.op_ref(), op2.op_ref());

        // Verify operand
        assert_eq!(op.ptr(&ctx), ptr_val);

        // Verify alloc_size attribute
        assert_eq!(op.alloc_size(&ctx), 16u64);
    }

    #[test]
    fn test_from_op_wrong_dialect() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let i32_ty = make_i32_type(&mut ctx);

        // Create an arith.const — should not match tribute_rt ops
        let c = trunk_ir::dialect::arith::Const::operands()
            .value(Attribute::Int(1))
            .results(i32_ty)
            .build(&mut ctx, loc);
        assert!(super::BoxInt::from_op(&ctx, c.op_ref()).is_err());
        assert!(super::UnboxInt::from_op(&ctx, c.op_ref()).is_err());
        assert!(super::Retain::from_op(&ctx, c.op_ref()).is_err());
        assert!(super::Release::from_op(&ctx, c.op_ref()).is_err());
    }
}
