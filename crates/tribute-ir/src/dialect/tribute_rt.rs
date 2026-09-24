//! Tribute runtime dialect — boxing, unboxing, and reference counting.

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

inventory::submit! { trunk_ir::op_interface::PureOps::register("tribute_rt", "box_int") }
inventory::submit! { trunk_ir::op_interface::PureOps::register("tribute_rt", "unbox_int") }
inventory::submit! { trunk_ir::op_interface::PureOps::register("tribute_rt", "box_nat") }
inventory::submit! { trunk_ir::op_interface::PureOps::register("tribute_rt", "unbox_nat") }
inventory::submit! { trunk_ir::op_interface::PureOps::register("tribute_rt", "box_float") }
inventory::submit! { trunk_ir::op_interface::PureOps::register("tribute_rt", "unbox_float") }
inventory::submit! { trunk_ir::op_interface::PureOps::register("tribute_rt", "box_bool") }
inventory::submit! { trunk_ir::op_interface::PureOps::register("tribute_rt", "unbox_bool") }

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
    fn test_box_int_round_trip() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let i32_ty = make_i32_type(&mut ctx);
        let ptr_ty = make_ptr_type(&mut ctx);

        // Create a value to box
        let c = trunk_ir::dialect::arith::Const::operands()
            .value(Attribute::Int(42))
            .results(i32_ty)
            .build(&mut ctx, loc);
        let val = c.result(&ctx);

        // Create tribute_rt.box_int
        let op = super::BoxInt::operands(val)
            .results(ptr_ty)
            .build(&mut ctx, loc);

        // Verify from_op round-trip
        let op2 =
            super::BoxInt::from_op(&ctx, op.op_ref()).expect("should match tribute_rt.box_int");
        assert_eq!(op.op_ref(), op2.op_ref());

        // Verify operand
        assert_eq!(op.value(&ctx), val);

        // Verify result type
        let result = op.result(&ctx);
        assert_eq!(ctx.value_ty(result), ptr_ty);

        // Verify dialect and op name constants
        assert_eq!(super::BoxInt::DIALECT_NAME, "tribute_rt");
        assert_eq!(super::BoxInt::OP_NAME, "box_int");
    }

    #[test]
    fn test_unbox_int_round_trip() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let i32_ty = make_i32_type(&mut ctx);
        let ptr_ty = make_ptr_type(&mut ctx);

        // Create a boxed value
        let c = trunk_ir::dialect::arith::Const::operands()
            .value(Attribute::Int(0))
            .results(ptr_ty)
            .build(&mut ctx, loc);
        let boxed_val = c.result(&ctx);

        // Create tribute_rt.unbox_int
        let op = super::UnboxInt::operands(boxed_val)
            .results(i32_ty)
            .build(&mut ctx, loc);

        // Verify from_op round-trip
        let op2 =
            super::UnboxInt::from_op(&ctx, op.op_ref()).expect("should match tribute_rt.unbox_int");
        assert_eq!(op.op_ref(), op2.op_ref());

        // Verify operand
        assert_eq!(op.value(&ctx), boxed_val);

        // Verify result type
        let result = op.result(&ctx);
        assert_eq!(ctx.value_ty(result), i32_ty);

        assert_eq!(super::UnboxInt::DIALECT_NAME, "tribute_rt");
        assert_eq!(super::UnboxInt::OP_NAME, "unbox_int");
    }

    #[test]
    fn test_retain_round_trip() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let ptr_ty = make_ptr_type(&mut ctx);

        // Create a ptr value
        let c = trunk_ir::dialect::arith::Const::operands()
            .value(Attribute::Int(0))
            .results(ptr_ty)
            .build(&mut ctx, loc);
        let ptr_val = c.result(&ctx);

        // Create tribute_rt.retain
        let op = super::Retain::operands(ptr_val)
            .results(ptr_ty)
            .build(&mut ctx, loc);

        // Verify from_op round-trip
        let op2 =
            super::Retain::from_op(&ctx, op.op_ref()).expect("should match tribute_rt.retain");
        assert_eq!(op.op_ref(), op2.op_ref());

        // Verify operand
        assert_eq!(op.ptr(&ctx), ptr_val);

        // Verify result type
        let result = op.result(&ctx);
        assert_eq!(ctx.value_ty(result), ptr_ty);

        assert_eq!(super::Retain::DIALECT_NAME, "tribute_rt");
        assert_eq!(super::Retain::OP_NAME, "retain");
    }

    #[test]
    fn test_release_round_trip() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let ptr_ty = make_ptr_type(&mut ctx);

        // Create a ptr value
        let c = trunk_ir::dialect::arith::Const::operands()
            .value(Attribute::Int(0))
            .results(ptr_ty)
            .build(&mut ctx, loc);
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

        assert_eq!(super::Release::DIALECT_NAME, "tribute_rt");
        assert_eq!(super::Release::OP_NAME, "release");
    }

    #[test]
    fn test_into_raw_round_trip() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let managed_ty = ctx.intern_type(
            TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("typeref"))
                .attr("name", Attribute::Symbol(Symbol::new("Box")))
                .build(),
        );
        let ptr_ty = make_ptr_type(&mut ctx);
        let value = trunk_ir::dialect::arith::Const::operands()
            .value(Attribute::Int(0))
            .results(managed_ty)
            .build(&mut ctx, loc)
            .result(&ctx);

        let op = super::IntoRaw::operands(value)
            .results(ptr_ty)
            .build(&mut ctx, loc);
        let round_trip =
            super::IntoRaw::from_op(&ctx, op.op_ref()).expect("should match tribute_rt.into_raw");

        assert_eq!(round_trip.op_ref(), op.op_ref());
        assert_eq!(op.value(&ctx), value);
        assert_eq!(ctx.value_ty(op.result(&ctx)), ptr_ty);
        assert_eq!(super::IntoRaw::DIALECT_NAME, "tribute_rt");
        assert_eq!(super::IntoRaw::OP_NAME, "into_raw");
    }

    #[test]
    fn test_box_float_round_trip() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let f64_ty = ctx.intern_type(TypeDataBuilder::new("core", "f64").build());
        let ptr_ty = make_ptr_type(&mut ctx);

        let c = trunk_ir::dialect::arith::Const::operands()
            .value(Attribute::Int(0))
            .results(f64_ty)
            .build(&mut ctx, loc);
        let val = c.result(&ctx);

        let op = super::BoxFloat::operands(val)
            .results(ptr_ty)
            .build(&mut ctx, loc);
        let op2 =
            super::BoxFloat::from_op(&ctx, op.op_ref()).expect("should match tribute_rt.box_float");
        assert_eq!(op.op_ref(), op2.op_ref());
        assert_eq!(op.value(&ctx), val);
        assert_eq!(ctx.value_ty(op.result(&ctx)), ptr_ty);
    }

    #[test]
    fn test_box_bool_round_trip() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let bool_ty = ctx.intern_type(TypeDataBuilder::new("core", "bool").build());
        let ptr_ty = make_ptr_type(&mut ctx);

        let c = trunk_ir::dialect::arith::Const::operands()
            .value(Attribute::Int(1))
            .results(bool_ty)
            .build(&mut ctx, loc);
        let val = c.result(&ctx);

        let op = super::BoxBool::operands(val)
            .results(ptr_ty)
            .build(&mut ctx, loc);
        let op2 =
            super::BoxBool::from_op(&ctx, op.op_ref()).expect("should match tribute_rt.box_bool");
        assert_eq!(op.op_ref(), op2.op_ref());
        assert_eq!(op.value(&ctx), val);
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

    #[test]
    fn test_dialect_name_function() {
        assert_eq!(super::DIALECT_NAME(), Symbol::new("tribute_rt"));
    }
}
