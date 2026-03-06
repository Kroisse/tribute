//! Arena-based tribute_rt dialect.

#[trunk_ir::arena_dialect]
mod tribute_rt {
    // Types
    struct Int;
    struct Nat;
    struct Float;
    struct Bool;
    struct Intref;
    struct Any;

    fn box_int(value: ()) -> result {}
    fn unbox_int(value: ()) -> result {}
    fn box_nat(value: ()) -> result {}
    fn unbox_nat(value: ()) -> result {}
    fn box_float(value: ()) -> result {}
    fn unbox_float(value: ()) -> result {}
    fn box_bool(value: ()) -> result {}
    fn unbox_bool(value: ()) -> result {}

    fn retain(ptr: ()) -> result {}

    #[attr(alloc_size: u64)]
    fn release(ptr: ()) {}
}

#[cfg(test)]
mod tests {
    use trunk_ir::Span;
    use trunk_ir::Symbol;
    use trunk_ir::arena::ops::ArenaDialectOp;
    use trunk_ir::arena::refs::PathRef;
    use trunk_ir::arena::types::Location;
    use trunk_ir::arena::{Attribute, IrContext, TypeDataBuilder, TypeInterner};

    fn dummy_location() -> Location {
        Location::new(PathRef::from_u32(0), Span::default())
    }

    fn make_i32_type(types: &mut TypeInterner) -> trunk_ir::arena::TypeRef {
        types.intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build())
    }

    fn make_ptr_type(types: &mut TypeInterner) -> trunk_ir::arena::TypeRef {
        types.intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ptr")).build())
    }

    #[test]
    fn test_box_int_round_trip() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let i32_ty = make_i32_type(&mut ctx.types);
        let ptr_ty = make_ptr_type(&mut ctx.types);

        // Create a value to box
        let c =
            trunk_ir::arena::dialect::arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(42));
        let val = c.result(&ctx);

        // Create tribute_rt.box_int
        let op = super::box_int(&mut ctx, loc, val, ptr_ty);

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
        let i32_ty = make_i32_type(&mut ctx.types);
        let ptr_ty = make_ptr_type(&mut ctx.types);

        // Create a boxed value
        let c =
            trunk_ir::arena::dialect::arith::r#const(&mut ctx, loc, ptr_ty, Attribute::IntBits(0));
        let boxed_val = c.result(&ctx);

        // Create tribute_rt.unbox_int
        let op = super::unbox_int(&mut ctx, loc, boxed_val, i32_ty);

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
        let ptr_ty = make_ptr_type(&mut ctx.types);

        // Create a ptr value
        let c =
            trunk_ir::arena::dialect::arith::r#const(&mut ctx, loc, ptr_ty, Attribute::IntBits(0));
        let ptr_val = c.result(&ctx);

        // Create tribute_rt.retain
        let op = super::retain(&mut ctx, loc, ptr_val, ptr_ty);

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
        let ptr_ty = make_ptr_type(&mut ctx.types);

        // Create a ptr value
        let c =
            trunk_ir::arena::dialect::arith::r#const(&mut ctx, loc, ptr_ty, Attribute::IntBits(0));
        let ptr_val = c.result(&ctx);

        // Create tribute_rt.release (no result, has alloc_size attr)
        let op = super::release(&mut ctx, loc, ptr_val, 16u64);

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
    fn test_box_float_round_trip() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let f64_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("f64")).build());
        let ptr_ty = make_ptr_type(&mut ctx.types);

        let c =
            trunk_ir::arena::dialect::arith::r#const(&mut ctx, loc, f64_ty, Attribute::IntBits(0));
        let val = c.result(&ctx);

        let op = super::box_float(&mut ctx, loc, val, ptr_ty);
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
        let bool_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("bool")).build());
        let ptr_ty = make_ptr_type(&mut ctx.types);

        let c =
            trunk_ir::arena::dialect::arith::r#const(&mut ctx, loc, bool_ty, Attribute::IntBits(1));
        let val = c.result(&ctx);

        let op = super::box_bool(&mut ctx, loc, val, ptr_ty);
        let op2 =
            super::BoxBool::from_op(&ctx, op.op_ref()).expect("should match tribute_rt.box_bool");
        assert_eq!(op.op_ref(), op2.op_ref());
        assert_eq!(op.value(&ctx), val);
    }

    #[test]
    fn test_from_op_wrong_dialect() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let i32_ty = make_i32_type(&mut ctx.types);

        // Create an arith.const — should not match tribute_rt ops
        let c =
            trunk_ir::arena::dialect::arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(1));
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
