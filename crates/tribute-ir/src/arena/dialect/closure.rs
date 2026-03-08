//! Arena-based closure dialect.

#[trunk_ir::dialect]
mod closure {
    // Types
    struct Closure<FuncType>;

    #[attr(func_ref: Symbol)]
    fn new(env: ()) -> result {}

    fn func(closure: ()) -> result {}

    fn env(closure: ()) -> result {}
}

#[cfg(test)]
mod tests {
    use trunk_ir::Span;
    use trunk_ir::Symbol;
    use trunk_ir::ops::DialectOp;
    use trunk_ir::refs::PathRef;
    use trunk_ir::types::Location;
    use trunk_ir::{Attribute, IrContext, TypeDataBuilder, TypeInterner};

    fn dummy_location() -> Location {
        Location::new(PathRef::from_u32(0), Span::default())
    }

    fn make_i32_type(types: &mut TypeInterner) -> trunk_ir::TypeRef {
        types.intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build())
    }

    fn make_closure_type(types: &mut TypeInterner) -> trunk_ir::TypeRef {
        types.intern(TypeDataBuilder::new(Symbol::new("closure"), Symbol::new("closure")).build())
    }

    #[test]
    fn test_closure_new_round_trip() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let i32_ty = make_i32_type(&mut ctx.types);
        let closure_ty = make_closure_type(&mut ctx.types);

        // Create an env value via arith.const
        let env_op =
            trunk_ir::dialect::arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(0));
        let env_val = env_op.result(&ctx);

        // Create closure.new with func_ref attribute
        let op = super::new(&mut ctx, loc, env_val, closure_ty, Symbol::new("my_func"));

        // Verify from_op round-trip
        let op2 = super::New::from_op(&ctx, op.op_ref()).expect("should match closure.new");
        assert_eq!(op.op_ref(), op2.op_ref());

        // Verify func_ref attribute
        assert_eq!(op.func_ref(&ctx), Symbol::new("my_func"));

        // Verify result type
        let result = op.result(&ctx);
        assert_eq!(ctx.value_ty(result), closure_ty);

        // Verify dialect and op name constants
        assert_eq!(super::New::DIALECT_NAME, "closure");
        assert_eq!(super::New::OP_NAME, "new");
    }

    #[test]
    fn test_closure_func_round_trip() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let i32_ty = make_i32_type(&mut ctx.types);
        let closure_ty = make_closure_type(&mut ctx.types);

        // Create a closure value
        let env_op =
            trunk_ir::dialect::arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(0));
        let env_val = env_op.result(&ctx);
        let closure_op = super::new(&mut ctx, loc, env_val, closure_ty, Symbol::new("f"));
        let closure_val = closure_op.result(&ctx);

        // Create closure.func
        let func_op = super::func(&mut ctx, loc, closure_val, i32_ty);

        // Verify from_op round-trip
        let func_op2 =
            super::Func::from_op(&ctx, func_op.op_ref()).expect("should match closure.func");
        assert_eq!(func_op.op_ref(), func_op2.op_ref());

        // Verify operand
        assert_eq!(func_op.closure(&ctx), closure_val);

        // Verify result type
        let result = func_op.result(&ctx);
        assert_eq!(ctx.value_ty(result), i32_ty);

        // Verify dialect and op name constants
        assert_eq!(super::Func::DIALECT_NAME, "closure");
        assert_eq!(super::Func::OP_NAME, "func");
    }

    #[test]
    fn test_closure_env_round_trip() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let i32_ty = make_i32_type(&mut ctx.types);
        let closure_ty = make_closure_type(&mut ctx.types);

        // Create a closure value
        let env_op =
            trunk_ir::dialect::arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(0));
        let env_val = env_op.result(&ctx);
        let closure_op = super::new(&mut ctx, loc, env_val, closure_ty, Symbol::new("f"));
        let closure_val = closure_op.result(&ctx);

        // Create closure.env
        let env_result_op = super::env(&mut ctx, loc, closure_val, i32_ty);

        // Verify from_op round-trip
        let env_result_op2 =
            super::Env::from_op(&ctx, env_result_op.op_ref()).expect("should match closure.env");
        assert_eq!(env_result_op.op_ref(), env_result_op2.op_ref());

        // Verify operand
        assert_eq!(env_result_op.closure(&ctx), closure_val);

        // Verify result type
        let result = env_result_op.result(&ctx);
        assert_eq!(ctx.value_ty(result), i32_ty);

        // Verify dialect and op name constants
        assert_eq!(super::Env::DIALECT_NAME, "closure");
        assert_eq!(super::Env::OP_NAME, "env");
    }

    #[test]
    fn test_closure_from_op_wrong_dialect() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let i32_ty = make_i32_type(&mut ctx.types);

        // Create an arith.const — should not match closure ops
        let c = trunk_ir::dialect::arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(1));
        assert!(super::New::from_op(&ctx, c.op_ref()).is_err());
        assert!(super::Func::from_op(&ctx, c.op_ref()).is_err());
        assert!(super::Env::from_op(&ctx, c.op_ref()).is_err());
    }

    #[test]
    fn test_closure_matches() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let i32_ty = make_i32_type(&mut ctx.types);
        let closure_ty = make_closure_type(&mut ctx.types);

        let env_op =
            trunk_ir::dialect::arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(0));
        let env_val = env_op.result(&ctx);

        let closure_new = super::new(&mut ctx, loc, env_val, closure_ty, Symbol::new("f"));

        assert!(super::New::matches(&ctx, closure_new.op_ref()));
        assert!(!super::Func::matches(&ctx, closure_new.op_ref()));
        assert!(!super::Env::matches(&ctx, closure_new.op_ref()));
    }

    #[test]
    fn test_dialect_name_function() {
        assert_eq!(super::DIALECT_NAME(), Symbol::new("closure"));
    }
}
