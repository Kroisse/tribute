//! Arena-based dialect definitions.
//!
//! Each module mirrors the corresponding Salsa-based dialect in `crate::dialect`.

pub mod adt;
pub mod arith;
pub mod cf;
pub mod clif;
pub mod cont;
pub mod core;
pub mod func;
pub mod mem;
pub mod scf;
pub mod trampoline;
pub mod wasm;

#[cfg(test)]
mod tests {
    use crate::Span;
    use crate::Symbol;
    use crate::arena::ops::ArenaDialectOp;
    use crate::arena::refs::PathRef;
    use crate::arena::types::Location;
    use crate::arena::{
        Attribute, BlockData, IrContext, RegionData, TypeData, TypeInterner, ValueDef,
    };

    fn dummy_location() -> Location {
        Location::new(PathRef::from_u32(0), Span::default())
    }

    fn make_i32_type(types: &mut TypeInterner) -> crate::arena::TypeRef {
        types.intern(TypeData {
            dialect: Symbol::new("core"),
            name: Symbol::new("i32"),
            params: Default::default(),
            attrs: Default::default(),
        })
    }

    fn make_func_type(types: &mut TypeInterner) -> crate::arena::TypeRef {
        types.intern(TypeData {
            dialect: Symbol::new("func"),
            name: Symbol::new("fn"),
            params: Default::default(),
            attrs: Default::default(),
        })
    }

    // ================================================================
    // Basic constructor → from_op → accessor round-trip
    // ================================================================

    #[test]
    fn test_arith_const_round_trip() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let i32_ty = make_i32_type(&mut ctx.types);

        // Create i32.const with value attribute
        let op = super::arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(42));

        // Verify from_op
        let op2 =
            super::arith::Const::from_op(&ctx, op.op_ref()).expect("should match arith.const");
        assert_eq!(op.op_ref(), op2.op_ref());

        // Verify accessor
        let val = op.value(&ctx);
        assert_eq!(val, Attribute::IntBits(42));

        // Verify result
        let result = op.result(&ctx);
        assert_eq!(ctx.value_ty(result), i32_ty);
    }

    #[test]
    fn test_func_call_round_trip() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let i32_ty = make_i32_type(&mut ctx.types);

        // Create two values to use as arguments: use arith.const to produce them
        let c1 = super::arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(1));
        let c2 = super::arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(2));
        let v1 = c1.result(&ctx);
        let v2 = c2.result(&ctx);

        // Create func.call with variadic args
        let call = super::func::call(&mut ctx, loc, [v1, v2], i32_ty, Symbol::new("add"));

        // Verify from_op
        let call2 =
            super::func::Call::from_op(&ctx, call.op_ref()).expect("should match func.call");
        assert_eq!(call.op_ref(), call2.op_ref());

        // Verify callee attribute
        assert_eq!(call.callee(&ctx), Symbol::new("add"));

        // Verify variadic args
        let args = call.args(&ctx);
        assert_eq!(args.len(), 2);
        assert_eq!(args[0], v1);
        assert_eq!(args[1], v2);

        // Verify result
        let result = call.result(&ctx);
        assert_eq!(ctx.value_ty(result), i32_ty);
    }

    #[test]
    fn test_func_return_no_result() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let i32_ty = make_i32_type(&mut ctx.types);

        let c1 = super::arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(99));
        let v1 = c1.result(&ctx);

        // func.return has no result, variadic operands
        let ret = super::func::r#return(&mut ctx, loc, [v1]);

        let ret2 =
            super::func::Return::from_op(&ctx, ret.op_ref()).expect("should match func.return");
        assert_eq!(ret.op_ref(), ret2.op_ref());

        let values = ret.values(&ctx);
        assert_eq!(values.len(), 1);
        assert_eq!(values[0], v1);
    }

    // ================================================================
    // Region and successor accessors
    // ================================================================

    #[test]
    fn test_func_with_region() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let func_ty = make_func_type(&mut ctx.types);

        // Create a region for the function body
        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: Default::default(),
            parent_region: None,
        });
        let region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec::smallvec![block],
            parent_op: None,
        });

        // Constructor order: ctx, location, attrs (sym_name, r#type), regions (body)
        let f = super::func::func(&mut ctx, loc, Symbol::new("main"), func_ty, region);

        // Verify from_op
        let f2 = super::func::Func::from_op(&ctx, f.op_ref()).expect("should match func.func");
        assert_eq!(f.op_ref(), f2.op_ref());

        // Verify attrs
        assert_eq!(f.sym_name(&ctx), Symbol::new("main"));

        // Verify region accessor
        assert_eq!(f.body(&ctx), region);
    }

    #[test]
    fn test_scf_if_with_two_regions() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let i32_ty = make_i32_type(&mut ctx.types);

        let cond_op = super::arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(1));
        let cond = cond_op.result(&ctx);

        // Create then and else regions
        let then_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: Default::default(),
            parent_region: None,
        });
        let then_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec::smallvec![then_block],
            parent_op: None,
        });

        let else_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: Default::default(),
            parent_region: None,
        });
        let else_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec::smallvec![else_block],
            parent_op: None,
        });

        let if_op = super::scf::r#if(&mut ctx, loc, cond, i32_ty, then_region, else_region);

        assert_eq!(if_op.cond(&ctx), cond);
        assert_eq!(if_op.then_region(&ctx), then_region);
        assert_eq!(if_op.else_region(&ctx), else_region);

        // Verify result
        let result = if_op.result(&ctx);
        assert_eq!(ctx.value_ty(result), i32_ty);
    }

    #[test]
    fn test_clif_brif_with_successors() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let i32_ty = make_i32_type(&mut ctx.types);

        let cond_op = super::arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(1));
        let cond = cond_op.result(&ctx);

        // Create successor blocks
        let then_dest = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: Default::default(),
            parent_region: None,
        });
        let else_dest = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: Default::default(),
            parent_region: None,
        });

        let brif = super::clif::brif(&mut ctx, loc, cond, then_dest, else_dest);

        assert_eq!(brif.cond(&ctx), cond);
        assert_eq!(brif.then_dest(&ctx), then_dest);
        assert_eq!(brif.else_dest(&ctx), else_dest);
    }

    // ================================================================
    // Optional attributes
    // ================================================================

    #[test]
    fn test_wasm_table_optional_attrs() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();

        // wasm.table has required min and optional max
        let table_op = super::wasm::table(&mut ctx, loc, Symbol::new("funcref"), 10, Some(100));

        let table2 =
            super::wasm::Table::from_op(&ctx, table_op.op_ref()).expect("should match wasm.table");
        assert_eq!(table_op.op_ref(), table2.op_ref());

        assert_eq!(table_op.reftype(&ctx), Symbol::new("funcref"));
        assert_eq!(table_op.min(&ctx), 10);
        assert_eq!(table_op.max(&ctx), Some(100));
    }

    #[test]
    fn test_wasm_table_optional_attr_none() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();

        let table_op = super::wasm::table(&mut ctx, loc, Symbol::new("funcref"), 5, None);
        assert_eq!(table_op.min(&ctx), 5);
        assert_eq!(table_op.max(&ctx), None);
    }

    // ================================================================
    // from_op fails on wrong dialect
    // ================================================================

    #[test]
    fn test_from_op_wrong_dialect() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let i32_ty = make_i32_type(&mut ctx.types);

        let c = super::arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(1));

        // Try to match as func.call — should fail
        let err = super::func::Call::from_op(&ctx, c.op_ref());
        assert!(err.is_err());
    }

    // ================================================================
    // ArenaDialectOp::matches
    // ================================================================

    #[test]
    fn test_matches() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let i32_ty = make_i32_type(&mut ctx.types);

        let c = super::arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(42));

        assert!(super::arith::Const::matches(&ctx, c.op_ref()));
        assert!(!super::func::Call::matches(&ctx, c.op_ref()));
    }

    // ================================================================
    // Variadic results (wasm.call → #[rest] results)
    // ================================================================

    #[test]
    fn test_wasm_call_variadic_results() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let i32_ty = make_i32_type(&mut ctx.types);

        let call = super::wasm::call(
            &mut ctx,
            loc,
            [],               // no args
            [i32_ty, i32_ty], // two result types
            Symbol::new("multi_return"),
        );

        let results = call.results(&ctx);
        assert_eq!(results.len(), 2);

        // Each result should have the correct type
        assert_eq!(ctx.value_ty(results[0]), i32_ty);
        assert_eq!(ctx.value_ty(results[1]), i32_ty);
    }

    // ================================================================
    // Ops with mixed fixed + variadic operands
    // ================================================================

    #[test]
    fn test_wasm_call_indirect_mixed_operands() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let i32_ty = make_i32_type(&mut ctx.types);

        // Create values for the call
        let c1 = super::arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(1));
        let c2 = super::arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(2));
        let v1 = c1.result(&ctx);
        let v2 = c2.result(&ctx);

        // func.call_indirect has: callee (fixed), args (variadic)
        let call = super::func::call_indirect(
            &mut ctx,
            loc,
            v1,     // callee
            [v2],   // args
            i32_ty, // result type
        );

        assert_eq!(call.callee(&ctx), v1);
        assert_eq!(call.args(&ctx), &[v2]);
    }

    // ================================================================
    // Value definition tracking
    // ================================================================

    #[test]
    fn test_result_value_def() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let i32_ty = make_i32_type(&mut ctx.types);

        let c = super::arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(42));
        let result = c.result(&ctx);

        match ctx.value_def(result) {
            ValueDef::OpResult(op, idx) => {
                assert_eq!(op, c.op_ref());
                assert_eq!(idx, 0);
            }
            _ => panic!("expected OpResult"),
        }
    }

    // ================================================================
    // DIALECT_NAME and OP_NAME constants
    // ================================================================

    #[test]
    fn test_dialect_name_and_op_name() {
        assert_eq!(super::arith::Const::DIALECT_NAME, "arith");
        assert_eq!(super::arith::Const::OP_NAME, "const");
        assert_eq!(super::func::Call::DIALECT_NAME, "func");
        assert_eq!(super::func::Call::OP_NAME, "call");
        assert_eq!(super::func::Return::DIALECT_NAME, "func");
        assert_eq!(super::func::Return::OP_NAME, "return");
        assert_eq!(super::wasm::I32Const::DIALECT_NAME, "wasm");
        assert_eq!(super::wasm::I32Const::OP_NAME, "i32_const");
    }

    // ================================================================
    // DIALECT_NAME() function
    // ================================================================

    #[test]
    fn test_dialect_name_function() {
        assert_eq!(super::func::DIALECT_NAME(), Symbol::new("func"));
        assert_eq!(super::arith::DIALECT_NAME(), Symbol::new("arith"));
        assert_eq!(super::wasm::DIALECT_NAME(), Symbol::new("wasm"));
        assert_eq!(super::clif::DIALECT_NAME(), Symbol::new("clif"));
    }
}
