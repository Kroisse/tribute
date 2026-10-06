//! Arena-based dialect definitions.
//!
//! Each module mirrors the corresponding Salsa-based dialect in `crate::dialect`.

pub mod arith;
pub mod cf;
pub mod clif;
pub mod core;
pub mod func;
pub mod mem;
pub mod scf;
pub mod wasm;
pub mod wasm_gc;

#[cfg(test)]
mod tests {
    use crate::Span;
    use crate::Symbol;
    use crate::SymbolPath;
    use crate::ops::{DialectOp, DialectType};
    use crate::refs::PathRef;
    use crate::types::Location;
    use crate::{Attribute, BlockData, IrContext, RegionData, TypeDataBuilder, ValueDef};

    fn dummy_location() -> Location {
        Location::new(PathRef::from_u32(0), Span::default())
    }

    fn make_i32_type(ctx: &mut IrContext) -> crate::TypeRef {
        ctx.intern_type(TypeDataBuilder::new("core", "i32").build())
    }

    fn make_func_type(ctx: &mut IrContext) -> crate::TypeRef {
        let nil_ty = super::core::nil(ctx).as_type_ref();
        super::func::func_sig(ctx, [], [nil_ty]).as_type_ref()
    }

    // ================================================================
    // Basic constructor → from_op → accessor round-trip
    // ================================================================

    #[test]
    fn test_arith_const_round_trip() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let i32_ty = make_i32_type(&mut ctx);

        // Create i32.const with value attribute
        let op = super::arith::Const::operands()
            .value(Attribute::Int(42))
            .results(i32_ty)
            .build(&mut ctx, loc);

        // Verify from_op
        let op2 =
            super::arith::Const::from_op(&ctx, op.op_ref()).expect("should match arith.const");
        assert_eq!(op.op_ref(), op2.op_ref());

        // Verify accessor
        let val = op.value(&ctx);
        assert_eq!(val, Attribute::Int(42));

        // Verify result
        let result = op.result(&ctx);
        assert_eq!(ctx.value_ty(result), i32_ty);
    }

    #[test]
    fn test_func_call_round_trip() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let i32_ty = make_i32_type(&mut ctx);

        // Create two values to use as arguments: use arith.const to produce them
        let c1 = super::arith::Const::operands()
            .value(Attribute::Int(1))
            .results(i32_ty)
            .build(&mut ctx, loc);
        let c2 = super::arith::Const::operands()
            .value(Attribute::Int(2))
            .results(i32_ty)
            .build(&mut ctx, loc);
        let v1 = c1.result(&ctx);
        let v2 = c2.result(&ctx);

        // Create func.call with variadic args
        let call = super::func::Call::operands([v1, v2])
            .callee(SymbolPath::from("add"))
            .results([i32_ty])
            .build(&mut ctx, loc);

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
        let i32_ty = make_i32_type(&mut ctx);

        let c1 = super::arith::Const::operands()
            .value(Attribute::Int(99))
            .results(i32_ty)
            .build(&mut ctx, loc);
        let v1 = c1.result(&ctx);

        // func.return has no result, variadic operands
        let ret = super::func::Return::operands([v1]).build(&mut ctx, loc);

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
        let func_ty = make_func_type(&mut ctx);

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
        let f = super::func::Func::operands()
            .sym_name("main")
            .r#type(func_ty)
            .regions(region)
            .build(&mut ctx, loc);

        // Verify from_op
        let f2 = super::func::Func::from_op(&ctx, f.op_ref()).expect("should match func.func");
        assert_eq!(f.op_ref(), f2.op_ref());

        // Verify attrs
        assert_eq!(f.sym_name(&ctx), "main");

        // Verify region accessor
        assert_eq!(f.body(&ctx), region);
    }

    #[test]
    fn test_scf_if_with_two_regions() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let i32_ty = make_i32_type(&mut ctx);

        let cond_op = super::arith::Const::operands()
            .value(Attribute::Int(1))
            .results(i32_ty)
            .build(&mut ctx, loc);
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

        let if_op = super::scf::If::operands(cond)
            .results(i32_ty)
            .regions(then_region, else_region)
            .build(&mut ctx, loc);

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
        let i32_ty = make_i32_type(&mut ctx);

        let cond_op = super::arith::Const::operands()
            .value(Attribute::Int(1))
            .results(i32_ty)
            .build(&mut ctx, loc);
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

        let brif = super::clif::Brif::operands(cond)
            .successors(then_dest, else_dest)
            .build(&mut ctx, loc);

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
        let table_op = super::wasm::Table::operands()
            .reftype("funcref")
            .min(10)
            .max(Some(100))
            .build(&mut ctx, loc);

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

        let table_op = super::wasm::Table::operands()
            .reftype("funcref")
            .min(5)
            .max(None)
            .build(&mut ctx, loc);
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
        let i32_ty = make_i32_type(&mut ctx);

        let c = super::arith::Const::operands()
            .value(Attribute::Int(1))
            .results(i32_ty)
            .build(&mut ctx, loc);

        // Try to match as func.call — should fail
        let err = super::func::Call::from_op(&ctx, c.op_ref());
        assert!(err.is_err());
    }

    // ================================================================
    // DialectOp::matches
    // ================================================================

    #[test]
    fn test_matches() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let i32_ty = make_i32_type(&mut ctx);

        let c = super::arith::Const::operands()
            .value(Attribute::Int(42))
            .results(i32_ty)
            .build(&mut ctx, loc);

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
        let i32_ty = make_i32_type(&mut ctx);

        let call = super::wasm::Call::operands([])
            .callee(SymbolPath::from("multi_return"))
            .results([i32_ty, i32_ty])
            .build(&mut ctx, loc);

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
        let i32_ty = make_i32_type(&mut ctx);

        // Create values for the call
        let c1 = super::arith::Const::operands()
            .value(Attribute::Int(1))
            .results(i32_ty)
            .build(&mut ctx, loc);
        let c2 = super::arith::Const::operands()
            .value(Attribute::Int(2))
            .results(i32_ty)
            .build(&mut ctx, loc);
        let v1 = c1.result(&ctx);
        let v2 = c2.result(&ctx);

        // func.call_indirect has: callee (fixed), args (variadic)
        let signature = super::func::func_sig(&mut ctx, [i32_ty], [i32_ty]).as_type_ref();
        let call = super::func::CallIndirect::operands(v1, [v2])
            .signature(signature)
            .build(&mut ctx, loc);

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
        let i32_ty = make_i32_type(&mut ctx);

        let c = super::arith::Const::operands()
            .value(Attribute::Int(42))
            .results(i32_ty)
            .build(&mut ctx, loc);
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
    // Arena dialect types
    // ================================================================

    #[test]
    fn test_nil_type_round_trip() {
        let mut ctx = IrContext::new();
        let nil = super::core::nil(&mut ctx);

        // from_type_ref round-trip
        let nil2 =
            super::core::Nil::from_type_ref(&ctx, nil.as_type_ref()).expect("should match Nil");
        assert_eq!(nil.as_type_ref(), nil2.as_type_ref());

        // matches
        assert!(super::core::Nil::matches(&ctx, nil.as_type_ref()));
    }

    #[test]
    fn test_array_type_with_param() {
        let mut ctx = IrContext::new();
        let i32_ty = make_i32_type(&mut ctx);
        let arr = super::core::array(&mut ctx, i32_ty);

        // element accessor
        assert_eq!(arr.element(&ctx), i32_ty);

        // from_type_ref
        let arr2 =
            super::core::Array::from_type_ref(&ctx, arr.as_type_ref()).expect("should match Array");
        assert_eq!(arr.as_type_ref(), arr2.as_type_ref());
        assert_eq!(arr2.element(&ctx), i32_ty);
    }

    #[test]
    fn test_ref_type_with_attr() {
        let mut ctx = IrContext::new();
        let ptr_ty = super::core::ptr(&mut ctx);
        let r = super::core::r#ref(&mut ctx, ptr_ty.as_type_ref(), true);

        assert_eq!(r.pointee(&ctx), ptr_ty.as_type_ref());
        assert!(r.nullable(&ctx));

        let r2 = super::core::Ref::from_type_ref(&ctx, r.as_type_ref()).expect("should match Ref");
        assert!(r2.nullable(&ctx));
    }

    #[test]
    fn test_type_matches_wrong_type() {
        let mut ctx = IrContext::new();
        let nil = super::core::nil(&mut ctx);
        // Nil should not match as Array
        assert!(!super::core::Array::matches(&ctx, nil.as_type_ref()));
        assert!(super::core::Array::from_type_ref(&ctx, nil.as_type_ref()).is_none());
    }

    #[test]
    fn test_type_into_type_ref() {
        let mut ctx = IrContext::new();
        let nil = super::core::nil(&mut ctx);
        let ty_ref: crate::TypeRef = nil.into();
        assert_eq!(ty_ref, nil.as_type_ref());
    }

    // ================================================================
    // Variadic type params (Tuple, Func)
    // ================================================================

    #[test]
    fn test_tuple_type_variadic() {
        let mut ctx = IrContext::new();
        let i32_ty = make_i32_type(&mut ctx);
        let func_ty = make_func_type(&mut ctx);

        let tup = super::core::tuple(&mut ctx, [i32_ty, func_ty]);

        assert!(super::core::Tuple::matches(&ctx, tup.as_type_ref()));
        let elements = tup.elements(&ctx);
        assert_eq!(elements.len(), 2);
        assert_eq!(elements[0], i32_ty);
        assert_eq!(elements[1], func_ty);
    }

    #[test]
    fn test_tuple_type_empty() {
        let mut ctx = IrContext::new();
        let tup = super::core::tuple(&mut ctx, []);
        assert_eq!(tup.elements(&ctx).len(), 0);
    }

    #[test]
    fn test_func_type_stores_inputs_and_results() {
        let mut ctx = IrContext::new();
        let i32_ty = make_i32_type(&mut ctx);
        let nil_ty = super::core::nil(&mut ctx).as_type_ref();
        let cases: [(&str, &[_], &[_]); 5] = [
            ("(i32, i32) -> i32", &[i32_ty, i32_ty], &[i32_ty]),
            ("(i32) -> i32", &[i32_ty], &[i32_ty]),
            ("() -> nil", &[], &[nil_ty]),
            ("() -> i32", &[], &[i32_ty]),
            ("() -> ()", &[], &[]),
        ];
        for (case, inputs, results) in cases {
            let f =
                super::func::func_sig(&mut ctx, inputs.iter().copied(), results.iter().copied());

            assert!(
                super::func::FuncSig::matches(&ctx, f.as_type_ref()),
                "{case}"
            );
            assert_eq!(f.inputs(&ctx), inputs, "{case}");
            assert_eq!(f.results(&ctx), results, "{case}");
            assert_eq!(f.single_result(&ctx), results.first().copied(), "{case}");
        }
    }

    /// Shared contract of the target dialects' multi-result `func_sig`
    /// types, checked once per dialect module.
    macro_rules! for_each_target_func_sig_dialect {
        ($check:ident) => {
            $check!(clif);
            $check!(wasm);
        };
    }

    #[test]
    fn target_func_sigs_own_zero_and_multiple_result_lists_and_metadata() {
        macro_rules! check {
            ($dialect:ident) => {{
                use super::$dialect::{FuncSig, func_sig, func_sig_with_attrs};
                let dialect = stringify!($dialect);
                let mut ctx = IrContext::new();
                let i32 = make_i32_type(&mut ctx);
                let i64 = ctx.intern_type(TypeDataBuilder::new("core", "i64").build());
                let kept = || {
                    let mut attrs = crate::AttributeMap::new();
                    attrs.insert(Symbol::new("kept"), Attribute::Type(i64));
                    attrs
                };
                let zero = func_sig(&mut ctx, [i32], []).as_type_ref();
                let one = func_sig(&mut ctx, [i32], [i64]).as_type_ref();
                let many = func_sig_with_attrs(&mut ctx, [i32], [i32, i64], kept()).as_type_ref();

                let zero_sig = FuncSig::from_type_ref(&ctx, zero).expect(dialect);
                assert!(zero_sig.is_resultless(&ctx), "{dialect}");
                let one_sig = FuncSig::from_type_ref(&ctx, one).expect(dialect);
                assert_eq!(one_sig.inputs(&ctx), [i32], "{dialect}");
                assert_eq!(one_sig.results(&ctx), [i64], "{dialect}");
                assert_ne!(
                    one,
                    super::func::func_sig(&mut ctx, [i32], [i64]).as_type_ref(),
                    "{dialect}"
                );
                let many_sig = FuncSig::from_type_ref(&ctx, many).expect(dialect);
                assert_eq!(many_sig.inputs(&ctx), [i32], "{dialect}");
                assert_eq!(many_sig.results(&ctx), [i32, i64], "{dialect}");
                assert_eq!(many_sig.non_reserved_attrs(&ctx).count(), 1, "{dialect}");
                assert_eq!(
                    many,
                    func_sig_with_attrs(&mut ctx, [i32], [i32, i64], kept()).as_type_ref(),
                    "{dialect}"
                );
            }};
        }
        for_each_target_func_sig_dialect!(check);
    }

    #[test]
    fn target_func_sigs_reject_malformed_delimiters_without_slicing() {
        macro_rules! check {
            ($dialect:ident) => {{
                use super::$dialect::{
                    FUNC_SIG, FuncSig, FuncSigTypeError, NUM_INPUTS_ATTR, NUM_RESULTS_ATTR,
                };
                let dialect = stringify!($dialect);
                let mut ctx = IrContext::new();
                let i32 = make_i32_type(&mut ctx);
                let sig =
                    |ctx: &mut IrContext, params: &[_], counts: &[(&'static str, Attribute)]| {
                        let mut builder = TypeDataBuilder::new(Symbol::new(dialect), FUNC_SIG());
                        for &param in params {
                            builder = builder.param(param);
                        }
                        for (name, value) in counts {
                            builder = builder.attr(*name, value.clone());
                        }
                        ctx.intern_type(builder.build())
                    };

                let malformed = sig(
                    &mut ctx,
                    &[i32],
                    &[
                        (NUM_INPUTS_ATTR, Attribute::Int(2)),
                        (NUM_RESULTS_ATTR, Attribute::Int(1)),
                    ],
                );
                assert!(
                    FuncSig::from_type_ref(&ctx, malformed).is_none(),
                    "{dialect}"
                );

                let missing = sig(&mut ctx, &[i32], &[(NUM_INPUTS_ATTR, Attribute::Int(1))]);
                assert_eq!(
                    FuncSig::validate(&ctx, missing),
                    Err(FuncSigTypeError::MissingCount(NUM_RESULTS_ATTR)),
                    "{dialect}"
                );

                let wrong_kind = sig(
                    &mut ctx,
                    &[i32],
                    &[
                        (
                            NUM_INPUTS_ATTR,
                            Attribute::SymbolRef(SymbolPath::from("one")),
                        ),
                        (NUM_RESULTS_ATTR, Attribute::Int(0)),
                    ],
                );
                assert_eq!(
                    FuncSig::validate(&ctx, wrong_kind),
                    Err(FuncSigTypeError::InvalidCount(NUM_INPUTS_ATTR)),
                    "{dialect}"
                );

                let overflow = sig(
                    &mut ctx,
                    &[],
                    &[
                        (NUM_INPUTS_ATTR, Attribute::Int(i128::from(u32::MAX))),
                        (NUM_RESULTS_ATTR, Attribute::Int(1)),
                    ],
                );
                assert_eq!(
                    FuncSig::validate(&ctx, overflow),
                    Err(FuncSigTypeError::CountOverflow),
                    "{dialect}"
                );
            }};
        }
        for_each_target_func_sig_dialect!(check);
    }

    #[test]
    fn test_func_type_result_lists_are_distinct() {
        let mut ctx = IrContext::new();
        let nil_ty = super::core::nil(&mut ctx).as_type_ref();
        let resultless = super::func::func_sig(&mut ctx, [], []);
        let unit = super::func::func_sig(&mut ctx, [], [nil_ty]);

        assert_ne!(resultless.as_type_ref(), unit.as_type_ref());
        assert!(resultless.is_resultless(&ctx));
        assert_eq!(resultless.single_result(&ctx), None);
        assert_eq!(unit.single_result(&ctx), Some(nil_ty));
    }

    #[test]
    fn test_func_type_counts_distinguish_the_same_flat_params() {
        let mut ctx = IrContext::new();
        let i32_ty = make_i32_type(&mut ctx);
        let one_input = super::func::func_sig(&mut ctx, [i32_ty], []);
        let one_result = super::func::func_sig(&mut ctx, [], [i32_ty]);

        assert_ne!(one_input.as_type_ref(), one_result.as_type_ref());
        assert_eq!(
            ctx.get_type(one_input.as_type_ref()).params.as_slice(),
            [i32_ty]
        );
        assert_eq!(
            ctx.get_type(one_result.as_type_ref()).params.as_slice(),
            [i32_ty]
        );
        assert_eq!(one_input.inputs(&ctx), [i32_ty]);
        assert!(one_input.results(&ctx).is_empty());
        assert!(one_result.inputs(&ctx).is_empty());
        assert_eq!(one_result.results(&ctx), [i32_ty]);
    }

    #[test]
    #[should_panic(expected = "func.func_sig currently supports at most one result")]
    fn test_func_type_constructor_rejects_multiple_results() {
        let mut ctx = IrContext::new();
        let i32_ty = make_i32_type(&mut ctx);
        let _ = super::func::func_sig(&mut ctx, [], [i32_ty, i32_ty]);
    }
}
