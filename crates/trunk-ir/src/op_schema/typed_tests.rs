//! Tests for operations declared with the typed `#[dialect]` syntax.

use super::*;
use crate::Symbol;
use crate::dialect::core::{BoolLike, I32, IntegerLike, Ptr};
use crate::dialect::func;
use crate::ops::DialectOp;
use crate::printer::print_op;
use crate::type_constraint::{ProjectionKind, TypeConstraint};
use crate::{BlockArgData, BlockData, Location, RegionData, TypeDataBuilder, TypeRef, ValueRef};

mod test_typed {
    use super::*;

    #[trunk_ir::dialect]
    mod test_typed {
        struct Pair<First, Second>;

        fn add<T: IntegerLike>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {}

        fn cmp<T: IntegerLike>(
            predicate: Attr<Symbol>,
            lhs: Value<T>,
            rhs: Value<T>,
        ) -> Value<impl BoolLike> {
        }

        fn first<P: Pair>(pair: Value<P>) -> Value<P::First> {}

        fn call<S: func::FuncSig>(
            sig: Attr<S::Type>,
            callee: Value<Ptr>,
            args: Values<S::Inputs>,
        ) -> Values<S::Results> {
        }

        fn call_qualified<S>(callee: Value<S>, args: Values<<S as func::FuncSig>::Inputs>)
        where
            S: func::FuncSig,
        {
        }

        fn ret(values: Variadic<_>) {}

        fn pack<T>(elements: Values<(T, T, impl IntegerLike)>) -> Value<_> {}

        fn select(cond: Value<impl BoolLike>, label: Option<Attr<Symbol>>) -> Option<Value<_>> {
            #[region(then_region)]
            {}
            #[region(else_region?)]
            {}
        }

        fn marker() -> Value<_> {}

        fn jump(args: Variadic<_>) {
            #[successor(dest)]
            {}
        }

        fn widen<T: IntegerLike>(value: Value<T>) -> Values<(T, I32)> {}

        #[verify]
        fn nonempty(values: Variadic<_>) {}

        fn maybe_call<S: func::FuncSig>(sig: Option<Attr<S::Type>>, args: Values<S::Inputs>) {}
    }

    impl crate::ops::Verify for Nonempty {
        fn verify(self, ctx: &IrContext) -> Result<(), String> {
            if self.values(ctx).is_empty() {
                return Err("needs at least one value".into());
            }
            Ok(())
        }
    }
}

fn location(ctx: &mut IrContext) -> Location {
    let path = ctx.intern_path("test.trb".to_owned());
    Location::new(path, crate::Span::new(0, 0))
}

fn scalar(ctx: &mut IrContext, name: &'static str) -> TypeRef {
    ctx.intern_type(TypeDataBuilder::new("core", name).build())
}

fn block_args(ctx: &mut IrContext, loc: Location, tys: &[TypeRef]) -> Vec<ValueRef> {
    let block = ctx.create_block(BlockData {
        location: loc,
        args: tys
            .iter()
            .map(|&ty| BlockArgData {
                ty,
                attrs: Default::default(),
            })
            .collect(),
        ops: Default::default(),
        parent_region: None,
    });
    ctx.block_args(block).to_vec()
}

fn empty_region(ctx: &mut IrContext, loc: Location) -> crate::RegionRef {
    ctx.create_region(RegionData {
        location: loc,
        blocks: Default::default(),
        parent_op: None,
    })
}

#[test]
fn typed_schema_records_variables_and_constraints() {
    let add = &test_typed::Add::DEF.schema;
    assert_eq!(add.type_vars.len(), 1);
    assert_eq!(add.type_vars[0].name, "T");
    assert_eq!(add.type_vars[0].bounds[0].name, "IntegerLike");
    assert!(matches!(
        add.operands[1].constraint,
        ValueConstraint::Each(TypeSpec::Var(0))
    ));
    assert!(matches!(
        add.result_constraint,
        ValueConstraint::Each(TypeSpec::Var(0))
    ));

    let cmp = &test_typed::Cmp::DEF.schema;
    let ValueConstraint::Each(TypeSpec::Anon(bounds)) = cmp.result_constraint else {
        panic!("expected an anonymous result bound");
    };
    assert_eq!(bounds[0].name, "BoolLike");
    assert_eq!(cmp.attributes[0].kind, AttributeKind::Symbol);

    let first = &test_typed::First::DEF.schema;
    assert_eq!(first.type_vars[0].bounds[0].name, "test_typed.pair");
    assert!(matches!(
        first.result_constraint,
        ValueConstraint::Each(TypeSpec::Proj(ProjectionRef {
            var: 0,
            bound: 0,
            index: 0
        }))
    ));

    let call = &test_typed::Call::DEF.schema;
    assert_eq!(call.attributes[0].kind, AttributeKind::Type);
    assert_eq!(call.attributes[0].binds, Some(0));
    let ValueConstraint::Each(TypeSpec::Anon(callee)) = call.operands[0].constraint else {
        panic!("expected an exact callee bound");
    };
    assert_eq!(callee[0].name, "core.ptr");
    assert_eq!(call.operands[1].arity, Arity::Variadic);
    assert!(matches!(
        call.operands[1].constraint,
        ValueConstraint::List(ListSpec::Proj(ProjectionRef { index: 0, .. }))
    ));
    assert!(matches!(call.results, ResultSchema::Variadic("results")));
    assert!(matches!(
        call.result_constraint,
        ValueConstraint::List(ListSpec::Proj(ProjectionRef { index: 1, .. }))
    ));

    let qualified = &test_typed::CallQualified::DEF.schema;
    assert_eq!(qualified.type_vars[0].bounds[0].name, "func.func_sig");
    assert!(matches!(
        qualified.operands[1].constraint,
        ValueConstraint::List(ListSpec::Proj(ProjectionRef {
            var: 0,
            bound: 0,
            index: 0
        }))
    ));

    let pack = &test_typed::Pack::DEF.schema;
    let ValueConstraint::List(ListSpec::Types(types)) = pack.operands[0].constraint else {
        panic!("expected an explicit type list");
    };
    assert!(matches!(
        types,
        [TypeSpec::Var(0), TypeSpec::Var(0), TypeSpec::Anon(_)]
    ));

    let select = &test_typed::Select::DEF.schema;
    assert!(matches!(select.results, ResultSchema::Optional("result")));
    assert!(select.attributes[0].optional);
    assert!(select.regions[1].optional);
}

#[test]
fn wildcard_entities_are_unconstrained() {
    let schema = &crate::dialect::func::Call::DEF.schema;
    assert!(schema.type_vars.is_empty());
    assert!(matches!(
        schema.operands[0].constraint,
        ValueConstraint::Each(TypeSpec::Any)
    ));
    assert!(matches!(
        schema.result_constraint,
        ValueConstraint::Each(TypeSpec::Any)
    ));
}

#[test]
fn generated_type_constraints_check_parameters_and_project() {
    let mut ctx = IrContext::new();
    let i32_ty = scalar(&mut ctx, "i32");
    let i1_ty = scalar(&mut ctx, "i1");
    let pair = test_typed::pair(&mut ctx, i32_ty, i1_ty).as_type_ref();
    let malformed = ctx.intern_type(
        TypeDataBuilder::new("test_typed", "pair")
            .param(i32_ty)
            .build(),
    );

    let desc = <test_typed::Pair as TypeConstraint>::DESC;
    assert!(desc.exact);
    assert_eq!(desc.projections[1].name, "Second");
    assert_eq!(desc.projections[1].kind, ProjectionKind::One);
    assert!((desc.matches)(&ctx, pair));
    assert!(!(desc.matches)(&ctx, malformed));
    assert!((desc.project)(&ctx, malformed, 0).is_none());
    assert!(matches!(
        (desc.project)(&ctx, pair, 1),
        Some(crate::type_constraint::Projected::One(ty)) if ty == i1_ty
    ));

    let sig = func::func_sig(&mut ctx, [i32_ty, i1_ty], [i32_ty]).as_type_ref();
    let sig_desc = <func::FuncSig as TypeConstraint>::DESC;
    assert!(matches!(
        (sig_desc.project)(&ctx, sig, 0),
        Some(crate::type_constraint::Projected::List([a, b])) if *a == i32_ty && *b == i1_ty
    ));
    assert!(matches!(
        (sig_desc.project)(&ctx, sig, 1),
        Some(crate::type_constraint::Projected::List([r])) if *r == i32_ty
    ));
    assert!((sig_desc.project)(&ctx, sig, 2).is_none());
    assert!((sig_desc.project)(&ctx, pair, 0).is_none());
    assert!(!(sig_desc.matches)(&ctx, pair));

    // Declared type attributes are part of the exact-bound invariant.
    let ref_desc = <crate::dialect::core::Ref as TypeConstraint>::DESC;
    let ref_with = |ctx: &mut IrContext, attr: Option<crate::Attribute>| {
        let mut builder = TypeDataBuilder::new("core", "ref").param(i32_ty);
        if let Some(attr) = attr {
            builder = builder.attr("nullable", attr);
        }
        ctx.intern_type(builder.build())
    };
    let nullable = ref_with(&mut ctx, Some(crate::Attribute::Bool(true)));
    let missing = ref_with(&mut ctx, None);
    let wrong_kind = ref_with(&mut ctx, Some(crate::Attribute::Int(1)));
    assert!((ref_desc.matches)(&ctx, nullable));
    assert!(!(ref_desc.matches)(&ctx, missing));
    assert!(!(ref_desc.matches)(&ctx, wrong_kind));

    let integer = <IntegerLike as TypeConstraint>::DESC;
    assert!((integer.matches)(&ctx, i1_ty));
    assert!(!(integer.matches)(&ctx, pair));
}

#[test]
fn fluent_builders_group_inputs_by_kind() {
    let mut ctx = IrContext::new();
    let loc = location(&mut ctx);
    let i32_ty = scalar(&mut ctx, "i32");
    let i1_ty = scalar(&mut ctx, "i1");
    let ptr_ty = crate::dialect::core::ptr(&mut ctx).as_type_ref();
    let sig = func::func_sig(&mut ctx, [i32_ty], [i1_ty]).as_type_ref();
    let args = block_args(&mut ctx, loc, &[i32_ty, i32_ty, ptr_ty, i1_ty]);
    let (a, b, callee, cond) = (args[0], args[1], args[2], args[3]);

    let add = test_typed::Add::operands(a, b).build(&mut ctx, loc);
    assert_eq!(add.lhs(&ctx), a);
    assert_eq!(add.result_ty(&ctx), i32_ty);

    let cmp = test_typed::Cmp::operands(a, b)
        .predicate(Symbol::new("slt"))
        .results(i1_ty)
        .build(&mut ctx, loc);
    assert_eq!(cmp.predicate(&ctx), Symbol::new("slt"));
    assert!(print_op(&ctx, cmp.op_ref()).contains("predicate = @slt"));

    let call = test_typed::Call::operands(callee, [a])
        .sig(sig)
        .build(&mut ctx, loc);
    assert_eq!(call.sig(&ctx), sig);
    assert_eq!(call.args(&ctx), [a]);
    assert_eq!(ctx.op_result_types(call.op_ref()), [i1_ty]);

    let ret = test_typed::Ret::operands([a, b]).build(&mut ctx, loc);
    assert_eq!(ret.values(&ctx), [a, b]);

    let then_region = empty_region(&mut ctx, loc);
    let declared = test_typed::Select::operands(cond)
        .results(None)
        .regions(then_region, None)
        .build(&mut ctx, loc);
    assert!(ctx.op_result_types(declared.op_ref()).is_empty());
    assert_eq!(ctx.op(declared.op_ref()).regions.len(), 1);
    assert!(ctx.op(declared.op_ref()).attributes.get("label").is_none());

    let then_region = empty_region(&mut ctx, loc);
    let else_region = empty_region(&mut ctx, loc);
    let labeled = test_typed::Select::operands(cond)
        .label(Symbol::new("l"))
        .results(i32_ty)
        .regions(then_region, else_region)
        .build(&mut ctx, loc);
    assert_eq!(labeled.label(&ctx), Some(Symbol::new("l")));
    assert_eq!(labeled.result_ty(&ctx), i32_ty);

    let marker = test_typed::Marker::operands()
        .results(i1_ty)
        .build(&mut ctx, loc);

    let dest = ctx.create_block(BlockData {
        location: loc,
        args: Vec::new(),
        ops: Default::default(),
        parent_region: None,
    });
    let jump = test_typed::Jump::operands([a])
        .successors(dest)
        .build(&mut ctx, loc);
    assert_eq!(jump.dest(&ctx), dest);

    for op in [
        add.op_ref(),
        cmp.op_ref(),
        call.op_ref(),
        ret.op_ref(),
        declared.op_ref(),
        labeled.op_ref(),
        marker.op_ref(),
        jump.op_ref(),
    ] {
        let def = crate::op_def::OpDef::of(&ctx, op).expect("typed ops are registered");
        assert_eq!(def.verify(&ctx, op), []);
    }
}

#[test]
#[should_panic(expected = "test_typed.cmp: missing attribute `predicate`")]
fn fluent_builder_rejects_missing_required_attribute() {
    let mut ctx = IrContext::new();
    let loc = location(&mut ctx);
    let i32_ty = scalar(&mut ctx, "i32");
    let args = block_args(&mut ctx, loc, &[i32_ty, i32_ty]);
    test_typed::Cmp::operands(args[0], args[1])
        .results(i32_ty)
        .build(&mut ctx, loc);
}

#[test]
#[should_panic(expected = "test_typed.cmp: missing result types")]
fn fluent_builder_rejects_missing_results() {
    let mut ctx = IrContext::new();
    let loc = location(&mut ctx);
    let i32_ty = scalar(&mut ctx, "i32");
    let args = block_args(&mut ctx, loc, &[i32_ty, i32_ty]);
    test_typed::Cmp::operands(args[0], args[1])
        .predicate(Symbol::new("slt"))
        .build(&mut ctx, loc);
}

#[test]
fn fluent_builders_infer_bound_projected_and_fixed_results() {
    let mut ctx = IrContext::new();
    let loc = location(&mut ctx);
    let i64_ty = scalar(&mut ctx, "i64");
    let i32_ty = scalar(&mut ctx, "i32");
    let i1_ty = scalar(&mut ctx, "i1");
    let pair_ty = test_typed::pair(&mut ctx, i64_ty, i1_ty).as_type_ref();
    let args = block_args(&mut ctx, loc, &[i64_ty, pair_ty]);

    let first = test_typed::First::operands(args[1]).build(&mut ctx, loc);
    assert_eq!(first.result_ty(&ctx), i64_ty);

    let widen = test_typed::Widen::operands(args[0]).build(&mut ctx, loc);
    assert_eq!(ctx.op_result_types(widen.op_ref()), [i64_ty, i32_ty]);
    assert_eq!(I32::type_ref(&mut ctx), i32_ty);
}

#[test]
#[should_panic(expected = "test_typed.first: P = core.i64 does not provide P::First")]
fn fluent_builder_rejects_unprojectable_sources() {
    let mut ctx = IrContext::new();
    let loc = location(&mut ctx);
    let i64_ty = scalar(&mut ctx, "i64");
    let args = block_args(&mut ctx, loc, &[i64_ty]);
    test_typed::First::operands(args[0]).build(&mut ctx, loc);
}

#[test]
#[should_panic(expected = "test_typed.call: cannot infer type variable `S`")]
fn fluent_builder_rejects_missing_binding_attribute() {
    let mut ctx = IrContext::new();
    let loc = location(&mut ctx);
    let ptr_ty = crate::dialect::core::ptr(&mut ctx).as_type_ref();
    let args = block_args(&mut ctx, loc, &[ptr_ty]);
    test_typed::Call::operands(args[0], []).build(&mut ctx, loc);
}

fn verify_errors(input: &str) -> String {
    let mut ctx = IrContext::new();
    let module = crate::parser::parse_test_module(&mut ctx, input);
    crate::validation::validate_operation_verifiers(&ctx, module).to_string()
}

#[test]
fn verifier_reports_constraints_before_bindings() {
    let text = verify_errors(
        r#"core.module @m {
  func.func @f(%a: core.f64, %b: core.i32) {
    %c = test_typed.add %a, %b : core.i32
    func.return
  }
}"#,
    );
    assert!(
        text.contains("operand #0 `lhs`: expected T: IntegerLike, found core.f64"),
        "{text}"
    );
    // Binding is checked only after every individual constraint passed.
    assert!(!text.contains("same type"), "{text}");
}

#[test]
fn verifier_reports_binding_mismatches() {
    let text = verify_errors(
        r#"core.module @m {
  func.func @f(%a: core.i32, %b: core.i64) {
    %c = test_typed.add %a, %b : core.i32
    %d = test_typed.add %a, %a : core.i64
    func.return
  }
}"#,
    );
    assert!(
        text.contains(
            "operand #1 `rhs`: expected same type as operand #0 `lhs` (T = core.i32), found core.i64"
        ),
        "{text}"
    );
    assert!(
        text.contains(
            "result #0 `result`: expected same type as operand #0 `lhs` (T = core.i32), found core.i64"
        ),
        "{text}"
    );
}

#[test]
fn verifier_reports_projection_and_list_mismatches() {
    let text = verify_errors(
        r#"core.module @m {
  func.func @f(%p: test_typed.pair(core.i64, core.i1), %callee: core.ptr, %x: core.i1) {
    %a = test_typed.first %p : core.i1
    %b = test_typed.call %callee, %x {sig = func.func_sig<(core.i32) -> core.i1>} : core.i1
    %c = test_typed.call %callee, %x {sig = core.i32} : core.i1
    func.return
  }
}"#,
    );
    assert!(
        text.contains("result #0 `result`: expected P::First = core.i64, found core.i1"),
        "{text}"
    );
    assert!(
        text.contains("operands `args`: expected S::Inputs = (core.i32), found (core.i1)"),
        "{text}"
    );
    assert!(
        text.contains("attribute `sig`: expected S: func.func_sig, found core.i32"),
        "{text}"
    );
}

#[test]
fn verifier_counts_explicit_lists_and_runs_verify_hooks_last() {
    let text = verify_errors(
        r#"core.module @m {
  func.func @f(%a: core.i32, %b: core.f64) {
    test_typed.pack %a, %a
    test_typed.nonempty
    test_typed.nonempty %b
    func.return
  }
}"#,
    );
    assert_eq!(
        test_typed::Pack::DEF.schema.operand_count().to_string(),
        "3"
    );
    assert!(text.contains("test_typed.pack"), "{text}");
    assert!(text.contains("expected 3 operand(s), found 2"), "{text}");
    assert!(text.contains("needs at least one value"), "{text}");
    assert_eq!(
        text.matches("needs at least one value").count(),
        1,
        "{text}"
    );
}

#[test]
fn verifier_rejects_projections_of_unbound_variables() {
    let text = verify_errors(
        r#"core.module @m {
  func.func @f(%x: core.i32) {
    test_typed.maybe_call %x
    test_typed.maybe_call %x {sig = func.func_sig<(core.i32) -> core.nil>}
    func.return
  }
}"#,
    );
    assert!(
        text.contains("operands `args`: cannot check S::Inputs because `S` is not bound"),
        "{text}"
    );
    assert_eq!(text.matches("test_typed.maybe_call").count(), 1, "{text}");
}

#[test]
fn verifier_checks_explicit_list_elements_and_result_segments() {
    let text = verify_errors(
        r#"core.module @m {
  func.func @f(%a: core.i32, %b: core.i64, %c: core.f64, %callee: core.ptr) {
    %p = test_typed.pack %a, %a, %c : core.nil
    %q = test_typed.pack %a, %b, %a : core.nil
    %r = test_typed.call %callee, %a, %b {sig = func.func_sig<(core.i32, core.i64) -> core.i1>} : core.i32
    func.return
  }
}"#,
    );
    assert!(
        text.contains("operand #2 `elements`: expected IntegerLike, found core.f64"),
        "{text}"
    );
    assert!(
        text.contains("operand #1 `elements`: expected same type as operand #0 `elements`"),
        "{text}"
    );
    assert!(
        text.contains("results `results`: expected S::Results = (core.i1), found (core.i32)"),
        "{text}"
    );
}

#[test]
fn scalar_bounds_match_one_core_type_and_verify_wasm_add() {
    let mut ctx = IrContext::new();
    let i32_ty = I32::type_ref(&mut ctx);
    let i64_ty = crate::dialect::core::I64::type_ref(&mut ctx);
    assert!(I32::matches(&ctx, i32_ty));
    assert!(!I32::matches(&ctx, i64_ty));
    let f64_ty = scalar(&mut ctx, "f64");
    assert!(crate::dialect::core::F64::matches(&ctx, f64_ty));

    let text = verify_errors(
        r#"core.module @m {
  func.func @f(%a: core.i32, %b: core.i1) {
    %c = wasm.i32_add %a, %b : core.i32
    %d = arith.addi %a, %a : core.i32
    func.return
  }
}"#,
    );
    assert!(
        text.contains("wasm.i32_add") && text.contains("expected core.i32, found core.i1"),
        "{text}"
    );
    assert!(!text.contains("arith.addi"), "{text}");
}
