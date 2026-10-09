//! Tests for operations declared with the typed `#[dialect]` syntax.

use super::*;
use crate::Symbol;
use crate::SymbolPath;
use crate::attr_kind::Dict;
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

        /// `Type` is an ordinary parameter and projection name.
        struct Holder<Type>;

        fn held<H: Holder>(holder: Value<H>) -> Value<H::Type> {}

        /// `TypeOf` binds a variable without a bound.
        fn typed_marker<T>(r#type: Attr<TypeOf<T>>) -> Value<T> {}

        fn add<T: IntegerLike>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {}

        fn cmp<T: IntegerLike>(
            predicate: Attr<String>,
            lhs: Value<T>,
            rhs: Value<T>,
        ) -> Value<impl BoolLike> {
        }

        fn first<P: Pair>(pair: Value<P>) -> Value<P::First> {}

        fn call<S: func::FuncSig>(
            sig: Attr<TypeOf<S>>,
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

        fn select(cond: Value<impl BoolLike>, label: Option<Attr<String>>) -> Option<Value<_>> {
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

        fn maybe_call<S: func::FuncSig>(sig: Option<Attr<TypeOf<S>>>, args: Values<S::Inputs>) {}

        fn labeled(labels: Attr<[String]>, sizes: Option<Attr<[u32]>>) {}

        fn export(linkage: Attr<Linkage>, history: Option<Attr<[Linkage]>>) {}

        fn annotated(notes: Attr<Dict<String>>, sizes: Option<Attr<Dict<[u32]>>>) {}

        /// `sig` takes a pair and an integer and returns the pair's first
        /// component; the result is the pair's second component.
        fn unpack<S, P: Pair>(sig: Value<S>) -> Value<P::Second>
        where
            S: func::FuncSig<Inputs = (P, impl IntegerLike), Results = (P::First)>,
        {
        }

        /// A pair whose first component is a pair; the result is the inner
        /// pair's first component.
        fn nested<P: Pair<First = Q>, Q: Pair>(outer: Value<P>) -> Value<Q::First> {}

        fn same_inputs<S, T: func::FuncSig>(a: Value<S>, b: Value<T>)
        where
            S: func::FuncSig<Inputs = T::Inputs>,
        {
        }

        fn maybe_unpack<S, T>(sig: Option<Attr<TypeOf<S>>>, value: Value<T>)
        where
            S: func::FuncSig<Inputs = (T,)>,
        {
        }
    }

    /// An attribute kind defined next to its dialect.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Linkage {
        Private,
        Public,
    }

    impl crate::attr_kind::AttrKind for Linkage {
        const KIND: AttributeKind = AttributeKind::SymbolRef;
        type Out<'ctx> = Linkage;
        type In = Linkage;

        fn read<'ctx>(_: &'ctx IrContext, attr: &'ctx Attribute) -> Linkage {
            match attr {
                Attribute::SymbolRef(name) if *name == "public" => Linkage::Public,
                _ => Linkage::Private,
            }
        }

        fn write(_: &mut IrContext, value: Linkage) -> Attribute {
            Attribute::SymbolRef(
                Symbol::new(match value {
                    Linkage::Private => "private",
                    Linkage::Public => "public",
                })
                .into(),
            )
        }
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
    let path = ctx.intern_path("test.trb");
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
    assert_eq!(cmp.attributes[0].kind, AttributeKind::String);

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
        .predicate("slt")
        .results(i1_ty)
        .build(&mut ctx, loc);
    assert_eq!(cmp.predicate(&ctx), Symbol::new("slt"));
    assert!(print_op(&ctx, cmp.op_ref()).contains("predicate = \"slt\""));

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
    assert_eq!(ctx.op_region_count(declared.op_ref()), 1);
    assert!(ctx.op(declared.op_ref()).attributes.get("label").is_none());

    let then_region = empty_region(&mut ctx, loc);
    let else_region = empty_region(&mut ctx, loc);
    let labeled = test_typed::Select::operands(cond)
        .label("l")
        .results(i32_ty)
        .regions(then_region, else_region)
        .build(&mut ctx, loc);
    assert_eq!(labeled.label(&ctx), Some("l"));
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
        .predicate("slt")
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
  func.func @f(%p: test_typed.pair<core.i64, core.i1>, %callee: core.ptr, %x: core.i1) {
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

#[test]
fn list_attributes_build_read_and_verify_their_elements() {
    let mut ctx = IrContext::new();
    let loc = location(&mut ctx);
    let schema = &test_typed::Labeled::DEF.schema;
    assert_eq!(
        schema.attributes[0].kind,
        AttributeKind::List(&AttributeKind::String)
    );
    assert_eq!(schema.attributes[1].kind.to_string(), "[u32]");

    let op = test_typed::Labeled::operands()
        .labels(["left", "right"])
        .sizes([4, 8])
        .build(&mut ctx, loc);
    assert_eq!(op.labels(&ctx).collect::<Vec<_>>(), ["left", "right"]);
    let right = op.labels_ref(&ctx).nth(1).unwrap();
    assert_eq!(ctx.str(right), "right");
    assert_eq!(
        op.sizes(&ctx).map(Iterator::collect::<Vec<_>>),
        Some(vec![4, 8])
    );
    assert!(
        test_typed::Labeled::DEF
            .verify(&ctx, op.op_ref())
            .is_empty()
    );

    let unsized_op = test_typed::Labeled::operands()
        .labels(Vec::<String>::new())
        .build(&mut ctx, loc);
    assert_eq!(unsized_op.labels(&ctx).len(), 0);
    assert!(unsized_op.sizes(&ctx).is_none());

    let mixed = op.op_ref();
    let label = ctx.string_attr("label");
    ctx.op_mut(mixed)
        .attributes
        .insert("labels", Attribute::List(vec![label, Attribute::Int(1)]));
    let violations = test_typed::Labeled::DEF.verify(&ctx, mixed);
    assert_eq!(
        violations
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>(),
        ["attribute `labels` must be a [String] attribute"]
    );
}

#[test]
fn a_dialect_defines_its_own_attribute_kind() {
    let mut ctx = IrContext::new();
    let loc = location(&mut ctx);
    let schema = &test_typed::Export::DEF.schema;
    assert_eq!(schema.attributes[0].kind, AttributeKind::SymbolRef);
    assert_eq!(
        schema.attributes[1].kind,
        AttributeKind::List(&AttributeKind::SymbolRef)
    );

    let op = test_typed::Export::operands()
        .linkage(test_typed::Linkage::Public)
        .history([test_typed::Linkage::Private, test_typed::Linkage::Public])
        .build(&mut ctx, loc);
    assert_eq!(op.linkage(&ctx), test_typed::Linkage::Public);
    assert_eq!(
        op.history(&ctx).map(Iterator::collect::<Vec<_>>),
        Some(vec![
            test_typed::Linkage::Private,
            test_typed::Linkage::Public
        ])
    );
    assert_eq!(
        ctx.op(op.op_ref()).attributes.get("linkage"),
        Some(&Attribute::SymbolRef(SymbolPath::from("public")))
    );
    assert!(test_typed::Export::DEF.verify(&ctx, op.op_ref()).is_empty());
}

#[test]
fn dict_attributes_build_read_and_verify_their_values() {
    let mut ctx = IrContext::new();
    let loc = location(&mut ctx);
    let schema = &test_typed::Annotated::DEF.schema;
    assert_eq!(
        schema.attributes[0].kind,
        AttributeKind::Dict(&AttributeKind::String)
    );
    assert_eq!(schema.attributes[1].kind.to_string(), "Dict<[u32]>");

    let op = test_typed::Annotated::operands()
        .notes(vec![
            (Symbol::new("b"), "second".into()),
            (Symbol::new("a"), "first".into()),
        ])
        .sizes(vec![(Symbol::new("a"), vec![1, 2])])
        .build(&mut ctx, loc);
    let notes = op.notes(&ctx);
    assert_eq!(notes.len(), 2);
    assert_eq!(notes.get("b"), Some("second"));
    assert_eq!(notes.get("missing"), None);
    assert_eq!(
        notes.iter().collect::<Vec<_>>(),
        [(Symbol::new("a"), "first"), (Symbol::new("b"), "second")]
    );
    let sizes = op.sizes(&ctx).unwrap();
    assert_eq!(
        sizes.get("a").map(Iterator::collect::<Vec<_>>),
        Some(vec![1, 2])
    );
    assert!(
        test_typed::Annotated::DEF
            .verify(&ctx, op.op_ref())
            .is_empty()
    );

    let mixed = op.op_ref();
    let entries = [(Symbol::new("a"), Attribute::Int(1))]
        .into_iter()
        .collect();
    ctx.op_mut(mixed)
        .attributes
        .insert("notes", Attribute::Dict(entries));
    let violations = test_typed::Annotated::DEF.verify(&ctx, mixed);
    assert_eq!(
        violations
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>(),
        ["attribute `notes` must be a Dict<String> attribute"]
    );
}

#[test]
fn bound_constraints_are_recorded_in_the_schema() {
    let unpack = &test_typed::Unpack::DEF.schema;
    assert_eq!(unpack.relations.len(), 2);
    let RelationTarget::List(inputs) = unpack.relations[0].target else {
        panic!("expected an explicit list");
    };
    assert_eq!(
        unpack.relations[0].projection,
        ProjectionRef {
            var: 0,
            bound: 0,
            index: 0
        }
    );
    assert!(matches!(inputs, [TypeSpec::Var(1), TypeSpec::Anon(_)]));
    let RelationTarget::List([TypeSpec::Proj(first)]) = unpack.relations[1].target else {
        panic!("expected a projection element");
    };
    assert_eq!(
        *first,
        ProjectionRef {
            var: 1,
            bound: 0,
            index: 0
        }
    );

    let nested = &test_typed::Nested::DEF.schema;
    assert!(matches!(
        nested.relations[0].target,
        RelationTarget::One(TypeSpec::Var(1))
    ));

    let same = &test_typed::SameInputs::DEF.schema;
    assert!(matches!(
        same.relations[0].target,
        RelationTarget::Proj(ProjectionRef {
            var: 1,
            index: 0,
            ..
        })
    ));
}

#[test]
fn type_attributes_bind_variables_and_type_is_an_ordinary_name() {
    let mut ctx = IrContext::new();
    let loc = location(&mut ctx);
    let i64_ty = scalar(&mut ctx, "i64");
    let holder = test_typed::holder(&mut ctx, i64_ty).as_type_ref();
    let args = block_args(&mut ctx, loc, &[holder]);

    let held = test_typed::Held::operands(args[0]).build(&mut ctx, loc);
    assert_eq!(held.result_ty(&ctx), i64_ty);
    let marker = test_typed::TypedMarker::operands()
        .r#type(i64_ty)
        .build(&mut ctx, loc);
    assert_eq!(marker.result_ty(&ctx), i64_ty);
    assert_eq!(
        test_typed::TypedMarker::DEF.schema.attributes[0].binds,
        Some(0)
    );

    for op in [held.op_ref(), marker.op_ref()] {
        let def = crate::op_def::OpDef::of(&ctx, op).expect("typed ops are registered");
        assert_eq!(def.verify(&ctx, op), []);
    }
}

#[test]
fn builders_infer_results_through_bound_constraints() {
    let mut ctx = IrContext::new();
    let loc = location(&mut ctx);
    let i64_ty = scalar(&mut ctx, "i64");
    let i32_ty = scalar(&mut ctx, "i32");
    let i1_ty = scalar(&mut ctx, "i1");
    let pair = test_typed::pair(&mut ctx, i64_ty, i1_ty).as_type_ref();
    let sig = func::func_sig(&mut ctx, [pair, i32_ty], [i64_ty]).as_type_ref();
    let outer = test_typed::pair(&mut ctx, pair, i32_ty).as_type_ref();
    let args = block_args(&mut ctx, loc, &[sig, outer]);

    let unpack = test_typed::Unpack::operands(args[0]).build(&mut ctx, loc);
    assert_eq!(unpack.result_ty(&ctx), i1_ty);
    let nested = test_typed::Nested::operands(args[1]).build(&mut ctx, loc);
    assert_eq!(nested.result_ty(&ctx), i64_ty);

    for op in [unpack.op_ref(), nested.op_ref()] {
        let def = crate::op_def::OpDef::of(&ctx, op).expect("typed ops are registered");
        assert_eq!(def.verify(&ctx, op), []);
    }
}

#[test]
#[should_panic(expected = "test_typed.unpack: cannot infer type variable `P`")]
fn builders_reject_inputs_a_bound_constraint_does_not_apply_to() {
    let mut ctx = IrContext::new();
    let loc = location(&mut ctx);
    let i32_ty = scalar(&mut ctx, "i32");
    let sig = func::func_sig(&mut ctx, [i32_ty], [i32_ty]).as_type_ref();
    let args = block_args(&mut ctx, loc, &[sig]);
    test_typed::Unpack::operands(args[0]).build(&mut ctx, loc);
}

#[test]
fn verifier_checks_types_reached_through_bound_constraints() {
    const PAIR: &str = "test_typed.pair<core.i64, core.i1>";
    let text = verify_errors(&format!(
        r#"core.module @m {{
  func.func @f(%ok: func.func_sig<({PAIR}, core.i32) -> core.i64>, %not_pair: func.func_sig<(core.i32, core.i32) -> core.i64>, %short: func.func_sig<({PAIR}) -> core.i64>, %first: func.func_sig<({PAIR}, core.i32) -> core.i1>, %float: func.func_sig<({PAIR}, core.f64) -> core.i64>) {{
    %a = test_typed.unpack %ok : core.i1
    %b = test_typed.unpack %ok : core.i64
    %c = test_typed.unpack %not_pair : core.i1
    %d = test_typed.unpack %short : core.i1
    %e = test_typed.unpack %first : core.i1
    %g = test_typed.unpack %float : core.i1
    func.return
  }}
}}"#
    ));
    assert_eq!(text.matches("test_typed.unpack").count(), 5, "{text}");
    for expected in [
        "result #0 `result`: expected P::Second = core.i1, found core.i64",
        "element #0 of `S::Inputs`: expected P: test_typed.pair, found core.i32",
        "constraint on S::Inputs: expected a list of 2 type(s), found (test_typed.pair<core.i64, core.i1>)",
        "element #0 of `S::Results`: expected P::First = core.i64, found core.i1",
        "element #1 of `S::Inputs`: expected IntegerLike, found core.f64",
    ] {
        assert!(text.contains(expected), "missing {expected:?} in\n{text}");
    }
}

#[test]
fn verifier_requires_variables_reached_twice_to_agree() {
    let text = verify_errors(
        r#"core.module @m {
  func.func @f(%x: core.i32, %y: core.i64) {
    test_typed.maybe_unpack %x {sig = func.func_sig<(core.i32) -> core.nil>}
    test_typed.maybe_unpack %y {sig = func.func_sig<(core.i32) -> core.nil>}
    test_typed.maybe_unpack %x
    func.return
  }
}"#,
    );
    assert_eq!(text.matches("test_typed.maybe_unpack").count(), 2, "{text}");
    assert!(
        text.contains(
            "element #0 of `S::Inputs`: expected same type as operand #0 `value` (T = core.i64), found core.i32"
        ),
        "{text}"
    );
    assert!(
        text.contains("bound constraint: cannot check S::Inputs because `S` is not bound"),
        "{text}"
    );
}

#[test]
fn verifier_compares_projections_equated_by_bound_constraints() {
    let text = verify_errors(
        r#"core.module @m {
  func.func @f(%a: func.func_sig<(core.i32) -> core.nil>, %b: func.func_sig<(core.i32) -> core.i1>, %c: func.func_sig<(core.i64) -> core.nil>) {
    test_typed.same_inputs %a, %b
    test_typed.same_inputs %a, %c
    func.return
  }
}"#,
    );
    assert_eq!(text.matches("test_typed.same_inputs").count(), 1, "{text}");
    assert!(
        text.contains("constraint on S::Inputs: expected T::Inputs = (core.i64), found (core.i32)"),
        "{text}"
    );
}
