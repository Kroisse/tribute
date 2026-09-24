//! Tests for operations declared with the typed `#[dialect]` syntax.

use super::*;
use crate::dialect::core::{BoolLike, IntegerLike, Ptr};
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
    let add = test_typed::Add::SCHEMA;
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

    let cmp = test_typed::Cmp::SCHEMA;
    let ValueConstraint::Each(TypeSpec::Anon(bounds)) = cmp.result_constraint else {
        panic!("expected an anonymous result bound");
    };
    assert_eq!(bounds[0].name, "BoolLike");
    assert_eq!(cmp.attributes[0].kind, AttributeKind::Symbol);

    let first = test_typed::First::SCHEMA;
    assert_eq!(first.type_vars[0].bounds[0].name, "test_typed.pair");
    assert!(matches!(
        first.result_constraint,
        ValueConstraint::Each(TypeSpec::Proj(ProjectionRef {
            var: 0,
            bound: 0,
            index: 0
        }))
    ));

    let call = test_typed::Call::SCHEMA;
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

    let qualified = test_typed::CallQualified::SCHEMA;
    assert_eq!(qualified.type_vars[0].bounds[0].name, "func.func_sig");
    assert!(matches!(
        qualified.operands[1].constraint,
        ValueConstraint::List(ListSpec::Proj(ProjectionRef {
            var: 0,
            bound: 0,
            index: 0
        }))
    ));

    let pack = test_typed::Pack::SCHEMA;
    let ValueConstraint::List(ListSpec::Types(types)) = pack.operands[0].constraint else {
        panic!("expected an explicit type list");
    };
    assert!(matches!(
        types,
        [TypeSpec::Var(0), TypeSpec::Var(0), TypeSpec::Anon(_)]
    ));

    let select = test_typed::Select::SCHEMA;
    assert!(matches!(select.results, ResultSchema::Optional("result")));
    assert!(select.attributes[0].optional);
    assert!(select.regions[1].optional);
}

#[test]
fn legacy_schemas_are_unconstrained() {
    let schema = crate::dialect::arith::Addi::SCHEMA;
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

    let add = test_typed::Add::operands(a, b)
        .results(i32_ty)
        .build(loc, &mut ctx);
    assert_eq!(add.lhs(&ctx), a);
    assert_eq!(add.result_ty(&ctx), i32_ty);

    let cmp = test_typed::Cmp::operands(a, b)
        .predicate(Symbol::new("slt"))
        .results(i1_ty)
        .build(loc, &mut ctx);
    assert_eq!(cmp.predicate(&ctx), Symbol::new("slt"));
    assert!(print_op(&ctx, cmp.op_ref()).contains("predicate = @slt"));

    let call = test_typed::Call::operands(callee, [a])
        .sig(sig)
        .results([i1_ty])
        .build(loc, &mut ctx);
    assert_eq!(call.sig(&ctx), sig);
    assert_eq!(call.args(&ctx), [a]);
    assert_eq!(call.results(&ctx).len(), 1);

    let ret = test_typed::Ret::operands([a, b]).build(loc, &mut ctx);
    assert_eq!(ret.values(&ctx), [a, b]);

    let then_region = empty_region(&mut ctx, loc);
    let declared = test_typed::Select::operands(cond)
        .results(None)
        .regions(then_region, None)
        .build(loc, &mut ctx);
    assert!(ctx.op_result_types(declared.op_ref()).is_empty());
    assert_eq!(ctx.op(declared.op_ref()).regions.len(), 1);
    assert!(ctx.op(declared.op_ref()).attributes.get("label").is_none());

    let then_region = empty_region(&mut ctx, loc);
    let else_region = empty_region(&mut ctx, loc);
    let labeled = test_typed::Select::operands(cond)
        .label(Symbol::new("l"))
        .results(i32_ty)
        .regions(then_region, else_region)
        .build(loc, &mut ctx);
    assert_eq!(labeled.label(&ctx), Some(Symbol::new("l")));
    assert_eq!(labeled.result_ty(&ctx), i32_ty);

    let marker = test_typed::Marker::builder()
        .results(i1_ty)
        .build(loc, &mut ctx);

    let dest = ctx.create_block(BlockData {
        location: loc,
        args: Vec::new(),
        ops: Default::default(),
        parent_region: None,
    });
    let jump = test_typed::Jump::operands([a])
        .successors(dest)
        .build(loc, &mut ctx);
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
        let schema = OpSchema::of(&ctx, op).expect("typed ops are registered");
        assert_eq!(schema.verify_structure(&ctx, op), []);
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
        .build(loc, &mut ctx);
}

#[test]
#[should_panic(expected = "test_typed.add: missing result types")]
fn fluent_builder_rejects_missing_results() {
    let mut ctx = IrContext::new();
    let loc = location(&mut ctx);
    let i32_ty = scalar(&mut ctx, "i32");
    let args = block_args(&mut ctx, loc, &[i32_ty, i32_ty]);
    test_typed::Add::operands(args[0], args[1]).build(loc, &mut ctx);
}
