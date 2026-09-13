//! Semantic effect unions survive declaration collection and generalization.
use salsa_test_macros::salsa_test;
use tribute_core::diagnostic::Diagnostic;
use tribute_front::{SourceCst, ast::TypeKind, typeck::TypeCheckOutput};

#[salsa::tracked]
fn checked(db: &dyn salsa::Database, source: SourceCst) -> TypeCheckOutput<'_> {
    let parsed = tribute_front::query::parsed_ast(db, source).unwrap();
    let spans = parsed.span_map(db).clone();
    let resolved =
        tribute_front::resolve::resolve_module(db, parsed.module(db).clone(), spans.clone());
    tribute_front::typeck::typecheck_module(db, resolved, spans)
}

#[salsa_test]
fn named_rows_remain_distinct_and_union_survives_scheme(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "union.trb",
        r#"
fn combine(left: fn() ->{e1} Nil, right: fn() ->{e2} Nil) ->{e1, e2} Nil {
    left()
    right()
}
"#,
    );
    let output = checked(db, source);
    let errors = checked::accumulated::<Diagnostic>(db, source);
    assert!(errors.is_empty(), "{errors:?}");
    let scheme = output.function_types(db)[0].1;
    let TypeKind::Func { params, effect, .. } = scheme.body(db).kind(db) else {
        panic!()
    };
    let tail = |ty: tribute_front::ast::Type<'_>| match ty.kind(db) {
        TypeKind::Func { effect, .. } => effect.rest(db).unwrap(),
        _ => panic!(),
    };
    assert_ne!(tail(params[0]), tail(params[1]));
    assert!(!scheme.row_unions(db).is_empty());
    assert!(
        scheme
            .row_unions(db)
            .iter()
            .any(|union| union.result.rest(db) == effect.rest(db))
    );
}

#[salsa_test]
fn relay_preserves_writer_argument_through_nested_lambda(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "relay.trb",
        r#"
extern "C" fn print(value: Nat) -> Nil
ability Writer(w) { op tell(value: w) -> Nil }
fn use_writer() ->{Writer(Nat)} Nat { Writer::tell(5)
5 }
fn run_writer(comp: fn() ->{e, Writer(w)} a) ->{e} a {
 handle comp() { do result { result }
op Writer::tell(v) { run_writer(fn() { resume Nil }) } }
}
fn relay(comp: fn() ->{e} a) ->{e} a { comp() }
fn main() { print(run_writer(fn() { relay(use_writer) })) }
"#,
    );
    let output = checked(db, source);
    let errors = checked::accumulated::<Diagnostic>(db, source);
    assert!(errors.is_empty(), "{errors:?}");
    let main = output
        .function_types(db)
        .iter()
        .find(|(name, _)| *name == trunk_ir::Symbol::new("main"))
        .unwrap()
        .1;
    assert!(
        main.type_params(db).is_empty(),
        "main must be concrete: {main:?}"
    );
    let references = &output.expression_types(db).function_instances;
    assert!(
        references
            .iter()
            .any(|(_, instance)| instance.function.qualified(db)
                == trunk_ir::Symbol::new("run_writer")
                && instance
                    .type_arguments
                    .iter()
                    .all(|ty| matches!(ty.kind(db), TypeKind::Nat))
                && instance.type_arguments.len() == 2),
        "{references:?}"
    );
}
