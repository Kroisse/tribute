//! Evidence is checked before lowering; execution observes the exact output bytes.
mod common;

use salsa_test_macros::salsa_test;
use tribute::pipeline::{parse_and_lower_ast, prepare_frontend_for_lowering};
use tribute_core::Diagnostic;
use tribute_front::{SourceCst, ast::TypeKind};
use trunk_ir::Symbol;

const NESTED: &str = include_str!("fixtures/three_abilities_nested.trb");

#[salsa_test]
fn nested_writer_instance_is_concrete_before_erasure(db: &salsa::DatabaseImpl) {
    let text = format!("{}\n{NESTED}", common::PRINT_EXTERNS);
    let source = SourceCst::from_source_str(db, "nested_instances.trb", &text);
    let typed = parse_and_lower_ast(db, source).unwrap();
    assert!(parse_and_lower_ast::accumulated::<Diagnostic>(db, source).is_empty());
    let main = typed
        .function_types(db)
        .iter()
        .find(|(name, _)| *name == Symbol::new("main"))
        .unwrap()
        .1;
    assert!(main.type_params(db).is_empty());
    assert!(
        typed
            .expression_types(db)
            .function_instances
            .iter()
            .any(
                |(_, instance)| instance.function.qualified(db) == Symbol::new("run_writer")
                    && instance.type_arguments.len() == 2
                    && instance
                        .type_arguments
                        .iter()
                        .all(|ty| matches!(ty.kind(db), TypeKind::Nat))
            )
    );
    let prepared = prepare_frontend_for_lowering(db, typed, source).unwrap();
    let writer = prepared
        .perform_operations(db)
        .iter()
        .find(|(_, op)| op.ability.qualified(db) == Symbol::new("Writer"))
        .unwrap()
        .1
        .clone();
    assert!(matches!(writer.ability_args.as_slice(), [ty] if matches!(ty.kind(db), TypeKind::Nat)));
    assert!(
        prepared
            .handler_operations(db)
            .iter()
            .any(|(_, op)| op.ability == writer.ability && op.ability_args == writer.ability_args)
    );
}

#[test]
fn nested_handlers_emit_exact_output() {
    let source = format!("{}\n{NESTED}", common::PRINT_EXTERNS);
    let output = common::compile_and_run_native("nested_instances.trb", &source);
    assert!(output.status.success(), "{:?}", output);
    assert_eq!(output.stdout, b"5\n");
}

#[test]
fn relay_and_closed_named_callback_emit_exact_output() {
    for callback in ["use_writer", "fn() { relay(use_writer) }"] {
        let source = format!(
            r#"{}
ability Writer(w) {{ op tell(value: w) -> Nil }}
fn use_writer() ->{{Writer(Nat)}} Nat {{
    Writer::tell(5)
    5
}}
fn run_writer(comp: fn() ->{{e, Writer(w)}} a) ->{{e}} a {{
    handle comp() {{
        do value {{ value }}
        op Writer::tell(value) {{ run_writer(fn() {{ resume Nil }}) }}
    }}
}}
fn relay(comp: fn() ->{{e}} a) ->{{e}} a {{ comp() }}
fn main() {{ __tribute_print_nat(run_writer({callback})) }}
"#,
            common::PRINT_EXTERNS
        );
        let output = common::compile_and_run_native("callback_instance.trb", &source);
        assert!(output.status.success(), "{callback}: {output:?}");
        assert_eq!(output.stdout, b"5\n", "{callback}");
    }
}
