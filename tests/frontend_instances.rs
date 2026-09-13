//! The tracked public specialization query rejects damaged instance evidence.
use salsa_test_macros::salsa_test;
use tribute::pipeline::{parse_and_lower_ast, prepare_frontend_for_lowering};
use tribute_core::diagnostic::Diagnostic;
use tribute_front::{
    SourceCst,
    ast::{FuncDefId, Type, TypeKind},
    typeck::TypeCheckOutput,
};
use trunk_ir::Symbol;

#[salsa::tracked]
fn prepare_damaged(db: &dyn salsa::Database, source: SourceCst, damage: u8) -> bool {
    let typed = parse_and_lower_ast(db, source).unwrap();
    let mut metadata = typed.expression_types(db).clone();
    let index = metadata
        .function_instances
        .iter()
        .position(|(_, instance)| instance.function.qualified(db) == Symbol::new("identity"))
        .unwrap();
    if damage == 0 {
        metadata.function_instances.remove(index);
    } else {
        let instance = &mut metadata.function_instances[index].1;
        match damage {
            1 => instance.function = FuncDefId::new(db, Symbol::new("wrong")),
            2 => instance.type_arguments.clear(),
            3 => instance.row_arguments.clear(),
            4 => instance.type_arguments[0] = Type::new(db, TypeKind::BoundVar { index: 0 }),
            5 => instance.callable = Type::new(db, TypeKind::Bool),
            _ => unreachable!(),
        }
    }
    let damaged = TypeCheckOutput::new(
        db,
        typed.module(db).clone(),
        typed.function_types(db).clone(),
        typed.constructor_types(db).clone(),
        metadata,
        typed.ability_conventions(db).clone(),
        typed.ability_definitions(db).clone(),
        typed.handler_operations(db).clone(),
        typed.perform_operations(db).clone(),
        typed.lambda_signatures(db).clone(),
        typed.exhaustive_cases(db).clone(),
        typed.well_known_types(db),
        typed.span_map(db).clone(),
    );
    prepare_frontend_for_lowering(db, damaged, source).is_some()
}

#[salsa_test]
fn damaged_instances_are_tracked_diagnostics(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "instance.trb",
        "fn identity(value: a) -> a { value }\nfn main() { let _ = identity(5) }",
    );
    for (damage, expected) in [
        (0, "MissingInstance"),
        (1, "WrongDeclaration"),
        (2, "TypeArgumentArity"),
        (3, "RowArgumentArity"),
        (4, "IncompleteTypeArgument"),
        (5, "InconsistentCallable"),
    ] {
        assert!(
            !prepare_damaged(db, source, damage),
            "damage {damage} must not produce a lowered frontend"
        );
        let messages: Vec<_> = prepare_damaged::accumulated::<Diagnostic>(db, source, damage)
            .iter()
            .map(|diagnostic| diagnostic.inner.message.to_string())
            .collect();
        assert!(
            messages.iter().any(|message| message.contains(expected)),
            "{messages:?}"
        );
    }
}
