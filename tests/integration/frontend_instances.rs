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

#[salsa::tracked(returns(copy))]
fn prepare_damaged(db: &dyn salsa::Database, source: SourceCst, damage: u8) -> bool {
    let typed = parse_and_lower_ast(db, source).unwrap();
    let mut metadata = typed.expression_types(db).clone();
    let mut handlers = typed.handler_operations(db).clone();
    let mut performs = typed.perform_operations(db).clone();
    let target = if damage >= 6 { "hidden" } else { "identity" };
    let index = metadata
        .function_instances
        .iter()
        .position(|(_, instance)| instance.function.qualified(db) == Symbol::new(target))
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
            6 => instance.type_arguments[0] = Type::new(db, TypeKind::Bool),
            7 => {
                let (_, handler) = handlers
                    .iter_mut()
                    .find(|(_, operation)| operation.ability.qualified(db) == Symbol::new("Writer"))
                    .unwrap();
                handler.ability_args[0] = Type::new(db, TypeKind::BoundVar { index: 999 });
            }
            8 => {
                let (_, perform) = performs
                    .iter_mut()
                    .find(|(_, operation)| operation.ability.qualified(db) == Symbol::new("Writer"))
                    .unwrap();
                perform.ability_args[0] = Type::new(db, TypeKind::BoundVar { index: 999 });
            }
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
        handlers,
        performs,
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

const EFFECT_ONLY_NOMINAL: &str = r#"
struct Packet(a) { value: a }
ability Writer(w) { op tell(value: w) -> Nil }
fn hidden() ->{Writer(w)} Nil { Nil }
fn emit() ->{Writer(Packet(Nat))} Nil { Writer::tell(Packet { value: 5 }) }
fn expects(action: fn() ->{Writer(Packet(Nat))} Nil) ->{Writer(Packet(Nat))} Nil { action() }
fn run(comp: fn() ->{e, Writer(w)} Nil) ->{e} Nil {
    handle comp() {
        do value { value }
        op Writer::tell(value) { run(fn() { resume Nil }) }
    }
}
fn main() {
    run(fn() {
        expects(hidden)
        emit()
    })
}
"#;

#[salsa_test]
fn inconsistent_effect_instances_block_lowering(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(db, "invalid_effect_instance.trb", EFFECT_ONLY_NOMINAL);
    for (damage, expected) in [
        (6, "InconsistentCallable"),
        (7, "IncompleteAbilityArgument"),
        (8, "IncompleteAbilityArgument"),
    ] {
        assert!(!prepare_damaged(db, source, damage));
        let errors = prepare_damaged::accumulated::<Diagnostic>(db, source, damage);
        assert!(
            errors
                .iter()
                .any(|error| error.inner.message.contains(expected)),
            "{errors:?}"
        );
    }
}

#[salsa_test]
fn effect_only_nominal_instance_matches_handlers_and_performs(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(db, "effect_only.trb", EFFECT_ONLY_NOMINAL);
    let typed = parse_and_lower_ast(db, source).unwrap();
    let errors = parse_and_lower_ast::accumulated::<Diagnostic>(db, source);
    assert!(errors.is_empty(), "{errors:?}");
    let prepared = prepare_frontend_for_lowering(db, typed, source).unwrap();
    let instance = prepared
        .expression_types(db)
        .function_instances
        .iter()
        .find(|(_, instance)| instance.function.qualified(db) == Symbol::new("hidden"))
        .unwrap()
        .1
        .clone();
    assert_eq!(instance.type_arguments.len(), 1);
    let argument = instance.type_arguments[0];
    assert!(matches!(argument.kind(db), TypeKind::Named { args, .. } if args.is_empty()));
    let handlers: Vec<_> = prepared
        .handler_operations(db)
        .iter()
        .filter(|(_, operation)| {
            operation.ability.qualified(db) == Symbol::new("Writer")
                && operation.ability_args == vec![argument]
        })
        .collect();
    let performs: Vec<_> = prepared
        .perform_operations(db)
        .iter()
        .filter(|(_, operation)| {
            operation.ability.qualified(db) == Symbol::new("Writer")
                && operation.ability_args == vec![argument]
        })
        .collect();
    assert!(!handlers.is_empty());
    assert!(!performs.is_empty());
    assert!(handlers.iter().all(|(_, handler)| {
        performs
            .iter()
            .all(|(_, perform)| handler.ability == perform.ability)
    }));
}
