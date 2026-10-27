//! The tracked public specialization query rejects damaged instance evidence.
use salsa_test_macros::salsa_test;
use tribute::pipeline::{frontend_diagnostics, parse_and_lower_ast, prepare_frontend_for_lowering};
use tribute_core::diagnostic::Diagnostic;
use tribute_front::{
    SourceCst,
    ast::{FuncDefId, Type, TypeKind},
    typeck::TypeCheckOutput,
};

#[salsa::tracked(returns(copy))]
fn prepare_damaged(db: &dyn salsa::Database, source: SourceCst, damage: u8) -> bool {
    let typed = parse_and_lower_ast(db, source).unwrap();
    let mut metadata = typed.expression_types(db).clone();
    let target = if damage >= 6 { "hidden" } else { "identity" };
    let node = metadata
        .function_instances
        .iter()
        .find(|(_, instance)| instance.function.qualified(db) == target)
        .unwrap()
        .0;
    if damage == 0 {
        metadata.function_instances.remove(&node);
    } else {
        let instance = metadata.function_instances.get_mut(&node).unwrap();
        match damage {
            1 => instance.function = FuncDefId::new(db, trunk_ir::SymbolPath::new(["wrong"])),
            2 => instance.type_arguments.clear(),
            3 => instance.row_arguments.clear(),
            4 => instance.type_arguments[0] = Type::new(db, TypeKind::BoundVar { index: 0 }),
            5 => instance.callable = Type::new(db, TypeKind::Bool),
            6 => instance.type_arguments[0] = Type::new(db, TypeKind::Bool),
            7 => {
                let (_, handler) = metadata
                    .handler_operations
                    .iter_mut()
                    .find(|(_, operation)| operation.ability.qualified(db) == "Writer")
                    .unwrap();
                handler.ability_args[0] = Type::new(db, TypeKind::BoundVar { index: 999 });
            }
            8 => {
                let (_, perform) = metadata
                    .perform_operations
                    .iter_mut()
                    .find(|(_, operation)| operation.ability.qualified(db) == "Writer")
                    .unwrap();
                perform.ability_args[0] = Type::new(db, TypeKind::BoundVar { index: 999 });
            }
            _ => unreachable!(),
        }
    }
    let damaged = TypeCheckOutput::new(
        db,
        typed.module(db).clone(),
        typed.declarations(db).clone(),
        metadata,
    );
    prepare_frontend_for_lowering(db, damaged, source).is_some()
}

#[salsa_test]
fn damaged_instances_are_tracked_diagnostics(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "instance.trb",
        "fn identity(value: a) -> a { value }\nfn main() -> Nil { let _ = identity(5) }",
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
fn main() -> Nil {
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
    let errors = frontend_diagnostics(db, source);
    assert!(errors.is_empty(), "{errors:?}");
    let prepared = prepare_frontend_for_lowering(db, typed, source).unwrap();
    let instance = prepared
        .expression_types(db)
        .function_instances
        .iter()
        .find(|(_, instance)| instance.function.qualified(db) == "hidden")
        .unwrap()
        .1
        .clone();
    assert_eq!(instance.type_arguments.len(), 1);
    let argument = instance.type_arguments[0];
    assert!(matches!(argument.kind(db), TypeKind::Named { args, .. } if args.is_empty()));
    let handlers: Vec<_> = prepared
        .expression_types(db)
        .handler_operations
        .iter()
        .filter(|(_, operation)| {
            operation.ability.qualified(db) == "Writer" && operation.ability_args == vec![argument]
        })
        .collect();
    let performs: Vec<_> = prepared
        .expression_types(db)
        .perform_operations
        .iter()
        .filter(|(_, operation)| {
            operation.ability.qualified(db) == "Writer" && operation.ability_args == vec![argument]
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

#[salsa::tracked(returns(copy))]
fn prepares(db: &dyn salsa::Database, source: SourceCst) -> bool {
    let typed = parse_and_lower_ast(db, source).unwrap();
    prepare_frontend_for_lowering(db, typed, source).is_some()
}

/// Every monomorphic function is lowered, so an incomplete instance in one
/// is rejected whether or not `main` reaches it.
#[salsa_test]
fn incomplete_instances_are_rejected_whether_or_not_reached(db: &salsa::DatabaseImpl) {
    for main in [
        "fn main() -> Nil { }",
        "fn main() -> Nil { let _ = unused() }",
    ] {
        let source = SourceCst::from_source_str(
            db,
            "incomplete_instance.trb",
            &format!(
                "fn count(xs: List(a)) -> Nat {{ 0 }}\nfn unused() -> Nat {{ count([]) }}\n{main}"
            ),
        );
        assert!(!prepares(db, source), "{main}");
        let messages: Vec<_> = prepares::accumulated::<Diagnostic>(db, source)
            .iter()
            .map(|diagnostic| diagnostic.inner.message.to_string())
            .collect();
        assert!(
            messages
                .iter()
                .any(|message| message.contains("IncompleteTypeArgument")),
            "{main}: {messages:?}"
        );
    }
}

/// The specialization output holds a generic function's instances, never
/// the function itself, nor nominal instances only its body mentions.
#[salsa_test]
fn generic_function_templates_leave_the_specialization_output(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "templates.trb",
        r#"
struct Packet(a) { value: a }
fn identity(value: a) -> a { value }
fn unreached(value: a) -> Packet(Bool) { Packet { value: True } }
fn main() -> Nil { let _ = identity(5) }
"#,
    );
    let typed = parse_and_lower_ast(db, source).unwrap();
    let specialized = prepare_frontend_for_lowering(db, typed, source).unwrap();
    let names: Vec<String> = specialized
        .module(db)
        .decls()
        .filter_map(|decl| match decl {
            tribute_front::ast::Decl::Function(function) => Some(function.name.to_string()),
            tribute_front::ast::Decl::Struct(declaration) => Some(declaration.name.to_string()),
            _ => None,
        })
        .collect();
    for absent in ["identity", "unreached", "Packet$Bool"] {
        assert!(
            !names.iter().any(|name| name == absent),
            "{absent}: {names:?}"
        );
    }
    for present in ["identity$Nat", "main", "Packet"] {
        assert!(
            names.iter().any(|name| name == present),
            "{present}: {names:?}"
        );
    }
    assert!(
        specialized
            .declarations(db)
            .function_types
            .iter()
            .all(|(path, _)| *path != ["identity"] && *path != ["unreached"])
    );
}
