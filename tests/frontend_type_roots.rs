//! Valid metadata-only roots must materialize nominal declarations before erasure.
use salsa_test_macros::salsa_test;
use tribute::pipeline::{parse_and_lower_ast, prepare_frontend_for_lowering};
use tribute_core::Diagnostic;
use tribute_front::{
    SourceCst,
    ast::{
        AbilityId, Decl, Effect, EffectRow, EffectVar, RowRemoval, RowUnion, Type, TypeDefId,
        TypeKind, TypeParam,
    },
    typeck::TypeCheckOutput,
};
use trunk_ir::Symbol;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Root {
    TypeArgument,
    RowArgument,
    Union,
    Removal,
    ConstructorUnion,
}

#[salsa::tracked(returns(copy))]
fn prepare_root<'db>(
    db: &'db dyn salsa::Database,
    source: SourceCst,
    root: Root,
) -> TypeCheckOutput<'db> {
    let typed = parse_and_lower_ast(db, source).unwrap();
    let packet_decl = typed
        .module(db)
        .decls
        .iter()
        .find_map(|decl| match decl {
            Decl::Struct(decl) if decl.name == Symbol::new("Packet") => Some(decl),
            _ => None,
        })
        .unwrap();
    let packet = Type::new(
        db,
        TypeKind::Named {
            id: TypeDefId::source(db, packet_decl.name, packet_decl.id),
            name: packet_decl.name,
            args: vec![Type::new(db, TypeKind::Nat)],
        },
    );
    let row = EffectRow::single(
        db,
        Effect {
            ability_id: AbilityId::source(db, Symbol::new("Marker")),
            args: vec![packet],
        },
    );
    let union = RowUnion {
        sources: vec![row],
        result: row,
    };
    let mut functions = typed.function_types(db).clone();
    let mut constructors = typed.constructor_types(db).clone();
    let mut metadata = typed.expression_types(db).clone();
    let (_, scheme) = functions
        .iter_mut()
        .find(|(name, _)| *name == Symbol::new("marker"))
        .unwrap();
    let instance = &mut metadata
        .function_instances
        .iter_mut()
        .find(|(_, instance)| instance.function.qualified(db) == Symbol::new("marker"))
        .unwrap()
        .1;

    // Explicitly quantified phantom binders and tautological constraints are
    // valid at this boundary. Each case adds Packet(Nat) to only one metadata
    // category; the source contains no construction or concrete signature.
    let builder = scheme.to_builder(db);
    match root {
        Root::TypeArgument => {
            *scheme = builder.type_params(vec![TypeParam::anonymous()]).build(db);
            instance.type_arguments = vec![packet];
        }
        Root::RowArgument => {
            *scheme = tribute_front::ast::TypeScheme::builder(
                scheme.type_params(db).clone(),
                vec![EffectVar { id: 999 }],
                scheme.body(db),
            )
            .build(db);
            instance.row_arguments = vec![row];
        }
        Root::Union => *scheme = builder.row_unions(vec![union]).build(db),
        Root::Removal => {
            *scheme = builder
                .row_removals(vec![RowRemoval {
                    source: row,
                    removed: row,
                    result: EffectRow::pure(db),
                }])
                .build(db)
        }
        Root::ConstructorUnion => {
            let (_, constructor) = constructors
                .schemes
                .iter_mut()
                .find(|(id, _)| id.qualified(db) == Symbol::new("Packet"))
                .unwrap();
            *constructor = constructor.to_builder(db).row_unions(vec![union]).build(db);
        }
    }
    instance.scheme = *scheme;
    let input = TypeCheckOutput::new(
        db,
        typed.module(db).clone(),
        functions,
        constructors,
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
    let prepared = prepare_frontend_for_lowering(db, input, source);
    let errors = prepare_frontend_for_lowering::accumulated::<Diagnostic>(db, input, source);
    assert!(
        errors.is_empty(),
        "invalid metadata for {root:?}: {errors:?}"
    );
    prepared.unwrap()
}

fn assert_root(db: &salsa::DatabaseImpl, root: Root) {
    let source = SourceCst::from_source_str(
        db,
        "metadata_roots.trb",
        r#"
struct Packet(a) { value: a }
ability Marker(a) { op mark(value: a) -> Nil }
fn marker() { Nil }
fn main() { marker() }
"#,
    );
    let prepared = prepare_root(db, source, root);
    let declaration = prepared
        .module(db)
        .decls
        .iter()
        .find_map(|decl| match decl {
            Decl::Struct(decl) if decl.name == Symbol::new("Packet$Nat") => Some(decl),
            _ => None,
        })
        .unwrap_or_else(|| panic!("{root:?} did not materialize Packet(Nat)"));
    assert!(declaration.type_params.is_empty());
    let marker = prepared
        .function_types(db)
        .iter()
        .find(|(name, _)| *name == Symbol::new("marker"))
        .unwrap()
        .1;
    let instance = &prepared
        .expression_types(db)
        .function_instances
        .iter()
        .find(|(_, instance)| instance.function.qualified(db) == Symbol::new("marker"))
        .unwrap()
        .1;
    let ty = match root {
        Root::TypeArgument => instance.type_arguments[0],
        Root::RowArgument => instance.row_arguments[0].effects(db)[0].args[0],
        Root::Union => marker.row_unions(db)[0].result.effects(db)[0].args[0],
        Root::Removal => marker.row_removals(db)[0].removed.effects(db)[0].args[0],
        Root::ConstructorUnion => {
            prepared
                .constructor_types(db)
                .schemes
                .iter()
                .find(|(id, _)| id.qualified(db) == Symbol::new("Packet"))
                .unwrap()
                .1
                .row_unions(db)[0]
                .result
                .effects(db)[0]
                .args[0]
        }
    };
    let TypeKind::Named { id, args, .. } = ty.kind(db) else {
        panic!("nominal type")
    };
    assert!(args.is_empty(), "{root:?} retained generic arguments");
    assert_eq!(id.qualified(db), declaration.name);
    let original = prepared
        .module(db)
        .decls
        .iter()
        .find_map(|decl| match decl {
            Decl::Struct(decl) if decl.name == Symbol::new("Packet") => Some(decl),
            _ => None,
        })
        .unwrap();
    assert_eq!(
        id.origin(db),
        TypeDefId::source(db, original.name, original.id).origin(db)
    );
}

#[salsa_test]
fn type_argument_is_a_nominal_root(db: &salsa::DatabaseImpl) {
    assert_root(db, Root::TypeArgument);
}

#[salsa_test]
fn row_argument_is_a_nominal_root(db: &salsa::DatabaseImpl) {
    assert_root(db, Root::RowArgument);
}

#[salsa_test]
fn retained_union_is_a_nominal_root(db: &salsa::DatabaseImpl) {
    assert_root(db, Root::Union);
}

#[salsa_test]
fn retained_removal_is_a_nominal_root(db: &salsa::DatabaseImpl) {
    assert_root(db, Root::Removal);
}

#[salsa_test]
fn constructor_scheme_is_a_nominal_root(db: &salsa::DatabaseImpl) {
    assert_root(db, Root::ConstructorUnion);
}
