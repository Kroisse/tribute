use salsa_test_macros::salsa_test;

use super::*;
use crate::ast::{AbilityId, NodeId, RowVar, TypeDefId, UniVarId, UniVarSource};
use trunk_ir::Symbol;

use crate::resolve::PackageName;

fn names() -> NameDisplay {
    NameDisplay::new(&PackageName::executable_root(), [])
}

fn effect<'db>(
    db: &'db dyn salsa::Database,
    name: &'static str,
    args: Vec<Type<'db>>,
) -> AbilityInstance<'db> {
    AbilityInstance {
        ability_id: AbilityId::source(db, trunk_ir::SymbolPath::new([name])),
        args,
    }
}

/// Preserve diagnostic spellings across the different type forms.
fn examples(db: &dyn salsa::Database) -> Vec<(Type<'_>, &'static str)> {
    let int = Type::new(db, TypeKind::Int);
    let bool_ty = Type::new(db, TypeKind::Bool);
    let pure = AbilityRow::pure(db);
    let named = |name, args| TypeKind::Named {
        id: TypeDefId::synthetic(db, trunk_ir::SymbolPath::new([name])),
        name: Symbol::new(name),
        args,
    };
    let cases = vec![
        (TypeKind::Int, "Int"),
        (TypeKind::Nat, "Nat"),
        (TypeKind::Float, "Float"),
        (TypeKind::Bool, "Bool"),
        (TypeKind::Bytes, "Bytes"),
        (TypeKind::Rune, "Rune"),
        (TypeKind::Nil, "Nil"),
        (TypeKind::Never, "Never"),
        (TypeKind::Error, "<error>"),
        (TypeKind::BoundVar { index: 0 }, "_0"),
        (TypeKind::BoundVar { index: 2 }, "_2"),
        (TypeKind::BoundVar { index: 10 }, "_10"),
        (TypeKind::BoundVar { index: u32::MAX }, "_4294967295"),
        (
            TypeKind::LocalBoundVar {
                scope: NodeId::from_raw(1),
                index: 10,
            },
            "_local10",
        ),
        (
            TypeKind::UniVar {
                id: UniVarId::new(db, UniVarSource::Anonymous(1), 0),
            },
            "_",
        ),
        (named("", vec![]), ""),
        (named("IntBox", vec![]), "IntBox"),
        (named("Int", vec![bool_ty]), "Int(Bool)"),
        (named("é", vec![int]), "é(Int)"),
        (named("효과", vec![]), "효과"),
        (TypeKind::Tuple(vec![]), "#()"),
        (TypeKind::Tuple(vec![int, bool_ty]), "#(Int, Bool)"),
        (
            TypeKind::Func {
                params: vec![],
                result: int,
                effect: pure,
            },
            "fn() -> Int",
        ),
        (
            TypeKind::Func {
                params: vec![int, bool_ty],
                result: int,
                effect: pure,
            },
            "fn(Int, Bool) -> Int",
        ),
        (
            TypeKind::Continuation {
                arg: int,
                result: bool_ty,
                effect: pure,
            },
            "Continuation(Int -> Bool)",
        ),
    ];
    cases
        .into_iter()
        .map(|(kind, spelling)| (Type::new(db, kind), spelling))
        .collect()
}

#[salsa_test]
fn type_and_effect_display_preserve_spelling(db: &salsa::DatabaseImpl) {
    for (ty, spelling) in examples(db) {
        assert_eq!(ty.display(db, &names()).to_string(), spelling);
        assert_eq!(
            effect(db, "State", vec![ty])
                .display(db, &names())
                .to_string(),
            format!("State({spelling})")
        );
    }
}

#[salsa_test]
fn ability_row_display_preserves_distinct_identities(db: &salsa::DatabaseImpl) {
    let int = Type::new(db, TypeKind::Int);
    let pure = AbilityRow::pure(db);
    let open = AbilityRow::open(db, RowVar { id: 987 });
    let named = |module| TypeKind::Named {
        id: TypeDefId::source(db, trunk_ir::SymbolPath::new([module, "T"])),
        name: Symbol::new("T"),
        args: vec![],
    };
    let pairs = [
        (named("A"), named("B")),
        (
            TypeKind::UniVar {
                id: UniVarId::new(db, UniVarSource::Anonymous(1), 0),
            },
            TypeKind::UniVar {
                id: UniVarId::new(db, UniVarSource::Anonymous(2), 0),
            },
        ),
        (
            TypeKind::LocalBoundVar {
                scope: NodeId::from_raw(1),
                index: 0,
            },
            TypeKind::LocalBoundVar {
                scope: NodeId::from_raw(2),
                index: 0,
            },
        ),
        (
            TypeKind::Func {
                params: vec![],
                result: int,
                effect: pure,
            },
            TypeKind::Func {
                params: vec![],
                result: int,
                effect: open,
            },
        ),
        (
            TypeKind::Continuation {
                arg: int,
                result: int,
                effect: pure,
            },
            TypeKind::Continuation {
                arg: int,
                result: int,
                effect: open,
            },
        ),
    ];
    for (a, b) in pairs {
        let a = effect(db, "State", vec![Type::new(db, a)]);
        let b = effect(db, "State", vec![Type::new(db, b)]);
        assert_ne!(a, b);
        assert_eq!(
            a.display(db, &names()).to_string(),
            b.display(db, &names()).to_string()
        );
        let expected = format!(
            "{{{}, {}}}",
            a.display(db, &names()),
            b.display(db, &names())
        );
        let row = AbilityRow::new(db, vec![a, b], None);
        assert_eq!(row.display(db, &names()).to_string(), expected);
    }
    let source = AbilityInstance {
        ability_id: AbilityId::source(db, trunk_ir::SymbolPath::new(["std", "io", "Io"])),
        args: vec![],
    };
    let builtin = AbilityInstance {
        ability_id: AbilityId::builtin_io(db),
        args: vec![],
    };
    assert_ne!(source, builtin);
    let row = AbilityRow::new(db, vec![source, builtin], None);
    assert_eq!(
        row.display(db, &names()).to_string(),
        "{std::io::Io, std::io::Io}"
    );
}

#[salsa_test]
fn ability_row_display_is_canonical(db: &salsa::DatabaseImpl) {
    use crate::typeck::SolveError;

    let pure = AbilityRow::pure(db);
    assert_eq!(pure.display(db, &names()).to_string(), "{}");
    assert_eq!(
        AbilityRow::open(db, RowVar { id: 99 })
            .display(db, &names())
            .to_string(),
        "{e}"
    );
    let mut effects = vec![
        effect(
            db,
            "State",
            vec![Type::new(db, TypeKind::BoundVar { index: 2 })],
        ),
        effect(db, "Console", vec![]),
        effect(
            db,
            "State",
            vec![Type::new(db, TypeKind::BoundVar { index: 10 })],
        ),
    ];
    let closed = AbilityRow::new(db, effects.clone(), None);
    assert_eq!(
        closed.display(db, &names()).to_string(),
        "{Console, State(_10), State(_2)}"
    );
    effects.reverse();
    assert_eq!(
        AbilityRow::new(db, effects.clone(), None)
            .display(db, &names())
            .to_string(),
        closed.display(db, &names()).to_string()
    );
    let open = AbilityRow::new(db, effects, Some(RowVar { id: 777 }));
    assert_eq!(
        open.display(db, &names()).to_string(),
        "{Console, State(_10), State(_2), e}"
    );
    let mismatch = SolveError::RowMismatch {
        expected: pure,
        actual: open,
    };
    assert_eq!(
        mismatch.display(db, &names()).to_string(),
        "ability row mismatch: expected `{}`, found `{Console, State(_10), State(_2), e}`"
    );
    let ambiguous = SolveError::AmbiguousAbility {
        expected: pure,
        actual: open,
    };
    assert_eq!(
        ambiguous.display(db, &names()).to_string(),
        "ambiguous ability: cannot match `{Console, State(_10), State(_2), e}` with `{}`, \
         more than one instance could correspond; annotate the ability's type arguments"
    );
}

#[salsa_test]
fn ability_row_displays_deep_types(db: &salsa::DatabaseImpl) {
    let mut ty = Type::new(db, TypeKind::Int);
    for _ in 0..32 {
        ty = Type::new(db, TypeKind::Tuple(vec![ty]));
    }
    let a = effect(db, "State", vec![ty]);
    let b = effect(db, "State", vec![ty, ty]);
    assert_eq!(
        a.display(db, &names()).to_string(),
        format!("State({}Int{})", "#(".repeat(32), ")".repeat(32))
    );
    let expected = format!(
        "{{{}, {}}}",
        a.display(db, &names()),
        b.display(db, &names())
    );
    let row = AbilityRow::new(db, vec![a, b], None);
    assert_eq!(row.display(db, &names()).to_string(), expected);
}

#[salsa_test]
fn a_library_prints_its_own_types_without_its_name(db: &salsa::DatabaseImpl) {
    let std = Symbol::new(crate::resolve::STD_PACKAGE);
    let lib = NameDisplay::new(&PackageName::library(Symbol::new("lib")), [&std]);
    let named = |identity: &str, args| {
        Type::new(
            db,
            TypeKind::Named {
                id: TypeDefId::source(db, trunk_ir::SymbolPath::new(identity.split("::"))),
                name: Symbol::new(identity),
                args,
            },
        )
    };
    let option = named("std::Option", vec![named("lib::geometry::Shape", vec![])]);
    let own_std = named("lib::std::Shape", vec![]);
    let user = named("pkg::Shape", vec![]);
    let pair = Type::new(db, TypeKind::Tuple(vec![option, own_std, user]));

    assert_eq!(
        pair.display(db, &lib).to_string(),
        "#(std::Option(geometry::Shape), lib::std::Shape, pkg::Shape)"
    );
    assert_eq!(
        pair.display(db, &NameDisplay::new(&PackageName::std(), []))
            .to_string(),
        "#(Option(lib::geometry::Shape), lib::std::Shape, pkg::Shape)"
    );
}

/// Abilities of the same name in different modules print apart, spelled
/// relative to the compiled package.
#[salsa_test]
fn ability_display_keeps_the_module_path(db: &salsa::DatabaseImpl) {
    let state = |module: &str| AbilityInstance {
        ability_id: AbilityId::source(db, trunk_ir::SymbolPath::new(["pkg", module, "State"])),
        args: vec![],
    };
    let row = AbilityRow::new(db, vec![state("a"), state("b")], None);
    assert_eq!(
        row.display(db, &names()).to_string(),
        "{a::State, b::State}"
    );
}
