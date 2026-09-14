use salsa_test_macros::salsa_test;

use super::*;
use crate::ast::{
    AbilityId, CallingConvention, EffectVar, NodeId, TypeDefId, UniVarId, UniVarSource,
};

fn effect<'db>(
    db: &'db dyn salsa::Database,
    name: &'static str,
    args: Vec<Type<'db>>,
) -> Effect<'db> {
    Effect {
        ability_id: AbilityId::source(db, Symbol::new(name)),
        args,
    }
}

/// Explicit spellings guard the shared renderer independently of the comparator.
fn examples(db: &dyn salsa::Database) -> Vec<(Type<'_>, &'static str)> {
    let int = Type::new(db, TypeKind::Int);
    let bool_ty = Type::new(db, TypeKind::Bool);
    let pure = EffectRow::pure(db);
    let named = |name, args| TypeKind::Named {
        id: TypeDefId::synthetic(db, Symbol::new(name)),
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
                minimum_convention: CallingConvention::Direct,
            },
            "fn() -> Int",
        ),
        (
            TypeKind::Func {
                params: vec![int, bool_ty],
                result: int,
                effect: pure,
                minimum_convention: CallingConvention::Direct,
            },
            "fn(Int, Bool) -> Int",
        ),
        (
            TypeKind::App {
                ctor: int,
                args: vec![],
            },
            "Int()",
        ),
        (
            TypeKind::App {
                ctor: int,
                args: vec![bool_ty],
            },
            "Int(Bool)",
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
        assert_eq!(ty.to_string(), spelling);
        assert_eq!(
            effect(db, "State", vec![ty]).to_string(),
            format!("State({spelling})")
        );
    }
}

#[test]
fn cmp_for_display_matches_string_order_without_attached_database() {
    let db = salsa::DatabaseImpl::default();
    let mut effects = Vec::new();
    for name in ["State", "Stateful", "é", "효과"] {
        effects.push(effect(&db, name, vec![]));
        for (ty, _) in examples(&db) {
            effects.push(effect(&db, name, vec![ty]));
            effects.push(effect(&db, name, vec![ty, Type::new(&db, TypeKind::Int)]));
        }
    }
    // Equal output with different fragment boundaries must compare equally.
    effects.push(effect(
        &db,
        "State",
        vec![Type::new(
            &db,
            TypeKind::Named {
                id: TypeDefId::synthetic(&db, Symbol::new("Int(Bool)")),
                name: Symbol::new("Int(Bool)"),
                args: vec![],
            },
        )],
    ));
    let rendered: Vec<_> = salsa::attach(&db, || effects.iter().map(ToString::to_string).collect());
    for (a, a_text) in effects.iter().zip(&rendered) {
        for (b, b_text) in effects.iter().zip(&rendered) {
            assert_eq!(
                a.cmp_for_display(&db, b),
                a_text.cmp(b_text),
                "{a_text:?} vs {b_text:?}"
            );
        }
    }
}

#[salsa_test]
fn display_order_does_not_compare_hidden_identity(db: &salsa::DatabaseImpl) {
    let int = Type::new(db, TypeKind::Int);
    let pure = EffectRow::pure(db);
    let open = EffectRow::open(db, EffectVar { id: 987 });
    let named = |declaration| TypeKind::Named {
        id: TypeDefId::source(db, Symbol::new("T"), NodeId::from_raw(declaration)),
        name: Symbol::new("T"),
        args: vec![],
    };
    let pairs = [
        (named(1), named(2)),
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
                minimum_convention: CallingConvention::Direct,
            },
            TypeKind::Func {
                params: vec![],
                result: int,
                effect: open,
                minimum_convention: CallingConvention::Cps,
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
        assert_eq!(a.cmp_for_display(db, &b), Ordering::Equal);
        assert_eq!(a.to_string(), b.to_string());
    }
    let source = effect(db, "std::io::Io", vec![]);
    let builtin = Effect {
        ability_id: AbilityId::builtin_io(db),
        args: vec![],
    };
    assert_ne!(source, builtin);
    assert_eq!(source.cmp_for_display(db, &builtin), Ordering::Equal);
}

#[salsa_test]
fn effect_row_display_is_canonical(db: &salsa::DatabaseImpl) {
    use crate::typeck::SolveError;

    let pure = EffectRow::pure(db);
    assert_eq!(pure.to_string(), "{}");
    assert_eq!(EffectRow::open(db, EffectVar { id: 99 }).to_string(), "{e}");
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
    let closed = EffectRow::new(db, effects.clone(), None);
    assert_eq!(closed.to_string(), "{Console, State(_10), State(_2)}");
    effects.reverse();
    assert_eq!(
        EffectRow::new(db, effects.clone(), None).to_string(),
        closed.to_string()
    );
    let open = EffectRow::new(db, effects, Some(EffectVar { id: 777 }));
    assert_eq!(open.to_string(), "{Console, State(_10), State(_2), e}");
    for error in [
        SolveError::RowMismatch {
            expected: pure,
            actual: open,
        },
        SolveError::AmbiguousEffect {
            expected: pure,
            actual: open,
        },
    ] {
        assert_eq!(
            error.to_string(),
            "effect mismatch: expected `{}`, found `{Console, State(_10), State(_2), e}`"
        );
    }
}

#[salsa_test]
fn display_comparison_handles_deep_types(db: &salsa::DatabaseImpl) {
    let mut ty = Type::new(db, TypeKind::Int);
    for _ in 0..32 {
        ty = Type::new(db, TypeKind::Tuple(vec![ty]));
    }
    let a = effect(db, "State", vec![ty]);
    let b = effect(db, "State", vec![ty, ty]);
    assert_eq!(
        a.to_string(),
        format!("State({}Int{})", "#(".repeat(32), ")".repeat(32))
    );
    assert_eq!(a.cmp_for_display(db, &b), a.to_string().cmp(&b.to_string()));
}
