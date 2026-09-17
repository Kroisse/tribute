use tribute_front::{
    ast::{EffectRow, EffectVar},
    typeck::SolveError,
};

#[test]
fn effect_errors_display_without_an_attached_database() {
    let db = salsa::DatabaseImpl::default();
    let expected = EffectRow::pure(&db);
    let actual = EffectRow::open(&db, EffectVar { id: 42 });

    for error in [
        SolveError::RowMismatch { expected, actual },
        SolveError::AmbiguousEffect { expected, actual },
    ] {
        assert_eq!(
            error.to_string(),
            "effect mismatch: expected `<effect row>`, found `<effect row>`"
        );
        salsa::Database::attach(&db, |_| {
            assert_eq!(
                error.to_string(),
                "effect mismatch: expected `{}`, found `{e}`"
            );
        });
    }
}
