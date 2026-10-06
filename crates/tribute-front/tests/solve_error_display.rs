use tribute_front::{
    ast::{EffectRow, EffectVar},
    typeck::SolveError,
};

#[test]
fn effect_errors_display_without_an_attached_database() {
    let db = salsa::DatabaseImpl::default();
    let expected = EffectRow::pure(&db);
    let actual = EffectRow::open(&db, EffectVar { id: 42 });

    let ambiguous = "more than one instance could correspond; annotate the effect's type arguments";
    for (error, detached, attached) in [
        (
            SolveError::RowMismatch { expected, actual },
            "effect mismatch: expected `<effect row>`, found `<effect row>`".to_owned(),
            "effect mismatch: expected `{}`, found `{e}`".to_owned(),
        ),
        (
            SolveError::AmbiguousEffect { expected, actual },
            format!(
                "ambiguous effect: cannot match `<effect row>` with `<effect row>`, {ambiguous}"
            ),
            format!("ambiguous effect: cannot match `{{e}}` with `{{}}`, {ambiguous}"),
        ),
    ] {
        assert_eq!(error.to_string(), detached);
        salsa::Database::attach(&db, |_| {
            assert_eq!(error.to_string(), attached);
        });
    }
}
