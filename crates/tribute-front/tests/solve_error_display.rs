use tribute_front::{
    ast::{AbilityRow, RowVar},
    typeck::SolveError,
};

#[test]
fn effect_errors_display_without_an_attached_database() {
    let db = salsa::DatabaseImpl::default();
    let expected = AbilityRow::pure(&db);
    let actual = AbilityRow::open(&db, RowVar { id: 42 });

    let ambiguous =
        "more than one instance could correspond; annotate the ability's type arguments";
    for (error, detached, attached) in [
        (
            SolveError::RowMismatch { expected, actual },
            "ability row mismatch: expected `<ability row>`, found `<ability row>`".to_owned(),
            "ability row mismatch: expected `{}`, found `{e}`".to_owned(),
        ),
        (
            SolveError::AmbiguousAbility { expected, actual },
            format!(
                "ambiguous ability: cannot match `<ability row>` with `<ability row>`, {ambiguous}"
            ),
            format!("ambiguous ability: cannot match `{{e}}` with `{{}}`, {ambiguous}"),
        ),
    ] {
        assert_eq!(error.to_string(), detached);
        salsa::Database::attach(&db, |_| {
            assert_eq!(error.to_string(), attached);
        });
    }
}
