use tribute_front::{
    ast::{AbilityRow, RowVar},
    resolve::{NameDisplay, PackageName},
    typeck::SolveError,
};

#[test]
fn effect_errors_display_with_the_database_they_are_given() {
    let db = salsa::DatabaseImpl::default();
    let names = NameDisplay::new(&PackageName::executable_root(), []);
    let expected = AbilityRow::pure(&db);
    let actual = AbilityRow::open(&db, RowVar { id: 42 });

    let ambiguous =
        "more than one instance could correspond; annotate the ability's type arguments";
    for (error, message) in [
        (
            SolveError::RowMismatch { expected, actual },
            "ability row mismatch: expected `{}`, found `{e}`".to_owned(),
        ),
        (
            SolveError::AmbiguousAbility { expected, actual },
            format!("ambiguous ability: cannot match `{{e}}` with `{{}}`, {ambiguous}"),
        ),
    ] {
        assert_eq!(error.display(&db, &names).to_string(), message);
    }
}
