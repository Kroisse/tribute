use salsa_test_macros::salsa_test;
use tribute_core::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use tribute_front::SourceCst;

fn type_errors(db: &dyn salsa::Database, source: SourceCst) -> Vec<String> {
    let _ = tribute_front::query::typed_module(db, source);
    tribute_front::query::typed_module::accumulated::<Diagnostic>(db, source)
        .into_iter()
        .filter(|diagnostic| {
            diagnostic.inner.severity == DiagnosticSeverity::Error
                && diagnostic.phase == CompilationPhase::TypeChecking
        })
        .map(|diagnostic| diagnostic.inner.message.clone())
        .collect()
}

#[salsa_test]
fn pure_local_identity_is_polymorphic(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn pair() -> #(Nat, Bool) {
    let identity = fn(value) value
    #(identity(1), identity(true))
}
"#,
    );

    let errors = type_errors(db, source);
    assert!(errors.is_empty(), "unexpected type errors: {errors:#?}");
}

#[salsa_test]
fn latent_function_effect_does_not_trigger_value_restriction(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability Source(s) {
    op get() -> s
}

extern "intrinsic" fn take_nat_reader(reader: fn() ->{Source(Nat)} Nat) ->{} Nil
extern "intrinsic" fn take_bool_reader(reader: fn() ->{Source(Bool)} Bool) ->{} Nil

fn accepted() ->{} Nil {
    let read = fn() Source::get()
    take_nat_reader(read)
    take_bool_reader(read)
}
"#,
    );

    let errors = type_errors(db, source);
    assert!(errors.is_empty(), "unexpected type errors: {errors:#?}");
}

#[salsa_test]
fn effectful_rhs_is_not_generalized(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability Source(s) {
    op get() -> s
}

extern "intrinsic" fn take_nat(value: Nat) ->{} Nil
extern "intrinsic" fn take_bool(value: Bool) ->{} Nil

fn restricted() -> Nil {
    let value = Source::get()
    take_nat(value)
    take_bool(value)
}
"#,
    );

    let errors = type_errors(db, source);
    assert!(
        errors
            .iter()
            .any(|error| error.contains("expected `Nat`, found `Bool`")
                || error.contains("expected `Bool`, found `Nat`")),
        "expected monomorphic value restriction error, got: {errors:#?}"
    );
}
