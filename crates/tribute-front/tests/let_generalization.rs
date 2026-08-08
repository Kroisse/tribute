use salsa_test_macros::salsa_test;
use tribute_core::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use tribute_front::{
    SourceCst,
    ast::{Type, TypeKind},
};

fn type_contains_univar(db: &dyn salsa::Database, ty: Type<'_>) -> bool {
    match ty.kind(db) {
        TypeKind::UniVar { .. } => true,
        TypeKind::Named { args, .. } => args.iter().any(|arg| type_contains_univar(db, *arg)),
        TypeKind::Func { params, result, .. } => {
            params.iter().any(|param| type_contains_univar(db, *param))
                || type_contains_univar(db, *result)
        }
        TypeKind::Tuple(elements) => elements
            .iter()
            .any(|element| type_contains_univar(db, *element)),
        TypeKind::App { ctor, args } => {
            type_contains_univar(db, *ctor) || args.iter().any(|arg| type_contains_univar(db, *arg))
        }
        TypeKind::Continuation { arg, result, .. } => {
            type_contains_univar(db, *arg) || type_contains_univar(db, *result)
        }
        _ => false,
    }
}

fn type_contains_local_bound(db: &dyn salsa::Database, ty: Type<'_>) -> bool {
    match ty.kind(db) {
        TypeKind::LocalBoundVar { .. } => true,
        TypeKind::Named { args, .. } => args.iter().any(|arg| type_contains_local_bound(db, *arg)),
        TypeKind::Func { params, result, .. } => {
            params
                .iter()
                .any(|param| type_contains_local_bound(db, *param))
                || type_contains_local_bound(db, *result)
        }
        TypeKind::Tuple(elements) => elements
            .iter()
            .any(|element| type_contains_local_bound(db, *element)),
        TypeKind::App { ctor, args } => {
            type_contains_local_bound(db, *ctor)
                || args.iter().any(|arg| type_contains_local_bound(db, *arg))
        }
        TypeKind::Continuation { arg, result, .. } => {
            type_contains_local_bound(db, *arg) || type_contains_local_bound(db, *result)
        }
        _ => false,
    }
}

fn phase_errors(
    db: &dyn salsa::Database,
    source: SourceCst,
    phase: CompilationPhase,
) -> Vec<String> {
    let _ = tribute_front::query::typed_module(db, source);
    tribute_front::query::typed_module::accumulated::<Diagnostic>(db, source)
        .into_iter()
        .filter(|diagnostic| {
            diagnostic.inner.severity == DiagnosticSeverity::Error && diagnostic.phase == phase
        })
        .map(|diagnostic| diagnostic.inner.message.clone())
        .collect()
}

fn type_errors(db: &dyn salsa::Database, source: SourceCst) -> Vec<String> {
    phase_errors(db, source, CompilationPhase::TypeChecking)
}

#[salsa_test]
fn pure_local_identity_is_polymorphic(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn pair() -> #(Nat, Bool) {
    let identity = fn(value) value
    #(identity(1), identity(True))
}
"#,
    );

    let errors = type_errors(db, source);
    assert!(errors.is_empty(), "unexpected type errors: {errors:#?}");
}

#[salsa_test]
fn local_callable_metadata_has_no_raw_solver_variables(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn pair() -> #(Nat, Bool) {
    let identity = fn(value) value
    #(identity(1), identity(True))
}
"#,
    );

    let output = tribute_front::query::type_check_output(db, source)
        .expect("type checking should produce output");
    assert!(
        output
            .lambda_signatures(db)
            .iter()
            .all(|(_, signature)| !type_contains_univar(db, signature.function_type)),
        "locally generalized callable metadata leaked a raw solver variable"
    );
    assert!(
        output
            .lambda_signatures(db)
            .iter()
            .any(|(_, signature)| type_contains_local_bound(db, signature.function_type)),
        "locally generalized callable metadata lost its lexical quantifier owner"
    );
}

#[salsa_test]
fn variant_destructuring_preserves_polymorphism(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
enum Wrapped(a) {
    Wrapped(a),
}

fn pair() -> #(Nat, Bool) {
    let Wrapped(identity) = Wrapped(fn(value) value)
    #(identity(1), identity(True))
}
"#,
    );

    let errors = type_errors(db, source);
    assert!(errors.is_empty(), "unexpected type errors: {errors:#?}");
}

#[salsa_test]
fn as_pattern_alias_preserves_polymorphism(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn pair() -> #(Nat, Bool) {
    let _ as identity = fn(value) value
    #(identity(1), identity(True))
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

#[salsa_test]
fn effectful_destructuring_is_not_generalized(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability Source(s) {
    op get() -> s
}

enum Wrapped(a) {
    Wrapped(a),
}

extern "intrinsic" fn take_nat(value: Nat) ->{} Nil
extern "intrinsic" fn take_bool(value: Bool) ->{} Nil

fn restricted() -> Nil {
    let Wrapped(value) = Wrapped(Source::get())
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
        "expected monomorphic destructuring error, got: {errors:#?}"
    );
}

#[salsa_test]
fn record_shorthand_binding_stays_free_in_later_scheme(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
struct Holder(a) { value: a }

ability Source(s) {
    op get() -> s
}

extern "intrinsic" fn take_nat(value: Nat) ->{} Nil
extern "intrinsic" fn take_bool(value: Bool) ->{} Nil

fn restricted() -> Nil {
    let Holder { value } = Holder(Source::get())
    let copy = value
    take_nat(copy)
    take_bool(copy)
}
"#,
    );

    let resolution_errors = phase_errors(db, source, CompilationPhase::NameResolution);
    assert!(
        resolution_errors.is_empty(),
        "record shorthand should resolve as a local binding: {resolution_errors:#?}"
    );

    let errors = type_errors(db, source);
    assert!(
        errors
            .iter()
            .any(|error| error.contains("expected `Nat`, found `Bool`")
                || error.contains("expected `Bool`, found `Nat`")),
        "expected captured shorthand type to remain monomorphic, got: {errors:#?}"
    );
}
