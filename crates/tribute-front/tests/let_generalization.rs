use salsa_test_macros::salsa_test;
use tribute_core::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use tribute_front::{
    SourceCst,
    ast::{Decl, Expr, ExprKind, HandlerKind, Pattern, PatternKind, Stmt, Type, TypeKind},
};

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

fn type_contains_univar(db: &dyn salsa::Database, ty: Type<'_>) -> bool {
    match ty.kind(db) {
        TypeKind::UniVar { .. } => true,
        TypeKind::Named { args, .. } => args.iter().any(|arg| type_contains_univar(db, *arg)),
        TypeKind::Func {
            params,
            result,
            effect,
            ..
        } => {
            params.iter().any(|param| type_contains_univar(db, *param))
                || type_contains_univar(db, *result)
                || effect
                    .effects(db)
                    .iter()
                    .flat_map(|entry| entry.args.iter())
                    .any(|arg| type_contains_univar(db, *arg))
        }
        TypeKind::Tuple(elements) => elements
            .iter()
            .any(|element| type_contains_univar(db, *element)),
        TypeKind::App { ctor, args } => {
            type_contains_univar(db, *ctor) || args.iter().any(|arg| type_contains_univar(db, *arg))
        }
        TypeKind::Continuation {
            arg,
            result,
            effect,
        } => {
            type_contains_univar(db, *arg)
                || type_contains_univar(db, *result)
                || effect
                    .effects(db)
                    .iter()
                    .flat_map(|entry| entry.args.iter())
                    .any(|entry| type_contains_univar(db, *entry))
        }
        _ => false,
    }
}

fn collect_local_bound_scopes(
    db: &dyn salsa::Database,
    ty: Type<'_>,
    scopes: &mut Vec<tribute_front::ast::NodeId>,
) {
    let mut bounds = Vec::new();
    collect_local_bounds(db, ty, &mut bounds);
    scopes.extend(bounds.into_iter().map(|(scope, _)| scope));
}

fn collect_local_bounds(
    db: &dyn salsa::Database,
    ty: Type<'_>,
    bounds: &mut Vec<(tribute_front::ast::NodeId, u32)>,
) {
    match ty.kind(db) {
        TypeKind::LocalBoundVar { scope, index } => bounds.push((*scope, *index)),
        TypeKind::Named { args, .. } => {
            for arg in args {
                collect_local_bounds(db, *arg, bounds);
            }
        }
        TypeKind::Func {
            params,
            result,
            effect,
            ..
        } => {
            for param in params {
                collect_local_bounds(db, *param, bounds);
            }
            collect_local_bounds(db, *result, bounds);
            for entry in effect.effects(db) {
                for arg in &entry.args {
                    collect_local_bounds(db, *arg, bounds);
                }
            }
        }
        TypeKind::Tuple(elements) => {
            for element in elements {
                collect_local_bounds(db, *element, bounds);
            }
        }
        TypeKind::App { ctor, args } => {
            collect_local_bounds(db, *ctor, bounds);
            for arg in args {
                collect_local_bounds(db, *arg, bounds);
            }
        }
        TypeKind::Continuation {
            arg,
            result,
            effect,
        } => {
            collect_local_bounds(db, *arg, bounds);
            collect_local_bounds(db, *result, bounds);
            for entry in effect.effects(db) {
                for arg in &entry.args {
                    collect_local_bounds(db, *arg, bounds);
                }
            }
        }
        _ => {}
    }
}

fn assert_typed_refs_resolved(
    db: &dyn salsa::Database,
    expr: &Expr<tribute_front::ast::TypedRef<'_>>,
) {
    match &*expr.kind {
        ExprKind::Var(reference) => assert!(
            !type_contains_univar(db, reference.ty),
            "typed reference leaked a raw UniVar: {}",
            reference.ty
        ),
        ExprKind::Cons { ctor, args } => {
            assert!(!type_contains_univar(db, ctor.ty));
            for arg in args {
                assert_typed_refs_resolved(db, arg);
            }
        }
        ExprKind::Record {
            type_name,
            fields,
            spread,
        } => {
            assert!(!type_contains_univar(db, type_name.ty));
            for (_, value) in fields {
                assert_typed_refs_resolved(db, value);
            }
            if let Some(spread) = spread {
                assert_typed_refs_resolved(db, spread);
            }
        }
        ExprKind::MethodCall { receiver, args, .. } => {
            assert_typed_refs_resolved(db, receiver);
            for arg in args {
                assert_typed_refs_resolved(db, arg);
            }
        }
        ExprKind::Call { callee, args } => {
            assert_typed_refs_resolved(db, callee);
            for arg in args {
                assert_typed_refs_resolved(db, arg);
            }
        }
        ExprKind::Block { stmts, value } => {
            for stmt in stmts {
                match stmt {
                    Stmt::Let { pattern, value, .. } => {
                        assert_pattern_refs_resolved(db, pattern);
                        assert_typed_refs_resolved(db, value);
                    }
                    Stmt::Expr { expr: value, .. } => {
                        assert_typed_refs_resolved(db, value);
                    }
                }
            }
            assert_typed_refs_resolved(db, value);
        }
        ExprKind::Lambda { body, .. } => assert_typed_refs_resolved(db, body),
        ExprKind::Case { scrutinee, arms } => {
            assert_typed_refs_resolved(db, scrutinee);
            for arm in arms {
                assert_pattern_refs_resolved(db, &arm.pattern);
                if let Some(guard) = &arm.guard {
                    assert_typed_refs_resolved(db, guard);
                }
                assert_typed_refs_resolved(db, &arm.body);
            }
        }
        ExprKind::Handle { body, handlers } => {
            assert_typed_refs_resolved(db, body);
            for handler in handlers {
                match &handler.kind {
                    HandlerKind::Do { binding } => assert_pattern_refs_resolved(db, binding),
                    HandlerKind::Fn {
                        ability, params, ..
                    }
                    | HandlerKind::Op {
                        ability, params, ..
                    } => {
                        assert!(!type_contains_univar(db, ability.ty));
                        for param in params {
                            assert_pattern_refs_resolved(db, param);
                        }
                    }
                }
                assert_typed_refs_resolved(db, &handler.body);
            }
        }
        ExprKind::Resume { arg, .. } => assert_typed_refs_resolved(db, arg),
        ExprKind::Tuple(values) | ExprKind::List(values) => {
            for value in values {
                assert_typed_refs_resolved(db, value);
            }
        }
        ExprKind::BinOp { lhs, rhs, .. } => {
            assert_typed_refs_resolved(db, lhs);
            assert_typed_refs_resolved(db, rhs);
        }
        _ => {}
    }
}

fn assert_pattern_refs_resolved(
    db: &dyn salsa::Database,
    pattern: &Pattern<tribute_front::ast::TypedRef<'_>>,
) {
    match &*pattern.kind {
        PatternKind::Variant { ctor, fields } => {
            assert!(!type_contains_univar(db, ctor.ty));
            for field in fields {
                assert_pattern_refs_resolved(db, field);
            }
        }
        PatternKind::Record {
            type_name, fields, ..
        } => {
            if let Some(type_name) = type_name {
                assert!(!type_contains_univar(db, type_name.ty));
            }
            for field in fields {
                if let Some(pattern) = &field.pattern {
                    assert_pattern_refs_resolved(db, pattern);
                }
            }
        }
        PatternKind::Tuple(patterns) | PatternKind::List(patterns) => {
            for pattern in patterns {
                assert_pattern_refs_resolved(db, pattern);
            }
        }
        PatternKind::ListRest { head, .. } => {
            for pattern in head {
                assert_pattern_refs_resolved(db, pattern);
            }
        }
        PatternKind::As { pattern, .. } => assert_pattern_refs_resolved(db, pattern),
        PatternKind::Wildcard
        | PatternKind::Bind { .. }
        | PatternKind::Literal(_)
        | PatternKind::Error => {}
    }
}

#[salsa_test]
fn local_generalization_stays_scoped(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability Signal {
    fn tick(value: Nat) -> Nil
}

fn local_identity_and_handler() -> Nil {
    let identity = fn(value) value
    handle Signal::tick(identity(1)) {
        do result { result }
        fn Signal::tick(value) { Nil }
    }
}
"#,
    );

    let errors = type_errors(db, source);
    assert!(errors.is_empty(), "unexpected type errors: {errors:#?}");

    let output = tribute_front::query::type_check_output(db, source)
        .expect("type checking should produce output");
    let scheme = output
        .function_types(db)
        .iter()
        .find(|(name, _)| name == "local_identity_and_handler")
        .map(|(_, scheme)| *scheme)
        .expect("function scheme should be present");
    assert!(
        scheme.type_params(db).is_empty(),
        "a body-local identity must not add a function scheme parameter"
    );
    assert!(!type_contains_univar(db, scheme.body(db)));

    let metadata = output.expression_types(db);
    for (node, ty) in &metadata.node_types {
        assert!(
            !type_contains_univar(db, *ty),
            "node type {node:?} leaked raw UniVar: {:?}",
            ty.kind(db)
        );
    }
    for (_, ty) in &metadata.call_callee_types {
        assert!(
            !type_contains_univar(db, *ty),
            "callee type leaked raw UniVar: {ty}"
        );
    }
    for (_, operation) in output.handler_operations(db) {
        for ty in operation
            .ability_args
            .iter()
            .chain(operation.params.iter())
            .chain([&operation.result])
        {
            assert!(
                !type_contains_univar(db, *ty),
                "handler metadata leaked raw UniVar: {ty}"
            );
        }
    }
    for (_, operation) in output.perform_operations(db) {
        for ty in operation
            .ability_args
            .iter()
            .chain(operation.params.iter())
            .chain([&operation.result])
        {
            assert!(
                !type_contains_univar(db, *ty),
                "perform metadata leaked raw UniVar: {ty}"
            );
        }
    }
    for (_, signature) in output.lambda_signatures(db) {
        assert!(
            !type_contains_univar(db, signature.function_type),
            "lambda metadata leaked raw UniVar: {}",
            signature.function_type
        );
    }

    let function = output
        .module(db)
        .decls
        .iter()
        .find_map(|decl| match decl {
            Decl::Function(function) if function.name == "local_identity_and_handler" => {
                Some(function)
            }
            _ => None,
        })
        .expect("typed function should be present");
    assert_typed_refs_resolved(db, &function.body);
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
fn tuple_local_generalizations_have_distinct_lexical_owners(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn pair() -> #(Nat, Bool) {
    let #(left, right) = #(fn(value) value, fn(value) value)
    #(left(1), right(True))
}
"#,
    );

    let errors = type_errors(db, source);
    assert!(errors.is_empty(), "unexpected type errors: {errors:#?}");

    let output = tribute_front::query::type_check_output(db, source)
        .expect("type checking should produce output");
    let mut scopes = Vec::new();
    for (_, signature) in output.lambda_signatures(db) {
        collect_local_bound_scopes(db, signature.function_type, &mut scopes);
    }
    scopes.sort_by_key(|scope| scope.raw());
    scopes.dedup();
    assert_eq!(
        scopes.len(),
        2,
        "independent destructured local schemes must retain distinct owners: {scopes:?}"
    );
}

#[salsa_test]
fn unresolved_body_only_variable_is_diagnosed_and_recovered(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn ambiguous() -> Nil {
    let identity = fn(value) value
    identity(1)
    []
    Nil
}
"#,
    );

    let errors = type_errors(db, source);
    assert!(
        errors
            .iter()
            .any(|error| error.contains("cannot infer a concrete type for")),
        "expected a body-local ambiguity diagnostic, got: {errors:#?}"
    );

    let output = tribute_front::query::type_check_output(db, source)
        .expect("error recovery should still produce typed output");
    let function = output
        .module(db)
        .decls
        .iter()
        .find_map(|decl| match decl {
            Decl::Function(function) if function.name == "ambiguous" => Some(function),
            _ => None,
        })
        .expect("recovered typed function should be present");
    assert_typed_refs_resolved(db, &function.body);
    for (_, signature) in output.lambda_signatures(db) {
        assert!(
            !type_contains_univar(db, signature.function_type),
            "recovered lambda metadata leaked raw UniVar: {}",
            signature.function_type
        );
    }
}

#[salsa_test]
fn multi_parameter_local_scheme_preserves_distinct_indices(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn pairs() -> #(#(Nat, Bool), #(Bool, Nat)) {
    let pair = fn(left, right) #(left, right)
    #(pair(1, True), pair(True, 1))
}
"#,
    );

    let errors = type_errors(db, source);
    assert!(errors.is_empty(), "unexpected type errors: {errors:#?}");

    let output = tribute_front::query::type_check_output(db, source)
        .expect("type checking should produce output");
    let mut bounds = Vec::new();
    for (_, signature) in output.lambda_signatures(db) {
        collect_local_bounds(db, signature.function_type, &mut bounds);
    }
    bounds.sort_by_key(|(scope, index)| (scope.raw(), *index));
    bounds.dedup();
    assert!(
        bounds
            .windows(2)
            .any(|pair| pair[0].0 == pair[1].0 && pair[0].1 != pair[1].1),
        "a multi-parameter local scheme must retain its distinct bound indices: {bounds:?}"
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
