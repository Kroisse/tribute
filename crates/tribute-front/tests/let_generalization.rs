use salsa_test_macros::salsa_test;
use tribute_core::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use tribute_front::{
    SourceCst,
    ast::{
        AbilityId, CallingConvention, Decl, Effect, EffectRow, Expr, ExprKind, Module, NodeId,
        Type, TypeKind, TypedRef, UniVarId, UniVarSource,
    },
};
use trunk_ir::Symbol;

fn type_contains<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    predicate: impl Fn(Type<'db>) -> bool + Copy,
) -> bool {
    if predicate(ty) {
        return true;
    }
    match ty.kind(db) {
        TypeKind::Named { args, .. } => args.iter().any(|arg| type_contains(db, *arg, predicate)),
        TypeKind::Func {
            params,
            result,
            effect,
            ..
        } => {
            params
                .iter()
                .any(|param| type_contains(db, *param, predicate))
                || type_contains(db, *result, predicate)
                || effect
                    .effects(db)
                    .iter()
                    .flat_map(|effect| effect.args.iter())
                    .any(|arg| type_contains(db, *arg, predicate))
        }
        TypeKind::Tuple(elements) => elements
            .iter()
            .any(|element| type_contains(db, *element, predicate)),
        TypeKind::App { ctor, args } => {
            type_contains(db, *ctor, predicate)
                || args.iter().any(|arg| type_contains(db, *arg, predicate))
        }
        TypeKind::Continuation {
            arg,
            result,
            effect,
        } => {
            type_contains(db, *arg, predicate)
                || type_contains(db, *result, predicate)
                || effect
                    .effects(db)
                    .iter()
                    .flat_map(|effect| effect.args.iter())
                    .any(|arg| type_contains(db, *arg, predicate))
        }
        _ => false,
    }
}

fn type_contains_univar(db: &dyn salsa::Database, ty: Type<'_>) -> bool {
    type_contains(db, ty, |ty| matches!(ty.kind(db), TypeKind::UniVar { .. }))
}

fn type_contains_out_of_scope_bound(db: &dyn salsa::Database, ty: Type<'_>, arity: u32) -> bool {
    type_contains(
        db,
        ty,
        |ty| matches!(ty.kind(db), TypeKind::BoundVar { index } if *index >= arity),
    )
}

fn local_call_callee_types<'db>(body: &Expr<TypedRef<'db>>) -> Vec<Type<'db>> {
    let ExprKind::Block { value, .. } = &*body.kind else {
        panic!("test function body must be a block");
    };
    let calls: &[Expr<TypedRef<'db>>] = match &*value.kind {
        ExprKind::Tuple(calls) => calls,
        ExprKind::Call { .. } => std::slice::from_ref(value),
        _ => panic!("test function body must end in local calls"),
    };
    calls
        .iter()
        .map(|call| match &*call.kind {
            ExprKind::Call { callee, .. } => match &*callee.kind {
                ExprKind::Var(typed_ref) => typed_ref.ty,
                _ => panic!("test call callee must be a local reference"),
            },
            _ => panic!("test body must contain calls"),
        })
        .collect()
}

fn body_node_ids(expr: &Expr<TypedRef<'_>>, ids: &mut Vec<NodeId>) {
    ids.push(expr.id);
    match &*expr.kind {
        ExprKind::Block { stmts, value } => {
            for stmt in stmts {
                match stmt {
                    tribute_front::ast::Stmt::Let { value, .. }
                    | tribute_front::ast::Stmt::Expr { expr: value, .. } => {
                        body_node_ids(value, ids);
                    }
                }
            }
            body_node_ids(value, ids);
        }
        ExprKind::Lambda { body, .. } => body_node_ids(body, ids),
        ExprKind::Tuple(elements) | ExprKind::List(elements) => {
            for element in elements {
                body_node_ids(element, ids);
            }
        }
        ExprKind::Call { callee, args } => {
            body_node_ids(callee, ids);
            for arg in args {
                body_node_ids(arg, ids);
            }
        }
        _ => {}
    }
}

fn func_ref_has_param_and_result(
    db: &dyn salsa::Database,
    ty: Type<'_>,
    expected: Type<'_>,
) -> bool {
    matches!(ty.kind(db), TypeKind::Func { params, result, .. }
        if params.as_slice() == [expected] && *result == expected)
}

fn is_local_identity_signature(db: &dyn salsa::Database, ty: Type<'_>) -> bool {
    local_identity_scope(db, ty).is_some()
}

fn local_identity_scope(db: &dyn salsa::Database, ty: Type<'_>) -> Option<NodeId> {
    let TypeKind::Func { params, result, .. } = ty.kind(db) else {
        return None;
    };
    let [param] = params.as_slice() else {
        return None;
    };
    matches!(
        (param.kind(db), result.kind(db)),
        (
            TypeKind::LocalBoundVar { scope, index: 0 },
            TypeKind::LocalBoundVar {
                scope: result_scope,
                index: 0,
            },
        ) if scope == result_scope
    )
    .then(|| match param.kind(db) {
        TypeKind::LocalBoundVar { scope, .. } => *scope,
        _ => unreachable!(),
    })
}

fn typed_function_body<'a, 'db>(
    module: &'a Module<TypedRef<'db>>,
    name: Symbol,
) -> &'a Expr<TypedRef<'db>> {
    module
        .decls
        .iter()
        .find_map(|decl| match decl {
            Decl::Function(function) if function.name == name => Some(&function.body),
            _ => None,
        })
        .unwrap_or_else(|| panic!("missing typed function body for {name}"))
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
fn local_callable_instantiations_keep_their_owner_after_solving(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn pair() -> #(Nat, Bool) {
    let identity = fn(value) value
    #(identity(1), identity(True))
}

fn pass(value: a) -> a {
    let identity = fn(other) other
    identity(value)
}
"#,
    );

    let output = tribute_front::query::type_check_output(db, source)
        .expect("type checking should produce output");
    let scheme = |name| {
        output
            .function_types(db)
            .iter()
            .find_map(|(candidate, scheme)| (*candidate == Symbol::new(name)).then_some(*scheme))
            .unwrap_or_else(|| panic!("missing function scheme for {name}"))
    };

    let pair_scheme = scheme("pair");
    let pass_scheme = scheme("pass");
    assert_eq!(pair_scheme.type_params(db).len(), 0);
    assert_eq!(pass_scheme.type_params(db).len(), 1);
    assert!(
        !type_contains_out_of_scope_bound(db, pair_scheme.body(db), 0),
        "pair must not acquire enclosing function binders"
    );
    assert!(
        !type_contains_out_of_scope_bound(db, pass_scheme.body(db), 1),
        "pass scheme must not contain phantom binders"
    );
    assert!(matches!(
        pass_scheme.body(db).kind(db),
        TypeKind::Func { params, result, .. }
            if matches!(params.as_slice(), [param] if matches!(param.kind(db), TypeKind::BoundVar { index: 0 }))
                && matches!(result.kind(db), TypeKind::BoundVar { index: 0 })
    ));
    for ty in output
        .function_types(db)
        .iter()
        .map(|(_, scheme)| scheme.body(db))
        .chain(
            output
                .expression_types(db)
                .node_types
                .iter()
                .map(|(_, ty)| *ty),
        )
        .chain(
            output
                .expression_types(db)
                .call_callee_types
                .iter()
                .map(|(_, ty)| *ty),
        )
        .chain(
            output
                .lambda_signatures(db)
                .iter()
                .map(|(_, signature)| signature.function_type),
        )
    {
        assert!(
            !type_contains_univar(db, ty),
            "typed body and metadata must not retain raw solver variables: {ty:?}"
        );
    }

    let module = output.module(db);
    for (function, name, arity) in [
        (Symbol::new("pair"), "pair", 0),
        (Symbol::new("pass"), "pass", 1),
    ] {
        let mut nodes = Vec::new();
        body_node_ids(typed_function_body(module, function), &mut nodes);
        for node in &nodes {
            let ty = output
                .expression_types(db)
                .node_types
                .iter()
                .find_map(|(candidate, ty)| (*candidate == *node).then_some(*ty))
                .unwrap_or_else(|| panic!("missing typed metadata for {name} node {node:?}"));
            assert!(
                !type_contains_out_of_scope_bound(db, ty, arity),
                "{name} node metadata exceeds its scheme arity {arity}: {ty:?}"
            );
        }
        for (lambda, signature) in output.lambda_signatures(db) {
            if nodes.contains(lambda) {
                assert!(
                    !type_contains_out_of_scope_bound(db, signature.function_type, arity),
                    "{name} lambda metadata exceeds its scheme arity {arity}"
                );
            }
        }
    }
    let pair_identities = local_call_callee_types(typed_function_body(module, Symbol::new("pair")));
    assert!(
        pair_identities
            .iter()
            .any(|ty| func_ref_has_param_and_result(db, *ty, Type::new(db, TypeKind::Nat))),
        "identity(1) must have a concrete Nat callable type"
    );
    assert!(
        pair_identities
            .iter()
            .any(|ty| func_ref_has_param_and_result(db, *ty, Type::new(db, TypeKind::Bool))),
        "identity(True) must have a concrete Bool callable type"
    );
    let pass_identities = local_call_callee_types(typed_function_body(module, Symbol::new("pass")));
    assert!(
        pass_identities.iter().any(|ty| {
            matches!(ty.kind(db), TypeKind::Func { params, result, .. }
                if matches!(params.as_slice(), [param] if matches!(param.kind(db), TypeKind::BoundVar { index: 0 }))
                    && matches!(result.kind(db), TypeKind::BoundVar { index: 0 }))
        }),
        "identity(value) must alias the enclosing function binder"
    );
    assert!(
        output
            .lambda_signatures(db)
            .iter()
            .all(|(_, signature)| is_local_identity_signature(db, signature.function_type)),
        "each identity definition must retain one LocalBoundVar owner"
    );
}

#[salsa_test]
fn effectful_local_callable_metadata_generalizes_effect_arguments(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability Source(s) {
    op get() -> s
}

fn pair() -> Nil {
    let read = fn() Source::get()
    Nil
}
"#,
    );

    let output = tribute_front::query::type_check_output(db, source)
        .expect("type checking should produce output");
    let (scope, signature) = output
        .lambda_signatures(db)
        .iter()
        .find(|(_, signature)| {
            matches!(signature.function_type.kind(db), TypeKind::Func { effect, .. } if !effect.effects(db).is_empty())
        })
        .expect("effectful local lambda signature should be present");
    let TypeKind::Func { effect, .. } = signature.function_type.kind(db) else {
        unreachable!("selected lambda signature must be a function");
    };
    assert!(
        effect
            .effects(db)
            .iter()
            .flat_map(|effect| effect.args.iter())
            .any(|arg| matches!(arg.kind(db), TypeKind::LocalBoundVar { .. })),
        "the lambda effect row must retain its local quantifier provenance"
    );
    let local = Type::new(
        db,
        TypeKind::LocalBoundVar {
            scope: *scope,
            index: 0,
        },
    );
    let univar = Type::new(
        db,
        TypeKind::UniVar {
            id: UniVarId::new(db, UniVarSource::Anonymous(1), 0),
        },
    );
    let effect = EffectRow::single(
        db,
        Effect {
            ability_id: AbilityId::source(db, Symbol::new("Source")),
            args: vec![local, univar],
        },
    );
    let nil = Type::new(db, TypeKind::Nil);
    let function = Type::new(
        db,
        TypeKind::Func {
            params: vec![],
            result: nil,
            effect,
            minimum_convention: CallingConvention::Direct,
        },
    );
    let continuation = Type::new(
        db,
        TypeKind::Continuation {
            arg: nil,
            result: nil,
            effect,
        },
    );
    for ty in [function, continuation] {
        assert!(type_contains_univar(db, ty));
        assert!(type_contains(db, ty, |entry| {
            matches!(entry.kind(db), TypeKind::LocalBoundVar { .. })
        }));
    }
}

#[salsa_test]
fn resume_lambda_metadata_uses_the_enclosing_handler_effect(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "resume_lambda_effect.trb",
        r#"
ability Writer(w) {
    op tell(value: w) -> Nil
}

fn run_writer(comp: fn() ->{e, Writer(w)} a) ->{e} a {
    handle comp() {
        do result { result }
        op Writer::tell(v) { run_writer(fn() { resume Nil }) }
    }
}
"#,
    );

    let output = tribute_front::query::type_check_output(db, source)
        .expect("type checking should produce output");
    let run_writer = output
        .function_types(db)
        .iter()
        .find_map(|(name, scheme)| (*name == Symbol::new("run_writer")).then_some(*scheme))
        .expect("run_writer scheme should be present");
    let TypeKind::Func { params, .. } = run_writer.body(db).kind(db) else {
        panic!("run_writer scheme must be callable");
    };
    let TypeKind::Func {
        effect: computation_effect,
        ..
    } = params[0].kind(db)
    else {
        panic!("run_writer computation parameter must be callable");
    };

    let signatures: Vec<_> = output
        .lambda_signatures(db)
        .iter()
        .map(|(_, signature)| signature.clone())
        .collect();
    assert_eq!(signatures.len(), 1, "fixture has one resume lambda");
    let TypeKind::Func {
        effect: lambda_effect,
        ..
    } = signatures[0].function_type.kind(db)
    else {
        panic!("resume lambda metadata must be callable");
    };
    assert_eq!(
        lambda_effect, computation_effect,
        "the resume lambda must preserve the enclosing handler computation effect"
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
