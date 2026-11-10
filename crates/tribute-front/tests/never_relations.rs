//! Frontend contracts for `Never` equality, coercion, and common results.

mod common;

use self::common::{ast_pipeline_diagnostics, ast_pipeline_error_messages};
use salsa_test_macros::salsa_test;
use tribute_front::{
    SourceCst,
    ast::{Decl, ExprKind, Type, TypeKind},
    resolve::{NameDisplay, PackageName},
};

fn names() -> NameDisplay {
    NameDisplay::new(&PackageName::executable_root(), [])
}

fn type_errors(db: &salsa::DatabaseImpl, name: &str, text: &str) -> Vec<String> {
    ast_pipeline_error_messages(db, SourceCst::from_source_str(db, name, text))
}

/// Return the recorded type for a function's final expression.  Fixtures in
/// this file deliberately use a direct block tail, so this stays focused on
/// the relation under test rather than walking arbitrary expressions.
fn function_tail_type<'db>(
    db: &'db salsa::DatabaseImpl,
    source: SourceCst,
    function_name: &str,
) -> Type<'db> {
    let checked = tribute_front::query::type_check_output(db, source)
        .expect("type checking should produce output");
    let body = checked
        .module(db)
        .decls()
        .find_map(|decl| match decl {
            Decl::Function(function) if function.name.with_str(|name| name == function_name) => {
                Some(&function.body)
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("missing function {function_name}"));
    let ExprKind::Block { value, .. } = &*body.kind else {
        panic!("{function_name} must have a block body");
    };
    checked
        .expression_types(db)
        .node_types
        .iter()
        .find_map(|(id, ty)| (*id == value.id).then_some(*ty))
        .unwrap_or_else(|| panic!("missing tail type for {function_name}"))
}

#[salsa_test]
fn ordinary_values_cannot_satisfy_never(db: &salsa::DatabaseImpl) {
    for (name, body) in [
        ("result", "fn bad() -> Never { 42 }"),
        (
            "argument",
            "fn take(x: Never) -> Nil { Nil } fn bad() -> Nil { take(42) }",
        ),
    ] {
        let errors = type_errors(db, name, body);
        assert!(
            errors
                .iter()
                .any(|error| error.contains("expected `Never`, found `Nat`")),
            "{name}: {errors:#?}"
        );
    }
}

#[salsa_test]
fn never_is_not_recursive_subtyping(db: &salsa::DatabaseImpl) {
    for (index, (actual, expected)) in [
        ("List(Never)", "List(Nat)"),
        ("List(Nat)", "List(Never)"),
        ("Box(Never)", "Box(Nat)"),
        ("Box(Nat)", "Box(Never)"),
        ("#(Never, Bool)", "#(Nat, Bool)"),
        ("#(Nat, Bool)", "#(Never, Bool)"),
        ("fn() -> Never", "fn() -> Nat"),
        ("fn() -> Nat", "fn() -> Never"),
        ("fn(Never) -> Nat", "fn(Nat) -> Nat"),
        ("fn(Nat) -> Nat", "fn(Never) -> Nat"),
    ]
    .into_iter()
    .enumerate()
    {
        let errors = type_errors(
            db,
            &format!("recursive_{index}.trb"),
            &format!(
                "struct Box(a) {{ value: a }}\nfn bad(value: {actual}) -> {expected} {{ value }}"
            ),
        );
        assert!(
            errors
                .iter()
                .any(|error| error.contains("type error") && error.contains("Never")),
            "{actual} -> {expected}: {errors:#?}"
        );
    }
}

#[salsa_test]
fn never_does_not_match_effect_type_arguments(db: &salsa::DatabaseImpl) {
    let errors = type_errors(
        db,
        "never_does_not_match_effect_type_arguments.trb",
        r#"
ability Mark(a) { op mark() -> Nil }

fn takes_never_mark(thunk: fn() ->{Mark(Never)} Nil) -> Nil { Nil }
fn nat_mark() ->{Mark(Nat)} Nil { Mark::mark() }
fn bad() -> Nil { takes_never_mark(nat_mark) }
"#,
    );

    assert!(
        !errors.is_empty(),
        "effect type arguments must use exact recursive equality"
    );
}

#[salsa_test]
fn top_level_never_eliminates_in_expression_contexts(db: &salsa::DatabaseImpl) {
    let errors = type_errors(
        db,
        "top_level_never_eliminates_in_expression_contexts.trb",
        r#"
ability Stop { op stop() -> Never }
struct Box { value: Nat }

fn abort() ->{Stop} Never { Stop::stop() }
fn takes_nat(value: Nat) -> Nat { value }
fn invoke(thunk: fn() ->{e} Nat) ->{e} Nat { thunk() }
fn return_context() ->{Stop} Nat { abort() }
fn factory() -> fn() ->{Stop} Nat { fn() { abort() } }

fn valid() ->{Stop} Nat {
    let inferred = abort()
    takes_nat(inferred)
    takes_nat(abort())
    Box { value: abort() }
    invoke(fn() { abort() })
}

"#,
    );

    assert!(
        errors.is_empty(),
        "top-level Never should eliminate: {errors:#?}"
    );
}

#[salsa_test]
fn list_elements_join_never_and_keep_empty_elements_fresh(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "list_elements_join_never_and_keep_empty_elements_fresh.trb",
        r#"
ability Stop { op stop() -> Never }
fn abort() ->{Stop} Never { Stop::stop() }

fn mixed() ->{Stop} List(Nat) { [abort(), 1] }
fn reversed() ->{Stop} List(Nat) { [1, abort()] }
fn all_never() ->{Stop} List(Never) { [abort(), abort()] }
fn empty() -> List(a) { [] }
"#,
    );

    let errors = ast_pipeline_error_messages(db, source);
    assert!(
        errors.is_empty(),
        "list joins should accept top-level Never: {errors:#?}"
    );
    let mixed = function_tail_type(db, source, "mixed");
    assert!(
        matches!(
            mixed.kind(db),
            TypeKind::Named { name, args, .. }
                if *name == "List"
                    && matches!(args.as_slice(), [arg] if matches!(arg.kind(db), TypeKind::Nat))
        ),
        "the non-Never list element must determine List(Nat), got {}",
        mixed.display(db, &names())
    );
    let empty = function_tail_type(db, source, "empty");
    assert!(
        matches!(
            empty.kind(db),
            TypeKind::Named { name, args, .. }
                if *name == "List"
                    && matches!(args.as_slice(), [arg] if matches!(
                        arg.kind(db),
                        TypeKind::UniVar { .. }
                            | TypeKind::BoundVar { .. }
                            | TypeKind::LocalBoundVar { .. }
                    ))
        ),
        "an empty list must retain a fresh element type, got {}",
        empty.display(db, &names())
    );
    for (name, is_never) in [("reversed", false), ("all_never", true)] {
        let TypeKind::Named { args, .. } = function_tail_type(db, source, name).kind(db) else {
            panic!("expected list");
        };
        assert_eq!(
            matches!(args[0].kind(db), TypeKind::Never),
            is_never,
            "{name}"
        );
    }
}

#[salsa_test]
fn case_common_results_are_order_independent(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "case_common_results_are_order_independent.trb",
        r#"
ability Stop { op stop() -> Never }
enum Choice { A, B, C }
fn abort() ->{Stop} Never { Stop::stop() }

fn nat_first(choice: Choice) ->{Stop} Nat {
    case choice {
        .A -> 1
        .B -> abort()
        .C -> 2
    }
}

fn never_first(choice: Choice) ->{Stop} Nat {
    case choice {
        .A -> abort()
        .B -> 1
        .C -> 2
    }
}

fn nil_first(choice: Bool) ->{Stop} Nil {
    case choice {
        True -> Nil
        False -> abort()
    }
}

fn never_before_nil(choice: Bool) ->{Stop} Nil {
    case choice {
        True -> abort()
        False -> Nil
    }
}

fn all_never(choice: Bool) ->{Stop} Never {
    case choice {
        True -> abort()
        False -> abort()
    }
}
"#,
    );
    let errors = ast_pipeline_error_messages(db, source);

    assert!(
        errors.is_empty(),
        "Never case joins should be order-independent: {errors:#?}"
    );
    let all_never = function_tail_type(db, source, "all_never");
    assert!(
        matches!(all_never.kind(db), TypeKind::Never),
        "all-Never case arms must retain Never, got {}",
        all_never.display(db, &names())
    );
    for name in ["nat_first", "never_first"] {
        assert!(
            matches!(function_tail_type(db, source, name).kind(db), TypeKind::Nat),
            "{name}"
        );
    }
    for name in ["nil_first", "never_before_nil"] {
        assert!(
            matches!(function_tail_type(db, source, name).kind(db), TypeKind::Nil),
            "{name}"
        );
    }
}

#[salsa_test]
fn reachable_case_mismatches_still_fail(db: &salsa::DatabaseImpl) {
    let errors = type_errors(
        db,
        "reachable_case_mismatches_still_fail.trb",
        r#"
ability Stop { op stop() -> Never }
enum Choice { A, B, C }
fn abort() ->{Stop} Never { Stop::stop() }

fn bad(choice: Choice) ->{Stop} Nat {
    case choice {
        .A -> abort()
        .B -> 1
        .C -> True
    }
}

fn never_in_middle(choice: Choice) ->{Stop} Nat {
    case choice {
        .A -> 1
        .B -> abort()
        .C -> True
    }
}

fn never_last(choice: Choice) ->{Stop} Nat {
    case choice {
        .A -> 1
        .B -> True
        .C -> abort()
    }
}
"#,
    );

    assert!(
        errors.len() >= 3,
        "each reachable Never/Nat/Bool arm order must fail, got {errors:#?}"
    );
}

#[salsa_test]
fn deferred_ufcs_call_keeps_its_actual_never_result(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "deferred_ufcs_call_keeps_its_actual_never_result.trb",
        r#"
struct Bomb {}

pub mod Bomb {
    pub extern "intrinsic" fn explode(value: Bomb) -> Never
}

fn after(bomb: Bomb) -> Nat { bomb.explode() }

fn deferred_join() -> Nat {
    let choose = fn(bomb) {
        case True {
            True -> bomb.explode()
            False -> 42
        }
    }
    choose(Bomb {})
}
"#,
    );

    let errors = ast_pipeline_error_messages(db, source);
    assert!(
        errors.is_empty(),
        "a deferred Never result should eliminate: {errors:#?}"
    );
    let actual = function_tail_type(db, source, "after");
    assert!(
        matches!(actual.kind(db), TypeKind::Never),
        "the deferred method call must retain actual Never before contextual coercion, got {}",
        actual.display(db, &names())
    );
}

/// A pure `let` quantifies no data type: whether its value is a lambda, a
/// join of lambdas, or a component of a destructured tuple, its uses at
/// `Nat` and `Bool` are rejected and its uses at `Nat` alone are accepted.
#[salsa_test]
fn pure_let_is_monomorphic_in_data_types(db: &salsa::DatabaseImpl) {
    let program = |second: &str| {
        format!(
            r#"
fn pair() -> #(Nat, {second}) {{
    let identity = fn(value) value
    #(identity(1), identity({second}::VALUE))
}}

fn generic_join() -> #(Nat, {second}) {{
    let select = fn(flag, value) {{
        case flag {{
            True -> value
            False -> value
        }}
    }}
    #(select(True, 1), select(False, {second}::VALUE))
}}

struct Bomb {{}}
pub mod Bomb {{ pub extern "intrinsic" fn explode(value: Bomb) -> Never }}

fn partial_generalization() -> #(Nat, {second}) {{
    let #(explode, identity) = #(fn(bomb) bomb.explode(), fn(value) value)
    explode(Bomb {{}})
    #(identity(1), identity({second}::VALUE))
}}
"#
        )
    };
    let rejected = program("Bool").replace("Bool::VALUE", "True");
    let errors = type_errors(db, "pure_let_rejected.trb", &rejected);
    assert_eq!(
        errors
            .iter()
            .filter(|error| error.contains("expected `Nat`, found `Bool`")
                || error.contains("expected `Bool`, found `Nat`"))
            .count(),
        3,
        "each function shares one data type: {errors:#?}"
    );
    let accepted = program("Nat").replace("Nat::VALUE", "2");
    let errors = type_errors(db, "pure_let_accepted.trb", &accepted);
    assert!(errors.is_empty(), "{errors:#?}");
}

#[salsa_test]
fn effectful_let_is_not_generalized(db: &salsa::DatabaseImpl) {
    let errors = type_errors(
        db,
        "effectful_let.trb",
        r#"
ability Choose { op choose() -> Bool }
fn bad() ->{Choose} #(Nat, Bool) {
    let identity = {
        Choose::choose()
        fn(value) value
    }
    #(identity(1), identity(True))
}

"#,
    );
    assert!(
        errors.iter().any(|error| error.contains("type error")),
        "{errors:#?}"
    );
}

#[salsa_test]
fn every_three_way_case_and_list_mismatch_is_rejected(db: &salsa::DatabaseImpl) {
    for (index, sources) in [
        ["abort()", "1", "True"],
        ["abort()", "True", "1"],
        ["1", "abort()", "True"],
        ["1", "True", "abort()"],
        ["True", "abort()", "1"],
        ["True", "1", "abort()"],
    ]
    .into_iter()
    .enumerate()
    {
        for is_list in [false, true] {
            let expression = if is_list {
                format!("[{}, {}, {}]", sources[0], sources[1], sources[2])
            } else {
                format!(
                    "case choice {{
 A -> {}
 B -> {}
 C -> {}
 }}",
                    sources[0], sources[1], sources[2]
                )
            };
            let source = format!(
                "ability Stop {{ op stop() -> Never }}\nenum Choice {{ A, B, C }}\nfn abort() ->{{Stop}} Never {{ Stop::stop() }}\nfn bad(choice: Choice) ->{{Stop}} {} {{ {expression} }}",
                if is_list { "List(Nat)" } else { "Nat" }
            );
            let errors = type_errors(db, &format!("order_{index}_{is_list}.trb"), &source);
            assert!(
                errors.iter().any(|error| error.contains("type error")
                    && error.contains("Nat")
                    && error.contains("Bool")),
                "{source}\n{errors:#?}"
            );
        }
    }
}

#[salsa_test]
fn handle_join_separates_aborting_results_from_resume_answers(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "handle_join_accepts_never_do_path_and_resumptive_answer.trb",
        r#"
ability Stop { op stop() -> Never }
ability Ask { op ask() -> Nat }

fn do_never() ->{Stop} Never { Stop::stop() }

fn handler_nat() -> Nat {
    handle Stop::stop() {
        do result { result }
        op Stop::stop() { 42 }
    }
}

fn resume_only() ->{Stop} Never {
    handle Ask::ask() {
        do result { do_never() }
        op Ask::ask() { resume 42 }
    }
}

fn resume_or_abort(flag: Bool) ->{Stop} Nat {
    handle Ask::ask() {
        do result { do_never() }
        op Ask::ask() {
            case flag {
                True -> resume 42
                False -> 7
            }
        }
    }
}

fn nested_resume_only(flag: Bool) ->{Stop} Never {
    handle Ask::ask() {
        do result { do_never() }
        op Ask::ask() {
            case flag {
                True -> resume 42
                False -> resume 7
            }
        }
    }
}
"#,
    );

    let errors = ast_pipeline_error_messages(db, source);
    assert!(
        errors.is_empty(),
        "Never do-path and resumptive answer must form a Nat handle result: {errors:#?}"
    );
    let result = function_tail_type(db, source, "handler_nat");
    assert!(
        matches!(result.kind(db), TypeKind::Nat),
        "the handle answer should be Nat, got {}",
        result.display(db, &names())
    );
    assert!(matches!(
        function_tail_type(db, source, "resume_only").kind(db),
        TypeKind::Never
    ));
    assert!(matches!(
        function_tail_type(db, source, "resume_or_abort").kind(db),
        TypeKind::Nat
    ));
    assert!(matches!(
        function_tail_type(db, source, "nested_resume_only").kind(db),
        TypeKind::Never
    ));
}

#[salsa_test]
fn callable_metadata_preserves_context_and_actual_never_body(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "lambda_metadata.trb",
        r#"
ability Stop { op stop() -> Never }
fn factory() -> fn() ->{Stop} Nat { fn() { Stop::stop() } }
"#,
    );
    let errors = ast_pipeline_error_messages(db, source);
    assert!(errors.is_empty(), "{errors:#?}");
    let checked = tribute_front::query::type_check_output(db, source).unwrap();
    let metadata = checked.expression_types(db);
    assert_eq!(checked.expression_types(db).lambda_signatures.len(), 1);
    let (lambda_id, signature) = &checked.expression_types(db).lambda_signatures[0];
    let TypeKind::Func { result, .. } = signature.function_type.kind(db) else {
        panic!("expected callable");
    };
    assert!(matches!(result.kind(db), TypeKind::Nat));
    let factory = checked
        .module(db)
        .decls()
        .find_map(|decl| match decl {
            Decl::Function(function) if function.name.with_str(|name| name == "factory") => {
                Some(function)
            }
            _ => None,
        })
        .unwrap();
    let ExprKind::Block { value: lambda, .. } = &*factory.body.kind else {
        panic!("block");
    };
    assert_eq!(lambda.id, *lambda_id);
    let ExprKind::Lambda { body, .. } = &*lambda.kind else {
        panic!("lambda");
    };
    let ExprKind::Block {
        value: operation, ..
    } = &*body.kind
    else {
        panic!("body block");
    };
    let actual = metadata
        .node_types
        .iter()
        .find(|(id, _)| *id == operation.id)
        .unwrap()
        .1;
    assert!(matches!(actual.kind(db), TypeKind::Never));
}

#[salsa_test]
fn separate_deferred_comparisons_keep_both_receivers(db: &salsa::DatabaseImpl) {
    let errors = type_errors(
        db,
        "separate_comparisons.trb",
        r#"
fn compare() -> #(Bool, Bool) {
    let #(nat_eq, float_eq) = #(fn(x) { x == x }, fn(y) { y == y })
    #(nat_eq(1), float_eq(1.0))
}
"#,
    );
    assert!(errors.is_empty(), "{errors:#?}");
}

#[salsa_test]
fn deferred_join_rejects_a_late_inhabited_mismatch(db: &salsa::DatabaseImpl) {
    let errors = type_errors(
        db,
        "late_mismatch.trb",
        r#"
struct Probe {}
pub mod Probe { pub fn answer(value: Probe) -> Bool { True } }
fn bad() -> Nat {
    let choose = fn(probe) {
        case True {
            True -> probe.answer()
            False -> 42
        }
    }
    choose(Probe {})
}
"#,
    );
    assert!(
        errors.iter().any(|error| error.contains("type error")
            && error.contains("Nat")
            && error.contains("Bool")),
        "{errors:#?}"
    );
}

/// A pure join of a callable generalizes its row variables but not its data
/// types: the thunks it runs may perform different effects, but must return
/// one type.
#[salsa_test]
fn a_pure_join_generalizes_callable_row_variables(db: &salsa::DatabaseImpl) {
    let program = |second: &str| {
        format!(
            r#"
ability Mark {{ op mark() -> Nil }}
extern "intrinsic" fn diverge() ->{{}} Never
fn both() ->{{Mark}} #(Nat, Nat) {{
    let apply = case True {{
        True -> diverge()
        False -> fn(thunk) thunk()
    }}
    #(apply(fn() {{ 1 }}), apply(fn() {{
        Mark::mark()
        {second}
    }}))
}}
"#
        )
    };
    let errors = type_errors(db, "generic_callable_join.trb", &program("2"));
    assert!(errors.is_empty(), "{errors:#?}");
    let errors = type_errors(db, "generic_callable_join.trb", &program("True"));
    assert!(
        errors
            .iter()
            .any(|error| error.contains("expected `Nat`, found `Bool`")
                || error.contains("expected `Bool`, found `Nat`")),
        "{errors:#?}"
    );
}

#[salsa_test]
fn source_logical_ir_keeps_never_operations_and_normal_branch_results(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "logical_never.trb",
        r#"
ability Stop { op stop() -> Never }
fn choose(flag: Bool) ->{Stop} Nat {
    case flag {
        True -> Stop::stop()
        False -> 42
    }
}
"#,
    );
    assert!(ast_pipeline_error_messages(db, source).is_empty());
    assert!(matches!(
        function_tail_type(db, source, "choose").kind(db),
        TypeKind::Nat
    ));
    let ir = common::run_ast_pipeline_with_ir(db, source);
    assert!(
        ir.lines()
            .any(|line| line.contains("tribute_control.perform") && line.ends_with("-> core.never")),
        "{ir}"
    );
    assert!(
        !ir.lines()
            .any(|line| line.contains("unrealized_conversion_cast")
                && line.ends_with(": core.never")),
        "normal branch values cannot be converted to Never:\n{ir}"
    );
}

#[salsa_test]
fn fn_handler_metadata_keeps_operation_result_and_actual_never_body(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "fn_handler_metadata.trb",
        r#"
ability Read { fn read() -> Nat }
fn diverge() ->{} Never { diverge() }
fn result() ->{} Nat {
    handle Read::read() {
        do value { value }
        fn Read::read() { diverge() }
    }
}
"#,
    );
    let errors = ast_pipeline_error_messages(db, source);
    assert!(errors.is_empty(), "{errors:#?}");
    let checked = tribute_front::query::type_check_output(db, source).unwrap();
    let (_, operation) = checked
        .expression_types(db)
        .handler_operations
        .iter()
        .find(|(_, operation)| operation.kind == tribute_front::ast::OpDeclKind::Fn)
        .unwrap();
    assert!(matches!(operation.result.kind(db), TypeKind::Nat));
    assert!(matches!(
        function_tail_type(db, source, "result").kind(db),
        TypeKind::Nat
    ));
    let function = checked
        .module(db)
        .decls()
        .find_map(|decl| match decl {
            Decl::Function(function) if function.name.with_str(|name| name == "result") => {
                Some(function)
            }
            _ => None,
        })
        .unwrap();
    let ExprKind::Block { value, .. } = &*function.body.kind else {
        panic!("block");
    };
    let ExprKind::Handle { handlers, .. } = &*value.kind else {
        panic!("handle");
    };
    let handler = handlers
        .iter()
        .find(|handler| matches!(handler.kind, tribute_front::ast::HandlerKind::Fn { .. }))
        .unwrap();
    let (_, actual) = checked
        .expression_types(db)
        .node_types
        .iter()
        .find(|(id, _)| *id == handler.body.id)
        .unwrap();
    assert!(matches!(actual.kind(db), TypeKind::Never));
}

#[salsa_test]
fn deferred_method_contextualizes_literal_callback_results(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "deferred_literal_callback.trb",
        r#"
struct Bomb {}
pub mod Bomb { pub extern "intrinsic" fn explode(value: Bomb) -> Never }
struct Runner {}
pub mod Runner {
    pub fn run(receiver: Runner, callback: fn() ->{e} Nat) ->{e} Nat { callback() }
}
fn late_method() -> Nat {
    let invoke = fn(receiver) {
        receiver.run(fn() { Bomb::explode(Bomb {}) })
    }
    invoke(Runner {})
}
"#,
    );
    let errors = ast_pipeline_error_messages(db, source);
    assert!(errors.is_empty(), "{errors:#?}");
    let checked = tribute_front::query::type_check_output(db, source).unwrap();
    let callbacks: Vec<_> = checked
        .expression_types(db)
        .lambda_signatures
        .iter()
        .filter_map(|(_, signature)| match signature.function_type.kind(db) {
            TypeKind::Func { params, result, .. } if params.is_empty() => Some(*result),
            _ => None,
        })
        .collect();
    assert_eq!(callbacks.len(), 1);
    assert!(matches!(callbacks[0].kind(db), TypeKind::Nat));
    assert!(
        checked
            .expression_types(db)
            .node_types
            .iter()
            .any(|(_, ty)| matches!(ty.kind(db), TypeKind::Never))
    );
}

#[salsa_test]
fn deferred_callback_context_does_not_relax_function_value_equality(db: &salsa::DatabaseImpl) {
    for (name, callback_type, argument, parameter) in [
        (
            "existing",
            "fn() -> Nat",
            "existing",
            "existing: fn() -> Never",
        ),
        ("ordinary", "fn() -> Never", "fn() { 42 }", ""),
    ] {
        let errors = type_errors(
            db,
            name,
            &format!(
                r#"
struct Runner {{}}
pub mod Runner {{
    pub fn run(receiver: Runner, callback: {callback_type}) -> Nat {{ 0 }}
}}
fn late_method({parameter}) -> Nat {{
    let invoke = fn(receiver) {{ receiver.run({argument}) }}
    invoke(Runner {{}})
}}
"#
            ),
        );
        assert!(
            errors
                .iter()
                .any(|error| error.contains("type error") && error.contains("Never")),
            "{name}: {errors:#?}"
        );
    }
}

#[salsa_test]
fn deferred_method_context_reaches_nested_literal_results(db: &salsa::DatabaseImpl) {
    let errors = type_errors(
        db,
        "nested_deferred_callback.trb",
        r#"
struct Bomb {}
pub mod Bomb { pub extern "intrinsic" fn explode(value: Bomb) -> Never }
struct Runner {}
pub mod Runner {
    pub fn run(receiver: Runner, callback: fn() ->{e} fn() ->{e} Nat) ->{e} Nat {
        let f = callback()
        f()
    }
}
fn late_method() -> Nat {
    let invoke = fn(receiver) {
        receiver.run(fn() { fn() { Bomb::explode(Bomb {}) } })
    }
    invoke(Runner {})
}
"#,
    );
    assert!(errors.is_empty(), "{errors:#?}");
}

#[salsa_test]
fn never_operation_arm_rejects_resume(db: &salsa::DatabaseImpl) {
    for (name, arm) in [
        ("direct", "op Stop::stop() { resume 0 }"),
        ("nested", "op Stop::stop() { let k = fn() { resume 0 }; 0 }"),
    ] {
        let errors = type_errors(
            db,
            name,
            &format!(
                r#"
ability Stop {{ op stop() -> Never }}

fn test() -> Nat {{
    handle Stop::stop() {{
        do result {{ 0 }}
        {arm}
    }}
}}
"#
            ),
        );
        let resume_errors = errors
            .iter()
            .filter(|error| error.contains("cannot `resume`") && error.contains("`Stop::stop`"))
            .count();
        assert_eq!(
            resume_errors, 1,
            "{name}: resume in a Never operation arm must be reported once: {errors:#?}"
        );
    }
}

#[salsa_test]
fn never_operation_arm_without_resume_is_valid(db: &salsa::DatabaseImpl) {
    let errors = type_errors(
        db,
        "never_operation_arm_without_resume_is_valid.trb",
        r#"
ability Stop { op stop() -> Never }

fn test() -> Nat {
    handle Stop::stop() {
        do result { 0 }
        op Stop::stop() { 1 }
    }
}
"#,
    );
    assert!(
        errors.is_empty(),
        "a Never operation arm may return the handler result: {errors:#?}"
    );
}

#[salsa_test]
fn empty_case_eliminates_empty_types(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "empty_case_eliminates_empty_types.trb",
        r#"
enum Void {}

fn from_never(x: Never) -> Nat { case x {} }
fn from_void(v: Void) -> Nat { case v {} }
fn void_wildcard(v: Void) -> Nat { case v { _ -> 1 } }
fn never_wildcard(x: Never) -> Nat { case x { _ -> 1 } }
fn named(x: Never) -> String { absurd(x) }
fn lift(o: Option(Never)) -> Option(Nat) {
    case o {
        .None -> .None
        .Some(x) -> .Some(absurd(x))
    }
}
fn apply(f: fn(Never) ->{} Nat, o: Option(Never)) -> Nat {
    case o {
        .None -> 0
        .Some(x) -> f(x)
    }
}
fn as_value(o: Option(Never)) -> Nat { apply(absurd, o) }
fn never_tail(v: Void) -> Never { case v {} }
"#,
    );
    let diagnostics: Vec<String> = ast_pipeline_diagnostics(db, source)
        .into_iter()
        .map(|diagnostic| diagnostic.message.to_owned())
        .collect();
    assert!(
        diagnostics.is_empty(),
        "a case without arms covers an empty type, and a wildcard over one is \
         not reported: {diagnostics:#?}"
    );
    let tail = function_tail_type(db, source, "never_tail");
    assert!(
        matches!(tail.kind(db), TypeKind::Never),
        "a case without arms has type Never, got {}",
        tail.display(db, &names())
    );
}

#[salsa_test]
fn empty_case_over_inhabited_types_is_rejected(db: &salsa::DatabaseImpl) {
    for (name, body) in [
        ("nat", "fn bad(n: Nat) -> Nat { case n {} }"),
        ("bool", "fn bad(b: Bool) -> Nat { case b {} }"),
        ("option", "fn bad(o: Option(Nat)) -> Nat { case o {} }"),
        ("nested", "fn bad(o: Option(Never)) -> Nat { case o {} }"),
        (
            "enum",
            "enum One { Only }\nfn bad(v: One) -> Nat { case v {} }",
        ),
    ] {
        let errors = type_errors(db, name, body);
        assert_eq!(
            errors,
            ["non-exhaustive case expression: no patterns provided"],
            "{name}"
        );
    }
}

#[salsa_test]
fn empty_fields_are_not_inferred_empty(db: &salsa::DatabaseImpl) {
    let errors = type_errors(
        db,
        "empty_fields_are_not_inferred_empty.trb",
        "fn bad(o: Option(Never)) -> Nat { case o { .None -> 0 } }",
    );
    assert_eq!(
        errors,
        ["non-exhaustive case expression on `std::Option(Never)`: missing patterns: .Some(_)"]
    );
}

#[salsa_test]
fn empty_enums_are_not_never(db: &salsa::DatabaseImpl) {
    let errors = type_errors(
        db,
        "empty_enums_are_not_never.trb",
        "enum Void {}\nfn bad(v: Void) -> Nat { v }",
    );
    assert!(
        errors
            .iter()
            .any(|error| error.contains("expected `Nat`, found `Void`")),
        "an empty enum value does not eliminate implicitly: {errors:#?}"
    );
}

#[salsa_test]
fn empty_case_lowers_to_a_never_value(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "empty_case_lowering.trb",
        r#"
enum Void {}
fn from_never(x: Never) -> Never { case x {} }
fn from_void(v: Void) -> Nat { case v {} }
fn eliminate(x: Never) ->{} a { case x {} }
fn named(x: Never) -> Nat { eliminate(x) }
"#,
    );
    assert!(ast_pipeline_error_messages(db, source).is_empty());
    let ir = common::run_ast_pipeline_with_ir(db, source);
    let lines: Vec<&str> = ir.lines().map(str::trim).collect();
    let body = |name: &str| -> Vec<&str> {
        let header = format!("tribute_control.func @\"pkg::{name}\"");
        let start = lines
            .iter()
            .position(|line| line.starts_with(&header))
            .unwrap_or_else(|| panic!("missing {name}:\n{ir}"));
        lines[start + 1..]
            .iter()
            .take_while(|line| **line != "}")
            .copied()
            .collect()
    };
    assert_eq!(
        body("from_never"),
        ["tribute_control.return %0"],
        "a case over Never is its scrutinee:\n{ir}"
    );
    assert_eq!(
        body("from_void"),
        [
            "%1 = core.unrealized_conversion_cast %0 : !\"pkg::Void\" to core.never",
            "%2 = core.unrealized_conversion_cast %1 : core.never to core.i32",
            "tribute_control.return %2",
        ],
        "a case over an empty enum eliminates its scrutinee as Never:\n{ir}"
    );
    assert_eq!(
        body("eliminate$Nat"),
        [
            "%1 = core.unrealized_conversion_cast %0 : core.never to core.i32",
            "tribute_control.return %1",
        ],
        "{ir}"
    );
}
