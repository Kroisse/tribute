//! Frontend contracts for `Never` equality, coercion, and common results.

mod common;

use self::common::{ast_pipeline_error_messages, run_ast_pipeline_with_ir};
use salsa_test_macros::salsa_test;
use tribute_front::{
    SourceCst,
    ast::{Decl, ExprKind, Type, TypeKind},
};
use trunk_ir::Symbol;

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
        .decls
        .iter()
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
fn invoke(thunk: fn() -> Nat) -> Nat { thunk() }
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
fn empty() { [] }
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
                if *name == Symbol::new("List")
                    && matches!(args.as_slice(), [arg] if matches!(arg.kind(db), TypeKind::Nat))
        ),
        "the non-Never list element must determine List(Nat), got {mixed}"
    );
    let empty = function_tail_type(db, source, "empty");
    assert!(
        matches!(
            empty.kind(db),
            TypeKind::Named { name, args, .. }
                if *name == Symbol::new("List")
                    && matches!(args.as_slice(), [arg] if matches!(
                        arg.kind(db),
                        TypeKind::UniVar { .. }
                            | TypeKind::BoundVar { .. }
                            | TypeKind::LocalBoundVar { .. }
                    ))
        ),
        "an empty list must retain a fresh element type, got {empty}"
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
        A -> 1
        B -> abort()
        C -> 2
    }
}

fn never_first(choice: Choice) ->{Stop} Nat {
    case choice {
        A -> abort()
        B -> 1
        C -> 2
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
        "all-Never case arms must retain Never, got {all_never}"
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
        A -> abort()
        B -> 1
        C -> True
    }
}

fn never_in_middle(choice: Choice) ->{Stop} Nat {
    case choice {
        A -> 1
        B -> abort()
        C -> True
    }
}

fn never_last(choice: Choice) ->{Stop} Nat {
    case choice {
        A -> 1
        B -> True
        C -> abort()
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
    extern "intrinsic" fn explode(value: Bomb) -> Never
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
        "the deferred method call must retain actual Never before contextual coercion, got {actual}"
    );
}

#[salsa_test]
fn pure_let_remains_polymorphic(db: &salsa::DatabaseImpl) {
    let errors = type_errors(
        db,
        "pure_let_remains_polymorphic.trb",
        r#"
fn pair() -> #(Nat, Bool) {
    let identity = fn(value) value
    #(identity(1), identity(True))
}

fn generic_join() -> #(Nat, Bool) {
    let select = fn(flag, value) {
        case flag {
            True -> value
            False -> value
        }
    }
    #(select(True, 1), select(False, True))
}

struct Bomb {}
pub mod Bomb { extern "intrinsic" fn explode(value: Bomb) -> Never }

fn partial_generalization() -> #(Nat, Bool) {
    let #(explode, identity) = #(fn(bomb) bomb.explode(), fn(value) value)
    explode(Bomb {})
    #(identity(1), identity(True))
}
"#,
    );

    assert!(
        errors.is_empty(),
        "pure let binding should generalize: {errors:#?}"
    );
}

#[salsa_test]
fn returned_callback_preserves_cps_convention_after_pure_let(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "returned_callback_preserves_cps_convention_after_pure_let.trb",
        r#"
extern "C" fn __tribute_print_nat(value: Nat) -> Nil

fn identity(f: fn(Nat) -> Nat) -> fn(Nat) -> Nat { f }

fn main() {
    let f = identity(fn(x: Nat) { x + 10 })
    let result = f(5)
    __tribute_print_nat(result)
}
"#,
    );

    let errors = ast_pipeline_error_messages(db, source);
    assert!(errors.is_empty(), "{errors:#?}");
    let ir = run_ast_pipeline_with_ir(db, source);
    let main = ir
        .lines()
        .find(|line| line.contains("func @main("))
        .expect("IR must contain main");
    assert!(
        main.contains("convention(cps)"),
        "main must remain CPS for the returned callback call:\n{ir}"
    );
    let lambda = ir
        .lines()
        .find(|line| line.contains("tribute_control.lambda("))
        .expect("IR must contain the callback lambda");
    assert!(
        lambda.contains("convention(cps)"),
        "the callback lambda must retain its CPS convention:\n{ir}"
    );
    assert!(
        ir.lines()
            .any(|line| line.contains("tribute_control.call_indirect")),
        "the returned callback must use an indirect call:\n{ir}"
    );
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
        "the handle answer should be Nat, got {result}"
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
    assert_eq!(checked.lambda_signatures(db).len(), 1);
    let (lambda_id, signature) = &checked.lambda_signatures(db)[0];
    let TypeKind::Func { result, .. } = signature.function_type.kind(db) else {
        panic!("expected callable");
    };
    assert!(matches!(result.kind(db), TypeKind::Nat));
    let factory = checked
        .module(db)
        .decls
        .iter()
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
pub mod Probe { fn answer(value: Probe) -> Bool { True } }
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

#[salsa_test]
fn a_pure_join_generalizes_callable_type_and_effect_variables(db: &salsa::DatabaseImpl) {
    let errors = type_errors(
        db,
        "generic_callable_join.trb",
        r#"
extern "intrinsic" fn diverge() ->{} Never
fn both() -> #(Nat, Bool) {
    let apply = case True {
        True -> diverge()
        False -> fn(thunk) thunk()
    }
    #(apply(fn() { 1 }), apply(fn() { True }))
}
"#,
    );
    assert!(errors.is_empty(), "{errors:#?}");
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
            .any(|line| line.contains("tribute_control.perform") && line.ends_with(": core.never")),
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
ability Stop { op stop() -> Never }
fn result() ->{Stop} Nat {
    handle Read::read() {
        do value { value }
        fn Read::read() { Stop::stop() }
    }
}
"#,
    );
    let errors = ast_pipeline_error_messages(db, source);
    assert!(errors.is_empty(), "{errors:#?}");
    let checked = tribute_front::query::type_check_output(db, source).unwrap();
    let (_, operation) = checked
        .handler_operations(db)
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
        .decls
        .iter()
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
pub mod Bomb { extern "intrinsic" fn explode(value: Bomb) -> Never }
struct Runner {}
pub mod Runner {
    fn run(receiver: Runner, callback: fn() -> Nat) -> Nat { callback() }
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
        .lambda_signatures(db)
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
    fn run(receiver: Runner, callback: {callback_type}) -> Nat {{ 0 }}
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
pub mod Bomb { extern "intrinsic" fn explode(value: Bomb) -> Never }
struct Runner {}
pub mod Runner {
    fn run(receiver: Runner, callback: fn() -> fn() -> Nat) -> Nat {
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
