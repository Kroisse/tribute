//! A `handle` that has an arm for an ability must have an arm for every
//! operation of that ability. Abilities without arms pass through.

mod common;

use self::common::ast_pipeline_error_messages;
use salsa_test_macros::salsa_test;
use tribute_front::SourceCst;

fn errors(db: &dyn salsa::Database, text: &str) -> Vec<String> {
    ast_pipeline_error_messages(db, SourceCst::from_source_str(db, "handler.trb", text))
}

const STATE: &str = r#"
ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
    op reset() -> Nil
}
"#;

#[salsa_test]
fn missing_one_operation_is_an_error(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        &format!(
            "{STATE}{}",
            r#"
fn run_state(comp: fn() ->{e, State(s)} a, init: s) ->{e} a {
    handle comp() {
        do result { result }
        op State::get() { run_state(fn() { resume init }, init) }
        op State::set(v) { run_state(fn() { resume Nil }, v) }
    }
}
"#
        ),
    );
    assert_eq!(
        errors,
        vec!["handling `State` is missing an arm for `reset`".to_string()]
    );
}

#[salsa_test]
fn missing_several_operations_lists_each(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        &format!(
            "{STATE}{}",
            r#"
fn run_state(comp: fn() ->{e, State(s)} a, init: s) ->{e} a {
    handle comp() {
        do result { result }
        op State::get() { run_state(fn() { resume init }, init) }
    }
}
"#
        ),
    );
    assert_eq!(
        errors,
        vec!["handling `State` is missing arms for `reset`, `set`".to_string()]
    );
}

#[salsa_test]
fn complete_handler_is_accepted(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        &format!(
            "{STATE}{}",
            r#"
fn run_state(comp: fn() ->{e, State(s)} a, init: s) ->{e} a {
    handle comp() {
        do result { result }
        op State::get() { run_state(fn() { resume init }, init) }
        op State::set(v) { run_state(fn() { resume Nil }, v) }
        op State::reset() { run_state(fn() { resume Nil }, init) }
    }
}
"#
        ),
    );
    assert_eq!(errors, Vec::<String>::new());
}

#[salsa_test]
fn ability_without_arms_passes_through(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
ability Log {
    fn log(msg: String) -> Nil
}

ability Counter {
    fn tick() -> Nil
    fn count() -> Int
}

fn only_log(comp: fn() ->{e, Log, Counter} a) ->{e, Counter} a {
    handle comp() {
        do result { result }
        fn Log::log(msg) { Nil }
    }
}

fn only_do(comp: fn() ->{e, Counter} a) ->{e, Counter} a {
    handle comp() {
        do result { result }
    }
}
"#,
    );
    assert_eq!(errors, Vec::<String>::new());
}

#[salsa_test]
fn each_incomplete_ability_is_reported(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
ability Log {
    fn log(msg: String) -> Nil
    fn flush() -> Nil
}

ability Counter {
    fn tick() -> Nil
    fn count() -> Int
}

fn both(comp: fn() ->{e, Log, Counter} a) ->{e} a {
    handle comp() {
        do result { result }
        fn Log::log(msg) { Nil }
        fn Counter::count() { +0 }
    }
}
"#,
    );
    let mut errors = errors;
    errors.sort();
    assert_eq!(
        errors,
        vec![
            "handling `Counter` is missing an arm for `tick`".to_string(),
            "handling `Log` is missing an arm for `flush`".to_string(),
        ]
    );
}

#[salsa_test]
fn qualified_ability_is_distinct_from_same_named_ability(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod other {
    pub ability Log {
        fn log(msg: String) -> Nil
        fn flush() -> Nil
    }
}

ability Log {
    fn log(msg: String) -> Nil
}

fn handle_other(comp: fn() ->{e, other::Log} a) ->{e} a {
    handle comp() {
        do result { result }
        fn other::Log::log(msg) { Nil }
    }
}

fn handle_local(comp: fn() ->{e, Log} a) ->{e} a {
    handle comp() {
        do result { result }
        fn Log::log(msg) { Nil }
    }
}
"#,
    );
    assert_eq!(
        errors,
        vec!["handling `Log` is missing an arm for `flush`".to_string()]
    );
}
