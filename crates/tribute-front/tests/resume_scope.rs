//! `resume` binds to the nearest enclosing `op` or `fn` handler arm. Only a
//! resumptive `op` arm provides a resumption; a `resume` anywhere else is a
//! type-checking error at the `resume`.

mod common;

use self::common::{ast_pipeline_error_messages, run_ast_pipeline};
use salsa_test_macros::salsa_test;
use tribute_front::SourceCst;

const ABILITIES: &str = r#"
ability Ask {
    op ask() -> Int
}

ability Tell {
    fn tell(value: Int) -> Nil
}
"#;

fn resume_errors(db: &dyn salsa::Database, name: &str, body: &str) -> Vec<String> {
    let source = SourceCst::from_source_str(db, name, &format!("{ABILITIES}{body}"));
    ast_pipeline_error_messages(db, source)
}

#[salsa_test]
fn resume_without_resumption_is_reported(db: &salsa::DatabaseImpl) {
    for (name, body) in [
        (
            "fn_arm_of_op_operation.trb",
            r#"
fn main() -> Nil {
    let _ = handle Ask::ask() {
        fn Ask::ask() { resume +1 }
    }
}
"#,
        ),
        (
            "fn_arm_of_fn_operation.trb",
            r#"
fn main() -> Nil {
    handle Tell::tell(+1) {
        fn Tell::tell(value) { resume Nil }
    }
}
"#,
        ),
        (
            "outside_any_arm.trb",
            r#"
fn main() -> Int {
    resume +1
}
"#,
        ),
        (
            "fn_arm_inside_op_arm.trb",
            r#"
fn main() -> Int {
    handle Ask::ask() {
        op Ask::ask() {
            handle Tell::tell(+1) {
                do _ { resume +2 }
                fn Tell::tell(value) { resume value }
            }
        }
    }
}
"#,
        ),
        (
            "lambda_inside_fn_arm.trb",
            r#"
fn main() -> Nil {
    let _ = handle Ask::ask() {
        fn Ask::ask() {
            let k = fn() { resume +1 }
            +1
        }
    }
}
"#,
        ),
    ] {
        let errors = resume_errors(db, name, body);
        let reported = errors
            .iter()
            .filter(|error| error.contains("`resume` is only valid under a resumptive `op`"))
            .count();
        assert_eq!(
            reported, 1,
            "{name}: the `resume` without a resumption must be reported once: {errors:#?}"
        );
    }
}

#[salsa_test]
fn resume_in_lambda_under_op_arm_compiles(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "lambda_under_op_arm.trb",
        &format!(
            "{ABILITIES}{}",
            r#"
fn run_ask(comp: fn() ->{e, Ask} a, value: Int) ->{e} a {
    handle comp() {
        do result { result }
        op Ask::ask() { run_ask(fn() { resume value }, value) }
    }
}

fn answer() -> Int {
    run_ask(fn() { Ask::ask() }, +1)
}
"#
        ),
    );
    run_ast_pipeline(db, source);
}
