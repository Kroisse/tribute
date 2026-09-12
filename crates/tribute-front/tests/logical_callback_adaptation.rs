//! Source-logical callback adaptation contracts.

mod common;

use self::common::run_ast_pipeline_with_ir;
use salsa_test_macros::salsa_test;
use tribute_front::SourceCst;

fn named_function_header<'a>(ir: &'a str, name: &str) -> &'a str {
    ir.lines()
        .find(|line| {
            line.trim_start()
                .starts_with(&format!("tribute_control.func @{name}("))
        })
        .unwrap_or_else(|| panic!("missing lowered function @{name}:\n{ir}"))
}

fn parameter_type(header: &str, index: usize) -> &str {
    header
        .split_once('(')
        .and_then(|(_, tail)| tail.split_once(") ->"))
        .and_then(|(params, _)| params.split(',').nth(index))
        .and_then(|parameter| parameter.rsplit_once(": "))
        .map(|(_, ty)| ty.trim())
        .unwrap_or_else(|| panic!("missing parameter {index} in {header}"))
}

fn value_name(line: &str) -> &str {
    line.trim_start()
        .split_once(" = ")
        .map(|(value, _)| value)
        .unwrap_or_else(|| panic!("missing result value in {line}"))
}

fn type_after_colon(line: &str) -> &str {
    line.rsplit_once(": ")
        .map(|(_, ty)| ty.trim())
        .unwrap_or_else(|| panic!("missing type in {line}"))
}

fn callable_convention<'a>(ir: &'a str, callable_ty: &str) -> &'a str {
    ir.lines()
        .find(|line| line.trim_start().starts_with(&format!("{callable_ty} =")))
        .and_then(|line| line.split("tribute.calling_convention = ").nth(1))
        .and_then(|suffix| suffix.split('}').next())
        .map(str::trim)
        .unwrap_or_else(|| panic!("missing callable convention for {callable_ty}:\n{ir}"))
}

fn named_ref_lines<'a>(ir: &'a str, name: &str) -> Vec<&'a str> {
    ir.lines()
        .filter(|line| line.contains(&format!("func_ref = @{name}")))
        .collect()
}

fn assert_not_cast(ir: &str, value: &str) {
    assert!(
        !ir.lines().any(|line| {
            line.contains("core.unrealized_conversion_cast")
                && line
                    .split("core.unrealized_conversion_cast")
                    .nth(1)
                    .is_some_and(|suffix| suffix.trim_start().starts_with(value))
        }),
        "the callable value {value} must not be adapted by a raw conversion cast:\n{ir}"
    );
}

#[salsa_test]
fn named_function_value_uses_its_callback_contract(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "logical_callback_adaptation.trb",
        r#"
fn apply(f: fn(Int) -> Int, x: Int) -> Int { f(x) }
fn double(value: Int) -> Int { value + value }
fn main() { apply(double, 21) }
"#,
    );

    let ir = run_ast_pipeline_with_ir(db, source);
    let refs = named_ref_lines(&ir, "double");
    assert_eq!(
        refs.len(),
        1,
        "expected one named callback reference:\n{ir}"
    );
    let double_ref = refs[0];
    let callback_ty = type_after_colon(double_ref);
    assert_eq!(
        parameter_type(named_function_header(&ir, "apply"), 0),
        callback_ty,
        "the fresh function reference must have apply's exact callback type:\n{ir}"
    );
    assert_eq!(callable_convention(&ir, callback_ty), "2");
    assert!(
        named_function_header(&ir, "double").contains("convention(direct)"),
        "the original named worker must remain Direct:\n{ir}"
    );
    assert_not_cast(&ir, value_name(double_ref));
}

#[salsa_test]
fn named_value_uses_distinct_contracts(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "logical_callback_distinct_contracts.trb",
        r#"
fn apply_direct(f: fn(Int) ->{} Int, x: Int) ->{} Int { f(x) }
fn apply_open(f: fn(Int) -> Int, x: Int) -> Int { f(x) }
fn double(value: Int) -> Int { value + value }
fn main() { apply_open(double, apply_direct(double, 21)) }
"#,
    );

    let ir = run_ast_pipeline_with_ir(db, source);
    let refs = named_ref_lines(&ir, "double");
    assert_eq!(
        refs.len(),
        2,
        "each use needs its own function reference:\n{ir}"
    );
    let reference_values = refs.iter().map(|line| value_name(line)).collect::<Vec<_>>();
    assert_ne!(reference_values[0], reference_values[1]);
    let conventions = refs
        .iter()
        .map(|line| callable_convention(&ir, type_after_colon(line)))
        .collect::<Vec<_>>();
    assert!(conventions.contains(&"0"));
    assert!(conventions.contains(&"2"));
    for value in reference_values {
        assert_not_cast(&ir, value);
    }
}

#[salsa_test]
fn lambda_value_uses_an_exact_compatible_callback_contract(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "logical_lambda_callback_adaptation.trb",
        r#"
fn apply(f: fn(Int) -> Int, x: Int) -> Int { f(x) }
fn main() { apply(fn(value) { value + value }, 21) }
"#,
    );

    let ir = run_ast_pipeline_with_ir(db, source);
    let lambda = ir
        .lines()
        .find(|line| line.contains("tribute_control.lambda"))
        .expect("logical lowering must materialize the callback lambda");
    assert!(
        lambda.contains("convention(cps)"),
        "the exact-compatible lambda must be emitted at the CPS callback convention:\n{ir}"
    );
    assert_not_cast(&ir, value_name(lambda));
}
