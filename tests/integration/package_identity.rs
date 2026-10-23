//! A user package's declarations keep their own identity: a user module
//! named `std` is not package `std`, and its symbols start with `pkg`.

use salsa_test_macros::salsa_test;
use tribute::pipeline::compile_frontend;
use tribute_front::SourceCst;
use trunk_ir::printer::print_module;

use super::common::assert_native_output;

const PROGRAM: &str = r#"
use io::{Io, print_line}

mod std {
    pub mod Int {
        pub fn probe(value: Int) ->{} Int { value + +100 }
    }
    pub ability Tick {
        op tick() -> Int
    }
}

fn count() ->{std::Tick} Int { std::Tick::tick() }

fn main() ->{Io} Nil {
    let ticked = handle count() {
        do result { result }
        op std::Tick::tick() { resume +2 }
    }
    print_line(Int::to_string(std::Int::probe(+1) + ticked))
}
"#;

#[salsa_test]
fn a_user_std_module_has_its_own_symbols(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(db, "package_identity.trb", PROGRAM);
    let (ctx, module) = compile_frontend(db, source).expect("frontend output");
    let ir = print_module(&ctx, module.op());

    assert!(
        ir.contains(r#"tribute_control.func @"pkg::std::Int::probe""#),
        "{ir}"
    );
    assert!(ir.contains(r#"callee = @"pkg::std::Int::probe""#), "{ir}");
    // The prelude's declarations keep their package paths.
    assert!(ir.contains(r#"@"std::Int::to_string""#), "{ir}");
    assert!(!ir.contains(r#"@"std::Int::probe""#), "{ir}");
}

#[test]
fn a_user_std_module_runs_its_own_functions() {
    assert_native_output("package_identity.trb", PROGRAM, "103");
}

/// A leading `::` reaches package `std` past a user module named `std`.
#[test]
fn a_leading_separator_runs_the_package_past_a_user_std_module() {
    assert_native_output(
        "leading_separator.trb",
        r#"
use ::std::io::{Io, print_line}

mod std {
    pub mod Int {
        pub fn probe(value: Int) ->{} Int { value + +100 }
    }
}

fn main() ->{Io} Nil {
    print_line(::std::Int::to_string(std::Int::probe(+1)))
}
"#,
        "101",
    );
}

/// Package `std` is resolved from its own root: a `pub use` at its root
/// exports a name, which other packages see by its short name and under
/// `std`.
#[test]
fn a_std_root_re_export_is_visible_to_users() {
    super::common::assert_output_in_std_on_both_targets_with_native_asan(
        "std_root_re_export.trb",
        r#"
pub mod extras {
    pub fn seven() ->{} Int { +7 }
}

pub use pkg::extras::seven
"#,
        r#"
use std::io::{Io, print_line}

fn main() ->{Io} Nil {
    print_line(Int::to_string(seven() + std::seven()))
}
"#,
        "14",
    );
}
