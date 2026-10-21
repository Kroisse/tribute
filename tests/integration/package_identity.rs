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
