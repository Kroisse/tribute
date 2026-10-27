//! Tests that an `extern "C"` declaration links under its declared name while
//! the IR names it by its source path.

mod common;

use self::common::run_ast_pipeline_with_ir;
use salsa_test_macros::salsa_test;
use tribute_front::SourceCst;

#[salsa_test]
fn extern_c_declarations_in_modules_share_their_link_name(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
mod a {
    pub extern "C" fn abs(value: Int) -> Int
}
mod b {
    pub extern "C" fn abs(value: Int) -> Int
}

fn main() -> Nil {
    let _ = a::abs(-3)
    let _ = b::abs(-4)
}
"#,
    );
    let ir = run_ast_pipeline_with_ir(db, source);
    for module in ["a", "b"] {
        let declaration = ir
            .lines()
            .find(|line| line.contains(&format!("func @\"pkg::{module}::abs\"")))
            .unwrap_or_else(|| panic!("{module}::abs is declared by its path:\n{ir}"));
        assert!(declaration.contains("link_name = \"abs\""), "{declaration}");
        assert!(
            ir.contains(&format!("callee = @\"pkg::{module}::abs\"")),
            "calls name {module}::abs by its path:\n{ir}"
        );
    }
}
