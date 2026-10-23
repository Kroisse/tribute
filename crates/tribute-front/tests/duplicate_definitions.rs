//! A module declares each name once among its functions, structs, enums, and
//! abilities; the later declaration is the error.

mod common;

use self::common::ast_pipeline_error_messages;
use salsa_test_macros::salsa_test;
use tribute_front::SourceCst;

fn errors(db: &dyn salsa::Database, text: &str) -> Vec<String> {
    ast_pipeline_error_messages(
        db,
        SourceCst::from_source_str(db, "duplicate_definitions.trb", text),
    )
}

fn assert_duplicate(errors: &[String], name: &str) {
    let expected = format!("duplicate definition of `{name}`");
    assert!(
        errors.contains(&expected),
        "expected `{expected}` in {errors:#?}"
    );
}

#[salsa_test]
fn a_struct_declared_twice_is_reported(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        "struct A { x: Nat }\nstruct A { y: Nat }\nfn main() -> Nil {}\n",
    );
    assert_duplicate(&errors, "A");
}

#[salsa_test]
fn an_enum_and_a_struct_of_one_name_are_reported(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        "enum A { One, Two }\nstruct A { x: Nat }\nfn main() -> Nil {}\n",
    );
    assert_duplicate(&errors, "A");
}

#[salsa_test]
fn a_function_declared_twice_is_reported(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        "fn f() -> Nat { 1 }\nfn f() -> Nat { 2 }\nfn main() -> Nil {}\n",
    );
    assert_duplicate(&errors, "f");
}

#[salsa_test]
fn an_ability_declared_twice_is_reported(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        "ability Ask { fn ask() -> Nat }\nability Ask { fn ask() -> Nat }\nfn main() -> Nil {}\n",
    );
    assert_duplicate(&errors, "Ask");
}

#[salsa_test]
fn a_duplicate_inside_an_inline_module_is_reported(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        "mod m {\n    pub struct A { x: Nat }\n    pub struct A { y: Nat }\n}\nfn main() -> Nil {}\n",
    );
    assert_duplicate(&errors, "A");
}

#[salsa_test]
fn a_type_and_its_companion_module_coexist(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        "struct A { x: Nat }\nmod A {\n    pub fn make() -> A { A(1) }\n}\nfn main() -> Nil {}\n",
    );
    assert_eq!(errors, Vec::<String>::new());
}

#[salsa_test]
fn one_name_in_different_modules_is_not_a_duplicate(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        "struct A { x: Nat }\nmod m {\n    pub struct A { y: Nat }\n    pub fn f() -> Nat { 1 }\n}\nfn f() -> Nat { 2 }\nfn main() -> Nil {}\n",
    );
    assert_eq!(errors, Vec::<String>::new());
}
