mod common;

use common::ast_pipeline_error_messages;
use salsa_test_macros::salsa_test;
use tribute_front::SourceCst;

#[salsa_test]
fn unqualified_annotation_uses_nested_module_type_identity(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "nested_nominal_annotation.trb",
        r#"
pub mod A {
    pub struct Thing { value: Nat }

    pub mod Inner {
        pub fn forward(thing: Thing) -> A::Thing {
            thing
        }
    }
}
"#,
    );

    assert_eq!(
        ast_pipeline_error_messages(db, source),
        Vec::<String>::new()
    );
}

#[salsa_test]
fn string_literal_keeps_canonical_prelude_identity_when_shadowed(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "shadowed_string_literal.trb",
        r#"
enum String {
    Fake
}

fn take_user_string(_value: String) {}

fn main() {
    take_user_string("hello")
}
"#,
    );

    let errors = ast_pipeline_error_messages(db, source);
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(
        errors[0].contains("expected `String`, found `String`"),
        "{errors:?}"
    );
}
