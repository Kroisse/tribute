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

#[salsa_test]
fn forward_nominal_annotations_use_source_identity(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "forward_nominal_annotations.trb",
        r#"
fn make_bare() -> Later {
    Later { value: 1 }
}

fn make_qualified() -> A::Later {
    A::Later { value: 2 }
}

fn read_bare(value: Later) -> Nat {
    value.value
}

fn read_qualified(value: A::Later) -> Nat {
    value.value
}

struct Holder {
    value: Wrapper(FieldLater)
}

struct Wrapper(a) {
    value: a
}

struct Later {
    value: Nat
}

pub mod A {
    pub struct Later {
        value: Nat
    }
}

struct FieldLater {
    value: Nat
}

fn unwrap(holder: Holder) -> FieldLater {
    holder.value.value
}
"#,
    );

    assert_eq!(
        ast_pipeline_error_messages(db, source),
        Vec::<String>::new()
    );
}

#[salsa_test]
fn nested_struct_fields_and_record_patterns_use_declaration_identity(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "nested_struct_fields.trb",
        r#"
pub mod A {
    pub struct Thing {
        value: Nat
    }

    pub fn field(thing: Thing) -> Nat {
        thing.value
    }

    pub fn destructure(thing: Thing) -> Nat {
        let A::Thing { value } = thing
        value
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
fn nested_record_pattern_preserves_field_type(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "nested_record_pattern_field_type.trb",
        r#"
pub mod A {
    pub struct Thing {
        value: String
    }

    pub fn invalid(thing: Thing) -> Nat {
        let A::Thing { value } = thing
        value
    }
}
"#,
    );

    let errors = ast_pipeline_error_messages(db, source);
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(
        errors[0].contains("expected `String`, found `Nat`"),
        "{errors:?}"
    );
}
