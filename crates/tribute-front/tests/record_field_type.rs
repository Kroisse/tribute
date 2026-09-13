//! Tests for record field type checking.
//!
//! These tests verify that record construction properly validates
//! field expression types against declared struct field types.

mod common;

use self::common::{ast_pipeline_diagnostics, run_ast_pipeline, run_ast_pipeline_with_ir};
use insta::assert_snapshot;
use salsa_test_macros::salsa_test;
use tribute_core::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use tribute_front::{
    SourceCst,
    ast::{Decl, Expr, ExprKind, Module, Type, TypeDefId, TypeKind, TypedRef},
    typeck::ExpressionTypeMetadata,
};
use trunk_ir::Span;

fn diagnostics(db: &dyn salsa::Database, text: &str) -> Vec<Diagnostic> {
    let source = SourceCst::from_source_str(db, "record_shape.trb", text);
    ast_pipeline_diagnostics(db, source)
}

fn shape_error(text: &str, record: &str, occurrence: usize, message: &str) -> Diagnostic {
    let start = text.match_indices(record).nth(occurrence).unwrap().0;
    Diagnostic::new(
        message,
        Span::new(start, start + record.len()),
        DiagnosticSeverity::Error,
        CompilationPhase::TypeChecking,
    )
}

fn typed_function_body<'a, 'db>(
    module: &'a Module<TypedRef<'db>>,
    name: &str,
) -> &'a Expr<TypedRef<'db>> {
    module
        .decls
        .iter()
        .find_map(|decl| match decl {
            Decl::Function(function) if function.name.with_str(|candidate| candidate == name) => {
                Some(&function.body)
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("missing typed function {name}"))
}

fn typed_function_tail<'a, 'db>(
    module: &'a Module<TypedRef<'db>>,
    name: &str,
) -> &'a Expr<TypedRef<'db>> {
    let body = typed_function_body(module, name);
    let ExprKind::Block { value, .. } = &*body.kind else {
        panic!("{name} must have a block body");
    };
    value
}

fn declared_struct_id<'db>(
    db: &'db dyn salsa::Database,
    module: &Module<TypedRef<'db>>,
    name: &str,
) -> TypeDefId<'db> {
    let structure = module
        .decls
        .iter()
        .find_map(|decl| match decl {
            Decl::Struct(structure) if structure.name.with_str(|candidate| candidate == name) => {
                Some(structure)
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("missing struct {name}"));
    TypeDefId::source(db, structure.name, structure.id)
}

fn node_type<'db>(
    metadata: &ExpressionTypeMetadata<'db>,
    expression: &Expr<TypedRef<'db>>,
) -> Type<'db> {
    metadata
        .node_types
        .iter()
        .find_map(|(id, ty)| (*id == expression.id).then_some(*ty))
        .unwrap_or_else(|| panic!("missing type metadata for node {:?}", expression.id))
}

fn record_type_name<'a, 'db>(record: &'a Expr<TypedRef<'db>>) -> &'a TypedRef<'db> {
    let ExprKind::Record { type_name, .. } = &*record.kind else {
        panic!("expected a record expression");
    };
    type_name
}

fn assert_named_args(
    db: &dyn salsa::Database,
    ty: Type<'_>,
    expected_id: TypeDefId<'_>,
    name: &str,
    args: &[TypeKind<'_>],
) {
    let TypeKind::Named {
        id,
        name: actual,
        args: actual_args,
    } = ty.kind(db)
    else {
        panic!("expected {name} named type, found {ty:?}");
    };
    assert_eq!(*id, expected_id, "{name} declaration identity");
    actual.with_str(|actual| assert_eq!(actual, name));
    assert_eq!(actual_args.len(), args.len(), "{name} argument count");
    for (actual, expected) in actual_args.iter().zip(args) {
        assert_eq!(actual.kind(db), expected, "{name} type argument");
    }
}

#[salsa_test]
fn record_shape_diagnostic_matrix(db: &salsa::DatabaseImpl) {
    for (fields, messages) in [
        (
            "x: +1, y: +2, z: +3",
            vec!["unknown field `z` for struct `Point`"],
        ),
        ("x: +1, x: +2, y: +3", vec!["duplicate field `x`"]),
        (
            "x: +1, x: +2, x: +3, y: +4",
            vec!["duplicate field `x`", "duplicate field `x`"],
        ),
        (
            "x: +1, y: +2, z: +3, z: +4",
            vec![
                "unknown field `z` for struct `Point`",
                "unknown field `z` for struct `Point`",
            ],
        ),
        ("x: +1", vec!["missing field: y"]),
        ("", vec!["missing field: x"]),
        (
            "x: +1, x: +2, z: +3",
            vec![
                "duplicate field `x`",
                "unknown field `z` for struct `Point`",
                "missing field: y",
            ],
        ),
        ("x: +1, ..base", vec![]),
        ("x: +1, x: +2, ..base", vec!["duplicate field `x`"]),
        (
            "z: +1, z: +2, ..base",
            vec![
                "unknown field `z` for struct `Point`",
                "unknown field `z` for struct `Point`",
            ],
        ),
    ] {
        let record = format!("Point {{ {fields} }}");
        let text = format!(
            "struct Point {{ x: Int, y: Int }}\nfn make(base: Point) -> Point {{ {record} }}"
        );
        let expected: Vec<_> = messages
            .into_iter()
            .map(|message| shape_error(&text, &record, 0, message))
            .collect();
        assert_eq!(diagnostics(db, &text), expected, "fields: {fields}");
    }
}

#[salsa_test]
fn record_shape_reports_the_first_missing_field_in_declaration_order(db: &salsa::DatabaseImpl) {
    let record = "Ordered { }";
    let text =
        format!("struct Ordered {{ z: Int, a: Int, m: Int }}\nfn make() -> Ordered {{ {record} }}");
    assert_eq!(
        diagnostics(db, &text),
        vec![shape_error(&text, record, 0, "missing field: z")]
    );
}

#[salsa_test]
fn record_shape_empty_struct_has_registered_fields(db: &salsa::DatabaseImpl) {
    let valid = "Empty { }";
    let invalid = "Empty { extra: +1 }";
    let text = format!(
        "struct Empty {{}}\nfn valid() -> Empty {{ {valid} }}\nfn invalid() -> Empty {{ {invalid} }}"
    );
    assert_eq!(
        diagnostics(db, &text),
        vec![shape_error(
            &text,
            invalid,
            0,
            "unknown field `extra` for struct `Empty`"
        )]
    );
}

#[salsa_test]
fn record_shape_argument_revisits_preserve_distinct_occurrences(db: &salsa::DatabaseImpl) {
    let record = "Point { x: +1, y: +2, z: +3 }";
    for occurrences in [1, 2] {
        let calls = format!("take({record})\n").repeat(occurrences);
        let text = format!(
            "struct Point {{ x: Int, y: Int }}\nfn take(p: Point) {{}}\nfn run() {{ {calls} }}"
        );
        let expected: Vec<_> = (0..occurrences)
            .map(|occurrence| {
                shape_error(
                    &text,
                    record,
                    occurrence,
                    "unknown field `z` for struct `Point`",
                )
            })
            .collect();
        assert_eq!(diagnostics(db, &text), expected);
    }
}

#[salsa_test]
fn record_shape_nested_errors_survive_argument_revisits(db: &salsa::DatabaseImpl) {
    let inner = "Point { x: +1 }";
    let outer = format!("Wrapper {{ point: {inner}, extra: +2 }}");
    let text = format!(
        "struct Point {{ x: Int, y: Int }}\nstruct Wrapper {{ point: Point }}\n\
         fn take(value: Wrapper) {{}}\nfn run() {{ take({outer}) }}"
    );
    assert_eq!(
        diagnostics(db, &text),
        vec![
            shape_error(
                &text,
                &outer,
                0,
                "unknown field `extra` for struct `Wrapper`"
            ),
            shape_error(&text, inner, 0, "missing field: y"),
        ]
    );
}

#[salsa_test]
fn record_shape_unknown_and_duplicate_fields_still_check_nested_rhs(db: &salsa::DatabaseImpl) {
    for (field, message) in [
        ("extra", "unknown field `extra` for struct `Wrapper`"),
        ("point", "duplicate field `point`"),
    ] {
        let inner = "Point { x: +1 }";
        let outer = format!("Wrapper {{ point: valid, {field}: {inner} }}");
        let text = format!(
            "struct Point {{ x: Int, y: Int }}\nstruct Wrapper {{ point: Point }}\n\
             fn make(valid: Point) -> Wrapper {{ {outer} }}"
        );
        assert_eq!(
            diagnostics(db, &text),
            vec![
                shape_error(&text, &outer, 0, message),
                shape_error(&text, inner, 0, "missing field: y"),
            ]
        );
    }
}

#[salsa_test]
fn record_shape_errors_preserve_rhs_type_errors(db: &salsa::DatabaseImpl) {
    for (field, rhs, shape_message) in [
        (
            "extra",
            "need_int(True)",
            "unknown field `extra` for struct `Point`",
        ),
        ("x", "True", "duplicate field `x`"),
    ] {
        let record = format!("Point {{ x: +1, y: +2, {field}: {rhs} }}");
        let text = format!(
            "struct Point {{ x: Int, y: Int }}\nfn need_int(value: Int) -> Int {{ value }}\n\
             fn make() -> Point {{ {record} }}"
        );
        let errors = diagnostics(db, &text);
        assert_eq!(errors.len(), 2, "{errors:#?}");
        assert_eq!(errors[0], shape_error(&text, &record, 0, shape_message));
        assert_eq!(errors[1].phase, CompilationPhase::TypeChecking);
        assert_eq!(errors[1].inner.severity, DiagnosticSeverity::Error);
        assert!(
            errors[1]
                .inner
                .message
                .contains("expected `Int`, found `Bool`"),
            "{errors:#?}"
        );
    }
}

#[salsa_test]
fn record_shape_unknown_field_preserves_nominal_spread_error(db: &salsa::DatabaseImpl) {
    let record = "Point { z: +1, ..base }";
    let text = format!(
        "struct Point {{ x: Int, y: Int }}\nstruct Other {{ x: Int, y: Int }}\n\
         fn make(base: Other) -> Point {{ {record} }}"
    );
    let errors = diagnostics(db, &text);
    assert_eq!(errors.len(), 2, "{errors:#?}");
    assert_eq!(
        errors[0],
        shape_error(&text, record, 0, "unknown field `z` for struct `Point`")
    );
    assert_eq!(errors[1].phase, CompilationPhase::TypeChecking);
    assert!(
        errors[1]
            .inner
            .message
            .contains("expected `Point`, found `Other`"),
        "{errors:#?}"
    );
}

#[salsa_test]
fn record_shape_uses_generic_declaration_fields(db: &salsa::DatabaseImpl) {
    let record = "Pair { first: +1, extra: True }";
    let text = format!(
        "struct Pair(a, b) {{ first: a, second: b }}\n\
         fn make() -> Pair(Int, Bool) {{ {record} }}"
    );
    assert_eq!(
        diagnostics(db, &text),
        vec![
            shape_error(&text, record, 0, "unknown field `extra` for struct `Pair`"),
            shape_error(&text, record, 0, "missing field: second"),
        ]
    );
}

#[salsa_test]
fn record_shape_uses_qualified_declaration_identity(db: &salsa::DatabaseImpl) {
    let a_record = "A::Point { y: +1 }";
    let b_record = "B::Point { x: +2 }";
    let text = format!(
        "pub mod A {{ pub struct Point {{ x: Int }} }}\n\
         pub mod B {{ pub struct Point {{ y: Int }} }}\n\
         fn make_a() -> A::Point {{ {a_record} }}\n\
         fn make_b() -> B::Point {{ {b_record} }}"
    );
    assert_eq!(
        diagnostics(db, &text),
        vec![
            shape_error(
                &text,
                a_record,
                0,
                "unknown field `y` for struct `A::Point`"
            ),
            shape_error(&text, a_record, 0, "missing field: x"),
            shape_error(
                &text,
                b_record,
                0,
                "unknown field `x` for struct `B::Point`"
            ),
            shape_error(&text, b_record, 0, "missing field: y"),
        ]
    );
}

#[salsa_test]
fn valid_record_shapes_keep_field_and_callable_inference(db: &salsa::DatabaseImpl) {
    let text = r#"
struct Point { x: Int, y: Int }
struct Pair(a, b) { first: a, second: b }
struct Callback { run: fn(Int) -> Int }
pub mod A { pub struct Point { x: Int } }
pub mod B { pub struct Point { y: Int } }

fn complete() -> Point { Point { y: +2, x: +1 } }
fn spread_only(base: Point) -> Point { Point { ..base } }
fn partial_override(base: Point) -> Point { Point { y: +3, ..base } }
fn full_override(base: Point) -> Point { Point { x: +3, y: +4, ..base } }
fn generic() -> Pair(Int, Bool) { Pair { first: +1, second: True } }
fn callable() -> Callback { Callback { run: fn(value) { value } } }
fn make_a() -> A::Point { A::Point { x: +1 } }
fn make_b() -> B::Point { B::Point { y: +2 } }
"#;
    assert_eq!(diagnostics(db, text), vec![]);
}

#[salsa_test]
fn generic_record_fields_infer_and_receive_contextual_arguments(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "generic_record_inference.trb",
        r#"
struct Pair(a, b) { first: a, second: b }

fn inferred() { Pair { first: +1, second: True } }
fn expected_return() -> Pair(Int, Bool) { Pair { first: +1, second: True } }
fn take(value: Pair(Int, Bool)) -> Pair(Int, Bool) { value }
fn expected_argument() -> Pair(Int, Bool) { take(Pair { first: +1, second: True }) }
"#,
    );
    assert_eq!(
        ast_pipeline_diagnostics(db, source),
        vec![],
        "generic record construction must type check"
    );

    let checked = tribute_front::query::type_check_output(db, source)
        .expect("type checking should produce output");
    let metadata = checked.expression_types(db);
    let module = checked.module(db);
    let pair = declared_struct_id(db, module, "Pair");
    let expected = [TypeKind::Int, TypeKind::Bool];
    for name in ["inferred", "expected_return"] {
        let record = typed_function_tail(module, name);
        assert_named_args(db, node_type(metadata, record), pair, "Pair", &expected);
        let ExprKind::Record { fields, .. } = &*record.kind else {
            panic!("{name} must end in a record");
        };
        assert_eq!(fields.len(), 2);
        assert!(matches!(
            node_type(metadata, &fields[0].1).kind(db),
            TypeKind::Int
        ));
        assert!(matches!(
            node_type(metadata, &fields[1].1).kind(db),
            TypeKind::Bool
        ));
    }
    let call = typed_function_tail(module, "expected_argument");
    let ExprKind::Call { args, .. } = &*call.kind else {
        panic!("expected_argument must end in a call");
    };
    let ExprKind::Record { fields, .. } = &*args[0].kind else {
        panic!("the call argument must be a record");
    };
    assert_named_args(db, node_type(metadata, &args[0]), pair, "Pair", &expected);
    assert!(matches!(
        node_type(metadata, &fields[0].1).kind(db),
        TypeKind::Int
    ));
    assert!(matches!(
        node_type(metadata, &fields[1].1).kind(db),
        TypeKind::Bool
    ));
}

#[salsa_test]
fn generic_record_instances_keep_independent_substitutions(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "independent_generic_records.trb",
        r#"
struct Pair(a, b) { first: a, second: b }

fn independent() -> #(Pair(Int, Bool), Pair(Bool, Int)) {
    #(Pair { first: +1, second: True }, Pair { first: False, second: +2 })
}
"#,
    );
    assert_eq!(ast_pipeline_diagnostics(db, source), vec![]);

    let checked = tribute_front::query::type_check_output(db, source).unwrap();
    let module = checked.module(db);
    let pair = declared_struct_id(db, module, "Pair");
    let tail = typed_function_tail(module, "independent");
    let ExprKind::Tuple(records) = &*tail.kind else {
        panic!("independent must end in a tuple");
    };
    assert_named_args(
        db,
        node_type(checked.expression_types(db), &records[0]),
        pair,
        "Pair",
        &[TypeKind::Int, TypeKind::Bool],
    );
    assert_named_args(
        db,
        node_type(checked.expression_types(db), &records[1]),
        pair,
        "Pair",
        &[TypeKind::Bool, TypeKind::Int],
    );
}

#[salsa_test]
fn generic_record_repeated_field_parameter_rejects_mismatched_rhs(db: &salsa::DatabaseImpl) {
    let text = r#"
struct Same(a) { first: a, second: a }
fn bad() -> Same(Int) { Same { first: +1, second: True } }
"#;
    let errors = diagnostics(db, text);
    assert_eq!(errors.len(), 1, "{errors:#?}");
    assert_eq!(errors[0].phase, CompilationPhase::TypeChecking);
    assert!(
        errors[0]
            .inner
            .message
            .contains("expected `Int`, found `Bool`"),
        "{errors:#?}"
    );
}

#[salsa_test]
fn generic_record_spread_preserves_all_type_arguments(db: &salsa::DatabaseImpl) {
    let valid = r#"
struct Pair(a, b) { first: a, second: b }
fn spread_only(base: Pair(Int, Bool)) -> Pair(Int, Bool) { Pair { ..base } }
fn partial(base: Pair(Int, Bool)) -> Pair(Int, Bool) { Pair { first: +1, ..base } }
fn full(base: Pair(Int, Bool)) -> Pair(Int, Bool) {
    Pair { first: +1, second: True, ..base }
}
"#;
    assert_eq!(diagnostics(db, valid), vec![]);

    let invalid = r#"
struct Pair(a, b) { first: a, second: b }
fn spread_only(base: Pair(Bool, Int)) -> Pair(Int, Bool) { Pair { ..base } }
fn partial(base: Pair(Bool, Int)) -> Pair(Int, Bool) { Pair { first: +1, ..base } }
fn full(base: Pair(Bool, Int)) -> Pair(Int, Bool) {
    Pair { first: +1, second: True, ..base }
}
"#;
    let errors = diagnostics(db, invalid);
    assert_eq!(errors.len(), 3, "{errors:#?}");
    for error in errors {
        assert_eq!(error.phase, CompilationPhase::TypeChecking);
        assert!(
            error.inner.message.contains("expected `Int`, found `Bool`"),
            "{error:#?}"
        );
    }
}

#[salsa_test]
fn generic_callable_record_field_keeps_context(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "generic_callable_record.trb",
        r#"
struct Mapper(a, b) { map: fn(a) ->{} b }
fn mapper() -> Mapper(Int, Bool) { Mapper { map: fn(value) { True } } }
"#,
    );
    assert_eq!(ast_pipeline_diagnostics(db, source), vec![]);
    let checked = tribute_front::query::type_check_output(db, source).unwrap();
    let module = checked.module(db);
    let mapper_id = declared_struct_id(db, module, "Mapper");
    let mapper = typed_function_tail(checked.module(db), "mapper");
    assert_named_args(
        db,
        node_type(checked.expression_types(db), mapper),
        mapper_id,
        "Mapper",
        &[TypeKind::Int, TypeKind::Bool],
    );
    let ExprKind::Record { fields, .. } = &*mapper.kind else {
        panic!("mapper must end in a record");
    };
    let lambda = &fields[0].1;
    let signature = checked
        .lambda_signatures(db)
        .iter()
        .find_map(|(id, signature)| (*id == lambda.id).then_some(signature))
        .expect("missing lambda signature");
    assert!(matches!(
        signature.function_type.kind(db),
        TypeKind::Func { params, result, .. }
            if matches!(params.as_slice(), [param] if matches!(param.kind(db), TypeKind::Int))
                && matches!(result.kind(db), TypeKind::Bool)
    ));
    let TypeKind::Func { params, result, .. } = record_type_name(mapper).ty.kind(db) else {
        panic!("mapper constructor must have a function type");
    };
    assert_eq!(*result, node_type(checked.expression_types(db), mapper));
    assert_eq!(params.as_slice(), &[signature.function_type]);
}

#[salsa_test]
fn generic_phantom_record_keeps_expected_type_argument(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "generic_phantom_record.trb",
        r#"
struct Tag(a) {}
fn tag() -> Tag(Int) { Tag {} }
"#,
    );
    assert_eq!(ast_pipeline_diagnostics(db, source), vec![]);
    let checked = tribute_front::query::type_check_output(db, source).unwrap();
    let module = checked.module(db);
    let tag_id = declared_struct_id(db, module, "Tag");
    let metadata = checked.expression_types(db);
    let expected = [TypeKind::Int];

    assert_named_args(
        db,
        node_type(metadata, typed_function_body(module, "tag")),
        tag_id,
        "Tag",
        &expected,
    );
    let record = typed_function_tail(module, "tag");
    assert_named_args(db, node_type(metadata, record), tag_id, "Tag", &expected);
    assert_named_args(db, record_type_name(record).ty, tag_id, "Tag", &expected);
}

#[salsa_test]
fn generic_nonempty_phantom_record_keeps_expected_type_argument(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "generic_nonempty_phantom_record.trb",
        r#"
struct Tagged(a) { value: Nat }
fn tagged() -> Tagged(Int) { Tagged { value: 1 } }
"#,
    );
    assert_eq!(ast_pipeline_diagnostics(db, source), vec![]);
    let checked = tribute_front::query::type_check_output(db, source).unwrap();
    let module = checked.module(db);
    let tagged_id = declared_struct_id(db, module, "Tagged");
    let metadata = checked.expression_types(db);
    let expected = [TypeKind::Int];

    assert_named_args(
        db,
        node_type(metadata, typed_function_body(module, "tagged")),
        tagged_id,
        "Tagged",
        &expected,
    );
    let record = typed_function_tail(module, "tagged");
    assert_named_args(
        db,
        node_type(metadata, record),
        tagged_id,
        "Tagged",
        &expected,
    );
    let TypeKind::Func { params, result, .. } = record_type_name(record).ty.kind(db) else {
        panic!("nonempty struct constructor must have a function type");
    };
    assert!(matches!(params.as_slice(), [param] if matches!(param.kind(db), TypeKind::Nat)));
    assert_named_args(db, *result, tagged_id, "Tagged", &expected);
}

#[salsa_test]
fn pure_let_phantom_record_remains_polymorphic(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "pure_let_phantom_record.trb",
        r#"
struct Tag(a) {}
fn tags() -> #(Tag(Int), Tag(Bool)) {
    let tag = Tag {}
    #(tag, tag)
}
"#,
    );
    assert_eq!(ast_pipeline_diagnostics(db, source), vec![]);
    let checked = tribute_front::query::type_check_output(db, source).unwrap();
    let module = checked.module(db);
    let tag_id = declared_struct_id(db, module, "Tag");
    let tail = typed_function_tail(module, "tags");
    let ExprKind::Tuple(uses) = &*tail.kind else {
        panic!("tags must end in a tuple");
    };
    assert_eq!(uses.len(), 2);
    assert_named_args(
        db,
        node_type(checked.expression_types(db), &uses[0]),
        tag_id,
        "Tag",
        &[TypeKind::Int],
    );
    assert_named_args(
        db,
        node_type(checked.expression_types(db), &uses[1]),
        tag_id,
        "Tag",
        &[TypeKind::Bool],
    );
}

#[salsa_test]
fn generic_phantom_record_spread_rejects_incompatible_argument(db: &salsa::DatabaseImpl) {
    let invalid = r#"
struct Tag(a) {}
fn tag_bad(base: Tag(Bool)) -> Tag(Int) { Tag { ..base } }
"#;
    let errors = diagnostics(db, invalid);
    assert_eq!(errors.len(), 1, "{errors:#?}");
    assert!(
        errors[0]
            .inner
            .message
            .contains("expected `Int`, found `Bool`"),
        "{errors:#?}"
    );
}

#[salsa_test]
fn generic_callable_record_rejects_wrong_result(db: &salsa::DatabaseImpl) {
    let invalid = r#"
struct Mapper(a, b) { map: fn(a) ->{} b }
fn mapper_bad() -> Mapper(Int, Bool) { Mapper { map: fn(value) { +1 } } }
"#;
    let errors = diagnostics(db, invalid);
    assert_eq!(errors.len(), 1, "{errors:#?}");
    assert!(
        errors[0]
            .inner
            .message
            .contains("expected `Bool`, found `Int`")
    );
}

#[salsa_test]
fn generic_record_never_field_coercion_is_directional(db: &salsa::DatabaseImpl) {
    let valid = r#"
ability Stop { op stop() -> Never }
struct Box(a) { value: a }
fn abort() ->{Stop} Never { Stop::stop() }
fn box_nat() ->{Stop} Box(Nat) { Box { value: abort() } }
"#;
    assert_eq!(diagnostics(db, valid), vec![]);

    let invalid = r#"
struct Box(a) { value: a }
fn bad() -> Box(Never) { Box { value: 1 } }
"#;
    let errors = diagnostics(db, invalid);
    assert_eq!(errors.len(), 1, "{errors:#?}");
    assert!(
        errors[0]
            .inner
            .message
            .contains("expected `Never`, found `Nat`"),
        "{errors:#?}"
    );
}

/// Test basic record construction with correct field types.
/// This should compile successfully.
#[salsa_test]
fn test_record_field_type_correct(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
struct Point { x: Int, y: Int }

fn make_point() -> Point {
    Point { x: 10, y: 20 }
}
"#,
    );

    run_ast_pipeline(db, source);
}

/// Test record construction with multiple field types.
#[salsa_test]
fn test_record_mixed_field_types(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
struct Person { name: String, age: Int, active: Bool }

fn make_person() -> Person {
    Person { name: "Alice", age: 30, active: True }
}
"#,
    );

    run_ast_pipeline(db, source);
}

/// Test record construction with spread operator.
/// The spread expression should be constrained to the struct type.
#[salsa_test]
fn test_record_spread_same_type(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
struct Point { x: Int, y: Int }

fn update_x(p: Point) -> Point {
    Point { x: 100, ..p }
}
"#,
    );

    run_ast_pipeline(db, source);
}

/// Test record construction with only spread (no explicit fields).
#[salsa_test]
fn test_record_spread_only(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
struct Config { debug: Bool, verbose: Bool }

fn copy_config(c: Config) -> Config {
    Config { ..c }
}
"#,
    );

    run_ast_pipeline(db, source);
}

/// Test record with generic type parameter.
#[salsa_test]
fn test_record_generic_type(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
struct Pair(a, b) { first: a, second: b }

fn make_pair() -> Pair(Int, Bool) {
    Pair { first: 42, second: True }
}
"#,
    );

    run_ast_pipeline(db, source);
}

/// Test record field type inference in let binding.
#[salsa_test]
fn test_record_field_type_inference(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
struct Point { x: Int, y: Int }

fn test() -> Int {
    let p = Point { x: 1, y: 2 }
    p.x
}
"#,
    );

    run_ast_pipeline(db, source);
}

// ========================================================================
// Snapshot Tests
// ========================================================================

/// Snapshot test for basic record construction IR.
#[salsa_test]
fn test_snapshot_record_construction(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
struct Point { x: Int, y: Int }

fn make_point() -> Point {
    Point { x: 10, y: 20 }
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Snapshot test for record with spread operator.
#[salsa_test]
fn test_snapshot_record_spread(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
struct Point { x: Int, y: Int }

fn update_x(p: Point) -> Point {
    Point { x: 100, ..p }
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Snapshot test for generic record construction.
#[salsa_test]
fn test_snapshot_record_generic(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
struct Pair(a, b) { first: a, second: b }

fn make_pair() -> Pair(Int, Bool) {
    Pair { first: 42, second: True }
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Snapshot test for record with spread only (no explicit fields).
/// All fields should be extracted via `adt.struct_get` from the base.
#[salsa_test]
fn test_snapshot_record_spread_only(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
struct Config { debug: Bool, verbose: Bool }

fn copy_config(c: Config) -> Config {
    Config { ..c }
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Snapshot test for record with all fields explicit plus spread.
/// Explicit fields should take priority over spread values.
#[salsa_test]
fn test_snapshot_record_spread_all_fields_explicit(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
struct Point { x: Int, y: Int }

fn replace_all(p: Point) -> Point {
    Point { x: 1, y: 2, ..p }
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Test record spread with a function call as the spread expression.
#[salsa_test]
fn test_record_spread_complex_expr(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
struct Point { x: Int, y: Int }

fn origin() -> Point {
    Point { x: 0, y: 0 }
}

fn shift_x() -> Point {
    Point { x: 10, ..origin() }
}
"#,
    );

    run_ast_pipeline(db, source);
}

// ========================================================================
// Forward Reference Tests
// ========================================================================

/// Test record construction where the function using the record appears
/// before the struct definition (forward reference).
///
/// This tests that prescan_struct_fields correctly registers field orders
/// before lowering, regardless of declaration order in the source.
#[salsa_test]
fn test_record_forward_reference(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn make_point() -> Point {
    Point { x: 1, y: 2 }
}

struct Point { x: Int, y: Int }
"#,
    );

    // Should compile without ICE, emitting adt.struct_new
    run_ast_pipeline(db, source);
}
