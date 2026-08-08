//! Frontend coverage for the canonical public Int text API.

mod common;

use self::common::{ast_pipeline_error_messages, run_ast_pipeline, run_ast_pipeline_with_ir};
use salsa_test_macros::salsa_test;
use tribute_front::SourceCst;
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;

#[salsa_test]
fn canonical_int_text_api_resolves_through_prelude(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "int_text_api.trb",
        r#"
fn parse(input: String) -> Result(Int, Int::ParseError) {
    Int::parse(input)
}

fn format(value: Int) -> String {
    Int::to_string(value)
}
"#,
    );

    let errors = ast_pipeline_error_messages(db, source);
    assert!(
        errors.is_empty(),
        "canonical Int text API must type-check through the prelude: {errors:?}"
    );
    run_ast_pipeline(db, source);
}

#[salsa_test]
fn generic_constructor_boxes_int_for_its_specialized_layout(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "generic_constructor_int.trb",
        r#"
enum Boxed(a) {
    Empty,
    Box(a),
}

fn format_box(value: Int) -> String {
    case Box(value) {
        Box(inner) -> Int::to_string(inner)
        Empty -> "empty"
    }
}
"#,
    );

    let ir = run_ast_pipeline_with_ir(db, source);
    let format_box = ir
        .split("tribute_control.func @format_box")
        .nth(1)
        .expect("format_box function must be lowered");
    let construct = format_box
        .find("adt.variant_new")
        .expect("generic constructor must construct Box");
    let cast = format_box
        .find("core.unrealized_conversion_cast")
        .expect("erased generic field must be recovered for Int::to_string");
    assert!(
        construct < cast,
        "logical generic construction must not use a cast as a control carrier before Box is built:\n{format_box}"
    );
}

/// The public typecheck-to-logical-lowering boundary carries deterministic,
/// exact operation declarations rather than reconstructing them from printed
/// operations. First source use is bounce, then echo; handler repeats dedupe.
#[salsa::tracked]
fn public_logical_output_declarations_inner(db: &dyn salsa::Database, source: SourceCst) {
    let parsed = tribute_front::query::parsed_ast(db, source).expect("fixture must parse");
    let ast = parsed.module(db).clone();
    let span_map = parsed.span_map(db).clone();
    let resolved = tribute_front::resolve::resolve_with_env(
        db,
        ast.clone(),
        tribute_front::resolve::build_env(db, &ast),
        span_map,
    );
    let checked =
        tribute_front::typeck::typecheck_module(db, resolved, parsed.span_map(db).clone());
    let typed =
        tribute_front::tdnr::resolve_tdnr(db, checked.module(db).clone(), std::iter::empty());
    let mut ir = IrContext::new();
    let output = tribute_front::ast_to_ir::TypedModule {
        ast: typed,
        span_map: checked.span_map(db).clone(),
        function_types: checked.function_types(db).iter().cloned().collect(),
        constructor_types: checked
            .nominal_types(db)
            .constructor_types
            .iter()
            .cloned()
            .collect(),
        node_types: checked
            .expression_types(db)
            .node_types
            .iter()
            .cloned()
            .collect(),
        call_callee_types: checked
            .expression_types(db)
            .call_callee_types
            .iter()
            .cloned()
            .collect(),
        specialized_call_callee_nodes: Default::default(),
        ability_conventions: checked.ability_conventions(db).iter().cloned().collect(),
        ability_definitions: tribute_front::typeck::ability_definitions_from_schemas(
            checked.ability_definitions(db),
        ),
        handler_operations: checked.handler_operations(db).iter().cloned().collect(),
        perform_operations: checked.perform_operations(db).iter().cloned().collect(),
        lambda_signatures: checked.lambda_signatures(db).iter().cloned().collect(),
        exhaustive_cases: checked.exhaustive_cases(db).iter().copied().collect(),
        well_known_types: checked.well_known_types(db),
    }
    .lower_to_ir(db, &mut ir, source.uri(db).as_str());
    let declarations = &output.operation_declarations;
    assert_eq!(
        declarations.len(),
        4,
        "repeats must dedupe within each instance"
    );
    assert_eq!(declarations[0].op_name, Symbol::new("bounce"));
    assert_eq!(declarations[0].kind, Symbol::new("op"));
    assert_eq!(declarations[1].op_name, Symbol::new("echo"));
    assert_eq!(declarations[1].kind, Symbol::new("fn"));
    assert_eq!(declarations[0].ability_ref, declarations[1].ability_ref);
    assert_eq!(declarations[2].op_name, Symbol::new("bounce"));
    assert_eq!(declarations[2].kind, Symbol::new("op"));
    assert_eq!(declarations[3].op_name, Symbol::new("echo"));
    assert_eq!(declarations[3].kind, Symbol::new("fn"));
    assert_eq!(declarations[2].ability_ref, declarations[3].ability_ref);
    assert_ne!(
        declarations[0].ability_ref, declarations[2].ability_ref,
        "equal operation names at distinct ability instantiations must not dedupe"
    );
    for declaration in declarations.iter().take(2) {
        assert_eq!(declaration.parameter_types.len(), 1);
        let parameter = ir.types.get(declaration.parameter_types[0]);
        let result = ir.types.get(declaration.result_type);
        assert_eq!(
            (parameter.dialect, parameter.name),
            (Symbol::new("core"), Symbol::new("i32"))
        );
        assert_eq!(
            (result.dialect, result.name),
            (Symbol::new("core"), Symbol::new("i32"))
        );
    }
    let ability = ir.types.get(declarations[0].ability_ref);
    assert_eq!(
        (ability.dialect, ability.name),
        (Symbol::new("core"), Symbol::new("ability_ref"))
    );
    assert_eq!(ability.params.len(), 1);
    assert_eq!(
        (
            ir.types.get(ability.params[0]).dialect,
            ir.types.get(ability.params[0]).name
        ),
        (Symbol::new("core"), Symbol::new("i32"))
    );
    let bool_ability = ir.types.get(declarations[2].ability_ref);
    assert_eq!(bool_ability.params.len(), 1);
    assert_eq!(
        (
            ir.types.get(bool_ability.params[0]).dialect,
            ir.types.get(bool_ability.params[0]).name
        ),
        (Symbol::new("core"), Symbol::new("i1"))
    );
    for declaration in declarations.iter().skip(2) {
        assert_eq!(declaration.parameter_types.len(), 1);
        let parameter = ir.types.get(declaration.parameter_types[0]);
        let result = ir.types.get(declaration.result_type);
        assert_eq!(
            (parameter.dialect, parameter.name),
            (Symbol::new("core"), Symbol::new("i1"))
        );
        assert_eq!(
            (result.dialect, result.name),
            (Symbol::new("core"), Symbol::new("i1"))
        );
    }
    let validation =
        tribute_ir::dialect::tribute_control::validate(&ir, output.module, declarations);
    assert!(
        validation.is_ok(),
        "public logical frontend output must pass complete validation: {validation}"
    );
}

#[salsa_test]
fn public_logical_output_retains_exact_operation_declarations(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "public_ability_metadata.trb",
        r#"
ability Audit(a) {
    fn echo(value: a) -> a
    op bounce(value: a) -> a
}

fn use() ->{Audit(Int)} Int {
    handle Audit::echo(Audit::bounce(+1)) {
        do result { result }
        fn Audit::echo(value) { value }
        op Audit::bounce(value) { resume value }
    }
}

fn use_bool() ->{Audit(Bool)} Bool {
    handle Audit::echo(Audit::bounce(True)) {
        do result { result }
        fn Audit::echo(value) { value }
        op Audit::bounce(value) { resume value }
    }
}
"#,
    );
    public_logical_output_declarations_inner(db, source);
}
