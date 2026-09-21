//! Frontend coverage for the canonical public Int text API.

mod common;

use self::common::{ast_pipeline_error_messages, run_ast_pipeline, run_ast_pipeline_with_ir};
use salsa_test_macros::salsa_test;
use tribute_front::SourceCst;
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::printer::print_module;

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
    let layout_cast = format_box
        .find("core.unrealized_conversion_cast")
        .expect("generic constructor field must be converted to its erased layout type");
    let construct = format_box
        .find("adt.variant_new")
        .expect("generic constructor must construct Box");
    let int_to_string = format_box
        .find("callee = @\"Int::to_string\"")
        .expect("Box payload must be passed to Int::to_string");
    let payload_recovery = format_box[..int_to_string]
        .rfind("core.unrealized_conversion_cast")
        .expect("erased Box payload must be recovered before Int::to_string");
    assert!(
        format_box[layout_cast..construct].contains(": tribute_rt.anyref"),
        "generic Box field must be converted to its erased layout type before construction:\n{format_box}"
    );
    assert!(
        format_box[payload_recovery..int_to_string].contains(": core.i32"),
        "pattern use must recover the erased Box payload as Int before Int::to_string:\n{format_box}"
    );
    assert!(
        layout_cast < construct && construct < payload_recovery && payload_recovery < int_to_string,
        "generic Box construction and pattern use must cross distinct layout and recovery boundaries:\n{format_box}"
    );
}

#[salsa_test]
fn generic_result_constructor_boxes_int_for_its_specialized_layout(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "generic_result_constructor_int.trb",
        r#"
enum Result(a, e) {
    Ok(a),
    Error(e),
}

fn ok_int(value: Int) -> Result(Int, String) {
    Ok(value)
}
"#,
    );

    let ir = run_ast_pipeline_with_ir(db, source);
    let ok_int = ir
        .split("tribute_control.func @ok_int")
        .nth(1)
        .expect("ok_int function must be lowered");
    let construct = ok_int
        .find("adt.variant_new")
        .expect("generic Result constructor must construct Ok");
    let cast = ok_int
        .find("core.unrealized_conversion_cast")
        .expect("generic Result field must be explicitly converted to its erased layout type");
    assert!(
        ok_int[cast..construct].contains(": tribute_rt.anyref"),
        "generic Result field must be converted to the resolved erased layout type:\n{ok_int}"
    );
    assert!(
        cast < construct,
        "logical generic construction must convert its Int field before adt.variant_new:\n{ok_int}"
    );
}

#[salsa_test]
fn concrete_constructor_does_not_insert_a_redundant_cast(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "concrete_constructor_int.trb",
        r#"
enum IntBox {
    IntBox(Int),
}

fn box_int(value: Int) -> IntBox {
    IntBox(value)
}
"#,
    );

    let ir = run_ast_pipeline_with_ir(db, source);
    let box_int = ir
        .split("tribute_control.func @box_int")
        .nth(1)
        .expect("box_int function must be lowered");
    assert!(box_int.contains("adt.variant_new"), "{box_int}");
    assert!(
        !box_int.contains("core.unrealized_conversion_cast"),
        "already-matching concrete constructor fields must not be converted:\n{box_int}"
    );
}

/// Direct-call metadata preserves the exact concrete instantiation used when
/// cloning a generic source function for source-logical lowering.
#[salsa_test]
fn generic_specialization_transports_direct_callee_metadata(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "generic_lambda_metadata.trb",
        r#"
fn apply(value: a) -> a {
    value
}

fn use_apply() -> Int { apply(+41) }
"#,
    );

    generic_specialization_transports_direct_callee_metadata_inner(db, source);
}

/// A generic extern discovered through a concrete clone has no AST body to
/// specialize, but source-logical lowering still needs its exact callable
/// scheme under the mangled identity.
#[salsa_test]
fn generic_extern_specialization_has_a_logical_signature(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "generic_extern_metadata.trb",
        r#"
extern "intrinsic" fn generic_intrinsic(value: a) -> a

fn through(value: a) -> a {
    generic_intrinsic(value)
}

fn use_through() -> Nat { through(0) }
"#,
    );

    generic_extern_specialization_has_a_logical_signature_inner(db, source);
}

#[salsa::tracked]
fn generic_extern_specialization_has_a_logical_signature_inner(
    db: &dyn salsa::Database,
    source: SourceCst,
) {
    let parsed = tribute_front::query::parsed_ast(db, source).expect("fixture must parse");
    let ast = parsed.module(db).clone();
    let checked = tribute_front::typeck::typecheck_module(
        db,
        tribute_front::resolve::resolve_with_env(
            db,
            ast.clone(),
            tribute_front::resolve::build_env(db, &ast),
            parsed.span_map(db).clone(),
        ),
        parsed.span_map(db).clone(),
    );
    let typed =
        tribute_front::tdnr::resolve_tdnr(db, checked.module(db).clone(), std::iter::empty());
    let mono = tribute_front::monomorphize::monomorphize_functions(
        db,
        typed,
        checked.function_types(db).iter().cloned().collect(),
        tribute_front::monomorphize::MonomorphizeMetadata {
            constructor_types: checked
                .constructor_types(db)
                .schemes
                .iter()
                .cloned()
                .collect(),
            specialized_enum_variants: checked
                .constructor_types(db)
                .specialized_enum_variants
                .iter()
                .cloned()
                .collect(),
            local_instances: checked
                .expression_types(db)
                .local_instances
                .iter()
                .cloned()
                .collect(),
            function_instances: checked
                .expression_types(db)
                .function_instances
                .iter()
                .cloned()
                .collect(),
            node_types: checked
                .expression_types(db)
                .node_types
                .iter()
                .cloned()
                .collect(),
            handler_operations: checked.handler_operations(db).iter().cloned().collect(),
            perform_operations: checked.perform_operations(db).iter().cloned().collect(),
            lambda_signatures: checked.lambda_signatures(db).iter().cloned().collect(),
            exhaustive_cases: checked.exhaustive_cases(db).iter().copied().collect(),
        },
    )
    .expect("checked instances must specialize");
    let mut ir = IrContext::new();
    let output = tribute_front::ast_to_ir::TypedModule {
        ast: mono.module,
        span_map: checked.span_map(db).clone(),
        function_types: mono.function_types.into_iter().collect(),
        constructor_types: mono.metadata.constructor_types,
        specialized_enum_variants: mono.metadata.specialized_enum_variants,
        node_types: mono.metadata.node_types,
        local_instances: mono.metadata.local_instances,
        ability_conventions: checked.ability_conventions(db).iter().cloned().collect(),
        ability_definitions: tribute_front::typeck::ability_definitions_from_schemas(
            checked.ability_definitions(db),
        ),
        handler_operations: mono.metadata.handler_operations,
        perform_operations: mono.metadata.perform_operations,
        lambda_signatures: mono.metadata.lambda_signatures,
        exhaustive_cases: mono.metadata.exhaustive_cases,
        well_known_types: checked.well_known_types(db),
        compiler_intrinsics: std::collections::HashMap::new(),
    }
    .lower_to_ir(db, &mut ir, source.uri(db).as_str());
    let ir_text = print_module(&ir, output.module.op());
    assert!(
        ir_text.contains("generic_intrinsic$Nat"),
        "logical lowering must resolve the concrete generic extern signature:\n{ir_text}"
    );
}

// The callers supply the tracked accumulator context.
fn lower_specialized_source(
    db: &dyn salsa::Database,
    source: SourceCst,
) -> (IrContext, tribute_front::ast_to_ir::FrontendIrModule) {
    let parsed = tribute_front::query::parsed_ast(db, source).expect("fixture must parse");
    let ast = parsed.module(db).clone();
    let checked = tribute_front::typeck::typecheck_module(
        db,
        tribute_front::resolve::resolve_with_env(
            db,
            ast.clone(),
            tribute_front::resolve::build_env(db, &ast),
            parsed.span_map(db).clone(),
        ),
        parsed.span_map(db).clone(),
    );
    let typed =
        tribute_front::tdnr::resolve_tdnr(db, checked.module(db).clone(), std::iter::empty());
    let mono = tribute_front::monomorphize::monomorphize_functions(
        db,
        typed,
        checked.function_types(db).iter().cloned().collect(),
        tribute_front::monomorphize::MonomorphizeMetadata {
            constructor_types: checked
                .constructor_types(db)
                .schemes
                .iter()
                .cloned()
                .collect(),
            specialized_enum_variants: checked
                .constructor_types(db)
                .specialized_enum_variants
                .iter()
                .cloned()
                .collect(),
            local_instances: checked
                .expression_types(db)
                .local_instances
                .iter()
                .cloned()
                .collect(),
            function_instances: checked
                .expression_types(db)
                .function_instances
                .iter()
                .cloned()
                .collect(),
            node_types: checked
                .expression_types(db)
                .node_types
                .iter()
                .cloned()
                .collect(),
            handler_operations: checked.handler_operations(db).iter().cloned().collect(),
            perform_operations: checked.perform_operations(db).iter().cloned().collect(),
            lambda_signatures: checked.lambda_signatures(db).iter().cloned().collect(),
            exhaustive_cases: checked.exhaustive_cases(db).iter().copied().collect(),
        },
    )
    .expect("checked instances must specialize");
    let mut ir = IrContext::new();
    let output = tribute_front::ast_to_ir::TypedModule {
        ast: mono.module,
        span_map: checked.span_map(db).clone(),
        function_types: mono.function_types.into_iter().collect(),
        constructor_types: mono.metadata.constructor_types,
        specialized_enum_variants: mono.metadata.specialized_enum_variants,
        node_types: mono.metadata.node_types,
        local_instances: mono.metadata.local_instances,
        ability_conventions: checked.ability_conventions(db).iter().cloned().collect(),
        ability_definitions: tribute_front::typeck::ability_definitions_from_schemas(
            checked.ability_definitions(db),
        ),
        handler_operations: mono.metadata.handler_operations,
        perform_operations: mono.metadata.perform_operations,
        lambda_signatures: mono.metadata.lambda_signatures,
        exhaustive_cases: mono.metadata.exhaustive_cases,
        well_known_types: checked.well_known_types(db),
        compiler_intrinsics: std::collections::HashMap::new(),
    }
    .lower_to_ir(db, &mut ir, source.uri(db).as_str());
    (ir, output)
}

#[salsa::tracked]
fn generic_specialization_transports_direct_callee_metadata_inner(
    db: &dyn salsa::Database,
    source: SourceCst,
) -> String {
    let (ir, output) = lower_specialized_source(db, source);
    let ir_text = print_module(&ir, output.module.op());
    assert!(
        ir_text.contains("tribute_control.func @\"apply$Int\""),
        "the specialized generic function must preserve its checked concrete call type:\n{ir_text}"
    );
    ir_text
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
        local_instances: checked
            .expression_types(db)
            .local_instances
            .iter()
            .cloned()
            .collect(),
        span_map: checked.span_map(db).clone(),
        function_types: checked.function_types(db).iter().cloned().collect(),
        constructor_types: checked
            .constructor_types(db)
            .schemes
            .iter()
            .cloned()
            .collect(),
        specialized_enum_variants: checked
            .constructor_types(db)
            .specialized_enum_variants
            .iter()
            .cloned()
            .collect(),
        node_types: checked
            .expression_types(db)
            .node_types
            .iter()
            .cloned()
            .collect(),
        ability_conventions: checked.ability_conventions(db).iter().cloned().collect(),
        ability_definitions: tribute_front::typeck::ability_definitions_from_schemas(
            checked.ability_definitions(db),
        ),
        handler_operations: checked.handler_operations(db).iter().cloned().collect(),
        perform_operations: checked.perform_operations(db).iter().cloned().collect(),
        lambda_signatures: checked.lambda_signatures(db).iter().cloned().collect(),
        exhaustive_cases: checked.exhaustive_cases(db).iter().copied().collect(),
        well_known_types: checked.well_known_types(db),
        compiler_intrinsics: std::collections::HashMap::new(),
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
    let validation = tribute_ir::dialect::tribute_control::validate(
        &ir,
        output.module,
        declarations,
        &output.compiler_intrinsics,
    );
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

#[salsa_test]
fn outer_generic_specialization_rekeys_local_binding_instances(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "specialized_local.trb",
        r#"
fn pure(f: fn(Int) ->{} Int, value: Int) ->{} Int { f(value) }
fn apply(value: a) -> a {
    let identity = fn(x: Int) {
        let retained = value
        x
    }
    let alias = identity
    let result = pure(alias, +3)
    value
}
fn use_int() ->{} Int { apply(+3) }
fn use_bool() ->{} Bool { apply(True) }
"#,
    );
    let ir = generic_specialization_transports_direct_callee_metadata_inner(db, source);
    assert!(ir.contains("apply$Bool"), "{ir}");
    assert!(ir.contains("tribute_control.lambda"), "{ir}");
    assert!(!ir.contains("unrealized_conversion_cast"), "{ir}");
}

#[salsa_test]
fn outer_generic_arguments_determine_local_callable_signature(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "outer_local_signature.trb",
        r#"
fn pure(f: fn(a) ->{} a, value: a) ->{} a { f(value) }
fn apply(value: a) ->{} a {
    let identity = fn(x: a) x
    let alias = identity
    pure(alias, value)
}
fn use_int() ->{} Int { apply(+3) }
fn use_bool() ->{} Bool { apply(True) }
"#,
    );
    assert_outer_local_signatures(db, source);
    let errors = assert_outer_local_signatures::accumulated::<tribute_core::Diagnostic>(db, source);
    assert!(errors.is_empty(), "{errors:?}");
}

#[salsa::tracked]
fn assert_outer_local_signatures(db: &dyn salsa::Database, source: SourceCst) {
    use std::ops::ControlFlow;
    use tribute_ir::dialect::tribute_control::{Call, CallingConvention, Func, FuncSig, Lambda};
    use trunk_ir::ops::{DialectOp, DialectType};
    use trunk_ir::walk::{WalkAction, walk_typed};

    let (ir, output) = lower_specialized_source(db, source);
    let validation = tribute_ir::dialect::tribute_control::validate(
        &ir,
        output.module,
        &output.operation_declarations,
        &output.compiler_intrinsics,
    );
    assert!(
        validation.is_ok(),
        "including retained generic bodies: {validation}"
    );
    let mut functions = std::collections::HashMap::new();
    let _: ControlFlow<()> =
        walk_typed::<Func, ()>(&ir, output.module.body(&ir).unwrap(), &mut |func| {
            functions.insert(func.sym_name(&ir), func);
            ControlFlow::Continue(WalkAction::Skip)
        });
    for (argument, primitive) in [("Int", "i32"), ("Bool", "i1")] {
        let parent = functions[&Symbol::from_dynamic(&format!("apply${argument}"))];
        let consumer = functions[&Symbol::from_dynamic(&format!("pure${argument}"))];
        let parent_signature = FuncSig::from_type_ref(&ir, parent.r#type(&ir)).unwrap();
        let data_type = parent_signature.result(&ir);
        assert_eq!(parent_signature.inputs(&ir), [data_type]);
        assert_eq!(ir.types.get(data_type).dialect, Symbol::new("core"));
        assert_eq!(
            ir.types.get(data_type).name,
            Symbol::from_dynamic(primitive)
        );
        let mut lambdas = Vec::new();
        let mut calls = Vec::new();
        let _: ControlFlow<()> = trunk_ir::walk::walk_region(&ir, parent.body(&ir), &mut |op| {
            if let Ok(lambda) = Lambda::from_op(&ir, op) {
                lambdas.push(lambda);
                return ControlFlow::Continue(WalkAction::Skip);
            }
            if let Ok(call) = Call::from_op(&ir, op) {
                calls.push(call);
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
        assert_eq!(lambdas.len(), 1, "{argument}");
        assert_eq!(calls.len(), 1, "{argument}");
        let value = lambdas[0].result(&ir);
        let signature = FuncSig::from_type_ref(&ir, ir.value_ty(value)).unwrap();
        assert_eq!(signature.inputs(&ir), [data_type]);
        assert_eq!(signature.result(&ir), data_type);
        assert_eq!(signature.convention(&ir), CallingConvention::Direct);
        assert_eq!(calls[0].callee(&ir), consumer.sym_name(&ir));
        assert_eq!(
            calls[0].args(&ir)[0],
            value,
            "pass the actual lambda, without a cast"
        );
        let consumer_signature = FuncSig::from_type_ref(&ir, consumer.r#type(&ir)).unwrap();
        assert_eq!(
            consumer_signature.inputs(&ir),
            [signature.as_type_ref(), data_type]
        );
        assert_eq!(consumer_signature.result(&ir), data_type);
    }
}
