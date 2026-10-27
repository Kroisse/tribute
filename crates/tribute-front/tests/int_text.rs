//! Frontend coverage for the canonical public Int text API.

mod common;

use self::common::{ast_pipeline_error_messages, run_ast_pipeline, run_ast_pipeline_with_ir};
use rustc_hash::FxHashMap as HashMap;
use salsa_test_macros::salsa_test;
use tribute_front::SourceCst;
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
fn generic_constructor_stores_int_in_its_specialized_layout(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "generic_constructor_int.trb",
        r#"
enum Boxed(a) {
    Empty,
    Box(a),
}

fn format_box(value: Int) -> String {
    case Boxed::Box(value) {
        .Box(inner) -> Int::to_string(inner)
        .Empty -> "empty"
    }
}
"#,
    );

    let ir = run_ast_pipeline_with_ir(db, source);
    assert!(
        ir.contains(r#"!"Boxed$Int" = adt.enum<"Boxed$Int" { Empty(), Box(core.i32) }>"#),
        "Boxed(Int) must be specialized to a layout with an Int field:\n{ir}"
    );
    let format_box = ir
        .split("tribute_control.func @format_box")
        .nth(1)
        .expect("format_box function must be lowered");
    let construct = format_box
        .find(r#"adt.variant_new %0 {tag = "Box", type = !"Boxed$Int"}"#)
        .expect("Box(value) must construct the specialized Boxed$Int");
    let int_to_string = format_box
        .find("callee = @\"std::Int::to_string\"")
        .expect("Box payload must be passed to Int::to_string");
    assert!(
        format_box[construct..int_to_string].contains(
            r#"adt.variant_get %9 {field = 0, tag = "Box", type = !"Boxed$Int"} : core.i32"#
        ),
        "pattern use must read the Int payload from the specialized layout:\n{format_box}"
    );
    assert!(
        !format_box.contains("tribute_rt.box_int") && !format_box.contains("tribute_rt.unbox_int"),
        "a specialized layout must hold Int without boxing:\n{format_box}"
    );
}

#[salsa_test]
fn generic_result_constructor_stores_int_in_its_specialized_layout(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "generic_result_constructor_int.trb",
        r#"
enum Result(a, e) {
    Ok(a),
    Error(e),
}

fn ok_int(value: Int) -> Result(Int, String) {
    .Ok(value)
}
"#,
    );

    let ir = run_ast_pipeline_with_ir(db, source);
    let ok_int = ir
        .split("tribute_control.func @ok_int")
        .nth(1)
        .expect("ok_int function must be lowered");
    assert!(
        ok_int.contains(r#"adt.variant_new %0 {tag = "Ok", type = !"Result$Int$std::String"}"#),
        ".Ok(value) must construct the specialized Result(Int, String) from the Int directly:\n{ok_int}"
    );
    assert!(
        !ok_int.contains("tribute_rt.box_int"),
        "a specialized layout must hold Int without boxing:\n{ok_int}"
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
    .IntBox(value)
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
    let ast = parsed.module(db);
    let checked = tribute_front::typeck::typecheck_module(
        db,
        &tribute_front::resolve::resolve_with_env(
            db,
            ast,
            &tribute_front::resolve::build_env(
                db,
                ast,
                tribute_front::resolve::Dependencies::none(),
            ),
            parsed.span_map(db),
        ),
        parsed.span_map(db),
    );
    let mono = tribute_front::monomorphize::monomorphize_functions(
        db,
        tribute_front::monomorphize::SpecializationInput::new(db, checked, None),
    )
    .expect("checked instances must specialize");
    let mut ir = IrContext::new();
    let output = tribute_front::ast_to_ir::LoweringInput::from_specialization(
        db,
        mono,
        tribute_front::ast_to_ir::LoweringEnvironment {
            span_map: tribute_front::ast::SpanMaps::new([parsed.span_map(db)]),
            ..Default::default()
        },
    )
    .lower_to_ir(db, &mut ir, source.uri(db).as_str());
    let ir_text = print_module(&ir, output.module.op());
    assert!(
        ir_text.contains("generic_intrinsic$Nat"),
        "logical lowering must resolve the concrete generic extern signature:\n{ir_text}"
    );
}

// The callers supply the tracked accumulator context.
fn lower_monomorphized_source(
    db: &dyn salsa::Database,
    source: SourceCst,
) -> (IrContext, tribute_front::ast_to_ir::FrontendIrModule) {
    let parsed = tribute_front::query::parsed_ast(db, source).expect("fixture must parse");
    let ast = parsed.module(db);
    let checked = tribute_front::typeck::typecheck_module(
        db,
        &tribute_front::resolve::resolve_with_env(
            db,
            ast,
            &tribute_front::resolve::build_env(
                db,
                ast,
                tribute_front::resolve::Dependencies::none(),
            ),
            parsed.span_map(db),
        ),
        parsed.span_map(db),
    );
    let mono = tribute_front::monomorphize::monomorphize_functions(
        db,
        tribute_front::monomorphize::SpecializationInput::new(db, checked, None),
    )
    .expect("checked instances must specialize");
    let mut ir = IrContext::new();
    let output = tribute_front::ast_to_ir::LoweringInput::from_specialization(
        db,
        mono,
        tribute_front::ast_to_ir::LoweringEnvironment {
            span_map: tribute_front::ast::SpanMaps::new([parsed.span_map(db)]),
            ..Default::default()
        },
    )
    .lower_to_ir(db, &mut ir, source.uri(db).as_str());
    (ir, output)
}

#[salsa::tracked(returns(clone))]
fn generic_specialization_transports_direct_callee_metadata_inner(
    db: &dyn salsa::Database,
    source: SourceCst,
) -> String {
    let (ir, output) = lower_monomorphized_source(db, source);
    let ir_text = print_module(&ir, output.module.op());
    assert!(
        ir_text.contains("tribute_control.func @\"apply$Int\""),
        "the specialized generic function must preserve its checked concrete call type:\n{ir_text}"
    );
    ir_text
}

/// The logical module declares each ability instance once, with its
/// operations in name order; repeated uses and handler arms add nothing.
#[salsa::tracked]
fn public_logical_output_declarations_inner(db: &dyn salsa::Database, source: SourceCst) {
    let parsed = tribute_front::query::parsed_ast(db, source).expect("fixture must parse");
    let ast = parsed.module(db);
    let span_map = parsed.span_map(db);
    let resolved = tribute_front::resolve::resolve_with_env(
        db,
        ast,
        &tribute_front::resolve::build_env(db, ast, tribute_front::resolve::Dependencies::none()),
        span_map,
    );
    let checked = tribute_front::typeck::typecheck_module(db, &resolved, span_map);
    let mono = tribute_front::monomorphize::monomorphize_functions(
        db,
        tribute_front::monomorphize::SpecializationInput::new(db, checked, None),
    )
    .expect("checked instances must specialize");
    let mut ir = IrContext::new();
    let output = tribute_front::ast_to_ir::LoweringInput::from_specialization(
        db,
        mono,
        tribute_front::ast_to_ir::LoweringEnvironment {
            span_map: tribute_front::ast::SpanMaps::new([span_map]),
            ..Default::default()
        },
    )
    .lower_to_ir(db, &mut ir, source.uri(db).as_str());
    let declarations =
        &tribute_ir::dialect::tribute_control::operation_declarations(&ir, output.module);
    assert_eq!(
        declarations.len(),
        4,
        "repeats must dedupe within each instance"
    );
    assert_eq!(ir.str(declarations[0].op_name), "bounce");
    assert_eq!(ir.str(declarations[0].kind), "op");
    assert_eq!(ir.str(declarations[1].op_name), "echo");
    assert_eq!(ir.str(declarations[1].kind), "fn");
    assert_eq!(declarations[0].ability, declarations[1].ability);
    assert_eq!(ir.str(declarations[2].op_name), "bounce");
    assert_eq!(ir.str(declarations[2].kind), "op");
    assert_eq!(ir.str(declarations[3].op_name), "echo");
    assert_eq!(ir.str(declarations[3].kind), "fn");
    assert_eq!(declarations[2].ability, declarations[3].ability);
    assert_ne!(
        declarations[0].ability, declarations[2].ability,
        "equal operation names at distinct ability instantiations must not dedupe"
    );
    for declaration in declarations.iter().take(2) {
        assert_eq!(declaration.parameter_types.len(), 1);
        let parameter = ir.get_type(declaration.parameter_types[0]);
        let result = ir.get_type(declaration.result_type);
        assert_eq!(parameter.name, trunk_ir::dialect::core::I32::NAME);
        assert_eq!(result.name, trunk_ir::dialect::core::I32::NAME);
    }
    for declaration in declarations.iter().skip(2) {
        assert_eq!(declaration.parameter_types.len(), 1);
        let parameter = ir.get_type(declaration.parameter_types[0]);
        let result = ir.get_type(declaration.result_type);
        assert_eq!(parameter.name, trunk_ir::dialect::core::I1::NAME);
        assert_eq!(result.name, trunk_ir::dialect::core::I1::NAME);
    }
    let validation = tribute_ir::dialect::tribute_control::validate(
        &ir,
        output.module,
        &output.compiler_intrinsics,
        &mut Default::default(),
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

fn use_int() ->{Audit(Int)} Int {
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
    use tribute_ir::dialect::tribute_control::{Call, Func, FuncSig, Lambda};
    use trunk_ir::ops::{DialectOp, DialectType};
    use trunk_ir::walk::{WalkAction, walk_typed};

    let (ir, output) = lower_monomorphized_source(db, source);
    let validation = tribute_ir::dialect::tribute_control::validate(
        &ir,
        output.module,
        &output.compiler_intrinsics,
        &mut Default::default(),
    );
    assert!(
        validation.is_ok(),
        "including retained generic bodies: {validation}"
    );
    let mut functions = HashMap::default();
    let _: ControlFlow<()> =
        walk_typed::<Func, ()>(&ir, output.module.body(&ir).unwrap(), &mut |func| {
            functions.insert(func.sym_name(&ir), func);
            ControlFlow::Continue(WalkAction::Skip)
        });
    for (argument, primitive) in [
        ("Int", trunk_ir::dialect::core::I32::NAME),
        ("Bool", trunk_ir::dialect::core::I1::NAME),
    ] {
        let parent = functions[format!("apply${argument}").as_str()];
        let consumer = functions[format!("pure${argument}").as_str()];
        let parent_signature = FuncSig::from_type_ref(&ir, parent.r#type(&ir)).unwrap();
        let data_type = parent_signature.result(&ir);
        assert_eq!(parent_signature.inputs(&ir), [data_type]);
        assert_eq!(ir.get_type(data_type).name, primitive);
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
        assert_eq!(calls[0].callee(&ir), [consumer.sym_name(&ir)]);
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
