//! Checked local callable instances must survive source-logical lowering.
mod common;

use salsa_test_macros::salsa_test;
use tribute_front::SourceCst;

#[salsa_test]
fn named_operator_preserves_target_and_consumer_conventions(db: &salsa::DatabaseImpl) {
    use std::ops::ControlFlow;
    use tribute_ir::dialect::tribute_control::{self, CallingConvention};
    use trunk_ir::Symbol;
    use trunk_ir::context::IrContext;
    use trunk_ir::ops::DialectOp;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::refs::ValueDef;
    use trunk_ir::walk::{WalkAction, walk_op};

    for argument in ["(Int::+)", "alias"] {
        let code = format!(
            r#"
fn pure(f: fn(Int, Int) ->{{}} Int) ->{{}} Int {{ f(+3, +4) }}
fn open(f: fn(Int, Int) ->{{e}} Int, x: Int, y: Int) ->{{e}} Int {{ f(x, y) }}
fn main() {{
    let add = (Int::+)
    let alias = add
    let first = pure({argument})
    open({argument}, first, +2)
    Nil
}}
"#
        );
        let source = SourceCst::from_source_str(db, "named_operator.trb", &code);
        let errors = common::ast_pipeline_error_messages(db, source);
        assert!(errors.is_empty(), "{argument}: {errors:?}");
        let text = common::run_ast_pipeline_with_ir(db, source);
        assert!(!text.contains("unrealized_conversion_cast"), "{text}");
        let mut ir = IrContext::new();
        let module = parse_test_module(&mut ir, &text);
        let mut conventions = Vec::new();
        let _ = walk_op::<()>(&ir, module.op(), &mut |op| {
            if let Ok(call) = tribute_control::Call::from_op(&ir, op)
                && [Symbol::new("pure"), Symbol::new("open")].contains(&call.callee(&ir))
            {
                let callback = ir.op_operands(op)[0];
                let ValueDef::OpResult(producer, _) = ir.value_def(callback) else {
                    panic!("expected named callback: {text}");
                };
                let reference = tribute_control::FuncRef::from_op(&ir, producer).unwrap();
                assert_eq!(reference.func_ref(&ir), Symbol::new("Int::+"));
                conventions.push(tribute_control::func_sig_convention(
                    &ir,
                    ir.value_ty(callback),
                ));
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
        assert_eq!(
            conventions,
            [
                Some(CallingConvention::Direct),
                Some(CallingConvention::Cps)
            ],
            "{argument}: {text}"
        );
    }
}

#[salsa_test]
fn fixed_local_identity_reaches_pure_consumer(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "local_identity.trb",
        r#"
fn apply(f: fn(Int) ->{} Int, x: Int) ->{} Int { f(x) }
fn main() ->{} Int {
    let identity = fn(x: Int) x
    apply(identity, +3)
}
"#,
    );
    let ir = common::run_ast_pipeline_with_ir(db, source);
    assert!(ir.contains("tribute_control.lambda"));
    assert!(
        ir.lines().any(|line| {
            line.contains("tribute_control.lambda(") && line.contains("convention(direct)")
        }),
        "a pure lambda passed to an effect-free consumer must be Direct:\n{ir}"
    );
    assert!(!ir.contains("unrealized_conversion_cast"), "{ir}");
}

#[salsa_test]
fn escaping_lambda_retains_open_callable_contract(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "escaping_lambda.trb",
        r#"
fn make() -> fn(Int) -> Int { fn(x) x }
fn main() {}
"#,
    );
    let ir = common::run_ast_pipeline_with_ir(db, source);
    assert!(
        ir.lines().any(|line| {
            line.contains("tribute_control.lambda(") && line.contains("convention(cps)")
        }),
        "a returned lambda must retain the open callable contract:\n{ir}"
    );
}

#[salsa_test]
fn open_effect_consumer_keeps_lambda_cps(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "open_lambda_consumer.trb",
        r#"
fn open(f: fn(Int) ->{e} Int, x: Int) ->{e} Int { f(x) }
fn main() { open(fn(x) x, +3) }
"#,
    );
    let ir = common::run_ast_pipeline_with_ir(db, source);
    assert!(
        ir.lines().any(|line| {
            line.contains("tribute_control.lambda(") && line.contains("convention(cps)")
        }),
        "an open-effect consumer must receive a CPS lambda:\n{ir}"
    );
}

#[salsa_test]
fn effectful_lambda_keeps_effect_convention(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "effectful_lambda_convention.trb",
        r#"
ability State(s) {
    op get() -> s
}

fn run(f: fn() ->{State(Int)} Int) -> Int {
    handle f() {
        do result { result }
        op State::get() { resume +0 }
    }
}

fn main() -> Int { run(fn() State::get()) }
"#,
    );
    let ir = common::run_ast_pipeline_with_ir(db, source);
    assert!(
        ir.contains("tribute_control.lambda() -> core.i32 convention(cps)"),
        "a lambda that performs an effect must remain CPS:\n{ir}"
    );
}

#[salsa_test]
fn fixed_captured_local_reaches_pure_consumer(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "local_capture.trb",
        r#"
fn apply(f: fn(Int) ->{} Int, x: Int) ->{} Int { f(x) }
fn captured(k: Int) ->{} Int {
    let action = fn(x: Int) k
    apply(action, +3)
}

fn main() ->{} Int { captured(+9) }
"#,
    );
    let ir = common::run_ast_pipeline_with_ir(db, source);
    assert!(ir.contains("tribute_control.lambda"));
    assert!(!ir.contains("unrealized_conversion_cast"), "{ir}");
}

#[salsa_test]
fn fixed_local_has_separate_pure_and_open_worker_surfaces(db: &salsa::DatabaseImpl) {
    for calls in [
        "let a = pure(action, +3)\nopen(action, a)",
        "let a = open(action, +3)\npure(action, a)",
    ] {
        let code = format!(
            r#"
fn pure(f: fn(Int) ->{{}} Int, x: Int) ->{{}} Int {{ f(x) }}
fn open(f: fn(Int) ->{{e}} Int, x: Int) ->{{e}} Int {{ f(x) }}
fn main() ->{{}} Int {{
    let action = fn(x: Int) x
    {calls}
}}
"#
        );
        let source = SourceCst::from_source_str(db, "multi_surface.trb", &code);
        let ir = common::run_ast_pipeline_with_ir(db, source);
        assert_eq!(ir.matches("tribute_control.lambda").count(), 2, "{ir}");
        assert!(!ir.contains("unrealized_conversion_cast"), "{ir}");
    }
}

#[salsa_test]
fn repeated_instance_and_alias_share_materialization(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "alias.trb",
        r#"
fn pure(f: fn(Int) ->{} Int, x: Int) ->{} Int { f(x) }
fn main() ->{} Int {
    let action = fn(x: Int) x
    let alias = action
    let first = pure(alias, +3)
    pure(action, first)
}
"#,
    );
    let ir = common::run_ast_pipeline_with_ir(db, source);
    assert_eq!(ir.matches("tribute_control.lambda").count(), 1, "{ir}");
    assert!(!ir.contains("unrealized_conversion_cast"), "{ir}");
}

#[salsa_test]
fn named_alias_uses_adapter_without_obsolete_capture(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "named_capture.trb",
        r#"
fn identity(x: Int) ->{} Int { x }
fn pure(f: fn(Int) ->{} Int, x: Int) ->{} Int { f(x) }
fn open(f: fn(Int) ->{e} Int, x: Int) ->{e} Int { f(x) }
fn main() ->{} Int {
    let named = identity
    let alias = named
    pure(fn(x: Int) open(alias, x), +3)
}
"#,
    );
    let ir = common::run_ast_pipeline_with_ir(db, source);
    assert!(ir.contains("captures []"), "{ir}");
    assert!(!ir.contains("unrealized_conversion_cast"), "{ir}");
}

#[salsa_test]
fn named_alias_adapts_for_indirect_consumer(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "indirect_consumer.trb",
        r#"
fn identity(x: Int) ->{} Int { x }
fn call(consumer: fn(fn(Int) ->{e} Int, Int) ->{e} Int) ->{e} Int {
    let named = identity
    let alias = named
    consumer(alias, +3)
}
fn main() {}
"#,
    );
    let ir = common::run_ast_pipeline_with_ir(db, source);
    assert!(!ir.contains("unrealized_conversion_cast"), "{ir}");
}

#[salsa_test]
fn shadowed_bindings_materialize_independently(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "shadowed.trb",
        r#"
fn pure(f: fn(Int) ->{} Int, x: Int) ->{} Int { f(x) }
fn main() ->{} Int {
    let action = fn(x: Int) x
    let first = pure(action, +3)
    let action = fn(x: Int) first
    pure(action, +9)
}
"#,
    );
    let ir = common::run_ast_pipeline_with_ir(db, source);
    assert_eq!(ir.matches("tribute_control.lambda").count(), 2, "{ir}");
    assert!(!ir.contains("unrealized_conversion_cast"), "{ir}");
}

#[salsa_test]
fn effectful_local_is_rejected_by_pure_consumer(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "effectful_local.trb",
        r#"
ability State(s) { fn get() -> s }
fn apply(f: fn(Int) ->{} Int, x: Int) ->{} Int { f(x) }
fn effectful(x: Int) ->{State(Int)} Int { State::get() + x }
fn invalid() ->{State(Int)} Int {
    let action = effectful
    apply(action, +3)
}
fn main() {}
"#,
    );
    let errors = common::ast_pipeline_error_messages(db, source);
    assert!(
        errors.iter().any(|error| error.contains("effect mismatch")),
        "{errors:?}"
    );
}

#[salsa_test]
fn local_named_consumer_retains_its_worker_parameter_convention(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "local_consumer.trb",
        r#"
fn open(f: fn(Int) ->{e} Int, x: Int) ->{e} Int { f(x) }
fn main() ->{} Int {
    let consume = open
    let alias = consume
    let action = fn(x: Int) x
    alias(action, +3)
}
"#,
    );
    let ir = common::run_ast_pipeline_with_ir(db, source);
    assert!(!ir.contains("unrealized_conversion_cast"), "{ir}");
}
