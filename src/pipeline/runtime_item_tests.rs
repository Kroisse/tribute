//! Runtime items through the pipeline: the frontend checks them and
//! publishes their types, target lowering calls runtime functions by
//! canonical path, and the cleanup after it removes the ones nothing calls.
//!
//! These tests stand a user module `rt` in for `std::runtime`, with their
//! own item set.

use super::*;
use rustc_hash::FxHashMap as HashMap;
use tribute_front::ast_to_ir::runtime_items::{RuntimeInputs, RuntimeItemErrorKind, check_runtime};
use tribute_ir::runtime_items::{RuntimeFunction, RuntimeItem, RuntimeItemKind, runtime_type};
use tribute_passes::abi_boundary::TargetKind;
use trunk_ir::SymbolPath;
use trunk_ir::refs::OpRef;
use trunk_ir::symbol_table::SymbolTable;
use trunk_ir::types::Attribute;

const RUNTIME: &[&str] = &["rt"];
const PROBE: &[&str] = &["rt", "probe"];
const ITEMS: &[RuntimeItem] = &[
    RuntimeItem {
        path: PROBE,
        kind: RuntimeItemKind::Function,
    },
    RuntimeItem {
        path: &["rt", "Cell"],
        kind: RuntimeItemKind::Struct,
    },
];

const PROGRAM: &str = r#"
mod rt {
    struct Cell { value: Int }

    fn probe(value: Int) ->{} Int {
        value
    }
}

fn main() -> Nil { }
"#;

fn source(db: &dyn salsa::Database, text: &str) -> SourceCst {
    SourceCst::from_source_str(db, "runtime_items.trb", text)
}

/// The definition `path` names in the root module, whatever its dialect.
fn definition(ctx: &IrContext, m: Module, path: &[&str]) -> Option<OpRef> {
    let reference = SymbolPath::new([tribute_ir::runtime_items::runtime_symbol(path)]);
    SymbolTable::collect(ctx, m).resolve_in(ctx, m.op(), &reference)
}

/// The program after the shared middle-end, with every source function
/// still present; the target entry is the first DCE it meets.
fn shared_output(db: &salsa::DatabaseImpl) -> (IrContext, Module) {
    salsa::Database::attach(db, |db| {
        compile_ast(db, source(db, PROGRAM))
            .expect("shared middle-end")
            .expect("frontend output")
    })
}

/// A stand-in for a target lowering that calls `rt::probe`: it finds the
/// function by path and calls it at the start of `main`.
fn call_probe_from_main(ctx: &mut IrContext, m: Module) -> OpRef {
    let probe = RuntimeFunction::lookup(ctx, m, PROBE).expect("the probe is kept alive");
    let main = definition(ctx, m, &["main"])
        .and_then(|op| func_dialect::Func::from_op(ctx, op).ok())
        .expect("main");
    let entry = ctx.region(main.body(ctx)).blocks[0];
    let first = ctx.block(entry).ops[0];
    let location = ctx.op(first).location;
    let input = probe.signature(ctx).inputs(ctx)[0];
    let argument = trunk_ir::dialect::arith::Const::operands()
        .value(Attribute::Int(7))
        .results(input)
        .build(ctx, location);
    ctx.insert_op_before(entry, first, argument.op_ref());
    let call = probe.call(ctx, location, [argument.result(ctx)]);
    ctx.insert_op_before(entry, first, call.op_ref());
    call.op_ref()
}

/// Run `stages` for `target` with the test runtime items.
fn run_target_stages(
    ctx: &mut IrContext,
    m: Module,
    target: TargetKind,
    stages: std::ops::RangeInclusive<Stage>,
) -> PassResult {
    let config = StageConfig {
        runtime: ITEMS,
        ..StageConfig::target(target)
    };
    run_stages(ctx, m, config, stages)
}

#[test]
fn frontend_publishes_runtime_types_with_the_prelude_types() {
    let db = salsa::DatabaseImpl::default();
    let (ctx, frontend) = salsa::Database::attach(&db, |db| {
        let source = source(db, PROGRAM);
        let typed = parse_and_lower_ast(db, source).expect("frontend output");
        // The prepared metadata is checked against the compiler's own item
        // set under `std::runtime`, which has no `rt` items; check the test
        // set here instead.
        let specialized =
            prepare_frontend_for_lowering(db, typed, source).expect("specialized frontend");
        let function_types: HashMap<_, _> = specialized
            .declarations(db)
            .function_types
            .iter()
            .cloned()
            .collect();
        let runtime_types = check_runtime(
            db,
            &RuntimeInputs {
                module: specialized.module(db),
                function_types: &function_types,
            },
            RUNTIME,
            ITEMS,
        )
        .expect("runtime items");
        let mut ctx = IrContext::new();
        let frontend = ast_to_ir::LoweringInput::from_specialization(
            db,
            specialized,
            ast_to_ir::LoweringEnvironment {
                span_map: frontend_spans(db, source),
                runtime_types,
                merged_sources: vec![crate::prelude::PRELUDE_URI.to_owned()],
            },
        )
        .lower_to_ir(db, &mut ctx, source.uri(db).as_str());
        (ctx, frontend)
    });
    let m = frontend.module;

    let cell = runtime_type(&ctx, m, &["rt", "Cell"]).expect("published runtime type");
    assert_eq!(
        trunk_ir::printer::print_type(&ctx, cell),
        r#"adt.struct<"rt::Cell"(value: core.i32)>"#
    );
    assert!(
        tribute_ir::metadata::WellKnownTypes::from_module(&ctx, m.op())
            .string
            .is_some(),
        "the prelude String shares the dictionary"
    );
}

#[test]
fn source_dce_keeps_runtime_functions() {
    for (items, kept) in [(ITEMS, true), (&[][..], false)] {
        let db = salsa::DatabaseImpl::default();
        let (mut ctx, m) = salsa::Database::attach(&db, |db| {
            compile_frontend(db, source(db, PROGRAM)).expect("frontend output")
        });
        let core_module = core_dialect::Module::from_op(&ctx, m.op()).expect("core.module");
        let mut pm = PassManager::new();
        pm.add_pass(
            tribute_passes::unreferenced_source_functions::EliminateUnreferencedSourceFunctions::keeping(items),
        );
        pm.run(&mut ctx, core_module, &mut AnalysisCache::new())
            .expect("source DCE");
        assert_eq!(definition(&ctx, m, PROBE).is_some(), kept);
    }
}

#[test]
fn source_dce_keeps_the_evidence_runtime_only_where_effects_lower_to_it() {
    const HANDLED: &str = r#"
ability Ask {
    fn ask() -> Nat
}

fn use_ask() ->{Ask} Nat {
    Ask::ask()
}

fn main() -> Nil {
    let _ = handle use_ask() {
        do result { result }
        fn Ask::ask() { 42 }
    }
}
"#;
    for (program, kept) in [(PROGRAM, false), (HANDLED, true)] {
        let db = salsa::DatabaseImpl::default();
        let (ctx, m) = salsa::Database::attach(&db, |db| {
            let frontend = compile_frontend_for_shared_route(db, source(db, program))
                .expect("frontend output");
            run_shared_middle_end(frontend).expect("shared middle-end")
        });
        assert_eq!(
            definition(&ctx, m, tribute_ir::runtime_items::evidence::LOOKUP).is_some(),
            kept,
            "{program}"
        );
    }
}

#[test]
fn unused_runtime_functions_do_not_reach_the_boundary_exit() {
    for target in [TargetKind::Native, TargetKind::Wasm] {
        let db = salsa::DatabaseImpl::default();
        let (mut ctx, m) = shared_output(&db);
        run_target_stages(&mut ctx, m, target, Stage::EnterTarget..=Stage::EnterTarget)
            .expect("target entry");
        assert!(
            RuntimeFunction::lookup(&ctx, m, PROBE).is_ok(),
            "{target:?}: the target entry keeps the probe"
        );

        run_target_stages(
            &mut ctx,
            m,
            target,
            Stage::LowerRuntimeCalls..=Stage::LowerRuntimeCalls,
        )
        .expect("runtime calls");
        run_target_stages(
            &mut ctx,
            m,
            target,
            Stage::ExitBoundary..=Stage::ExitBoundary,
        )
        .expect("boundary exit");
        assert_eq!(
            definition(&ctx, m, PROBE),
            None,
            "{target:?}: an uncalled runtime function reached the boundary exit"
        );
    }
}

#[test]
fn target_lowering_calls_runtime_functions_by_path() {
    for target in [TargetKind::Native, TargetKind::Wasm] {
        let db = salsa::DatabaseImpl::default();
        let (mut ctx, m) = shared_output(&db);
        run_target_stages(&mut ctx, m, target, Stage::EnterTarget..=Stage::EnterTarget)
            .expect("target entry");
        let call = call_probe_from_main(&mut ctx, m);
        run_target_stages(
            &mut ctx,
            m,
            target,
            Stage::LowerRuntimeCalls..=Stage::LowerRuntimeCalls,
        )
        .expect("runtime calls");
        run_target_stages(
            &mut ctx,
            m,
            target,
            Stage::ExitBoundary..=Stage::ExitBoundary,
        )
        .expect("boundary exit");

        let probe = RuntimeFunction::lookup(&ctx, m, PROBE)
            .unwrap_or_else(|error| panic!("{target:?}: {error}"));
        let callee = func_dialect::Call::from_op(&ctx, call)
            .ok()
            .and_then(|call| {
                SymbolTable::collect(&ctx, m).resolve(&ctx, call.op_ref(), call.callee(&ctx))
            });
        assert_eq!(callee, Some(probe.function().op_ref()), "{target:?}");

        let emitted = emit_from_boundary_exit(&mut ctx, m, target)
            .unwrap_or_else(|error| panic!("{target:?} emission: {error}"));
        assert!(!emitted.is_empty());
    }
}

mod check {
    use super::*;

    const ACCEPTED: &str = r#"
mod rt {
    ability State(s) {
        op get() -> s
    }

    struct Cell { value: Int }

    struct Boxed(a) { value: a }

    fn probe(value: Int) ->{} Int {
        value
    }

    fn read(cell: Cell) ->{} Int {
        cell.value
    }

    fn countdown(n: Int) ->{} Int {
        case n == +0 {
            True -> +0
            False -> countdown(n - +1)
        }
    }
}

fn main() -> Nil { }
"#;

    const REJECTED: &str = r#"
mod rt {
    ability State(s) {
        op get() -> s
    }

    fn inferred(value: Int) -> Int {
        value
    }

    fn identity(value: a) ->{} a {
        value
    }

    fn stateful() ->{State(Int)} Int {
        State::get()
    }

    fn apply(value: Int) ->{} Int {
        let f = fn(x: Int) x
        f(value)
    }

    fn depth(n: Int) ->{} Int {
        case n == +0 {
            True -> +0
            False -> depth(n - +1) + +1
        }
    }

    fn handled() ->{} Int {
        handle stateful() {
            do result { result }
            op State::get() { +0 }
        }
    }
}

fn helper(value: Int) -> Int {
    value
}

fn main() -> Nil { }
"#;

    fn item(path: &'static [&'static str], kind: RuntimeItemKind) -> RuntimeItem {
        RuntimeItem { path, kind }
    }

    /// Check `items` and the functions of `rt` in `text`, returning the
    /// number of published types or the failed kinds.
    fn check(text: &str, items: &[RuntimeItem]) -> Result<usize, Vec<RuntimeItemErrorKind>> {
        let db = salsa::DatabaseImpl::default();
        salsa::Database::attach(&db, |db| {
            let source = source(db, text);
            let typed = parse_and_lower_ast(db, source).expect("typechecked source");
            let diagnostics = parse_and_lower_ast::accumulated::<Diagnostic>(db, source);
            assert!(diagnostics.is_empty(), "{diagnostics:?}");
            let function_types: HashMap<_, _> = typed
                .declarations(db)
                .function_types
                .iter()
                .cloned()
                .collect();
            let inputs = RuntimeInputs {
                module: typed.module(db),
                function_types: &function_types,
            };
            check_runtime(db, &inputs, RUNTIME, items)
                .map(|types| types.0.len())
                .map_err(|errors| errors.into_iter().map(|error| error.kind).collect())
        })
    }

    #[test]
    fn accepts_runtime_code_within_the_constraints() {
        use RuntimeItemKind::*;
        assert_eq!(
            check(
                ACCEPTED,
                &[
                    item(&["rt", "probe"], Function),
                    item(&["rt", "read"], Function),
                    item(&["rt", "Cell"], Struct),
                ]
            ),
            Ok(1)
        );
    }

    #[test]
    fn rejects_missing_and_mismatched_items() {
        use RuntimeItemErrorKind::*;
        use RuntimeItemKind::*;
        assert_eq!(
            check(
                ACCEPTED,
                &[
                    item(&["rt", "absent"], Function),
                    item(&["rt", "Cell"], Function),
                    item(&["rt", "probe"], Struct),
                    item(&["rt", "State"], Function),
                    item(&["rt", "Boxed"], Struct),
                ]
            ),
            Err(vec![
                Missing,
                WrongKind("function with a body"),
                WrongKind("struct"),
                WrongKind("function with a body"),
                Polymorphic,
            ])
        );
    }

    #[test]
    fn reports_errors_at_their_declarations() {
        let db = salsa::DatabaseImpl::default();
        salsa::Database::attach(&db, |db| {
            let source = source(db, REJECTED);
            let typed = parse_and_lower_ast(db, source).expect("typechecked source");
            let function_types: HashMap<_, _> = typed
                .declarations(db)
                .function_types
                .iter()
                .cloned()
                .collect();
            let inputs = RuntimeInputs {
                module: typed.module(db),
                function_types: &function_types,
            };
            let errors = check_runtime(db, &inputs, RUNTIME, &[]).expect_err("rejected");
            let span_map = tribute_front::query::span_map(db, source).expect("parsed source");
            let diagnostic = errors[0].diagnostic(&tribute_front::ast::SpanMaps::new([span_map]));
            assert_eq!(
                diagnostic.inner.message,
                "runtime `rt::inferred` must have an empty ability row, written `->{}`"
            );
            assert_eq!(diagnostic.inner.severity, DiagnosticSeverity::Error);
            assert!(REJECTED[diagnostic.inner.span.start..].starts_with("fn inferred"));
        });
    }

    #[test]
    fn rejects_every_runtime_function_outside_the_constraints() {
        use RuntimeItemErrorKind::*;
        assert_eq!(
            check(REJECTED, &[]),
            Err(vec![
                NotDirect,
                Polymorphic,
                NotDirect,
                ClosureCall,
                NonTailRecursion,
                Handler,
            ])
        );
    }
}
