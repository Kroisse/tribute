//! Salsa integration tests for AST-based compilation pipeline (arena IR).

use salsa::{Database as _, Setter as _};
use salsa_test_macros::salsa_test;
use tree_sitter::Parser;
use tribute::{SourceCst, TributeDatabaseImpl, compile_frontend};
use tribute_ir::dialect::tribute_control;
use tribute_passes::diagnostic::Diagnostic;
use trunk_ir::dialect::arith;
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::{Attribute, ValueDef};
use trunk_ir::{IrContext, Module};

/// The frontend owns source-logical definitions, before physical CPS parameters.
fn source_function(ctx: &IrContext, module: &Module, name: &str) -> tribute_control::Func {
    let matches: Vec<_> = module
        .ops(ctx)
        .iter()
        .filter_map(|&op| {
            tribute_control::Func::from_op(ctx, op)
                .ok()
                .filter(|function| function.sym_name(ctx) == name)
        })
        .collect();
    assert_eq!(matches.len(), 1, "expected one source function {name}");
    matches[0]
}

fn assert_source_signature(
    ctx: &IrContext,
    module: &Module,
    name: &str,
    arity: usize,
    result: &str,
) {
    let function = source_function(ctx, module, name);
    let signature = tribute_control::FuncSig::from_type_ref(ctx, function.r#type(ctx))
        .expect("complete logical signature");
    assert_eq!(signature.inputs(ctx).len(), arity);
    assert!(
        signature
            .inputs(ctx)
            .iter()
            .all(|&ty| ctx.types.get(ty).dialect == "core" && ctx.types.get(ty).name == "i32")
    );
    assert_eq!(ctx.types.get(signature.result(ctx)).name, result);
    let entry = ctx.region(function.body(ctx)).blocks[0];
    assert_eq!(
        ctx.block_args(entry)
            .iter()
            .map(|&arg| ctx.value_ty(arg))
            .collect::<Vec<_>>(),
        signature.inputs(ctx)
    );
}

fn returned_constant(ctx: &IrContext, module: &Module, name: &str) -> i128 {
    let function = source_function(ctx, module, name);
    let entry = ctx.region(function.body(ctx)).blocks[0];
    let terminator = *ctx.block(entry).ops.last().expect("function terminator");
    let returned = tribute_control::Return::from_op(ctx, terminator).expect("logical return");
    let ValueDef::OpResult(op, _) = ctx.value_def(returned.value(ctx)) else {
        panic!("expected returned constant");
    };
    let constant = arith::Const::from_op(ctx, op).expect("returned value must be constant");
    let Attribute::Int(value) = constant.value(ctx) else {
        panic!("expected integer");
    };
    value
}

fn compile_frontend_with_diagnostics(
    db: &dyn salsa::Database,
    source: SourceCst,
) -> Result<(IrContext, Module), Vec<&Diagnostic>> {
    compile_frontend(db, source).ok_or_else(|| {
        tribute::pipeline::parse_and_lower_ast::accumulated::<Diagnostic>(db, source)
    })
}

fn expect_compilation_success(db: &dyn salsa::Database, source: SourceCst) -> (IrContext, Module) {
    compile_frontend_with_diagnostics(db, source)
        .unwrap_or_else(|diagnostics| panic!("compilation should succeed: {diagnostics:?}"))
}

#[salsa_test]
fn test_salsa_database_examples(db: &salsa::DatabaseImpl) {
    // Example source code
    let examples = vec![
        (
            "hello.trb",
            r#"fn main() { print_line("Hello, World!") }"#,
            vec!["main"],
        ),
        (
            "calc.trb",
            r#"fn main() { let _ = 1 + 2 + 3 }"#,
            vec!["main"],
        ),
        (
            "complex.trb",
            r#"
fn factorial(n: Nat) -> Nat {
  case n {
    0 -> 1
    _ -> n * factorial(n - 1)
  }
}

fn main() {
  let _ = factorial(5)
}
"#,
            vec!["factorial", "main"],
        ),
    ];

    for (filename, source_code, expected_funcs) in examples {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_tribute::LANGUAGE.into())
            .expect("Failed to set language");
        let tree = parser.parse(source_code, None).expect("tree");
        let source_file = SourceCst::from_path(db, filename, source_code.into(), Some(tree));
        let (ctx, module) = expect_compilation_success(db, source_file);

        for func_name in expected_funcs {
            let (arity, result) = if func_name == "factorial" {
                (1, "i32")
            } else {
                (0, "nil")
            };
            assert_source_signature(&ctx, &module, func_name, arity, result);
        }
    }
}

#[salsa_test]
fn test_compile_frontend_with_diagnostics_returns_errors(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(db, "invalid.trb", "fn main() -> Int { true }");

    let Err(diagnostics) = compile_frontend_with_diagnostics(db, source) else {
        panic!("invalid source should return diagnostics");
    };

    assert!(!diagnostics.is_empty());
}

#[test]
fn test_salsa_incremental_computation_detailed() {
    // Demonstrate incremental computation via compile_frontend
    let mut db = TributeDatabaseImpl::default();
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_tribute::LANGUAGE.into())
        .expect("Failed to set language");
    let text = "fn value() -> Nat { 3 } fn main() {}";
    let tree = parser.parse(text, None).expect("tree");
    let source_file = SourceCst::from_path(&db, "incremental.trb", text.into(), Some(tree));
    let (ctx1, module1) = expect_compilation_success(&db, source_file);
    assert_source_signature(&ctx1, &module1, "value", 0, "i32");
    assert_eq!(returned_constant(&ctx1, &module1, "value"), 3);

    let updated_text = "fn value() -> Nat { 10 } fn main() {}";
    let updated_tree = parser.parse(updated_text, None).expect("tree");
    source_file.set_text(&mut db).to(updated_text.into());
    source_file.set_tree(&mut db).to(Some(updated_tree));
    let (ctx2, module2) = expect_compilation_success(&db, source_file);
    assert_source_signature(&ctx2, &module2, "value", 0, "i32");
    assert_eq!(returned_constant(&ctx2, &module2, "value"), 10);

    let (ctx3, module3) = expect_compilation_success(&db, source_file);
    assert_source_signature(&ctx3, &module3, "value", 0, "i32");
    assert_eq!(returned_constant(&ctx3, &module3, "value"), 10);
    assert_eq!(
        returned_constant(&ctx2, &module2, "value"),
        returned_constant(&ctx3, &module3, "value")
    );
}

#[salsa_test]
fn test_salsa_multiple_functions(db: &salsa::DatabaseImpl) {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_tribute::LANGUAGE.into())
        .expect("Failed to set language");
    let text = r#"
fn add(a: Nat, b: Nat) -> Nat { a + b }
fn multiply(a: Nat, b: Nat) -> Nat { a * b }
fn main() { print_line("test") }
"#;
    let tree = parser.parse(text, None).expect("tree");
    let source = SourceCst::from_path(db, "multi.trb", text.into(), Some(tree));
    let (ctx, module) = expect_compilation_success(db, source);

    assert_source_signature(&ctx, &module, "add", 2, "i32");
    assert_source_signature(&ctx, &module, "multiply", 2, "i32");
    assert_source_signature(&ctx, &module, "main", 0, "nil");
}

#[test]
fn test_salsa_database_isolation() {
    // Test that different database instances are isolated
    let module1_name = TributeDatabaseImpl::default().attach(|db| {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_tribute::LANGUAGE.into())
            .expect("Failed to set language");
        let text = "fn main() { let _ = 1 + 2 }";
        let tree = parser.parse(text, None).expect("tree");
        let source1 = SourceCst::from_path(db, "test1.trb", text.into(), Some(tree));
        let (ctx, module) = expect_compilation_success(db, source1);
        module.name(&ctx).map(|s| s.to_string())
    });

    let module2_name = TributeDatabaseImpl::default().attach(|db| {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_tribute::LANGUAGE.into())
            .expect("Failed to set language");
        let text = "fn main() { let _ = 3 * 4 }";
        let tree = parser.parse(text, None).expect("tree");
        let source2 = SourceCst::from_path(db, "test2.trb", text.into(), Some(tree));
        let (ctx, module) = expect_compilation_success(db, source2);
        module.name(&ctx).map(|s| s.to_string())
    });

    // Module names are derived from file paths
    assert_eq!(module1_name, Some("test1".to_string()));
    assert_eq!(module2_name, Some("test2".to_string()));
}

#[salsa_test]
fn test_function_lowering(db: &salsa::DatabaseImpl) {
    let source = "fn main() { let _ = 1 + 2 }";
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_tribute::LANGUAGE.into())
        .expect("Failed to set language");
    let tree = parser.parse(source, None).expect("tree");
    let source_file = SourceCst::from_path(db, "func_test.trb", source.into(), Some(tree));
    let (ctx, module) = expect_compilation_success(db, source_file);

    assert_source_signature(&ctx, &module, "main", 0, "nil");
}
