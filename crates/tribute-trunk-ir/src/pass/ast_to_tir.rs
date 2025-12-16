//! AST to TrunkIR lowering pass.
//!
//! This pass converts the parsed AST into TrunkIR operations.
//! At this stage, names are unresolved (using `src` dialect ops).

use crate::{Attribute, BlockBuilder, Type, Value, arith, core, func, src};
use tribute_ast::{CallExpression, Expr, FunctionDefinition, ItemKind, Program, Statement};
use tribute_core::{Location, PathId, Span, Spanned};

/// Lower an AST program to a TrunkIR module.
pub fn lower_program<'db>(
    db: &'db dyn salsa::Database,
    path: PathId<'db>,
    program: Program<'db>,
) -> core::Module<'db> {
    let location = Location::new(path, Span::new(0, 0));

    core::Module::build(db, location, "main", |top| {
        for item in program.items(db) {
            if let ItemKind::Function(func_def) = item.kind(db) {
                let func = lower_function(db, path, *func_def);
                top.op(func);
            }
        }
    })
}

/// Lower a function definition to a func.func operation.
fn lower_function<'db>(
    db: &'db dyn salsa::Database,
    path: PathId<'db>,
    func_def: FunctionDefinition<'db>,
) -> func::Func<'db> {
    let name = func_def.name(db);
    let span = func_def.span(db);
    let location = Location::new(path, span);

    // For now, assume no type annotations (all Unknown)
    let params: Vec<Type> = func_def
        .parameters(db)
        .iter()
        .map(|_| Type::Unit) // Placeholder until type inference
        .collect();

    // Result type is also unknown until inference
    let results = vec![Type::Unit];

    func::Func::build(db, location, &name, params, results, |entry| {
        let body = func_def.body(db);

        // Lower each statement, keeping track of the last value
        let mut last_value: Option<Value<'db>> = None;

        for stmt in &body.statements {
            match stmt {
                Statement::Expression(spanned_expr) => {
                    last_value = Some(lower_expr(db, path, entry, spanned_expr));
                }
                Statement::Let(_let_stmt) => {
                    // TODO: Handle let bindings
                    todo!("let bindings not yet implemented")
                }
            }
        }

        // Return the last expression value (or unit if empty)
        if let Some(value) = last_value {
            entry.op(func::Return::value(db, location, value));
        } else {
            entry.op(func::Return::empty(db, location));
        }
    })
}

/// Lower an expression to TrunkIR operations.
fn lower_expr<'db>(
    db: &'db dyn salsa::Database,
    path: PathId<'db>,
    block: &mut BlockBuilder<'db>,
    spanned: &Spanned<Expr>,
) -> Value<'db> {
    let (expr, span) = spanned;
    let location = Location::new(path, *span);

    match expr {
        // Literals → arith.const
        Expr::Nat(n) => {
            let op = block.op(arith::Const::i64(db, location, *n as i64));
            op.result(db)
        }
        Expr::Int(n) => {
            let op = block.op(arith::Const::i64(db, location, *n));
            op.result(db)
        }
        Expr::Float(f) => {
            let op = block.op(arith::Const::f64(db, location, *f));
            op.result(db)
        }
        Expr::Bool(b) => {
            // Use i1 for booleans
            let op = block.op(arith::Const::new(
                db,
                location,
                Type::I { bits: 1 },
                Attribute::Int(if *b { 1 } else { 0 }),
            ));
            op.result(db)
        }
        Expr::Nil => {
            // Nil as unit constant
            let op = block.op(arith::Const::new(db, location, Type::Unit, Attribute::Unit));
            op.result(db)
        }

        // Variable reference → src.var
        Expr::Identifier(name) => {
            let op = block.op(src::Var::new(
                db,
                location,
                Type::Unit, // Unknown until resolution
                Attribute::String(name.clone()),
            ));
            op.result(db)
        }

        // Function call → src.call
        Expr::Call(CallExpression {
            function,
            arguments,
        }) => {
            // Lower arguments first
            let args: Vec<Value<'db>> = arguments
                .iter()
                .map(|arg| lower_expr(db, path, block, arg))
                .collect();

            let op = block.op(src::Call::new(
                db,
                location,
                args,
                Type::Unit, // Unknown until resolution
                Attribute::String(function.clone()),
            ));
            op.result(db)
        }

        // Not yet implemented
        Expr::StringInterpolation(_) => todo!("string interpolation"),
        Expr::BytesInterpolation(_) => todo!("bytes interpolation"),
        Expr::Rune(_) => todo!("rune literals"),
        Expr::Binary(_) => todo!("binary expressions"),
        Expr::MethodCall(_) => todo!("method calls"),
        Expr::Match(_) => todo!("match expressions"),
        Expr::Lambda(_) => todo!("lambda expressions"),
        Expr::Block(_) => todo!("block expressions"),
        Expr::List(_) => todo!("list literals"),
        Expr::Tuple(_, _) => todo!("tuple literals"),
        Expr::Record(_) => todo!("record expressions"),
        Expr::OperatorFn(_) => todo!("operator functions"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DialectOp;
    use salsa::Database;
    use std::path::PathBuf;
    use tribute_core::TributeDatabaseImpl;

    /// Helper tracked function to create AST and lower it.
    #[salsa::tracked]
    fn lower_simple_function_helper(db: &dyn salsa::Database) -> crate::Operation<'_> {
        let path = PathId::new(db, PathBuf::from("test.tr"));

        // Create a simple AST: fn main() { 42 }
        let body = tribute_ast::Block {
            statements: vec![Statement::Expression((Expr::Nat(42), Span::new(14, 16)))],
        };

        let func_def =
            FunctionDefinition::new(db, "main".to_string(), vec![], None, body, Span::new(0, 18));

        let item = tribute_ast::Item::new(db, ItemKind::Function(func_def), Span::new(0, 18));

        let program = Program::new(db, vec![item]);

        // Lower to TrunkIR and return the underlying operation
        lower_program(db, path, program).operation()
    }

    #[test]
    fn test_lower_simple_function() {
        TributeDatabaseImpl::default().attach(|db| {
            let op = lower_simple_function_helper(db);
            let module = core::Module::from_operation(db, op).unwrap();

            // Verify module structure
            assert_eq!(module.name(db), "main");

            // Get the function from the module's body
            let body_region = module.body(db);
            let blocks = body_region.blocks(db);
            assert_eq!(blocks.len(), 1);

            let top_block = &blocks[0];
            let ops = top_block.operations(db);
            assert_eq!(ops.len(), 1); // One func.func

            // Verify it's a func.func operation
            let func_op = func::Func::from_operation(db, ops[0]).unwrap();
            assert_eq!(func_op.name(db), "main");
        });
    }
}
