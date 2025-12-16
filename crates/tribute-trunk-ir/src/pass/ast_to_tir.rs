//! AST to TrunkIR lowering pass.
//!
//! This pass converts the parsed AST into TrunkIR operations.
//! At this stage, names are unresolved (using `src` dialect ops).

use std::collections::HashMap;

use crate::{Attribute, BlockBuilder, Region, Type, Value, arith, core, func, src};
use tribute_ast::{
    BinaryExpression, BinaryOperator, CallExpression, Expr, FunctionDefinition, ItemKind,
    LambdaExpression, LetStatement, Pattern, Program, Statement,
};
use tribute_core::{Location, PathId, Span, Spanned};

/// Context for lowering, tracking local variable bindings.
struct LoweringCtx<'db> {
    /// Map from variable names to their SSA values.
    bindings: HashMap<String, Value<'db>>,
}

impl<'db> LoweringCtx<'db> {
    fn new() -> Self {
        Self {
            bindings: HashMap::new(),
        }
    }

    /// Bind a name to a value.
    fn bind(&mut self, name: String, value: Value<'db>) {
        self.bindings.insert(name, value);
    }

    /// Look up a binding by name.
    fn lookup(&self, name: &str) -> Option<Value<'db>> {
        self.bindings.get(name).copied()
    }

    /// Execute a closure in a new scope. Bindings created inside are discarded after.
    fn scoped<R>(&mut self, f: impl FnOnce(&mut Self) -> R) -> R {
        let saved = self.bindings.clone();
        let result = f(self);
        self.bindings = saved;
        result
    }
}

/// Lower an AST program to a TrunkIR module.
#[salsa::tracked]
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
        let mut ctx = LoweringCtx::new();

        // Lower each statement, keeping track of the last value
        let mut last_value: Option<Value<'db>> = None;

        for stmt in &body.statements {
            match stmt {
                Statement::Expression(spanned_expr) => {
                    last_value = Some(lower_expr(db, path, &mut ctx, entry, spanned_expr));
                }
                Statement::Let(let_stmt) => {
                    lower_let(db, path, &mut ctx, entry, let_stmt);
                    // Let statements don't produce a value
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

/// Lower a let statement, binding the pattern to the value.
fn lower_let<'db>(
    db: &'db dyn salsa::Database,
    path: PathId<'db>,
    ctx: &mut LoweringCtx<'db>,
    block: &mut BlockBuilder<'db>,
    let_stmt: &LetStatement,
) {
    let value = lower_expr(db, path, ctx, block, &let_stmt.value);
    bind_pattern(ctx, &let_stmt.pattern, value);
}

/// Bind a pattern to a value, adding bindings to the context.
fn bind_pattern<'db>(ctx: &mut LoweringCtx<'db>, pattern: &Pattern, value: Value<'db>) {
    match pattern {
        Pattern::Identifier(name) => {
            ctx.bind(name.clone(), value);
        }
        Pattern::Wildcard => {
            // Discard the value, no binding
        }
        Pattern::As(inner, name) => {
            // Bind the whole value to the name, then recurse on inner pattern
            ctx.bind(name.clone(), value);
            bind_pattern(ctx, inner, value);
        }
        // Patterns that require runtime matching - todo for now
        Pattern::Literal(_) => todo!("literal pattern matching"),
        Pattern::Constructor(_) => todo!("constructor pattern matching"),
        Pattern::Tuple(_, _) => todo!("tuple pattern destructuring"),
        Pattern::List(_) => todo!("list pattern matching"),
        Pattern::Handler(_) => todo!("handler patterns"),
    }
}

/// Lower an expression to TrunkIR operations.
fn lower_expr<'db>(
    db: &'db dyn salsa::Database,
    path: PathId<'db>,
    ctx: &mut LoweringCtx<'db>,
    block: &mut BlockBuilder<'db>,
    spanned: &Spanned<Expr>,
) -> Value<'db> {
    let (expr, span) = spanned;
    let location = Location::new(path, *span);

    match expr {
        // Literals → arith.const
        Expr::Nat(n) => {
            let op = block.op(arith::Const::u64(db, location, *n));
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
                (*b).into(),
            ));
            op.result(db)
        }
        Expr::Nil => {
            // Nil as unit constant
            let op = block.op(arith::Const::new(db, location, Type::Unit, Attribute::Unit));
            op.result(db)
        }

        // Variable reference: check local bindings first, then emit src.var for unresolved
        Expr::Identifier(name) => {
            if let Some(value) = ctx.lookup(name) {
                // Found in local bindings - use the SSA value directly
                value
            } else {
                // Unresolved - emit src.var for later resolution
                let op = block.op(src::Var::new(
                    db,
                    location,
                    Type::Unit, // Unknown until resolution
                    Attribute::String(name.clone()),
                ));
                op.result(db)
            }
        }

        // Function call → src.call
        Expr::Call(CallExpression {
            function,
            arguments,
        }) => {
            // Lower arguments first
            let args: Vec<Value<'db>> = arguments
                .iter()
                .map(|arg| lower_expr(db, path, ctx, block, arg))
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

        // Binary expressions
        Expr::Binary(BinaryExpression {
            left,
            operator,
            qualifier: _, // TODO: Handle qualified operators
            right,
        }) => {
            let lhs = lower_expr(db, path, ctx, block, left);
            let rhs = lower_expr(db, path, ctx, block, right);
            lower_binary_op(db, location, block, operator.clone(), lhs, rhs)
        }

        // Not yet implemented
        Expr::StringInterpolation(_) => todo!("string interpolation"),
        Expr::BytesInterpolation(_) => todo!("bytes interpolation"),
        Expr::Rune(_) => todo!("rune literals"),
        Expr::MethodCall(_) => todo!("method calls"),
        Expr::Match(_) => todo!("match expressions"),
        Expr::Lambda(LambdaExpression {
            parameters,
            return_type: _, // TODO: use type annotation when available
            body,
        }) => {
            // Parameter types are unknown until type inference
            let param_types: Vec<Type> = parameters.iter().map(|_| Type::Unit).collect();
            let result_type = Type::Unit;

            // Create body block with parameters as block arguments
            let mut body_block = BlockBuilder::new(db, location).args(param_types.clone());

            // Lower body in a new scope with parameters bound to block arguments
            let result_value = ctx.scoped(|ctx| {
                // Bind parameters to block arguments
                // Note: we need the built block to get arg values, so we track indices
                for (i, param) in parameters.iter().enumerate() {
                    // Create a placeholder - will be resolved when block is built
                    // For now, emit src.var for parameter references
                    // The block args will be connected during a later pass
                    let param_value = body_block.op(src::Var::new(
                        db,
                        location,
                        Type::Unit,
                        Attribute::String(param.name.clone()),
                    ));
                    ctx.bind(param.name.clone(), param_value.result(db));
                    let _ = i; // suppress unused warning for now
                }

                // Lower the body expression
                lower_expr(db, path, ctx, &mut body_block, body)
            });

            // Add yield to return the lambda's result
            body_block.op(src::Yield::new(db, location, result_value));

            // Create the function type for the lambda
            let func_type = Type::Function {
                params: param_types,
                results: vec![result_type],
            };

            // Create the src.lambda operation
            let region = Region::new(db, location, vec![body_block.build()]);
            let lambda_op = block.op(src::Lambda::new(
                db,
                location,
                Type::Unit,
                Attribute::Type(func_type),
                region,
            ));
            lambda_op.result(db)
        }
        Expr::Block(statements) => {
            // Create body block for the block expression
            let mut body_block = BlockBuilder::new(db, location);

            // Lower statements in a new scope
            let result_value = ctx.scoped(|ctx| {
                lower_statements(db, path, ctx, &mut body_block, statements, location)
            });

            // Add yield to return the value from the block
            body_block.op(src::Yield::new(db, location, result_value));

            // Create the src.block operation
            let region = Region::new(db, location, vec![body_block.build()]);
            let block_op = block.op(src::Block::new(db, location, Type::Unit, region));
            block_op.result(db)
        }
        Expr::List(_) => todo!("list literals"),
        Expr::Tuple(_, _) => todo!("tuple literals"),
        Expr::Record(_) => todo!("record expressions"),
        Expr::OperatorFn(_) => todo!("operator functions"),
    }
}

/// Lower a sequence of statements, returning the last expression's value.
fn lower_statements<'db>(
    db: &'db dyn salsa::Database,
    path: PathId<'db>,
    ctx: &mut LoweringCtx<'db>,
    block: &mut BlockBuilder<'db>,
    statements: &[Statement],
    location: Location<'db>,
) -> Value<'db> {
    let mut last_value: Option<Value<'db>> = None;

    for stmt in statements {
        match stmt {
            Statement::Expression(spanned_expr) => {
                last_value = Some(lower_expr(db, path, ctx, block, spanned_expr));
            }
            Statement::Let(let_stmt) => {
                lower_let(db, path, ctx, block, let_stmt);
                last_value = None; // Let statements don't produce a value
            }
        }
    }

    // Return the last expression value, or unit if empty/ends with let
    last_value.unwrap_or_else(|| {
        let op = block.op(arith::Const::new(db, location, Type::Unit, Attribute::Unit));
        op.result(db)
    })
}

/// Lower a binary operation to the appropriate TrunkIR op.
fn lower_binary_op<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    block: &mut BlockBuilder<'db>,
    operator: BinaryOperator,
    lhs: Value<'db>,
    rhs: Value<'db>,
) -> Value<'db> {
    // Result type is unknown until type inference
    let result_ty = Type::Unit;

    match operator {
        // Arithmetic operations → arith dialect
        BinaryOperator::Add => block
            .op(arith::Add::new(db, location, lhs, rhs, result_ty))
            .result(db),
        BinaryOperator::Subtract => block
            .op(arith::Sub::new(db, location, lhs, rhs, result_ty))
            .result(db),
        BinaryOperator::Multiply => block
            .op(arith::Mul::new(db, location, lhs, rhs, result_ty))
            .result(db),
        BinaryOperator::Divide => block
            .op(arith::Div::new(db, location, lhs, rhs, result_ty))
            .result(db),
        BinaryOperator::Modulo => block
            .op(arith::Rem::new(db, location, lhs, rhs, result_ty))
            .result(db),

        // Comparison operations → arith dialect (result is i1)
        BinaryOperator::Equal => block
            .op(arith::CmpEq::new(
                db,
                location,
                lhs,
                rhs,
                Type::I { bits: 1 },
            ))
            .result(db),
        BinaryOperator::NotEqual => block
            .op(arith::CmpNe::new(
                db,
                location,
                lhs,
                rhs,
                Type::I { bits: 1 },
            ))
            .result(db),
        BinaryOperator::LessThan => block
            .op(arith::CmpLt::new(
                db,
                location,
                lhs,
                rhs,
                Type::I { bits: 1 },
            ))
            .result(db),
        BinaryOperator::LessEqual => block
            .op(arith::CmpLe::new(
                db,
                location,
                lhs,
                rhs,
                Type::I { bits: 1 },
            ))
            .result(db),
        BinaryOperator::GreaterThan => block
            .op(arith::CmpGt::new(
                db,
                location,
                lhs,
                rhs,
                Type::I { bits: 1 },
            ))
            .result(db),
        BinaryOperator::GreaterEqual => block
            .op(arith::CmpGe::new(
                db,
                location,
                lhs,
                rhs,
                Type::I { bits: 1 },
            ))
            .result(db),

        // Logical operations → arith bitwise (works for i1 booleans)
        BinaryOperator::And => block
            .op(arith::And::new(db, location, lhs, rhs, Type::I { bits: 1 }))
            .result(db),
        BinaryOperator::Or => block
            .op(arith::Or::new(db, location, lhs, rhs, Type::I { bits: 1 }))
            .result(db),

        // Concat needs type-directed resolution → src.binop
        BinaryOperator::Concat => {
            let op = block.op(src::Binop::new(
                db,
                location,
                lhs,
                rhs,
                result_ty,
                Attribute::String("concat".to_string()),
            ));
            op.result(db)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DialectOp;
    use salsa::Database;
    use std::path::PathBuf;
    use tribute_ast::{BinaryExpression, Parameter};
    use tribute_core::TributeDatabaseImpl;

    /// Helper tracked function to create AST and lower it.
    #[salsa::tracked]
    fn lower_simple_function_helper(db: &dyn salsa::Database) -> core::Module<'_> {
        let path = PathId::new(db, PathBuf::from("test.tr"));

        // Create a simple AST: fn main() { 42 }
        let body = tribute_ast::Block {
            statements: vec![Statement::Expression((Expr::Nat(42), Span::new(14, 16)))],
        };

        let func_def =
            FunctionDefinition::new(db, "main".to_string(), vec![], None, body, Span::new(0, 18));

        let item = tribute_ast::Item::new(db, ItemKind::Function(func_def), Span::new(0, 18));

        let program = Program::new(db, vec![item]);

        lower_program(db, path, program)
    }

    #[test]
    fn test_lower_simple_function() {
        TributeDatabaseImpl::default().attach(|db| {
            let module = lower_simple_function_helper(db);

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

    /// Helper to create AST with binary expression: fn main() { 1 + 2 }
    #[salsa::tracked]
    fn lower_binary_expr_helper(db: &dyn salsa::Database) -> core::Module<'_> {
        let path = PathId::new(db, PathBuf::from("test.tr"));

        // Create AST: fn main() { 1 + 2 }
        let binary_expr = Expr::Binary(BinaryExpression {
            left: Box::new((Expr::Nat(1), Span::new(14, 15))),
            operator: BinaryOperator::Add,
            qualifier: None,
            right: Box::new((Expr::Nat(2), Span::new(18, 19))),
        });

        let body = tribute_ast::Block {
            statements: vec![Statement::Expression((binary_expr, Span::new(14, 19)))],
        };

        let func_def =
            FunctionDefinition::new(db, "main".to_string(), vec![], None, body, Span::new(0, 21));

        let item = tribute_ast::Item::new(db, ItemKind::Function(func_def), Span::new(0, 21));
        let program = Program::new(db, vec![item]);

        lower_program(db, path, program)
    }

    #[test]
    fn test_lower_binary_expression() {
        TributeDatabaseImpl::default().attach(|db| {
            let module = lower_binary_expr_helper(db);

            // Get the function
            let body_region = module.body(db);
            let blocks = body_region.blocks(db);
            let func_op = func::Func::from_operation(db, blocks[0].operations(db)[0]).unwrap();

            // Get the function's entry block
            let func_body = func_op.body(db);
            let func_blocks = func_body.blocks(db);
            assert_eq!(func_blocks.len(), 1);

            let entry_block = &func_blocks[0];
            let ops = entry_block.operations(db);

            // Should have: arith.const(1), arith.const(2), arith.add, func.return
            assert_eq!(ops.len(), 4);

            // Verify the add operation
            let add_op = arith::Add::from_operation(db, ops[2]).unwrap();
            assert!(add_op.lhs(db) != add_op.rhs(db)); // lhs and rhs should be different values
        });
    }

    /// Helper to create AST with let binding: fn main() { let x = 42; x }
    #[salsa::tracked]
    fn lower_let_binding_helper(db: &dyn salsa::Database) -> core::Module<'_> {
        let path = PathId::new(db, PathBuf::from("test.tr"));

        // Create AST: fn main() { let x = 42; x }
        let let_stmt = Statement::Let(LetStatement {
            pattern: Pattern::Identifier("x".to_string()),
            value: (Expr::Nat(42), Span::new(12, 14)),
        });
        let ref_expr =
            Statement::Expression((Expr::Identifier("x".to_string()), Span::new(16, 17)));

        let body = tribute_ast::Block {
            statements: vec![let_stmt, ref_expr],
        };

        let func_def =
            FunctionDefinition::new(db, "main".to_string(), vec![], None, body, Span::new(0, 19));

        let item = tribute_ast::Item::new(db, ItemKind::Function(func_def), Span::new(0, 19));
        let program = Program::new(db, vec![item]);

        lower_program(db, path, program)
    }

    #[test]
    fn test_lower_let_binding() {
        TributeDatabaseImpl::default().attach(|db| {
            let module = lower_let_binding_helper(db);

            // Get the function
            let body_region = module.body(db);
            let blocks = body_region.blocks(db);
            let func_op = func::Func::from_operation(db, blocks[0].operations(db)[0]).unwrap();

            // Get the function's entry block
            let func_body = func_op.body(db);
            let func_blocks = func_body.blocks(db);
            let entry_block = &func_blocks[0];
            let ops = entry_block.operations(db);

            // Should have: arith.const(42), func.return
            // No src.var because 'x' is resolved to the const value directly
            assert_eq!(ops.len(), 2);

            // Verify first op is arith.const
            let const_op = arith::Const::from_operation(db, ops[0]).unwrap();
            let const_value = const_op.result(db);

            // Verify return uses the same value
            let ret_op = func::Return::from_operation(db, ops[1]).unwrap();
            assert_eq!(ret_op.operands(db).len(), 1);
            assert_eq!(ret_op.operands(db)[0], const_value);
        });
    }

    /// Helper to create AST with block expression: fn main() { { let x = 1; x + 2 } }
    #[salsa::tracked]
    fn lower_block_expr_helper(db: &dyn salsa::Database) -> core::Module<'_> {
        let path = PathId::new(db, PathBuf::from("test.tr"));

        // Create AST: fn main() { { let x = 1; x + 2 } }
        // Inner block: let x = 1; x + 2
        let inner_let = Statement::Let(LetStatement {
            pattern: Pattern::Identifier("x".to_string()),
            value: (Expr::Nat(1), Span::new(16, 17)),
        });
        let inner_expr = Statement::Expression((
            Expr::Binary(BinaryExpression {
                left: Box::new((Expr::Identifier("x".to_string()), Span::new(19, 20))),
                operator: BinaryOperator::Add,
                qualifier: None,
                right: Box::new((Expr::Nat(2), Span::new(23, 24))),
            }),
            Span::new(19, 24),
        ));

        let block_expr = Expr::Block(vec![inner_let, inner_expr]);

        let body = tribute_ast::Block {
            statements: vec![Statement::Expression((block_expr, Span::new(12, 26)))],
        };

        let func_def =
            FunctionDefinition::new(db, "main".to_string(), vec![], None, body, Span::new(0, 28));

        let item = tribute_ast::Item::new(db, ItemKind::Function(func_def), Span::new(0, 28));
        let program = Program::new(db, vec![item]);

        lower_program(db, path, program)
    }

    #[test]
    fn test_lower_block_expression() {
        TributeDatabaseImpl::default().attach(|db| {
            let module = lower_block_expr_helper(db);

            // Get the function
            let body_region = module.body(db);
            let blocks = body_region.blocks(db);
            let func_op = func::Func::from_operation(db, blocks[0].operations(db)[0]).unwrap();

            // Get the function's entry block
            let func_body = func_op.body(db);
            let func_blocks = func_body.blocks(db);
            let entry_block = &func_blocks[0];
            let ops = entry_block.operations(db);

            // Should have: src.block, func.return
            assert_eq!(ops.len(), 2);

            // Verify we have src.block
            let block_op = src::Block::from_operation(db, ops[0]).unwrap();

            // Get the block's body region
            let block_body = block_op.body(db);
            let block_blocks = block_body.blocks(db);
            assert_eq!(block_blocks.len(), 1);

            let inner_block = &block_blocks[0];
            let inner_ops = inner_block.operations(db);

            // Inner block should have: arith.const(1), arith.const(2), arith.add, src.yield
            assert_eq!(inner_ops.len(), 4);

            // Verify the operations inside the block
            let const_1 = arith::Const::from_operation(db, inner_ops[0]).unwrap();
            let const_2 = arith::Const::from_operation(db, inner_ops[1]).unwrap();
            let add_op = arith::Add::from_operation(db, inner_ops[2]).unwrap();

            // lhs should be x (const_1), rhs should be 2 (const_2)
            assert_eq!(add_op.lhs(db), const_1.result(db));
            assert_eq!(add_op.rhs(db), const_2.result(db));

            // Yield should use the add result
            let yield_op = src::Yield::from_operation(db, inner_ops[3]).unwrap();
            assert_eq!(yield_op.value(db), add_op.result(db));

            // Return should use the block's result
            let ret_op = func::Return::from_operation(db, ops[1]).unwrap();
            assert_eq!(ret_op.operands(db)[0], block_op.result(db));
        });
    }

    /// Helper to create AST with lambda: fn main() { fn(x) x + 1 }
    #[salsa::tracked]
    fn lower_lambda_helper(db: &dyn salsa::Database) -> core::Module<'_> {
        let path = PathId::new(db, PathBuf::from("test.tr"));

        // Create AST: fn main() { fn(x) x + 1 }
        let lambda = Expr::Lambda(LambdaExpression {
            parameters: vec![Parameter {
                name: "x".to_string(),
                ty: None,
            }],
            return_type: None,
            body: Box::new((
                Expr::Binary(BinaryExpression {
                    left: Box::new((Expr::Identifier("x".to_string()), Span::new(18, 19))),
                    operator: BinaryOperator::Add,
                    qualifier: None,
                    right: Box::new((Expr::Nat(1), Span::new(22, 23))),
                }),
                Span::new(18, 23),
            )),
        });

        let body = tribute_ast::Block {
            statements: vec![Statement::Expression((lambda, Span::new(12, 24)))],
        };

        let func_def =
            FunctionDefinition::new(db, "main".to_string(), vec![], None, body, Span::new(0, 26));

        let item = tribute_ast::Item::new(db, ItemKind::Function(func_def), Span::new(0, 26));
        let program = Program::new(db, vec![item]);

        lower_program(db, path, program)
    }

    #[test]
    fn test_lower_lambda_expression() {
        TributeDatabaseImpl::default().attach(|db| {
            let module = lower_lambda_helper(db);

            // Get the function
            let body_region = module.body(db);
            let blocks = body_region.blocks(db);
            let func_op = func::Func::from_operation(db, blocks[0].operations(db)[0]).unwrap();

            // Get the function's entry block
            let func_body = func_op.body(db);
            let func_blocks = func_body.blocks(db);
            let entry_block = &func_blocks[0];
            let ops = entry_block.operations(db);

            // Should have: src.lambda, func.return
            assert_eq!(ops.len(), 2);

            // Verify we have src.lambda
            let lambda_op = src::Lambda::from_operation(db, ops[0]).unwrap();

            // Verify lambda has the type attribute
            let ty = lambda_op.r#type(db);
            assert!(matches!(ty, Attribute::Type(Type::Function { .. })));

            // Get the lambda's body region
            let lambda_body = lambda_op.body(db);
            let lambda_blocks = lambda_body.blocks(db);
            assert_eq!(lambda_blocks.len(), 1);

            let inner_block = &lambda_blocks[0];

            // Lambda should have one block argument (the parameter x)
            assert_eq!(inner_block.args(db).len(), 1);

            let inner_ops = inner_block.operations(db);

            // Inner block should have: src.var(x), arith.const(1), arith.add, src.yield
            assert_eq!(inner_ops.len(), 4);

            // Verify the parameter is represented as src.var
            let var_x = src::Var::from_operation(db, inner_ops[0]).unwrap();
            assert_eq!(var_x.name(db), &Attribute::String("x".to_string()));

            // Verify const and add
            let const_1 = arith::Const::from_operation(db, inner_ops[1]).unwrap();
            let add_op = arith::Add::from_operation(db, inner_ops[2]).unwrap();
            assert_eq!(add_op.lhs(db), var_x.result(db));
            assert_eq!(add_op.rhs(db), const_1.result(db));

            // Verify yield uses the add result
            let yield_op = src::Yield::from_operation(db, inner_ops[3]).unwrap();
            assert_eq!(yield_op.value(db), add_op.result(db));

            // Return should use the lambda's result
            let ret_op = func::Return::from_operation(db, ops[1]).unwrap();
            assert_eq!(ret_op.operands(db)[0], lambda_op.result(db));
        });
    }

    /// Helper to create AST with complex lambda: fn main() { fn(x, y) { let z = x * y; z + x } }
    #[salsa::tracked]
    fn lower_complex_lambda_helper(db: &dyn salsa::Database) -> core::Module<'_> {
        let path = PathId::new(db, PathBuf::from("test.tr"));

        // Create AST: fn main() { fn(x, y) { let z = x * y; z + x } }
        // Inner block: let z = x * y; z + x
        let let_stmt = Statement::Let(LetStatement {
            pattern: Pattern::Identifier("z".to_string()),
            value: (
                Expr::Binary(BinaryExpression {
                    left: Box::new((Expr::Identifier("x".to_string()), Span::new(22, 23))),
                    operator: BinaryOperator::Multiply,
                    qualifier: None,
                    right: Box::new((Expr::Identifier("y".to_string()), Span::new(26, 27))),
                }),
                Span::new(22, 27),
            ),
        });
        let result_expr = Statement::Expression((
            Expr::Binary(BinaryExpression {
                left: Box::new((Expr::Identifier("z".to_string()), Span::new(29, 30))),
                operator: BinaryOperator::Add,
                qualifier: None,
                right: Box::new((Expr::Identifier("x".to_string()), Span::new(33, 34))),
            }),
            Span::new(29, 34),
        ));

        let lambda = Expr::Lambda(LambdaExpression {
            parameters: vec![
                Parameter {
                    name: "x".to_string(),
                    ty: None,
                },
                Parameter {
                    name: "y".to_string(),
                    ty: None,
                },
            ],
            return_type: None,
            body: Box::new((Expr::Block(vec![let_stmt, result_expr]), Span::new(18, 36))),
        });

        let body = tribute_ast::Block {
            statements: vec![Statement::Expression((lambda, Span::new(12, 38)))],
        };

        let func_def =
            FunctionDefinition::new(db, "main".to_string(), vec![], None, body, Span::new(0, 40));

        let item = tribute_ast::Item::new(db, ItemKind::Function(func_def), Span::new(0, 40));
        let program = Program::new(db, vec![item]);

        lower_program(db, path, program)
    }

    #[test]
    fn test_lower_complex_lambda() {
        TributeDatabaseImpl::default().attach(|db| {
            let module = lower_complex_lambda_helper(db);

            // Get the function
            let body_region = module.body(db);
            let blocks = body_region.blocks(db);
            let func_op = func::Func::from_operation(db, blocks[0].operations(db)[0]).unwrap();

            // Get the function's entry block
            let func_body = func_op.body(db);
            let func_blocks = func_body.blocks(db);
            let entry_block = &func_blocks[0];
            let ops = entry_block.operations(db);

            // Should have: src.lambda, func.return
            assert_eq!(ops.len(), 2);

            // Verify we have src.lambda
            let lambda_op = src::Lambda::from_operation(db, ops[0]).unwrap();

            // Verify lambda has two parameters
            let ty = lambda_op.r#type(db);
            match ty {
                Attribute::Type(Type::Function { params, .. }) => {
                    assert_eq!(params.len(), 2);
                }
                _ => panic!("expected function type"),
            }

            // Get the lambda's body region
            let lambda_body = lambda_op.body(db);
            let lambda_blocks = lambda_body.blocks(db);
            assert_eq!(lambda_blocks.len(), 1);

            let lambda_entry = &lambda_blocks[0];

            // Lambda should have two block arguments (x, y)
            assert_eq!(lambda_entry.args(db).len(), 2);

            let lambda_ops = lambda_entry.operations(db);

            // Lambda body should have: src.var(x), src.var(y), src.block, src.yield
            assert_eq!(lambda_ops.len(), 4);

            // Verify parameters are src.var
            let var_x = src::Var::from_operation(db, lambda_ops[0]).unwrap();
            assert_eq!(var_x.name(db), &Attribute::String("x".to_string()));
            let var_y = src::Var::from_operation(db, lambda_ops[1]).unwrap();
            assert_eq!(var_y.name(db), &Attribute::String("y".to_string()));

            // Verify src.block for the body
            let block_op = src::Block::from_operation(db, lambda_ops[2]).unwrap();

            // Get the block's inner ops
            let block_body = block_op.body(db);
            let block_blocks = block_body.blocks(db);
            let inner_block = &block_blocks[0];
            let inner_ops = inner_block.operations(db);

            // Inner block: src.var(x), src.var(y), arith.mul (for z=x*y),
            //              src.var(z), src.var(x), arith.add, src.yield
            // Note: x and y are looked up from context (bound to lambda's src.var ops)
            // z is bound via let, so z+x uses the mul result directly

            // Actually, let's trace through:
            // - let z = x * y: looks up x (finds var_x), y (finds var_y), emits mul, binds z to mul result
            // - z + x: looks up z (finds mul result), x (finds var_x), emits add
            // So inner block should have: arith.mul, arith.add, src.yield
            assert_eq!(inner_ops.len(), 3);

            let mul_op = arith::Mul::from_operation(db, inner_ops[0]).unwrap();
            let add_op = arith::Add::from_operation(db, inner_ops[1]).unwrap();

            // mul uses var_x and var_y
            assert_eq!(mul_op.lhs(db), var_x.result(db));
            assert_eq!(mul_op.rhs(db), var_y.result(db));

            // add uses mul result (z) and var_x
            assert_eq!(add_op.lhs(db), mul_op.result(db));
            assert_eq!(add_op.rhs(db), var_x.result(db));

            // yield uses add result
            let yield_op = src::Yield::from_operation(db, inner_ops[2]).unwrap();
            assert_eq!(yield_op.value(db), add_op.result(db));
        });
    }
}
