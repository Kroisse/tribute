//! AST to TrunkIR lowering pass.
//!
//! This pass converts the parsed AST into TrunkIR operations.
//! At this stage, names are unresolved (using `src` dialect ops).

use std::collections::HashMap;

use crate::{Attribute, BlockBuilder, Type, Value, arith, core, func, src};
use tribute_ast::{
    BinaryExpression, BinaryOperator, CallExpression, Expr, FunctionDefinition, ItemKind,
    LetStatement, Pattern, Program, Statement,
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
        Expr::Lambda(_) => todo!("lambda expressions"),
        Expr::Block(_) => todo!("block expressions"),
        Expr::List(_) => todo!("list literals"),
        Expr::Tuple(_, _) => todo!("tuple literals"),
        Expr::Record(_) => todo!("record expressions"),
        Expr::OperatorFn(_) => todo!("operator functions"),
    }
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
    use tribute_ast::BinaryExpression;
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
}
