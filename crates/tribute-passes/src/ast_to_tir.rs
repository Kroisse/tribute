//! AST to TrunkIR lowering pass.
//!
//! This pass converts the parsed AST into TrunkIR operations.
//! At this stage, names are unresolved (using `src` dialect ops).

use std::collections::HashMap;

use tribute_ast::{
    BinaryExpression, BinaryOperator, BytesInterpolation, CallExpression, ConstructorArgs, Expr,
    FunctionDefinition, ItemKind, LambdaExpression, LetStatement, ListPattern, MatchArm,
    MatchExpression, MethodCallExpression, OperatorFnExpression, Pattern, Program,
    RecordExpression, RecordField, Statement, StringInterpolation, TypeRef,
};
use tribute_core::{Location, PathId, Span, Spanned};
use tribute_trunk_ir::{
    Attribute, BlockBuilder, DialectType, IdVec, Region, Symbol, Type, Value,
    dialect::{adt, arith, case, core, func, list, src, ty},
    idvec,
};

/// Create a symbol from a string.
fn sym<'db>(db: &'db dyn salsa::Database, name: &str) -> Symbol<'db> {
    Symbol::new(db, name)
}

/// Create a symbol reference (path) from a single name.
fn sym_ref<'db>(db: &'db dyn salsa::Database, name: &str) -> IdVec<Symbol<'db>> {
    idvec![Symbol::new(db, name)]
}

/// Context for lowering, tracking local variable bindings and type variable generation.
struct LoweringCtx<'db> {
    /// Map from variable names to their SSA values.
    bindings: HashMap<String, Value<'db>>,
    /// Map from type variable names to their Type representations.
    type_var_bindings: HashMap<String, Type<'db>>,
    /// Counter for generating unique type variable IDs.
    next_type_var_id: u64,
}

impl<'db> LoweringCtx<'db> {
    fn new() -> Self {
        Self {
            bindings: HashMap::new(),
            type_var_bindings: HashMap::new(),
            next_type_var_id: 0,
        }
    }

    /// Generate a fresh type variable with a unique ID.
    fn fresh_type_var(&mut self, db: &'db dyn salsa::Database) -> Type<'db> {
        let id = self.next_type_var_id;
        self.next_type_var_id += 1;
        ty::var_with_id(db, id)
    }

    /// Get or create a named type variable.
    /// Same name always returns the same type variable within a scope.
    fn named_type_var(&mut self, db: &'db dyn salsa::Database, name: &str) -> Type<'db> {
        if let Some(&ty) = self.type_var_bindings.get(name) {
            ty
        } else {
            let ty = self.fresh_type_var(db);
            self.type_var_bindings.insert(name.to_string(), ty);
            ty
        }
    }

    /// Resolve a TypeRef to an IR Type.
    fn resolve_type_ref(&mut self, db: &'db dyn salsa::Database, type_ref: &TypeRef) -> Type<'db> {
        match type_ref {
            TypeRef::Named(name) => {
                // Concrete named types - emit src.type for later resolution
                src::unresolved_type(db, name, idvec![])
            }
            TypeRef::Variable(name) => {
                // Type variable - get or create with consistent ID
                self.named_type_var(db, name)
            }
            TypeRef::Generic { name, args } => {
                // Generic type - resolve args and emit src.type with params
                let params: IdVec<Type<'db>> = args
                    .iter()
                    .map(|arg| self.resolve_type_ref(db, arg))
                    .collect();
                src::unresolved_type(db, name, params)
            }
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
        let saved_bindings = self.bindings.clone();
        let saved_type_vars = self.type_var_bindings.clone();
        let result = f(self);
        self.bindings = saved_bindings;
        self.type_var_bindings = saved_type_vars;
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
    let mut ctx = LoweringCtx::new();

    // Resolve parameter types from annotations, or create fresh type vars
    let params: IdVec<Type> = func_def
        .parameters(db)
        .iter()
        .map(|param| {
            param
                .ty
                .as_ref()
                .map(|ty| ctx.resolve_type_ref(db, ty))
                .unwrap_or_else(|| ctx.fresh_type_var(db))
        })
        .collect();

    // Resolve return type from annotation, or create fresh type var
    let result = func_def
        .return_type(db)
        .as_ref()
        .map(|ty| ctx.resolve_type_ref(db, ty))
        .unwrap_or_else(|| ctx.fresh_type_var(db));

    func::Func::build(db, location, &name, params, result, |entry| {
        let body = func_def.body(db);

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
    let location = Location::new(path, let_stmt.value.1);
    let value = lower_expr(db, path, ctx, block, &let_stmt.value);
    bind_pattern(db, ctx, block, &let_stmt.pattern, value, location);
}

/// Bind a pattern to a value, adding bindings to the context.
/// For patterns that require destructuring, emits extraction operations.
fn bind_pattern<'db>(
    db: &'db dyn salsa::Database,
    ctx: &mut LoweringCtx<'db>,
    block: &mut BlockBuilder<'db>,
    pattern: &Pattern,
    value: Value<'db>,
    location: Location<'db>,
) {
    let infer_ty = ctx.fresh_type_var(db);
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
            bind_pattern(db, ctx, block, inner, value, location);
        }
        Pattern::Literal(_) => {
            // Literal patterns in let bindings are unusual but valid
            // They don't introduce bindings, just assert the value matches
            // No bindings needed here
        }
        Pattern::Constructor(ctor) => {
            // For let bindings, constructor patterns are for destructuring
            // e.g., let Some(x) = opt
            match &ctor.args {
                ConstructorArgs::None => {}
                ConstructorArgs::Positional(patterns) => {
                    for (i, pat) in patterns.iter().enumerate() {
                        let field_value = block
                            .op(adt::variant_get(
                                db,
                                location,
                                value,
                                infer_ty,
                                u64::try_from(i).unwrap().into(),
                            ))
                            .result(db);
                        bind_pattern(db, ctx, block, pat, field_value, location);
                    }
                }
                ConstructorArgs::Named { fields, .. } => {
                    for (i, field) in fields.iter().enumerate() {
                        let field_value = block
                            .op(adt::variant_get(
                                db,
                                location,
                                value,
                                infer_ty,
                                u64::try_from(i).unwrap().into(),
                            ))
                            .result(db);
                        bind_pattern(db, ctx, block, &field.pattern, field_value, location);
                    }
                }
            }
        }
        Pattern::Tuple(first, rest) => {
            // Destructure tuple: let #(a, b, c) = tuple
            let first_value = block
                .op(src::call(
                    db,
                    location,
                    vec![value],
                    infer_ty,
                    sym_ref(db, "tuple_get_0"),
                ))
                .result(db);
            bind_pattern(db, ctx, block, first, first_value, location);

            for (i, pat) in rest.iter().enumerate() {
                let elem_value = block
                    .op(src::call(
                        db,
                        location,
                        vec![value],
                        infer_ty,
                        sym_ref(db, &format!("tuple_get_{}", i + 1)),
                    ))
                    .result(db);
                bind_pattern(db, ctx, block, pat, elem_value, location);
            }
        }
        Pattern::List(ListPattern { elements, rest }) => {
            // Destructure list: let [a, b, ..rest] = list
            for (i, pat) in elements.iter().enumerate() {
                let index_value = block
                    .op(arith::Const::i64(db, location, i64::try_from(i).unwrap()))
                    .result(db);
                let elem_ty = ctx.fresh_type_var(db);
                let elem_value = block
                    .op(list::get(
                        db,
                        location,
                        value,
                        index_value,
                        infer_ty,
                        elem_ty,
                    ))
                    .result(db);
                bind_pattern(db, ctx, block, pat, elem_value, location);
            }

            // Handle rest pattern: ..rest binds to list[n..]
            if let Some(Some(rest_name)) = rest {
                let start_value = block
                    .op(arith::Const::i64(
                        db,
                        location,
                        i64::try_from(elements.len()).unwrap(),
                    ))
                    .result(db);
                let len_value = block
                    .op(list::len(db, location, value, infer_ty))
                    .result(db);
                let elem_ty = ctx.fresh_type_var(db);
                let rest_value = block
                    .op(list::slice(
                        db,
                        location,
                        value,
                        start_value,
                        len_value,
                        infer_ty,
                        elem_ty,
                    ))
                    .result(db);
                ctx.bind(rest_name.clone(), rest_value);
            }
        }
        Pattern::Handler(_) => {
            // Handler patterns don't make sense in let bindings
            // They are for ability effect handling in match expressions
        }
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
    let unit_ty = core::Nil::new(db).as_type();
    let infer_ty = ctx.fresh_type_var(db); // Type to be inferred

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
            let op = block.op(arith::r#const(
                db,
                location,
                core::I1::new(db).as_type(),
                (*b).into(),
            ));
            op.result(db)
        }
        Expr::Nil => {
            // Nil as unit constant
            let op = block.op(arith::r#const(db, location, unit_ty, Attribute::Unit));
            op.result(db)
        }

        // Variable reference: check local bindings first, then emit src.var for unresolved
        Expr::Identifier(name) => {
            if let Some(value) = ctx.lookup(name) {
                // Found in local bindings - use the SSA value directly
                value
            } else {
                // Unresolved - emit src.var for later resolution
                let op = block.op(src::var(db, location, infer_ty, sym(db, name)));
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

            let op = block.op(src::call(
                db,
                location,
                args,
                infer_ty,
                sym_ref(db, function),
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
            lower_binary_op(db, ctx, location, block, operator.clone(), lhs, rhs)
        }

        // Rune (Unicode codepoint) → arith.const i32
        Expr::Rune(c) => {
            let op = block.op(arith::r#const(
                db,
                location,
                core::I32::new(db).as_type(),
                u64::from(u32::from(*c)).into(),
            ));
            op.result(db)
        }

        // String interpolation → concatenate string parts and expressions
        Expr::StringInterpolation(interp) => {
            lower_string_interpolation(db, path, ctx, block, interp, location)
        }

        // Bytes interpolation → concatenate bytes parts and expressions
        Expr::BytesInterpolation(interp) => {
            lower_bytes_interpolation(db, path, ctx, block, interp, location)
        }
        // Method call (UFCS): x.f(y, z) → f(x, y, z)
        Expr::MethodCall(MethodCallExpression {
            receiver,
            method,
            arguments,
        }) => {
            // Lower receiver first, then arguments
            let receiver_value = lower_expr(db, path, ctx, block, receiver);
            let mut args = vec![receiver_value];
            for arg in arguments {
                args.push(lower_expr(db, path, ctx, block, arg));
            }

            let op = block.op(src::call(db, location, args, infer_ty, sym_ref(db, method)));
            op.result(db)
        }

        // Match expression → scf.case
        Expr::Match(MatchExpression { value, arms }) => {
            lower_match_expr(db, path, ctx, block, value, arms, location)
        }
        Expr::Lambda(LambdaExpression {
            parameters,
            return_type: _, // TODO: use type annotation when available
            body,
        }) => {
            // Parameter types are unknown until type inference
            let param_types: IdVec<Type<'_>> =
                std::iter::repeat_n(infer_ty, parameters.len()).collect();
            let result_type = infer_ty;
            let mut body_block = BlockBuilder::new(db, location).args(param_types.clone());

            // Lower body in a new scope with parameters bound to block arguments
            let result_value = ctx.scoped(|ctx| {
                // Bind parameters to block arguments
                // Note: we need the built block to get arg values, so we track indices
                for (i, param) in parameters.iter().enumerate() {
                    // Create a placeholder - will be resolved when block is built
                    // For now, emit src.var for parameter references
                    // The block args will be connected during a later pass
                    let param_value =
                        body_block.op(src::var(db, location, infer_ty, sym(db, &param.name)));
                    ctx.bind(param.name.clone(), param_value.result(db));
                    let _ = i; // suppress unused warning for now
                }

                // Lower the body expression
                lower_expr(db, path, ctx, &mut body_block, body)
            });

            // Add yield to return the lambda's result
            body_block.op(src::r#yield(db, location, result_value));

            // Create the function type for the lambda
            let func_type = core::Func::new(db, param_types, result_type).as_type();

            // Create the src.lambda operation
            let region = Region::new(db, location, idvec![body_block.build()]);
            let lambda_op = block.op(src::lambda(db, location, infer_ty, func_type, region));
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
            body_block.op(src::r#yield(db, location, result_value));

            // Create the src.block operation
            let region = Region::new(db, location, idvec![body_block.build()]);
            let block_op = block.op(src::block(db, location, infer_ty, region));
            block_op.result(db)
        }
        // List literal → list.new
        Expr::List(elements) => {
            // Lower all elements
            let values: Vec<Value<'db>> = elements
                .iter()
                .map(|elem| lower_expr(db, path, ctx, block, elem))
                .collect();

            // elem_type is inferred - use fresh type variable
            let elem_ty = ctx.fresh_type_var(db);
            let op = block.op(list::new(db, location, values, infer_ty, elem_ty));
            op.result(db)
        }
        Expr::Tuple(first, rest) => {
            // Lower all tuple elements
            let first_value = lower_expr(db, path, ctx, block, first);
            let mut elements = vec![first_value];
            for elem in rest {
                elements.push(lower_expr(db, path, ctx, block, elem));
            }

            // Create the src.tuple operation
            let tuple_op = block.op(src::tuple(db, location, elements, infer_ty));
            tuple_op.result(db)
        }
        // Record expression: User { name: "Alice", age: 30 }
        Expr::Record(RecordExpression { type_name, fields }) => {
            lower_record_expr(db, path, ctx, block, type_name, fields, location)
        }

        // Operator as function: (+), (Int::+)
        Expr::OperatorFn(OperatorFnExpression { op, qualifier }) => {
            match qualifier {
                Some(q) => {
                    // Qualified path: Int::+ → src.path(["Int", "+"])
                    let path = idvec![sym(db, q), sym(db, op)];
                    let op = block.op(src::path(db, location, infer_ty, path));
                    op.result(db)
                }
                None => {
                    // Unqualified: (+) → src.var("+")
                    let op = block.op(src::var(db, location, infer_ty, sym(db, op)));
                    op.result(db)
                }
            }
        }
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
        let op = block.op(arith::r#const(
            db,
            location,
            core::Nil::new(db).as_type(),
            Attribute::Unit,
        ));
        op.result(db)
    })
}

/// Lower a binary operation to the appropriate TrunkIR op.
fn lower_binary_op<'db>(
    db: &'db dyn salsa::Database,
    ctx: &mut LoweringCtx<'db>,
    location: Location<'db>,
    block: &mut BlockBuilder<'db>,
    operator: BinaryOperator,
    lhs: Value<'db>,
    rhs: Value<'db>,
) -> Value<'db> {
    // Result type is unknown until type inference
    let infer_ty = ctx.fresh_type_var(db);
    let bool_ty = core::I1::new(db).as_type();

    match operator {
        // Arithmetic operations → arith dialect
        BinaryOperator::Add => block
            .op(arith::add(db, location, lhs, rhs, infer_ty))
            .result(db),
        BinaryOperator::Subtract => block
            .op(arith::sub(db, location, lhs, rhs, infer_ty))
            .result(db),
        BinaryOperator::Multiply => block
            .op(arith::mul(db, location, lhs, rhs, infer_ty))
            .result(db),
        BinaryOperator::Divide => block
            .op(arith::div(db, location, lhs, rhs, infer_ty))
            .result(db),
        BinaryOperator::Modulo => block
            .op(arith::rem(db, location, lhs, rhs, infer_ty))
            .result(db),

        // Comparison operations → arith dialect (result is i1)
        BinaryOperator::Equal => block
            .op(arith::cmp_eq(db, location, lhs, rhs, bool_ty))
            .result(db),
        BinaryOperator::NotEqual => block
            .op(arith::cmp_ne(db, location, lhs, rhs, bool_ty))
            .result(db),
        BinaryOperator::LessThan => block
            .op(arith::cmp_lt(db, location, lhs, rhs, bool_ty))
            .result(db),
        BinaryOperator::LessEqual => block
            .op(arith::cmp_le(db, location, lhs, rhs, bool_ty))
            .result(db),
        BinaryOperator::GreaterThan => block
            .op(arith::cmp_gt(db, location, lhs, rhs, bool_ty))
            .result(db),
        BinaryOperator::GreaterEqual => block
            .op(arith::cmp_ge(db, location, lhs, rhs, bool_ty))
            .result(db),

        // Logical operations → arith bitwise (works for i1 booleans)
        BinaryOperator::And => block
            .op(arith::and(db, location, lhs, rhs, bool_ty))
            .result(db),
        BinaryOperator::Or => block
            .op(arith::or(db, location, lhs, rhs, bool_ty))
            .result(db),

        // Concat needs type-directed resolution → src.binop
        BinaryOperator::Concat => {
            let op = block.op(src::binop(
                db,
                location,
                lhs,
                rhs,
                infer_ty,
                sym(db, "concat"),
            ));
            op.result(db)
        }
    }
}

/// Lower a string interpolation expression.
/// `"hello \{name}!"` → string_const("hello ") <> to_string(name) <> string_const("!")
fn lower_string_interpolation<'db>(
    db: &'db dyn salsa::Database,
    path: PathId<'db>,
    ctx: &mut LoweringCtx<'db>,
    block: &mut BlockBuilder<'db>,
    interp: &StringInterpolation,
    location: Location<'db>,
) -> Value<'db> {
    let string_ty = core::String::new(db).as_type();

    // Start with the leading string part
    let mut result = block
        .op(adt::string_const(
            db,
            location,
            string_ty,
            interp.leading.clone(),
        ))
        .result(db);

    // For each segment: concat(result, to_string(expr), trailing)
    for segment in &interp.segments {
        // Lower the interpolated expression
        let expr_value = lower_expr(db, path, ctx, block, &segment.interpolation);

        // Call to_string on the expression (will be resolved later)
        let str_value = block
            .op(src::call(
                db,
                location,
                vec![expr_value],
                string_ty,
                sym_ref(db, "to_string"),
            ))
            .result(db);

        // Concat result with str_value
        result = block
            .op(src::binop(
                db,
                location,
                result,
                str_value,
                string_ty,
                sym(db, "concat"),
            ))
            .result(db);

        // Concat with trailing string if non-empty
        if !segment.trailing.is_empty() {
            let trailing_value = block
                .op(adt::string_const(
                    db,
                    location,
                    string_ty,
                    segment.trailing.clone(),
                ))
                .result(db);

            result = block
                .op(src::binop(
                    db,
                    location,
                    result,
                    trailing_value,
                    string_ty,
                    sym(db, "concat"),
                ))
                .result(db);
        }
    }

    result
}

/// Lower a bytes interpolation expression.
fn lower_bytes_interpolation<'db>(
    db: &'db dyn salsa::Database,
    path: PathId<'db>,
    ctx: &mut LoweringCtx<'db>,
    block: &mut BlockBuilder<'db>,
    interp: &BytesInterpolation,
    location: Location<'db>,
) -> Value<'db> {
    let bytes_ty = core::Bytes::new(db).as_type();

    // Start with the leading bytes part
    let mut result = block
        .op(adt::bytes_const(
            db,
            location,
            bytes_ty,
            Attribute::Bytes(interp.leading.clone()),
        ))
        .result(db);

    // For each segment: concat(result, to_bytes(expr), trailing)
    for segment in &interp.segments {
        // Lower the interpolated expression
        let expr_value = lower_expr(db, path, ctx, block, &segment.interpolation);

        // Call to_bytes on the expression (will be resolved later)
        let bytes_value = block
            .op(src::call(
                db,
                location,
                vec![expr_value],
                bytes_ty,
                sym_ref(db, "to_bytes"),
            ))
            .result(db);

        // Concat result with bytes_value
        result = block
            .op(src::binop(
                db,
                location,
                result,
                bytes_value,
                bytes_ty,
                sym(db, "concat"),
            ))
            .result(db);

        // Concat with trailing bytes if non-empty
        if !segment.trailing.is_empty() {
            let trailing_value = block
                .op(adt::bytes_const(
                    db,
                    location,
                    bytes_ty,
                    Attribute::Bytes(segment.trailing.clone()),
                ))
                .result(db);

            result = block
                .op(src::binop(
                    db,
                    location,
                    result,
                    trailing_value,
                    bytes_ty,
                    sym(db, "concat"),
                ))
                .result(db);
        }
    }

    result
}

/// Lower a match expression to scf.case.
fn lower_match_expr<'db>(
    db: &'db dyn salsa::Database,
    path: PathId<'db>,
    ctx: &mut LoweringCtx<'db>,
    block: &mut BlockBuilder<'db>,
    value: &Spanned<Expr>,
    arms: &[MatchArm],
    location: Location<'db>,
) -> Value<'db> {
    // Lower the scrutinee
    let scrutinee = lower_expr(db, path, ctx, block, value);

    // Build the body region containing case.arm operations
    let mut body_block = BlockBuilder::new(db, location);

    for arm in arms {
        // Each arm becomes a case.arm operation with its own body region
        // For now, we handle only the first guarded branch (no guards support yet)
        let branch = arm
            .branches
            .first()
            .expect("match arm should have at least one branch");

        let mut arm_block = BlockBuilder::new(db, location);

        // Lower the branch body in a new scope with pattern bindings
        let result_value = ctx.scoped(|ctx| {
            // Bind pattern variables (simplified - actual pattern matching happens later)
            bind_pattern_for_match(db, ctx, &mut arm_block, &arm.pattern, scrutinee, location);

            // Handle guard if present
            if branch.guard.is_some() {
                // Guards require more complex control flow - for now emit the body
                // A proper implementation would emit conditional branching
            }

            lower_expr(db, path, ctx, &mut arm_block, &branch.value)
        });

        // Yield the result from the arm
        arm_block.op(case::r#yield(db, location, result_value));
        let arm_region = Region::new(db, location, idvec![arm_block.build()]);

        // Create the case.arm operation with pattern attribute
        let pattern_attr = pattern_to_attribute(&arm.pattern);
        body_block.op(case::arm(db, location, pattern_attr, arm_region));
    }

    let body_region = Region::new(db, location, idvec![body_block.build()]);

    // Create the case.case operation
    let case_op = block.op(case::r#case(
        db,
        location,
        scrutinee,
        core::Nil::new(db).as_type(),
        body_region,
    ));
    case_op.result(db)
}

/// Bind pattern variables for match expressions.
/// This is a simplified version that just binds identifiers.
fn bind_pattern_for_match<'db>(
    db: &'db dyn salsa::Database,
    ctx: &mut LoweringCtx<'db>,
    block: &mut BlockBuilder<'db>,
    pattern: &Pattern,
    scrutinee: Value<'db>,
    location: Location<'db>,
) {
    let infer_ty = ctx.fresh_type_var(db);
    match pattern {
        Pattern::Identifier(name) => {
            ctx.bind(name.clone(), scrutinee);
        }
        Pattern::Wildcard => {
            // No binding needed
        }
        Pattern::As(inner, name) => {
            ctx.bind(name.clone(), scrutinee);
            bind_pattern_for_match(db, ctx, block, inner, scrutinee, location);
        }
        Pattern::Literal(_lit) => {
            // Literal patterns don't introduce bindings, just matching
            // The actual comparison happens in pattern compilation
        }
        Pattern::Constructor(ctor) => {
            // For constructors, we need to extract fields
            // For now, bind any nested identifier patterns
            match &ctor.args {
                ConstructorArgs::None => {}
                ConstructorArgs::Positional(patterns) => {
                    for (i, pat) in patterns.iter().enumerate() {
                        // Extract field i from the variant
                        let field_value = block
                            .op(adt::variant_get(
                                db,
                                location,
                                scrutinee,
                                infer_ty,
                                u64::try_from(i).expect("unexpected index").into(),
                            ))
                            .result(db);
                        bind_pattern_for_match(db, ctx, block, pat, field_value, location);
                    }
                }
                ConstructorArgs::Named { fields, .. } => {
                    for (i, field) in fields.iter().enumerate() {
                        let field_value = block
                            .op(adt::variant_get(
                                db,
                                location,
                                scrutinee,
                                infer_ty,
                                u64::try_from(i).expect("unexpected index").into(),
                            ))
                            .result(db);
                        bind_pattern_for_match(
                            db,
                            ctx,
                            block,
                            &field.pattern,
                            field_value,
                            location,
                        );
                    }
                }
            }
        }
        Pattern::Tuple(first, rest) => {
            // Extract tuple elements
            let first_value = block
                .op(src::call(
                    db,
                    location,
                    vec![scrutinee],
                    infer_ty,
                    sym_ref(db, "tuple_get_0"),
                ))
                .result(db);
            bind_pattern_for_match(db, ctx, block, first, first_value, location);

            for (i, pat) in rest.iter().enumerate() {
                let elem_value = block
                    .op(src::call(
                        db,
                        location,
                        vec![scrutinee],
                        infer_ty,
                        sym_ref(db, &format!("tuple_get_{}", i + 1)),
                    ))
                    .result(db);
                bind_pattern_for_match(db, ctx, block, pat, elem_value, location);
            }
        }
        Pattern::List(ListPattern { elements, rest }) => {
            // Extract list elements
            for (i, pat) in elements.iter().enumerate() {
                let index_value = block
                    .op(arith::Const::i64(db, location, i64::try_from(i).unwrap()))
                    .result(db);
                let elem_ty = ctx.fresh_type_var(db);
                let elem_value = block
                    .op(list::get(
                        db,
                        location,
                        scrutinee,
                        index_value,
                        infer_ty,
                        elem_ty,
                    ))
                    .result(db);
                bind_pattern_for_match(db, ctx, block, pat, elem_value, location);
            }

            // Handle rest pattern (..tail or ..)
            if let Some(Some(rest_name)) = rest {
                // Bind the rest of the list to the name
                let start_value = block
                    .op(arith::Const::i64(
                        db,
                        location,
                        i64::try_from(elements.len()).unwrap(),
                    ))
                    .result(db);
                let len_value = block
                    .op(list::len(db, location, scrutinee, infer_ty))
                    .result(db);
                let elem_ty = ctx.fresh_type_var(db);
                let rest_value = block
                    .op(list::slice(
                        db,
                        location,
                        scrutinee,
                        start_value,
                        len_value,
                        infer_ty,
                        elem_ty,
                    ))
                    .result(db);
                ctx.bind(rest_name.clone(), rest_value);
            }
        }
        Pattern::Handler(_) => {
            // Handler patterns are for ability handling, not regular match
            // This should not appear in regular match expressions
        }
    }
}

/// Lower a record expression.
fn lower_record_expr<'db>(
    db: &'db dyn salsa::Database,
    path: PathId<'db>,
    ctx: &mut LoweringCtx<'db>,
    block: &mut BlockBuilder<'db>,
    type_name: &str,
    fields: &[RecordField],
    location: Location<'db>,
) -> Value<'db> {
    // Collect field values
    let mut field_values = Vec::new();
    let mut field_names = Vec::new();

    for field in fields {
        match field {
            RecordField::Field { name, value } => {
                field_names.push(name.clone());
                field_values.push(lower_expr(db, path, ctx, block, value));
            }
            RecordField::Shorthand(name) => {
                field_names.push(name.clone());
                // Shorthand: name means name: name, look up the variable
                if let Some(value) = ctx.lookup(name) {
                    field_values.push(value);
                } else {
                    // Unresolved - emit src.var
                    let var_op = block.op(src::var(
                        db,
                        location,
                        core::Nil::new(db).as_type(),
                        sym(db, name),
                    ));
                    field_values.push(var_op.result(db));
                }
            }
            RecordField::Spread(expr) => {
                // Spread: ..expr - the expression should be a record of the same type
                // For now, emit as a special call that will be resolved later
                let spread_value = lower_expr(db, path, ctx, block, expr);
                // This needs special handling during resolution
                field_names.push("..".to_string());
                field_values.push(spread_value);
            }
        }
    }

    // Emit src.call with the type name as constructor
    // Field names are encoded in the attribute
    let op = block.op(src::call(
        db,
        location,
        field_values,
        core::Nil::new(db).as_type(),
        sym_ref(db, type_name),
    ));
    op.result(db)
}

/// Convert an AST pattern to an IR attribute representation.
fn pattern_to_attribute<'db>(pattern: &Pattern) -> Attribute<'db> {
    use case::pattern as pat;

    match pattern {
        Pattern::Wildcard => pat::wildcard(),
        Pattern::Identifier(name) => pat::ident(name),
        Pattern::Literal(lit) => {
            use tribute_ast::LiteralPattern;
            match lit {
                LiteralPattern::Nat(n) => pat::int(i64::try_from(*n).unwrap()),
                LiteralPattern::Int(n) => pat::int(*n),
                LiteralPattern::Float(_) => Attribute::String(format!("lit:{:?}", lit)),
                LiteralPattern::String(s) => pat::string(s),
                LiteralPattern::StringInterpolation(_) => {
                    Attribute::String(format!("lit:{:?}", lit))
                }
                LiteralPattern::Bytes(_) => Attribute::String(format!("lit:{:?}", lit)),
                LiteralPattern::BytesInterpolation(_) => {
                    Attribute::String(format!("lit:{:?}", lit))
                }
                LiteralPattern::Rune(c) => Attribute::String(format!("lit:?{}", c)),
                LiteralPattern::Bool(b) => pat::bool(*b),
                LiteralPattern::Nil => Attribute::String("Nil".to_string()),
            }
        }
        Pattern::Constructor(ctor) => match &ctor.args {
            ConstructorArgs::None => pat::unit_variant(&ctor.name),
            ConstructorArgs::Positional(pats) => {
                let fields: Vec<_> = pats.iter().map(pattern_to_attribute).collect();
                pat::variant(&ctor.name, &fields)
            }
            ConstructorArgs::Named { fields, .. } => {
                // Named fields: encode as "Name { f1: p1, f2: p2 }"
                let fs: Vec<_> = fields
                    .iter()
                    .map(|f| {
                        let inner = pattern_to_attribute(&f.pattern);
                        let inner_str = match &inner {
                            Attribute::String(s) => s.clone(),
                            _ => format!("{:?}", inner),
                        };
                        format!("{}: {}", f.name, inner_str)
                    })
                    .collect();
                Attribute::String(format!("{} {{ {} }}", ctor.name, fs.join(", ")))
            }
        },
        Pattern::Tuple(first, rest) => {
            let mut elems = vec![pattern_to_attribute(first)];
            elems.extend(rest.iter().map(pattern_to_attribute));
            pat::tuple(&elems)
        }
        Pattern::List(ListPattern { elements, rest }) => {
            let head: Vec<_> = elements.iter().map(pattern_to_attribute).collect();
            match rest {
                Some(rest_name) => pat::list_rest(&head, rest_name.as_deref()),
                None => pat::list(&head),
            }
        }
        Pattern::Handler(hp) => Attribute::String(format!("handler:{:?}", hp)),
        Pattern::As(inner, name) => pat::as_pattern(pattern_to_attribute(inner), name),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa::Database;
    use std::path::PathBuf;
    use tribute_ast::{BinaryExpression, Parameter};
    use tribute_core::TributeDatabaseImpl;
    use tribute_trunk_ir::DialectOp;

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
            let func_ty = ty else {
                panic!("expected type attribute");
            };
            assert!(func_ty.is_function(db));

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
            assert_eq!(var_x.name(db).text(db), "x");

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
            let func_ty = ty else {
                panic!("expected type attribute");
            };
            let params = func_ty.function_params(db).expect("expected function type");
            assert_eq!(params.len(), 2);

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
            assert_eq!(var_x.name(db).text(db), "x");
            let var_y = src::Var::from_operation(db, lambda_ops[1]).unwrap();
            assert_eq!(var_y.name(db).text(db), "y");

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

    /// Helper to create AST with tuple: fn main() { #(1, 2, 3) }
    #[salsa::tracked]
    fn lower_tuple_helper(db: &dyn salsa::Database) -> core::Module<'_> {
        let path = PathId::new(db, PathBuf::from("test.tr"));

        // Create AST: fn main() { #(1, 2, 3) }
        let tuple_expr = Expr::Tuple(
            Box::new((Expr::Nat(1), Span::new(14, 15))),
            vec![
                (Expr::Nat(2), Span::new(17, 18)),
                (Expr::Nat(3), Span::new(20, 21)),
            ],
        );

        let body = tribute_ast::Block {
            statements: vec![Statement::Expression((tuple_expr, Span::new(12, 23)))],
        };

        let func_def =
            FunctionDefinition::new(db, "main".to_string(), vec![], None, body, Span::new(0, 25));

        let item = tribute_ast::Item::new(db, ItemKind::Function(func_def), Span::new(0, 25));
        let program = Program::new(db, vec![item]);

        lower_program(db, path, program)
    }

    #[test]
    fn test_lower_tuple_expression() {
        TributeDatabaseImpl::default().attach(|db| {
            let module = lower_tuple_helper(db);

            // Get the function
            let body_region = module.body(db);
            let blocks = body_region.blocks(db);
            let func_op = func::Func::from_operation(db, blocks[0].operations(db)[0]).unwrap();

            // Get the function's entry block
            let func_body = func_op.body(db);
            let func_blocks = func_body.blocks(db);
            let entry_block = &func_blocks[0];
            let ops = entry_block.operations(db);

            // Should have: arith.const(1), arith.const(2), arith.const(3), src.tuple, func.return
            assert_eq!(ops.len(), 5);

            // Verify the constants
            let const_1 = arith::Const::from_operation(db, ops[0]).unwrap();
            let const_2 = arith::Const::from_operation(db, ops[1]).unwrap();
            let const_3 = arith::Const::from_operation(db, ops[2]).unwrap();

            // Verify the tuple operation
            let tuple_op = src::Tuple::from_operation(db, ops[3]).unwrap();
            let elements = tuple_op.elements(db);
            assert_eq!(elements.len(), 3);
            assert_eq!(elements[0], const_1.result(db));
            assert_eq!(elements[1], const_2.result(db));
            assert_eq!(elements[2], const_3.result(db));

            // Return should use the tuple's result
            let ret_op = func::Return::from_operation(db, ops[4]).unwrap();
            assert_eq!(ret_op.operands(db)[0], tuple_op.result(db));
        });
    }

    /// Helper to create AST with rune literal: fn main() { ?a }
    #[salsa::tracked]
    fn lower_rune_helper(db: &dyn salsa::Database) -> core::Module<'_> {
        let path = PathId::new(db, PathBuf::from("test.tr"));

        // Create AST: fn main() { ?a }
        let rune_expr = Expr::Rune('a');

        let body = tribute_ast::Block {
            statements: vec![Statement::Expression((rune_expr, Span::new(12, 14)))],
        };

        let func_def =
            FunctionDefinition::new(db, "main".to_string(), vec![], None, body, Span::new(0, 16));

        let item = tribute_ast::Item::new(db, ItemKind::Function(func_def), Span::new(0, 16));
        let program = Program::new(db, vec![item]);

        lower_program(db, path, program)
    }

    #[test]
    fn test_lower_rune_expression() {
        TributeDatabaseImpl::default().attach(|db| {
            let module = lower_rune_helper(db);

            // Get the function
            let body_region = module.body(db);
            let blocks = body_region.blocks(db);
            let func_op = func::Func::from_operation(db, blocks[0].operations(db)[0]).unwrap();

            // Get the function's entry block
            let func_body = func_op.body(db);
            let func_blocks = func_body.blocks(db);
            let entry_block = &func_blocks[0];
            let ops = entry_block.operations(db);

            // Should have: arith.const('a'), func.return
            assert_eq!(ops.len(), 2);

            // Verify first op is arith.const with i32 type
            let const_op = arith::Const::from_operation(db, ops[0]).unwrap();
            assert_eq!(const_op.result_ty(db), core::I32::new(db).as_type());

            // Verify the value is 'a' (97)
            assert_eq!(const_op.value(db), Attribute::IntBits(97));
        });
    }

    /// Helper to create AST with method call: fn main() { x.f(y) }
    #[salsa::tracked]
    fn lower_method_call_helper(db: &dyn salsa::Database) -> core::Module<'_> {
        let path = PathId::new(db, PathBuf::from("test.tr"));

        // Create AST: fn main() { x.f(y) }
        let method_call = Expr::MethodCall(MethodCallExpression {
            receiver: Box::new((Expr::Identifier("x".to_string()), Span::new(12, 13))),
            method: "f".to_string(),
            arguments: vec![(Expr::Identifier("y".to_string()), Span::new(16, 17))],
        });

        let body = tribute_ast::Block {
            statements: vec![Statement::Expression((method_call, Span::new(12, 19)))],
        };

        let func_def =
            FunctionDefinition::new(db, "main".to_string(), vec![], None, body, Span::new(0, 21));

        let item = tribute_ast::Item::new(db, ItemKind::Function(func_def), Span::new(0, 21));
        let program = Program::new(db, vec![item]);

        lower_program(db, path, program)
    }

    #[test]
    fn test_lower_method_call() {
        TributeDatabaseImpl::default().attach(|db| {
            let module = lower_method_call_helper(db);

            // Get the function
            let body_region = module.body(db);
            let blocks = body_region.blocks(db);
            let func_op = func::Func::from_operation(db, blocks[0].operations(db)[0]).unwrap();

            // Get the function's entry block
            let func_body = func_op.body(db);
            let func_blocks = func_body.blocks(db);
            let entry_block = &func_blocks[0];
            let ops = entry_block.operations(db);

            // Should have: src.var(x), src.var(y), src.call(f, [x, y]), func.return
            assert_eq!(ops.len(), 4);

            // Verify src.var for x
            let var_x = src::Var::from_operation(db, ops[0]).unwrap();
            assert_eq!(var_x.name(db).text(db), "x");

            // Verify src.var for y
            let var_y = src::Var::from_operation(db, ops[1]).unwrap();
            assert_eq!(var_y.name(db).text(db), "y");

            // Verify src.call with UFCS: x.f(y) → f(x, y)
            let call_op = src::Call::from_operation(db, ops[2]).unwrap();
            assert_eq!(call_op.name(db).to_vec(), vec![Symbol::new(db, "f")]);
            let args = call_op.operands(db);
            assert_eq!(args.len(), 2);
            assert_eq!(args[0], var_x.result(db)); // receiver first
            assert_eq!(args[1], var_y.result(db)); // then arguments
        });
    }

    /// Helper to create AST with list: fn main() { [1, 2, 3] }
    #[salsa::tracked]
    fn lower_list_helper(db: &dyn salsa::Database) -> core::Module<'_> {
        let path = PathId::new(db, PathBuf::from("test.tr"));

        // Create AST: fn main() { [1, 2, 3] }
        let list_expr = Expr::List(vec![
            (Expr::Nat(1), Span::new(13, 14)),
            (Expr::Nat(2), Span::new(16, 17)),
            (Expr::Nat(3), Span::new(19, 20)),
        ]);

        let body = tribute_ast::Block {
            statements: vec![Statement::Expression((list_expr, Span::new(12, 22)))],
        };

        let func_def =
            FunctionDefinition::new(db, "main".to_string(), vec![], None, body, Span::new(0, 24));

        let item = tribute_ast::Item::new(db, ItemKind::Function(func_def), Span::new(0, 24));
        let program = Program::new(db, vec![item]);

        lower_program(db, path, program)
    }

    #[test]
    fn test_lower_list_expression() {
        TributeDatabaseImpl::default().attach(|db| {
            let module = lower_list_helper(db);

            // Get the function
            let body_region = module.body(db);
            let blocks = body_region.blocks(db);
            let func_op = func::Func::from_operation(db, blocks[0].operations(db)[0]).unwrap();

            // Get the function's entry block
            let func_body = func_op.body(db);
            let func_blocks = func_body.blocks(db);
            let entry_block = &func_blocks[0];
            let ops = entry_block.operations(db);

            // Should have: arith.const(1), arith.const(2), arith.const(3), list.new([1,2,3]), func.return
            assert_eq!(ops.len(), 5);

            // Verify the list.new operation
            let list_new = list::New::from_operation(db, ops[3]).unwrap();
            assert_eq!(list_new.elements(db).len(), 3);
        });
    }

    /// Helper to create AST with record: fn main() { User { name, age: 30 } }
    #[salsa::tracked]
    fn lower_record_helper(db: &dyn salsa::Database) -> core::Module<'_> {
        let path = PathId::new(db, PathBuf::from("test.tr"));

        // Create AST: fn main() { User { name, age: 30 } }
        let record_expr = Expr::Record(RecordExpression {
            type_name: "User".to_string(),
            fields: vec![
                RecordField::Shorthand("name".to_string()),
                RecordField::Field {
                    name: "age".to_string(),
                    value: (Expr::Nat(30), Span::new(30, 32)),
                },
            ],
        });

        let body = tribute_ast::Block {
            statements: vec![Statement::Expression((record_expr, Span::new(12, 35)))],
        };

        let func_def =
            FunctionDefinition::new(db, "main".to_string(), vec![], None, body, Span::new(0, 37));

        let item = tribute_ast::Item::new(db, ItemKind::Function(func_def), Span::new(0, 37));
        let program = Program::new(db, vec![item]);

        lower_program(db, path, program)
    }

    #[test]
    fn test_lower_record_expression() {
        TributeDatabaseImpl::default().attach(|db| {
            let module = lower_record_helper(db);

            // Get the function
            let body_region = module.body(db);
            let blocks = body_region.blocks(db);
            let func_op = func::Func::from_operation(db, blocks[0].operations(db)[0]).unwrap();

            // Get the function's entry block
            let func_body = func_op.body(db);
            let func_blocks = func_body.blocks(db);
            let entry_block = &func_blocks[0];
            let ops = entry_block.operations(db);

            // Should have: src.var(name), arith.const(30), src.call(User, [name, 30]), func.return
            assert_eq!(ops.len(), 4);

            // Verify src.var for shorthand field "name"
            let var_name = src::Var::from_operation(db, ops[0]).unwrap();
            assert_eq!(var_name.name(db).text(db), "name");

            // Verify arith.const for age
            let const_30 = arith::Const::from_operation(db, ops[1]).unwrap();
            assert_eq!(const_30.value(db), Attribute::IntBits(30));

            // Verify src.call with User constructor
            let record_call = src::Call::from_operation(db, ops[2]).unwrap();
            assert_eq!(record_call.name(db).to_vec(), vec![Symbol::new(db, "User")]);
            let args = record_call.operands(db);
            assert_eq!(args.len(), 2);
            assert_eq!(args[0], var_name.result(db));
            assert_eq!(args[1], const_30.result(db));
        });
    }

    /// Helper to create AST with operator function: fn main() { (+) }
    #[salsa::tracked]
    fn lower_operator_fn_helper(db: &dyn salsa::Database) -> core::Module<'_> {
        let path = PathId::new(db, PathBuf::from("test.tr"));

        // Create AST: fn main() { (+) }
        let op_fn_expr = Expr::OperatorFn(OperatorFnExpression {
            op: "+".to_string(),
            qualifier: None,
        });

        let body = tribute_ast::Block {
            statements: vec![Statement::Expression((op_fn_expr, Span::new(12, 15)))],
        };

        let func_def =
            FunctionDefinition::new(db, "main".to_string(), vec![], None, body, Span::new(0, 17));

        let item = tribute_ast::Item::new(db, ItemKind::Function(func_def), Span::new(0, 17));
        let program = Program::new(db, vec![item]);

        lower_program(db, path, program)
    }

    #[test]
    fn test_lower_operator_fn() {
        TributeDatabaseImpl::default().attach(|db| {
            let module = lower_operator_fn_helper(db);

            // Get the function
            let body_region = module.body(db);
            let blocks = body_region.blocks(db);
            let func_op = func::Func::from_operation(db, blocks[0].operations(db)[0]).unwrap();

            // Get the function's entry block
            let func_body = func_op.body(db);
            let func_blocks = func_body.blocks(db);
            let entry_block = &func_blocks[0];
            let ops = entry_block.operations(db);

            // Should have: src.var(+), func.return
            assert_eq!(ops.len(), 2);

            // Verify src.var for operator function "+"
            let var_op = src::Var::from_operation(db, ops[0]).unwrap();
            assert_eq!(var_op.name(db).text(db), "+");
        });
    }

    /// Helper to create AST with qualified operator function: fn main() { (Int::+) }
    #[salsa::tracked]
    fn lower_qualified_operator_fn_helper(db: &dyn salsa::Database) -> core::Module<'_> {
        let path = PathId::new(db, PathBuf::from("test.tr"));

        // Create AST: fn main() { (Int::+) }
        let op_fn_expr = Expr::OperatorFn(OperatorFnExpression {
            op: "+".to_string(),
            qualifier: Some("Int".to_string()),
        });

        let body = tribute_ast::Block {
            statements: vec![Statement::Expression((op_fn_expr, Span::new(12, 19)))],
        };

        let func_def =
            FunctionDefinition::new(db, "main".to_string(), vec![], None, body, Span::new(0, 21));

        let item = tribute_ast::Item::new(db, ItemKind::Function(func_def), Span::new(0, 21));
        let program = Program::new(db, vec![item]);

        lower_program(db, path, program)
    }

    #[test]
    fn test_lower_qualified_operator_fn() {
        TributeDatabaseImpl::default().attach(|db| {
            let module = lower_qualified_operator_fn_helper(db);

            // Get the function
            let body_region = module.body(db);
            let blocks = body_region.blocks(db);
            let func_op = func::Func::from_operation(db, blocks[0].operations(db)[0]).unwrap();

            // Get the function's entry block
            let func_body = func_op.body(db);
            let func_blocks = func_body.blocks(db);
            let entry_block = &func_blocks[0];
            let ops = entry_block.operations(db);

            // Should have: src.path(["Int", "+"]), func.return
            assert_eq!(ops.len(), 2);

            // Verify src.path for qualified operator function "Int::+"
            let path_op = src::Path::from_operation(db, ops[0]).unwrap();
            let path = path_op.path(db);
            assert_eq!(path.len(), 2);
            assert_eq!(path[0].text(db), "Int");
            assert_eq!(path[1].text(db), "+");
        });
    }

    /// Helper to create AST with type variables: fn identity(x: t) -> t { x }
    #[salsa::tracked]
    fn lower_type_var_helper(db: &dyn salsa::Database) -> core::Module<'_> {
        let path = PathId::new(db, PathBuf::from("test.tr"));

        // Create AST: fn identity(x: t) -> t { x }
        let body = tribute_ast::Block {
            statements: vec![Statement::Expression((
                Expr::Identifier("x".to_string()),
                Span::new(28, 29),
            ))],
        };

        let func_def = FunctionDefinition::new(
            db,
            "identity".to_string(),
            vec![Parameter {
                name: "x".to_string(),
                ty: Some(TypeRef::Variable("t".to_string())),
            }],
            Some(TypeRef::Variable("t".to_string())),
            body,
            Span::new(0, 31),
        );

        let item = tribute_ast::Item::new(db, ItemKind::Function(func_def), Span::new(0, 31));
        let program = Program::new(db, vec![item]);

        lower_program(db, path, program)
    }

    #[test]
    fn test_type_variable_consistency() {
        TributeDatabaseImpl::default().attach(|db| {
            let module = lower_type_var_helper(db);

            // Get the function
            let body_region = module.body(db);
            let blocks = body_region.blocks(db);
            let func_op = func::Func::from_operation(db, blocks[0].operations(db)[0]).unwrap();

            // Get function type and extract params/result
            let func_ty = func_op.ty(db);
            let params = func_ty.function_params(db).expect("Expected function type");
            let result_ty = func_ty.function_result(db).expect("Expected function type");

            assert_eq!(params.len(), 1);
            let param_ty = params[0];

            // Both should be ty.var with the same ID
            // Verify both are ty.var dialect types
            assert_eq!(param_ty.dialect(db).text(db), "type");
            assert_eq!(param_ty.name(db).text(db), "var");
            assert_eq!(result_ty.dialect(db).text(db), "type");
            assert_eq!(result_ty.name(db).text(db), "var");
            // Same type variable name "t" should get the same ID
            assert_eq!(
                param_ty.attrs(db),
                result_ty.attrs(db),
                "Type variable t should have same ID"
            );

            // Additionally, verify they are the exact same Type (interned)
            assert_eq!(
                param_ty, result_ty,
                "Same type variable name should produce identical Type"
            );
        });
    }
}
