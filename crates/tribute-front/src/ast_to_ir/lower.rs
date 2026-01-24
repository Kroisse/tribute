//! Core lowering logic.
//!
//! Transforms AST declarations and expressions to TrunkIR operations.

use trunk_ir::dialect::{adt, arith, core, func};
use trunk_ir::{Attribute, DialectType, IdVec, Location, PathId, Symbol};

use crate::ast::{Decl, Expr, ExprKind, FuncDecl, Module, ResolvedRef, SpanMap, Stmt, TypedRef};

use super::context::IrLoweringCtx;

/// Lower a module to TrunkIR.
pub fn lower_module<'db>(
    db: &'db dyn salsa::Database,
    path: PathId<'db>,
    span_map: SpanMap,
    module: Module<TypedRef<'db>>,
) -> core::Module<'db> {
    // Use module's NodeId to get location
    let module_location = span_map.get_or_default(module.id);
    let location = Location::new(path, module_location);
    let module_name = module.name.unwrap_or_else(|| Symbol::new("main"));

    core::Module::build(db, location, module_name, |top| {
        let mut ctx = IrLoweringCtx::new(db, path, span_map.clone());

        for decl in module.decls {
            lower_decl(&mut ctx, top, decl);
        }
    })
}

/// Lower a declaration to TrunkIR.
fn lower_decl<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    top: &mut trunk_ir::BlockBuilder<'db>,
    decl: Decl<TypedRef<'db>>,
) {
    match decl {
        Decl::Function(func) => lower_function(ctx, top, func),
        Decl::Struct(_) => {
            // TODO: Lower struct declarations
        }
        Decl::Enum(_) => {
            // TODO: Lower enum declarations
        }
        Decl::Const(_) => {
            // TODO: Lower const declarations
        }
        Decl::Ability(_) => {
            // TODO: Lower ability declarations
        }
        Decl::Use(_) => {
            // Use declarations don't generate IR
        }
    }
}

/// Lower a function declaration.
fn lower_function<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    top: &mut trunk_ir::BlockBuilder<'db>,
    func: FuncDecl<TypedRef<'db>>,
) {
    let location = ctx.location(func.id);
    let func_name = func.name;

    // Build parameter types
    let param_types: IdVec<trunk_ir::Type<'db>> =
        func.params.iter().map(|_| ctx.int_type()).collect();

    // Build return type
    let return_ty = ctx.unit_type();

    // Create function operation
    let func_op = func::Func::build(
        ctx.db,
        location,
        func_name,
        param_types,
        return_ty,
        |body| {
            ctx.enter_scope();

            // Lower function body
            if let Some(result) = lower_expr(ctx, body, func.body) {
                body.op(func::Return::value(ctx.db, location, result));
            } else {
                // Return unit
                let unit = body.op(arith::r#const(
                    ctx.db,
                    location,
                    ctx.unit_type(),
                    Attribute::Unit,
                ));
                body.op(func::Return::value(ctx.db, location, unit.result(ctx.db)));
            }

            ctx.exit_scope();
        },
    );

    top.op(func_op);
}

/// Lower an expression to TrunkIR.
fn lower_expr<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    block: &mut trunk_ir::BlockBuilder<'db>,
    expr: Expr<TypedRef<'db>>,
) -> Option<trunk_ir::Value<'db>> {
    let location = ctx.location(expr.id);

    match *expr.kind {
        ExprKind::IntLit(n) => {
            let op = block.op(arith::Const::i64(ctx.db, location, n));
            Some(op.result(ctx.db))
        }

        ExprKind::FloatLit(f) => {
            let op = block.op(arith::Const::f64(ctx.db, location, f.value()));
            Some(op.result(ctx.db))
        }

        ExprKind::BoolLit(b) => {
            let ty = ctx.bool_type();
            let op = block.op(arith::r#const(ctx.db, location, ty, b.into()));
            Some(op.result(ctx.db))
        }

        ExprKind::StringLit(ref s) => {
            let ty = core::String::new(ctx.db).as_type();
            let op = block.op(adt::string_const(ctx.db, location, ty, s.clone()));
            Some(op.result(ctx.db))
        }

        ExprKind::Nil => {
            let ty = ctx.unit_type();
            let op = block.op(arith::r#const(ctx.db, location, ty, Attribute::Unit));
            Some(op.result(ctx.db))
        }

        ExprKind::Var(ref typed_ref) => match &typed_ref.resolved {
            ResolvedRef::Local { id, .. } => ctx.lookup(*id),
            ResolvedRef::Function { id } => {
                let func_name = id.name(ctx.db);
                let func_ty = ctx.convert_type(typed_ref.ty);
                let op = block.op(func::constant(ctx.db, location, func_ty, func_name));
                Some(op.result(ctx.db))
            }
            ResolvedRef::Constructor { variant, .. } => {
                let func_ty = ctx.convert_type(typed_ref.ty);
                let op = block.op(func::constant(ctx.db, location, func_ty, *variant));
                Some(op.result(ctx.db))
            }
            ResolvedRef::Builtin(_) | ResolvedRef::Module { .. } => None,
        },

        ExprKind::BinOp { op, lhs, rhs } => {
            let lhs_val = lower_expr(ctx, block, lhs)?;
            let rhs_val = lower_expr(ctx, block, rhs)?;
            lower_binop(ctx, block, op, lhs_val, rhs_val, location)
        }

        ExprKind::Block(stmts) => lower_block(ctx, block, stmts),

        // For other expressions, return unit as placeholder
        _ => {
            let ty = ctx.unit_type();
            let op = block.op(arith::r#const(ctx.db, location, ty, Attribute::Unit));
            Some(op.result(ctx.db))
        }
    }
}

/// Lower a binary operation.
fn lower_binop<'db>(
    ctx: &IrLoweringCtx<'db>,
    block: &mut trunk_ir::BlockBuilder<'db>,
    op: crate::ast::BinOpKind,
    lhs: trunk_ir::Value<'db>,
    rhs: trunk_ir::Value<'db>,
    location: Location<'db>,
) -> Option<trunk_ir::Value<'db>> {
    use crate::ast::BinOpKind;

    // Get the result type - for arithmetic ops, use i64; for comparisons, use bool
    let int_ty = ctx.int_type();
    let bool_ty = ctx.bool_type();

    let result = match op {
        BinOpKind::Add => block
            .op(arith::add(ctx.db, location, lhs, rhs, int_ty))
            .result(ctx.db),
        BinOpKind::Sub => block
            .op(arith::sub(ctx.db, location, lhs, rhs, int_ty))
            .result(ctx.db),
        BinOpKind::Mul => block
            .op(arith::mul(ctx.db, location, lhs, rhs, int_ty))
            .result(ctx.db),
        BinOpKind::Div => block
            .op(arith::div(ctx.db, location, lhs, rhs, int_ty))
            .result(ctx.db),
        BinOpKind::Mod => block
            .op(arith::rem(ctx.db, location, lhs, rhs, int_ty))
            .result(ctx.db),
        BinOpKind::Eq => block
            .op(arith::cmp_eq(ctx.db, location, lhs, rhs, bool_ty))
            .result(ctx.db),
        BinOpKind::Ne => block
            .op(arith::cmp_ne(ctx.db, location, lhs, rhs, bool_ty))
            .result(ctx.db),
        BinOpKind::Lt => block
            .op(arith::cmp_lt(ctx.db, location, lhs, rhs, bool_ty))
            .result(ctx.db),
        BinOpKind::Le => block
            .op(arith::cmp_le(ctx.db, location, lhs, rhs, bool_ty))
            .result(ctx.db),
        BinOpKind::Gt => block
            .op(arith::cmp_gt(ctx.db, location, lhs, rhs, bool_ty))
            .result(ctx.db),
        BinOpKind::Ge => block
            .op(arith::cmp_ge(ctx.db, location, lhs, rhs, bool_ty))
            .result(ctx.db),
        BinOpKind::And => block
            .op(arith::and(ctx.db, location, lhs, rhs, bool_ty))
            .result(ctx.db),
        BinOpKind::Or => block
            .op(arith::or(ctx.db, location, lhs, rhs, bool_ty))
            .result(ctx.db),
        BinOpKind::Concat => {
            // String concatenation not yet supported in trunk-ir
            // Return unit as placeholder
            let ty = ctx.unit_type();
            block
                .op(arith::r#const(ctx.db, location, ty, Attribute::Unit))
                .result(ctx.db)
        }
    };
    Some(result)
}

/// Lower a block of statements.
fn lower_block<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    block: &mut trunk_ir::BlockBuilder<'db>,
    stmts: Vec<Stmt<TypedRef<'db>>>,
) -> Option<trunk_ir::Value<'db>> {
    ctx.enter_scope();
    let mut last_value = None;

    for stmt in stmts {
        match stmt {
            Stmt::Let {
                id: _,
                pattern,
                ty: _,
                value,
            } => {
                if let Some(val) = lower_expr(ctx, block, value) {
                    // Simple case: bind pattern is just a name
                    if let crate::ast::PatternKind::Bind { name: _ } = &*pattern.kind {
                        // TODO: Properly track bindings
                    }
                    last_value = Some(val);
                }
            }
            Stmt::Expr { id: _, expr } => {
                last_value = lower_expr(ctx, block, expr);
            }
            Stmt::Return { id, expr } => {
                if let Some(val) = lower_expr(ctx, block, expr) {
                    let location = ctx.location(id);
                    block.op(func::Return::value(ctx.db, location, val));
                }
                break;
            }
        }
    }

    ctx.exit_scope();
    last_value
}
