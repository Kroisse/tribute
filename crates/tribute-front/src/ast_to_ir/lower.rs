//! Core lowering logic.
//!
//! Transforms AST declarations and expressions to TrunkIR operations.

use trunk_ir::dialect::{adt, arith, core, func};
use trunk_ir::{Attribute, DialectType, IdVec, Location, PathId, Symbol};

use crate::ast::{
    Decl, Expr, ExprKind, FuncDecl, Module, ResolvedRef, SpanMap, Stmt, TypeAnnotation,
    TypeAnnotationKind, TypeKind, TypedRef,
};

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

    // Build parameter types from annotations, defaulting to int for untyped params
    let param_types: IdVec<trunk_ir::Type<'db>> = func
        .params
        .iter()
        .map(|p| convert_annotation_to_ir_type(ctx, p.ty.as_ref()))
        .collect();

    // Build return type from annotation, defaulting to unit
    let return_ty = func
        .return_ty
        .as_ref()
        .map(|ann| convert_annotation_to_ir_type(ctx, Some(ann)))
        .unwrap_or_else(|| ctx.unit_type());

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
        ExprKind::NatLit(n) => {
            let op = block.op(arith::Const::i64(ctx.db, location, n as i64));
            Some(op.result(ctx.db))
        }

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

        ExprKind::BytesLit(ref _bytes) => {
            // TODO: implement bytes constant lowering
            None
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
            // Determine operand type for selecting int vs float operations
            let is_float = is_float_expr(ctx.db, &lhs);
            let lhs_val = lower_expr(ctx, block, lhs)?;
            let rhs_val = lower_expr(ctx, block, rhs)?;
            lower_binop(ctx, block, op, lhs_val, rhs_val, is_float, location)
        }

        ExprKind::Block { stmts, value } => lower_block(ctx, block, stmts, value),

        // For other expressions, return unit as placeholder
        _ => {
            let ty = ctx.unit_type();
            let op = block.op(arith::r#const(ctx.db, location, ty, Attribute::Unit));
            Some(op.result(ctx.db))
        }
    }
}

/// Check if an expression evaluates to a float type.
///
/// This examines the expression structure and TypedRef to determine
/// whether to use floating-point operations.
fn is_float_expr<'db>(db: &'db dyn salsa::Database, expr: &Expr<TypedRef<'db>>) -> bool {
    match &*expr.kind {
        // Literal types
        ExprKind::FloatLit(_) => true,
        ExprKind::NatLit(_) | ExprKind::IntLit(_) => false,

        // For variables, check the TypedRef's type
        ExprKind::Var(typed_ref) => matches!(typed_ref.ty.kind(db), TypeKind::Float),

        // For binary operations, check both lhs and rhs
        ExprKind::BinOp { lhs, rhs, .. } => is_float_expr(db, lhs) || is_float_expr(db, rhs),

        // For blocks, check the value expression
        ExprKind::Block { value, .. } => is_float_expr(db, value),

        // Default to non-float
        _ => false,
    }
}

/// Lower a binary operation.
///
/// Uses the `is_float` flag to select between integer and floating-point operations.
fn lower_binop<'db>(
    ctx: &IrLoweringCtx<'db>,
    block: &mut trunk_ir::BlockBuilder<'db>,
    op: crate::ast::BinOpKind,
    lhs: trunk_ir::Value<'db>,
    rhs: trunk_ir::Value<'db>,
    is_float: bool,
    location: Location<'db>,
) -> Option<trunk_ir::Value<'db>> {
    use crate::ast::BinOpKind;

    let bool_ty = ctx.bool_type();
    let result_ty = if is_float {
        core::F64::new(ctx.db).as_type()
    } else {
        ctx.int_type()
    };

    let result = match op {
        BinOpKind::Add => block
            .op(arith::add(ctx.db, location, lhs, rhs, result_ty))
            .result(ctx.db),
        BinOpKind::Sub => block
            .op(arith::sub(ctx.db, location, lhs, rhs, result_ty))
            .result(ctx.db),
        BinOpKind::Mul => block
            .op(arith::mul(ctx.db, location, lhs, rhs, result_ty))
            .result(ctx.db),
        BinOpKind::Div => block
            .op(arith::div(ctx.db, location, lhs, rhs, result_ty))
            .result(ctx.db),
        BinOpKind::Mod => block
            .op(arith::rem(ctx.db, location, lhs, rhs, result_ty))
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
    value: Expr<TypedRef<'db>>,
) -> Option<trunk_ir::Value<'db>> {
    ctx.enter_scope();

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
                    if let crate::ast::PatternKind::Bind {
                        local_id: Some(local_id),
                        ..
                    } = &*pattern.kind
                    {
                        // Register the binding so Var expressions can find it
                        ctx.bind(*local_id, val);
                    }
                    // TODO: Handle other pattern kinds (tuple, record, etc.)
                }
            }
            Stmt::Expr { id: _, expr } => {
                let _ = lower_expr(ctx, block, expr);
            }
        }
    }

    let result = lower_expr(ctx, block, value);
    ctx.exit_scope();
    result
}

/// Convert a type annotation to a TrunkIR type.
///
/// Falls back to int_type() for unannotated parameters.
fn convert_annotation_to_ir_type<'db>(
    ctx: &IrLoweringCtx<'db>,
    annotation: Option<&TypeAnnotation>,
) -> trunk_ir::Type<'db> {
    let Some(ann) = annotation else {
        // No annotation - default to int
        return ctx.int_type();
    };

    match &ann.kind {
        TypeAnnotationKind::Named(name) => {
            // Map well-known type names using Symbol's PartialEq<&str>
            // Note: Both Int and Nat map to i64 in IR
            if *name == "Int" || *name == "Nat" {
                ctx.int_type()
            } else if *name == "Float" {
                core::F64::new(ctx.db).as_type()
            } else if *name == "Bool" {
                ctx.bool_type()
            } else if *name == "String" {
                core::String::new(ctx.db).as_type()
            } else if *name == "Bytes" {
                core::Bytes::new(ctx.db).as_type()
            } else if *name == "()" {
                ctx.unit_type()
            } else {
                // Unknown named type - use placeholder
                ctx.unit_type()
            }
        }
        TypeAnnotationKind::Path(_) => {
            // Qualified path - use placeholder for now
            ctx.unit_type()
        }
        TypeAnnotationKind::App { ctor, .. } => {
            // Parameterized type - convert the constructor
            convert_annotation_to_ir_type(ctx, Some(ctor))
        }
        _ => ctx.unit_type(),
    }
}
