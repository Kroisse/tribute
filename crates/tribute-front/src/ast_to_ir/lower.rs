//! Core lowering logic.
//!
//! Transforms AST declarations and expressions to TrunkIR operations.

use salsa::Accumulator;
use tribute_core::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use trunk_ir::dialect::{adt, arith, core, func};
use trunk_ir::{Attribute, DialectType, Location, PathId, Symbol};

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
        Decl::Ability(_) => {
            // TODO: Lower ability declarations
        }
        Decl::Use(_) => {
            // Use declarations don't generate IR
        }
        Decl::Module(m) => {
            // Inline module: recursively lower nested declarations
            if let Some(body) = m.body {
                for inner_decl in body {
                    lower_decl(ctx, top, inner_decl);
                }
            }
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

    // Build parameter types with optional names
    let params: Vec<(trunk_ir::Type<'db>, Option<Symbol>)> = func
        .params
        .iter()
        .map(|p| {
            let ty = convert_annotation_to_ir_type(ctx, p.ty.as_ref());
            (ty, Some(p.name))
        })
        .collect();

    // Build return type from annotation, defaulting to unit
    let return_ty = func
        .return_ty
        .as_ref()
        .map(|ann| convert_annotation_to_ir_type(ctx, Some(ann)))
        .unwrap_or_else(|| ctx.unit_type());

    // Create function operation with named params
    let func_op = func::Func::build_with_named_params(
        ctx.db,
        location,
        func_name,
        None,
        params.clone(),
        return_ty,
        None,
        |body, arg_values| {
            ctx.enter_scope();

            // Bind parameters to their block argument values
            // Use index-based matching to handle params with None local_id correctly
            for (i, param) in func.params.iter().enumerate() {
                if let Some(local_id) = param.local_id {
                    ctx.bind(local_id, arg_values[i]);
                }
            }

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
            // i31ref range check for WASM target compatibility.
            // i31 range: -2^30 to 2^30-1, NatLit (positive): 0 to 2^30-1
            const I31_MAX: u64 = (1 << 30) - 1; // 1,073,741,823

            if n > I31_MAX {
                Diagnostic {
                    message: format!(
                        "natural number literal {} exceeds i31 range (max: {})",
                        n, I31_MAX
                    ),
                    span: location.span,
                    severity: DiagnosticSeverity::Error,
                    phase: CompilationPhase::Lowering,
                }
                .accumulate(ctx.db);
                None
            } else {
                let op = block.op(arith::Const::i64(ctx.db, location, n as i64));
                Some(op.result(ctx.db))
            }
        }

        ExprKind::IntLit(n) => {
            let op = block.op(arith::Const::i64(ctx.db, location, n));
            Some(op.result(ctx.db))
        }

        ExprKind::RuneLit(c) => {
            // Rune is lowered as i32 (Unicode code point, matching core::I32).
            // Unicode code points max out at 0x10FFFF (1,114,111), which is within
            // i31 range (max 1,073,741,823), so no range check is needed.
            let op = block.op(arith::Const::i32(ctx.db, location, c as i32));
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
            // Bytes literal lowering not yet supported - emit diagnostic
            Diagnostic {
                message: "bytes literal not yet supported in IR lowering".to_string(),
                span: location.span,
                severity: DiagnosticSeverity::Warning,
                phase: CompilationPhase::Lowering,
            }
            .accumulate(ctx.db);
            // Return unit as placeholder
            let ty = ctx.unit_type();
            let op = block.op(arith::r#const(ctx.db, location, ty, Attribute::Unit));
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
            // Determine operand type for selecting int vs float operations.
            // Check both operands: mixed int+float or float+int should use float operations.
            let is_float = is_float_expr(ctx.db, &lhs) || is_float_expr(ctx.db, &rhs);
            let lhs_val = lower_expr(ctx, block, lhs)?;
            let rhs_val = lower_expr(ctx, block, rhs)?;
            lower_binop(ctx, block, op, lhs_val, rhs_val, is_float, location)
        }

        ExprKind::Block { stmts, value } => lower_block(ctx, block, stmts, value),

        // === Expressions not yet implemented ===
        // Each of these emits a diagnostic and returns a unit placeholder.
        // This makes it clear which expressions are missing lowering support.
        ExprKind::Call { .. } => {
            Diagnostic {
                message: "function call not yet supported in IR lowering".to_string(),
                span: location.span,
                severity: DiagnosticSeverity::Warning,
                phase: CompilationPhase::Lowering,
            }
            .accumulate(ctx.db);
            let ty = ctx.unit_type();
            let op = block.op(arith::r#const(ctx.db, location, ty, Attribute::Unit));
            Some(op.result(ctx.db))
        }

        ExprKind::Cons { .. } => {
            Diagnostic {
                message: "constructor application not yet supported in IR lowering".to_string(),
                span: location.span,
                severity: DiagnosticSeverity::Warning,
                phase: CompilationPhase::Lowering,
            }
            .accumulate(ctx.db);
            let ty = ctx.unit_type();
            let op = block.op(arith::r#const(ctx.db, location, ty, Attribute::Unit));
            Some(op.result(ctx.db))
        }

        ExprKind::Record { .. } => {
            Diagnostic {
                message: "record construction not yet supported in IR lowering".to_string(),
                span: location.span,
                severity: DiagnosticSeverity::Warning,
                phase: CompilationPhase::Lowering,
            }
            .accumulate(ctx.db);
            let ty = ctx.unit_type();
            let op = block.op(arith::r#const(ctx.db, location, ty, Attribute::Unit));
            Some(op.result(ctx.db))
        }

        ExprKind::FieldAccess { .. } => {
            Diagnostic {
                message: "field access not yet supported in IR lowering".to_string(),
                span: location.span,
                severity: DiagnosticSeverity::Warning,
                phase: CompilationPhase::Lowering,
            }
            .accumulate(ctx.db);
            let ty = ctx.unit_type();
            let op = block.op(arith::r#const(ctx.db, location, ty, Attribute::Unit));
            Some(op.result(ctx.db))
        }

        ExprKind::MethodCall { .. } => {
            Diagnostic {
                message: "method call not yet supported in IR lowering".to_string(),
                span: location.span,
                severity: DiagnosticSeverity::Warning,
                phase: CompilationPhase::Lowering,
            }
            .accumulate(ctx.db);
            let ty = ctx.unit_type();
            let op = block.op(arith::r#const(ctx.db, location, ty, Attribute::Unit));
            Some(op.result(ctx.db))
        }

        ExprKind::Case { .. } => {
            Diagnostic {
                message: "case expression not yet supported in IR lowering".to_string(),
                span: location.span,
                severity: DiagnosticSeverity::Warning,
                phase: CompilationPhase::Lowering,
            }
            .accumulate(ctx.db);
            let ty = ctx.unit_type();
            let op = block.op(arith::r#const(ctx.db, location, ty, Attribute::Unit));
            Some(op.result(ctx.db))
        }

        ExprKind::Lambda { .. } => {
            Diagnostic {
                message: "lambda expression not yet supported in IR lowering".to_string(),
                span: location.span,
                severity: DiagnosticSeverity::Warning,
                phase: CompilationPhase::Lowering,
            }
            .accumulate(ctx.db);
            let ty = ctx.unit_type();
            let op = block.op(arith::r#const(ctx.db, location, ty, Attribute::Unit));
            Some(op.result(ctx.db))
        }

        ExprKind::Handle { .. } => {
            Diagnostic {
                message: "handle expression not yet supported in IR lowering".to_string(),
                span: location.span,
                severity: DiagnosticSeverity::Warning,
                phase: CompilationPhase::Lowering,
            }
            .accumulate(ctx.db);
            let ty = ctx.unit_type();
            let op = block.op(arith::r#const(ctx.db, location, ty, Attribute::Unit));
            Some(op.result(ctx.db))
        }

        ExprKind::Tuple(_) => {
            Diagnostic {
                message: "tuple expression not yet supported in IR lowering".to_string(),
                span: location.span,
                severity: DiagnosticSeverity::Warning,
                phase: CompilationPhase::Lowering,
            }
            .accumulate(ctx.db);
            let ty = ctx.unit_type();
            let op = block.op(arith::r#const(ctx.db, location, ty, Attribute::Unit));
            Some(op.result(ctx.db))
        }

        ExprKind::List(_) => {
            Diagnostic {
                message: "list expression not yet supported in IR lowering".to_string(),
                span: location.span,
                severity: DiagnosticSeverity::Warning,
                phase: CompilationPhase::Lowering,
            }
            .accumulate(ctx.db);
            let ty = ctx.unit_type();
            let op = block.op(arith::r#const(ctx.db, location, ty, Attribute::Unit));
            Some(op.result(ctx.db))
        }

        ExprKind::Error => {
            // Error expression from parsing - just return unit placeholder
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
            // String concatenation not yet supported in trunk-ir - emit diagnostic
            Diagnostic {
                message: "string concatenation not yet supported in IR lowering".to_string(),
                span: location.span,
                severity: DiagnosticSeverity::Warning,
                phase: CompilationPhase::Lowering,
            }
            .accumulate(ctx.db);
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
                    match &*pattern.kind {
                        crate::ast::PatternKind::Bind {
                            local_id: Some(local_id),
                            ..
                        } => {
                            // Register the binding so Var expressions can find it
                            ctx.bind(*local_id, val);
                        }
                        crate::ast::PatternKind::Wildcard => {
                            // Wildcard binds nothing, value is computed for side effects
                        }
                        _ => {
                            // Pattern destructuring not yet supported in IR lowering
                            let location = ctx.location(pattern.id);
                            Diagnostic {
                                message: "pattern destructuring not yet supported in IR lowering"
                                    .to_string(),
                                span: location.span,
                                severity: DiagnosticSeverity::Warning,
                                phase: CompilationPhase::Lowering,
                            }
                            .accumulate(ctx.db);
                        }
                    }
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
            } else if *name == "Rune" {
                // Rune is a Unicode code point, represented as i32
                core::I32::new(ctx.db).as_type()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{
        BinOpKind, Decl, Expr, ExprKind, FloatBits, FuncDecl, LocalId, Module, NodeId, ParamDecl,
        Pattern, PatternKind, ResolvedRef, Stmt, Type as AstType, TypeKind, TypedRef,
    };
    use salsa_test_macros::salsa_test;
    use trunk_ir::DialectOp;
    use trunk_ir::dialect::func;

    fn fresh_node_id() -> NodeId {
        NodeId::from_raw(0)
    }

    /// Tracked wrapper for lower_module to establish Salsa context.
    #[salsa::tracked]
    fn test_lower<'db>(
        db: &'db dyn salsa::Database,
        path: PathId<'db>,
        span_map: SpanMap,
        module: Module<TypedRef<'db>>,
    ) -> core::Module<'db> {
        lower_module(db, path, span_map, module)
    }

    /// Get operations from a module's body.
    fn get_module_ops<'db>(
        db: &'db dyn salsa::Database,
        module: &core::Module<'db>,
    ) -> Vec<trunk_ir::Operation<'db>> {
        let body = module.body(db);
        body.blocks(db)
            .iter()
            .flat_map(|block| block.operations(db).iter().copied())
            .collect()
    }

    /// Get operations from a function's body.
    fn get_func_body_ops<'db>(
        db: &'db dyn salsa::Database,
        func: &func::Func<'db>,
    ) -> Vec<trunk_ir::Operation<'db>> {
        let body = func.body(db);
        body.blocks(db)
            .iter()
            .flat_map(|block| block.operations(db).iter().copied())
            .collect()
    }

    /// Create a typed reference for a local variable.
    fn local_ref<'db>(db: &'db dyn salsa::Database, id: LocalId, name: Symbol) -> TypedRef<'db> {
        TypedRef::new(
            ResolvedRef::local(id, name),
            AstType::new(db, TypeKind::Int),
        )
    }

    /// Create a simple int literal expression.
    fn int_lit_expr<'db>(n: i64) -> Expr<TypedRef<'db>> {
        Expr::new(fresh_node_id(), ExprKind::IntLit(n))
    }

    /// Create a simple nat literal expression.
    fn nat_lit_expr<'db>(n: u64) -> Expr<TypedRef<'db>> {
        Expr::new(fresh_node_id(), ExprKind::NatLit(n))
    }

    /// Create a float literal expression.
    fn float_lit_expr<'db>(f: f64) -> Expr<TypedRef<'db>> {
        Expr::new(fresh_node_id(), ExprKind::FloatLit(FloatBits::new(f)))
    }

    /// Create a bool literal expression.
    fn bool_lit_expr<'db>(b: bool) -> Expr<TypedRef<'db>> {
        Expr::new(fresh_node_id(), ExprKind::BoolLit(b))
    }

    /// Create a string literal expression.
    fn string_lit_expr<'db>(s: &str) -> Expr<TypedRef<'db>> {
        Expr::new(fresh_node_id(), ExprKind::StringLit(s.to_string()))
    }

    /// Create a nil expression.
    fn nil_expr<'db>() -> Expr<TypedRef<'db>> {
        Expr::new(fresh_node_id(), ExprKind::Nil)
    }

    /// Create a variable expression.
    fn var_expr<'db>(typed_ref: TypedRef<'db>) -> Expr<TypedRef<'db>> {
        Expr::new(fresh_node_id(), ExprKind::Var(typed_ref))
    }

    /// Create a binary operation expression.
    fn binop_expr<'db>(
        op: BinOpKind,
        lhs: Expr<TypedRef<'db>>,
        rhs: Expr<TypedRef<'db>>,
    ) -> Expr<TypedRef<'db>> {
        Expr::new(fresh_node_id(), ExprKind::BinOp { op, lhs, rhs })
    }

    /// Create a block expression.
    fn block_expr<'db>(
        stmts: Vec<Stmt<TypedRef<'db>>>,
        value: Expr<TypedRef<'db>>,
    ) -> Expr<TypedRef<'db>> {
        Expr::new(fresh_node_id(), ExprKind::Block { stmts, value })
    }

    /// Create a let statement with a simple bind pattern.
    fn let_stmt<'db>(
        local_id: LocalId,
        name: Symbol,
        value: Expr<TypedRef<'db>>,
    ) -> Stmt<TypedRef<'db>> {
        let pattern = Pattern::new(
            fresh_node_id(),
            PatternKind::Bind {
                name,
                local_id: Some(local_id),
            },
        );
        Stmt::Let {
            id: fresh_node_id(),
            pattern,
            ty: None,
            value,
        }
    }

    /// Create a simple function with no parameters.
    fn simple_func<'db>(name: Symbol, body: Expr<TypedRef<'db>>) -> FuncDecl<TypedRef<'db>> {
        FuncDecl {
            id: fresh_node_id(),
            is_pub: false,
            name,
            type_params: vec![],
            params: vec![],
            return_ty: None,
            effects: None,
            body,
        }
    }

    /// Create a simple module with declarations.
    fn simple_module<'db>(decls: Vec<Decl<TypedRef<'db>>>) -> Module<TypedRef<'db>> {
        Module::new(fresh_node_id(), Some(Symbol::new("test")), decls)
    }

    // ========================================================================
    // Literal Lowering Tests
    // ========================================================================

    #[salsa_test]
    fn test_lower_nat_literal(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let span_map = SpanMap::default();

        let module = simple_module(vec![Decl::Function(simple_func(
            Symbol::new("main"),
            nat_lit_expr(42),
        ))]);

        let ir_module = test_lower(db, path, span_map, module);

        // Verify module was created
        let ops = get_module_ops(db, &ir_module);
        assert!(!ops.is_empty(), "Module should have operations");

        // Find the func.func operation
        let func_op = ops.iter().find(|op| op.name(db) == "func").unwrap();
        let func_typed = func::Func::from_operation(db, *func_op).unwrap();

        // Check the body contains operations
        let body_ops = get_func_body_ops(db, &func_typed);
        assert!(body_ops.len() >= 2, "Body should have const and return ops");

        // First op should be arith.const
        let first_op = body_ops[0];
        assert_eq!(first_op.dialect(db), "arith");
        assert_eq!(first_op.name(db), "const");
    }

    #[salsa_test]
    fn test_lower_int_literal(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let span_map = SpanMap::default();

        let module = simple_module(vec![Decl::Function(simple_func(
            Symbol::new("main"),
            int_lit_expr(-123),
        ))]);

        let ir_module = test_lower(db, path, span_map, module);
        let ops = get_module_ops(db, &ir_module);
        let func_op = ops.iter().find(|op| op.name(db) == "func").unwrap();
        let func_typed = func::Func::from_operation(db, *func_op).unwrap();
        let body_ops = get_func_body_ops(db, &func_typed);

        // First op should be arith.const
        let first_op = body_ops[0];
        assert_eq!(first_op.dialect(db), "arith");
        assert_eq!(first_op.name(db), "const");
    }

    #[salsa_test]
    fn test_lower_float_literal(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let span_map = SpanMap::default();

        let module = simple_module(vec![Decl::Function(simple_func(
            Symbol::new("main"),
            float_lit_expr(3.5),
        ))]);

        let ir_module = test_lower(db, path, span_map, module);
        let ops = get_module_ops(db, &ir_module);
        let func_op = ops.iter().find(|op| op.name(db) == "func").unwrap();
        let func_typed = func::Func::from_operation(db, *func_op).unwrap();
        let body_ops = get_func_body_ops(db, &func_typed);

        // First op should be arith.const for float
        let first_op = body_ops[0];
        assert_eq!(first_op.dialect(db), "arith");
        assert_eq!(first_op.name(db), "const");
    }

    #[salsa_test]
    fn test_lower_bool_true(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let span_map = SpanMap::default();

        let module = simple_module(vec![Decl::Function(simple_func(
            Symbol::new("main"),
            bool_lit_expr(true),
        ))]);
        let ir_module = test_lower(db, path, span_map, module);
        let ops = get_module_ops(db, &ir_module);
        let func_op = ops.iter().find(|op| op.name(db) == "func").unwrap();
        let func_typed = func::Func::from_operation(db, *func_op).unwrap();
        let body_ops = get_func_body_ops(db, &func_typed);

        // First op should be arith.const for bool
        let first_op = body_ops[0];
        assert_eq!(first_op.dialect(db), "arith");
        assert_eq!(first_op.name(db), "const");
    }

    #[salsa_test]
    fn test_lower_bool_false(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let span_map = SpanMap::default();

        let module = simple_module(vec![Decl::Function(simple_func(
            Symbol::new("main"),
            bool_lit_expr(false),
        ))]);
        let ir_module = test_lower(db, path, span_map, module);
        let ops = get_module_ops(db, &ir_module);
        let func_op = ops.iter().find(|op| op.name(db) == "func").unwrap();
        let func_typed = func::Func::from_operation(db, *func_op).unwrap();
        let body_ops = get_func_body_ops(db, &func_typed);

        let first_op = body_ops[0];
        assert_eq!(first_op.dialect(db), "arith");
        assert_eq!(first_op.name(db), "const");
    }

    #[salsa_test]
    fn test_lower_string_literal(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let span_map = SpanMap::default();

        let module = simple_module(vec![Decl::Function(simple_func(
            Symbol::new("main"),
            string_lit_expr("hello"),
        ))]);

        let ir_module = test_lower(db, path, span_map, module);
        let ops = get_module_ops(db, &ir_module);
        let func_op = ops.iter().find(|op| op.name(db) == "func").unwrap();
        let func_typed = func::Func::from_operation(db, *func_op).unwrap();
        let body_ops = get_func_body_ops(db, &func_typed);

        // First op should be adt.string_const
        let first_op = body_ops[0];
        assert_eq!(first_op.dialect(db), "adt");
        assert_eq!(first_op.name(db), "string_const");
    }

    #[salsa_test]
    fn test_lower_nil(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let span_map = SpanMap::default();

        let module = simple_module(vec![Decl::Function(simple_func(
            Symbol::new("main"),
            nil_expr(),
        ))]);

        let ir_module = test_lower(db, path, span_map, module);
        let ops = get_module_ops(db, &ir_module);
        let func_op = ops.iter().find(|op| op.name(db) == "func").unwrap();
        let func_typed = func::Func::from_operation(db, *func_op).unwrap();
        let body_ops = get_func_body_ops(db, &func_typed);

        // Nil produces arith.const with Unit attribute
        let first_op = body_ops[0];
        assert_eq!(first_op.dialect(db), "arith");
        assert_eq!(first_op.name(db), "const");
    }

    // ========================================================================
    // Binary Operation Lowering Tests
    // ========================================================================

    #[salsa_test]
    fn test_lower_arithmetic_add(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let span_map = SpanMap::default();

        // Test addition: 10 + 20
        let add_expr = binop_expr(BinOpKind::Add, int_lit_expr(10), int_lit_expr(20));
        let module = simple_module(vec![Decl::Function(simple_func(
            Symbol::new("main"),
            add_expr,
        ))]);

        let ir_module = test_lower(db, path, span_map, module);
        let ops = get_module_ops(db, &ir_module);
        let func_op = ops.iter().find(|op| op.name(db) == "func").unwrap();
        let func_typed = func::Func::from_operation(db, *func_op).unwrap();
        let body_ops = get_func_body_ops(db, &func_typed);

        // Should have: const(10), const(20), add, return
        assert!(body_ops.len() >= 4);

        // Find the add operation
        let add_op = body_ops.iter().find(|op| op.name(db) == "add");
        assert!(add_op.is_some(), "Should have an add operation");
    }

    #[salsa_test]
    fn test_lower_comparison_eq(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let span_map = SpanMap::default();

        // Test equality: 5 == 5
        let eq_expr = binop_expr(BinOpKind::Eq, int_lit_expr(5), int_lit_expr(5));
        let module = simple_module(vec![Decl::Function(simple_func(
            Symbol::new("main"),
            eq_expr,
        ))]);

        let ir_module = test_lower(db, path, span_map, module);
        let ops = get_module_ops(db, &ir_module);
        let func_op = ops.iter().find(|op| op.name(db) == "func").unwrap();
        let func_typed = func::Func::from_operation(db, *func_op).unwrap();
        let body_ops = get_func_body_ops(db, &func_typed);

        // Find the cmp_eq operation
        let cmp_op = body_ops.iter().find(|op| op.name(db) == "cmp_eq");
        assert!(cmp_op.is_some(), "Should have a cmp_eq operation");
    }

    #[salsa_test]
    fn test_lower_logical_and(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let span_map = SpanMap::default();

        // Test logical and: true && false
        let and_expr = binop_expr(BinOpKind::And, bool_lit_expr(true), bool_lit_expr(false));
        let module = simple_module(vec![Decl::Function(simple_func(
            Symbol::new("main"),
            and_expr,
        ))]);

        let ir_module = test_lower(db, path, span_map, module);
        let ops = get_module_ops(db, &ir_module);
        let func_op = ops.iter().find(|op| op.name(db) == "func").unwrap();
        let func_typed = func::Func::from_operation(db, *func_op).unwrap();
        let body_ops = get_func_body_ops(db, &func_typed);

        // Find the and operation
        let and_op = body_ops.iter().find(|op| op.name(db) == "and");
        assert!(and_op.is_some(), "Should have an and operation");
    }

    #[salsa_test]
    fn test_lower_float_mul(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let span_map = SpanMap::default();

        // Test float multiplication: 2.5 * 3.0
        let mul_expr = binop_expr(BinOpKind::Mul, float_lit_expr(2.5), float_lit_expr(3.0));
        let module = simple_module(vec![Decl::Function(simple_func(
            Symbol::new("main"),
            mul_expr,
        ))]);

        let ir_module = test_lower(db, path, span_map, module);
        let ops = get_module_ops(db, &ir_module);
        let func_op = ops.iter().find(|op| op.name(db) == "func").unwrap();
        let func_typed = func::Func::from_operation(db, *func_op).unwrap();
        let body_ops = get_func_body_ops(db, &func_typed);

        // Should have: const(2.5), const(3.0), mul, return
        let mul_op = body_ops.iter().find(|op| op.name(db) == "mul");
        assert!(mul_op.is_some(), "Should have a mul operation");
    }

    // ========================================================================
    // Block and Let Binding Tests
    // ========================================================================

    #[salsa_test]
    fn test_lower_block_with_let(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let span_map = SpanMap::default();

        // { let x = 10; x + 5 }
        let x_name = Symbol::new("x");
        let local_id = LocalId::new(0);
        let let_x = let_stmt(local_id, x_name, int_lit_expr(10));
        let x_ref = local_ref(db, local_id, x_name);
        let x_plus_5 = binop_expr(BinOpKind::Add, var_expr(x_ref), int_lit_expr(5));
        let block = block_expr(vec![let_x], x_plus_5);

        let module = simple_module(vec![Decl::Function(simple_func(
            Symbol::new("main"),
            block,
        ))]);

        let ir_module = test_lower(db, path, span_map, module);
        let ops = get_module_ops(db, &ir_module);
        let func_op = ops.iter().find(|op| op.name(db) == "func").unwrap();
        let func_typed = func::Func::from_operation(db, *func_op).unwrap();
        let body_ops = get_func_body_ops(db, &func_typed);

        // Should have: const(10), const(5), add, return
        // The variable lookup uses the value from let binding
        assert!(body_ops.len() >= 3);
        let add_op = body_ops.iter().find(|op| op.name(db) == "add");
        assert!(add_op.is_some(), "Should have an add operation");
    }

    #[salsa_test]
    fn test_lower_nested_block(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let span_map = SpanMap::default();

        // { let x = 1; { let y = 2; x + y } }
        let x_name = Symbol::new("x");
        let y_name = Symbol::new("y");
        let x_id = LocalId::new(0);
        let y_id = LocalId::new(1);

        let let_x = let_stmt(x_id, x_name, int_lit_expr(1));
        let let_y = let_stmt(y_id, y_name, int_lit_expr(2));

        let x_ref = local_ref(db, x_id, x_name);
        let y_ref = local_ref(db, y_id, y_name);
        let x_plus_y = binop_expr(BinOpKind::Add, var_expr(x_ref), var_expr(y_ref));

        let inner_block = block_expr(vec![let_y], x_plus_y);
        let outer_block = block_expr(vec![let_x], inner_block);

        let module = simple_module(vec![Decl::Function(simple_func(
            Symbol::new("main"),
            outer_block,
        ))]);

        let ir_module = test_lower(db, path, span_map, module);
        let ops = get_module_ops(db, &ir_module);
        let func_op = ops.iter().find(|op| op.name(db) == "func").unwrap();
        let func_typed = func::Func::from_operation(db, *func_op).unwrap();
        let body_ops = get_func_body_ops(db, &func_typed);

        // Should have add operation with correct operands
        let add_op = body_ops.iter().find(|op| op.name(db) == "add");
        assert!(add_op.is_some(), "Should have an add operation");
    }

    // ========================================================================
    // Function Lowering Tests
    // ========================================================================

    #[salsa_test]
    fn test_lower_function_with_params(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let span_map = SpanMap::default();

        // fn add(a, b) { a + b }
        // Note: without type annotations, params default to int
        let a_name = Symbol::new("a");
        let b_name = Symbol::new("b");
        let a_id = LocalId::new(0);
        let b_id = LocalId::new(1);
        let a_ref = local_ref(db, a_id, a_name);
        let b_ref = local_ref(db, b_id, b_name);
        let body = binop_expr(BinOpKind::Add, var_expr(a_ref), var_expr(b_ref));

        let func = FuncDecl {
            id: fresh_node_id(),
            is_pub: false,
            name: Symbol::new("add"),
            type_params: vec![],
            params: vec![
                ParamDecl {
                    id: fresh_node_id(),
                    name: a_name,
                    ty: None,
                    local_id: Some(LocalId::new(0)),
                },
                ParamDecl {
                    id: fresh_node_id(),
                    name: b_name,
                    ty: None,
                    local_id: Some(LocalId::new(1)),
                },
            ],
            return_ty: None,
            effects: None,
            body,
        };

        let module = simple_module(vec![Decl::Function(func)]);

        let ir_module = test_lower(db, path, span_map, module);
        let ops = get_module_ops(db, &ir_module);
        let func_op = ops.iter().find(|op| op.name(db) == "func").unwrap();
        let func_typed = func::Func::from_operation(db, *func_op).unwrap();

        // Check function exists and has correct name
        assert_eq!(func_typed.sym_name(db), Symbol::new("add"));
    }

    #[salsa_test]
    fn test_lower_module_with_multiple_functions(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let span_map = SpanMap::default();

        let func1 = simple_func(Symbol::new("foo"), int_lit_expr(1));
        let func2 = simple_func(Symbol::new("bar"), int_lit_expr(2));
        let module = simple_module(vec![Decl::Function(func1), Decl::Function(func2)]);

        let ir_module = test_lower(db, path, span_map, module);
        let ops = get_module_ops(db, &ir_module);

        // Should have 2 function operations
        let func_ops: Vec<_> = ops.iter().filter(|op| op.name(db) == "func").collect();
        assert_eq!(func_ops.len(), 2, "Should have 2 function definitions");
    }

    // ========================================================================
    // is_float_expr Tests
    // ========================================================================

    #[salsa_test]
    fn test_is_float_expr_literals(db: &salsa::DatabaseImpl) {
        // Float literal is float
        let float_expr = float_lit_expr(1.0);
        assert!(is_float_expr(db, &float_expr));

        // Int and Nat literals are not float
        let int_expr = int_lit_expr(1);
        assert!(!is_float_expr(db, &int_expr));

        let nat_expr = nat_lit_expr(1);
        assert!(!is_float_expr(db, &nat_expr));
    }

    #[salsa_test]
    fn test_is_float_expr_binop(db: &salsa::DatabaseImpl) {
        // Float + Int should be detected as float
        let mixed_expr = binop_expr(BinOpKind::Add, float_lit_expr(1.0), int_lit_expr(2));
        assert!(is_float_expr(db, &mixed_expr));

        // Int + Int should not be float
        let int_expr = binop_expr(BinOpKind::Add, int_lit_expr(1), int_lit_expr(2));
        assert!(!is_float_expr(db, &int_expr));
    }

    #[salsa_test]
    fn test_is_float_expr_block(db: &salsa::DatabaseImpl) {
        // Block returning float should be float
        let block = block_expr(vec![], float_lit_expr(1.0));
        assert!(is_float_expr(db, &block));

        // Block returning int should not be float
        let block2 = block_expr(vec![], int_lit_expr(1));
        assert!(!is_float_expr(db, &block2));
    }

    // ========================================================================
    // Type Annotation Conversion Tests
    // ========================================================================

    #[salsa_test]
    fn test_convert_annotation_to_ir_type_named(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let span_map = SpanMap::default();
        let ctx = super::super::context::IrLoweringCtx::new(db, path, span_map);

        // Test Int annotation
        let int_ann = TypeAnnotation {
            id: fresh_node_id(),
            kind: TypeAnnotationKind::Named(Symbol::new("Int")),
        };
        let int_ty = convert_annotation_to_ir_type(&ctx, Some(&int_ann));
        assert_eq!(int_ty, ctx.int_type());

        // Test Float annotation
        let float_ann = TypeAnnotation {
            id: fresh_node_id(),
            kind: TypeAnnotationKind::Named(Symbol::new("Float")),
        };
        let float_ty = convert_annotation_to_ir_type(&ctx, Some(&float_ann));
        assert_eq!(float_ty, core::F64::new(db).as_type());

        // Test Bool annotation
        let bool_ann = TypeAnnotation {
            id: fresh_node_id(),
            kind: TypeAnnotationKind::Named(Symbol::new("Bool")),
        };
        let bool_ty = convert_annotation_to_ir_type(&ctx, Some(&bool_ann));
        assert_eq!(bool_ty, ctx.bool_type());
    }

    #[salsa_test]
    fn test_convert_annotation_to_ir_type_default(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let span_map = SpanMap::default();
        let ctx = super::super::context::IrLoweringCtx::new(db, path, span_map);

        // No annotation should default to int
        let ty = convert_annotation_to_ir_type(&ctx, None);
        assert_eq!(ty, ctx.int_type());
    }

    // ========================================================================
    // NatLit Overflow Tests
    // ========================================================================

    #[salsa_test]
    fn test_nat_literal_within_i31_range(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let span_map = SpanMap::default();

        // i31 max value: 2^30 - 1 = 1,073,741,823
        let max_i31: u64 = (1 << 30) - 1;
        let module = simple_module(vec![Decl::Function(simple_func(
            Symbol::new("main"),
            nat_lit_expr(max_i31),
        ))]);

        let ir_module = test_lower(db, path, span_map, module);
        let ops = get_module_ops(db, &ir_module);
        let func_op = ops.iter().find(|op| op.name(db) == "func").unwrap();
        let func_typed = func::Func::from_operation(db, *func_op).unwrap();
        let body_ops = get_func_body_ops(db, &func_typed);

        // Should have const and return ops (no error)
        let const_op = body_ops.iter().find(|op| op.name(db) == "const");
        assert!(
            const_op.is_some(),
            "Should have a const operation for valid i31 value"
        );
    }

    #[salsa_test]
    fn test_nat_literal_exceeds_i31_range_returns_unit(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let span_map = SpanMap::default();

        // Value exceeding i31 max: 2^30 = 1,073,741,824
        let exceeds_i31: u64 = 1 << 30;
        let module = simple_module(vec![Decl::Function(simple_func(
            Symbol::new("main"),
            nat_lit_expr(exceeds_i31),
        ))]);

        let ir_module = test_lower(db, path, span_map, module);
        let ops = get_module_ops(db, &ir_module);
        let func_op = ops.iter().find(|op| op.name(db) == "func").unwrap();
        let func_typed = func::Func::from_operation(db, *func_op).unwrap();
        let body_ops = get_func_body_ops(db, &func_typed);

        // When NatLit exceeds i31 range, lower_expr returns None,
        // so the function body should only have unit const + return (no i64 const).
        // The function body will have 2 ops: arith.const (unit) and func.return.
        assert_eq!(
            body_ops.len(),
            2,
            "Overflow NatLit should result in unit return only"
        );
    }
}
