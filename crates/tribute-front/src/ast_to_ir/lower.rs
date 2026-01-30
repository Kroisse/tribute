//! Core lowering logic.
//!
//! Transforms AST declarations and expressions to TrunkIR operations.

use std::collections::{HashMap, HashSet};

use salsa::Accumulator;
use tribute_core::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use trunk_ir::dialect::{adt, arith, core, func, scf};
use trunk_ir::{
    Attribute, BlockBuilder, DialectOp, DialectType, Location, PathId, Region, Symbol, idvec,
};

use tribute_ir::ModulePathExt;
use tribute_ir::dialect::{closure, tribute, tribute_rt};

use super::context::CaptureInfo;

use crate::ast::{
    Arm, CtorId, Decl, EnumDecl, Expr, ExprKind, ExternFuncDecl, FuncDecl, LiteralPattern, LocalId,
    Module, Param, Pattern, PatternKind, ResolvedRef, SpanMap, Stmt, StructDecl, TypeAnnotation,
    TypeAnnotationKind, TypeKind, TypeScheme, TypedRef,
};

use super::context::IrLoweringCtx;

// =============================================================================
// IrBuilder
// =============================================================================

/// Builder for emitting TrunkIR operations within a block.
///
/// Combines the lowering context and block builder to provide
/// a unified API for expression lowering.
struct IrBuilder<'a, 'db> {
    ctx: &'a mut IrLoweringCtx<'db>,
    block: &'a mut trunk_ir::BlockBuilder<'db>,
}

impl<'a, 'db> IrBuilder<'a, 'db> {
    fn new(ctx: &'a mut IrLoweringCtx<'db>, block: &'a mut trunk_ir::BlockBuilder<'db>) -> Self {
        Self { ctx, block }
    }

    fn db(&self) -> &'db dyn salsa::Database {
        self.ctx.db
    }

    fn location(&self, id: crate::ast::NodeId) -> Location<'db> {
        self.ctx.location(id)
    }

    /// Emit a nil value (Tribute's unit type).
    fn emit_nil(&mut self, location: Location<'db>) -> trunk_ir::Value<'db> {
        let ty = self.ctx.nil_type();
        self.block
            .op(arith::r#const(self.db(), location, ty, Attribute::Unit))
            .result(self.db())
    }

    /// Emit diagnostic for unimplemented expression and return nil placeholder.
    fn emit_unsupported(
        &mut self,
        location: Location<'db>,
        feature: &str,
    ) -> Option<trunk_ir::Value<'db>> {
        Diagnostic {
            message: format!("{feature} not yet supported in IR lowering"),
            span: location.span,
            severity: DiagnosticSeverity::Warning,
            phase: CompilationPhase::Lowering,
        }
        .accumulate(self.db());
        Some(self.emit_nil(location))
    }

    /// Get the result type from a function type, or use the type directly.
    fn call_result_type(&self, ty: &crate::ast::Type<'db>) -> trunk_ir::Type<'db> {
        match ty.kind(self.db()) {
            TypeKind::Func { result, .. } => self.ctx.convert_type(*result),
            _ => self.ctx.convert_type(*ty),
        }
    }

    /// Lower arguments and propagate errors properly.
    ///
    /// Unlike filter_map which silently drops failed arguments (changing arity),
    /// this returns None if any argument fails to lower.
    fn collect_args(
        &mut self,
        args: impl IntoIterator<Item = Expr<TypedRef<'db>>>,
    ) -> Option<Vec<trunk_ir::Value<'db>>> {
        args.into_iter().map(|a| lower_expr(self, a)).collect()
    }
}

/// Lower a module to TrunkIR.
pub fn lower_module<'db>(
    db: &'db dyn salsa::Database,
    path: PathId<'db>,
    span_map: SpanMap,
    module: Module<TypedRef<'db>>,
    function_types: HashMap<Symbol, TypeScheme<'db>>,
) -> core::Module<'db> {
    // Use module's NodeId to get location
    let module_location = span_map.get_or_default(module.id);
    let location = Location::new(path, module_location);
    let module_name = module.name.unwrap_or_else(|| Symbol::new("main"));
    let module_path = smallvec::smallvec![module_name];

    // Create context outside the build closure so we can access lifted_functions after
    let mut ctx = IrLoweringCtx::new(db, path, span_map.clone(), function_types, module_path);

    // Build the module body
    let mut top = BlockBuilder::new(db, location);
    for decl in module.decls {
        lower_decl(&mut ctx, &mut top, decl);
    }

    // Prepend lifted functions to the module
    let lifted_functions = ctx.take_lifted_functions();
    let mut all_ops = lifted_functions;
    all_ops.extend(top.build().operations(db).iter().copied());

    // Build the final module with all operations
    let block = trunk_ir::Block::new(
        db,
        trunk_ir::BlockId::fresh(),
        location,
        trunk_ir::IdVec::new(),
        all_ops.into_iter().collect(),
    );
    let region = Region::new(db, location, idvec![block]);
    core::Module::create(db, location, module_name, region)
}

/// Lower a declaration to TrunkIR.
fn lower_decl<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    top: &mut trunk_ir::BlockBuilder<'db>,
    decl: Decl<TypedRef<'db>>,
) {
    match decl {
        Decl::Function(func) => lower_function(ctx, top, func),
        Decl::ExternFunction(func) => lower_extern_function(ctx, top, func),
        Decl::Struct(s) => lower_struct_decl(ctx, top, s),
        Decl::Enum(e) => lower_enum_decl(ctx, top, e),
        Decl::Ability(_) => {
            // TODO: Lower ability declarations
        }
        Decl::Use(_) => {
            // Use declarations don't generate IR
        }
        Decl::Module(m) => {
            // Inline module: recursively lower nested declarations
            if let Some(body) = m.body {
                ctx.enter_module(m.name);
                for inner_decl in body {
                    lower_decl(ctx, top, inner_decl);
                }
                ctx.exit_module();
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

    // Use TypeScheme from type checking if available, otherwise fall back to annotations
    let (param_ir_types, return_ty) =
        if let Some(scheme) = ctx.lookup_function_type(func_name).cloned() {
            let body = scheme.body(ctx.db);
            match body.kind(ctx.db) {
                TypeKind::Func { params, result, .. } => {
                    let p: Vec<_> = params.iter().map(|t| ctx.convert_type(*t)).collect();
                    let r = ctx.convert_type(*result);
                    (p, r)
                }
                _ => fallback_from_annotations(ctx, &func),
            }
        } else {
            fallback_from_annotations(ctx, &func)
        };

    let params: Vec<(trunk_ir::Type<'db>, Option<Symbol>)> = param_ir_types
        .into_iter()
        .zip(func.params.iter())
        .map(|(ty, p)| (ty, Some(p.name)))
        .collect();

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
                    ctx.bind(local_id, param.name, arg_values[i]);
                }
            }

            // Lower function body
            let mut builder = IrBuilder::new(ctx, body);
            if let Some(result) = lower_expr(&mut builder, func.body) {
                builder
                    .block
                    .op(func::Return::value(builder.db(), location, result));
            } else {
                // Return unit
                let nil = builder.emit_nil(location);
                builder
                    .block
                    .op(func::Return::value(builder.db(), location, nil));
            }

            ctx.exit_scope();
        },
    );

    top.op(func_op);
}

/// Lower an extern function declaration.
///
/// Extern functions have no body; we emit `func.func` with an empty body
/// containing `func.unreachable`.
fn lower_extern_function<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    top: &mut trunk_ir::BlockBuilder<'db>,
    func: ExternFuncDecl,
) {
    let location = ctx.location(func.id);
    let func_name = func.name;

    // Derive types from the TypeScheme registered during type checking.
    // Extern functions always have a TypeScheme from collect_extern_function_signature.
    let (param_ir_types, return_ty) = {
        let scheme = ctx
            .lookup_function_type(func_name)
            .cloned()
            .expect("extern function should have TypeScheme from type checking");
        let body = scheme.body(ctx.db);
        match body.kind(ctx.db) {
            TypeKind::Func { params, result, .. } => {
                let p: Vec<_> = params.iter().map(|t| ctx.convert_type(*t)).collect();
                let r = ctx.convert_type(*result);
                (p, r)
            }
            other => {
                unreachable!("extern function `{func_name}` has non-function TypeScheme: {other:?}")
            }
        }
    };

    let params: Vec<(trunk_ir::Type<'db>, Option<Symbol>)> = param_ir_types
        .into_iter()
        .zip(func.params.iter())
        .map(|(ty, p)| (ty, Some(p.name)))
        .collect();

    let func_op = func::Func::build_with_named_params(
        ctx.db,
        location,
        func_name,
        None,
        params,
        return_ty,
        None,
        |body, _arg_values| {
            // Extern functions have no body — emit unreachable
            body.op(func::unreachable(ctx.db, location));
        },
    );

    top.op(func_op);
}

/// Lower an expression to TrunkIR.
fn lower_expr<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    expr: Expr<TypedRef<'db>>,
) -> Option<trunk_ir::Value<'db>> {
    let location = builder.location(expr.id);

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
                .accumulate(builder.db());
                None
            } else {
                let op = builder
                    .block
                    .op(arith::Const::i64(builder.db(), location, n as i64));
                Some(op.result(builder.db()))
            }
        }

        ExprKind::IntLit(n) => {
            // i31ref range check for WASM target compatibility.
            // i31 range: -2^30 to 2^30-1
            const I31_MIN: i64 = -(1 << 30); // -1,073,741,824
            const I31_MAX: i64 = (1 << 30) - 1; // 1,073,741,823

            if !(I31_MIN..=I31_MAX).contains(&n) {
                Diagnostic {
                    message: format!(
                        "integer literal {} exceeds i31 range ({} to {})",
                        n, I31_MIN, I31_MAX
                    ),
                    span: location.span,
                    severity: DiagnosticSeverity::Error,
                    phase: CompilationPhase::Lowering,
                }
                .accumulate(builder.db());
                None
            } else {
                let op = builder
                    .block
                    .op(arith::Const::i64(builder.db(), location, n));
                Some(op.result(builder.db()))
            }
        }

        ExprKind::RuneLit(c) => {
            // Rune is lowered as i32 (Unicode code point, matching core::I32).
            // Unicode code points max out at 0x10FFFF (1,114,111), which is within
            // i31 range (max 1,073,741,823), so no range check is needed.
            let op = builder
                .block
                .op(arith::Const::i32(builder.db(), location, c as i32));
            Some(op.result(builder.db()))
        }

        ExprKind::FloatLit(f) => {
            let op = builder
                .block
                .op(arith::Const::f64(builder.db(), location, f.value()));
            Some(op.result(builder.db()))
        }

        ExprKind::BoolLit(b) => {
            let ty = builder.ctx.bool_type();
            let op = builder
                .block
                .op(arith::r#const(builder.db(), location, ty, b.into()));
            Some(op.result(builder.db()))
        }

        ExprKind::StringLit(ref s) => {
            let ty = core::String::new(builder.db()).as_type();
            let op = builder
                .block
                .op(adt::string_const(builder.db(), location, ty, s.clone()));
            Some(op.result(builder.db()))
        }

        ExprKind::BytesLit(ref _bytes) => builder.emit_unsupported(location, "bytes literal"),

        ExprKind::Nil => Some(builder.emit_nil(location)),

        ExprKind::Var(ref typed_ref) => match &typed_ref.resolved {
            ResolvedRef::Local { id, .. } => builder.ctx.lookup(*id),
            ResolvedRef::Function { id } => {
                let func_name = Symbol::from_dynamic(&id.qualified_name(builder.db()));
                let func_ty = builder.ctx.convert_type(typed_ref.ty);
                let op =
                    builder
                        .block
                        .op(func::constant(builder.db(), location, func_ty, func_name));
                Some(op.result(builder.db()))
            }
            ResolvedRef::Constructor { variant, .. } => {
                let func_ty = builder.ctx.convert_type(typed_ref.ty);
                let op =
                    builder
                        .block
                        .op(func::constant(builder.db(), location, func_ty, *variant));
                Some(op.result(builder.db()))
            }
            ResolvedRef::Builtin(_) | ResolvedRef::Module { .. } => None,
        },

        ExprKind::BinOp { op, lhs, rhs } => {
            // TODO: Short-circuit semantics for &&/|| are unimplemented.
            // Currently both operands are evaluated unconditionally via lower_expr,
            // which breaks short-circuit behavior. To fix: emit conditional branches
            // based on lhs value for And/Or, skipping rhs evaluation when appropriate.
            // See: ExprKind::BinOp, lower_expr, lower_binop

            // Determine operand type for selecting int vs float operations.
            // Check both operands: mixed int+float or float+int should use float operations.
            let is_float = is_float_expr(builder.db(), &lhs) || is_float_expr(builder.db(), &rhs);
            let lhs_val = lower_expr(builder, lhs)?;
            let rhs_val = lower_expr(builder, rhs)?;
            lower_binop(
                builder.ctx,
                builder.block,
                op,
                lhs_val,
                rhs_val,
                is_float,
                location,
            )
        }

        ExprKind::Block { stmts, value } => lower_block(builder, stmts, value),

        ExprKind::Call { callee, args } => {
            // Lower arguments first, propagating errors properly
            let arg_values = builder.collect_args(args)?;

            match *callee.kind {
                ExprKind::Var(ref typed_ref) => match &typed_ref.resolved {
                    ResolvedRef::Function { id } => {
                        let callee_name = Symbol::from_dynamic(&id.qualified_name(builder.db()));
                        let result_ty = builder.call_result_type(&typed_ref.ty);
                        let op = builder.block.op(func::call(
                            builder.db(),
                            location,
                            arg_values,
                            result_ty,
                            callee_name,
                        ));
                        Some(op.result(builder.db()))
                    }
                    ResolvedRef::Local { id, .. } => {
                        let callee_val = builder.ctx.lookup(*id)?;
                        let result_ty = builder.call_result_type(&typed_ref.ty);
                        let op = builder.block.op(func::call_indirect(
                            builder.db(),
                            location,
                            callee_val,
                            arg_values,
                            result_ty,
                        ));
                        Some(op.result(builder.db()))
                    }
                    ResolvedRef::Constructor { variant, .. } => {
                        let result_ty = builder.call_result_type(&typed_ref.ty);
                        let op = builder.block.op(adt::variant_new(
                            builder.db(),
                            location,
                            arg_values,
                            result_ty,
                            result_ty,
                            *variant,
                        ));
                        Some(op.result(builder.db()))
                    }
                    _ => builder.emit_unsupported(location, "builtin/module call"),
                },
                _ => {
                    // General expression callee → indirect call
                    let callee_val = lower_expr(builder, callee)?;
                    let result_ty = tribute_ir::dialect::tribute_rt::any_type(builder.db());
                    let op = builder.block.op(func::call_indirect(
                        builder.db(),
                        location,
                        callee_val,
                        arg_values,
                        result_ty,
                    ));
                    Some(op.result(builder.db()))
                }
            }
        }

        ExprKind::Cons { ctor, args } => {
            // Lower arguments, propagating errors properly
            let arg_values = builder.collect_args(args)?;

            match &ctor.resolved {
                ResolvedRef::Constructor { variant, .. } => {
                    let result_ty = builder.call_result_type(&ctor.ty);
                    let op = builder.block.op(adt::variant_new(
                        builder.db(),
                        location,
                        arg_values,
                        result_ty,
                        result_ty,
                        *variant,
                    ));
                    Some(op.result(builder.db()))
                }
                _ => builder.emit_unsupported(location, "non-constructor in Cons"),
            }
        }

        ExprKind::Tuple(elements) => {
            // Lower elements, propagating errors properly
            let values: Vec<_> = elements
                .iter()
                .map(|elem| lower_expr(builder, elem.clone()))
                .collect::<Option<Vec<_>>>()?;
            let result_ty = tribute_rt::any_type(builder.db());
            let op = builder
                .block
                .op(tribute::tuple(builder.db(), location, values, result_ty));
            Some(op.result(builder.db()))
        }

        ExprKind::Record {
            type_name,
            fields,
            spread,
        } => {
            let db = builder.db();
            let result_ty = tribute_rt::any_type(db);

            // Spread syntax is not yet supported
            if spread.is_some() {
                return builder.emit_unsupported(location, "record spread syntax");
            }

            // Extract struct type name and CtorId
            let struct_name = extract_type_name(db, &type_name.resolved);
            let ctor_id = extract_ctor_id(&type_name.resolved);
            let struct_ty = adt::typeref(db, struct_name);

            // Get field order from struct definition
            let Some(field_order) = builder.ctx.get_struct_field_order(ctor_id) else {
                // Struct not found - emit diagnostic and return nil
                Diagnostic {
                    message: format!("unknown struct: {}", struct_name),
                    span: location.span,
                    severity: DiagnosticSeverity::Error,
                    phase: CompilationPhase::Lowering,
                }
                .accumulate(db);
                return Some(builder.emit_nil(location));
            };
            let field_order = field_order.clone();

            // Build a set of valid field names for validation
            let valid_fields: HashSet<Symbol> = field_order.iter().copied().collect();

            // Lower field values into a map for quick lookup
            // Also check for duplicate and unknown fields
            let mut field_map: HashMap<Symbol, trunk_ir::Value<'db>> = HashMap::new();
            for (name, expr) in fields {
                // Check for unknown field
                if !valid_fields.contains(&name) {
                    Diagnostic {
                        message: format!("unknown field `{}` for struct `{}`", name, struct_name),
                        span: location.span,
                        severity: DiagnosticSeverity::Error,
                        phase: CompilationPhase::Lowering,
                    }
                    .accumulate(db);
                    continue;
                }

                // Check for duplicate field
                if field_map.contains_key(&name) {
                    Diagnostic {
                        message: format!("duplicate field `{}`", name),
                        span: location.span,
                        severity: DiagnosticSeverity::Error,
                        phase: CompilationPhase::Lowering,
                    }
                    .accumulate(db);
                    continue;
                }

                let val = lower_expr(builder, expr.clone())?;
                field_map.insert(name, val);
            }

            // Collect field values in definition order
            let mut ordered_values: Vec<trunk_ir::Value<'db>> =
                Vec::with_capacity(field_order.len());
            for field_name in &field_order {
                if let Some(val) = field_map.get(field_name) {
                    ordered_values.push(*val);
                } else {
                    // Missing field - emit diagnostic and return nil
                    Diagnostic {
                        message: format!("missing field: {}", field_name),
                        span: location.span,
                        severity: DiagnosticSeverity::Error,
                        phase: CompilationPhase::Lowering,
                    }
                    .accumulate(db);
                    return Some(builder.emit_nil(location));
                }
            }

            // Generate adt.struct_new directly
            let op = builder.block.op(adt::struct_new(
                db,
                location,
                ordered_values,
                struct_ty,
                result_ty,
            ));
            Some(op.result(db))
        }

        // === Expressions not yet implemented ===
        ExprKind::MethodCall { .. } => {
            unreachable!("MethodCall should be desugared before IR lowering")
        }
        ExprKind::Case { scrutinee, arms } => {
            let scrutinee_val = lower_expr(builder, scrutinee)?;
            let result_ty = tribute_rt::any_type(builder.db());
            let location = builder.location(expr.id);
            lower_case_chain(
                builder.ctx,
                builder.block,
                location,
                scrutinee_val,
                result_ty,
                &arms,
            )
        }
        ExprKind::Lambda { params, body } => lower_lambda(builder, location, &params, &body),
        ExprKind::Handle { .. } => builder.emit_unsupported(location, "handle expression"),
        ExprKind::List(_) => builder.emit_unsupported(location, "list expression"),

        ExprKind::Error => {
            // Error expression from parsing - just return unit placeholder
            Some(builder.emit_nil(location))
        }
    }
}

// =============================================================================
// Capture Analysis
// =============================================================================

/// Collect all LocalIds referenced in an expression (free variables).
fn collect_free_vars<'db>(expr: &Expr<TypedRef<'db>>, free_vars: &mut HashSet<LocalId>) {
    match &*expr.kind {
        ExprKind::Var(typed_ref) => {
            if let ResolvedRef::Local { id, .. } = &typed_ref.resolved {
                free_vars.insert(*id);
            }
        }
        ExprKind::IntLit(_)
        | ExprKind::NatLit(_)
        | ExprKind::FloatLit(_)
        | ExprKind::BoolLit(_)
        | ExprKind::StringLit(_)
        | ExprKind::BytesLit(_)
        | ExprKind::RuneLit(_)
        | ExprKind::Nil
        | ExprKind::Error => {}
        ExprKind::BinOp { lhs, rhs, .. } => {
            collect_free_vars(lhs, free_vars);
            collect_free_vars(rhs, free_vars);
        }
        ExprKind::Call { callee, args } => {
            collect_free_vars(callee, free_vars);
            for arg in args {
                collect_free_vars(arg, free_vars);
            }
        }
        ExprKind::Cons { args, .. } => {
            for arg in args {
                collect_free_vars(arg, free_vars);
            }
        }
        ExprKind::MethodCall { receiver, args, .. } => {
            collect_free_vars(receiver, free_vars);
            for arg in args {
                collect_free_vars(arg, free_vars);
            }
        }
        ExprKind::Block { stmts, value } => {
            for stmt in stmts {
                match stmt {
                    Stmt::Let { value, .. } => collect_free_vars(value, free_vars),
                    Stmt::Expr { expr, .. } => collect_free_vars(expr, free_vars),
                }
            }
            collect_free_vars(value, free_vars);
        }
        ExprKind::Record { fields, .. } => {
            for (_, value) in fields {
                collect_free_vars(value, free_vars);
            }
        }
        ExprKind::Tuple(elements) => {
            for elem in elements {
                collect_free_vars(elem, free_vars);
            }
        }
        ExprKind::Case { scrutinee, arms } => {
            collect_free_vars(scrutinee, free_vars);
            for arm in arms {
                if let Some(guard) = &arm.guard {
                    collect_free_vars(guard, free_vars);
                }
                collect_free_vars(&arm.body, free_vars);
            }
        }
        ExprKind::Lambda { body, .. } => {
            // Note: nested lambda's body is traversed, but its params are bound
            // within that lambda, so they won't be in our outer scope anyway
            collect_free_vars(body, free_vars);
        }
        ExprKind::Handle { body, .. } => {
            collect_free_vars(body, free_vars);
            // TODO: traverse handler arms as well
        }
        ExprKind::List(elements) => {
            for elem in elements {
                collect_free_vars(elem, free_vars);
            }
        }
    }
}

/// Analyze captures for a lambda expression.
/// Returns the list of captured variables (those referenced but not lambda params).
fn analyze_captures<'db>(
    ctx: &IrLoweringCtx<'db>,
    params: &[Param],
    body: &Expr<TypedRef<'db>>,
) -> Vec<CaptureInfo<'db>> {
    // Collect all LocalIds referenced in the body
    let mut free_vars = HashSet::new();
    collect_free_vars(body, &mut free_vars);

    // Get param LocalIds to exclude
    let param_ids: HashSet<LocalId> = params.iter().filter_map(|p| p.local_id).collect();

    // Filter to only those that are:
    // 1. Not lambda parameters
    // 2. Actually bound in the current scope
    let mut captures = Vec::new();
    for (local_id, name, value) in ctx.all_bindings() {
        if free_vars.contains(&local_id) && !param_ids.contains(&local_id) {
            let ty = tribute_rt::any_type(ctx.db); // Type-erased for now
            captures.push(CaptureInfo {
                name,
                local_id,
                ty,
                value,
            });
        }
    }

    captures
}

// =============================================================================
// Lambda Lowering (Closure Conversion)
// =============================================================================

/// Lower a lambda expression to a closure.
///
/// This performs closure conversion:
/// 1. Analyze captures (free variables referenced in body)
/// 2. Create a lifted function at module level
/// 3. Create an env struct with captured values
/// 4. Return a closure.new with the env and function reference
fn lower_lambda<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location<'db>,
    params: &[Param],
    body: &Expr<TypedRef<'db>>,
) -> Option<trunk_ir::Value<'db>> {
    let db = builder.db();
    let any_ty = tribute_rt::any_type(db);

    // Step 1: Analyze captures
    let captures = analyze_captures(builder.ctx, params, body);

    // Step 2: Generate unique name for the lifted function
    let lifted_name = builder.ctx.gen_lambda_name();

    // Step 3: Build the lifted function
    // Signature: (env, param1, param2, ...) -> result
    // All types are erased to any for simplicity
    let mut all_param_types: trunk_ir::IdVec<trunk_ir::Type<'db>> = trunk_ir::IdVec::new();

    // First param is env (or nil if no captures)
    let env_ty = if captures.is_empty() {
        builder.ctx.nil_type()
    } else {
        // Create env struct type
        let fields: Vec<(Symbol, trunk_ir::Type<'db>)> = captures
            .iter()
            .enumerate()
            .map(|(i, cap)| (Symbol::from_dynamic(&format!("_{}", i)), cap.ty))
            .collect();
        let env_name = lifted_name.join_path(Symbol::new("env"));
        adt::struct_type(db, env_name, fields)
    };
    all_param_types.push(env_ty);

    // Then the original lambda params
    for _ in params {
        all_param_types.push(any_ty);
    }

    // Build the lifted function body
    let lifted_func = func::Func::build_with_named_params(
        db,
        location,
        lifted_name,
        None, // visibility
        {
            let mut param_specs = Vec::new();
            param_specs.push((env_ty, Some(Symbol::new("__env"))));
            for param in params {
                param_specs.push((any_ty, Some(param.name)));
            }
            param_specs
        },
        any_ty, // return type
        None,   // effect type
        |body_block, arg_values| {
            // arg_values[0] is env, arg_values[1..] are params
            let env_value = arg_values[0];

            // Create a new context for the lambda body
            builder.ctx.enter_scope();

            // Bind lambda parameters (skip env at index 0)
            for (i, param) in params.iter().enumerate() {
                if let Some(local_id) = param.local_id {
                    builder.ctx.bind(local_id, param.name, arg_values[i + 1]);
                }
            }

            // Extract captured values from env and bind them
            for (i, cap) in captures.iter().enumerate() {
                let extracted = body_block
                    .op(adt::struct_get(
                        db, location, env_value, cap.ty, env_ty, i as u64,
                    ))
                    .result(db);
                builder.ctx.bind(cap.local_id, cap.name, extracted);
            }

            // Lower the lambda body
            let mut inner_builder = IrBuilder::new(builder.ctx, body_block);
            if let Some(result) = lower_expr(&mut inner_builder, body.clone()) {
                inner_builder
                    .block
                    .op(func::Return::value(db, location, result));
            } else {
                // Return nil on error
                let nil = inner_builder.emit_nil(location);
                inner_builder
                    .block
                    .op(func::Return::value(db, location, nil));
            }

            builder.ctx.exit_scope();
        },
    );

    // Add the lifted function to the module
    builder.ctx.add_lifted_function(lifted_func.as_operation());

    // Step 4: Create env struct with captured values
    let capture_values: Vec<trunk_ir::Value<'db>> = captures.iter().map(|c| c.value).collect();

    let env_value = if captures.is_empty() {
        // No captures - create nil value
        builder.emit_nil(location)
    } else {
        builder
            .block
            .op(adt::struct_new(
                db,
                location,
                capture_values,
                env_ty,
                env_ty,
            ))
            .result(db)
    };

    // Step 5: Create closure.new
    let func_type = core::Func::new(
        db,
        std::iter::repeat_n(any_ty, params.len()).collect(),
        any_ty,
    )
    .as_type();
    let closure_ty = closure::Closure::new(db, func_type);

    let closure_op = builder.block.op(closure::new(
        db,
        location,
        env_value,
        *closure_ty,
        lifted_name,
    ));
    Some(closure_op.result(db))
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
            let ty = ctx.nil_type();
            block
                .op(arith::r#const(ctx.db, location, ty, Attribute::Unit))
                .result(ctx.db)
        }
    };
    Some(result)
}

/// Lower a block of statements.
fn lower_block<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    stmts: Vec<Stmt<TypedRef<'db>>>,
    value: Expr<TypedRef<'db>>,
) -> Option<trunk_ir::Value<'db>> {
    builder.ctx.enter_scope();

    for stmt in stmts {
        match stmt {
            Stmt::Let {
                id: _,
                pattern,
                ty: _,
                value,
            } => {
                if let Some(val) = lower_expr(builder, value) {
                    match &*pattern.kind {
                        crate::ast::PatternKind::Bind {
                            name,
                            local_id: Some(local_id),
                        } => {
                            // Register the binding so Var expressions can find it
                            builder.ctx.bind(*local_id, *name, val);
                        }
                        crate::ast::PatternKind::Wildcard => {
                            // Wildcard binds nothing, value is computed for side effects
                        }
                        _ => {
                            // Pattern destructuring not yet supported in IR lowering
                            let location = builder.ctx.location(pattern.id);
                            Diagnostic {
                                message: "pattern destructuring not yet supported in IR lowering"
                                    .to_string(),
                                span: location.span,
                                severity: DiagnosticSeverity::Warning,
                                phase: CompilationPhase::Lowering,
                            }
                            .accumulate(builder.db());
                        }
                    }
                }
            }
            Stmt::Expr { id: _, expr } => {
                let _ = lower_expr(builder, expr);
            }
        }
    }

    let result = lower_expr(builder, value);
    builder.ctx.exit_scope();
    result
}

// =============================================================================
// Struct Declaration Lowering
// =============================================================================

/// Lower a struct declaration to TrunkIR.
///
/// Generates:
/// 1. A `tribute.struct_def` operation with field definitions
/// 2. An accessor module containing getter functions for each field
fn lower_struct_decl<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    top: &mut BlockBuilder<'db>,
    decl: StructDecl,
) {
    let db = ctx.db;
    let location = ctx.location(decl.id);
    let name = decl.name;
    let struct_ty = adt::typeref(db, name);

    // 0. Register struct field order for lowering Record expressions
    let ctor_id = CtorId::new(db, ctx.module_path().clone(), name);
    let field_names: Vec<Symbol> = decl
        .fields
        .iter()
        .map(|f| f.name.unwrap_or_else(|| Symbol::new("_")))
        .collect();
    ctx.register_struct_fields(ctor_id, field_names);

    // 1. Build fields region for struct_def
    let mut fields_block = BlockBuilder::new(db, location);
    for field in &decl.fields {
        let field_name = field.name.unwrap_or_else(|| Symbol::new("_"));
        let field_ty = convert_annotation_to_ir_type(ctx, Some(&field.ty));
        fields_block.op(tribute::field_def(db, location, field_name, field_ty));
    }
    let fields_region = Region::new(db, location, idvec![fields_block.build()]);

    // 2. Emit struct_def
    top.op(tribute::struct_def(
        db,
        location,
        struct_ty,
        name,
        fields_region,
    ));

    // 3. Generate accessor module with getter functions
    let fields: Vec<(Symbol, trunk_ir::Type<'db>)> = decl
        .fields
        .iter()
        .map(|f| {
            let field_name = f.name.unwrap_or_else(|| Symbol::new("_"));
            let field_ty = convert_annotation_to_ir_type(ctx, Some(&f.ty));
            (field_name, field_ty)
        })
        .collect();

    // Build qualified accessor names using module_path
    // This must match what TDNR generates via build_qualified_field_name
    let module_path = ctx.module_path();

    let accessors_module = core::Module::build(db, location, name, |module_builder| {
        for (idx, (field_name, field_type)) in fields.iter().enumerate() {
            // Generate getter: fn qualified_name(self: StructType) -> FieldType
            // Qualified name: module_path::struct_name::field_name
            let qualified_name = if module_path.is_empty() {
                Symbol::from_dynamic(&format!("{}::{}", name, field_name))
            } else {
                let module_path_str = module_path
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
                    .join("::");
                Symbol::from_dynamic(&format!("{}::{}::{}", module_path_str, name, field_name))
            };

            let getter = func::Func::build(
                db,
                location,
                qualified_name,
                idvec![struct_ty],
                *field_type,
                |entry| {
                    let self_value = entry.block_arg(db, 0);
                    let field_value = entry.op(adt::struct_get(
                        db,
                        location,
                        self_value,
                        *field_type,
                        struct_ty,
                        idx as u64,
                    ));
                    entry.op(func::Return::value(db, location, field_value.result(db)));
                },
            );
            module_builder.op(getter);
        }
    });

    top.op(accessors_module);
}

// =============================================================================
// Enum Declaration Lowering
// =============================================================================

/// Lower an enum declaration to TrunkIR.
fn lower_enum_decl<'db>(ctx: &mut IrLoweringCtx<'db>, top: &mut BlockBuilder<'db>, decl: EnumDecl) {
    let location = ctx.location(decl.id);
    let name = decl.name;

    // Build adt.enum type with variant information
    let enum_variants: Vec<(Symbol, Vec<trunk_ir::Type<'db>>)> = decl
        .variants
        .iter()
        .map(|v| {
            let field_types: Vec<_> = v
                .fields
                .iter()
                .map(|f| convert_annotation_to_ir_type(ctx, Some(&f.ty)))
                .collect();
            (v.name, field_types)
        })
        .collect();

    let result_ty = adt::enum_type(ctx.db, name, enum_variants);

    // Build variants region containing tribute.variant_def operations
    let mut variants_block = BlockBuilder::new(ctx.db, location);
    for variant in &decl.variants {
        // Build fields region for this variant
        let mut variant_fields_block = BlockBuilder::new(ctx.db, location);
        for field in &variant.fields {
            let field_name = field.name.unwrap_or_else(|| Symbol::new("_"));
            let field_type = convert_annotation_to_ir_type(ctx, Some(&field.ty));
            variant_fields_block.op(tribute::field_def(ctx.db, location, field_name, field_type));
        }
        let variant_fields_region =
            Region::new(ctx.db, location, idvec![variant_fields_block.build()]);

        variants_block.op(tribute::variant_def(
            ctx.db,
            location,
            variant.name,
            variant_fields_region,
        ));
    }
    let variants_region = Region::new(ctx.db, location, idvec![variants_block.build()]);

    top.op(tribute::enum_def(
        ctx.db,
        location,
        result_ty,
        name,
        variants_region,
    ));
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Extract the type name from a ResolvedRef.
///
/// Used for record construction to get the struct/enum name.
/// Record type_name must always resolve to a Constructor.
fn extract_type_name<'db>(db: &'db dyn salsa::Database, resolved: &ResolvedRef<'db>) -> Symbol {
    match resolved {
        ResolvedRef::Constructor { id, .. } => id.type_name(db),
        _ => unreachable!("Record type must be a constructor: {:?}", resolved),
    }
}

/// Extract the CtorId from a ResolvedRef.
///
/// Used for record construction to get the constructor ID.
/// Record type_name must always resolve to a Constructor.
fn extract_ctor_id<'db>(resolved: &ResolvedRef<'db>) -> CtorId<'db> {
    match resolved {
        ResolvedRef::Constructor { id, .. } => *id,
        _ => unreachable!("Record type must be a constructor: {:?}", resolved),
    }
}

// =============================================================================
// Case Expression Lowering
// =============================================================================

/// Check if a pattern is unconditional (always matches without runtime checks).
///
/// Returns true for `_` (wildcard) and `x` (bind) patterns, which match any value.
/// Returns false for patterns that require runtime checks (variant, literal, etc.).
fn is_unconditional_pattern(pattern: &Pattern<TypedRef<'_>>) -> bool {
    matches!(
        &*pattern.kind,
        PatternKind::Wildcard | PatternKind::Bind { .. }
    )
}

/// Lower a case expression as a chain of scf.if operations.
///
/// Pattern: `case scrutinee { p1 => e1, p2 => e2, ... }` becomes a nested
/// if-else chain testing each pattern in order.
fn lower_case_chain<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    block: &mut BlockBuilder<'db>,
    location: Location<'db>,
    scrutinee: trunk_ir::Value<'db>,
    result_ty: trunk_ir::Type<'db>,
    arms: &[Arm<TypedRef<'db>>],
) -> Option<trunk_ir::Value<'db>> {
    match arms {
        [] => {
            // No arms — exhaustiveness failure fallback, emit nil
            let ty = core::Nil::new(ctx.db).as_type();
            let op = block.op(arith::r#const(ctx.db, location, ty, Attribute::Unit));
            Some(op.result(ctx.db))
        }
        [last] if last.guard.is_none() && is_unconditional_pattern(&last.pattern) => {
            // Single unconditional arm (wildcard or bind, no guard)
            // — bind pattern and lower body directly (no scf.if needed)
            ctx.enter_scope();
            bind_pattern_fields(ctx, block, location, scrutinee, &last.pattern);
            let result = {
                let mut builder = IrBuilder::new(ctx, block);
                lower_expr(&mut builder, last.body.clone())
            };
            ctx.exit_scope();
            result
        }
        [first, rest @ ..] => {
            // Multi-arm: emit pattern check → build then/else regions → scf.if

            // 1. Emit pattern condition check
            let pattern_cond = {
                let mut builder = IrBuilder::new(ctx, block);
                emit_pattern_check(&mut builder, location, scrutinee, &first.pattern)
            }?;

            // 2. Build then region (handles guard internally if present)
            let then_region = if let Some(guard_expr) = &first.guard {
                build_guarded_arm_region(
                    ctx, location, scrutinee, first, guard_expr, result_ty, rest,
                )
            } else {
                build_arm_region(ctx, location, scrutinee, first, result_ty)
            };

            // 3. Build else region (recursive)
            let else_region = build_else_chain_region(ctx, location, scrutinee, result_ty, rest);

            // 4. Emit scf.if in current block
            let if_op = block.op(scf::r#if(
                ctx.db,
                location,
                pattern_cond,
                result_ty,
                then_region,
                else_region,
            ));
            Some(if_op.as_operation().result(ctx.db, 0))
        }
    }
}

/// Emit a condition check for a pattern match.
///
/// Returns a boolean value indicating whether the pattern matches the scrutinee.
fn emit_pattern_check<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location<'db>,
    scrutinee: trunk_ir::Value<'db>,
    pattern: &Pattern<TypedRef<'db>>,
) -> Option<trunk_ir::Value<'db>> {
    let bool_ty = builder.ctx.bool_type();

    match &*pattern.kind {
        PatternKind::Wildcard | PatternKind::Bind { .. } | PatternKind::Error => {
            // Always matches
            Some(
                builder
                    .block
                    .op(arith::r#const(builder.db(), location, bool_ty, true.into()))
                    .result(builder.db()),
            )
        }
        PatternKind::Variant { ctor, .. } => {
            // Test if scrutinee is of the specific variant
            let (variant_name, enum_ty) = match &ctor.resolved {
                ResolvedRef::Constructor { variant, .. } => {
                    (*variant, builder.ctx.convert_type(ctor.ty))
                }
                _ => {
                    unreachable!("non-constructor in variant pattern: {:?}", ctor.resolved)
                }
            };
            let op = builder.block.op(adt::variant_is(
                builder.db(),
                location,
                scrutinee,
                bool_ty,
                enum_ty,
                variant_name,
            ));
            Some(op.as_operation().result(builder.db(), 0))
        }
        PatternKind::Literal(lit) => emit_literal_check(builder, location, scrutinee, lit),
        PatternKind::Tuple(elements) => {
            // Tuple patterns always match structurally (untagged)
            // We need to recursively check all element patterns
            let mut conditions = Vec::new();
            let any_ty = tribute_rt::any_type(builder.db());

            for (i, elem_pat) in elements.iter().enumerate() {
                // Extract the element from the tuple
                let elem_val = builder
                    .block
                    .op(adt::struct_get(
                        builder.db(),
                        location,
                        scrutinee,
                        any_ty,
                        any_ty,
                        i as u64,
                    ))
                    .result(builder.db());

                // Recursively check the element pattern
                if let Some(cond) = emit_pattern_check(builder, location, elem_val, elem_pat) {
                    conditions.push(cond);
                }
            }

            // Combine all conditions with AND
            if conditions.is_empty() {
                // Empty tuple or all wildcards - always matches
                Some(
                    builder
                        .block
                        .op(arith::r#const(builder.db(), location, bool_ty, true.into()))
                        .result(builder.db()),
                )
            } else {
                let mut result = conditions[0];
                for cond in conditions.into_iter().skip(1) {
                    result = builder
                        .block
                        .op(arith::and(builder.db(), location, result, cond, bool_ty))
                        .result(builder.db());
                }
                Some(result)
            }
        }
        _ => {
            unreachable!("unsupported pattern in IR lowering: {:?}", pattern.kind)
        }
    }
}

/// Emit a literal equality check.
fn emit_literal_check<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location<'db>,
    scrutinee: trunk_ir::Value<'db>,
    lit: &LiteralPattern,
) -> Option<trunk_ir::Value<'db>> {
    let bool_ty = builder.ctx.bool_type();
    match lit {
        LiteralPattern::Int(n) => {
            let const_val = builder
                .block
                .op(arith::Const::i64(builder.db(), location, *n))
                .result(builder.db());
            Some(
                builder
                    .block
                    .op(arith::cmp_eq(
                        builder.db(),
                        location,
                        scrutinee,
                        const_val,
                        bool_ty,
                    ))
                    .result(builder.db()),
            )
        }
        LiteralPattern::Bool(b) => {
            let const_val = builder
                .block
                .op(arith::r#const(builder.db(), location, bool_ty, (*b).into()))
                .result(builder.db());
            Some(
                builder
                    .block
                    .op(arith::cmp_eq(
                        builder.db(),
                        location,
                        scrutinee,
                        const_val,
                        bool_ty,
                    ))
                    .result(builder.db()),
            )
        }
        _ => {
            unreachable!("unsupported literal pattern in IR lowering: {:?}", lit)
        }
    }
}

/// Build a region for a case arm with a guard condition.
///
/// This creates a nested `scf.if` structure:
/// - Outer region: pattern bindings + guard evaluation + inner if
/// - Inner if: guard succeeded → arm body, guard failed → fall through to rest
///
/// This ensures `bind_pattern_fields` (which may emit `variant_cast`) is only
/// executed inside the pattern-matched region, not unconditionally.
fn build_guarded_arm_region<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    location: Location<'db>,
    scrutinee: trunk_ir::Value<'db>,
    arm: &Arm<TypedRef<'db>>,
    guard_expr: &Expr<TypedRef<'db>>,
    result_ty: trunk_ir::Type<'db>,
    rest: &[Arm<TypedRef<'db>>],
) -> Region<'db> {
    let mut block = BlockBuilder::new(ctx.db, location);

    // 1. Bind pattern fields (now safe — we're inside pattern-matched region)
    ctx.enter_scope();
    bind_pattern_fields(ctx, &mut block, location, scrutinee, &arm.pattern);

    // 2. Evaluate guard condition
    let guard_cond = {
        let mut builder = IrBuilder::new(ctx, &mut block);
        lower_expr(&mut builder, guard_expr.clone())
    };

    let guard_cond = guard_cond.unwrap_or_else(|| {
        // Guard lowering failed — emit false to skip this arm
        let bool_ty = ctx.bool_type();
        block
            .op(arith::r#const(
                ctx.db,
                location,
                bool_ty,
                Attribute::Bool(false),
            ))
            .result(ctx.db)
    });

    // 3. Build inner then region (arm body)
    let inner_then_region = {
        let mut inner_block = BlockBuilder::new(ctx.db, location);
        let result = {
            let mut builder = IrBuilder::new(ctx, &mut inner_block);
            lower_expr(&mut builder, arm.body.clone())
        };
        let yield_val = result.unwrap_or_else(|| {
            let ty = core::Nil::new(ctx.db).as_type();
            inner_block
                .op(arith::r#const(ctx.db, location, ty, Attribute::Unit))
                .result(ctx.db)
        });
        inner_block.op(scf::r#yield(ctx.db, location, vec![yield_val]));
        Region::new(ctx.db, location, idvec![inner_block.build()])
    };

    ctx.exit_scope();

    // 4. Build inner else region (fall through to remaining arms)
    let inner_else_region = build_else_chain_region(ctx, location, scrutinee, result_ty, rest);

    // 5. Emit inner scf.if for guard
    let inner_if_op = block.op(scf::r#if(
        ctx.db,
        location,
        guard_cond,
        result_ty,
        inner_then_region,
        inner_else_region,
    ));
    let inner_result = inner_if_op.as_operation().result(ctx.db, 0);

    // 6. Yield inner result
    block.op(scf::r#yield(ctx.db, location, vec![inner_result]));
    Region::new(ctx.db, location, idvec![block.build()])
}

/// Build a region for a single case arm body.
fn build_arm_region<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    location: Location<'db>,
    scrutinee: trunk_ir::Value<'db>,
    arm: &Arm<TypedRef<'db>>,
    _result_ty: trunk_ir::Type<'db>,
) -> Region<'db> {
    let mut block = BlockBuilder::new(ctx.db, location);

    ctx.enter_scope();
    bind_pattern_fields(ctx, &mut block, location, scrutinee, &arm.pattern);

    let result = {
        let mut builder = IrBuilder::new(ctx, &mut block);
        lower_expr(&mut builder, arm.body.clone())
    };

    ctx.exit_scope();

    let yield_val = result.unwrap_or_else(|| {
        let ty = core::Nil::new(ctx.db).as_type();
        block
            .op(arith::r#const(ctx.db, location, ty, Attribute::Unit))
            .result(ctx.db)
    });
    block.op(scf::r#yield(ctx.db, location, vec![yield_val]));
    Region::new(ctx.db, location, idvec![block.build()])
}

/// Build an else region containing a recursive case chain.
fn build_else_chain_region<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    location: Location<'db>,
    scrutinee: trunk_ir::Value<'db>,
    result_ty: trunk_ir::Type<'db>,
    arms: &[Arm<TypedRef<'db>>],
) -> Region<'db> {
    let mut block = BlockBuilder::new(ctx.db, location);
    let result = lower_case_chain(ctx, &mut block, location, scrutinee, result_ty, arms);

    let yield_val = result.unwrap_or_else(|| {
        let ty = core::Nil::new(ctx.db).as_type();
        block
            .op(arith::r#const(ctx.db, location, ty, Attribute::Unit))
            .result(ctx.db)
    });
    block.op(scf::r#yield(ctx.db, location, vec![yield_val]));
    Region::new(ctx.db, location, idvec![block.build()])
}

/// Bind pattern fields to SSA values in the current scope.
fn bind_pattern_fields<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    block: &mut BlockBuilder<'db>,
    location: Location<'db>,
    scrutinee: trunk_ir::Value<'db>,
    pattern: &Pattern<TypedRef<'db>>,
) {
    match &*pattern.kind {
        PatternKind::Bind {
            name,
            local_id: Some(id),
        } => {
            ctx.bind(*id, *name, scrutinee);
        }
        PatternKind::Variant { ctor, fields } => {
            let (variant_name, enum_ty) = match &ctor.resolved {
                ResolvedRef::Constructor { variant, .. } => (*variant, ctx.convert_type(ctor.ty)),
                _ => unreachable!("non-constructor in variant pattern: {:?}", ctor.resolved),
            };

            // Cast scrutinee to the specific variant type
            let cast_val = block
                .op(adt::variant_cast(
                    ctx.db,
                    location,
                    scrutinee,
                    enum_ty,
                    enum_ty,
                    variant_name,
                ))
                .as_operation()
                .result(ctx.db, 0);

            // Extract each field and recursively bind
            let any_ty = tribute_rt::any_type(ctx.db);
            for (i, field_pat) in fields.iter().enumerate() {
                let field_val = block
                    .op(adt::variant_get(
                        ctx.db, location, cast_val, any_ty, i as u64,
                    ))
                    .as_operation()
                    .result(ctx.db, 0);
                bind_pattern_fields(ctx, block, location, field_val, field_pat);
            }
        }
        PatternKind::Wildcard | PatternKind::Literal(_) | PatternKind::Error => {
            // No bindings needed
        }
        PatternKind::Bind { local_id: None, .. } => {
            // Bind pattern without local_id (e.g., from unresolved names) — no binding needed
        }
        PatternKind::Tuple(elements) => {
            // Extract each element from the tuple and recursively bind
            let any_ty = tribute_rt::any_type(ctx.db);
            for (i, elem_pat) in elements.iter().enumerate() {
                let elem_val = block
                    .op(adt::struct_get(
                        ctx.db, location, scrutinee, any_ty, any_ty, i as u64,
                    ))
                    .result(ctx.db);
                bind_pattern_fields(ctx, block, location, elem_val, elem_pat);
            }
        }
        _ => {
            unreachable!(
                "unsupported pattern in bind_pattern_fields: {:?}",
                pattern.kind
            )
        }
    }
}

/// Fallback: derive parameter and return types from annotations when TypeScheme is unavailable.
fn fallback_from_annotations<'db>(
    ctx: &IrLoweringCtx<'db>,
    func: &FuncDecl<TypedRef<'db>>,
) -> (Vec<trunk_ir::Type<'db>>, trunk_ir::Type<'db>) {
    let params = func
        .params
        .iter()
        .map(|p| convert_annotation_to_ir_type(ctx, p.ty.as_ref()))
        .collect();
    let ret = func
        .return_ty
        .as_ref()
        .map(|ann| convert_annotation_to_ir_type(ctx, Some(ann)))
        .unwrap_or_else(|| ctx.nil_type());
    (params, ret)
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
            } else if *name == "Nil" {
                ctx.nil_type()
            } else {
                // Unknown named type - use placeholder
                ctx.nil_type()
            }
        }
        TypeAnnotationKind::Path(_) => {
            // Qualified path - use placeholder for now
            ctx.nil_type()
        }
        TypeAnnotationKind::App { ctor, .. } => {
            // Parameterized type - convert the constructor
            convert_annotation_to_ir_type(ctx, Some(ctor))
        }
        _ => ctx.nil_type(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{
        BinOpKind, CtorId, Decl, Expr, ExprKind, FloatBits, FuncDecl, FuncDefId, LocalId, Module,
        NodeId, ParamDecl, Pattern, PatternKind, ResolvedRef, Stmt, Type as AstType, TypeKind,
        TypedRef,
    };
    use insta::assert_debug_snapshot;
    use salsa_test_macros::salsa_test;
    use trunk_ir::DialectOp;
    use trunk_ir::SymbolVec;
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
        lower_module(db, path, span_map, module, HashMap::new())
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
    // TypeScheme-based Function Lowering Tests
    // ========================================================================

    /// Wrapper for lower_module with function_types that uses #[salsa::tracked]
    /// to provide proper Salsa context for accumulator operations.
    #[salsa::tracked]
    fn test_lower_with_scheme<'db>(
        db: &'db dyn salsa::Database,
        path: PathId<'db>,
        span_map: SpanMap,
        module: Module<TypedRef<'db>>,
        scheme_entries: Vec<(Symbol, crate::ast::TypeScheme<'db>)>,
    ) -> core::Module<'db> {
        let function_types: HashMap<Symbol, crate::ast::TypeScheme<'db>> =
            scheme_entries.into_iter().collect();
        lower_module(db, path, span_map, module, function_types)
    }

    #[salsa_test]
    fn test_lower_function_with_type_scheme(db: &salsa::DatabaseImpl) {
        use crate::ast::{EffectRow, Type as AstType, TypeKind, TypeScheme};

        let path = PathId::new(db, "test.trb".to_owned());
        let span_map = SpanMap::default();

        // Create function: fn identity(x) { x }
        let x_name = Symbol::new("x");
        let x_id = LocalId::new(0);
        let x_ref = local_ref(db, x_id, x_name);
        let body = var_expr(x_ref);

        let func = FuncDecl {
            id: fresh_node_id(),
            is_pub: false,
            name: Symbol::new("identity"),
            type_params: vec![],
            params: vec![ParamDecl {
                id: fresh_node_id(),
                name: x_name,
                ty: None,
                local_id: Some(x_id),
            }],
            return_ty: None,
            effects: None,
            body,
        };

        let module = simple_module(vec![Decl::Function(func)]);

        // Create TypeScheme: forall a. fn(a) -> a
        // After convert_type, BoundVar → tribute_rt.any
        let bound_var = AstType::new(db, TypeKind::BoundVar { index: 0 });
        let effect = EffectRow::pure(db);
        let func_ty = AstType::new(
            db,
            TypeKind::Func {
                params: vec![bound_var],
                result: bound_var,
                effect,
            },
        );
        let scheme = TypeScheme::new(
            db,
            vec![crate::ast::TypeParam {
                name: Some(Symbol::new("a")),
                kind: None,
            }],
            func_ty,
        );

        let scheme_entries = vec![(Symbol::new("identity"), scheme)];

        let ir_module = test_lower_with_scheme(db, path, span_map, module, scheme_entries);
        let ops = get_module_ops(db, &ir_module);
        let func_op = ops.iter().find(|op| op.name(db) == "func").unwrap();
        let func_typed = func::Func::from_operation(db, *func_op).unwrap();

        // The function type should use tribute_rt.any for param/return (type erasure)
        let func_ir_ty = func_typed.r#type(db);
        let any_ty = tribute_ir::dialect::tribute_rt::any_type(db);
        let expected_ty =
            trunk_ir::dialect::core::Func::new(db, vec![any_ty].into(), any_ty).as_type();
        assert_eq!(func_ir_ty, expected_ty);
    }

    #[salsa_test]
    fn test_lower_function_fallback_without_scheme(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let span_map = SpanMap::default();

        // fn add(a, b) { a + b } — no TypeScheme, no annotations → defaults to int
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
                    local_id: Some(a_id),
                },
                ParamDecl {
                    id: fresh_node_id(),
                    name: b_name,
                    ty: None,
                    local_id: Some(b_id),
                },
            ],
            return_ty: None,
            effects: None,
            body,
        };

        let module = simple_module(vec![Decl::Function(func)]);

        // No function_types → fallback to annotation conversion
        let ir_module = test_lower(db, path, span_map, module);
        let ops = get_module_ops(db, &ir_module);
        let func_op = ops.iter().find(|op| op.name(db) == "func").unwrap();
        let func_typed = func::Func::from_operation(db, *func_op).unwrap();

        // Without annotations, params default to int (i64), return defaults to unit (nil)
        let func_ir_ty = func_typed.r#type(db);
        let i64_ty = trunk_ir::dialect::core::I64::new(db).as_type();
        let nil_ty = trunk_ir::dialect::core::Nil::new(db).as_type();
        let expected_ty =
            trunk_ir::dialect::core::Func::new(db, vec![i64_ty, i64_ty].into(), nil_ty).as_type();
        assert_eq!(func_ir_ty, expected_ty);
    }

    // ========================================================================
    // Type Annotation Conversion Tests
    // ========================================================================

    #[salsa_test]
    fn test_convert_annotation_to_ir_type_named(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let span_map = SpanMap::default();
        let ctx = super::super::context::IrLoweringCtx::new(
            db,
            path,
            span_map,
            HashMap::new(),
            smallvec::smallvec![Symbol::new("test")],
        );

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
        let ctx = super::super::context::IrLoweringCtx::new(
            db,
            path,
            span_map,
            HashMap::new(),
            smallvec::smallvec![Symbol::new("test")],
        );

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

    #[salsa_test]
    fn test_int_literal_within_i31_range(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let span_map = SpanMap::default();

        // i31 range: -2^30 to 2^30-1
        let i31_min: i64 = -(1 << 30); // -1,073,741,824
        let module = simple_module(vec![Decl::Function(simple_func(
            Symbol::new("main"),
            int_lit_expr(i31_min),
        ))]);

        let ir_module = test_lower(db, path, span_map, module);
        let ops = get_module_ops(db, &ir_module);
        let func_op = ops.iter().find(|op| op.name(db) == "func").unwrap();
        let func_typed = func::Func::from_operation(db, *func_op).unwrap();
        let body_ops = get_func_body_ops(db, &func_typed);

        let const_op = body_ops.iter().find(|op| op.name(db) == "const");
        assert!(
            const_op.is_some(),
            "Should have a const operation for valid i31 value"
        );
    }

    #[salsa_test]
    fn test_int_literal_exceeds_i31_max_returns_unit(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let span_map = SpanMap::default();

        // Value exceeding i31 max: 2^30 = 1,073,741,824
        let exceeds_i31_max: i64 = 1 << 30;
        let module = simple_module(vec![Decl::Function(simple_func(
            Symbol::new("main"),
            int_lit_expr(exceeds_i31_max),
        ))]);

        let ir_module = test_lower(db, path, span_map, module);
        let ops = get_module_ops(db, &ir_module);
        let func_op = ops.iter().find(|op| op.name(db) == "func").unwrap();
        let func_typed = func::Func::from_operation(db, *func_op).unwrap();
        let body_ops = get_func_body_ops(db, &func_typed);

        // When IntLit exceeds i31 range, lower_expr returns None
        assert_eq!(
            body_ops.len(),
            2,
            "Overflow IntLit should result in unit return only"
        );
    }

    #[salsa_test]
    fn test_int_literal_below_i31_min_returns_unit(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let span_map = SpanMap::default();

        // Value below i31 min: -(2^30 + 1) = -1,073,741,825
        let below_i31_min: i64 = -(1 << 30) - 1;
        let module = simple_module(vec![Decl::Function(simple_func(
            Symbol::new("main"),
            int_lit_expr(below_i31_min),
        ))]);

        let ir_module = test_lower(db, path, span_map, module);
        let ops = get_module_ops(db, &ir_module);
        let func_op = ops.iter().find(|op| op.name(db) == "func").unwrap();
        let func_typed = func::Func::from_operation(db, *func_op).unwrap();
        let body_ops = get_func_body_ops(db, &func_typed);

        // When IntLit exceeds i31 range, lower_expr returns None
        assert_eq!(
            body_ops.len(),
            2,
            "Underflow IntLit should result in unit return only"
        );
    }

    // ========================================================================
    // Extern Function Lowering Tests
    // ========================================================================

    #[salsa_test]
    fn test_lower_extern_function_basic(db: &salsa::DatabaseImpl) {
        use crate::ast::{
            EffectRow, ExternFuncDecl, Type as AstType, TypeAnnotation, TypeAnnotationKind,
            TypeKind, TypeScheme,
        };

        let path = PathId::new(db, "test.trb".to_owned());
        let span_map = SpanMap::default();

        // extern "intrinsic" fn __add(a: Int, b: Int) -> Int
        let extern_func = ExternFuncDecl {
            id: fresh_node_id(),
            is_pub: false,
            name: Symbol::new("__add"),
            abi: Symbol::new("intrinsic"),
            params: vec![
                ParamDecl {
                    id: fresh_node_id(),
                    name: Symbol::new("a"),
                    ty: Some(TypeAnnotation {
                        id: fresh_node_id(),
                        kind: TypeAnnotationKind::Named(Symbol::new("Int")),
                    }),
                    local_id: None,
                },
                ParamDecl {
                    id: fresh_node_id(),
                    name: Symbol::new("b"),
                    ty: Some(TypeAnnotation {
                        id: fresh_node_id(),
                        kind: TypeAnnotationKind::Named(Symbol::new("Int")),
                    }),
                    local_id: None,
                },
            ],
            return_ty: TypeAnnotation {
                id: fresh_node_id(),
                kind: TypeAnnotationKind::Named(Symbol::new("Int")),
            },
        };

        let module = simple_module(vec![Decl::ExternFunction(extern_func)]);

        // Create TypeScheme: fn(Int, Int) -> Int
        let int_ty = AstType::new(db, TypeKind::Int);
        let effect = EffectRow::pure(db);
        let func_ty = AstType::new(
            db,
            TypeKind::Func {
                params: vec![int_ty, int_ty],
                result: int_ty,
                effect,
            },
        );
        let scheme = TypeScheme::new(db, vec![], func_ty);
        let scheme_entries = vec![(Symbol::new("__add"), scheme)];

        let ir_module = test_lower_with_scheme(db, path, span_map, module, scheme_entries);
        let ops = get_module_ops(db, &ir_module);

        // Should produce a func.func operation
        let func_op = ops.iter().find(|op| op.name(db) == "func");
        assert!(func_op.is_some(), "Should have a func operation");

        let func_typed = func::Func::from_operation(db, *func_op.unwrap()).unwrap();
        assert_eq!(func_typed.sym_name(db), Symbol::new("__add"));

        // Body should contain only func.unreachable
        let body_ops = get_func_body_ops(db, &func_typed);
        assert_eq!(
            body_ops.len(),
            1,
            "Extern func body should have only unreachable"
        );
        assert_eq!(body_ops[0].dialect(db), "func");
        assert_eq!(body_ops[0].name(db), "unreachable");
    }

    #[salsa_test]
    fn test_lower_extern_function_type(db: &salsa::DatabaseImpl) {
        use crate::ast::{
            EffectRow, ExternFuncDecl, Type as AstType, TypeAnnotation, TypeAnnotationKind,
            TypeKind, TypeScheme,
        };

        let path = PathId::new(db, "test.trb".to_owned());
        let span_map = SpanMap::default();

        // extern fn __negate(x: Int) -> Int
        let extern_func = ExternFuncDecl {
            id: fresh_node_id(),
            is_pub: false,
            name: Symbol::new("__negate"),
            abi: Symbol::new("C"),
            params: vec![ParamDecl {
                id: fresh_node_id(),
                name: Symbol::new("x"),
                ty: Some(TypeAnnotation {
                    id: fresh_node_id(),
                    kind: TypeAnnotationKind::Named(Symbol::new("Int")),
                }),
                local_id: None,
            }],
            return_ty: TypeAnnotation {
                id: fresh_node_id(),
                kind: TypeAnnotationKind::Named(Symbol::new("Int")),
            },
        };

        let module = simple_module(vec![Decl::ExternFunction(extern_func)]);

        // Create TypeScheme: fn(Int) -> Int
        let int_ty = AstType::new(db, TypeKind::Int);
        let effect = EffectRow::pure(db);
        let func_ty = AstType::new(
            db,
            TypeKind::Func {
                params: vec![int_ty],
                result: int_ty,
                effect,
            },
        );
        let scheme = TypeScheme::new(db, vec![], func_ty);
        let scheme_entries = vec![(Symbol::new("__negate"), scheme)];

        let ir_module = test_lower_with_scheme(db, path, span_map, module, scheme_entries);
        let ops = get_module_ops(db, &ir_module);
        let func_op = ops.iter().find(|op| op.name(db) == "func").unwrap();
        let func_typed = func::Func::from_operation(db, *func_op).unwrap();

        // Verify the function type: fn(i64) -> i64
        let func_ir_ty = func_typed.r#type(db);
        let i64_ty = trunk_ir::dialect::core::I64::new(db).as_type();
        let expected_ty =
            trunk_ir::dialect::core::Func::new(db, vec![i64_ty].into(), i64_ty).as_type();
        assert_eq!(func_ir_ty, expected_ty);
    }

    #[salsa_test]
    fn test_lower_module_with_extern_and_regular_functions(db: &salsa::DatabaseImpl) {
        use crate::ast::{
            EffectRow, ExternFuncDecl, Type as AstType, TypeAnnotation, TypeAnnotationKind,
            TypeKind, TypeScheme,
        };

        let path = PathId::new(db, "test.trb".to_owned());
        let span_map = SpanMap::default();

        // extern fn __add(a: Int, b: Int) -> Int
        let extern_func = ExternFuncDecl {
            id: fresh_node_id(),
            is_pub: false,
            name: Symbol::new("__add"),
            abi: Symbol::new("intrinsic"),
            params: vec![
                ParamDecl {
                    id: fresh_node_id(),
                    name: Symbol::new("a"),
                    ty: Some(TypeAnnotation {
                        id: fresh_node_id(),
                        kind: TypeAnnotationKind::Named(Symbol::new("Int")),
                    }),
                    local_id: None,
                },
                ParamDecl {
                    id: fresh_node_id(),
                    name: Symbol::new("b"),
                    ty: Some(TypeAnnotation {
                        id: fresh_node_id(),
                        kind: TypeAnnotationKind::Named(Symbol::new("Int")),
                    }),
                    local_id: None,
                },
            ],
            return_ty: TypeAnnotation {
                id: fresh_node_id(),
                kind: TypeAnnotationKind::Named(Symbol::new("Int")),
            },
        };

        // fn main() { 42 }
        let regular_func = simple_func(Symbol::new("main"), int_lit_expr(42));

        let module = simple_module(vec![
            Decl::ExternFunction(extern_func),
            Decl::Function(regular_func),
        ]);

        // TypeScheme for __add
        let int_ty = AstType::new(db, TypeKind::Int);
        let effect = EffectRow::pure(db);
        let func_ty = AstType::new(
            db,
            TypeKind::Func {
                params: vec![int_ty, int_ty],
                result: int_ty,
                effect,
            },
        );
        let scheme = TypeScheme::new(db, vec![], func_ty);
        let scheme_entries = vec![(Symbol::new("__add"), scheme)];

        let ir_module = test_lower_with_scheme(db, path, span_map, module, scheme_entries);
        let ops = get_module_ops(db, &ir_module);

        // Should have 2 function operations
        let func_ops: Vec<_> = ops.iter().filter(|op| op.name(db) == "func").collect();
        assert_eq!(func_ops.len(), 2, "Should have extern + regular function");
    }

    // ========================================================================
    // Snapshot Tests
    // ========================================================================

    #[salsa_test]
    fn test_snapshot_nat_literal(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let module = simple_module(vec![Decl::Function(simple_func(
            Symbol::new("main"),
            nat_lit_expr(42),
        ))]);
        let ir_module = test_lower(db, path, SpanMap::default(), module);
        assert_debug_snapshot!(ir_module);
    }

    #[salsa_test]
    fn test_snapshot_binop_add(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let add_expr = binop_expr(BinOpKind::Add, int_lit_expr(10), int_lit_expr(20));
        let module = simple_module(vec![Decl::Function(simple_func(
            Symbol::new("main"),
            add_expr,
        ))]);
        let ir_module = test_lower(db, path, SpanMap::default(), module);
        assert_debug_snapshot!(ir_module);
    }

    #[salsa_test]
    fn test_snapshot_let_binding(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
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
        let ir_module = test_lower(db, path, SpanMap::default(), module);
        assert_debug_snapshot!(ir_module);
    }

    #[salsa_test]
    fn test_snapshot_extern_function(db: &salsa::DatabaseImpl) {
        use crate::ast::{
            EffectRow, ExternFuncDecl, Type as AstType, TypeAnnotation, TypeAnnotationKind,
            TypeKind, TypeScheme,
        };

        let path = PathId::new(db, "test.trb".to_owned());

        let extern_func = ExternFuncDecl {
            id: fresh_node_id(),
            is_pub: false,
            name: Symbol::new("__add"),
            abi: Symbol::new("intrinsic"),
            params: vec![
                ParamDecl {
                    id: fresh_node_id(),
                    name: Symbol::new("a"),
                    ty: Some(TypeAnnotation {
                        id: fresh_node_id(),
                        kind: TypeAnnotationKind::Named(Symbol::new("Int")),
                    }),
                    local_id: None,
                },
                ParamDecl {
                    id: fresh_node_id(),
                    name: Symbol::new("b"),
                    ty: Some(TypeAnnotation {
                        id: fresh_node_id(),
                        kind: TypeAnnotationKind::Named(Symbol::new("Int")),
                    }),
                    local_id: None,
                },
            ],
            return_ty: TypeAnnotation {
                id: fresh_node_id(),
                kind: TypeAnnotationKind::Named(Symbol::new("Int")),
            },
        };

        let module = simple_module(vec![Decl::ExternFunction(extern_func)]);

        let int_ty = AstType::new(db, TypeKind::Int);
        let effect = EffectRow::pure(db);
        let func_ty = AstType::new(
            db,
            TypeKind::Func {
                params: vec![int_ty, int_ty],
                result: int_ty,
                effect,
            },
        );
        let scheme = TypeScheme::new(db, vec![], func_ty);
        let scheme_entries = vec![(Symbol::new("__add"), scheme)];

        let ir_module =
            test_lower_with_scheme(db, path, SpanMap::default(), module, scheme_entries);
        assert_debug_snapshot!(ir_module);
    }

    #[salsa_test]
    fn test_snapshot_mixed_module(db: &salsa::DatabaseImpl) {
        use crate::ast::{
            EffectRow, ExternFuncDecl, Type as AstType, TypeAnnotation, TypeAnnotationKind,
            TypeKind, TypeScheme,
        };

        let path = PathId::new(db, "test.trb".to_owned());

        let extern_func = ExternFuncDecl {
            id: fresh_node_id(),
            is_pub: false,
            name: Symbol::new("__add"),
            abi: Symbol::new("intrinsic"),
            params: vec![
                ParamDecl {
                    id: fresh_node_id(),
                    name: Symbol::new("a"),
                    ty: Some(TypeAnnotation {
                        id: fresh_node_id(),
                        kind: TypeAnnotationKind::Named(Symbol::new("Int")),
                    }),
                    local_id: None,
                },
                ParamDecl {
                    id: fresh_node_id(),
                    name: Symbol::new("b"),
                    ty: Some(TypeAnnotation {
                        id: fresh_node_id(),
                        kind: TypeAnnotationKind::Named(Symbol::new("Int")),
                    }),
                    local_id: None,
                },
            ],
            return_ty: TypeAnnotation {
                id: fresh_node_id(),
                kind: TypeAnnotationKind::Named(Symbol::new("Int")),
            },
        };

        let regular_func = simple_func(Symbol::new("main"), nat_lit_expr(42));

        let module = simple_module(vec![
            Decl::ExternFunction(extern_func),
            Decl::Function(regular_func),
        ]);

        let int_ty = AstType::new(db, TypeKind::Int);
        let effect = EffectRow::pure(db);
        let func_ty = AstType::new(
            db,
            TypeKind::Func {
                params: vec![int_ty, int_ty],
                result: int_ty,
                effect,
            },
        );
        let scheme = TypeScheme::new(db, vec![], func_ty);
        let scheme_entries = vec![(Symbol::new("__add"), scheme)];

        let ir_module =
            test_lower_with_scheme(db, path, SpanMap::default(), module, scheme_entries);
        assert_debug_snapshot!(ir_module);
    }

    // ========================================================================
    // Call Expression Lowering Tests
    // ========================================================================

    /// Create a call expression.
    fn call_expr<'db>(
        callee: Expr<TypedRef<'db>>,
        args: Vec<Expr<TypedRef<'db>>>,
    ) -> Expr<TypedRef<'db>> {
        Expr::new(fresh_node_id(), ExprKind::Call { callee, args })
    }

    /// Create a constructor expression (Cons).
    fn cons_expr<'db>(ctor: TypedRef<'db>, args: Vec<Expr<TypedRef<'db>>>) -> Expr<TypedRef<'db>> {
        Expr::new(fresh_node_id(), ExprKind::Cons { ctor, args })
    }

    /// Tracked helper: build a direct call module and lower it.
    /// Creates `fn main() { <callee_name>(arg1, arg2) }` with a TypeScheme for the callee.
    #[salsa::tracked]
    fn test_lower_direct_call<'db>(
        db: &'db dyn salsa::Database,
        path: PathId<'db>,
        callee_name: Symbol,
        arg1: i64,
        arg2: i64,
    ) -> core::Module<'db> {
        let int_ty = AstType::new(db, TypeKind::Int);
        let effect = crate::ast::EffectRow::pure(db);
        let func_ty = AstType::new(
            db,
            TypeKind::Func {
                params: vec![int_ty, int_ty],
                result: int_ty,
                effect,
            },
        );
        let func_id = FuncDefId::new(db, SymbolVec::new(), callee_name);
        let typed_ref = TypedRef::new(ResolvedRef::Function { id: func_id }, func_ty);
        let callee = Expr::new(fresh_node_id(), ExprKind::Var(typed_ref));
        let call = call_expr(callee, vec![int_lit_expr(arg1), int_lit_expr(arg2)]);
        let module = simple_module(vec![Decl::Function(simple_func(Symbol::new("main"), call))]);

        let scheme = crate::ast::TypeScheme::new(db, vec![], func_ty);
        let function_types: HashMap<Symbol, crate::ast::TypeScheme<'db>> =
            [(callee_name, scheme)].into();
        lower_module(db, path, SpanMap::default(), module, function_types)
    }

    /// Tracked helper: build a constructor (Cons) module and lower it.
    #[salsa::tracked]
    fn test_lower_cons<'db>(
        db: &'db dyn salsa::Database,
        path: PathId<'db>,
        type_name: Symbol,
        variant_name: Symbol,
        args: Vec<i64>,
    ) -> core::Module<'db> {
        let int_ty = AstType::new(db, TypeKind::Int);
        let named_ty = AstType::new(
            db,
            TypeKind::Named {
                name: type_name,
                args: vec![int_ty],
            },
        );
        let ctor_id = CtorId::new(db, SymbolVec::new(), type_name);
        let ctor_ref = TypedRef::new(
            ResolvedRef::Constructor {
                id: ctor_id,
                variant: variant_name,
            },
            named_ty,
        );
        let arg_exprs: Vec<_> = args.into_iter().map(int_lit_expr).collect();
        let cons = cons_expr(ctor_ref, arg_exprs);
        let module = simple_module(vec![Decl::Function(simple_func(Symbol::new("main"), cons))]);
        lower_module(db, path, SpanMap::default(), module, HashMap::new())
    }

    /// Tracked helper: build a Call with Constructor callee and lower it.
    #[salsa::tracked]
    fn test_lower_call_ctor<'db>(
        db: &'db dyn salsa::Database,
        path: PathId<'db>,
        type_name: Symbol,
        variant_name: Symbol,
    ) -> core::Module<'db> {
        let int_ty = AstType::new(db, TypeKind::Int);
        let named_ty = AstType::new(
            db,
            TypeKind::Named {
                name: type_name,
                args: vec![int_ty],
            },
        );
        let ctor_id = CtorId::new(db, SymbolVec::new(), type_name);
        let callee_ref = TypedRef::new(
            ResolvedRef::Constructor {
                id: ctor_id,
                variant: variant_name,
            },
            named_ty,
        );
        let callee = var_expr(callee_ref);
        let call = call_expr(callee, vec![int_lit_expr(42)]);
        let module = simple_module(vec![Decl::Function(simple_func(Symbol::new("main"), call))]);
        lower_module(db, path, SpanMap::default(), module, HashMap::new())
    }

    #[salsa_test]
    fn test_lower_direct_function_call(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());

        let ir_module = test_lower_direct_call(db, path, Symbol::new("add"), 1, 2);
        let ops = get_module_ops(db, &ir_module);
        let func_op = ops.iter().find(|op| op.name(db) == "func").unwrap();
        let func_typed = func::Func::from_operation(db, *func_op).unwrap();
        let body_ops = get_func_body_ops(db, &func_typed);

        // Should have: const(1), const(2), call, return
        let call_op = body_ops.iter().find(|op| op.name(db) == "call");
        assert!(call_op.is_some(), "Should have a func.call operation");
        assert_eq!(
            call_op.unwrap().dialect(db),
            "func",
            "Call should be in func dialect"
        );
    }

    #[salsa_test]
    fn test_lower_indirect_call(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let span_map = SpanMap::default();

        // fn main(f) { f(42) }
        // f is a local variable of function type
        let f_name = Symbol::new("f");
        let f_id = LocalId::new(0);

        let int_ty = AstType::new(db, TypeKind::Int);
        let effect = crate::ast::EffectRow::pure(db);
        let func_ty = AstType::new(
            db,
            TypeKind::Func {
                params: vec![int_ty],
                result: int_ty,
                effect,
            },
        );
        let f_ref = TypedRef::new(ResolvedRef::local(f_id, f_name), func_ty);
        let callee = var_expr(f_ref);
        let call = call_expr(callee, vec![int_lit_expr(42)]);

        let func = FuncDecl {
            id: fresh_node_id(),
            is_pub: false,
            name: Symbol::new("main"),
            type_params: vec![],
            params: vec![ParamDecl {
                id: fresh_node_id(),
                name: f_name,
                ty: None,
                local_id: Some(f_id),
            }],
            return_ty: None,
            effects: None,
            body: call,
        };

        let module = simple_module(vec![Decl::Function(func)]);

        let ir_module = test_lower(db, path, span_map, module);
        let ops = get_module_ops(db, &ir_module);
        let func_op = ops.iter().find(|op| op.name(db) == "func").unwrap();
        let func_typed = func::Func::from_operation(db, *func_op).unwrap();
        let body_ops = get_func_body_ops(db, &func_typed);

        // Should have: const(42), call_indirect, return
        let call_op = body_ops.iter().find(|op| op.name(db) == "call_indirect");
        assert!(
            call_op.is_some(),
            "Should have a func.call_indirect operation"
        );
    }

    #[salsa_test]
    fn test_lower_call_with_multiple_args(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());

        let ir_module = test_lower_direct_call(db, path, Symbol::new("add"), 10, 20);
        let ops = get_module_ops(db, &ir_module);
        let func_op = ops.iter().find(|op| op.name(db) == "func").unwrap();
        let func_typed = func::Func::from_operation(db, *func_op).unwrap();
        let body_ops = get_func_body_ops(db, &func_typed);

        // Should have: const(10), const(20), call, return = 4 ops
        assert!(body_ops.len() >= 4, "Should have at least 4 ops");

        let call_op = body_ops.iter().find(|op| op.name(db) == "call");
        assert!(call_op.is_some(), "Should have a func.call operation");

        // Verify call operands: should have 2 arguments
        let call_operation = call_op.unwrap();
        assert_eq!(
            call_operation.operands(db).len(),
            2,
            "Call should have 2 arguments"
        );
    }

    // ========================================================================
    // Constructor (Cons) Lowering Tests
    // ========================================================================

    #[salsa_test]
    fn test_lower_nullary_constructor(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());

        let ir_module =
            test_lower_cons(db, path, Symbol::new("Option"), Symbol::new("None"), vec![]);
        let ops = get_module_ops(db, &ir_module);
        let func_op = ops.iter().find(|op| op.name(db) == "func").unwrap();
        let func_typed = func::Func::from_operation(db, *func_op).unwrap();
        let body_ops = get_func_body_ops(db, &func_typed);

        let variant_op = body_ops.iter().find(|op| op.name(db) == "variant_new");
        assert!(
            variant_op.is_some(),
            "Should have an adt.variant_new operation"
        );
        assert_eq!(
            variant_op.unwrap().dialect(db),
            "adt",
            "variant_new should be in adt dialect"
        );
        assert_eq!(variant_op.unwrap().operands(db).len(), 0);
    }

    #[salsa_test]
    fn test_lower_unary_constructor(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());

        let ir_module = test_lower_cons(
            db,
            path,
            Symbol::new("Option"),
            Symbol::new("Some"),
            vec![42],
        );
        let ops = get_module_ops(db, &ir_module);
        let func_op = ops.iter().find(|op| op.name(db) == "func").unwrap();
        let func_typed = func::Func::from_operation(db, *func_op).unwrap();
        let body_ops = get_func_body_ops(db, &func_typed);

        let variant_op = body_ops.iter().find(|op| op.name(db) == "variant_new");
        assert!(
            variant_op.is_some(),
            "Should have an adt.variant_new operation"
        );
        assert_eq!(
            variant_op.unwrap().operands(db).len(),
            1,
            "Some(42) should have 1 operand"
        );
    }

    #[salsa_test]
    fn test_lower_call_constructor_via_call(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());

        let ir_module = test_lower_call_ctor(db, path, Symbol::new("Option"), Symbol::new("Some"));
        let ops = get_module_ops(db, &ir_module);
        let func_op = ops.iter().find(|op| op.name(db) == "func").unwrap();
        let func_typed = func::Func::from_operation(db, *func_op).unwrap();
        let body_ops = get_func_body_ops(db, &func_typed);

        let variant_op = body_ops.iter().find(|op| op.name(db) == "variant_new");
        assert!(
            variant_op.is_some(),
            "Constructor call should emit adt.variant_new"
        );
    }

    // ========================================================================
    // Snapshot Tests for Call and Constructor
    // ========================================================================

    #[salsa_test]
    fn test_snapshot_function_call(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let ir_module = test_lower_direct_call(db, path, Symbol::new("add"), 10, 20);
        assert_debug_snapshot!(ir_module);
    }

    #[salsa_test]
    fn test_snapshot_constructor(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let ir_module = test_lower_cons(
            db,
            path,
            Symbol::new("Option"),
            Symbol::new("Some"),
            vec![42],
        );
        assert_debug_snapshot!(ir_module);
    }

    // ========================================================================
    // Case Expression Lowering Tests
    // ========================================================================

    /// Create a case expression.
    fn case_expr<'db>(
        scrutinee: Expr<TypedRef<'db>>,
        arms: Vec<Arm<TypedRef<'db>>>,
    ) -> Expr<TypedRef<'db>> {
        Expr::new(fresh_node_id(), ExprKind::Case { scrutinee, arms })
    }

    /// Create a case arm.
    fn arm<'db>(pattern: Pattern<TypedRef<'db>>, body: Expr<TypedRef<'db>>) -> Arm<TypedRef<'db>> {
        Arm {
            id: fresh_node_id(),
            pattern,
            guard: None,
            body,
        }
    }

    /// Create a wildcard pattern.
    fn wildcard_pattern<'db>() -> Pattern<TypedRef<'db>> {
        Pattern::new(fresh_node_id(), PatternKind::Wildcard)
    }

    /// Create a bind pattern.
    fn bind_pattern<'db>(name: Symbol, local_id: LocalId) -> Pattern<TypedRef<'db>> {
        Pattern::new(
            fresh_node_id(),
            PatternKind::Bind {
                name,
                local_id: Some(local_id),
            },
        )
    }

    /// Create a variant pattern.
    fn variant_pattern<'db>(
        ctor_ref: TypedRef<'db>,
        fields: Vec<Pattern<TypedRef<'db>>>,
    ) -> Pattern<TypedRef<'db>> {
        Pattern::new(
            fresh_node_id(),
            PatternKind::Variant {
                ctor: ctor_ref,
                fields,
            },
        )
    }

    /// Create a literal int pattern.
    fn literal_pattern_int<'db>(n: i64) -> Pattern<TypedRef<'db>> {
        Pattern::new(
            fresh_node_id(),
            PatternKind::Literal(LiteralPattern::Int(n)),
        )
    }

    #[salsa_test]
    fn test_lower_case_wildcard(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let span_map = SpanMap::default();

        // case x { _ => 42 }
        let x_name = Symbol::new("x");
        let x_id = LocalId::new(0);
        let x_ref = local_ref(db, x_id, x_name);
        let scrutinee = var_expr(x_ref);

        let arms = vec![arm(wildcard_pattern(), int_lit_expr(42))];
        let case = case_expr(scrutinee, arms);

        // Wrap in a function with param x
        let func = FuncDecl {
            id: fresh_node_id(),
            is_pub: false,
            name: Symbol::new("main"),
            type_params: vec![],
            params: vec![ParamDecl {
                id: fresh_node_id(),
                name: x_name,
                ty: None,
                local_id: Some(x_id),
            }],
            return_ty: None,
            effects: None,
            body: case,
        };
        let module = simple_module(vec![Decl::Function(func)]);

        let ir_module = test_lower(db, path, span_map, module);
        let ops = get_module_ops(db, &ir_module);
        let func_op = ops.iter().find(|op| op.name(db) == "func").unwrap();
        let func_typed = func::Func::from_operation(db, *func_op).unwrap();
        let body_ops = get_func_body_ops(db, &func_typed);

        // Single wildcard arm should NOT produce scf.if — just const + return
        let if_op = body_ops.iter().find(|op| op.name(db) == "if");
        assert!(
            if_op.is_none(),
            "Wildcard-only case should not produce scf.if"
        );

        let const_op = body_ops.iter().find(|op| op.name(db) == "const");
        assert!(const_op.is_some(), "Should have a const operation for 42");
    }

    #[salsa_test]
    fn test_lower_case_literal(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let span_map = SpanMap::default();

        // case n { 1 => 10, _ => 20 }
        let n_name = Symbol::new("n");
        let n_id = LocalId::new(0);
        let n_ref = local_ref(db, n_id, n_name);
        let scrutinee = var_expr(n_ref);

        let arms = vec![
            arm(literal_pattern_int(1), int_lit_expr(10)),
            arm(wildcard_pattern(), int_lit_expr(20)),
        ];
        let case = case_expr(scrutinee, arms);

        let func = FuncDecl {
            id: fresh_node_id(),
            is_pub: false,
            name: Symbol::new("main"),
            type_params: vec![],
            params: vec![ParamDecl {
                id: fresh_node_id(),
                name: n_name,
                ty: None,
                local_id: Some(n_id),
            }],
            return_ty: None,
            effects: None,
            body: case,
        };
        let module = simple_module(vec![Decl::Function(func)]);

        let ir_module = test_lower(db, path, span_map, module);
        let ops = get_module_ops(db, &ir_module);
        let func_op = ops.iter().find(|op| op.name(db) == "func").unwrap();
        let func_typed = func::Func::from_operation(db, *func_op).unwrap();
        let body_ops = get_func_body_ops(db, &func_typed);

        // Should have cmp_eq for literal check and scf.if
        let cmp_op = body_ops.iter().find(|op| op.name(db) == "cmp_eq");
        assert!(cmp_op.is_some(), "Should have cmp_eq for literal pattern");

        let if_op = body_ops.iter().find(|op| op.name(db) == "if");
        assert!(if_op.is_some(), "Should have scf.if for case branch");
    }

    /// Tracked helper for building a case-with-variant module.
    #[salsa::tracked]
    fn test_lower_case_variant_module<'db>(
        db: &'db dyn salsa::Database,
        path: PathId<'db>,
    ) -> core::Module<'db> {
        let int_ty = AstType::new(db, TypeKind::Int);
        let option_ty = AstType::new(
            db,
            TypeKind::Named {
                name: Symbol::new("Option"),
                args: vec![int_ty],
            },
        );

        let x_name = Symbol::new("x");
        let x_id = LocalId::new(0);
        let x_ref = TypedRef::new(ResolvedRef::local(x_id, x_name), option_ty);
        let scrutinee = var_expr(x_ref);

        let v_name = Symbol::new("v");
        let v_id = LocalId::new(1);

        // Some(v) pattern
        let some_ctor_id = CtorId::new(db, SymbolVec::new(), Symbol::new("Option"));
        let some_ctor_ref = TypedRef::new(
            ResolvedRef::Constructor {
                id: some_ctor_id,
                variant: Symbol::new("Some"),
            },
            option_ty,
        );
        let some_pattern = variant_pattern(some_ctor_ref, vec![bind_pattern(v_name, v_id)]);
        let v_ref = TypedRef::new(ResolvedRef::local(v_id, v_name), int_ty);

        // None pattern
        let none_ctor_id = CtorId::new(db, SymbolVec::new(), Symbol::new("Option"));
        let none_ctor_ref = TypedRef::new(
            ResolvedRef::Constructor {
                id: none_ctor_id,
                variant: Symbol::new("None"),
            },
            option_ty,
        );
        let none_pattern = variant_pattern(none_ctor_ref, vec![]);

        let arms = vec![
            arm(some_pattern, var_expr(v_ref)),
            arm(none_pattern, int_lit_expr(0)),
        ];
        let case = case_expr(scrutinee, arms);

        let func = FuncDecl {
            id: fresh_node_id(),
            is_pub: false,
            name: Symbol::new("main"),
            type_params: vec![],
            params: vec![ParamDecl {
                id: fresh_node_id(),
                name: x_name,
                ty: None,
                local_id: Some(x_id),
            }],
            return_ty: None,
            effects: None,
            body: case,
        };
        let module = simple_module(vec![Decl::Function(func)]);
        lower_module(db, path, SpanMap::default(), module, HashMap::new())
    }

    #[salsa_test]
    fn test_lower_case_variant(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());

        let ir_module = test_lower_case_variant_module(db, path);
        let ops = get_module_ops(db, &ir_module);
        let func_op = ops.iter().find(|op| op.name(db) == "func").unwrap();
        let func_typed = func::Func::from_operation(db, *func_op).unwrap();
        let body_ops = get_func_body_ops(db, &func_typed);

        // Should have variant_is for pattern check and scf.if for branching
        let variant_is_op = body_ops.iter().find(|op| op.name(db) == "variant_is");
        assert!(
            variant_is_op.is_some(),
            "Should have adt.variant_is for variant pattern"
        );

        let if_op = body_ops.iter().find(|op| op.name(db) == "if");
        assert!(
            if_op.is_some(),
            "Should have scf.if for case variant branch"
        );
    }

    // ========================================================================
    // Enum Declaration Lowering Tests
    // ========================================================================

    #[salsa_test]
    fn test_lower_enum_decl(db: &salsa::DatabaseImpl) {
        use crate::ast::{EnumDecl, VariantDecl};

        let path = PathId::new(db, "test.trb".to_owned());
        let span_map = SpanMap::default();

        // enum Color { Red, Green, Blue }
        let enum_decl = EnumDecl {
            id: fresh_node_id(),
            is_pub: false,
            name: Symbol::new("Color"),
            type_params: vec![],
            variants: vec![
                VariantDecl {
                    id: fresh_node_id(),
                    name: Symbol::new("Red"),
                    fields: vec![],
                },
                VariantDecl {
                    id: fresh_node_id(),
                    name: Symbol::new("Green"),
                    fields: vec![],
                },
                VariantDecl {
                    id: fresh_node_id(),
                    name: Symbol::new("Blue"),
                    fields: vec![],
                },
            ],
        };

        let module = simple_module(vec![Decl::Enum(enum_decl)]);
        let ir_module = test_lower(db, path, span_map, module);
        let ops = get_module_ops(db, &ir_module);

        // Should have tribute.enum_def operation
        let enum_op = ops.iter().find(|op| op.name(db) == "enum_def");
        assert!(
            enum_op.is_some(),
            "Should have a tribute.enum_def operation"
        );
        assert_eq!(
            enum_op.unwrap().dialect(db),
            "tribute",
            "enum_def should be in tribute dialect"
        );
    }

    #[salsa_test]
    fn test_lower_enum_decl_with_fields(db: &salsa::DatabaseImpl) {
        use crate::ast::{EnumDecl, FieldDecl, TypeAnnotation, TypeAnnotationKind, VariantDecl};

        let path = PathId::new(db, "test.trb".to_owned());
        let span_map = SpanMap::default();

        // enum Shape { Circle(Float), Rect(Float, Float) }
        let enum_decl = EnumDecl {
            id: fresh_node_id(),
            is_pub: false,
            name: Symbol::new("Shape"),
            type_params: vec![],
            variants: vec![
                VariantDecl {
                    id: fresh_node_id(),
                    name: Symbol::new("Circle"),
                    fields: vec![FieldDecl {
                        id: fresh_node_id(),
                        is_pub: false,
                        name: None,
                        ty: TypeAnnotation {
                            id: fresh_node_id(),
                            kind: TypeAnnotationKind::Named(Symbol::new("Float")),
                        },
                    }],
                },
                VariantDecl {
                    id: fresh_node_id(),
                    name: Symbol::new("Rect"),
                    fields: vec![
                        FieldDecl {
                            id: fresh_node_id(),
                            is_pub: false,
                            name: None,
                            ty: TypeAnnotation {
                                id: fresh_node_id(),
                                kind: TypeAnnotationKind::Named(Symbol::new("Float")),
                            },
                        },
                        FieldDecl {
                            id: fresh_node_id(),
                            is_pub: false,
                            name: None,
                            ty: TypeAnnotation {
                                id: fresh_node_id(),
                                kind: TypeAnnotationKind::Named(Symbol::new("Float")),
                            },
                        },
                    ],
                },
            ],
        };

        let module = simple_module(vec![Decl::Enum(enum_decl)]);
        let ir_module = test_lower(db, path, span_map, module);
        let ops = get_module_ops(db, &ir_module);

        let enum_op = ops.iter().find(|op| op.name(db) == "enum_def");
        assert!(
            enum_op.is_some(),
            "Should have a tribute.enum_def operation"
        );
    }

    // ========================================================================
    // Snapshot Tests for Case and Enum
    // ========================================================================

    #[salsa_test]
    fn test_snapshot_case_variant(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let ir_module = test_lower_case_variant_module(db, path);
        assert_debug_snapshot!(ir_module);
    }

    #[salsa_test]
    fn test_snapshot_enum_decl(db: &salsa::DatabaseImpl) {
        use crate::ast::{EnumDecl, VariantDecl};

        let path = PathId::new(db, "test.trb".to_owned());
        let enum_decl = EnumDecl {
            id: fresh_node_id(),
            is_pub: false,
            name: Symbol::new("Color"),
            type_params: vec![],
            variants: vec![
                VariantDecl {
                    id: fresh_node_id(),
                    name: Symbol::new("Red"),
                    fields: vec![],
                },
                VariantDecl {
                    id: fresh_node_id(),
                    name: Symbol::new("Green"),
                    fields: vec![],
                },
                VariantDecl {
                    id: fresh_node_id(),
                    name: Symbol::new("Blue"),
                    fields: vec![],
                },
            ],
        };
        let module = simple_module(vec![Decl::Enum(enum_decl)]);
        let ir_module = test_lower(db, path, SpanMap::default(), module);
        assert_debug_snapshot!(ir_module);
    }

    // ========================================================================
    // Struct Declaration Tests
    // ========================================================================

    #[salsa_test]
    fn test_lower_struct_decl(db: &dyn salsa::Database) {
        use crate::ast::{FieldDecl, StructDecl};

        let path = PathId::new(db, "test");
        let struct_decl = StructDecl {
            id: fresh_node_id(),
            is_pub: false,
            name: Symbol::new("Point"),
            type_params: vec![],
            fields: vec![
                FieldDecl {
                    id: fresh_node_id(),
                    is_pub: false,
                    name: Some(Symbol::new("x")),
                    ty: TypeAnnotation {
                        id: fresh_node_id(),
                        kind: TypeAnnotationKind::Named(Symbol::new("Int")),
                    },
                },
                FieldDecl {
                    id: fresh_node_id(),
                    is_pub: false,
                    name: Some(Symbol::new("y")),
                    ty: TypeAnnotation {
                        id: fresh_node_id(),
                        kind: TypeAnnotationKind::Named(Symbol::new("Int")),
                    },
                },
            ],
        };
        let module = simple_module(vec![Decl::Struct(struct_decl)]);
        let ir_module = test_lower(db, path, SpanMap::default(), module);
        let ops = get_module_ops(db, &ir_module);

        // Should have two operations: tribute.struct_def and core.module (accessors)
        assert_eq!(ops.len(), 2, "Expected struct_def and accessor module");

        // First operation should be tribute.struct_def
        let struct_def_op = ops[0];
        assert_eq!(struct_def_op.dialect(db), tribute::DIALECT_NAME());
        assert_eq!(struct_def_op.name(db), tribute::STRUCT_DEF());

        // Second operation should be core.module (accessor module)
        let accessor_module_op = ops[1];
        assert_eq!(accessor_module_op.dialect(db), core::DIALECT_NAME());
        assert_eq!(accessor_module_op.name(db), core::MODULE());

        // Accessor module should contain 2 functions (getters for x and y)
        let accessor_module =
            core::Module::from_operation(db, accessor_module_op).expect("Expected core.module");
        let accessor_ops = get_module_ops(db, &accessor_module);
        assert_eq!(
            accessor_ops.len(),
            2,
            "Expected 2 accessor functions (x and y)"
        );

        // Both should be func.func operations
        for op in &accessor_ops {
            assert_eq!(op.dialect(db), func::DIALECT_NAME());
            assert_eq!(op.name(db), func::FUNC());
        }
    }

    // ========================================================================
    // Tuple Expression Tests
    // ========================================================================

    #[salsa_test]
    fn test_lower_tuple(db: &dyn salsa::Database) {
        let path = PathId::new(db, "test");
        let module = simple_module(vec![Decl::Function(FuncDecl {
            id: fresh_node_id(),
            is_pub: false,
            name: Symbol::new("make_tuple"),
            type_params: vec![],
            params: vec![],
            return_ty: None,
            effects: None,
            body: Expr::new(
                fresh_node_id(),
                ExprKind::Tuple(vec![int_lit_expr(1), int_lit_expr(2), int_lit_expr(3)]),
            ),
        })]);

        let ir_module = test_lower(db, path, SpanMap::default(), module);
        let ops = get_module_ops(db, &ir_module);

        // Should have a function
        assert_eq!(ops.len(), 1);
        let func_op = func::Func::from_operation(db, ops[0]).expect("Expected func.func");
        let body_ops = get_func_body_ops(db, &func_op);

        // Body should contain tribute.tuple operation
        let tuple_ops: Vec<_> = body_ops
            .iter()
            .filter(|op| {
                op.dialect(db) == tribute::DIALECT_NAME() && op.name(db) == tribute::TUPLE()
            })
            .collect();
        assert_eq!(tuple_ops.len(), 1, "Expected one tribute.tuple operation");

        // Tuple should have 3 operands
        let tuple_op = tuple_ops[0];
        assert_eq!(
            tuple_op.operands(db).len(),
            3,
            "Tuple should have 3 elements"
        );
    }

    // ========================================================================
    // Error Propagation Tests (Bug Fix Verification)
    // ========================================================================

    /// Tracked helper: build a call with one failing argument.
    #[salsa::tracked]
    fn test_lower_call_with_overflow_arg<'db>(
        db: &'db dyn salsa::Database,
        path: PathId<'db>,
    ) -> core::Module<'db> {
        let int_ty = AstType::new(db, TypeKind::Int);
        let effect = crate::ast::EffectRow::pure(db);
        let func_ty = AstType::new(
            db,
            TypeKind::Func {
                params: vec![int_ty, int_ty],
                result: int_ty,
                effect,
            },
        );
        let func_id = FuncDefId::new(db, SymbolVec::new(), Symbol::new("add"));
        let typed_ref = TypedRef::new(ResolvedRef::Function { id: func_id }, func_ty);
        let callee = Expr::new(fresh_node_id(), ExprKind::Var(typed_ref));

        // Second argument exceeds i31 range → should fail
        let exceeds_i31: u64 = 1 << 30;
        let call = call_expr(callee, vec![int_lit_expr(1), nat_lit_expr(exceeds_i31)]);

        let module = simple_module(vec![Decl::Function(simple_func(Symbol::new("main"), call))]);
        lower_module(db, path, SpanMap::default(), module, HashMap::new())
    }

    #[salsa_test]
    fn test_call_propagates_arg_error(db: &salsa::DatabaseImpl) {
        // When one argument fails to lower (e.g., overflow), the entire call should fail
        let path = PathId::new(db, "test.trb".to_owned());
        let ir_module = test_lower_call_with_overflow_arg(db, path);
        let ops = get_module_ops(db, &ir_module);
        let func_op = ops.iter().find(|op| op.name(db) == "func").unwrap();
        let func_typed = func::Func::from_operation(db, *func_op).unwrap();
        let body_ops = get_func_body_ops(db, &func_typed);

        // With proper error propagation, the call should NOT be emitted
        // Instead, we get a fallback unit return (2 ops: const unit + return)
        let call_op = body_ops.iter().find(|op| op.name(db) == "call");
        assert!(
            call_op.is_none(),
            "Call with failing arg should not emit func.call"
        );
    }

    /// Tracked helper: build a Cons with one failing argument.
    #[salsa::tracked]
    fn test_lower_cons_with_overflow_arg<'db>(
        db: &'db dyn salsa::Database,
        path: PathId<'db>,
    ) -> core::Module<'db> {
        let int_ty = AstType::new(db, TypeKind::Int);
        let named_ty = AstType::new(
            db,
            TypeKind::Named {
                name: Symbol::new("Pair"),
                args: vec![int_ty],
            },
        );
        let ctor_id = CtorId::new(db, SymbolVec::new(), Symbol::new("Pair"));
        let ctor_ref = TypedRef::new(
            ResolvedRef::Constructor {
                id: ctor_id,
                variant: Symbol::new("Pair"),
            },
            named_ty,
        );

        // Second argument exceeds i31 range → should fail
        let exceeds_i31: u64 = 1 << 30;
        let cons = cons_expr(ctor_ref, vec![int_lit_expr(1), nat_lit_expr(exceeds_i31)]);

        let module = simple_module(vec![Decl::Function(simple_func(Symbol::new("main"), cons))]);
        lower_module(db, path, SpanMap::default(), module, HashMap::new())
    }

    #[salsa_test]
    fn test_cons_propagates_arg_error(db: &salsa::DatabaseImpl) {
        // When one constructor argument fails, the entire Cons should fail
        let path = PathId::new(db, "test.trb".to_owned());
        let ir_module = test_lower_cons_with_overflow_arg(db, path);
        let ops = get_module_ops(db, &ir_module);
        let func_op = ops.iter().find(|op| op.name(db) == "func").unwrap();
        let func_typed = func::Func::from_operation(db, *func_op).unwrap();
        let body_ops = get_func_body_ops(db, &func_typed);

        // With proper error propagation, variant_new should NOT be emitted
        let variant_op = body_ops.iter().find(|op| op.name(db) == "variant_new");
        assert!(
            variant_op.is_none(),
            "Cons with failing arg should not emit adt.variant_new"
        );
    }

    #[salsa_test]
    fn test_tuple_propagates_element_error(db: &salsa::DatabaseImpl) {
        // When one tuple element fails, the entire Tuple should fail
        let path = PathId::new(db, "test.trb".to_owned());
        let span_map = SpanMap::default();

        // Third element exceeds i31 range → should fail
        let exceeds_i31: u64 = 1 << 30;
        let tuple = Expr::new(
            fresh_node_id(),
            ExprKind::Tuple(vec![
                int_lit_expr(1),
                int_lit_expr(2),
                nat_lit_expr(exceeds_i31),
            ]),
        );

        let module = simple_module(vec![Decl::Function(simple_func(
            Symbol::new("main"),
            tuple,
        ))]);
        let ir_module = test_lower(db, path, span_map, module);
        let ops = get_module_ops(db, &ir_module);
        let func_op = ops.iter().find(|op| op.name(db) == "func").unwrap();
        let func_typed = func::Func::from_operation(db, *func_op).unwrap();
        let body_ops = get_func_body_ops(db, &func_typed);

        // With proper error propagation, tuple should NOT be emitted
        let tuple_op = body_ops
            .iter()
            .find(|op| op.dialect(db) == "tribute" && op.name(db) == "tuple");
        assert!(
            tuple_op.is_none(),
            "Tuple with failing element should not emit tribute.tuple"
        );
    }

    // ========================================================================
    // Pattern Guard Tests
    // ========================================================================

    /// Create a case arm with a guard.
    fn arm_with_guard<'db>(
        pattern: Pattern<TypedRef<'db>>,
        guard: Expr<TypedRef<'db>>,
        body: Expr<TypedRef<'db>>,
    ) -> Arm<TypedRef<'db>> {
        Arm {
            id: fresh_node_id(),
            pattern,
            guard: Some(guard),
            body,
        }
    }

    /// Tracked helper: build case with simple guard (bind pattern).
    #[salsa::tracked]
    fn test_lower_case_with_simple_guard<'db>(
        db: &'db dyn salsa::Database,
        path: PathId<'db>,
    ) -> core::Module<'db> {
        // case n { x if x > 0 => 1, _ => 0 }
        let n_name = Symbol::new("n");
        let n_id = LocalId::new(0);
        let x_name = Symbol::new("x");
        let x_id = LocalId::new(1);

        let int_ty = AstType::new(db, TypeKind::Int);
        let n_ref = TypedRef::new(ResolvedRef::local(n_id, n_name), int_ty);
        let scrutinee = var_expr(n_ref);

        // Guard: x > 0
        let x_ref = TypedRef::new(ResolvedRef::local(x_id, x_name), int_ty);
        let guard = binop_expr(BinOpKind::Gt, var_expr(x_ref), int_lit_expr(0));

        let arms = vec![
            arm_with_guard(bind_pattern(x_name, x_id), guard, int_lit_expr(1)),
            arm(wildcard_pattern(), int_lit_expr(0)),
        ];
        let case = case_expr(scrutinee, arms);

        let func = FuncDecl {
            id: fresh_node_id(),
            is_pub: false,
            name: Symbol::new("main"),
            type_params: vec![],
            params: vec![ParamDecl {
                id: fresh_node_id(),
                name: n_name,
                ty: None,
                local_id: Some(n_id),
            }],
            return_ty: None,
            effects: None,
            body: case,
        };
        let module = simple_module(vec![Decl::Function(func)]);
        lower_module(db, path, SpanMap::default(), module, HashMap::new())
    }

    #[salsa_test]
    fn test_lower_case_with_guard(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let ir_module = test_lower_case_with_simple_guard(db, path);
        let ops = get_module_ops(db, &ir_module);
        let func_op = ops.iter().find(|op| op.name(db) == "func").unwrap();
        let func_typed = func::Func::from_operation(db, *func_op).unwrap();
        let body_ops = get_func_body_ops(db, &func_typed);

        // With the new implementation:
        // - Outer scf.if checks pattern_cond (const true for bind pattern)
        // - Inner scf.if (in then region) checks guard_cond
        // - cmp_gt and inner scf.if are inside the outer then region, not at top level

        let if_op = body_ops.iter().find(|op| op.name(db) == "if");
        assert!(if_op.is_some(), "Should have scf.if for case branch");

        // arith.and should NOT exist — guard is checked via nested scf.if
        let and_op = body_ops.iter().find(|op| op.name(db) == "and");
        assert!(
            and_op.is_none(),
            "Should NOT have arith.and — guard uses nested scf.if"
        );

        // cmp_gt is inside the then region, not at function body level
        // We verify the structure is correct by checking scf.if exists
    }

    /// Tracked helper: build case with variant pattern and guard.
    #[salsa::tracked]
    fn test_lower_case_variant_with_guard<'db>(
        db: &'db dyn salsa::Database,
        path: PathId<'db>,
    ) -> core::Module<'db> {
        // case opt { Some(x) if x > 0 => x, _ => 0 }
        let int_ty = AstType::new(db, TypeKind::Int);
        let option_ty = AstType::new(
            db,
            TypeKind::Named {
                name: Symbol::new("Option"),
                args: vec![int_ty],
            },
        );

        let opt_name = Symbol::new("opt");
        let opt_id = LocalId::new(0);
        let x_name = Symbol::new("x");
        let x_id = LocalId::new(1);

        let opt_ref = TypedRef::new(ResolvedRef::local(opt_id, opt_name), option_ty);
        let scrutinee = var_expr(opt_ref);

        // Some(x) pattern
        let some_ctor_id = CtorId::new(db, SymbolVec::new(), Symbol::new("Option"));
        let some_ctor_ref = TypedRef::new(
            ResolvedRef::Constructor {
                id: some_ctor_id,
                variant: Symbol::new("Some"),
            },
            option_ty,
        );
        let some_pattern = variant_pattern(some_ctor_ref, vec![bind_pattern(x_name, x_id)]);

        // Guard: x > 0
        let x_ref = TypedRef::new(ResolvedRef::local(x_id, x_name), int_ty);
        let guard = binop_expr(BinOpKind::Gt, var_expr(x_ref.clone()), int_lit_expr(0));

        let arms = vec![
            arm_with_guard(some_pattern, guard, var_expr(x_ref)),
            arm(wildcard_pattern(), int_lit_expr(0)),
        ];
        let case = case_expr(scrutinee, arms);

        let func = FuncDecl {
            id: fresh_node_id(),
            is_pub: false,
            name: Symbol::new("main"),
            type_params: vec![],
            params: vec![ParamDecl {
                id: fresh_node_id(),
                name: opt_name,
                ty: None,
                local_id: Some(opt_id),
            }],
            return_ty: None,
            effects: None,
            body: case,
        };
        let module = simple_module(vec![Decl::Function(func)]);
        lower_module(db, path, SpanMap::default(), module, HashMap::new())
    }

    #[salsa_test]
    fn test_lower_case_guard_with_variant(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let ir_module = test_lower_case_variant_with_guard(db, path);
        let ops = get_module_ops(db, &ir_module);
        let func_op = ops.iter().find(|op| op.name(db) == "func").unwrap();
        let func_typed = func::Func::from_operation(db, *func_op).unwrap();
        let body_ops = get_func_body_ops(db, &func_typed);

        // With the new implementation:
        // - Outer scf.if checks variant_is (at function body level)
        // - Inner scf.if (in then region) checks guard_cond after bind_pattern_fields
        // - variant_cast and cmp_gt are inside the outer then region, not at top level

        // variant_is should be at top level (pattern check)
        let variant_is_op = body_ops.iter().find(|op| op.name(db) == "variant_is");
        assert!(
            variant_is_op.is_some(),
            "Should have adt.variant_is for Some pattern"
        );

        // scf.if should exist for pattern branch
        let if_op = body_ops.iter().find(|op| op.name(db) == "if");
        assert!(if_op.is_some(), "Should have scf.if for case branch");

        // arith.and should NOT exist — guard is checked via nested scf.if
        let and_op = body_ops.iter().find(|op| op.name(db) == "and");
        assert!(
            and_op.is_none(),
            "Should NOT have arith.and — guard uses nested scf.if"
        );

        // variant_cast should NOT be at top level — it's inside the then region
        let variant_cast_op = body_ops.iter().find(|op| op.name(db) == "variant_cast");
        assert!(
            variant_cast_op.is_none(),
            "variant_cast should be inside then region, not at function body level"
        );
    }

    // ========================================================================
    // Bug Fix Regression Tests
    // ========================================================================

    /// Test that single-arm variant patterns emit proper runtime checks.
    ///
    /// Bug #2 fix: Single arm with variant pattern should NOT use fast-path.
    /// It must emit variant_is and scf.if for proper pattern matching.
    #[salsa::tracked]
    fn test_single_arm_variant_helper<'db>(
        db: &'db dyn salsa::Database,
        path: PathId<'db>,
    ) -> core::Module<'db> {
        // case opt { Some(x) => x }
        let int_ty = AstType::new(db, TypeKind::Int);
        let option_ty = AstType::new(
            db,
            TypeKind::Named {
                name: Symbol::new("Option"),
                args: vec![int_ty],
            },
        );

        let opt_name = Symbol::new("opt");
        let opt_id = LocalId::new(0);
        let x_name = Symbol::new("x");
        let x_id = LocalId::new(1);

        let opt_ref = TypedRef::new(ResolvedRef::local(opt_id, opt_name), option_ty);
        let scrutinee = var_expr(opt_ref);

        // Some(x) pattern
        let some_ctor_id = CtorId::new(db, SymbolVec::new(), Symbol::new("Option"));
        let some_ctor_ref = TypedRef::new(
            ResolvedRef::Constructor {
                id: some_ctor_id,
                variant: Symbol::new("Some"),
            },
            option_ty,
        );
        let some_pattern = variant_pattern(some_ctor_ref, vec![bind_pattern(x_name, x_id)]);

        let x_ref = TypedRef::new(ResolvedRef::local(x_id, x_name), int_ty);
        let arms = vec![arm(some_pattern, var_expr(x_ref))];
        let case = case_expr(scrutinee, arms);

        let func = FuncDecl {
            id: fresh_node_id(),
            is_pub: false,
            name: Symbol::new("main"),
            type_params: vec![],
            params: vec![ParamDecl {
                id: fresh_node_id(),
                name: opt_name,
                ty: None,
                local_id: Some(opt_id),
            }],
            return_ty: None,
            effects: None,
            body: case,
        };
        let module = simple_module(vec![Decl::Function(func)]);
        lower_module(db, path, SpanMap::default(), module, HashMap::new())
    }

    #[salsa_test]
    fn test_single_arm_variant_emits_check(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let ir_module = test_single_arm_variant_helper(db, path);
        let ops = get_module_ops(db, &ir_module);
        let func_op = ops.iter().find(|op| op.name(db) == "func").unwrap();
        let func_typed = func::Func::from_operation(db, *func_op).unwrap();
        let body_ops = get_func_body_ops(db, &func_typed);

        // Single variant arm should NOT use fast-path — must emit variant_is and scf.if
        let variant_is_op = body_ops.iter().find(|op| op.name(db) == "variant_is");
        assert!(
            variant_is_op.is_some(),
            "Single variant arm must emit adt.variant_is check"
        );

        let if_op = body_ops.iter().find(|op| op.name(db) == "if");
        assert!(
            if_op.is_some(),
            "Single variant arm must emit scf.if for conditional branching"
        );
    }

    /// Test that single-arm with guard is not ignored.
    ///
    /// Bug #2 fix: Single arm with guard should NOT use fast-path.
    #[salsa::tracked]
    fn test_single_arm_guard_helper<'db>(
        db: &'db dyn salsa::Database,
        path: PathId<'db>,
    ) -> core::Module<'db> {
        // case n { x if x > 0 => x }
        let int_ty = AstType::new(db, TypeKind::Int);
        let n_name = Symbol::new("n");
        let n_id = LocalId::new(0);
        let x_name = Symbol::new("x");
        let x_id = LocalId::new(1);

        let n_ref = TypedRef::new(ResolvedRef::local(n_id, n_name), int_ty);
        let scrutinee = var_expr(n_ref);

        // Guard: x > 0
        let x_ref = TypedRef::new(ResolvedRef::local(x_id, x_name), int_ty);
        let guard = binop_expr(BinOpKind::Gt, var_expr(x_ref.clone()), int_lit_expr(0));

        let arms = vec![arm_with_guard(
            bind_pattern(x_name, x_id),
            guard,
            var_expr(x_ref),
        )];
        let case = case_expr(scrutinee, arms);

        let func = FuncDecl {
            id: fresh_node_id(),
            is_pub: false,
            name: Symbol::new("main"),
            type_params: vec![],
            params: vec![ParamDecl {
                id: fresh_node_id(),
                name: n_name,
                ty: None,
                local_id: Some(n_id),
            }],
            return_ty: None,
            effects: None,
            body: case,
        };
        let module = simple_module(vec![Decl::Function(func)]);
        lower_module(db, path, SpanMap::default(), module, HashMap::new())
    }

    #[salsa_test]
    fn test_single_arm_with_guard_not_ignored(db: &salsa::DatabaseImpl) {
        let path = PathId::new(db, "test.trb".to_owned());
        let ir_module = test_single_arm_guard_helper(db, path);
        let ops = get_module_ops(db, &ir_module);
        let func_op = ops.iter().find(|op| op.name(db) == "func").unwrap();
        let func_typed = func::Func::from_operation(db, *func_op).unwrap();
        let body_ops = get_func_body_ops(db, &func_typed);

        // Single arm with guard should NOT use fast-path — must emit scf.if
        let if_op = body_ops.iter().find(|op| op.name(db) == "if");
        assert!(
            if_op.is_some(),
            "Single arm with guard must emit scf.if (guard must be evaluated)"
        );
    }

    // =========================================================================
    // Tuple Pattern Tests
    // =========================================================================

    #[salsa_test]
    fn test_tuple_pattern_lowering(db: &salsa::DatabaseImpl) {
        // Test: case pair { #(a, b) => a + b }
        let path = PathId::new(db, "test.trb".to_owned());
        let int_ty = AstType::new(db, TypeKind::Int);

        // Create bind patterns for 'a' and 'b' with LocalIds
        let local_a = LocalId::new(0);
        let local_b = LocalId::new(1);
        let pair_id = LocalId::new(2);

        let bind_a = Pattern::new(
            fresh_node_id(),
            PatternKind::Bind {
                name: Symbol::new("a"),
                local_id: Some(local_a),
            },
        );
        let bind_b = Pattern::new(
            fresh_node_id(),
            PatternKind::Bind {
                name: Symbol::new("b"),
                local_id: Some(local_b),
            },
        );

        // Tuple pattern #(a, b)
        let tuple_pattern: Pattern<TypedRef<'_>> =
            Pattern::new(fresh_node_id(), PatternKind::Tuple(vec![bind_a, bind_b]));

        // Arm body: a + b (using variables)
        let a_ref = TypedRef::new(ResolvedRef::local(local_a, Symbol::new("a")), int_ty);
        let b_ref = TypedRef::new(ResolvedRef::local(local_b, Symbol::new("b")), int_ty);
        let body = binop_expr(BinOpKind::Add, var_expr(a_ref), var_expr(b_ref));

        let arm = Arm {
            id: fresh_node_id(),
            pattern: tuple_pattern,
            guard: None,
            body,
        };

        // Scrutinee: a tuple variable
        let tuple_ty = AstType::new(db, TypeKind::Tuple(vec![int_ty, int_ty]));
        let pair_ref = TypedRef::new(ResolvedRef::local(pair_id, Symbol::new("pair")), tuple_ty);
        let scrutinee = var_expr(pair_ref);

        // Case expression
        let case_expr = Expr::new(
            fresh_node_id(),
            ExprKind::Case {
                scrutinee,
                arms: vec![arm],
            },
        );

        // Function definition
        let func_decl = FuncDecl {
            id: fresh_node_id(),
            name: Symbol::new("test_fn"),
            type_params: vec![],
            params: vec![ParamDecl {
                id: fresh_node_id(),
                name: Symbol::new("pair"),
                ty: None,
                local_id: Some(pair_id),
            }],
            return_ty: None,
            effects: None,
            body: block_expr(vec![], case_expr),
            is_pub: false,
        };

        let module = simple_module(vec![Decl::Function(func_decl)]);

        let ir_module = test_lower(db, path, SpanMap::default(), module);
        let ops = get_module_ops(db, &ir_module);

        // Should have a function
        let func_op = ops.iter().find(|op| op.name(db) == "func");
        assert!(func_op.is_some(), "Should emit func operation");

        let func_typed = func::Func::from_operation(db, *func_op.unwrap()).unwrap();
        let body_ops = get_func_body_ops(db, &func_typed);

        // Should have struct_get operations for extracting tuple elements
        let struct_gets: Vec<_> = body_ops
            .iter()
            .filter(|op| op.name(db) == "struct_get")
            .collect();
        assert!(
            struct_gets.len() >= 2,
            "Should emit at least 2 struct_get operations for tuple element extraction, got {}",
            struct_gets.len()
        );
    }

    // =========================================================================
    // Lambda Expression Tests
    // =========================================================================

    #[salsa_test]
    fn test_lambda_expression_lowering(db: &salsa::DatabaseImpl) {
        // Test: fn(x) { x + 1 }
        let path = PathId::new(db, "test.trb".to_owned());
        let int_ty = AstType::new(db, TypeKind::Int);

        let local_x = LocalId::new(0);

        // Lambda body: x + 1
        let x_ref = TypedRef::new(ResolvedRef::local(local_x, Symbol::new("x")), int_ty);
        let body = binop_expr(BinOpKind::Add, var_expr(x_ref), nat_lit_expr(1));

        // Lambda: fn(x) { x + 1 }
        let lambda = Expr::new(
            fresh_node_id(),
            ExprKind::Lambda {
                params: vec![Param {
                    id: fresh_node_id(),
                    name: Symbol::new("x"),
                    ty: None,
                    local_id: Some(local_x),
                }],
                body,
            },
        );

        // Function that returns the lambda
        let func_decl = simple_func(Symbol::new("make_adder"), lambda);

        let module = simple_module(vec![Decl::Function(func_decl)]);

        let ir_module = test_lower(db, path, SpanMap::default(), module);
        let ops = get_module_ops(db, &ir_module);

        // Should have the original function
        let func_ops: Vec<_> = ops.iter().filter(|op| op.name(db) == "func").collect();
        assert!(
            func_ops.len() >= 2,
            "Should have at least 2 func operations (lifted lambda + original)"
        );

        // Find the make_adder function (not the lifted lambda)
        let make_adder = func_ops.iter().find(|op| {
            if let Ok(f) = func::Func::from_operation(db, ***op) {
                f.sym_name(db).last_segment() == "make_adder"
            } else {
                false
            }
        });
        assert!(make_adder.is_some(), "Should have make_adder function");

        let func_typed = func::Func::from_operation(db, **make_adder.unwrap()).unwrap();
        let body_ops = get_func_body_ops(db, &func_typed);

        // Should have a closure.new operation (not tribute.lambda)
        let closure_op = body_ops.iter().find(|op| op.name(db) == "new");
        assert!(
            closure_op.is_some(),
            "Should emit closure.new operation for lambda expression"
        );

        // Check that a lifted lambda function exists
        let lifted = func_ops.iter().find(|op| {
            if let Ok(f) = func::Func::from_operation(db, ***op) {
                f.sym_name(db)
                    .last_segment()
                    .with_str(|s| s.starts_with("__lambda_"))
            } else {
                false
            }
        });
        assert!(lifted.is_some(), "Should have lifted __lambda_ function");
    }

    #[salsa_test]
    fn test_lambda_with_multiple_params(db: &salsa::DatabaseImpl) {
        // Test: fn(a, b, c) { a + b + c }
        let path = PathId::new(db, "test.trb".to_owned());
        let int_ty = AstType::new(db, TypeKind::Int);

        let local_a = LocalId::new(0);
        let local_b = LocalId::new(1);
        let local_c = LocalId::new(2);

        // Lambda body: (a + b) + c
        let a_ref = TypedRef::new(ResolvedRef::local(local_a, Symbol::new("a")), int_ty);
        let b_ref = TypedRef::new(ResolvedRef::local(local_b, Symbol::new("b")), int_ty);
        let c_ref = TypedRef::new(ResolvedRef::local(local_c, Symbol::new("c")), int_ty);

        let add_ab = binop_expr(BinOpKind::Add, var_expr(a_ref), var_expr(b_ref));
        let body = binop_expr(BinOpKind::Add, add_ab, var_expr(c_ref));

        // Lambda: fn(a, b, c) { a + b + c }
        let lambda = Expr::new(
            fresh_node_id(),
            ExprKind::Lambda {
                params: vec![
                    Param {
                        id: fresh_node_id(),
                        name: Symbol::new("a"),
                        ty: None,
                        local_id: Some(local_a),
                    },
                    Param {
                        id: fresh_node_id(),
                        name: Symbol::new("b"),
                        ty: None,
                        local_id: Some(local_b),
                    },
                    Param {
                        id: fresh_node_id(),
                        name: Symbol::new("c"),
                        ty: None,
                        local_id: Some(local_c),
                    },
                ],
                body,
            },
        );

        // Function that returns the lambda
        let func_decl = simple_func(Symbol::new("triple_sum"), lambda);

        let module = simple_module(vec![Decl::Function(func_decl)]);

        let ir_module = test_lower(db, path, SpanMap::default(), module);
        let ops = get_module_ops(db, &ir_module);

        // Find all func operations
        let func_ops: Vec<_> = ops.iter().filter(|op| op.name(db) == "func").collect();
        assert!(
            func_ops.len() >= 2,
            "Should have at least 2 func operations (lifted lambda + original)"
        );

        // Find the lifted lambda function
        let lifted = func_ops.iter().find(|op| {
            if let Ok(f) = func::Func::from_operation(db, ***op) {
                f.sym_name(db)
                    .last_segment()
                    .with_str(|s| s.starts_with("__lambda_"))
            } else {
                false
            }
        });
        assert!(lifted.is_some(), "Should have lifted __lambda_ function");

        let lifted_func = func::Func::from_operation(db, **lifted.unwrap()).unwrap();
        let body = lifted_func.body(db);
        let entry_block = body.blocks(db)[0];

        // Check that the lifted function has 4 block arguments: env + 3 params
        let args = entry_block.args(db);
        assert_eq!(
            args.len(),
            4,
            "Lifted lambda with 3 params should have 4 block arguments (env + 3 params)"
        );
    }
}
