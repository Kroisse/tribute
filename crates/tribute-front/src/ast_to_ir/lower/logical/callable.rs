//! Callables and control: functions, calls, `become`, lambdas, handlers,
//! evidence plans, and row arguments.

use salsa::Accumulator;
use tribute_core::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use tribute_ir::dialect::tribute_control;
use tribute_ir::dialect::tribute_control::{OperationDeclaration, Row, RowBinder};
use trunk_ir::context::{BlockArgData, BlockData, IrContext, OperationDataBuilder, RegionData};
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{BlockRef, OpRef, TypeRef, ValueRef};
use trunk_ir::types::{Attribute, Location};
use trunk_ir::{Symbol, SymbolPath};

use crate::ast::{
    Decl, DeclEntry, Expr, ExprKind, FuncDecl, HandlerArm, HandlerKind, OpDeclKind, ResolvedRef,
    TypeKind, TypedRef,
};
use crate::ast_to_ir::context::IrLoweringCtx;
use crate::ast_to_ir::lower::{FuncSignature, IrBuilder};

use super::declarations::{Declarations, PerformMetadata, ensure_prelude_declaration};
use super::nominal::{expr_type, expr_type_for_id, function_signature};
use super::values::{bind_pattern, lower_expr, lower_local_reference};
use super::{local_callables, op, result};

/// Whether lowering ended `block` with a source `become`. Such a block has
/// already left its callable and takes no `return` or `yield`.
pub(super) fn ends_with_tail_call(ir: &IrContext, block: BlockRef) -> bool {
    ir.block(block).ops.last().is_some_and(|&op| {
        tribute_control::TailCall::matches(ir, op)
            || tribute_control::TailCallIndirect::matches(ir, op)
    })
}

/// Attach the typechecked evidence selection of the source node `node`.
pub(super) fn attach_evidence_plan<'db>(
    ctx: &IrLoweringCtx<'db>,
    ir: &mut IrContext,
    op: OpRef,
    node: crate::ast::NodeId,
    declarations: &mut Declarations<'db>,
) {
    let Some(plan) = declarations.evidence_plans.get(&node).cloned() else {
        return;
    };
    let steps: Vec<_> = plan
        .iter()
        .map(|step| lower_evidence_step(ctx, step, declarations))
        .collect();
    if let Some(plan) = tribute_control::EvidenceStep::plan_attribute(ir, steps) {
        ir.op_attrs_mut(op)
            .insert(tribute_control::EVIDENCE_PLAN_ATTR, plan);
    }
}

pub(super) fn lower_evidence_step<'db>(
    ctx: &IrLoweringCtx<'db>,
    step: &crate::typeck::EvidenceStep<'db>,
    declarations: &mut Declarations<'db>,
) -> tribute_control::EvidenceStep {
    use crate::typeck::EvidenceStep;
    let mut symbol_of = |instance: &crate::ast::AbilityInstance<'db>| {
        let symbol = ctx.ability_symbol(&instance.ability_id.qualified(ctx.db), &instance.args);
        declarations.instance(&symbol, instance.ability_id, &instance.args);
        symbol
    };
    match step {
        EvidenceStep::Mask(instance) => tribute_control::EvidenceStep::Mask(symbol_of(instance)),
        EvidenceStep::Dup(instance) => tribute_control::EvidenceStep::Dup(symbol_of(instance)),
        EvidenceStep::Push(instance) => tribute_control::EvidenceStep::Push(symbol_of(instance)),
        EvidenceStep::Select(index) => tribute_control::EvidenceStep::Select(*index),
        EvidenceStep::Tails(plans) => tribute_control::EvidenceStep::Tails(
            plans
                .iter()
                .map(|plan| {
                    plan.iter()
                        .map(|step| lower_evidence_step(ctx, step, declarations))
                        .collect()
                })
                .collect(),
        ),
    }
}

/// Lower a named function value at its exact source-logical callable contract.
///
/// A `func_ref` may take a callable type whose row includes its target's, but
/// it must preserve the source parameter and result types. This is the only
/// callable adaptation accepted at the source-logical boundary: a generic
/// `unrealized_conversion_cast` cannot change a callable ABI.
pub(super) fn lower_function_ref<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    name: SymbolPath,
    reference: RowArguments,
    declarations: &Declarations<'db>,
    expected_ty: Option<TypeRef>,
) -> ValueRef {
    let signature = FuncSignature::lookup_logical(builder.ctx, builder.ir, &name)
        .unwrap_or_else(|| panic!("missing logical signature for function reference {name}"));
    ensure_prelude_declaration(builder, location, &name, &signature);
    let args = reference.rows(builder.ctx, declarations, &name, &signature);
    let worker_ty = signature
        .instantiate(builder.ir, &args)
        .callable_type(builder.ir);
    let ty = expected_ty
        .filter(|expected_ty| {
            let Some(expected) = tribute_control::FuncSig::from_type_ref(builder.ir, *expected_ty)
            else {
                return false;
            };
            let worker = tribute_control::FuncSig::from_type_ref(builder.ir, worker_ty)
                .expect("worker function reference must have a logical callable type");
            expected.result(builder.ir) == worker.result(builder.ir)
                && expected.inputs(builder.ir) == worker.inputs(builder.ir)
                && expected.row(builder.ir).includes(&worker.row(builder.ir))
        })
        .unwrap_or(worker_ty);
    let symbol = builder.ctx.function_symbol(&name);
    let row_args = FuncSignature::row_args_attribute(builder.ir, &args);
    let op = op(
        builder.ir,
        builder.block,
        location,
        tribute_control::FuncRef::DEF,
        |builder| {
            let builder = builder
                .result(ty)
                .attr("func_ref", Attribute::SymbolRef(SymbolPath::new([symbol])));
            match row_args {
                Some(row_args) => builder.attr(tribute_control::ROW_ARGS_ATTR, row_args),
                None => builder,
            }
        },
    );
    builder.ctx.record_function_ref(op, name);
    result(builder.ir, op)
}

/// Where a reference finds the row arguments it passes to its callee.
#[derive(Clone)]
pub(super) enum RowArguments {
    /// The function instance type checking selected for this node.
    Instance(crate::ast::NodeId),
    /// These rows, of a reference already lowered.
    Rows(Vec<Row>),
}

impl RowArguments {
    /// The rows a reference to `name` passes, in the current definition's
    /// binders: one for each binder `signature` quantifies.
    pub(super) fn rows<'db>(
        self,
        ctx: &IrLoweringCtx<'db>,
        declarations: &Declarations<'db>,
        name: &SymbolPath,
        signature: &FuncSignature,
    ) -> Vec<Row> {
        let quantified = tribute_control::quantified_count(&signature.binders);
        let rows = match self {
            Self::Rows(rows) => rows,
            Self::Instance(_) if quantified == 0 => vec![],
            Self::Instance(node) => declarations
                .function_instances
                .get(&node)
                .unwrap_or_else(|| panic!("reference to {name} selects no function instance"))
                .row_arguments
                .iter()
                .map(|row| ctx.ability_row(*row))
                .collect(),
        };
        assert_eq!(
            rows.len(),
            quantified,
            "a reference to {name} passes one row for each quantified binder"
        );
        rows
    }
}

/// Lower a named function value under a callable parameter's exact contract.
///
/// This deliberately creates a fresh `func_ref`; a source value may be used
/// under more than one callable type in the same function.
pub(super) fn lower_expr_for_callable_parameter<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    expr: Expr<TypedRef<'db>>,
    expected_ty: TypeRef,
    declarations: &mut Declarations<'db>,
) -> Option<ValueRef> {
    match &*expr.kind {
        ExprKind::Var(reference) => {
            if let ResolvedRef::Local { id, .. } = reference.resolved {
                if local_callables::lookup(builder.ctx, expr.id, declarations).is_some() {
                    return lower_local_reference(builder, expr.id, id, declarations);
                }
                if let Some(value) = builder.ctx.lookup(id)
                    && let trunk_ir::refs::ValueDef::OpResult(producer, _) =
                        builder.ir.value_def(value)
                    && let Some(name) = builder.ctx.function_ref_path(producer).cloned()
                {
                    let rows = tribute_control::row_args(builder.ir, producer)
                        .expect("a lowered func_ref has well-formed row arguments");
                    return Some(lower_function_ref(
                        builder,
                        builder.location(expr.id),
                        name,
                        RowArguments::Rows(rows),
                        declarations,
                        Some(expected_ty),
                    ));
                }
                return lower_expr(builder, expr, declarations);
            }
            let ResolvedRef::Function { id } = reference.resolved else {
                return lower_expr(builder, expr, declarations);
            };
            tribute_control::FuncSig::from_type_ref(builder.ir, expected_ty)
                .map(|_| {
                    lower_function_ref(
                        builder,
                        builder.location(expr.id),
                        id.path(builder.ctx.db).clone(),
                        RowArguments::Instance(expr.id),
                        declarations,
                        Some(expected_ty),
                    )
                })
                .or_else(|| lower_expr(builder, expr, declarations))
        }
        ExprKind::Lambda { params, body } => {
            let signature = declarations
                .lambda_signatures
                .get(&expr.id)
                .cloned()
                .unwrap_or_else(|| panic!("missing solved logical lambda signature"));
            lower_lambda(
                builder,
                builder.location(expr.id),
                signature,
                params.clone(),
                body.clone(),
                declarations,
                Some(expected_ty),
            )
        }
        _ => lower_expr(builder, expr, declarations),
    }
}

/// Each use of a let-bound lambda in `function`.
pub(super) fn local_instantiations<'db>(
    db: &'db dyn salsa::Database,
    function: &FuncDecl<TypedRef<'db>>,
    declarations: &Declarations<'db>,
) -> Vec<crate::monomorphize::LocalInstantiation<'db>> {
    crate::monomorphize::local_instantiations(
        db,
        function,
        |node| declarations.local_instances.get(node),
        |node| declarations.lambda_signatures.get(node),
    )
}

/// Number the row variables of each function definition.
pub(super) fn prescan_row_binders<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    entries: &[DeclEntry<TypedRef<'db>>],
    declarations: &Declarations<'db>,
) {
    for entry in entries {
        let Decl::Function(function) = &entry.decl else {
            continue;
        };
        let name = entry.module.child(function.name.clone());
        let Some(scheme) = ctx.lookup_function_type(&name).copied() else {
            continue;
        };
        // A let-bound lambda, and each copy of it, quantifies the rows its
        // own type spells.
        let mut local = Vec::new();
        for (variable, _) in local_instantiations(ctx.db, function, declarations)
            .into_iter()
            .flat_map(|use_| use_.rows)
        {
            if !local.contains(&variable) {
                local.push(variable);
            }
        }
        let body = crate::ast_to_ir::row_binders::BodyTypes {
            node_types: ctx.node_types(),
            handler_operations: &declarations.handler_operations,
            perform_operations: &declarations.perform_operations,
        };
        let binders =
            crate::ast_to_ir::row_binders::of_definition(ctx.db, scheme, function, &body, &local);
        ctx.register_row_binders(name, binders);
    }
}

/// Every ability instance a row of the module names, in first-occurrence
/// order.
pub(super) fn row_effects<'db>(
    ctx: &IrLoweringCtx<'db>,
    entries: &[DeclEntry<TypedRef<'db>>],
    declarations: &Declarations<'db>,
) -> Vec<crate::ast::AbilityInstance<'db>> {
    fn visit<'db>(
        db: &'db dyn salsa::Database,
        ty: crate::ast::Type<'db>,
        found: &mut Vec<crate::ast::AbilityInstance<'db>>,
    ) {
        let row = |effect: crate::ast::AbilityRow<'db>, found: &mut Vec<_>| {
            for ability in effect.instances(db) {
                for argument in &ability.args {
                    visit(db, *argument, found);
                }
                if !found.contains(ability) {
                    found.push(ability.clone());
                }
            }
        };
        match ty.kind(db) {
            TypeKind::Named { args, .. } | TypeKind::Tuple(args) => {
                for arg in args {
                    visit(db, *arg, found);
                }
            }
            TypeKind::App { ctor, args } => {
                visit(db, *ctor, found);
                for arg in args {
                    visit(db, *arg, found);
                }
            }
            TypeKind::Func {
                params,
                result,
                effect,
            } => {
                for param in params {
                    visit(db, *param, found);
                }
                visit(db, *result, found);
                row(*effect, found);
            }
            TypeKind::Continuation {
                arg,
                result,
                effect,
            } => {
                visit(db, *arg, found);
                visit(db, *result, found);
                row(*effect, found);
            }
            _ => {}
        }
    }
    let db = ctx.db;
    let mut found = Vec::new();
    for entry in entries {
        if let Decl::Function(function) = &entry.decl
            && let Some(scheme) =
                ctx.lookup_function_type(&entry.module.child(function.name.clone()))
        {
            visit(db, scheme.body(db), &mut found);
        }
    }
    for ty in ctx.node_types().values() {
        visit(db, *ty, &mut found);
    }
    for instance in declarations.function_instances.values() {
        visit(db, instance.callable, &mut found);
    }
    found
}

/// Plan the local callables of every function. A plan gives a pure use of a
/// lambda its own instance where it can; a use it leaves alone reads the
/// lambda as it was written.
pub(super) fn plan_local_callables<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    entries: &[DeclEntry<TypedRef<'db>>],
    declarations: &mut Declarations<'db>,
) {
    for entry in entries {
        let prefix = &entry.module;
        let Decl::Function(function) = &entry.decl else {
            continue;
        };
        let name = prefix.child(function.name.clone());
        let Some(scheme) = ctx.lookup_function_type(&name).copied() else {
            continue;
        };
        let outer = ctx.enter_definition(&name);
        let plan = local_callables::Plan::collect(
            ctx,
            ir,
            &function.body,
            declarations,
            scheme.type_params(ctx.db).len(),
        );
        ctx.leave_definition(outer);
        declarations
            .function_local_callables
            .insert(function.id, plan);
    }
}

pub(super) fn lower_function<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    top: BlockRef,
    function: FuncDecl<TypedRef<'db>>,
    declarations: &mut Declarations<'db>,
) {
    let location = ctx.location(function.id);
    let parent_type_parameters = ctx
        .lookup_function_type(&ctx.qualify_name(&function.name))
        .expect("function has a typechecked signature")
        .type_params(ctx.db)
        .len();
    let signature = function_signature(ctx, ir, &function);
    let callable = signature.callable_type(ir);
    let inferred_row = function.effects.is_none()
        || crate::is_root_main(&function.name, ctx.module_path().is_root());
    let args = function
        .params
        .iter()
        .zip(signature.param_types.iter())
        .map(|(parameter, ty)| {
            let mut attrs = trunk_ir::types::AttributeMap::default();
            let name = parameter.name.with_str(|name| ir.string_attr(name));
            attrs.view_mut(ir).insert("bind_name", name);
            BlockArgData { ty: *ty, attrs }
        })
        .collect();
    let entry = ir.create_block(BlockData {
        location,
        args,
        ops: Default::default(),
        parent_region: None,
    });
    {
        let mut scope = ctx.scope();
        for (index, parameter) in function.params.iter().enumerate() {
            if let Some(id) = parameter.local_id {
                scope.bind(
                    id,
                    parameter.name.clone(),
                    ir.block_arg(entry, index as u32),
                );
            }
        }
        declarations.local_uses = local_instantiations(scope.db, &function, declarations);
        declarations.local_callables = declarations
            .function_local_callables
            .remove(&function.id)
            .unwrap_or_else(|| {
                local_callables::Plan::collect(
                    &mut scope,
                    ir,
                    &function.body,
                    declarations,
                    parent_type_parameters,
                )
            });
        let value = lower_expr(
            &mut IrBuilder::new(&mut scope, ir, entry),
            function.body,
            declarations,
        );
        if value.is_some() || !ends_with_tail_call(ir, entry) {
            let value = value.expect("typechecked source expression failed logical IR lowering");
            let value = IrBuilder::new(&mut scope, ir, entry).cast_if_needed(
                location,
                value,
                signature.return_type,
            );
            op(
                ir,
                entry,
                location,
                tribute_control::Return::DEF,
                |builder| builder.operand(value),
            );
        }
    }
    let body = ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![entry],
        parent_op: None,
    });
    let name = ctx.qualify_name(&function.name).to_symbol();
    let function = tribute_control::func_declaration(ir, location, &name, callable);
    signature.attach_rows(ir, function.op_ref());
    if inferred_row {
        ir.op_attrs_mut(function.op_ref())
            .insert(tribute_control::INFERRED_ROW_ATTR, Attribute::Bool(true));
    }
    ir.push_op_region(function.op_ref(), body);
    ir.push_op(top, function.op_ref());
}

pub(super) fn operation_kind_text(kind: OpDeclKind) -> &'static str {
    match kind {
        OpDeclKind::Fn => "fn",
        OpDeclKind::Op => "op",
    }
}

pub(super) fn call_operation_metadata<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    declarations: &Declarations<'db>,
    metadata: PerformMetadata<'_, 'db>,
) -> (SymbolPath, OperationDeclaration) {
    let PerformMetadata {
        ability,
        operation,
        kind,
        semantic,
    } = metadata;
    let schema = declarations
        .schemas
        .get(&ability)
        .unwrap_or_else(|| panic!("missing resolved ability schema for perform {operation}"));
    let operation_schema = schema
        .operations
        .get(&operation)
        .unwrap_or_else(|| panic!("missing resolved operation schema for perform {operation}"));
    if semantic.ability != ability || semantic.kind != kind || operation_schema.kind != kind {
        panic!("handler operation kind disagrees with resolved schema");
    }
    if semantic.params.len() != operation_schema.param_types.len() {
        panic!("perform argument arity disagrees with resolved operation schema");
    }
    let args = semantic.ability_args.clone();
    if args.len() != schema.type_params.len() {
        panic!("typed ability operation effect has wrong type argument arity");
    }
    let expected_params = operation_schema
        .param_types
        .iter()
        .map(
            |ty| match crate::typeck::subst::substitute_bound_vars(builder.db(), *ty, &args) {
                crate::typeck::subst::SubstResult::Ok(ty) => Some(ty),
                crate::typeck::subst::SubstResult::OutOfBounds { .. } => None,
            },
        )
        .collect::<Option<Vec<_>>>()
        .unwrap_or_else(|| panic!("perform parameter substitution failed"));
    let expected_result = match crate::typeck::subst::substitute_bound_vars(
        builder.db(),
        operation_schema.return_type,
        &args,
    ) {
        crate::typeck::subst::SubstResult::Ok(ty) => ty,
        crate::typeck::subst::SubstResult::OutOfBounds { .. } => {
            panic!("perform result substitution failed")
        }
    };
    if expected_params != semantic.params || expected_result != semantic.result {
        panic!(
            "typed ability operation call disagrees with resolved operation schema: expected params={expected_params:?}, result={expected_result:?}; actual params={:?}, result={:?}",
            semantic.params, semantic.result
        );
    }
    let instance = builder
        .ctx
        .ability_symbol(&ability.qualified(builder.db()), &args);
    let parameters = semantic
        .params
        .iter()
        .map(|param| builder.ctx.convert_logical_type(builder.ir, *param))
        .collect::<Vec<_>>();
    let result = builder
        .ctx
        .convert_logical_type(builder.ir, semantic.result);
    (
        instance.clone(),
        OperationDeclaration::new(
            instance,
            builder.ir.intern_symbol_text(&operation),
            builder.ir.intern_str(operation_kind_text(kind)),
            parameters,
            result,
        ),
    )
}

pub(super) fn adapt_operation_arguments(
    builder: &mut IrBuilder<'_, '_>,
    location: Location,
    values: Vec<ValueRef>,
    parameter_types: &[TypeRef],
) -> Option<Vec<ValueRef>> {
    if values.len() != parameter_types.len() {
        return None;
    }
    Some(
        values
            .into_iter()
            .zip(parameter_types.iter().copied())
            .map(|(value, ty)| builder.cast_if_needed(location, value, ty))
            .collect(),
    )
}

pub(super) fn adapt_operation_arguments_or_recover<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    values: Vec<ValueRef>,
    declaration: &OperationDeclaration,
    ability: &Symbol,
    operation: &Symbol,
) -> Result<Vec<ValueRef>, ValueRef> {
    let argument_count = values.len();
    match adapt_operation_arguments(builder, location, values, &declaration.parameter_types) {
        Some(values) => Ok(values),
        None => {
            Diagnostic::new(
                format!(
                    "ability operation `{}::{}` has {} arguments, expected {}",
                    ability,
                    operation,
                    argument_count,
                    declaration.parameter_types.len()
                ),
                location.span,
                DiagnosticSeverity::Error,
                CompilationPhase::Lowering,
            )
            .accumulate(builder.db());
            Err(builder.emit_nil(location))
        }
    }
}

/// Call the named source function, casting each argument to its parameter
/// type. Returns the call's result at the callee's logical result type.
pub(in crate::ast_to_ir::lower) fn emit_named_call(
    builder: &mut IrBuilder<'_, '_>,
    location: Location,
    name: SymbolPath,
    values: Vec<ValueRef>,
) -> ValueRef {
    let signature = FuncSignature::lookup_logical(builder.ctx, builder.ir, &name)
        .unwrap_or_else(|| panic!("missing logical signature for call {name}"));
    // A call the compiler adds performs nothing: each row it passes is empty.
    let args = vec![Row::default(); tribute_control::quantified_count(&signature.binders)];
    let call = named_call(
        builder,
        location,
        &name,
        &signature,
        &args,
        values,
        CallPosition::Value,
    );
    result(builder.ir, call)
}

/// Whether a source call produces a value or is the operand of `become`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum CallPosition {
    Value,
    Tail,
}

/// Call `name`, whose signature is `declared` in its own binders, passing
/// the row arguments `args`.
pub(super) fn named_call(
    builder: &mut IrBuilder<'_, '_>,
    location: Location,
    name: &SymbolPath,
    declared: &FuncSignature,
    args: &[Row],
    values: Vec<ValueRef>,
    position: CallPosition,
) -> OpRef {
    ensure_prelude_declaration(builder, location, name, declared);
    let signature = declared.instantiate(builder.ir, args);
    let row_args = FuncSignature::row_args_attribute(builder.ir, args);
    if values.len() != signature.param_types.len() {
        panic!("typechecked call arity disagrees with logical signature for {name}");
    }
    let values: Vec<_> = values
        .into_iter()
        .zip(signature.param_types.iter().copied())
        .map(|(value, ty)| builder.cast_if_needed(location, value, ty))
        .collect();
    let symbol = builder.ctx.function_symbol(name);
    let callee = Attribute::SymbolRef(SymbolPath::new([symbol]));
    let with_rows = |builder: OperationDataBuilder| match row_args {
        Some(row_args) => builder.attr(tribute_control::ROW_ARGS_ATTR, row_args),
        None => builder,
    };
    match position {
        CallPosition::Value => op(
            builder.ir,
            builder.block,
            location,
            tribute_control::Call::DEF,
            |builder| {
                with_rows(
                    builder
                        .operands(values)
                        .result(signature.return_type)
                        .attr("callee", callee),
                )
            },
        ),
        CallPosition::Tail => op(
            builder.ir,
            builder.block,
            location,
            tribute_control::TailCall::DEF,
            |builder| with_rows(builder.operands(values).attr("callee", callee)),
        ),
    }
}

/// Emit an indirect call of `callable`, or end the block with a tail call.
pub(super) fn indirect_call(
    builder: &mut IrBuilder<'_, '_>,
    location: Location,
    callable: ValueRef,
    values: Vec<ValueRef>,
    position: CallPosition,
) -> OpRef {
    let callable_ty = builder.ir.value_ty(callable);
    let callable_signature = tribute_control::FuncSig::from_type_ref(builder.ir, callable_ty)
        .unwrap_or_else(|| panic!("indirect call target is not a logical callable"));
    let parameters = callable_signature.inputs(builder.ir).to_vec();
    if values.len() != parameters.len() {
        panic!("typechecked indirect call arity disagrees with logical callable");
    }
    let values: Vec<_> = values
        .into_iter()
        .zip(parameters.iter().copied())
        .map(|(value, ty)| builder.cast_if_needed(location, value, ty))
        .collect();
    match position {
        CallPosition::Value => {
            let call_result = callable_signature.result(builder.ir);
            op(
                builder.ir,
                builder.block,
                location,
                tribute_control::CallIndirect::DEF,
                |builder| {
                    builder
                        .operand(callable)
                        .operands(values)
                        .result(call_result)
                },
            )
        }
        CallPosition::Tail => op(
            builder.ir,
            builder.block,
            location,
            tribute_control::TailCallIndirect::DEF,
            |builder| builder.operand(callable).operands(values),
        ),
    }
}

/// Lower the operand of `become`. The tail call ends the current block, so
/// no value continues from it.
pub(super) fn lower_become<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    call: Expr<TypedRef<'db>>,
    declarations: &mut Declarations<'db>,
) -> Option<ValueRef> {
    let ExprKind::Call { callee, args, .. } = *call.kind else {
        // Typechecking has reported any other operand.
        return Some(builder.emit_nil(location));
    };
    let result_ty = expr_type_for_id(builder, call.id);
    lower_call(
        builder,
        location,
        call.id,
        result_ty,
        callee,
        args,
        declarations,
        CallPosition::Tail,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn lower_call<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    call_id: crate::ast::NodeId,
    result_ty: TypeRef,
    callee: Expr<TypedRef<'db>>,
    args: Vec<Expr<TypedRef<'db>>>,
    declarations: &mut Declarations<'db>,
    position: CallPosition,
) -> Option<ValueRef> {
    // A resolved variable callee is atomic. Any other callee expression is a
    // strict child and must be evaluated before the argument list.
    let indirect_callee = if matches!(&*callee.kind, ExprKind::Var(_)) {
        None
    } else {
        Some(lower_expr(builder, callee.clone(), declarations)?)
    };
    let named_signature = match &*callee.kind {
        ExprKind::Var(reference) => match reference.resolved {
            ResolvedRef::Function { id } => {
                let name = id.path(builder.ctx.db);
                let declared = FuncSignature::lookup_logical(builder.ctx, builder.ir, name)
                    .unwrap_or_else(|| panic!("missing logical signature for call {name}"));
                let args = RowArguments::Instance(callee.id).rows(
                    builder.ctx,
                    declarations,
                    name,
                    &declared,
                );
                let signature = declared.instantiate(builder.ir, &args);
                Some((declared, args, signature))
            }
            _ => None,
        },
        _ => None,
    };
    let callable_value = indirect_callee.or_else(|| match &*callee.kind {
        ExprKind::Var(reference) => match reference.resolved {
            ResolvedRef::Local { id, .. } => {
                lower_local_reference(builder, callee.id, id, declarations)
            }
            _ => None,
        },
        _ => None,
    });
    let parameter_types = named_signature
        .as_ref()
        .map(|(_, _, signature)| signature.param_types.clone())
        .or_else(|| {
            let value = callable_value?;
            let signature =
                tribute_control::FuncSig::from_type_ref(builder.ir, builder.ir.value_ty(value))?;
            Some(signature.inputs(builder.ir).to_vec())
        });
    let mut values = args
        .into_iter()
        .enumerate()
        .map(|(index, arg)| {
            parameter_types
                .as_ref()
                .and_then(|parameters| parameters.get(index).copied())
                .map(|expected_ty| {
                    lower_expr_for_callable_parameter(
                        builder,
                        arg.clone(),
                        expected_ty,
                        declarations,
                    )
                })
                .unwrap_or_else(|| lower_expr(builder, arg, declarations))
        })
        .collect::<Option<Vec<_>>>()?;
    if let ExprKind::Var(reference) = *callee.kind {
        match reference.resolved {
            ResolvedRef::AbilityOp {
                ability,
                op: operation,
                kind,
            } => {
                let semantic = declarations
                    .perform_operations
                    .get(&call_id)
                    .unwrap_or_else(|| {
                        panic!("missing exact typed metadata for ability operation call")
                    });
                let ability_arguments = semantic.ability_args.clone();
                let (instance, declaration) = call_operation_metadata(
                    builder,
                    declarations,
                    PerformMetadata {
                        ability,
                        operation: operation.clone(),
                        kind,
                        semantic,
                    },
                );
                values = match adapt_operation_arguments_or_recover(
                    builder,
                    location,
                    values,
                    &declaration,
                    &ability.qualified(builder.db()),
                    &operation,
                ) {
                    Ok(values) => values,
                    Err(nil) => return Some(nil),
                };
                let perform = op(
                    builder.ir,
                    builder.block,
                    location,
                    tribute_control::Perform::DEF,
                    |builder| {
                        builder
                            .operands(values)
                            .result(declaration.result_type)
                            .attr("ability", Attribute::SymbolRef(instance))
                            .attr("op_name", Attribute::String(declaration.op_name))
                            .attr("operation_kind", Attribute::String(declaration.kind))
                    },
                );
                declarations.record(
                    builder.ir,
                    declaration,
                    ability,
                    &ability_arguments,
                    location,
                    builder.db(),
                );
                let value = result(builder.ir, perform);
                Some(builder.cast_if_needed(location, value, result_ty))
            }
            ResolvedRef::Function { id } => {
                let name = id.path(builder.ctx.db);
                let (declared, args, _) = named_signature
                    .as_ref()
                    .expect("a named callee has a signature");
                let call = named_call(builder, location, name, declared, args, values, position);
                finish_call(
                    builder,
                    location,
                    call,
                    call_id,
                    result_ty,
                    declarations,
                    position,
                )
            }
            ResolvedRef::Local { id, .. } => {
                let callable = callable_value
                    .unwrap_or_else(|| panic!("missing logical callable binding for local {id:?}"));
                let call = indirect_call(builder, location, callable, values, position);
                finish_call(
                    builder,
                    location,
                    call,
                    call_id,
                    result_ty,
                    declarations,
                    position,
                )
            }
            _ => panic!("unsupported call target at source-logical boundary"),
        }
    } else {
        let callable = indirect_callee
            .expect("non-variable logical callee must be evaluated before its arguments");
        let call = indirect_call(builder, location, callable, values, position);
        finish_call(
            builder,
            location,
            call,
            call_id,
            result_ty,
            declarations,
            position,
        )
    }
}

/// Attach a source call's evidence selection and produce its value, if it
/// has one.
pub(super) fn finish_call<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    call: OpRef,
    call_id: crate::ast::NodeId,
    result_ty: TypeRef,
    declarations: &mut Declarations<'db>,
    position: CallPosition,
) -> Option<ValueRef> {
    attach_evidence_plan(builder.ctx, builder.ir, call, call_id, declarations);
    match position {
        CallPosition::Value => {
            let value = result(builder.ir, call);
            Some(builder.cast_if_needed(location, value, result_ty))
        }
        CallPosition::Tail => None,
    }
}

pub(super) fn lower_lambda<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    signature: crate::typeck::LambdaSignature<'db>,
    params: Vec<crate::ast::Param>,
    body: Expr<TypedRef<'db>>,
    declarations: &mut Declarations<'db>,
    expected_ty: Option<TypeRef>,
) -> Option<ValueRef> {
    let TypeKind::Func {
        params: signature_params,
        result: signature_result,
        effect,
    } = signature.function_type.kind(builder.db())
    else {
        panic!("solved logical lambda signature is not a function type");
    };
    let own_row = builder.ctx.ability_row(*effect);
    let param_types = signature_params
        .iter()
        .map(|ty| builder.ctx.convert_logical_type(builder.ir, *ty))
        .collect::<Vec<_>>();
    let result_type = builder
        .ctx
        .convert_logical_type(builder.ir, *signature_result);
    // The lambda takes the expected callable type when that type's row
    // requires at least the class of its own: the row includes its own, or
    // its tail is a binder whose class is always `Cps`.
    let row = expected_ty
        .and_then(|expected_ty| {
            let expected = tribute_control::FuncSig::from_type_ref(builder.ir, expected_ty)?;
            let row = expected.row(builder.ir);
            let always_cps = row.tail.is_some_and(|tail| {
                usize::try_from(tail)
                    .ok()
                    .and_then(|tail| builder.ctx.current_row_binders().get(tail))
                    .is_some_and(|(_, binder)| {
                        matches!(binder, RowBinder::Body | RowBinder::AbilityArg)
                    })
            });
            (expected.result(builder.ir) == result_type
                && expected.inputs(builder.ir) == param_types
                && (always_cps || row.includes(&own_row)))
            .then_some(row)
        })
        .unwrap_or(own_row);
    let entry = builder.ir.create_block(BlockData {
        location,
        args: param_types
            .iter()
            .map(|ty| BlockArgData {
                ty: *ty,
                attrs: Default::default(),
            })
            .collect(),
        ops: Default::default(),
        parent_region: None,
    });
    {
        let mut scope = builder.ctx.scope();
        for (index, parameter) in params.iter().enumerate() {
            if let Some(id) = parameter.local_id {
                scope.bind(
                    id,
                    parameter.name.clone(),
                    builder.ir.block_arg(entry, index as u32),
                );
            }
        }
        let value = lower_expr(
            &mut IrBuilder::new(&mut scope, builder.ir, entry),
            body,
            declarations,
        );
        if value.is_some() || !ends_with_tail_call(builder.ir, entry) {
            let value = IrBuilder::new(&mut scope, builder.ir, entry).cast_if_needed(
                location,
                value?,
                result_type,
            );
            op(
                builder.ir,
                entry,
                location,
                tribute_control::Return::DEF,
                |builder| builder.operand(value),
            );
        }
    }
    let region = builder.ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![entry],
        parent_op: None,
    });
    let captures = local_callables::captures(builder.ctx, builder.ir, region);
    let callable = tribute_control::FuncSig::params(param_types, result_type)
        .row(row)
        .intern(builder.ir)
        .as_type_ref();
    let lambda = op(
        builder.ir,
        builder.block,
        location,
        tribute_control::Lambda::DEF,
        |builder| builder.operands(captures).result(callable).region(region),
    );
    Some(result(builder.ir, lambda))
}

pub(super) fn lower_handle<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    handle_id: crate::ast::NodeId,
    result_ty: TypeRef,
    body: Expr<TypedRef<'db>>,
    handlers: Vec<HandlerArm<TypedRef<'db>>>,
    declarations: &mut Declarations<'db>,
) -> Option<ValueRef> {
    let body_ty = expr_type(builder, &body);
    let body_block = builder.ir.create_block(BlockData {
        location,
        args: vec![],
        ops: Default::default(),
        parent_region: None,
    });
    let body_value = lower_expr(
        &mut IrBuilder::new(builder.ctx, builder.ir, body_block),
        body,
        declarations,
    )?;
    let body_value = IrBuilder::new(builder.ctx, builder.ir, body_block)
        .cast_if_needed(location, body_value, body_ty);
    let body_yield = op(
        builder.ir,
        body_block,
        location,
        tribute_control::Yield::DEF,
        |builder| builder.operand(body_value),
    );
    let _ = body_yield;
    let body_region = builder.ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![body_block],
        parent_op: None,
    });
    let completion_block = builder.ir.create_block(BlockData {
        location,
        args: vec![BlockArgData {
            ty: body_ty,
            attrs: Default::default(),
        }],
        ops: Default::default(),
        parent_region: None,
    });
    let complete = builder.ir.block_arg(completion_block, 0);
    let do_arm = handlers
        .iter()
        .find(|arm| matches!(arm.kind, HandlerKind::Do { .. }));
    let completion_value = if let Some(arm) = do_arm {
        let mut scope = builder.ctx.scope();
        if let HandlerKind::Do { binding } = &arm.kind {
            bind_pattern(
                &mut IrBuilder::new(&mut scope, builder.ir, completion_block),
                binding,
                complete,
            );
        }
        let value = lower_expr(
            &mut IrBuilder::new(&mut scope, builder.ir, completion_block),
            arm.body.clone(),
            declarations,
        )?;
        IrBuilder::new(&mut scope, builder.ir, completion_block)
            .cast_if_needed(location, value, result_ty)
    } else {
        complete
    };
    op(
        builder.ir,
        completion_block,
        location,
        tribute_control::Yield::DEF,
        |builder| builder.operand(completion_value),
    );
    let completion_region = builder.ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![completion_block],
        parent_op: None,
    });
    let handlers_block = builder.ir.create_block(BlockData {
        location,
        args: vec![],
        ops: Default::default(),
        parent_region: None,
    });
    for arm in handlers
        .into_iter()
        .filter(|arm| !matches!(arm.kind, HandlerKind::Do { .. }))
    {
        lower_handler(
            builder.ctx,
            builder.ir,
            handlers_block,
            location,
            result_ty,
            arm,
            declarations,
        )
        .expect("typechecked handler failed logical IR metadata lowering");
    }
    let handlers_region = builder.ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![handlers_block],
        parent_op: None,
    });
    let handle = op(
        builder.ir,
        builder.block,
        location,
        tribute_control::Handle::DEF,
        |builder| {
            builder
                .result(result_ty)
                .region(body_region)
                .region(completion_region)
                .region(handlers_region)
        },
    );
    attach_evidence_plan(builder.ctx, builder.ir, handle, handle_id, declarations);
    Some(result(builder.ir, handle))
}

pub(super) fn lower_handler<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    table: BlockRef,
    location: Location,
    answer_ty: TypeRef,
    arm: HandlerArm<TypedRef<'db>>,
    declarations: &mut Declarations<'db>,
) -> Option<()> {
    let (ability, operation, kind, params, resume) = match arm.kind {
        HandlerKind::Fn {
            ability,
            op,
            params,
        } => (ability, op, OpDeclKind::Fn, params, None),
        HandlerKind::Op {
            ability,
            op,
            params,
            resume_local_id,
        } => (ability, op, OpDeclKind::Op, params, resume_local_id),
        HandlerKind::Do { .. } => return Some(()),
    };
    let ability_id = match ability.resolved {
        ResolvedRef::Ability { id } => id,
        ResolvedRef::TypeDef { id } => {
            crate::ast::AbilityId::source(ctx.db, id.path(ctx.db).clone())
        }
        _ => panic!("handler ability did not resolve to an ability definition"),
    };
    let schema = declarations
        .schemas
        .get(&ability_id)
        .unwrap_or_else(|| panic!("missing resolved ability schema for handler {operation}"));
    let operation_schema = schema
        .operations
        .get(&operation)
        .unwrap_or_else(|| panic!("missing resolved operation schema for handler {operation}"));
    if operation_schema.kind != kind {
        panic!("handler operation kind disagrees with resolved schema");
    }
    let semantic = declarations
        .handler_operations
        .get(&arm.id)
        .unwrap_or_else(|| panic!("missing typed semantic operation for handler {operation}"));
    if semantic.ability != ability_id || semantic.kind != kind {
        panic!("typed handler operation identity disagrees with source handler arm");
    }
    let arguments = semantic.ability_args.clone();
    if arguments.len() != schema.type_params.len() {
        panic!("typed handler ability argument arity disagrees with resolved schema");
    }
    let expected_params = operation_schema
        .param_types
        .iter()
        .map(
            |ty| match crate::typeck::subst::substitute_bound_vars(ctx.db, *ty, &arguments) {
                crate::typeck::subst::SubstResult::Ok(ty) => ty,
                crate::typeck::subst::SubstResult::OutOfBounds { .. } => {
                    panic!("handler parameter substitution failed")
                }
            },
        )
        .collect::<Vec<_>>();
    let expected_result = match crate::typeck::subst::substitute_bound_vars(
        ctx.db,
        operation_schema.return_type,
        &arguments,
    ) {
        crate::typeck::subst::SubstResult::Ok(ty) => ty,
        crate::typeck::subst::SubstResult::OutOfBounds { .. } => {
            panic!("handler result substitution failed")
        }
    };
    if semantic.params != expected_params || semantic.result != expected_result {
        panic!("typed handler signature disagrees with resolved operation schema");
    }
    if params.len() != semantic.params.len() {
        panic!("handler parameter arity disagrees with typed semantic signature");
    }
    let instance = ctx.ability_symbol(&ability_id.qualified(ctx.db), &arguments);
    let parameter_types: Vec<_> = expected_params
        .into_iter()
        .map(|ty| ctx.convert_logical_type(ir, ty))
        .collect();
    let operation_result = ctx.convert_logical_type(ir, expected_result);
    let op_name = ir.intern_symbol_text(&operation);
    let kind_name = ir.intern_str(operation_kind_text(kind));
    let declaration = OperationDeclaration::new(
        instance.clone(),
        op_name,
        kind_name,
        parameter_types.clone(),
        operation_result,
    );
    declarations.record(ir, declaration, ability_id, &arguments, location, ctx.db);
    let is_never = trunk_ir::dialect::core::Never::matches(ir, operation_result);
    let mut block_args: Vec<_> = parameter_types
        .iter()
        .map(|ty| BlockArgData {
            ty: *ty,
            attrs: Default::default(),
        })
        .collect();
    if kind == OpDeclKind::Op && !is_never {
        let token =
            tribute_ir::dialect::tribute_control::ResumeToken::params(operation_result, answer_ty)
                .intern(ir)
                .as_type_ref();
        block_args.push(BlockArgData {
            ty: token,
            attrs: Default::default(),
        });
    }
    let block = ir.create_block(BlockData {
        location,
        args: block_args,
        ops: Default::default(),
        parent_region: None,
    });
    {
        let mut scope = ctx.scope();
        for (index, pattern) in params.iter().enumerate() {
            let value = ir.block_arg(block, index as u32);
            bind_pattern(&mut IrBuilder::new(&mut scope, ir, block), pattern, value);
        }
        if let (Some(id), true) = (resume, kind == OpDeclKind::Op && !is_never) {
            scope.bind_resume(
                id,
                Symbol::new("resume"),
                ir.block_arg(block, parameter_types.len() as u32),
            );
        }
        let value = lower_expr(
            &mut IrBuilder::new(&mut scope, ir, block),
            arm.body,
            declarations,
        )?;
        let expected = if kind == OpDeclKind::Fn {
            operation_result
        } else {
            answer_ty
        };
        let value = IrBuilder::new(&mut scope, ir, block).cast_if_needed(location, value, expected);
        op(
            ir,
            block,
            location,
            tribute_control::Yield::DEF,
            |builder| builder.operand(value),
        );
    }
    let region = ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![block],
        parent_op: None,
    });
    op(
        ir,
        table,
        location,
        tribute_control::Handler::DEF,
        |builder| {
            builder
                .attr("ability", Attribute::SymbolRef(instance))
                .attr("op_name", Attribute::String(op_name))
                .attr("kind", Attribute::String(kind_name))
                .attr("operation_result_type", Attribute::Type(operation_result))
                .region(region)
        },
    );
    Some(())
}
