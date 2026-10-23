//! Source-logical AST to TrunkIR lowering.
//!
//! It emits only the documented `tribute_control` boundary and ordinary value
//! dialects; shared CPS construction belongs to `tribute-passes`.

use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;

use salsa::Accumulator;
use tribute_core::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use tribute_ir::dialect::adt;
use tribute_ir::dialect::{
    list,
    tribute_control::{self, CompilerIntrinsicDeclaration, OperationDeclaration, Row, RowBinder},
};
use trunk_ir::context::{BlockArgData, BlockData, IrContext, OperationDataBuilder, RegionData};
use trunk_ir::dialect::{arith, core, scf};
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{BlockRef, OpRef, PathRef, TypeRef, ValueRef};
use trunk_ir::rewrite::Module as IrModule;
use trunk_ir::types::{Attribute, Location};
use trunk_ir::{Symbol, SymbolPath};

use crate::SortedMap;
use crate::ast::{
    AbilityRow, Arm, CtorId, Decl, DeclEntry, Expr, ExprKind, ExternFuncDecl, FieldInit, FuncDecl,
    HandlerArm, HandlerKind, OpDeclKind, Pattern, PatternKind, ResolvedRef, Stmt, TypeKind,
    TypedRef,
};

use super::super::context::{IrLoweringCtx, LogicalGeneratedSignature};
use super::super::{FrontendIrModule, TypedModule};
use super::{FuncSignature, IrBuilder};

mod local_callables;

struct Declarations<'db> {
    // TypeRef is an arena index and therefore has deterministic total order only
    // through its debug representation.  Preserve first source use explicitly.
    values: Vec<OperationDeclaration>,
    /// The ability instances operations were recorded for, in first-use order.
    instances: Vec<(
        SymbolPath,
        crate::ast::AbilityId<'db>,
        Vec<crate::ast::Type<'db>>,
    )>,
    compiler_intrinsics: Vec<CompilerIntrinsicDeclaration>,
    schemas: HashMap<crate::ast::AbilityId<'db>, crate::typeck::AbilityInfo<'db>>,
    handler_operations:
        SortedMap<crate::ast::NodeId, crate::typeck::InstantiatedHandlerOperation<'db>>,
    perform_operations:
        SortedMap<crate::ast::NodeId, crate::typeck::InstantiatedPerformOperation<'db>>,
    lambda_signatures: SortedMap<crate::ast::NodeId, crate::typeck::LambdaSignature<'db>>,
    exhaustive_cases: HashSet<crate::ast::NodeId>,
    evidence_plans: SortedMap<crate::ast::NodeId, Vec<crate::typeck::EvidenceStep<'db>>>,
    local_instances: SortedMap<crate::ast::NodeId, crate::typeck::LocalCallableInstance<'db>>,
    /// The instance each reference to a generic function selects.
    function_instances: SortedMap<crate::ast::NodeId, crate::typeck::FunctionInstance<'db>>,
    local_callables: local_callables::Plan<'db>,
    /// The plan of each function, by its declaration, until it is lowered.
    function_local_callables: HashMap<crate::ast::NodeId, local_callables::Plan<'db>>,
    /// The uses of the let-bound lambdas of the function being lowered.
    local_uses: Vec<crate::monomorphize::LocalInstantiation<'db>>,
}

/// Fully typed semantic inputs for one source ability-operation call.
struct PerformMetadata<'a, 'db> {
    ability: crate::ast::AbilityId<'db>,
    operation: Symbol,
    kind: OpDeclKind,
    semantic: &'a crate::typeck::InstantiatedPerformOperation<'db>,
}

impl<'db> Declarations<'db> {
    /// Note that the module names the ability instance `symbol`.
    fn instance(
        &mut self,
        symbol: &SymbolPath,
        ability: crate::ast::AbilityId<'db>,
        arguments: &[crate::ast::Type<'db>],
    ) {
        if !self.instances.iter().any(|(known, ..)| known == symbol) {
            self.instances
                .push((symbol.clone(), ability, arguments.to_vec()));
        }
    }

    fn record(
        &mut self,
        ir: &IrContext,
        declaration: OperationDeclaration,
        ability: crate::ast::AbilityId<'db>,
        arguments: &[crate::ast::Type<'db>],
        location: Location,
        db: &dyn salsa::Database,
    ) {
        self.instance(&declaration.ability, ability, arguments);
        if let Some(existing) = self.values.iter().find(|candidate| {
            candidate.ability == declaration.ability && candidate.op_name == declaration.op_name
        }) {
            if existing != &declaration {
                Diagnostic::new(
                    format!(
                        "conflicting instantiated declaration for ability operation {}",
                        ir.str(declaration.op_name)
                    ),
                    location.span,
                    DiagnosticSeverity::Error,
                    CompilationPhase::Lowering,
                )
                .accumulate(db);
                panic!("conflicting source-logical ability operation declaration");
            }
            return;
        }
        self.values.push(declaration);
    }
}

fn op(
    ir: &mut IrContext,
    block: BlockRef,
    location: Location,
    def: &'static trunk_ir::OpDef,
    build: impl FnOnce(OperationDataBuilder) -> OperationDataBuilder,
) -> OpRef {
    let data = build(OperationDataBuilder::new(location, def)).build(ir);
    let op = ir.create_op(data);
    ir.push_op(block, op);
    op
}

/// Whether lowering ended `block` with a source `become`. Such a block has
/// already left its callable and takes no `return` or `yield`.
fn ends_with_tail_call(ir: &IrContext, block: BlockRef) -> bool {
    ir.block(block).ops.last().is_some_and(|&op| {
        tribute_control::TailCall::matches(ir, op)
            || tribute_control::TailCallIndirect::matches(ir, op)
    })
}

/// Attach the typechecked evidence selection of the source node `node`.
fn attach_evidence_plan<'db>(
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

fn lower_evidence_step<'db>(
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

fn result(ir: &IrContext, op: OpRef) -> ValueRef {
    ir.op_result(op, 0)
}

/// Lower a named function value at its exact source-logical callable contract.
///
/// A `func_ref` may take a callable type whose row includes its target's, but
/// it must preserve the source parameter and result types. This is the only
/// callable adaptation accepted at the source-logical boundary: a generic
/// `unrealized_conversion_cast` cannot change a callable ABI.
fn lower_function_ref<'db>(
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
enum RowArguments {
    /// The function instance type checking selected for this node.
    Instance(crate::ast::NodeId),
    /// These rows, of a reference already lowered.
    Rows(Vec<Row>),
}

impl RowArguments {
    /// The rows a reference to `name` passes, in the current definition's
    /// binders: one for each binder `signature` quantifies.
    fn rows<'db>(
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
fn lower_expr_for_callable_parameter<'db>(
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

pub(super) fn lower_module<'db>(
    typed: TypedModule<'db>,
    db: &'db dyn salsa::Database,
    ir: &mut IrContext,
    path: PathRef,
) -> FrontendIrModule {
    let TypedModule {
        ast,
        span_map,
        function_types,
        constructor_types,
        specialized_enum_variants,
        node_types,
        local_instances,
        function_instances,
        ability_definitions,
        handler_operations,
        perform_operations,
        lambda_signatures,
        exhaustive_cases,
        evidence_plans,
        well_known_types,
        compiler_intrinsics,
        runtime_types,
        merged_sources,
    } = typed;
    let source_paths = merged_sources
        .iter()
        .map(|uri| (crate::ast::node_id::source_hash(uri), ir.intern_path(uri)))
        .collect();
    let location = Location::new(path, span_map.get_or_default(ast.id));
    let module_name = ast.name.unwrap_or_else(|| Symbol::new("main"));
    let mut ctx = IrLoweringCtx::new(db, path, span_map, function_types, node_types)
        .with_compiler_intrinsics(compiler_intrinsics)
        .with_runtime_types(runtime_types)
        .with_source_paths(source_paths)
        .with_literal_equalities(crate::ast_to_ir::context::LiteralEqualities {
            string: well_known_types
                .string_equality
                .map(|id| id.path(db).clone()),
            bytes: well_known_types
                .bytes_equality
                .map(|id| id.path(db).clone()),
        });
    let module_block = ir.create_block(BlockData {
        location,
        args: vec![],
        ops: Default::default(),
        parent_region: None,
    });
    ctx.set_module_block(module_block);
    let mut declarations = Declarations {
        values: vec![],
        instances: vec![],
        compiler_intrinsics: vec![],
        schemas: ability_definitions,
        handler_operations,
        perform_operations,
        lambda_signatures,
        exhaustive_cases,
        evidence_plans,
        local_instances,
        function_instances,
        local_callables: local_callables::Plan::default(),
        function_local_callables: HashMap::default(),
        local_uses: Vec::new(),
    };
    prescan_row_binders(&mut ctx, &ast.entries, &declarations);
    let row_effects = row_effects(&ctx, &ast.entries, &declarations);
    let mut well_known_type_prescan = super::decl::WellKnownTypePrescan::new(well_known_types);
    let mut published_runtime_types = Vec::new();
    collect_logical_nominal_identities(&mut ctx, &ast.entries);
    prescan_logical_nominal_layouts(
        &mut ctx,
        ir,
        &ast.entries,
        &mut well_known_type_prescan,
        &mut published_runtime_types,
        &constructor_types,
        &specialized_enum_variants,
    );
    prescan_struct_accessor_signatures(&mut ctx, ir, &ast.entries);
    prescan_source_functions(&mut ctx, &ast.entries);
    plan_local_callables(&mut ctx, ir, &ast.entries, &mut declarations);
    let well_known_types = well_known_type_prescan.finish();
    for entry in ast.entries {
        lower_decl(&mut ctx, ir, module_block, entry, &mut declarations);
    }
    // An instance only a row names still needs its declaration.
    for effect in row_effects {
        let symbol = ctx.ability_symbol(&effect.ability_id.qualified(db), &effect.args);
        declarations.instance(&symbol, effect.ability_id, &effect.args);
    }
    ctx.set_module(SymbolPath::root());
    let region = ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![module_block],
        parent_op: None,
    });
    let module = core::Module::operands()
        .sym_name(ir.intern_symbol_text(&module_name))
        .regions(region)
        .build(ir, location);
    well_known_types.attach(ir, module.op_ref());
    for (path, layout) in published_runtime_types {
        tribute_ir::metadata::publish_prelude_type(ir, module.op_ref(), &path.to_string(), layout);
    }
    let module = IrModule::new(ir, module.op_ref()).expect("valid core.module operation");
    let operations = instance_operations(&ctx, ir, &declarations);
    tribute_control::declare_abilities(
        ir,
        module,
        declarations.instances.iter().map(|(symbol, ..)| symbol),
        &operations,
    );
    FrontendIrModule {
        module,
        compiler_intrinsics: declarations.compiler_intrinsics,
    }
}

/// Every operation of each ability instance the module names, with the
/// parameter and result types of that instance.
fn instance_operations<'db>(
    ctx: &IrLoweringCtx<'db>,
    ir: &mut IrContext,
    declarations: &Declarations<'db>,
) -> Vec<OperationDeclaration> {
    let instantiate = |ty, arguments: &[crate::ast::Type<'db>]| {
        match crate::typeck::subst::substitute_bound_vars(ctx.db, ty, arguments) {
            crate::typeck::subst::SubstResult::Ok(ty) => ty,
            crate::typeck::subst::SubstResult::OutOfBounds { .. } => {
                panic!("ability operation substitution failed")
            }
        }
    };
    let mut operations = Vec::new();
    for (instance, ability, arguments) in &declarations.instances {
        let schema = &declarations.schemas[ability];
        let mut declared: Vec<_> = schema.operations.values().collect();
        declared.sort_by(|a, b| a.name.with_str(|a| b.name.with_str(|b| a.cmp(b))));
        for operation in declared {
            let parameters: Vec<_> = operation
                .param_types
                .iter()
                .map(|ty| ctx.convert_logical_type(ir, instantiate(*ty, arguments)))
                .collect();
            let result =
                ctx.convert_logical_type(ir, instantiate(operation.return_type, arguments));
            operations.push(OperationDeclaration::new(
                instance.clone(),
                ir.intern_symbol_text(&operation.name),
                ir.intern_str(operation_kind_text(operation.kind)),
                parameters,
                result,
            ));
        }
    }
    operations
}

/// Each use of a let-bound lambda in `function`.
fn local_instantiations<'db>(
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
fn prescan_row_binders<'db>(
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
fn row_effects<'db>(
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
fn plan_local_callables<'db>(
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

impl<'db> TypedModule<'db> {
    pub(crate) fn lower_module(
        self,
        db: &'db dyn salsa::Database,
        ir: &mut IrContext,
        path: PathRef,
    ) -> FrontendIrModule {
        lower_module(self, db, ir, path)
    }
}

fn lower_decl<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    top: BlockRef,
    entry: DeclEntry<TypedRef<'db>>,
    declarations: &mut Declarations<'db>,
) {
    ctx.set_module(entry.module);
    match entry.decl {
        Decl::Function(function) => {
            let outer = ctx.enter_definition(&ctx.qualify_name(&function.name));
            lower_function(ctx, ir, top, function, declarations);
            ctx.leave_definition(outer);
        }
        Decl::ExternFunction(function) => lower_extern(ctx, ir, top, function, declarations),
        Decl::Struct(declaration) => lower_struct_accessors(ctx, ir, top, declaration),
        // Nominal declarations remain types used by ordinary ADT value ops.
        // Struct accessors are emitted above as source-logical callables.
        Decl::Enum(_) | Decl::Ability(_) | Decl::Use(_) => {}
    }
}

/// Register nominal layouts used exclusively by source-logical lowering.
///
/// Constructor schemes are the authoritative semantic field types, so records,
/// constructors, accessors, and pattern extraction share the same recursive
/// `tribute_control` callable layouts.
fn prescan_logical_nominal_layouts<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    entries: &[DeclEntry<TypedRef<'db>>],
    well_known_types: &mut super::decl::WellKnownTypePrescan,
    runtime_types: &mut Vec<(SymbolPath, TypeRef)>,
    constructors: &HashMap<crate::ast::CtorId<'db>, crate::ast::TypeScheme<'db>>,
    specialized_enum_variants: &SortedMap<crate::ast::NodeId, crate::ast::TypeScheme<'db>>,
) {
    for entry in entries {
        let prefix = &entry.module;
        match &entry.decl {
            Decl::Struct(structure) => {
                let qualified = prefix.child(structure.name.clone());
                let ctor = CtorId::new(ctx.db, qualified);
                let field_types =
                    constructor_fields(ctx, constructors, ctor, structure.fields.len());
                let fields = structure
                    .fields
                    .iter()
                    .zip(field_types)
                    .map(|(field, ty)| {
                        (
                            field.name.clone().unwrap_or_else(|| Symbol::new("_")),
                            ctx.convert_logical_type(ir, ty),
                        )
                    })
                    .collect::<Vec<_>>();
                let name = super::qualified_type_name(ctx.db, &ctor);
                let layout = ctx.adt_struct_type(ir, &name, &fields);
                ctx.register_type(name.clone(), layout);
                ir.register_type_alias(name, layout);
                if let Some(path) = ctx.runtime_type_path(structure.id) {
                    runtime_types.push((path.clone(), layout));
                }
            }
            Decl::Enum(enumeration) => {
                let qualified = prefix.child(enumeration.name.clone());
                let enum_ctor = CtorId::new(ctx.db, qualified.clone());
                let variants = enumeration
                    .variants
                    .iter()
                    .map(|variant| {
                        let variant_ctor =
                            CtorId::new(ctx.db, prefix.child(variant.name.clone()));
                        let fields = if enumeration.id.variant().is_some() {
                            let scheme = *specialized_enum_variants.get(&variant.id)
                                .expect("missing specialized enum variant schema");
                            let result = match scheme.body(ctx.db).kind(ctx.db) {
                                TypeKind::Func { result, .. } => *result,
                                _ => scheme.body(ctx.db),
                            };
                            assert!(matches!(result.kind(ctx.db), TypeKind::Named { id, args, .. }
                                if args.is_empty() && *id.path(ctx.db) == qualified
                                    && id.origin(ctx.db) == crate::ast::TypeOrigin::Source { disambiguator: enumeration.disambiguator }),
                                "specialized enum variant schema has wrong owner");
                            constructor_schema_fields(ctx, scheme, variant.fields.len())
                        } else {
                            constructor_fields(ctx, constructors, variant_ctor, variant.fields.len())
                        }
                        .into_iter()
                        .map(|ty| ctx.convert_logical_type(ir, ty))
                        .collect();
                        (variant.name.clone(), fields)
                    })
                    .collect::<Vec<_>>();
                let name = super::qualified_type_name(ctx.db, &enum_ctor);
                let location = ctx.location(enumeration.id);
                let definition = crate::ast::TypeDefId::declared(
                    ctx.db,
                    qualified.clone(),
                    enumeration.disambiguator,
                );
                let layout = if well_known_types.is_string(definition) {
                    ctx.adt_enum_type_with_definition(ir, &name, &variants, location)
                } else {
                    ctx.adt_enum_type(ir, &name, &variants)
                };
                ctx.register_type(name.clone(), layout);
                ir.register_type_alias(name, layout);
                for variant in &enumeration.variants {
                    if let Some(names) = variant
                        .fields
                        .iter()
                        .map(|field| field.name.clone())
                        .collect()
                    {
                        ctx.register_variant_field_names(layout, variant.name.clone(), names);
                    }
                }
                if well_known_types.is_string(definition) {
                    well_known_types.record_string(layout);
                }
            }
            _ => {}
        }
    }
}

/// First phase of logical nominal prescan. All identities must be known before
/// converting any field: recursive and forward references are logical nominal
/// values even while their concrete layout is still being assembled.
fn collect_logical_nominal_identities<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    entries: &[DeclEntry<TypedRef<'db>>],
) {
    for entry in entries {
        let prefix = &entry.module;
        match &entry.decl {
            Decl::Struct(structure) => {
                let qualified = prefix.child(structure.name.clone());
                let ctor = CtorId::new(ctx.db, qualified);
                ctx.declare_logical_nominal(super::qualified_type_name(ctx.db, &ctor));
                ctx.register_struct_fields(
                    ctor,
                    structure
                        .fields
                        .iter()
                        .map(|field| field.name.clone().unwrap_or_else(|| Symbol::new("_")))
                        .collect(),
                );
            }
            Decl::Enum(enumeration) => {
                let qualified = prefix.child(enumeration.name.clone());
                let ctor = CtorId::new(ctx.db, qualified);
                ctx.declare_logical_nominal(super::qualified_type_name(ctx.db, &ctor));
            }
            _ => {}
        }
    }
}

fn constructor_fields<'db>(
    ctx: &IrLoweringCtx<'db>,
    constructors: &HashMap<crate::ast::CtorId<'db>, crate::ast::TypeScheme<'db>>,
    constructor: CtorId<'db>,
    field_count: usize,
) -> Vec<crate::ast::Type<'db>> {
    let scheme = constructors
        .get(&constructor)
        .unwrap_or_else(|| panic!("missing typechecked constructor schema for {constructor:?}"));
    constructor_schema_fields(ctx, *scheme, field_count)
}

fn constructor_schema_fields<'db>(
    ctx: &IrLoweringCtx<'db>,
    scheme: crate::ast::TypeScheme<'db>,
    field_count: usize,
) -> Vec<crate::ast::Type<'db>> {
    match scheme.body(ctx.db()).kind(ctx.db()) {
        TypeKind::Func { params, .. } => {
            assert_eq!(
                params.len(),
                field_count,
                "constructor field schema mismatch"
            );
            params.clone()
        }
        _ if field_count == 0 => vec![],
        _ => panic!("non-nullary constructor is missing callable field schema"),
    }
}

/// The names of the setter and the modifier of the field whose getter is
/// `getter`: `T::f::set` and `T::f::modify`.
fn field_update_names(getter: &SymbolPath) -> [SymbolPath; 2] {
    crate::ast::FIELD_LENS_FUNCTIONS.map(|name| getter.child(name))
}

fn prescan_struct_accessor_signatures<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    entries: &[DeclEntry<TypedRef<'db>>],
) {
    for entry in entries {
        ctx.set_module(entry.module.clone());
        let Decl::Struct(declaration) = &entry.decl else {
            continue;
        };
        let prefix = ctx.module_path().clone();
        let qualified = prefix.child(declaration.name.clone());
        let type_name = super::qualified_type_name(ctx.db, &CtorId::new(ctx.db, qualified));
        let struct_type = ctx.adt_typeref(ir, &type_name);
        let layout = ctx
            .get_type(&type_name)
            .unwrap_or_else(|| panic!("missing logical struct layout for accessor"));
        let layout_fields = tribute_ir::dialect::adt::layout::get_struct_fields(ir, layout)
            .unwrap_or_else(|| panic!("malformed logical struct layout"));
        for (index, field) in declaration.fields.iter().enumerate() {
            let field_name = field.name.clone().unwrap_or_else(|| Symbol::new("_"));
            let getter_name = prefix
                .child(declaration.name.clone())
                .child(field_name.clone());
            let field_type = layout_fields
                .get(index)
                .map(|(_, ty)| *ty)
                .unwrap_or_else(|| panic!("missing logical struct accessor field type"));
            let closed = |param_types, return_type| LogicalGeneratedSignature {
                param_types,
                return_type,
                row: Row::default(),
                binders: vec![],
            };
            ctx.register_logical_generated_signature(
                &getter_name,
                closed(vec![struct_type], field_type),
            );
            if field.name.is_none() {
                continue;
            }
            let [setter_name, modifier_name] = field_update_names(&getter_name);
            ctx.register_logical_generated_signature(
                &setter_name,
                closed(vec![struct_type, field_type], struct_type),
            );
            // The modifier performs whatever its callback performs.
            let row = Row::new([], Some(0));
            let callback = tribute_control::FuncSig::params([field_type], field_type)
                .row(row.clone())
                .intern(ir)
                .as_type_ref();
            ctx.register_logical_generated_signature(
                &modifier_name,
                LogicalGeneratedSignature {
                    param_types: vec![struct_type, callback],
                    return_type: struct_type,
                    row: row.clone(),
                    binders: vec![RowBinder::Param],
                },
            );
        }
    }
}

fn prescan_source_functions<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    entries: &[DeclEntry<TypedRef<'db>>],
) {
    for entry in entries {
        ctx.set_module(entry.module.clone());
        match &entry.decl {
            Decl::Function(function) => {
                ctx.register_logical_source_function(ctx.qualify_name(&function.name));
            }
            Decl::ExternFunction(function) => {
                ctx.register_logical_source_function(ctx.qualify_name(&function.name));
            }
            _ => {}
        }
    }
}

fn lower_struct_accessors<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    top: BlockRef,
    declaration: crate::ast::StructDecl,
) {
    let location = ctx.location(declaration.id);
    let prefix = ctx.module_path().clone();
    let qualified = prefix.child(declaration.name.clone());
    let type_name = super::qualified_type_name(ctx.db, &CtorId::new(ctx.db, qualified));
    let struct_type = ctx.adt_typeref(ir, &type_name);
    let layout_type = ctx
        .get_type(&type_name)
        .unwrap_or_else(|| panic!("prescan did not register struct layout {type_name}"));
    for (index, field) in declaration.fields.iter().enumerate() {
        let field_name = field.name.clone().unwrap_or_else(|| Symbol::new("_"));
        let getter_name = prefix
            .child(declaration.name.clone())
            .child(field_name.clone());
        let field_type = tribute_ir::dialect::adt::layout::get_struct_fields(ir, layout_type)
            .and_then(|fields| fields.get(index).map(|(_, ty)| *ty))
            .unwrap_or_else(|| panic!("missing logical struct accessor field type"));
        let entry = ir.create_block(BlockData {
            location,
            args: vec![BlockArgData {
                ty: struct_type,
                attrs: Default::default(),
            }],
            ops: Default::default(),
            parent_region: None,
        });
        let field_value = adt::StructGet::operands(ir.block_arg(entry, 0))
            .r#type(layout_type)
            .field(index as u32)
            .results(field_type)
            .build(ir, location);
        ir.push_op(entry, field_value.op_ref());
        let field_result = field_value.result(ir);
        op(
            ir,
            entry,
            location,
            tribute_control::Return::DEF,
            |builder| builder.operand(field_result),
        );
        let body = ir.create_region(RegionData {
            location,
            blocks: trunk_ir::smallvec::smallvec![entry],
            parent_op: None,
        });
        let callable = tribute_control::FuncSig::params([struct_type], field_type)
            .intern(ir)
            .as_type_ref();
        let getter =
            tribute_control::func_declaration(ir, location, &getter_name.to_symbol(), callable);
        ir.push_op_region(getter.op_ref(), body);
        ir.push_op(top, getter.op_ref());

        if field.name.is_none() {
            continue;
        }
        let field_types: Vec<TypeRef> =
            tribute_ir::dialect::adt::layout::get_struct_fields(ir, layout_type)
                .unwrap_or_else(|| panic!("malformed logical struct layout"))
                .iter()
                .map(|(_, ty)| *ty)
                .collect();
        let [setter_name, modifier_name] = field_update_names(&getter_name);
        let callback_row = Row::new([], Some(0));
        let callback = tribute_control::FuncSig::params([field_type], field_type)
            .row(callback_row.clone())
            .intern(ir)
            .as_type_ref();
        for (name, argument_ty, modifies) in [
            (setter_name, field_type, false),
            (modifier_name, callback, true),
        ] {
            let entry = ir.create_block(BlockData {
                location,
                args: [struct_type, argument_ty]
                    .map(|ty| BlockArgData {
                        ty,
                        attrs: Default::default(),
                    })
                    .to_vec(),
                ops: Default::default(),
                parent_region: None,
            });
            let subject = ir.block_arg(entry, 0);
            let argument = ir.block_arg(entry, 1);
            // The modifier applies its callback to the field's current value.
            let replacement = if modifies {
                let current = adt::StructGet::operands(subject)
                    .r#type(layout_type)
                    .field(index as u32)
                    .results(field_type)
                    .build(ir, location);
                ir.push_op(entry, current.op_ref());
                let current = current.result(ir);
                let call = op(
                    ir,
                    entry,
                    location,
                    tribute_control::CallIndirect::DEF,
                    |builder| {
                        builder
                            .operand(argument)
                            .operand(current)
                            .result(field_type)
                    },
                );
                result(ir, call)
            } else {
                argument
            };
            let values: Vec<ValueRef> = field_types
                .iter()
                .enumerate()
                .map(|(other, ty)| {
                    if other == index {
                        return replacement;
                    }
                    let get = adt::StructGet::operands(subject)
                        .r#type(layout_type)
                        .field(other as u32)
                        .results(*ty)
                        .build(ir, location);
                    ir.push_op(entry, get.op_ref());
                    get.result(ir)
                })
                .collect();
            let updated = adt::StructNew::operands(values)
                .r#type(layout_type)
                .results(struct_type)
                .build(ir, location);
            ir.push_op(entry, updated.op_ref());
            let updated = updated.result(ir);
            op(
                ir,
                entry,
                location,
                tribute_control::Return::DEF,
                |builder| builder.operand(updated),
            );
            let body = ir.create_region(RegionData {
                location,
                blocks: trunk_ir::smallvec::smallvec![entry],
                parent_op: None,
            });
            let row = if modifies {
                callback_row.clone()
            } else {
                Row::default()
            };
            let callable =
                tribute_control::FuncSig::params([struct_type, argument_ty], struct_type)
                    .row(row)
                    .intern(ir)
                    .as_type_ref();
            let function =
                tribute_control::func_declaration(ir, location, &name.to_symbol(), callable);
            if modifies {
                let param = ir.string_attr(RowBinder::Param.keyword());
                ir.op_attrs_mut(function.op_ref()).insert(
                    tribute_control::ROW_BINDERS_ATTR,
                    Attribute::List(vec![param]),
                );
            }
            ir.push_op_region(function.op_ref(), body);
            ir.push_op(top, function.op_ref());
        }
    }
}

fn function_signature<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    function: &FuncDecl<TypedRef<'db>>,
) -> FuncSignature {
    let qualified = ctx.qualify_name(&function.name);
    // Nested declarations are exported under their qualified identity; a
    // short-name lookup can silently select an unrelated root declaration.
    FuncSignature::lookup_logical(ctx, ir, &qualified)
        .or_else(|| {
            (qualified != [&function.name])
                .then(|| {
                    FuncSignature::lookup_logical(ctx, ir, std::slice::from_ref(&function.name))
                })
                .flatten()
        })
        .unwrap_or_else(|| {
            panic!(
                "missing typechecked signature for function {}",
                function.name
            )
        })
}

fn lower_function<'db>(
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

fn lower_extern<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    top: BlockRef,
    decl: ExternFuncDecl,
    declarations: &mut Declarations<'db>,
) {
    let location = ctx.location(decl.id);
    let qualified = ctx.qualify_name(&decl.name);
    let signature = if qualified == [&decl.name] {
        FuncSignature::lookup_logical(ctx, ir, &qualified)
    } else {
        FuncSignature::lookup_logical(ctx, ir, &qualified)
            .or_else(|| FuncSignature::lookup_logical(ctx, ir, std::slice::from_ref(&decl.name)))
    }
    .unwrap_or_else(|| {
        panic!(
            "missing typechecked signature for extern function {}",
            decl.name
        )
    });
    let callable = signature.callable_type(ir);
    let name = ctx.function_symbol(&ctx.qualify_name(&decl.name));
    let function = tribute_control::func_declaration(ir, location, &name, callable);
    signature.attach_rows(ir, function.op_ref());
    if let Some(identity) = ctx.compiler_intrinsic(decl.id) {
        let identity_text = ir.intern_symbol_text(&identity);
        ir.op_attrs_mut(function.op_ref()).insert(
            tribute_control::COMPILER_INTRINSIC_ATTR,
            Attribute::String(identity_text),
        );
        declarations
            .compiler_intrinsics
            .push(CompilerIntrinsicDeclaration::new(
                SymbolPath::new([name]),
                identity,
            ));
        declarations
            .compiler_intrinsics
            .sort_by_key(|declaration| (declaration.symbol.clone(), declaration.identity.clone()));
    }
    let abi = decl.abi.with_str(|abi| ir.string_attr(abi));
    ir.op_attrs_mut(function.op_ref()).insert("abi", abi);
    // C linkage has one flat namespace: the declaration links under its
    // declared name, wherever it is declared.
    if decl.abi == "C" {
        let link_name = ir.intern_symbol_text(&decl.name);
        ir.op_attrs_mut(function.op_ref()).insert(
            trunk_ir::symbol_table::LINK_NAME,
            Attribute::String(link_name),
        );
    }
    ir.push_op(top, function.op_ref());
}

fn ensure_prelude_declaration(
    builder: &mut IrBuilder<'_, '_>,
    location: Location,
    name: &SymbolPath,
    signature: &FuncSignature,
) {
    if builder.ctx.is_logical_source_function(name)
        || builder
            .ctx
            .lookup_logical_generated_signature(name)
            .is_some()
        || !builder.ctx.mark_logical_extern_emitted(name.clone())
    {
        return;
    }
    let callable = signature.callable_type(builder.ir);
    let symbol = builder.ctx.function_symbol(name);
    let declaration = tribute_control::func_declaration(builder.ir, location, &symbol, callable);
    signature.attach_rows(builder.ir, declaration.op_ref());
    let top = builder
        .ctx
        .module_block()
        .expect("logical module block must be set before lowering calls");
    builder.ir.push_op(top, declaration.op_ref());
}

fn expr_type(builder: &mut IrBuilder<'_, '_>, expr: &Expr<TypedRef<'_>>) -> TypeRef {
    builder
        .ctx
        .get_node_type(expr.id)
        .copied()
        .map(|ty| builder.ctx.convert_logical_type(builder.ir, ty))
        .unwrap_or_else(|| panic!("missing typechecked expression type"))
}

fn operation_kind_text(kind: OpDeclKind) -> &'static str {
    match kind {
        OpDeclKind::Fn => "fn",
        OpDeclKind::Op => "op",
    }
}

fn call_operation_metadata<'db>(
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

fn adapt_operation_arguments(
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

fn adapt_operation_arguments_or_recover<'db>(
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

fn lower_expr<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    expr: Expr<TypedRef<'db>>,
    declarations: &mut Declarations<'db>,
) -> Option<ValueRef> {
    let location = builder.location(expr.id);
    match *expr.kind {
        ExprKind::NatLit(value) => {
            let ty = builder.ctx.i32_type(builder.ir);
            let value = arith::Const::operands()
                .value(Attribute::Int(value as i128))
                .results(ty)
                .build(builder.ir, location);
            builder.ir.push_op(builder.block, value.op_ref());
            Some(value.result(builder.ir))
        }
        ExprKind::IntLit(value) => {
            let ty = builder.ctx.i32_type(builder.ir);
            let value = arith::Const::operands()
                .value(Attribute::Int(value as i128))
                .results(ty)
                .build(builder.ir, location);
            builder.ir.push_op(builder.block, value.op_ref());
            Some(value.result(builder.ir))
        }
        ExprKind::BoolLit(value) => {
            let ty = builder.ctx.bool_type(builder.ir);
            let value = arith::Const::operands()
                .value(Attribute::Bool(value))
                .results(ty)
                .build(builder.ir, location);
            builder.ir.push_op(builder.block, value.op_ref());
            Some(value.result(builder.ir))
        }
        ExprKind::FloatLit(value) => {
            let ty = builder.ctx.f64_type(builder.ir);
            let value = arith::Const::operands()
                .value(Attribute::FloatBits(value.value().to_bits()))
                .results(ty)
                .build(builder.ir, location);
            builder.ir.push_op(builder.block, value.op_ref());
            Some(value.result(builder.ir))
        }
        ExprKind::Nil => Some(builder.emit_nil(location)),
        ExprKind::RuneLit(value) => {
            let ty = builder.ctx.i32_type(builder.ir);
            let value = arith::Const::operands()
                .value(Attribute::Int(value as i32 as i128))
                .results(ty)
                .build(builder.ir, location);
            builder.ir.push_op(builder.block, value.op_ref());
            Some(value.result(builder.ir))
        }
        ExprKind::BytesLit(value) => {
            let ty = builder.ctx.bytes_type(builder.ir);
            let value = adt::BytesConst::operands()
                .value(value.into())
                .results(ty)
                .build(builder.ir, location);
            builder.ir.push_op(builder.block, value.op_ref());
            Some(value.result(builder.ir))
        }
        ExprKind::StringLit(value) => {
            let ty = builder.ctx.anyref_type(builder.ir);
            let value = builder.ir.intern_str(&value);
            let value = adt::StringConst::operands()
                .value(value)
                .results(ty)
                .build(builder.ir, location);
            builder.ir.push_op(builder.block, value.op_ref());
            Some(value.result(builder.ir))
        }
        ExprKind::Var(reference) => match reference.resolved {
            ResolvedRef::Local { id, .. } => Some(
                lower_local_reference(builder, expr.id, id, declarations)
                    .unwrap_or_else(|| panic!("missing logical binding for local {id:?}")),
            ),
            ResolvedRef::Function { id } => {
                let name = id.path(builder.ctx.db);
                Some(lower_function_ref(
                    builder,
                    location,
                    name.clone(),
                    RowArguments::Instance(expr.id),
                    declarations,
                    None,
                ))
            }
            ResolvedRef::AbilityOp { .. } => {
                // Typechecking has already diagnosed this non-call operation
                // reference.  Preserve error recovery without inventing a
                // perform declaration from malformed source.
                Some(builder.emit_nil(location))
            }
            _ => panic!("unsupported resolved reference at source-logical boundary"),
        },
        ExprKind::Block { stmts, value } => {
            let mut scope = builder.ctx.scope();
            let mut inner = IrBuilder::new(&mut scope, builder.ir, builder.block);
            for statement in stmts {
                lower_statement(&mut inner, statement, declarations);
            }
            lower_expr(&mut inner, value, declarations)
        }
        ExprKind::BinOp {
            op: binop,
            lhs,
            rhs,
        } => {
            let lhs = lower_expr(builder, lhs, declarations)?;
            let bool_ty = builder.ctx.bool_type(builder.ir);
            let then_block = builder.ir.create_block(BlockData {
                location,
                args: vec![],
                ops: Default::default(),
                parent_region: None,
            });
            let then_value = {
                let mut inner = IrBuilder::new(builder.ctx, builder.ir, then_block);
                match binop {
                    crate::ast::BinOpKind::And => {
                        lower_expr(&mut inner, rhs.clone(), declarations)?
                    }
                    crate::ast::BinOpKind::Or => emit_bool(&mut inner, location, true),
                }
            };
            let then_yield = scf::Yield::operands([then_value]).build(builder.ir, location);
            builder.ir.push_op(then_block, then_yield.op_ref());
            let then_region = builder.ir.create_region(RegionData {
                location,
                blocks: trunk_ir::smallvec::smallvec![then_block],
                parent_op: None,
            });
            let else_block = builder.ir.create_block(BlockData {
                location,
                args: vec![],
                ops: Default::default(),
                parent_region: None,
            });
            let else_value = {
                let mut inner = IrBuilder::new(builder.ctx, builder.ir, else_block);
                match binop {
                    crate::ast::BinOpKind::And => emit_bool(&mut inner, location, false),
                    crate::ast::BinOpKind::Or => lower_expr(&mut inner, rhs, declarations)?,
                }
            };
            let else_yield = scf::Yield::operands([else_value]).build(builder.ir, location);
            builder.ir.push_op(else_block, else_yield.op_ref());
            let else_region = builder.ir.create_region(RegionData {
                location,
                blocks: trunk_ir::smallvec::smallvec![else_block],
                parent_op: None,
            });
            let branch = scf::If::operands(lhs)
                .results(bool_ty)
                .regions(then_region, else_region)
                .build(builder.ir, location);
            builder.ir.push_op(builder.block, branch.op_ref());
            Some(branch.result(builder.ir))
        }
        ExprKind::Call { callee, args, .. } => {
            let result_ty = builder
                .ctx
                .get_node_type(expr.id)
                .copied()
                .map(|ty| builder.ctx.convert_logical_type(builder.ir, ty))
                .unwrap_or_else(|| panic!("missing typechecked result for call"));
            lower_call(
                builder,
                location,
                expr.id,
                result_ty,
                callee,
                args,
                declarations,
                CallPosition::Value,
            )
        }
        ExprKind::Become { call } => lower_become(builder, location, call, declarations),
        ExprKind::Lambda { params, body } => {
            let signature = declarations
                .lambda_signatures
                .get(&expr.id)
                .cloned()
                .unwrap_or_else(|| panic!("missing solved logical lambda signature"));
            lower_lambda(
                builder,
                location,
                signature,
                params,
                body,
                declarations,
                None,
            )
        }
        ExprKind::Handle { body, handlers } => {
            let result_ty = builder
                .ctx
                .get_node_type(expr.id)
                .copied()
                .map(|ty| builder.ctx.convert_logical_type(builder.ir, ty))
                .unwrap_or_else(|| panic!("missing typechecked result for handle"));
            lower_handle(
                builder,
                location,
                expr.id,
                result_ty,
                body,
                handlers,
                declarations,
            )
        }
        ExprKind::Cons { ctor, args } => {
            lower_constructor(builder, location, expr.id, ctor, args, declarations)
        }
        ExprKind::Tuple(elements) => {
            lower_tuple(builder, location, expr.id, elements, declarations)
        }
        ExprKind::List(elements) => lower_list(builder, location, expr.id, elements, declarations),
        ExprKind::Record {
            type_name,
            fields,
            spread,
        } => lower_record(
            builder,
            location,
            expr.id,
            type_name,
            fields,
            spread,
            declarations,
        ),
        ExprKind::Case { scrutinee, arms } => {
            let scrutinee = lower_expr(builder, scrutinee, declarations)?;
            let result_ty = expr_type_for_id(builder, expr.id);
            let exhaustive = declarations.exhaustive_cases.contains(&expr.id);
            // The unguarded arms of an exhaustive case match every value, so
            // the arms after the last of them never run; typechecking reports
            // them as unreachable. Dropping them leaves no path on which every
            // arm fails.
            let arms = match arms.iter().rposition(|arm| arm.guard.is_none()) {
                Some(last) if exhaustive => &arms[..=last],
                _ => &arms[..],
            };
            lower_case_chain(
                builder,
                location,
                scrutinee,
                result_ty,
                arms,
                exhaustive,
                declarations,
            )
        }
        ExprKind::Resume { arg, local_id } => {
            let token = builder.ctx.lookup_resume(local_id?)?;
            let (input_ty, _) =
                tribute_control::resume_token_parts(builder.ir, builder.ir.value_ty(token))
                    .expect("typechecked resume local must lower to a resume token");
            let value = lower_expr(builder, arg, declarations)?;
            let value = builder.cast_if_needed(location, value, input_ty);
            let resume =
                tribute_control::Resume::operands(token, value).build(builder.ir, location);
            attach_evidence_plan(
                builder.ctx,
                builder.ir,
                resume.op_ref(),
                expr.id,
                declarations,
            );
            builder.ir.push_op(builder.block, resume.op_ref());
            Some(resume.result(builder.ir))
        }
        ExprKind::Error => Some(builder.emit_nil(location)),
        _ => panic!("unsupported source expression at source-logical boundary"),
    }
}

fn lower_constructor<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    id: crate::ast::NodeId,
    ctor: TypedRef<'db>,
    args: Vec<Expr<TypedRef<'db>>>,
    declarations: &mut Declarations<'db>,
) -> Option<ValueRef> {
    let ResolvedRef::Constructor { variant, .. } = ctor.resolved.clone() else {
        panic!("non-constructor in source logical constructor expression");
    };
    let values = args
        .into_iter()
        .map(|arg| lower_expr(builder, arg, declarations))
        .collect::<Option<Vec<_>>>()?;
    let result_ty = expr_type_for_id(builder, id);
    let type_attr = super::resolve_enum_type_attr_for_constructor(
        builder.ctx,
        builder.ir,
        &ctor.resolved,
        ctor.ty,
    );
    let values = super::expr::cast_variant_args(builder, location, values, type_attr, &variant);
    let tag = builder.ir.intern_symbol_text(&variant);
    let variant = adt::VariantNew::operands(values)
        .r#type(type_attr)
        .tag(tag)
        .results(result_ty)
        .build(builder.ir, location);
    builder.ir.push_op(builder.block, variant.op_ref());
    Some(variant.result(builder.ir))
}

fn lower_tuple<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    id: crate::ast::NodeId,
    elements: Vec<Expr<TypedRef<'db>>>,
    declarations: &mut Declarations<'db>,
) -> Option<ValueRef> {
    let values = elements
        .into_iter()
        .map(|element| lower_expr(builder, element, declarations))
        .collect::<Option<Vec<_>>>()?;
    let (_, type_attr) = super::get_or_create_logical_tuple_type(builder.ctx, builder.ir, id)
        .unwrap_or_else(|| panic!("missing typechecked tuple layout"));
    let result_ty = expr_type_for_id(builder, id);
    let tuple = adt::StructNew::operands(values)
        .r#type(type_attr)
        .results(result_ty)
        .build(builder.ir, location);
    builder.ir.push_op(builder.block, tuple.op_ref());
    Some(tuple.result(builder.ir))
}

fn lower_list<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    id: crate::ast::NodeId,
    elements: Vec<Expr<TypedRef<'db>>>,
    declarations: &mut Declarations<'db>,
) -> Option<ValueRef> {
    let values = elements
        .into_iter()
        .map(|element| lower_expr(builder, element, declarations))
        .collect::<Option<Vec<_>>>()?;
    let list_source = builder
        .ctx
        .get_node_type(id)
        .copied()
        .unwrap_or_else(|| panic!("missing typechecked list type"));
    let TypeKind::Named {
        id: list_id, args, ..
    } = list_source.kind(builder.db())
    else {
        panic!("source list expression did not have List type");
    };
    if !list_id.is_builtin_list(builder.db()) || args.len() != 1 {
        panic!("source list expression did not have unary built-in List type");
    }
    let element_ty = builder.ctx.convert_logical_type(builder.ir, args[0]);
    let list_ty = expr_type_for_id(builder, id);
    let empty = list::Empty::operands()
        .element_type(element_ty)
        .results(list_ty)
        .build(builder.ir, location);
    builder.ir.push_op(builder.block, empty.op_ref());
    let mut value = empty.result(builder.ir);
    for element in values.into_iter().rev() {
        let prepend = list::Prepend::operands(element, value)
            .element_type(element_ty)
            .build(builder.ir, location);
        builder.ir.push_op(builder.block, prepend.op_ref());
        value = prepend.result(builder.ir);
    }
    Some(value)
}

fn lower_record<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    id: crate::ast::NodeId,
    type_name: TypedRef<'db>,
    fields: Vec<FieldInit<TypedRef<'db>>>,
    spread: Option<Expr<TypedRef<'db>>>,
    declarations: &mut Declarations<'db>,
) -> Option<ValueRef> {
    if let ResolvedRef::Constructor { variant, .. } = type_name.resolved.clone() {
        let layout = super::resolve_enum_type_attr_for_constructor(
            builder.ctx,
            builder.ir,
            &type_name.resolved,
            type_name.ty,
        );
        if tribute_ir::dialect::adt::layout::get_enum_variants(builder.ir, layout).is_some() {
            return lower_variant_record(
                builder,
                location,
                id,
                layout,
                &variant,
                fields,
                declarations,
            );
        }
    }
    let ctor = super::extract_ctor_id(&type_name.resolved);
    let struct_name = super::extract_type_name(builder.db(), &type_name.resolved);
    let field_order = builder
        .ctx
        .get_struct_field_order(ctor)
        .cloned()
        .unwrap_or_else(|| panic!("prescan did not register struct field order {struct_name}"));
    let layout = builder
        .ctx
        .get_type(&super::qualified_type_name(builder.db(), &ctor))
        .unwrap_or_else(|| panic!("prescan did not register struct layout {struct_name}"));
    // A record spread is evaluated before every explicit field.  The values
    // are nevertheless assembled in declaration layout order below, so these
    // two ordering concerns must stay separate.
    let spread = match spread {
        Some(spread) => Some(lower_expr(builder, spread, declarations)?),
        None => None,
    };
    // Lower explicit fields in source order, then place their already-evaluated
    // values in declaration layout order.
    let mut values = HashMap::default();
    for FieldInit {
        name, value: field, ..
    } in fields
    {
        if !field_order.contains(&name) || values.contains_key(&name) {
            panic!("typechecked record has an invalid field layout");
        }
        values.insert(name, lower_expr(builder, field, declarations)?);
    }
    let mut ordered = Vec::with_capacity(field_order.len());
    for (index, name) in field_order.iter().enumerate() {
        if let Some(value) = values.get(name) {
            ordered.push(*value);
        } else {
            let base =
                spread.unwrap_or_else(|| panic!("typechecked record is missing field {name}"));
            // The layout owns concrete field types.  `struct_get` needs a
            // result type, obtained from the matching getter expression type
            // only after normal typechecking; use layout metadata directly.
            let field_types =
                tribute_ir::dialect::adt::layout::get_struct_fields(builder.ir, layout)
                    .unwrap_or_else(|| panic!("prescanned struct layout is malformed"));
            let get = adt::StructGet::operands(base)
                .r#type(layout)
                .field(index as u32)
                .results(field_types[index].1)
                .build(builder.ir, location);
            builder.ir.push_op(builder.block, get.op_ref());
            ordered.push(get.result(builder.ir));
        }
    }
    let result_ty = expr_type_for_id(builder, id);
    let record = adt::StructNew::operands(ordered)
        .r#type(layout)
        .results(result_ty)
        .build(builder.ir, location);
    builder.ir.push_op(builder.block, record.op_ref());
    Some(record.result(builder.ir))
}

/// Lower a record literal that constructs a named-field enum variant. Fields
/// are evaluated in source order and assembled in declaration order; type
/// checking rejects spreads, so every field is written.
fn lower_variant_record<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    id: crate::ast::NodeId,
    layout: TypeRef,
    variant: &Symbol,
    fields: Vec<FieldInit<TypedRef<'db>>>,
    declarations: &mut Declarations<'db>,
) -> Option<ValueRef> {
    let field_order = builder
        .ctx
        .variant_field_names(layout, variant.clone())
        .unwrap_or_else(|| panic!("prescan did not register field names of variant {variant}"));
    let mut values = HashMap::default();
    for FieldInit {
        name, value: field, ..
    } in fields
    {
        if !field_order.contains(&name) || values.contains_key(&name) {
            panic!("typechecked variant record has an invalid field layout");
        }
        values.insert(name, lower_expr(builder, field, declarations)?);
    }
    let ordered = field_order
        .iter()
        .map(|name| {
            *values
                .get(name)
                .unwrap_or_else(|| panic!("typechecked variant record is missing field {name}"))
        })
        .collect();
    let ordered = super::expr::cast_variant_args(builder, location, ordered, layout, variant);
    let result_ty = expr_type_for_id(builder, id);
    let tag = builder.ir.intern_symbol_text(variant);
    let value = adt::VariantNew::operands(ordered)
        .r#type(layout)
        .tag(tag)
        .results(result_ty)
        .build(builder.ir, location);
    builder.ir.push_op(builder.block, value.op_ref());
    Some(value.result(builder.ir))
}

fn expr_type_for_id(builder: &mut IrBuilder<'_, '_>, id: crate::ast::NodeId) -> TypeRef {
    builder
        .ctx
        .get_node_type(id)
        .copied()
        .map(|ty| builder.ctx.convert_logical_type(builder.ir, ty))
        .unwrap_or_else(|| panic!("missing typechecked expression type"))
}

/// Lower source pattern selection without introducing a control-dialect
/// conditional.  Pattern tests and guards are ordinary structured values, so
/// the logical boundary keeps them in `scf.if`.
fn lower_case_chain<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    scrutinee: ValueRef,
    result_ty: TypeRef,
    arms: &[Arm<TypedRef<'db>>],
    exhaustive: bool,
    declarations: &mut Declarations<'db>,
) -> Option<ValueRef> {
    match arms {
        // Typechecking rejects a case it does not prove exhaustive, and an
        // exhaustive chain ends with an unguarded arm lowered without a test.
        [] => panic!("a case whose arms can all fail reached lowering"),
        [last] if exhaustive && last.guard.is_none() => {
            let mut scope = builder.ctx.scope();
            super::case::bind_logical_pattern_fields(
                &mut scope,
                builder.ir,
                builder.block,
                location,
                scrutinee,
                &last.pattern,
            );
            lower_expr(
                &mut IrBuilder::new(&mut scope, builder.ir, builder.block),
                last.body.clone(),
                declarations,
            )
        }
        [first, rest @ ..] => {
            let condition = super::case::emit_logical_pattern_check(
                builder,
                location,
                scrutinee,
                &first.pattern,
            )?;
            let then_region = build_case_arm_region(
                builder.ctx,
                builder.ir,
                CaseArmRequest {
                    location,
                    scrutinee,
                    arm: first,
                    rest,
                    result_ty,
                    exhaustive,
                },
                declarations,
            )?;
            let else_region = build_case_else_region(
                builder.ctx,
                builder.ir,
                location,
                scrutinee,
                rest,
                result_ty,
                exhaustive,
                declarations,
            )?;
            let branch = scf::If::operands(condition)
                .results(result_ty)
                .regions(then_region, else_region)
                .build(builder.ir, location);
            builder.ir.push_op(builder.block, branch.op_ref());
            Some(branch.result(builder.ir))
        }
    }
}

struct CaseArmRequest<'a, 'db> {
    location: Location,
    scrutinee: ValueRef,
    arm: &'a Arm<TypedRef<'db>>,
    rest: &'a [Arm<TypedRef<'db>>],
    result_ty: TypeRef,
    exhaustive: bool,
}

fn build_case_arm_region<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    request: CaseArmRequest<'_, 'db>,
    declarations: &mut Declarations<'db>,
) -> Option<trunk_ir::refs::RegionRef> {
    let CaseArmRequest {
        location,
        scrutinee,
        arm,
        rest,
        result_ty,
        exhaustive,
    } = request;
    let block = ir.create_block(BlockData {
        location,
        args: vec![],
        ops: Default::default(),
        parent_region: None,
    });
    let value = {
        let mut scope = ctx.scope();
        super::case::bind_logical_pattern_fields(
            &mut scope,
            ir,
            block,
            location,
            scrutinee,
            &arm.pattern,
        );
        let mut nested = IrBuilder::new(&mut scope, ir, block);
        if let Some(guard) = &arm.guard {
            (|| {
                let condition = lower_expr(&mut nested, guard.clone(), declarations)?;
                let then_region = build_case_body_region(
                    nested.ctx,
                    nested.ir,
                    location,
                    arm.body.clone(),
                    result_ty,
                    declarations,
                )?;
                let else_region = build_case_else_region(
                    nested.ctx,
                    nested.ir,
                    location,
                    scrutinee,
                    rest,
                    result_ty,
                    exhaustive,
                    declarations,
                )?;
                let branch = scf::If::operands(condition)
                    .results(result_ty)
                    .regions(then_region, else_region)
                    .build(nested.ir, location);
                nested.ir.push_op(nested.block, branch.op_ref());
                Some(branch.result(nested.ir))
            })()
        } else {
            lower_expr(&mut nested, arm.body.clone(), declarations)
        }
    };
    finish_case_region(ctx, ir, location, block, value, result_ty)
}

/// End a structured case region with its value, or leave it ended by a
/// source `become`.
fn finish_case_region(
    ctx: &mut IrLoweringCtx<'_>,
    ir: &mut IrContext,
    location: Location,
    block: BlockRef,
    value: Option<ValueRef>,
    result_ty: TypeRef,
) -> Option<trunk_ir::refs::RegionRef> {
    if value.is_some() || !ends_with_tail_call(ir, block) {
        let value = IrBuilder::new(ctx, ir, block).cast_if_needed(location, value?, result_ty);
        let yield_op = scf::Yield::operands([value]).build(ir, location);
        ir.push_op(block, yield_op.op_ref());
    }
    Some(ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![block],
        parent_op: None,
    }))
}

fn build_case_body_region<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    location: Location,
    body: Expr<TypedRef<'db>>,
    result_ty: TypeRef,
    declarations: &mut Declarations<'db>,
) -> Option<trunk_ir::refs::RegionRef> {
    let block = ir.create_block(BlockData {
        location,
        args: vec![],
        ops: Default::default(),
        parent_region: None,
    });
    let value = lower_expr(&mut IrBuilder::new(ctx, ir, block), body, declarations);
    finish_case_region(ctx, ir, location, block, value, result_ty)
}

#[allow(clippy::too_many_arguments)]
fn build_case_else_region<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    location: Location,
    scrutinee: ValueRef,
    arms: &[Arm<TypedRef<'db>>],
    result_ty: TypeRef,
    exhaustive: bool,
    declarations: &mut Declarations<'db>,
) -> Option<trunk_ir::refs::RegionRef> {
    let block = ir.create_block(BlockData {
        location,
        args: vec![],
        ops: Default::default(),
        parent_region: None,
    });
    let value = lower_case_chain(
        &mut IrBuilder::new(ctx, ir, block),
        location,
        scrutinee,
        result_ty,
        arms,
        exhaustive,
        declarations,
    );
    finish_case_region(ctx, ir, location, block, value, result_ty)
}

fn emit_bool(builder: &mut IrBuilder<'_, '_>, location: Location, value: bool) -> ValueRef {
    let ty = builder.ctx.bool_type(builder.ir);
    let op = arith::Const::operands()
        .value(Attribute::Bool(value))
        .results(ty)
        .build(builder.ir, location);
    builder.ir.push_op(builder.block, op.op_ref());
    op.result(builder.ir)
}

/// Declare on the lambda `binding` binds the local binders its uses
/// instantiate.
fn quantify_local_rows<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    binding: crate::ast::NodeId,
    lambda: ValueRef,
    declarations: &Declarations<'db>,
) {
    let Some(use_) = declarations
        .local_uses
        .iter()
        .find(|use_| use_.binding == binding && !use_.rows.is_empty())
    else {
        return;
    };
    let trunk_ir::refs::ValueDef::OpResult(op, _) = builder.ir.value_def(lambda) else {
        return;
    };
    let binders: Vec<_> = use_
        .rows
        .iter()
        .map(|(variable, _)| {
            let tail = builder
                .ctx
                .ability_row(AbilityRow::open(builder.ctx.db, *variable))
                .tail
                .expect("an open row has a tail");
            Attribute::Int(i128::from(tail))
        })
        .collect();
    builder.ir.op_attrs_mut(op).insert(
        tribute_control::QUANTIFIED_ROWS_ATTR,
        Attribute::List(binders),
    );
}

/// The value the local reference `node` reads: a lambda that quantifies
/// rows is instantiated with the rows this use passes.
fn lower_local_reference<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    node: crate::ast::NodeId,
    local: crate::ast::LocalId,
    declarations: &Declarations<'db>,
) -> Option<ValueRef> {
    let value = local_callables::lookup(builder.ctx, node, declarations)
        .or_else(|| builder.ctx.lookup(local))?;
    let trunk_ir::refs::ValueDef::OpResult(lambda, _) = builder.ir.value_def(value) else {
        return Some(value);
    };
    let quantified = tribute_control::quantified_rows(builder.ir, lambda)
        .expect("a lowered lambda has well-formed quantified rows");
    if quantified.is_empty() || !tribute_control::Lambda::matches(builder.ir, lambda) {
        return Some(value);
    }
    let use_ = declarations
        .local_uses
        .iter()
        .find(|use_| use_.site == node)
        .expect("typeck records an instance for every use of a binding or for none");
    let rows: Vec<_> = use_
        .rows
        .iter()
        .map(|(_, row)| builder.ctx.ability_row(*row))
        .collect();
    let ty = tribute_control::substitute_rows(
        builder.ir,
        builder.ir.value_ty(value),
        tribute_control::RowArgs::Binders(&quantified, &rows),
    )
    .expect("instantiation substitutes only the lambda's binders");
    let location = builder.location(node);
    let instantiate = tribute_control::Instantiate::operands(value)
        .row_args(rows)
        .results(ty)
        .build(builder.ir, location);
    builder.ir.push_op(builder.block, instantiate.op_ref());
    Some(instantiate.result(builder.ir))
}

fn lower_statement<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    statement: Stmt<TypedRef<'db>>,
    declarations: &mut Declarations<'db>,
) {
    match statement {
        Stmt::Let { pattern, value, .. } => {
            let lowered = local_callables::materialize(builder, &pattern, &value, declarations)
                .or_else(|| {
                    let lambda = matches!(&*value.kind, ExprKind::Lambda { .. });
                    let lowered = lower_expr(builder, value, declarations)?;
                    if lambda {
                        quantify_local_rows(builder, pattern.id, lowered, declarations);
                    }
                    Some(lowered)
                });
            if let Some(value) = lowered {
                bind_pattern(builder, &pattern, value);
            }
        }
        Stmt::Expr { expr, .. } => {
            let _ = lower_expr(builder, expr, declarations);
        }
    }
}

fn bind_pattern<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    pattern: &Pattern<TypedRef<'db>>,
    value: ValueRef,
) {
    match &*pattern.kind {
        PatternKind::Bind {
            name,
            local_id: Some(id),
        } => builder.ctx.bind(*id, name.clone(), value),
        PatternKind::Wildcard => {}
        _ => super::case::bind_logical_pattern_fields(
            builder.ctx,
            builder.ir,
            builder.block,
            builder.location(pattern.id),
            value,
            pattern,
        ),
    }
}

/// Call the named source function, casting each argument to its parameter
/// type. Returns the call's result at the callee's logical result type.
pub(super) fn emit_named_call(
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
enum CallPosition {
    Value,
    Tail,
}

/// Call `name`, whose signature is `declared` in its own binders, passing
/// the row arguments `args`.
fn named_call(
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
fn indirect_call(
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
fn lower_become<'db>(
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
fn lower_call<'db>(
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
fn finish_call<'db>(
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

fn lower_lambda<'db>(
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

fn lower_handle<'db>(
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

fn lower_handler<'db>(
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

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::context::BlockData;
    use trunk_ir::location::Span;

    #[salsa::tracked(returns(copy))]
    fn operation_arguments_use_resolved_parameter_types_inner(db: &dyn salsa::Database) -> bool {
        let mut ir = IrContext::new();
        let path = ir.intern_path("logical.trb");
        let mut ctx = IrLoweringCtx::new(
            db,
            path,
            crate::ast::SpanMap::default(),
            HashMap::default(),
            SortedMap::default(),
        );
        let location = Location::new(path, Span::new(0, 0));
        let block = ir.create_block(BlockData {
            location,
            args: vec![],
            ops: Default::default(),
            parent_region: None,
        });
        let anyref = ctx.anyref_type(&mut ir);
        let value = adt::RefNull::operands()
            .r#type(anyref)
            .results(anyref)
            .build(&mut ir, location);
        ir.push_op(block, value.op_ref());
        let value = value.result(&ir);
        let parameter = ctx.adt_typeref(&mut ir, &Symbol::new("String"));
        let declaration = OperationDeclaration::new(
            SymbolPath::new(["Test"]),
            ir.intern_str("throw"),
            ir.intern_str("op"),
            [parameter],
            parameter,
        );
        let mut scope = ctx.scope();
        let mut builder = IrBuilder::new(&mut scope, &mut ir, block);
        let adapted = adapt_operation_arguments_or_recover(
            &mut builder,
            location,
            vec![value],
            &declaration,
            &Symbol::new("Test"),
            &Symbol::new("throw"),
        )
        .expect("matching operation arity should adapt arguments");
        assert_eq!(builder.ir.value_ty(adapted[0]), parameter);

        let mismatch = OperationDeclaration::new(
            SymbolPath::new(["Test"]),
            declaration.op_name,
            declaration.kind,
            [],
            parameter,
        );
        let recovered = adapt_operation_arguments_or_recover(
            &mut builder,
            location,
            vec![value],
            &mismatch,
            &Symbol::new("Test"),
            &Symbol::new("throw"),
        )
        .expect_err("mismatched operation arity should recover with nil");
        let nil_type = builder.ctx.nil_type(builder.ir);
        assert_eq!(builder.ir.value_ty(recovered), nil_type);
        true
    }

    #[test]
    fn operation_arguments_use_resolved_parameter_types() {
        let db = salsa::DatabaseImpl::new();
        assert!(operation_arguments_use_resolved_parameter_types_inner(&db));
        let diagnostics =
            operation_arguments_use_resolved_parameter_types_inner::accumulated::<Diagnostic>(&db);
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(
            diagnostics[0].inner.message,
            "ability operation `Test::throw` has 1 arguments, expected 0"
        );
    }
}
