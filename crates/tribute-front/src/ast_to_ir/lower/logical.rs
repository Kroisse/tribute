//! Source-logical AST to TrunkIR lowering.
//!
//! This is intentionally separate from the superseded frontend CPS lowering.
//! It emits only the documented `tribute_control` boundary and ordinary value
//! dialects; shared CPS construction belongs to `tribute-passes`.

use std::collections::{HashMap, HashSet};

use salsa::Accumulator;
use tribute_core::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use tribute_ir::dialect::{
    list,
    tribute_control::{self, CompilerIntrinsicDeclaration, OperationDeclaration},
};
use trunk_ir::Symbol;
use trunk_ir::context::{BlockArgData, BlockData, IrContext, OperationDataBuilder, RegionData};
use trunk_ir::dialect::{adt, arith, core, scf};
use trunk_ir::ops::DialectType;
use trunk_ir::refs::{BlockRef, OpRef, PathRef, TypeRef, ValueRef};
use trunk_ir::rewrite::Module as IrModule;
use trunk_ir::types::{Attribute, Location};

use crate::ast::{
    Arm, CallingConvention, CtorId, Decl, EffectRow, Expr, ExprKind, ExternFuncDecl, FuncDecl,
    HandlerArm, HandlerKind, OpDeclKind, Pattern, PatternKind, ResolvedRef, Stmt, TypeKind,
    TypedRef,
};

use super::super::context::IrLoweringCtx;
use super::super::{FrontendIrModule, TypedModule};
use super::{FuncSignature, IrBuilder, expr};

struct Declarations<'db> {
    // TypeRef is an arena index and therefore has deterministic total order only
    // through its debug representation.  Preserve first source use explicitly.
    values: Vec<OperationDeclaration>,
    compiler_intrinsics: Vec<CompilerIntrinsicDeclaration>,
    schemas: std::collections::HashMap<crate::ast::AbilityId<'db>, crate::typeck::AbilityInfo<'db>>,
    handler_operations: std::collections::HashMap<
        crate::ast::NodeId,
        crate::typeck::InstantiatedHandlerOperation<'db>,
    >,
    perform_operations: std::collections::HashMap<
        crate::ast::NodeId,
        crate::typeck::InstantiatedPerformOperation<'db>,
    >,
    lambda_signatures:
        std::collections::HashMap<crate::ast::NodeId, crate::typeck::LambdaSignature<'db>>,
    exhaustive_cases: std::collections::HashSet<crate::ast::NodeId>,
}

/// Fully typed semantic inputs for one source ability-operation call.
struct PerformMetadata<'a, 'db> {
    ability: crate::ast::AbilityId<'db>,
    operation: Symbol,
    kind: OpDeclKind,
    semantic: &'a crate::typeck::InstantiatedPerformOperation<'db>,
}

impl<'db> Declarations<'db> {
    fn record(
        &mut self,
        declaration: OperationDeclaration,
        location: Location,
        db: &dyn salsa::Database,
    ) {
        if let Some(existing) = self.values.iter().find(|candidate| {
            candidate.ability_ref == declaration.ability_ref
                && candidate.op_name == declaration.op_name
        }) {
            if existing != &declaration {
                Diagnostic::new(
                    format!(
                        "conflicting instantiated declaration for ability operation {}",
                        declaration.op_name
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
    name: &str,
    build: impl FnOnce(OperationDataBuilder) -> OperationDataBuilder,
) -> OpRef {
    let data = build(OperationDataBuilder::new(
        location,
        Symbol::new("tribute_control"),
        Symbol::from_dynamic(name),
    ))
    .build(ir);
    let op = ir.create_op(data);
    ir.push_op(block, op);
    op
}

fn result(ir: &IrContext, op: OpRef) -> ValueRef {
    ir.op_result(op, 0)
}

fn control_convention(convention: CallingConvention) -> tribute_control::CallingConvention {
    match convention {
        CallingConvention::Direct => tribute_control::CallingConvention::Direct,
        CallingConvention::EvidenceDirect => tribute_control::CallingConvention::EvidenceDirect,
        CallingConvention::Cps => tribute_control::CallingConvention::Cps,
    }
}

fn callable_type(
    ir: &mut IrContext,
    result: TypeRef,
    params: impl IntoIterator<Item = TypeRef>,
    convention: CallingConvention,
) -> TypeRef {
    tribute_control::callable(ir, result, params, control_convention(convention)).as_type_ref()
}

fn declaration_name(prefix: &mut String, name: Symbol) -> Symbol {
    if name.with_str(|text| text.contains("::")) {
        name
    } else {
        crate::qualified_symbol(prefix, name)
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
        node_types,
        ability_conventions,
        ability_definitions,
        handler_operations,
        perform_operations,
        lambda_signatures,
        exhaustive_cases,
        well_known_types,
        compiler_intrinsics,
    } = typed;
    let location = Location::new(path, span_map.get_or_default(ast.id));
    let module_name = ast.name.unwrap_or_else(|| Symbol::new("main"));
    let mut ctx = IrLoweringCtx::new(
        db,
        path,
        span_map,
        function_types,
        ability_conventions,
        smallvec::smallvec![module_name],
        node_types,
    )
    .with_compiler_intrinsics(compiler_intrinsics);
    let module_block = ir.create_block(BlockData {
        location,
        args: vec![],
        ops: Default::default(),
        parent_region: None,
    });
    ctx.set_module_block(module_block);
    let mut declarations = Declarations {
        values: vec![],
        compiler_intrinsics: vec![],
        schemas: ability_definitions,
        handler_operations,
        perform_operations,
        lambda_signatures,
        exhaustive_cases,
    };
    prescan_definition_conventions(&mut ctx, &ast.decls, &mut String::new());
    promote_definition_conventions_to_fixed_point(&mut ctx, &ast.decls, &mut String::new());
    let mut well_known_type_prescan = super::decl::WellKnownTypePrescan::new(well_known_types);
    collect_logical_nominal_identities(&mut ctx, &ast.decls, &mut String::new());
    prescan_logical_nominal_layouts(
        &mut ctx,
        ir,
        &ast.decls,
        &mut String::new(),
        &mut well_known_type_prescan,
        &constructor_types,
    );
    prescan_struct_accessor_signatures(&mut ctx, ir, &ast.decls);
    prescan_source_functions(&mut ctx, &ast.decls);
    let well_known_types = well_known_type_prescan.finish();
    for declaration in ast.decls {
        lower_decl(&mut ctx, ir, module_block, declaration, &mut declarations);
    }
    let region = ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![module_block],
        parent_op: None,
    });
    let module = core::module(ir, location, module_name, region);
    well_known_types.attach(ir, module.op_ref());
    FrontendIrModule {
        module: IrModule::new(ir, module.op_ref()).expect("valid core.module operation"),
        operation_declarations: declarations.values,
        compiler_intrinsics: declarations.compiler_intrinsics,
    }
}

/// Seed worker conventions from each body's concrete residual effects. The
/// semantic function type remains untouched: an omitted annotation can still
/// expose an open-row callable at a first-class call site.
fn prescan_definition_conventions<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    declarations: &[Decl<TypedRef<'db>>],
    prefix: &mut String,
) {
    for declaration in declarations {
        match declaration {
            Decl::Function(function) => {
                let name = declaration_name(prefix, function.name);
                let Some(scheme) = ctx.lookup_function_type(name).copied() else {
                    continue;
                };
                let body = scheme.body(ctx.db);
                let Some(mut convention) = ctx.calling_convention_for_type(body) else {
                    continue;
                };
                if (function.effects.is_none()
                    || crate::is_root_main(function.name, prefix.is_empty()))
                    && let TypeKind::Func { effect, .. } = body.kind(ctx.db)
                {
                    convention = ctx.calling_convention_for_effect_row(EffectRow::new(
                        ctx.db,
                        effect.effects(ctx.db).clone(),
                        None,
                    ));
                }
                ctx.register_definition_convention(name, convention);
            }
            Decl::Module(module) => {
                if let Some(body) = &module.body {
                    let saved = crate::push_prefix(prefix, module.name);
                    prescan_definition_conventions(ctx, body, prefix);
                    prefix.truncate(saved);
                }
            }
            _ => {}
        }
    }
}

/// Strengthen workers until direct calls and structured logical evaluation no
/// longer leave a Direct/EvidenceDirect worker responsible for CPS control.
fn promote_definition_conventions_to_fixed_point<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    declarations: &[Decl<TypedRef<'db>>],
    prefix: &mut String,
) {
    loop {
        let mut changed = false;
        promote_definition_conventions_pass(ctx, declarations, prefix, &mut changed);
        if !changed {
            return;
        }
    }
}

fn promote_definition_conventions_pass<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    declarations: &[Decl<TypedRef<'db>>],
    prefix: &mut String,
    changed: &mut bool,
) {
    for declaration in declarations {
        match declaration {
            Decl::Function(function) => {
                let name = declaration_name(prefix, function.name);
                if ctx.function_calling_convention(name) != Some(CallingConvention::Cps)
                    && expr::logical_evaluation_control_class(ctx, &function.body)
                        == expr::EvaluationControlClass::Cps
                {
                    ctx.register_definition_convention(name, CallingConvention::Cps);
                    *changed = true;
                }
            }
            Decl::Module(module) => {
                if let Some(body) = &module.body {
                    let saved = crate::push_prefix(prefix, module.name);
                    promote_definition_conventions_pass(ctx, body, prefix, changed);
                    prefix.truncate(saved);
                }
            }
            _ => {}
        }
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
    declaration: Decl<TypedRef<'db>>,
    declarations: &mut Declarations<'db>,
) {
    match declaration {
        Decl::Function(function) => lower_function(ctx, ir, top, function, declarations),
        Decl::ExternFunction(function) => lower_extern(ctx, ir, top, function, declarations),
        Decl::Struct(declaration) => lower_struct_accessors(ctx, ir, top, declaration),
        Decl::Module(module) => {
            if let Some(body) = module.body {
                ctx.enter_module(module.name);
                for declaration in body {
                    lower_decl(ctx, ir, top, declaration, declarations);
                }
                ctx.exit_module();
            }
        }
        // Nominal declarations remain types used by ordinary ADT value ops.
        // Struct accessors are emitted above as source-logical callables.
        Decl::Enum(_) | Decl::Ability(_) | Decl::Use(_) => {}
    }
}

/// Register nominal layouts used exclusively by source-logical lowering.
///
/// Legacy lowering retains `decl::prescan_struct_fields`, whose annotation
/// converter deliberately erases physical callable representation.  Here the
/// constructor schemes are the authoritative semantic field types, so records,
/// constructors, accessors, and pattern extraction share the same recursive
/// `tribute_control` callable layouts.
fn prescan_logical_nominal_layouts<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    declarations: &[Decl<TypedRef<'db>>],
    prefix: &mut String,
    well_known_types: &mut super::decl::WellKnownTypePrescan,
    constructors: &HashMap<crate::ast::CtorId<'db>, crate::ast::TypeScheme<'db>>,
) {
    for declaration in declarations {
        match declaration {
            Decl::Struct(structure) => {
                let qualified = crate::qualified_symbol(prefix, structure.name);
                let ctor = CtorId::new(ctx.db, qualified);
                let field_types =
                    constructor_fields(ctx, constructors, ctor, structure.fields.len());
                let fields = structure
                    .fields
                    .iter()
                    .zip(field_types)
                    .map(|(field, ty)| {
                        (
                            field.name.unwrap_or_else(|| Symbol::new("_")),
                            ctx.convert_logical_type(ir, ty),
                        )
                    })
                    .collect::<Vec<_>>();
                let name = super::qualified_type_name(ctx.db, &ctor);
                let layout = ctx.adt_struct_type(ir, name, &fields);
                ctx.register_type(name, layout);
            }
            Decl::Enum(enumeration) => {
                let qualified = crate::qualified_symbol(prefix, enumeration.name);
                let enum_ctor = CtorId::new(ctx.db, qualified);
                let variants = enumeration
                    .variants
                    .iter()
                    .map(|variant| {
                        let variant_ctor =
                            CtorId::new(ctx.db, crate::qualified_symbol(prefix, variant.name));
                        let fields = constructor_fields(
                            ctx,
                            constructors,
                            variant_ctor,
                            variant.fields.len(),
                        )
                        .into_iter()
                        .map(|ty| ctx.convert_logical_type(ir, ty))
                        .collect();
                        (variant.name, fields)
                    })
                    .collect::<Vec<_>>();
                let name = super::qualified_type_name(ctx.db, &enum_ctor);
                let definition = crate::typeck::DefinitionIdentity::new(
                    enumeration.id,
                    ctx.location(enumeration.id).span,
                );
                let layout = if well_known_types.is_string(definition) {
                    ctx.adt_enum_type_with_definition(ir, name, &variants, definition)
                } else {
                    ctx.adt_enum_type(ir, name, &variants)
                };
                ctx.register_type(name, layout);
                if well_known_types.is_string(definition) {
                    well_known_types.record_string(layout);
                }
            }
            Decl::Module(module) => {
                if let Some(body) = &module.body {
                    let saved = crate::push_prefix(prefix, module.name);
                    prescan_logical_nominal_layouts(
                        ctx,
                        ir,
                        body,
                        prefix,
                        well_known_types,
                        constructors,
                    );
                    prefix.truncate(saved);
                }
            }
            Decl::Function(_) | Decl::ExternFunction(_) | Decl::Ability(_) | Decl::Use(_) => {}
        }
    }
}

/// First phase of logical nominal prescan. All identities must be known before
/// converting any field: recursive and forward references are logical nominal
/// values even while their concrete layout is still being assembled.
fn collect_logical_nominal_identities<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    declarations: &[Decl<TypedRef<'db>>],
    prefix: &mut String,
) {
    for declaration in declarations {
        match declaration {
            Decl::Struct(structure) => {
                let qualified = crate::qualified_symbol(prefix, structure.name);
                let ctor = CtorId::new(ctx.db, qualified);
                ctx.declare_logical_nominal(super::qualified_type_name(ctx.db, &ctor));
                ctx.register_struct_fields(
                    ctor,
                    structure
                        .fields
                        .iter()
                        .map(|field| field.name.unwrap_or_else(|| Symbol::new("_")))
                        .collect(),
                );
            }
            Decl::Enum(enumeration) => {
                let qualified = crate::qualified_symbol(prefix, enumeration.name);
                let ctor = CtorId::new(ctx.db, qualified);
                ctx.declare_logical_nominal(super::qualified_type_name(ctx.db, &ctor));
            }
            Decl::Module(module) => {
                if let Some(body) = &module.body {
                    let saved = crate::push_prefix(prefix, module.name);
                    collect_logical_nominal_identities(ctx, body, prefix);
                    prefix.truncate(saved);
                }
            }
            Decl::Function(_) | Decl::ExternFunction(_) | Decl::Ability(_) | Decl::Use(_) => {}
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

fn prescan_struct_accessor_signatures<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    declarations: &[Decl<TypedRef<'db>>],
) {
    for declaration in declarations {
        match declaration {
            Decl::Struct(declaration) => {
                let mut prefix = String::new();
                for segment in ctx.module_path().iter().skip(1) {
                    crate::push_prefix(&mut prefix, *segment);
                }
                let qualified = crate::qualified_symbol(&mut prefix, declaration.name);
                let type_name = super::qualified_type_name(ctx.db, &CtorId::new(ctx.db, qualified));
                let struct_type = ctx.adt_typeref(ir, type_name);
                let layout = ctx
                    .get_type(type_name)
                    .unwrap_or_else(|| panic!("missing logical struct layout for accessor"));
                let layout_fields = trunk_ir::adt_layout::get_struct_fields(ir, layout)
                    .unwrap_or_else(|| panic!("malformed logical struct layout"));
                for (index, field) in declaration.fields.iter().enumerate() {
                    let field_name = field.name.unwrap_or_else(|| Symbol::new("_"));
                    let getter_name = if prefix.is_empty() {
                        Symbol::from_dynamic(&format!("{}::{}", declaration.name, field_name))
                    } else {
                        Symbol::from_dynamic(&format!(
                            "{}::{}::{}",
                            prefix, declaration.name, field_name
                        ))
                    };
                    let field_type = layout_fields
                        .get(index)
                        .map(|(_, ty)| *ty)
                        .unwrap_or_else(|| panic!("missing logical struct accessor field type"));
                    ctx.register_logical_generated_signature(
                        getter_name,
                        vec![struct_type],
                        field_type,
                        CallingConvention::Direct,
                    );
                }
            }
            Decl::Module(module) => {
                if let Some(body) = &module.body {
                    ctx.enter_module(module.name);
                    prescan_struct_accessor_signatures(ctx, ir, body);
                    ctx.exit_module();
                }
            }
            Decl::Function(_)
            | Decl::ExternFunction(_)
            | Decl::Enum(_)
            | Decl::Ability(_)
            | Decl::Use(_) => {}
        }
    }
}

fn prescan_source_functions<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    declarations: &[Decl<TypedRef<'db>>],
) {
    for declaration in declarations {
        match declaration {
            Decl::Function(function) => {
                ctx.register_logical_source_function(ctx.qualify_name(function.name));
            }
            Decl::ExternFunction(function) => {
                ctx.register_logical_source_function(ctx.qualify_name(function.name));
            }
            Decl::Module(module) => {
                if let Some(body) = &module.body {
                    ctx.enter_module(module.name);
                    prescan_source_functions(ctx, body);
                    ctx.exit_module();
                }
            }
            Decl::Struct(_) | Decl::Enum(_) | Decl::Ability(_) | Decl::Use(_) => {}
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
    let mut prefix = String::new();
    for segment in ctx.module_path().iter().skip(1) {
        crate::push_prefix(&mut prefix, *segment);
    }
    let qualified = crate::qualified_symbol(&mut prefix, declaration.name);
    let type_name = super::qualified_type_name(ctx.db, &CtorId::new(ctx.db, qualified));
    let struct_type = ctx.adt_typeref(ir, type_name);
    let layout_type = ctx
        .get_type(type_name)
        .unwrap_or_else(|| panic!("prescan did not register struct layout {type_name}"));
    for (index, field) in declaration.fields.iter().enumerate() {
        let field_name = field.name.unwrap_or_else(|| Symbol::new("_"));
        let getter_name = if prefix.is_empty() {
            Symbol::from_dynamic(&format!("{}::{}", declaration.name, field_name))
        } else {
            Symbol::from_dynamic(&format!("{}::{}::{}", prefix, declaration.name, field_name))
        };
        let field_type = trunk_ir::adt_layout::get_struct_fields(ir, layout_type)
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
        let field_value = adt::struct_get(
            ir,
            location,
            ir.block_arg(entry, 0),
            field_type,
            layout_type,
            index as u32,
        );
        ir.push_op(entry, field_value.op_ref());
        let field_result = field_value.result(ir);
        op(ir, entry, location, "return", |builder| {
            builder.operand(field_result)
        });
        let body = ir.create_region(RegionData {
            location,
            blocks: trunk_ir::smallvec::smallvec![entry],
            parent_op: None,
        });
        let callable = callable_type(ir, field_type, [struct_type], CallingConvention::Direct);
        let getter = tribute_control::func_declaration(ir, location, getter_name, callable);
        ir.op_mut(getter.op_ref()).regions.push(body);
        ir.region_mut(body).parent_op = Some(getter.op_ref());
        ir.push_op(top, getter.op_ref());
    }
}

fn function_signature<'db>(
    ctx: &IrLoweringCtx<'db>,
    ir: &mut IrContext,
    function: &FuncDecl<TypedRef<'db>>,
) -> FuncSignature {
    let qualified = ctx.qualify_name(function.name);
    let mut signature = (if qualified == function.name {
        FuncSignature::lookup_logical(ctx, ir, function.name)
    } else {
        // Nested declarations are exported under their qualified identity; a
        // short-name lookup can silently select an unrelated root declaration.
        FuncSignature::lookup_logical(ctx, ir, qualified)
            .or_else(|| FuncSignature::lookup_logical(ctx, ir, function.name))
    })
    .unwrap_or_else(|| {
        panic!(
            "missing typechecked signature for function {}",
            function.name
        )
    });
    signature.convention = ctx
        .function_calling_convention(qualified)
        .unwrap_or(signature.convention);
    signature
}

fn lower_function<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    top: BlockRef,
    function: FuncDecl<TypedRef<'db>>,
    declarations: &mut Declarations<'db>,
) {
    let location = ctx.location(function.id);
    let root_export_convention = crate::is_root_main(function.name, ctx.module_path().len() == 1)
        .then(|| {
            let name = ctx.qualify_name(function.name);
            let scheme = ctx
                .lookup_function_type(name)
                .expect("root main has a typechecked logical signature");
            ctx.calling_convention_for_type(scheme.body(ctx.db))
                .expect("root main has a function type")
        });
    let signature = function_signature(ctx, ir, &function);
    let callable = callable_type(
        ir,
        signature.return_type,
        signature.param_types.iter().copied(),
        signature.convention,
    );
    let entry = ir.create_block(BlockData {
        location,
        args: function
            .params
            .iter()
            .zip(signature.param_types.iter())
            .map(|(parameter, ty)| {
                let mut attrs = trunk_ir::types::AttributeMap::default();
                attrs.insert(Symbol::new("bind_name"), Attribute::Symbol(parameter.name));
                BlockArgData { ty: *ty, attrs }
            })
            .collect(),
        ops: Default::default(),
        parent_region: None,
    });
    {
        let mut scope = ctx.scope();
        for (index, parameter) in function.params.iter().enumerate() {
            if let Some(id) = parameter.local_id {
                scope.bind(id, parameter.name, ir.block_arg(entry, index as u32));
            }
        }
        let value = lower_expr(
            &mut IrBuilder::new(&mut scope, ir, entry),
            function.body,
            declarations,
        )
        .expect("typechecked source expression failed logical IR lowering");
        let value = IrBuilder::new(&mut scope, ir, entry).cast_if_needed(
            location,
            value,
            signature.return_type,
        );
        op(ir, entry, location, "return", |builder| {
            builder.operand(value)
        });
    }
    let body = ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![entry],
        parent_op: None,
    });
    let name = ctx.qualify_name(function.name);
    let function = tribute_control::func_declaration(ir, location, name, callable);
    ir.op_mut(function.op_ref()).regions.push(body);
    ir.region_mut(body).parent_op = Some(function.op_ref());
    if let Some(convention) = root_export_convention
        && convention != signature.convention
    {
        assert_ne!(convention, CallingConvention::Cps);
        ir.op_mut(function.op_ref()).attributes.insert(
            Symbol::new("tribute.root_export_convention"),
            Attribute::Int(convention as i128),
        );
        ir.op_mut(function.op_ref()).attributes.insert(
            Symbol::new("tribute.root_source_result"),
            Attribute::Type(signature.return_type),
        );
    }
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
    let qualified = ctx.qualify_name(decl.name);
    let signature = if qualified == decl.name {
        FuncSignature::lookup_logical(ctx, ir, decl.name)
    } else {
        FuncSignature::lookup_logical(ctx, ir, qualified)
            .or_else(|| FuncSignature::lookup_logical(ctx, ir, decl.name))
    }
    .unwrap_or_else(|| {
        panic!(
            "missing typechecked signature for extern function {}",
            decl.name
        )
    });
    let callable = callable_type(
        ir,
        signature.return_type,
        signature.param_types,
        signature.convention,
    );
    let name = ctx.qualify_name(decl.name);
    let function = tribute_control::func_declaration(ir, location, name, callable);
    if let Some(identity) = ctx.compiler_intrinsic(decl.id) {
        ir.op_mut(function.op_ref()).attributes.insert(
            Symbol::new(tribute_control::COMPILER_INTRINSIC_ATTR),
            Attribute::Symbol(identity),
        );
        declarations
            .compiler_intrinsics
            .push(CompilerIntrinsicDeclaration::new(name, identity, callable));
        declarations
            .compiler_intrinsics
            .sort_by_key(|declaration| (declaration.symbol, declaration.identity));
    }
    ir.op_mut(function.op_ref())
        .attributes
        .insert(Symbol::new("abi"), Attribute::String(decl.abi.to_string()));
    ir.push_op(top, function.op_ref());
}

fn ensure_prelude_declaration(
    builder: &mut IrBuilder<'_, '_>,
    location: Location,
    name: Symbol,
    signature: &FuncSignature,
) {
    if builder.ctx.is_logical_source_function(name)
        || builder
            .ctx
            .lookup_logical_generated_signature(name)
            .is_some()
        || !builder.ctx.mark_logical_extern_emitted(name)
    {
        return;
    }
    let callable = callable_type(
        builder.ir,
        signature.return_type,
        signature.param_types.iter().copied(),
        signature.convention,
    );
    let declaration = tribute_control::func_declaration(builder.ir, location, name, callable);
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

fn call_operation_metadata<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    declarations: &Declarations<'db>,
    metadata: PerformMetadata<'_, 'db>,
) -> (TypeRef, OperationDeclaration) {
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
    let ability_arguments = args
        .into_iter()
        .map(|arg| builder.ctx.convert_logical_type(builder.ir, arg))
        .collect::<Vec<_>>();
    let ability_ref = builder.ctx.ability_ref_type(
        builder.ir,
        ability.qualified(builder.db()),
        &ability_arguments,
    );
    let parameters = semantic
        .params
        .iter()
        .map(|param| builder.ctx.convert_logical_type(builder.ir, *param))
        .collect::<Vec<_>>();
    let result = builder
        .ctx
        .convert_logical_type(builder.ir, semantic.result);
    (
        ability_ref,
        OperationDeclaration::new(
            ability_ref,
            operation,
            Symbol::new(match kind {
                OpDeclKind::Fn => "fn",
                OpDeclKind::Op => "op",
            }),
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
    ability: Symbol,
    operation: Symbol,
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
            let value = arith::r#const(builder.ir, location, ty, Attribute::Int(value as i128));
            builder.ir.push_op(builder.block, value.op_ref());
            Some(value.result(builder.ir))
        }
        ExprKind::IntLit(value) => {
            let ty = builder.ctx.i32_type(builder.ir);
            let value = arith::r#const(builder.ir, location, ty, Attribute::Int(value as i128));
            builder.ir.push_op(builder.block, value.op_ref());
            Some(value.result(builder.ir))
        }
        ExprKind::BoolLit(value) => {
            let ty = builder.ctx.bool_type(builder.ir);
            let value = arith::r#const(builder.ir, location, ty, Attribute::Bool(value));
            builder.ir.push_op(builder.block, value.op_ref());
            Some(value.result(builder.ir))
        }
        ExprKind::FloatLit(value) => {
            let ty = builder.ctx.f64_type(builder.ir);
            let value = arith::r#const(
                builder.ir,
                location,
                ty,
                Attribute::FloatBits(value.value().to_bits()),
            );
            builder.ir.push_op(builder.block, value.op_ref());
            Some(value.result(builder.ir))
        }
        ExprKind::Nil => Some(builder.emit_nil(location)),
        ExprKind::RuneLit(value) => {
            let ty = builder.ctx.i32_type(builder.ir);
            let value = arith::r#const(
                builder.ir,
                location,
                ty,
                Attribute::Int(value as i32 as i128),
            );
            builder.ir.push_op(builder.block, value.op_ref());
            Some(value.result(builder.ir))
        }
        ExprKind::BytesLit(value) => {
            let ty = builder.ctx.bytes_type(builder.ir);
            let value = adt::bytes_const(builder.ir, location, ty, value.into());
            builder.ir.push_op(builder.block, value.op_ref());
            Some(value.result(builder.ir))
        }
        ExprKind::StringLit(value) => {
            let ty = builder.ctx.anyref_type(builder.ir);
            let value = adt::string_const(builder.ir, location, ty, value);
            builder.ir.push_op(builder.block, value.op_ref());
            Some(value.result(builder.ir))
        }
        ExprKind::Var(reference) => match reference.resolved {
            ResolvedRef::Local { id, .. } => Some(
                builder
                    .ctx
                    .lookup(id)
                    .unwrap_or_else(|| panic!("missing logical binding for local {id:?}")),
            ),
            ResolvedRef::Function { id } => {
                let name = id.qualified(builder.ctx.db);
                let mut signature = FuncSignature::lookup_logical(builder.ctx, builder.ir, name)
                    .unwrap_or_else(|| {
                        panic!("missing logical signature for function reference {name}")
                    });
                signature.convention = builder
                    .ctx
                    .function_calling_convention(name)
                    .unwrap_or(signature.convention);
                ensure_prelude_declaration(builder, location, name, &signature);
                let ty = callable_type(
                    builder.ir,
                    signature.return_type,
                    signature.param_types,
                    signature.convention,
                );
                let op = op(builder.ir, builder.block, location, "func_ref", |builder| {
                    builder.result(ty).attr("func_ref", Attribute::Symbol(name))
                });
                Some(result(builder.ir, op))
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
            let then_yield = scf::r#yield(builder.ir, location, [then_value]);
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
            let else_yield = scf::r#yield(builder.ir, location, [else_value]);
            builder.ir.push_op(else_block, else_yield.op_ref());
            let else_region = builder.ir.create_region(RegionData {
                location,
                blocks: trunk_ir::smallvec::smallvec![else_block],
                parent_op: None,
            });
            let branch = scf::r#if(builder.ir, location, lhs, bool_ty, then_region, else_region);
            builder.ir.push_op(builder.block, branch.op_ref());
            Some(branch.result(builder.ir))
        }
        ExprKind::Call { callee, args } => {
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
            )
        }
        ExprKind::Lambda { params, body } => {
            let signature = declarations
                .lambda_signatures
                .get(&expr.id)
                .cloned()
                .unwrap_or_else(|| panic!("missing solved logical lambda signature"));
            lower_lambda(builder, location, signature, params, body, declarations)
        }
        ExprKind::Handle { body, handlers } => {
            let result_ty = builder
                .ctx
                .get_node_type(expr.id)
                .copied()
                .map(|ty| builder.ctx.convert_logical_type(builder.ir, ty))
                .unwrap_or_else(|| panic!("missing typechecked result for handle"));
            lower_handle(builder, location, result_ty, body, handlers, declarations)
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
            lower_case_chain(
                builder,
                location,
                scrutinee,
                result_ty,
                &arms,
                exhaustive,
                declarations,
            )
        }
        ExprKind::Resume { arg, local_id } => {
            let token = builder.ctx.lookup_resume(local_id?)?;
            let (input_ty, answer_ty) =
                tribute_control::resume_token_parts(builder.ir, builder.ir.value_ty(token))
                    .expect("typechecked resume local must lower to a resume token");
            let value = lower_expr(builder, arg, declarations)?;
            let value = builder.cast_if_needed(location, value, input_ty);
            let resume = op(builder.ir, builder.block, location, "resume", |builder| {
                builder.operand(token).operand(value).result(answer_ty)
            });
            Some(result(builder.ir, resume))
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
    let ResolvedRef::Constructor { variant, .. } = ctor.resolved else {
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
    let variant = adt::variant_new(builder.ir, location, values, result_ty, type_attr, variant);
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
    let tuple = adt::struct_new(builder.ir, location, values, result_ty, type_attr);
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
    let empty = list::empty(builder.ir, location, list_ty, element_ty);
    builder.ir.push_op(builder.block, empty.op_ref());
    let mut value = empty.result(builder.ir);
    for element in values.into_iter().rev() {
        let prepend = list::prepend(builder.ir, location, element, value, list_ty, element_ty);
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
    fields: Vec<(Symbol, Expr<TypedRef<'db>>)>,
    spread: Option<Expr<TypedRef<'db>>>,
    declarations: &mut Declarations<'db>,
) -> Option<ValueRef> {
    let ctor = super::extract_ctor_id(&type_name.resolved);
    let struct_name = super::extract_type_name(builder.db(), &type_name.resolved);
    let field_order = builder
        .ctx
        .get_struct_field_order(ctor)
        .cloned()
        .unwrap_or_else(|| panic!("prescan did not register struct field order {struct_name}"));
    let layout = builder
        .ctx
        .get_type(super::qualified_type_name(builder.db(), &ctor))
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
    let mut values = HashMap::new();
    for (name, field) in fields {
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
            let field_types = trunk_ir::adt_layout::get_struct_fields(builder.ir, layout)
                .unwrap_or_else(|| panic!("prescanned struct layout is malformed"));
            let get = adt::struct_get(
                builder.ir,
                location,
                base,
                field_types[index].1,
                layout,
                index as u32,
            );
            builder.ir.push_op(builder.block, get.op_ref());
            ordered.push(get.result(builder.ir));
        }
    }
    let result_ty = expr_type_for_id(builder, id);
    let record = adt::struct_new(builder.ir, location, ordered, result_ty, layout);
    builder.ir.push_op(builder.block, record.op_ref());
    Some(record.result(builder.ir))
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
        // Typechecking has proved this path unreachable (for example the
        // false branch after exhaustive Bool literal arms).  The arena has no
        // source-logical unreachable producer, so keep it as an isolated
        // polymorphic conversion value rather than introducing `func.*`.
        [] => Some(unreachable_case_value(builder, location, result_ty)),
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
            let branch = scf::r#if(
                builder.ir,
                location,
                condition,
                result_ty,
                then_region,
                else_region,
            );
            builder.ir.push_op(builder.block, branch.op_ref());
            Some(branch.result(builder.ir))
        }
    }
}

fn unreachable_case_value(
    builder: &mut IrBuilder<'_, '_>,
    location: Location,
    result_ty: TypeRef,
) -> ValueRef {
    let nil = builder.emit_nil(location);
    builder.cast_if_needed(location, nil, result_ty)
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
            let branch = scf::r#if(
                nested.ir,
                location,
                condition,
                result_ty,
                then_region,
                else_region,
            );
            nested.ir.push_op(nested.block, branch.op_ref());
            branch.result(nested.ir)
        } else {
            lower_expr(&mut nested, arm.body.clone(), declarations)?
        }
    };
    let value = IrBuilder::new(ctx, ir, block).cast_if_needed(location, value, result_ty);
    let yield_op = scf::r#yield(ir, location, [value]);
    ir.push_op(block, yield_op.op_ref());
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
    let value = lower_expr(&mut IrBuilder::new(ctx, ir, block), body, declarations)?;
    let value = IrBuilder::new(ctx, ir, block).cast_if_needed(location, value, result_ty);
    let yield_op = scf::r#yield(ir, location, [value]);
    ir.push_op(block, yield_op.op_ref());
    Some(ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![block],
        parent_op: None,
    }))
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
    )?;
    let value = IrBuilder::new(ctx, ir, block).cast_if_needed(location, value, result_ty);
    let yield_op = scf::r#yield(ir, location, [value]);
    ir.push_op(block, yield_op.op_ref());
    Some(ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![block],
        parent_op: None,
    }))
}

fn emit_bool(builder: &mut IrBuilder<'_, '_>, location: Location, value: bool) -> ValueRef {
    let ty = builder.ctx.bool_type(builder.ir);
    let op = arith::r#const(builder.ir, location, ty, Attribute::Bool(value));
    builder.ir.push_op(builder.block, op.op_ref());
    op.result(builder.ir)
}

fn lower_statement<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    statement: Stmt<TypedRef<'db>>,
    declarations: &mut Declarations<'db>,
) {
    match statement {
        Stmt::Let { pattern, value, .. } => {
            if let Some(value) = lower_expr(builder, value, declarations) {
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
        } => builder.ctx.bind(*id, *name, value),
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

fn lower_call<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    call_id: crate::ast::NodeId,
    result_ty: TypeRef,
    callee: Expr<TypedRef<'db>>,
    args: Vec<Expr<TypedRef<'db>>>,
    declarations: &mut Declarations<'db>,
) -> Option<ValueRef> {
    // A resolved variable callee is atomic. Any other callee expression is a
    // strict child and must be evaluated before the argument list.
    let indirect_callee = if matches!(&*callee.kind, ExprKind::Var(_)) {
        None
    } else {
        Some(lower_expr(builder, callee.clone(), declarations)?)
    };
    let mut values = args
        .into_iter()
        .map(|arg| lower_expr(builder, arg, declarations))
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
                let (ability_ref, declaration) = call_operation_metadata(
                    builder,
                    declarations,
                    PerformMetadata {
                        ability,
                        operation,
                        kind,
                        semantic,
                    },
                );
                values = match adapt_operation_arguments_or_recover(
                    builder,
                    location,
                    values,
                    &declaration,
                    ability.qualified(builder.db()),
                    operation,
                ) {
                    Ok(values) => values,
                    Err(nil) => return Some(nil),
                };
                let perform = op(builder.ir, builder.block, location, "perform", |builder| {
                    builder
                        .operands(values)
                        .result(declaration.result_type)
                        .attr("ability_ref", Attribute::Type(ability_ref))
                        .attr("op_name", Attribute::Symbol(operation))
                        .attr(
                            "operation_kind",
                            Attribute::Symbol(Symbol::new(match kind {
                                OpDeclKind::Fn => "fn",
                                OpDeclKind::Op => "op",
                            })),
                        )
                });
                declarations.record(declaration, location, builder.db());
                let value = result(builder.ir, perform);
                Some(builder.cast_if_needed(location, value, result_ty))
            }
            ResolvedRef::Function { id } => {
                let name = id.qualified(builder.ctx.db);
                let signature = FuncSignature::lookup_logical(builder.ctx, builder.ir, name)
                    .unwrap_or_else(|| panic!("missing logical signature for call {name}"));
                ensure_prelude_declaration(builder, location, name, &signature);
                if values.len() != signature.param_types.len() {
                    panic!("typechecked call arity disagrees with logical signature for {name}");
                }
                values = values
                    .into_iter()
                    .zip(signature.param_types.iter().copied())
                    .map(|(value, ty)| builder.cast_if_needed(location, value, ty))
                    .collect();
                let call = op(builder.ir, builder.block, location, "call", |builder| {
                    builder
                        .operands(values)
                        .result(signature.return_type)
                        .attr("callee", Attribute::Symbol(name))
                });
                let value = result(builder.ir, call);
                Some(builder.cast_if_needed(location, value, result_ty))
            }
            ResolvedRef::Local { id, .. } => {
                let callable = builder
                    .ctx
                    .lookup(id)
                    .unwrap_or_else(|| panic!("missing logical callable binding for local {id:?}"));
                let callable_ty = builder.ir.value_ty(callable);
                let callable_signature =
                    tribute_control::Callable::from_type_ref(builder.ir, callable_ty)
                        .unwrap_or_else(|| panic!("local call target is not a logical callable"));
                let parameters = callable_signature.params(builder.ir).to_vec();
                if values.len() != parameters.len() {
                    panic!("typechecked indirect call arity disagrees with logical callable");
                }
                values = values
                    .into_iter()
                    .zip(parameters.iter().copied())
                    .map(|(value, ty)| builder.cast_if_needed(location, value, ty))
                    .collect();
                let call_result = callable_signature.result(builder.ir);
                let call = op(
                    builder.ir,
                    builder.block,
                    location,
                    "call_indirect",
                    |builder| {
                        builder
                            .operand(callable)
                            .operands(values)
                            .result(call_result)
                    },
                );
                let value = result(builder.ir, call);
                Some(builder.cast_if_needed(location, value, result_ty))
            }
            _ => panic!("unsupported call target at source-logical boundary"),
        }
    } else {
        let callable = indirect_callee
            .expect("non-variable logical callee must be evaluated before its arguments");
        let callable_ty = builder.ir.value_ty(callable);
        let callable_signature = tribute_control::Callable::from_type_ref(builder.ir, callable_ty)
            .unwrap_or_else(|| panic!("indirect call target is not a logical callable"));
        let parameters = callable_signature.params(builder.ir).to_vec();
        if values.len() != parameters.len() {
            panic!("typechecked indirect call arity disagrees with logical callable");
        }
        values = values
            .into_iter()
            .zip(parameters.iter().copied())
            .map(|(value, ty)| builder.cast_if_needed(location, value, ty))
            .collect();
        let call_result = callable_signature.result(builder.ir);
        let call = op(
            builder.ir,
            builder.block,
            location,
            "call_indirect",
            |builder| {
                builder
                    .operand(callable)
                    .operands(values)
                    .result(call_result)
            },
        );
        let value = result(builder.ir, call);
        Some(builder.cast_if_needed(location, value, result_ty))
    }
}

fn lower_lambda<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    signature: crate::typeck::LambdaSignature<'db>,
    params: Vec<crate::ast::Param>,
    body: Expr<TypedRef<'db>>,
    declarations: &mut Declarations<'db>,
) -> Option<ValueRef> {
    let TypeKind::Func {
        params: signature_params,
        result: signature_result,
        ..
    } = signature.function_type.kind(builder.db())
    else {
        panic!("solved logical lambda signature is not a function type");
    };
    let param_types = signature_params
        .iter()
        .map(|ty| builder.ctx.convert_logical_type(builder.ir, *ty))
        .collect::<Vec<_>>();
    let result_type = builder
        .ctx
        .convert_logical_type(builder.ir, *signature_result);
    let convention = signature.convention;
    let mut free = HashSet::new();
    super::lambda::collect_free_vars(&body, &mut free);
    for parameter in &params {
        if let Some(id) = parameter.local_id {
            free.remove(&id);
        }
    }
    let captures: Vec<_> = builder
        .ctx
        .all_bindings()
        .filter(|(id, _, _)| free.contains(id))
        .map(|(_, _, value)| value)
        .collect();
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
                    parameter.name,
                    builder.ir.block_arg(entry, index as u32),
                );
            }
        }
        let value = lower_expr(
            &mut IrBuilder::new(&mut scope, builder.ir, entry),
            body,
            declarations,
        )?;
        let value = IrBuilder::new(&mut scope, builder.ir, entry).cast_if_needed(
            location,
            value,
            result_type,
        );
        op(builder.ir, entry, location, "return", |builder| {
            builder.operand(value)
        });
    }
    let region = builder.ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![entry],
        parent_op: None,
    });
    let callable = callable_type(builder.ir, result_type, param_types, convention);
    let lambda = op(builder.ir, builder.block, location, "lambda", |builder| {
        builder.operands(captures).result(callable).region(region)
    });
    Some(result(builder.ir, lambda))
}

fn lower_handle<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
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
    let body_yield = op(builder.ir, body_block, location, "yield", |builder| {
        builder.operand(body_value)
    });
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
    op(builder.ir, completion_block, location, "yield", |builder| {
        builder.operand(completion_value)
    });
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
    let handle = op(builder.ir, builder.block, location, "handle", |builder| {
        builder
            .result(result_ty)
            .region(body_region)
            .region(completion_region)
            .region(handlers_region)
    });
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
        ResolvedRef::TypeDef { id } => crate::ast::AbilityId::source(ctx.db, id.qualified(ctx.db)),
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
    let ability_arguments: Vec<_> = arguments
        .iter()
        .map(|arg| ctx.convert_logical_type(ir, *arg))
        .collect();
    let ability_ref = ctx.ability_ref_type(ir, ability_id.qualified(ctx.db), &ability_arguments);
    let parameter_types: Vec<_> = expected_params
        .into_iter()
        .map(|ty| ctx.convert_logical_type(ir, ty))
        .collect();
    let operation_result = ctx.convert_logical_type(ir, expected_result);
    let declaration = OperationDeclaration::new(
        ability_ref,
        operation,
        Symbol::new(match kind {
            OpDeclKind::Fn => "fn",
            OpDeclKind::Op => "op",
        }),
        parameter_types.clone(),
        operation_result,
    );
    declarations.record(declaration, location, ctx.db);
    let is_never = ir.types.get(operation_result).dialect == Symbol::new("core")
        && ir.types.get(operation_result).name == Symbol::new("never");
    let mut block_args: Vec<_> = parameter_types
        .iter()
        .map(|ty| BlockArgData {
            ty: *ty,
            attrs: Default::default(),
        })
        .collect();
    if kind == OpDeclKind::Op && !is_never {
        let token = ir.types.intern(
            trunk_ir::types::TypeDataBuilder::new(
                Symbol::new("tribute_control"),
                Symbol::new("resume_token"),
            )
            .param(operation_result)
            .param(answer_ty)
            .build(),
        );
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
        op(ir, block, location, "yield", |builder| {
            builder.operand(value)
        });
    }
    let region = ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![block],
        parent_op: None,
    });
    op(ir, table, location, "handler", |builder| {
        builder
            .attr("ability_ref", Attribute::Type(ability_ref))
            .attr("op_name", Attribute::Symbol(operation))
            .attr(
                "kind",
                Attribute::Symbol(Symbol::new(match kind {
                    OpDeclKind::Fn => "fn",
                    OpDeclKind::Op => "op",
                })),
            )
            .attr("operation_result_type", Attribute::Type(operation_result))
            .region(region)
    });
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::context::BlockData;
    use trunk_ir::location::Span;

    #[salsa::tracked]
    fn operation_arguments_use_resolved_parameter_types_inner<'db>(
        db: &'db dyn salsa::Database,
    ) -> bool {
        let mut ir = IrContext::new();
        let path = ir.paths.intern("logical.trb".to_owned());
        let mut ctx = IrLoweringCtx::new(
            db,
            path,
            crate::ast::SpanMap::default(),
            HashMap::new(),
            HashMap::new(),
            smallvec::smallvec![Symbol::new("test")],
            HashMap::new(),
        );
        let location = Location::new(path, Span::new(0, 0));
        let block = ir.create_block(BlockData {
            location,
            args: vec![],
            ops: Default::default(),
            parent_region: None,
        });
        let anyref = ctx.anyref_type(&mut ir);
        let value = adt::ref_null(&mut ir, location, anyref, anyref);
        ir.push_op(block, value.op_ref());
        let value = value.result(&ir);
        let parameter = ctx.adt_typeref(&mut ir, Symbol::new("String"));
        let declaration = OperationDeclaration::new(
            parameter,
            Symbol::new("throw"),
            Symbol::new("op"),
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
            Symbol::new("Test"),
            Symbol::new("throw"),
        )
        .expect("matching operation arity should adapt arguments");
        assert_eq!(builder.ir.value_ty(adapted[0]), parameter);

        let mismatch = OperationDeclaration::new(
            parameter,
            Symbol::new("throw"),
            Symbol::new("op"),
            [],
            parameter,
        );
        let recovered = adapt_operation_arguments_or_recover(
            &mut builder,
            location,
            vec![value],
            &mismatch,
            Symbol::new("Test"),
            Symbol::new("throw"),
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
