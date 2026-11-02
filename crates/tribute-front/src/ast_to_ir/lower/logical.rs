//! Source-logical AST to TrunkIR lowering.
//!
//! It emits only the documented `tribute_control` boundary and ordinary value
//! dialects; shared CPS construction belongs to `tribute-passes`.

use rustc_hash::FxHashMap as HashMap;
use tribute_ir::dialect::tribute_control;
use trunk_ir::context::{BlockData, IrContext, OperationDataBuilder, RegionData};
use trunk_ir::dialect::core;
use trunk_ir::refs::{BlockRef, OpRef, PathRef, ValueRef};
use trunk_ir::rewrite::Module as IrModule;
use trunk_ir::types::Location;
use trunk_ir::{Symbol, SymbolPath};

use crate::ast::{Decl, DeclEntry, TypedRef};
use crate::ast_to_ir::context::IrLoweringCtx;
use crate::ast_to_ir::{FrontendIrModule, LoweringInput};

use callable::{lower_function, plan_local_callables, prescan_row_binders, row_effects};
use declarations::{
    Declarations, instance_operations, lower_extern, lower_struct_accessors,
    prescan_source_functions, prescan_struct_accessor_signatures,
};
use nominal::{collect_logical_nominal_identities, prescan_logical_nominal_layouts};

mod callable;
mod declarations;
mod local_callables;
mod nominal;
mod values;

pub(super) use callable::emit_named_call;

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

fn result(ir: &IrContext, op: OpRef) -> ValueRef {
    ir.op_result(op, 0)
}

pub(super) fn lower_module<'db>(
    typed: LoweringInput<'db>,
    db: &'db dyn salsa::Database,
    ir: &mut IrContext,
    path: PathRef,
) -> FrontendIrModule {
    let LoweringInput {
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
        root_entry: None,
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
    if let Some(entry) = declarations.root_entry.take() {
        tribute_ir::metadata::set_root_entry(ir, module.op_ref(), SymbolPath::new([entry]));
    }
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

impl<'db> LoweringInput<'db> {
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

#[cfg(test)]
mod tests {
    use super::callable::adapt_operation_arguments;
    use super::*;
    use crate::SortedMap;
    use crate::ast_to_ir::lower::IrBuilder;
    use salsa_test_macros::salsa_test;
    use tribute_ir::dialect::adt;
    use tribute_ir::dialect::tribute_control::OperationDeclaration;
    use trunk_ir::context::BlockData;
    use trunk_ir::location::Span;

    #[salsa_test]
    fn operation_arguments_use_resolved_parameter_types(db: &dyn salsa::Database) {
        let mut ir = IrContext::new();
        let path = ir.intern_path("logical.trb");
        let mut ctx = IrLoweringCtx::new(
            db,
            path,
            crate::ast::SpanMaps::default(),
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
        let adapted = adapt_operation_arguments(&mut builder, location, vec![value], &declaration);
        assert_eq!(builder.ir.value_ty(adapted[0]), parameter);
    }
}
