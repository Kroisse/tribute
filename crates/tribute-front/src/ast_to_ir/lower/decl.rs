//! Module-level and declaration lowering.

use tribute_core::{CallableAbi, set_calling_convention};
use tribute_ir::dialect::ability;
use trunk_ir::Symbol;
use trunk_ir::context::{BlockArgData, BlockData, IrContext, RegionData};
use trunk_ir::dialect::{adt, core, func};
use trunk_ir::refs::{BlockRef, PathRef, TypeRef};
use trunk_ir::rewrite::Module as IrModule;
use trunk_ir::types::{Attribute, Location};

use crate::ast::{
    CallingConvention, CtorId, Decl, EffectRow, ExternFuncDecl, FuncDecl, TypeKind, TypedRef,
};

use super::super::TypedModule;
use super::super::context::IrLoweringCtx;
use super::{FuncSignature, IrBuilder, convert_annotation_to_ir_type, expr, qualified_type_name};

struct PendingWellKnownType {
    definition: crate::typeck::DefinitionIdentity,
    ir_type: Option<TypeRef>,
}

struct WellKnownTypePrescan {
    string: Option<PendingWellKnownType>,
    list: Option<PendingWellKnownType>,
}

impl WellKnownTypePrescan {
    fn new(types: crate::typeck::WellKnownTypes<'_>) -> Self {
        Self {
            string: types.string.map(|ty| PendingWellKnownType {
                definition: ty.definition,
                ir_type: None,
            }),
            list: types.list.map(|ty| PendingWellKnownType {
                definition: ty.definition,
                ir_type: None,
            }),
        }
    }

    fn is_string(&self, definition: crate::typeck::DefinitionIdentity) -> bool {
        self.string
            .as_ref()
            .is_some_and(|string| string.definition == definition)
    }

    fn record_string(&mut self, ir_type: TypeRef) {
        if let Some(string) = &mut self.string {
            string.ir_type = Some(ir_type);
        }
    }

    fn is_list(&self, definition: crate::typeck::DefinitionIdentity) -> bool {
        self.list
            .as_ref()
            .is_some_and(|list| list.definition == definition)
    }

    fn record_list(&mut self, ir_type: TypeRef) {
        if let Some(list) = &mut self.list {
            list.ir_type = Some(ir_type);
        }
    }

    fn finish(self) -> tribute_ir::metadata::WellKnownTypes {
        tribute_ir::metadata::WellKnownTypes {
            string: self.string.and_then(|string| string.ir_type),
            list: self.list.and_then(|list| list.ir_type),
        }
    }
}

/// Pre-scan declarations to register struct field orders.
fn prescan_struct_fields<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    decls: &[Decl<TypedRef<'db>>],
    prefix: &mut String,
    well_known_types: &mut WellKnownTypePrescan,
) {
    for decl in decls {
        match decl {
            Decl::Struct(s) => {
                let qualified = crate::qualified_symbol(prefix, s.name);
                let ctor_id = CtorId::new(ctx.db, qualified);
                let field_names: Vec<Symbol> = s
                    .fields
                    .iter()
                    .map(|f| f.name.unwrap_or_else(|| Symbol::new("_")))
                    .collect();
                ctx.register_struct_fields(ctor_id, field_names);

                // Build full struct IR type with field names and types
                let ir_fields: Vec<(Symbol, TypeRef)> = s
                    .fields
                    .iter()
                    .map(|f| {
                        let name = f.name.unwrap_or_else(|| Symbol::new("_"));
                        let ty = convert_annotation_to_ir_type(ctx, ir, Some(&f.ty));
                        (name, ty)
                    })
                    .collect();
                let qualified = qualified_type_name(ctx.db, &ctor_id);
                let struct_ir_type = ctx.adt_struct_type(ir, qualified, &ir_fields);
                ctx.register_type(qualified, struct_ir_type);
            }
            Decl::Enum(e) => {
                let qualified = crate::qualified_symbol(prefix, e.name);
                let ctor_id = CtorId::new(ctx.db, qualified);

                let ir_variants: Vec<(Symbol, Vec<TypeRef>)> = e
                    .variants
                    .iter()
                    .map(|v| {
                        let field_types: Vec<TypeRef> = v
                            .fields
                            .iter()
                            .map(|f| convert_annotation_to_ir_type(ctx, ir, Some(&f.ty)))
                            .collect();
                        (v.name, field_types)
                    })
                    .collect();
                let qualified = qualified_type_name(ctx.db, &ctor_id);
                let definition =
                    crate::typeck::DefinitionIdentity::new(e.id, ctx.location(e.id).span);
                let enum_ir_type = if well_known_types.is_string(definition)
                    || well_known_types.is_list(definition)
                {
                    ctx.adt_enum_type_with_definition(ir, qualified, &ir_variants, definition)
                } else {
                    ctx.adt_enum_type(ir, qualified, &ir_variants)
                };
                ctx.register_type(qualified, enum_ir_type);
                if well_known_types.is_string(definition) {
                    well_known_types.record_string(enum_ir_type);
                }
                if well_known_types.is_list(definition) {
                    well_known_types.record_list(enum_ir_type);
                }
            }
            Decl::Module(m) => {
                if let Some(body) = &m.body {
                    let saved = crate::push_prefix(prefix, m.name);
                    prescan_struct_fields(ctx, ir, body, prefix, well_known_types);
                    prefix.truncate(saved);
                }
            }
            _ => {}
        }
    }
}

/// Record physical worker conventions separately from semantic function types.
///
/// An omitted effect annotation denotes an open row, but its generalized tail
/// does not by itself require the definition's worker to use CPS. Concrete
/// residual effects still contribute their ability-level requirements. An
/// explicit annotations use the convention derived from their closed row.
fn prescan_definition_conventions<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    decls: &[Decl<TypedRef<'db>>],
    prefix: &mut String,
) {
    for decl in decls {
        match decl {
            Decl::Function(func_decl) => {
                let qualified = crate::qualified_symbol(prefix, func_decl.name);
                let scheme = ctx
                    .lookup_function_type(qualified)
                    .or_else(|| ctx.lookup_function_type(func_decl.name));
                let Some(scheme) = scheme.copied() else {
                    continue;
                };
                let body = scheme.body(ctx.db);
                let Some(mut convention) = ctx.calling_convention_for_type(body) else {
                    continue;
                };

                if func_decl.effects.is_none()
                    && let TypeKind::Func { effect, .. } = body.kind(ctx.db)
                {
                    let concrete = EffectRow::new(ctx.db, effect.effects(ctx.db).clone(), None);
                    convention = ctx.calling_convention_for_effect_row(concrete);
                }
                ctx.register_definition_convention(qualified, convention);
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

impl<'db> TypedModule<'db> {
    /// Lower this module when its source path has already been interned.
    pub(crate) fn lower_module(
        self,
        db: &'db dyn salsa::Database,
        ir: &mut IrContext,
        path: PathRef,
        options: super::super::AstToIrOptions,
    ) -> IrModule {
        let Self {
            ast,
            span_map,
            function_types,
            node_types,
            ability_conventions,
            well_known_types,
        } = self;
        let module_location = span_map.get_or_default(ast.id);
        let location = Location::new(path, module_location);
        let module_name = ast.name.unwrap_or_else(|| Symbol::new("main"));
        let module_path = smallvec::smallvec![module_name];

        let mut ctx = IrLoweringCtx::new(
            db,
            path,
            span_map.clone(),
            function_types,
            ability_conventions,
            module_path,
            node_types,
        )
        .with_options(options);

        prescan_definition_conventions(&mut ctx, &ast.decls, &mut String::new());

        // Pre-scan: register all struct field orders before lowering any declarations.
        let mut well_known_types = WellKnownTypePrescan::new(well_known_types);
        prescan_struct_fields(
            &mut ctx,
            ir,
            &ast.decls,
            &mut String::new(),
            &mut well_known_types,
        );

        let well_known_types = well_known_types.finish();
        ctx.set_well_known_types(well_known_types);

        // Create the module block (top-down: create block first, push ops into it)
        let module_block = ir.create_block(BlockData {
            location,
            args: vec![],
            ops: Default::default(),
            parent_region: None,
        });
        ctx.set_module_block(module_block);

        // Lower each declaration, pushing ops into the module block
        for decl in ast.decls {
            lower_decl(&mut ctx, ir, module_block, decl);
        }

        // Create region and module op
        let module_region = ir.create_region(RegionData {
            location,
            blocks: trunk_ir::smallvec::smallvec![module_block],
            parent_op: None,
        });

        let module_op = core::module(ir, location, module_name, module_region);
        well_known_types.attach(ir, module_op.op_ref());
        IrModule::new(ir, module_op.op_ref()).expect("valid core.module operation")
    }
}

/// Lower a declaration to arena TrunkIR.
fn lower_decl<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    top: BlockRef,
    decl: Decl<TypedRef<'db>>,
) {
    match decl {
        Decl::Function(func) => lower_function(ctx, ir, top, func),
        Decl::ExternFunction(func) => lower_extern_function(ctx, ir, top, func),
        Decl::Struct(s) => lower_struct_decl(ctx, ir, top, s),
        Decl::Enum(_) => {
            // Enum type is already registered in the type environment.
        }
        Decl::Ability(_) => {
            // Ability declarations are purely type-level metadata.
        }
        Decl::Use(_) => {
            // Use declarations don't generate IR
        }
        Decl::Module(m) => {
            if let Some(body) = m.body {
                ctx.enter_module(m.name);
                for inner_decl in body {
                    lower_decl(ctx, ir, top, inner_decl);
                }
                ctx.exit_module();
            }
        }
    }
}

/// Lower a function declaration to arena TrunkIR.
fn lower_function<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    top: BlockRef,
    func_decl: FuncDecl<TypedRef<'db>>,
) {
    let location = ctx.location(func_decl.id);
    let func_name = func_decl.name;

    // Use TypeScheme from type checking if available, otherwise fall back to annotations
    let sig = FuncSignature::lookup(ctx, ir, func_name)
        .unwrap_or_else(|| fallback_from_annotations(ctx, ir, &func_decl));
    let FuncSignature {
        param_types: param_ir_types,
        return_type: return_ty,
        convention: semantic_convention,
    } = sig;
    let qualified_name = ctx.qualify_name(func_name);
    let convention = ctx
        .function_calling_convention(qualified_name)
        .unwrap_or(semantic_convention);
    let anyref_ty = ctx.anyref_type(ir);
    let evidence_ty = ability::evidence_adt_type_ref(ir);
    let abi = CallableAbi::new(convention, param_ir_types.iter().copied(), return_ty);

    // Create entry block with parameter args
    let mut block_args: Vec<BlockArgData> = param_ir_types
        .iter()
        .zip(func_decl.params.iter())
        .map(|(&ty, p)| {
            let mut arg = BlockArgData {
                ty,
                attrs: Default::default(),
            };
            arg.attrs
                .insert(Symbol::new("bind_name"), Attribute::Symbol(p.name));
            arg
        })
        .collect();

    // Effectful functions receive evidence. Only CPS functions also receive done_k
    // and return through it on normal completion.
    let block_args = if convention.needs_evidence() {
        let mut ev_arg = BlockArgData {
            ty: evidence_ty,
            attrs: Default::default(),
        };
        ev_arg.attrs.insert(
            Symbol::new("bind_name"),
            Attribute::Symbol(Symbol::new("__evidence")),
        );
        let mut args = vec![ev_arg];
        if convention.needs_done_k() {
            let mut done_k_arg = BlockArgData {
                ty: anyref_ty,
                attrs: Default::default(),
            };
            done_k_arg.attrs.insert(
                Symbol::new("bind_name"),
                Attribute::Symbol(Symbol::new("__done_k")),
            );
            args.push(done_k_arg);
        }
        args.append(&mut block_args);
        args
    } else {
        block_args
    };

    let entry_block = ir.create_block(BlockData {
        location,
        args: block_args,
        ops: Default::default(),
        parent_region: None,
    });

    // Bind parameters to their block argument values
    {
        let mut scope = ctx.scope();
        let param_offset = abi.source_param_offset() as u32;
        for (i, param) in func_decl.params.iter().enumerate() {
            if let Some(local_id) = param.local_id {
                let arg_val = ir.block_arg(entry_block, i as u32 + param_offset);
                scope.bind(local_id, param.name, arg_val);
            }
        }

        // For CPS functions: set done_k and use CPS body lowering.
        // The done_k continuation is called at the end of the continuation chain
        // instead of func.return, handled by build_cps_continuation.
        if convention.needs_done_k() {
            let prev_done_k = scope.done_k;
            let prev_evidence = scope.evidence;
            let evidence_val = ir.block_arg(entry_block, 0);
            let done_k_val = ir.block_arg(entry_block, 1);
            scope.evidence = Some(evidence_val);
            scope.done_k = Some(done_k_val);

            let mut builder = IrBuilder::new(&mut scope, ir, entry_block);
            let result = expr::lower_block_cps_for_body(&mut builder, func_decl.body);

            if let Some((body_result, is_cps)) = result {
                let result_anyref = builder.cast_if_needed(location, body_result, anyref_ty);
                if is_cps {
                    // CPS: effectful call returned; emit func.return with the result.
                    // The callee already called done_k internally via continuation chain.
                    let ret = func::r#return(builder.ir, location, [result_anyref]);
                    builder.ir.push_op(builder.block, ret.op_ref());
                } else {
                    // Pure result: call done_k(result) instead of func.return
                    super::emit_done_k_call(&mut builder, location, done_k_val, result_anyref);
                }
            }
            // ability.perform → lower_ability_perform will add func.return

            scope.done_k = prev_done_k;
            scope.evidence = prev_evidence;
        } else {
            // Direct and EvidenceDirect functions return their source result normally.
            // EvidenceDirect still threads evidence through nested effectful calls.
            let prev_evidence = scope.evidence;
            if convention.needs_evidence() {
                scope.evidence = Some(ir.block_arg(entry_block, 0));
            }
            let mut builder = IrBuilder::new(&mut scope, ir, entry_block);
            if let Some(result) = expr::lower_expr(&mut builder, func_decl.body) {
                let result = builder.cast_if_needed(location, result, return_ty);
                let ret_op = func::r#return(builder.ir, location, [result]);
                builder.ir.push_op(builder.block, ret_op.op_ref());
            } else {
                let nil = builder.emit_nil(location);
                let ret_op = func::r#return(builder.ir, location, [nil]);
                builder.ir.push_op(builder.block, ret_op.op_ref());
            }
            scope.evidence = prev_evidence;
        }
    }

    // Build function type
    let final_param_types = abi.lowered_params(evidence_ty, anyref_ty);
    let final_return_ty = abi.lowered_result(anyref_ty);
    let func_type = ctx.func_type(ir, &final_param_types, final_return_ty);

    // Create body region and func op
    let body_region = ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![entry_block],
        parent_op: None,
    });

    let qualified_name = ctx.qualify_name(func_name);
    let func_op = func::func(ir, location, qualified_name, func_type, body_region);
    set_calling_convention(ir, func_op.op_ref(), convention);
    ir.push_op(top, func_op.op_ref());
}

/// Lower an extern function declaration.
fn lower_extern_function<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    top: BlockRef,
    func_decl: ExternFuncDecl,
) {
    let location = ctx.location(func_decl.id);
    let func_name = func_decl.name;
    let qualified_name = ctx.qualify_name(func_name);

    let FuncSignature {
        param_types: param_ir_types,
        return_type: return_ty,
        ..
    } = FuncSignature::lookup(ctx, ir, qualified_name).unwrap_or_else(|| {
        panic!(
            "extern function '{}' should have TypeScheme from type checking",
            qualified_name
        )
    });

    // Create entry block with parameter args
    let block_args: Vec<BlockArgData> = param_ir_types
        .iter()
        .zip(func_decl.params.iter())
        .map(|(&ty, p)| {
            let mut arg = BlockArgData {
                ty,
                attrs: Default::default(),
            };
            arg.attrs
                .insert(Symbol::new("bind_name"), Attribute::Symbol(p.name));
            arg
        })
        .collect();

    let entry_block = ir.create_block(BlockData {
        location,
        args: block_args,
        ops: Default::default(),
        parent_region: None,
    });

    // Extern functions have no body — emit unreachable
    let unreachable_op = func::unreachable(ir, location);
    ir.push_op(entry_block, unreachable_op.op_ref());

    // Build function type and create func op
    let func_type = ctx.func_type(ir, &param_ir_types, return_ty);

    let body_region = ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![entry_block],
        parent_op: None,
    });

    let func_op = func::func(ir, location, qualified_name, func_type, body_region);

    // Mark as extern so the backend treats it as Import linkage
    let abi_str = func_decl.abi.to_string();
    ir.op_mut(func_op.op_ref()).attributes.insert(
        Symbol::new("abi"),
        trunk_ir::types::Attribute::String(abi_str),
    );

    ir.push_op(top, func_op.op_ref());
}

/// Lower a struct declaration to arena TrunkIR.
fn lower_struct_decl<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    top: BlockRef,
    decl: crate::ast::StructDecl,
) {
    let location = ctx.location(decl.id);
    let name = decl.name;

    // Generate accessor module with getter functions
    let fields: Vec<(Symbol, TypeRef)> = decl
        .fields
        .iter()
        .map(|f| {
            let field_name = f.name.unwrap_or_else(|| Symbol::new("_"));
            let field_ty = convert_annotation_to_ir_type(ctx, ir, Some(&f.ty));
            (field_name, field_ty)
        })
        .collect();

    // Use the registered full struct IR type for adt.struct_get
    let qualified_key = {
        let module_path = ctx.module_path();
        let mut prefix = String::new();
        // Skip the first segment (top-level module name) to match prescan
        for seg in module_path.iter().skip(1) {
            crate::push_prefix(&mut prefix, *seg);
        }
        let qualified = crate::qualified_symbol(&mut prefix, name);
        let ctor_id = CtorId::new(ctx.db, qualified);
        qualified_type_name(ctx.db, &ctor_id)
    };
    let struct_get_ty = ctx.get_type(qualified_key).unwrap_or_else(|| {
        panic!(
            "ICE: struct type `{}` not registered during prescan",
            qualified_key
        )
    });
    let struct_ty = ctx.adt_typeref(ir, qualified_key);

    let module_path = ctx.module_path().clone();

    // Create accessor module: module block → getter funcs
    let accessor_block = ir.create_block(BlockData {
        location,
        args: vec![],
        ops: Default::default(),
        parent_region: None,
    });

    for (idx, (field_name, field_type)) in fields.iter().enumerate() {
        // Generate getter: fn qualified_name(self: StructType) -> FieldType
        // Skip the first segment (top-level module name) to match resolve convention.
        let nested: Vec<_> = module_path.iter().skip(1).collect();
        let qualified_name = if nested.is_empty() {
            Symbol::from_dynamic(&format!("{}::{}", name, field_name))
        } else {
            let nested_str = nested
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join("::");
            Symbol::from_dynamic(&format!("{}::{}::{}", nested_str, name, field_name))
        };

        // Create getter function: entry block with one arg (self)
        let getter_entry = ir.create_block(BlockData {
            location,
            args: vec![BlockArgData {
                ty: struct_ty,
                attrs: Default::default(),
            }],
            ops: Default::default(),
            parent_region: None,
        });

        let self_value = ir.block_arg(getter_entry, 0);
        let field_op = adt::struct_get(
            ir,
            location,
            self_value,
            *field_type,
            struct_get_ty,
            idx as u32,
        );
        ir.push_op(getter_entry, field_op.op_ref());
        let field_result = field_op.result(ir);

        let ret_op = func::r#return(ir, location, [field_result]);
        ir.push_op(getter_entry, ret_op.op_ref());

        let getter_body = ir.create_region(RegionData {
            location,
            blocks: trunk_ir::smallvec::smallvec![getter_entry],
            parent_op: None,
        });

        let getter_func_type = ctx.func_type(ir, &[struct_ty], *field_type);
        let getter_op = func::func(ir, location, qualified_name, getter_func_type, getter_body);
        ir.push_op(accessor_block, getter_op.op_ref());
    }

    let accessor_region = ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![accessor_block],
        parent_op: None,
    });

    let accessor_module = core::module(ir, location, name, accessor_region);
    ir.push_op(top, accessor_module.op_ref());
}

/// Fallback: derive parameter and return types from annotations.
fn fallback_from_annotations<'db>(
    ctx: &IrLoweringCtx<'db>,
    ir: &mut IrContext,
    func: &FuncDecl<TypedRef<'db>>,
) -> FuncSignature {
    let param_types = func
        .params
        .iter()
        .map(|p| convert_annotation_to_ir_type(ctx, ir, p.ty.as_ref()))
        .collect();
    let return_type = func
        .return_ty
        .as_ref()
        .map(|ann| convert_annotation_to_ir_type(ctx, ir, Some(ann)))
        .unwrap_or_else(|| ctx.nil_type(ir));
    FuncSignature {
        param_types,
        return_type,
        convention: CallingConvention::Direct,
    }
}
