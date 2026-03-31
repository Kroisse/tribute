//! Module-level and declaration lowering.

use std::collections::HashMap;

use trunk_ir::Symbol;
use trunk_ir::context::{BlockArgData, BlockData, IrContext, RegionData};
use trunk_ir::dialect::{adt, core, func};
use trunk_ir::refs::{BlockRef, PathRef, TypeRef};
use trunk_ir::rewrite::Module as IrModule;
use trunk_ir::types::{Attribute, Location};

use crate::ast::{
    CtorId, Decl, ExternFuncDecl, FuncDecl, Module, NodeId, SpanMap, TypeKind, TypeScheme, TypedRef,
};

use super::super::context::IrLoweringCtx;
use super::{IrBuilder, convert_annotation_to_ir_type, expr, qualified_type_name};

/// Pre-scan declarations to register struct field orders.
fn prescan_struct_fields<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    decls: &[Decl<TypedRef<'db>>],
    prefix: &mut String,
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
                let enum_ir_type = ctx.adt_enum_type(ir, qualified, &ir_variants);
                ctx.register_type(qualified, enum_ir_type);
            }
            Decl::Module(m) => {
                if let Some(body) = &m.body {
                    let saved = crate::push_prefix(prefix, m.name);
                    prescan_struct_fields(ctx, ir, body, prefix);
                    prefix.truncate(saved);
                }
            }
            _ => {}
        }
    }
}

/// Lower a module to arena TrunkIR.
///
/// Returns an `Module` inside the given `IrContext`.
pub fn lower_module<'db>(
    db: &'db dyn salsa::Database,
    ir: &mut IrContext,
    path: PathRef,
    span_map: SpanMap,
    module: Module<TypedRef<'db>>,
    function_types: HashMap<Symbol, TypeScheme<'db>>,
    node_types: HashMap<NodeId, crate::ast::Type<'db>>,
) -> IrModule {
    let module_location = span_map.get_or_default(module.id);
    let location = Location::new(path, module_location);
    let module_name = module.name.unwrap_or_else(|| Symbol::new("main"));
    let module_path = smallvec::smallvec![module_name];

    let mut ctx = IrLoweringCtx::new(
        db,
        path,
        span_map.clone(),
        function_types,
        module_path,
        node_types,
    );

    // Pre-scan: register all struct field orders before lowering any declarations.
    prescan_struct_fields(&mut ctx, ir, &module.decls, &mut String::new());

    // Create the module block (top-down: create block first, push ops into it)
    let module_block = ir.create_block(BlockData {
        location,
        args: vec![],
        ops: Default::default(),
        parent_region: None,
    });
    ctx.set_module_block(module_block);

    // Lower each declaration, pushing ops into the module block
    for decl in module.decls {
        lower_decl(&mut ctx, ir, module_block, decl);
    }

    // Create region and module op
    let module_region = ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![module_block],
        parent_op: None,
    });

    let module_op = core::module(ir, location, module_name, module_region);
    IrModule::new(ir, module_op.op_ref()).expect("valid core.module operation")
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
    let (param_ir_types, return_ty, effect_ir) =
        if let Some(scheme) = ctx.lookup_function_type(func_name).cloned() {
            let body = scheme.body(ctx.db);
            match body.kind(ctx.db) {
                TypeKind::Func {
                    params,
                    result,
                    effect,
                } => {
                    let p: Vec<TypeRef> = params.iter().map(|t| ctx.convert_type(ir, *t)).collect();
                    let r = ctx.convert_type(ir, *result);
                    let e = if effect.is_pure(ctx.db) {
                        None
                    } else {
                        Some(ctx.convert_effect_row(ir, *effect))
                    };
                    (p, r, e)
                }
                _ => {
                    let (p, r) = fallback_from_annotations(ctx, ir, &func_decl);
                    (p, r, None)
                }
            }
        } else {
            let (p, r) = fallback_from_annotations(ctx, ir, &func_decl);
            (p, r, None)
        };

    let is_effectful = effect_ir.is_some();
    let anyref_ty = ctx.anyref_type(ir);

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

    // Effectful functions receive a done_k (continuation closure) as the last parameter.
    // The function calls done_k(result) instead of func.return on normal completion.
    if is_effectful {
        let mut done_k_arg = BlockArgData {
            ty: anyref_ty,
            attrs: Default::default(),
        };
        done_k_arg.attrs.insert(
            Symbol::new("bind_name"),
            Attribute::Symbol(Symbol::new("__done_k")),
        );
        block_args.push(done_k_arg);
    }

    let entry_block = ir.create_block(BlockData {
        location,
        args: block_args,
        ops: Default::default(),
        parent_region: None,
    });

    // Bind parameters to their block argument values
    {
        let mut scope = ctx.scope();
        for (i, param) in func_decl.params.iter().enumerate() {
            if let Some(local_id) = param.local_id {
                let arg_val = ir.block_arg(entry_block, i as u32);
                scope.bind(local_id, param.name, arg_val);
            }
        }

        // For effectful functions: set done_k and use CPS body lowering.
        // The done_k continuation is called at the end of the continuation chain
        // instead of func.return, handled by build_cps_continuation.
        if is_effectful {
            let prev_done_k = scope.done_k;
            let done_k_idx = func_decl.params.len() as u32;
            let done_k_val = ir.block_arg(entry_block, done_k_idx);
            scope.done_k = Some(done_k_val);

            let mut builder = IrBuilder::new(&mut scope, ir, entry_block);
            let result = expr::lower_block_cps_for_body(&mut builder, func_decl.body);

            if let Some((body_result, is_cps)) = result
                && !is_cps
            {
                // Pure result: call done_k(result) instead of func.return
                let result_anyref = builder.cast_if_needed(location, body_result, anyref_ty);
                super::emit_done_k_call(&mut builder, location, done_k_val, result_anyref);
            }
            // If is_cps: ability.perform or effectful call already handles tail-call

            scope.done_k = prev_done_k;
        } else {
            // Pure function: lower body normally
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
        }
    }

    // Build function type
    // Effectful functions: add done_k param and return anyref
    let (final_param_types, final_return_ty) = if is_effectful {
        let mut params = param_ir_types.clone();
        params.push(anyref_ty);
        (params, anyref_ty)
    } else {
        (param_ir_types.clone(), return_ty)
    };
    let func_type = ctx.func_type_with_effect(ir, &final_param_types, final_return_ty, effect_ir);

    // Create body region and func op
    let body_region = ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![entry_block],
        parent_op: None,
    });

    let qualified_name = ctx.qualify_name(func_name);
    let func_op = func::func(ir, location, qualified_name, func_type, body_region);
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

    let (param_ir_types, return_ty, effect_ir) = {
        let scheme = ctx
            .lookup_function_type(qualified_name)
            .cloned()
            .unwrap_or_else(|| {
                panic!(
                    "extern function '{}' should have TypeScheme from type checking",
                    qualified_name
                )
            });
        let body = scheme.body(ctx.db);
        match body.kind(ctx.db) {
            TypeKind::Func {
                params,
                result,
                effect,
            } => {
                let p: Vec<TypeRef> = params.iter().map(|t| ctx.convert_type(ir, *t)).collect();
                let r = ctx.convert_type(ir, *result);
                let e = if effect.is_pure(ctx.db) {
                    None
                } else {
                    Some(ctx.convert_effect_row(ir, *effect))
                };
                (p, r, e)
            }
            other => {
                unreachable!("extern function `{func_name}` has non-function TypeScheme: {other:?}")
            }
        }
    };

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
    let func_type = ctx.func_type_with_effect(ir, &param_ir_types, return_ty, effect_ir);

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
    let struct_ty = ctx.adt_typeref(ir, name);

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

        let getter_func_type = ctx.func_type_with_effect(ir, &[struct_ty], *field_type, None);
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
) -> (Vec<TypeRef>, TypeRef) {
    let params = func
        .params
        .iter()
        .map(|p| convert_annotation_to_ir_type(ctx, ir, p.ty.as_ref()))
        .collect();
    let ret = func
        .return_ty
        .as_ref()
        .map(|ann| convert_annotation_to_ir_type(ctx, ir, Some(ann)))
        .unwrap_or_else(|| ctx.nil_type(ir));
    (params, ret)
}
