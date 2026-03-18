//! Lower `closure.lambda` operations to `func.func` + `closure.new`.
//!
//! This pass performs closure conversion on high-level `closure.lambda` ops:
//!
//! ```text
//! // Before:
//! %k = closure.lambda [%x, %y] {
//!   ^bb0(%param: anyref):
//!     %r = arith.add %x, %param
//!     func.return %r
//! }
//!
//! // After:
//! func.func @__lambda_0(%ev: Evidence, %env: anyref, %param: anyref) -> anyref {
//!   %env_cast = adt.ref_cast %env : env_struct
//!   %x = adt.struct_get %env_cast, 0
//!   %y = adt.struct_get %env_cast, 1
//!   %r = arith.add %x, %param
//!   func.return %r
//! }
//! %env = adt.struct_new(%x, %y)
//! %k = closure.new @__lambda_0, %env
//! ```

use trunk_ir::Symbol;
use trunk_ir::context::{BlockArgData, BlockData, IrContext, RegionData};
use trunk_ir::dialect::{adt, core, func};
use trunk_ir::ir_mapping::IrMapping;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{BlockRef, OpRef, TypeRef, ValueRef};
use trunk_ir::rewrite::Module;
use trunk_ir::types::{Attribute, TypeDataBuilder};

use tribute_ir::dialect::ability as arena_ability;
use tribute_ir::dialect::closure as arena_closure;
use tribute_ir::dialect::tribute_rt;

/// Lower all `closure.lambda` ops in the module to `func.func` + `closure.new`.
pub fn lower_closure_lambda(ctx: &mut IrContext, module: Module) {
    let module_block = match module.first_block(ctx) {
        Some(b) => b,
        None => return,
    };

    let mut counter: u32 = 0;

    // Iterate until no more closure.lambda ops remain.
    // Multiple iterations handle nested lambdas: the first pass converts
    // outermost lambdas, which may reveal inner closure.lambda ops that were
    // inside the cloned body regions.
    loop {
        let lambdas = collect_closure_lambdas(ctx, module);
        if lambdas.is_empty() {
            break;
        }

        for lambda_ref in lambdas {
            lower_single_lambda(ctx, module_block, lambda_ref, &mut counter);
        }
    }
}

// ============================================================================
// Collection
// ============================================================================

/// Collect all `closure.lambda` ops in the module (DFS, outermost first).
fn collect_closure_lambdas(ctx: &IrContext, module: Module) -> Vec<OpRef> {
    let mut result = Vec::new();
    for op in module.ops(ctx) {
        collect_lambdas_in_op(ctx, op, &mut result);
    }
    result
}

fn collect_lambdas_in_op(ctx: &IrContext, op: OpRef, result: &mut Vec<OpRef>) {
    if arena_closure::Lambda::matches(ctx, op) {
        // Don't recurse into the lambda body — nested lambdas will be
        // found on the next iteration after this one is lowered.
        result.push(op);
        return;
    }
    for &region in ctx.op(op).regions.iter() {
        for &block in ctx.region(region).blocks.iter() {
            for &child_op in ctx.block(block).ops.iter() {
                collect_lambdas_in_op(ctx, child_op, result);
            }
        }
    }
}

// ============================================================================
// Single lambda lowering
// ============================================================================

fn lower_single_lambda(
    ctx: &mut IrContext,
    module_block: BlockRef,
    lambda_ref: OpRef,
    counter: &mut u32,
) {
    let Ok(lambda_op) = arena_closure::Lambda::from_op(ctx, lambda_ref) else {
        return;
    };

    let location = ctx.op(lambda_ref).location;
    let result_ty = ctx.op_result_types(lambda_ref)[0];

    // Gather captures and their types.
    let captures: Vec<ValueRef> = ctx.op_operands(lambda_ref).to_vec();
    let capture_types: Vec<TypeRef> = captures.iter().map(|&v| ctx.value_ty(v)).collect();

    let body_region = lambda_op.body(ctx);

    // Generate unique lifted name.
    let lifted_name = Symbol::from_dynamic(&format!("__lambda_{}", *counter));
    *counter += 1;

    // Original entry block args = lambda formal parameters.
    let orig_entry = ctx.region(body_region).blocks[0];
    let orig_param_count = ctx.block(orig_entry).args.len();
    let orig_param_types: Vec<TypeRef> = (0..orig_param_count)
        .map(|i| ctx.block(orig_entry).args[i].ty)
        .collect();

    // Determine function return type from the closure result type.
    let anyref_ty = tribute_rt::anyref(ctx).as_type_ref();
    let func_result_ty = extract_return_type_from_closure(ctx, result_ty).unwrap_or(anyref_ty);
    let evidence_ty = arena_ability::evidence_adt_type_ref(ctx);

    // Build env struct type for captures.
    let env_struct_ty = if captures.is_empty() {
        None
    } else {
        let fields: Vec<(Symbol, TypeRef)> = capture_types
            .iter()
            .enumerate()
            .map(|(i, &ty)| (Symbol::from_dynamic(&format!("_{i}")), ty))
            .collect();
        let env_name = Symbol::from_dynamic(&format!("{lifted_name}::env"));
        Some(make_adt_struct_type(ctx, env_name, &fields))
    };

    // === Build the lifted function ===
    let func_body_region = build_lifted_body(
        ctx,
        location,
        body_region,
        &LiftBodyParams {
            captures: &captures,
            capture_types: &capture_types,
            env_struct_ty,
            orig_param_types: &orig_param_types,
            evidence_ty,
            anyref_ty,
        },
    );

    // Function type: (evidence, env, params...) -> result, preserving effect row.
    let effect = extract_effect_from_closure(ctx, result_ty);
    let mut all_param_tys = vec![evidence_ty, anyref_ty];
    all_param_tys.extend_from_slice(&orig_param_types);
    let func_ty =
        core::func(ctx, func_result_ty, all_param_tys.iter().copied(), effect).as_type_ref();

    let func_op = func::func(ctx, location, lifted_name, func_ty, func_body_region);
    ctx.push_op(module_block, func_op.op_ref());

    // === Replace closure.lambda at the original site ===
    let parent_block = ctx
        .op(lambda_ref)
        .parent_block
        .expect("closure.lambda should be in a block");

    // Pack captures into env struct (or null).
    let closure_env = if captures.is_empty() {
        let null_op = adt::ref_null(ctx, location, anyref_ty, anyref_ty);
        ctx.insert_op_before(parent_block, lambda_ref, null_op.op_ref());
        null_op.result(ctx)
    } else {
        let env_ty = env_struct_ty.unwrap();
        let struct_op = adt::struct_new(ctx, location, captures.clone(), env_ty, env_ty);
        ctx.insert_op_before(parent_block, lambda_ref, struct_op.op_ref());
        struct_op.result(ctx)
    };

    // Create closure.new replacing the lambda.
    let closure_func_ty =
        core::func(ctx, func_result_ty, orig_param_types.iter().copied(), None).as_type_ref();
    let closure_ty = arena_closure::closure(ctx, closure_func_ty).as_type_ref();
    let closure_new_op = arena_closure::new(ctx, location, closure_env, closure_ty, lifted_name);
    ctx.insert_op_before(parent_block, lambda_ref, closure_new_op.op_ref());

    // RAUW: lambda result → closure.new result.
    let old_result = ctx.op_result(lambda_ref, 0);
    let new_result = closure_new_op.result(ctx);
    ctx.replace_all_uses(old_result, new_result);

    // Remove the original closure.lambda op.
    ctx.remove_op_from_block(parent_block, lambda_ref);
}

// ============================================================================
// Body construction
// ============================================================================

/// Parameters for building a lifted function body.
struct LiftBodyParams<'a> {
    captures: &'a [ValueRef],
    capture_types: &'a [TypeRef],
    env_struct_ty: Option<TypeRef>,
    orig_param_types: &'a [TypeRef],
    evidence_ty: TypeRef,
    anyref_ty: TypeRef,
}

/// Build the lifted function body region by cloning the lambda body
/// with capture values remapped to env struct field extractions.
fn build_lifted_body(
    ctx: &mut IrContext,
    location: trunk_ir::types::Location,
    orig_body: trunk_ir::refs::RegionRef,
    params: &LiftBodyParams<'_>,
) -> trunk_ir::refs::RegionRef {
    let captures = params.captures;
    let capture_types = params.capture_types;
    let env_struct_ty = params.env_struct_ty;
    let orig_param_types = params.orig_param_types;
    let evidence_ty = params.evidence_ty;
    let anyref_ty = params.anyref_ty;
    let orig_blocks: Vec<BlockRef> = ctx.region(orig_body).blocks.to_vec();
    let orig_entry = orig_blocks[0];
    let orig_param_count = orig_param_types.len();

    let mut mapping = IrMapping::new();

    // --- Pass 1: Create new entry block with [evidence, env, params...] ---
    let mut new_entry_args = Vec::new();
    new_entry_args.push(BlockArgData {
        ty: evidence_ty,
        attrs: make_bind_name_attrs("__evidence"),
    });
    new_entry_args.push(BlockArgData {
        ty: anyref_ty,
        attrs: make_bind_name_attrs("__env"),
    });
    for (i, &param_ty) in orig_param_types.iter().enumerate() {
        new_entry_args.push(BlockArgData {
            ty: param_ty,
            attrs: ctx.block(orig_entry).args[i].attrs.clone(),
        });
    }

    let new_entry = ctx.create_block(BlockData {
        location,
        args: new_entry_args,
        ops: Default::default(),
        parent_region: None,
    });

    mapping.map_block(orig_entry, new_entry);

    // Map old param args → new param args (shifted by 2 for evidence + env).
    for i in 0..orig_param_count {
        let old_arg = ctx.block_arg(orig_entry, i as u32);
        let new_arg = ctx.block_arg(new_entry, (i + 2) as u32);
        mapping.map_value(old_arg, new_arg);
    }

    // Insert env extraction ops at the start of the new entry block.
    let raw_env_val = ctx.block_arg(new_entry, 1);
    if let Some(env_ty) = env_struct_ty {
        let cast_op = adt::ref_cast(ctx, location, raw_env_val, env_ty, env_ty);
        ctx.push_op(new_entry, cast_op.op_ref());
        let env_val = cast_op.result(ctx);

        for (i, &cap_val) in captures.iter().enumerate() {
            let cap_ty = capture_types[i];
            let get_op = adt::struct_get(ctx, location, env_val, cap_ty, env_ty, i as u32);
            ctx.push_op(new_entry, get_op.op_ref());
            mapping.map_value(cap_val, get_op.result(ctx));
        }
    }

    // --- Pass 1b: Create remaining blocks (non-entry) ---
    let mut new_blocks = vec![new_entry];
    for &orig_block in &orig_blocks[1..] {
        let arg_data: Vec<BlockArgData> = ctx
            .block(orig_block)
            .args
            .iter()
            .map(|a| BlockArgData {
                ty: a.ty,
                attrs: a.attrs.clone(),
            })
            .collect();
        let block_loc = ctx.block(orig_block).location;
        let new_block = ctx.create_block(BlockData {
            location: block_loc,
            args: arg_data,
            ops: Default::default(),
            parent_region: None,
        });
        mapping.map_block(orig_block, new_block);

        let arg_count = ctx.block(orig_block).args.len();
        for j in 0..arg_count {
            let old_arg = ctx.block_arg(orig_block, j as u32);
            let new_arg = ctx.block_arg(new_block, j as u32);
            mapping.map_value(old_arg, new_arg);
        }
        new_blocks.push(new_block);
    }

    // --- Pass 2: Clone ops from original blocks to new blocks ---
    for (idx, &orig_block) in orig_blocks.iter().enumerate() {
        let new_block = new_blocks[idx];
        let ops: Vec<OpRef> = ctx.block(orig_block).ops.to_vec();
        for op in ops {
            ctx.clone_op_into_block(new_block, op, &mut mapping);
        }
    }

    // Create the function body region.
    ctx.create_region(RegionData {
        location,
        blocks: new_blocks.into(),
        parent_op: None,
    })
}

// ============================================================================
// Helpers
// ============================================================================

/// Extract return type from `closure.closure<core.func<Return, Params...>>`.
fn extract_return_type_from_closure(ctx: &IrContext, closure_ty: TypeRef) -> Option<TypeRef> {
    let func_ty = extract_func_type_from_closure(ctx, closure_ty)?;
    let func_data = ctx.types.get(func_ty);
    // core.func params[0] = return type
    func_data.params.first().copied()
}

/// Extract effect attribute from the func type inside a closure type.
fn extract_effect_from_closure(ctx: &IrContext, closure_ty: TypeRef) -> Option<TypeRef> {
    let func_ty = extract_func_type_from_closure(ctx, closure_ty)?;
    let func_data = ctx.types.get(func_ty);
    match func_data.attrs.get(&Symbol::new("effect")) {
        Some(Attribute::Type(ty)) => Some(*ty),
        _ => None,
    }
}

/// Extract the `core.func` TypeRef from a `closure.closure<core.func<...>>`.
fn extract_func_type_from_closure(ctx: &IrContext, closure_ty: TypeRef) -> Option<TypeRef> {
    let data = ctx.types.get(closure_ty);
    if data.dialect != Symbol::new("closure") || data.name != Symbol::new("closure") {
        return None;
    }
    let func_ty = *data.params.first()?;
    let func_data = ctx.types.get(func_ty);
    if func_data.dialect != Symbol::new("core") || func_data.name != Symbol::new("func") {
        return None;
    }
    Some(func_ty)
}

/// Create an `adt.struct` type with name and fields.
fn make_adt_struct_type(
    ctx: &mut IrContext,
    name: Symbol,
    fields: &[(Symbol, TypeRef)],
) -> TypeRef {
    let fields_attr: Vec<Attribute> = fields
        .iter()
        .map(|(field_name, field_type)| {
            Attribute::List(vec![
                Attribute::Symbol(*field_name),
                Attribute::Type(*field_type),
            ])
        })
        .collect();

    ctx.types.intern(
        TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("struct"))
            .attr("name", Attribute::Symbol(name))
            .attr("fields", Attribute::List(fields_attr))
            .build(),
    )
}

/// Create a `bind_name` attribute map for a block argument.
fn make_bind_name_attrs(name: &str) -> std::collections::BTreeMap<Symbol, Attribute> {
    let mut attrs = std::collections::BTreeMap::new();
    attrs.insert(
        Symbol::new("bind_name"),
        Attribute::Symbol(Symbol::from_dynamic(name)),
    );
    attrs
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::context::RegionData;
    use trunk_ir::dialect::{arith, core as arena_core};
    use trunk_ir::refs::PathRef;
    use trunk_ir::types::Location;
    use trunk_ir::{Attribute, IrContext, OperationDataBuilder, Span};

    fn test_ctx() -> (IrContext, Location) {
        let ctx = IrContext::new();
        let loc = Location::new(PathRef::from_u32(0), Span::default());
        (ctx, loc)
    }

    fn make_module(ctx: &mut IrContext, loc: Location) -> (Module, BlockRef) {
        let module_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: Default::default(),
            parent_region: None,
        });
        let module_region = ctx.create_region(RegionData {
            location: loc,
            blocks: trunk_ir::smallvec::smallvec![module_block],
            parent_op: None,
        });
        let module_op = OperationDataBuilder::new(loc, Symbol::new("core"), Symbol::new("module"))
            .attr("sym_name", Attribute::Symbol(Symbol::new("test")))
            .region(module_region)
            .build(ctx);
        let module_ref = ctx.create_op(module_op);
        (Module::new(ctx, module_ref).unwrap(), module_block)
    }

    fn make_anyref_ty(ctx: &mut IrContext) -> TypeRef {
        tribute_rt::anyref(ctx).as_type_ref()
    }

    fn make_i32_ty(ctx: &mut IrContext) -> TypeRef {
        ctx.types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build())
    }

    #[test]
    fn test_lower_lambda_no_captures() {
        let (mut ctx, loc) = test_ctx();
        let (module, module_block) = make_module(&mut ctx, loc);
        let anyref_ty = make_anyref_ty(&mut ctx);
        let i32_ty = make_i32_ty(&mut ctx);

        // Build a func.func containing a closure.lambda with no captures:
        //   func.func @test_fn() {
        //     %k = closure.lambda [] {
        //       ^bb0(%x: i32):
        //         func.return %x
        //     }
        //   }

        // Lambda body: ^bb0(%x: i32): func.return %x
        let lambda_entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![BlockArgData {
                ty: i32_ty,
                attrs: make_bind_name_attrs("x"),
            }],
            ops: Default::default(),
            parent_region: None,
        });
        let x_val = ctx.block_arg(lambda_entry, 0);
        let ret_op = func::r#return(&mut ctx, loc, [x_val]);
        ctx.push_op(lambda_entry, ret_op.op_ref());

        let lambda_body_region = ctx.create_region(RegionData {
            location: loc,
            blocks: trunk_ir::smallvec::smallvec![lambda_entry],
            parent_op: None,
        });

        // closure type: closure.closure<core.func<i32, i32>>
        let func_ty = arena_core::func(&mut ctx, i32_ty, [i32_ty], None).as_type_ref();
        let closure_ty = arena_closure::closure(&mut ctx, func_ty).as_type_ref();

        // closure.lambda [] { ... } -> closure_ty
        let lambda_op = arena_closure::lambda(
            &mut ctx,
            loc,
            Vec::<ValueRef>::new(),
            closure_ty,
            lambda_body_region,
        );

        // Wrap in a func.func
        let outer_entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: Default::default(),
            parent_region: None,
        });
        ctx.push_op(outer_entry, lambda_op.op_ref());

        let outer_body = ctx.create_region(RegionData {
            location: loc,
            blocks: trunk_ir::smallvec::smallvec![outer_entry],
            parent_op: None,
        });
        let outer_func_ty =
            arena_core::func(&mut ctx, anyref_ty, std::iter::empty::<TypeRef>(), None)
                .as_type_ref();
        let outer_func = func::func(
            &mut ctx,
            loc,
            Symbol::new("test_fn"),
            outer_func_ty,
            outer_body,
        );
        ctx.push_op(module_block, outer_func.op_ref());

        // Run the pass.
        lower_closure_lambda(&mut ctx, module);

        // Verify: module should now have 2 functions (test_fn + __lambda_0).
        let ops = module.ops(&ctx);
        assert_eq!(ops.len(), 2, "expected 2 top-level ops after lowering");

        // The original function should contain closure.new instead of closure.lambda.
        let test_fn = func::Func::from_op(&ctx, ops[0]).unwrap();
        let test_fn_body = test_fn.body(&ctx);
        let test_fn_entry = ctx.region(test_fn_body).blocks[0];
        let test_fn_ops: Vec<OpRef> = ctx.block(test_fn_entry).ops.to_vec();

        // Should have: adt.ref_null (env) + closure.new
        assert!(test_fn_ops.len() >= 2, "expected at least 2 ops in test_fn");
        let last_op = test_fn_ops[test_fn_ops.len() - 1];
        assert!(
            arena_closure::New::matches(&ctx, last_op),
            "last op should be closure.new"
        );

        // The lifted function should exist.
        let lifted = func::Func::from_op(&ctx, ops[1]).unwrap();
        assert_eq!(lifted.sym_name(&ctx), Symbol::from_dynamic("__lambda_0"));

        // Lifted function should have 3 params: evidence, env, x
        let lifted_ty = lifted.r#type(&ctx);
        let lifted_ty_data = ctx.types.get(lifted_ty);
        // params[0] = return, params[1..] = param types
        assert_eq!(lifted_ty_data.params.len(), 4); // return + evidence + env + x
    }

    #[test]
    fn test_lower_lambda_with_captures() {
        let (mut ctx, loc) = test_ctx();
        let (module, module_block) = make_module(&mut ctx, loc);
        let anyref_ty = make_anyref_ty(&mut ctx);
        let i32_ty = make_i32_ty(&mut ctx);

        // Build:
        //   func.func @test_fn(%a: i32) {
        //     %k = closure.lambda [%a] {
        //       ^bb0(%x: i32):
        //         %r = arith.add %a, %x
        //         func.return %r
        //     }
        //   }

        // Outer function entry block with one param %a
        let outer_entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![BlockArgData {
                ty: i32_ty,
                attrs: make_bind_name_attrs("a"),
            }],
            ops: Default::default(),
            parent_region: None,
        });
        let a_val = ctx.block_arg(outer_entry, 0);

        // Lambda body: ^bb0(%x: i32): %r = arith.add %a, %x; func.return %r
        let lambda_entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![BlockArgData {
                ty: i32_ty,
                attrs: make_bind_name_attrs("x"),
            }],
            ops: Default::default(),
            parent_region: None,
        });
        let x_val = ctx.block_arg(lambda_entry, 0);

        let add_op = arith::add(&mut ctx, loc, a_val, x_val, i32_ty);
        ctx.push_op(lambda_entry, add_op.op_ref());
        let add_result = add_op.result(&ctx);

        let ret_op = func::r#return(&mut ctx, loc, [add_result]);
        ctx.push_op(lambda_entry, ret_op.op_ref());

        let lambda_body_region = ctx.create_region(RegionData {
            location: loc,
            blocks: trunk_ir::smallvec::smallvec![lambda_entry],
            parent_op: None,
        });

        let func_ty = arena_core::func(&mut ctx, i32_ty, [i32_ty], None).as_type_ref();
        let closure_ty = arena_closure::closure(&mut ctx, func_ty).as_type_ref();

        // closure.lambda [%a] { ... }
        let lambda_op =
            arena_closure::lambda(&mut ctx, loc, vec![a_val], closure_ty, lambda_body_region);
        ctx.push_op(outer_entry, lambda_op.op_ref());

        let outer_body = ctx.create_region(RegionData {
            location: loc,
            blocks: trunk_ir::smallvec::smallvec![outer_entry],
            parent_op: None,
        });
        let outer_func_ty = arena_core::func(&mut ctx, anyref_ty, [i32_ty], None).as_type_ref();
        let outer_func = func::func(
            &mut ctx,
            loc,
            Symbol::new("test_fn"),
            outer_func_ty,
            outer_body,
        );
        ctx.push_op(module_block, outer_func.op_ref());

        // Run the pass.
        lower_closure_lambda(&mut ctx, module);

        // Verify: 2 functions.
        let ops = module.ops(&ctx);
        assert_eq!(ops.len(), 2);

        // Original function should now have: adt.struct_new + closure.new
        let test_fn = func::Func::from_op(&ctx, ops[0]).unwrap();
        let test_fn_body = test_fn.body(&ctx);
        let test_fn_entry = ctx.region(test_fn_body).blocks[0];
        let test_fn_ops: Vec<OpRef> = ctx.block(test_fn_entry).ops.to_vec();

        let has_struct_new = test_fn_ops
            .iter()
            .any(|&op| adt::StructNew::matches(&ctx, op));
        assert!(has_struct_new, "should have adt.struct_new for env packing");

        let has_closure_new = test_fn_ops
            .iter()
            .any(|&op| arena_closure::New::matches(&ctx, op));
        assert!(has_closure_new, "should have closure.new");

        // Lifted function body should reference env extraction, not the original capture.
        let lifted = func::Func::from_op(&ctx, ops[1]).unwrap();
        let lifted_body = lifted.body(&ctx);
        let lifted_entry = ctx.region(lifted_body).blocks[0];
        let lifted_ops: Vec<OpRef> = ctx.block(lifted_entry).ops.to_vec();

        // Should have: ref_cast + struct_get (env extraction) + add + return
        assert!(
            lifted_ops.len() >= 4,
            "expected at least 4 ops in lifted fn, got {}",
            lifted_ops.len()
        );
        assert!(
            adt::RefCast::matches(&ctx, lifted_ops[0]),
            "first op should be adt.ref_cast"
        );
        assert!(
            adt::StructGet::matches(&ctx, lifted_ops[1]),
            "second op should be adt.struct_get"
        );
    }
}
