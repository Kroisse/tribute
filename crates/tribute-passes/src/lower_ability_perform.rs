//! Lower `ability.perform` operations to YieldResult::Shift construction.
//!
//! In CPS-based effect handling, `ability.perform` carries an explicit
//! continuation closure. This pass converts it to:
//!
//! ```text
//! // Input:
//! %yr = ability.perform %continuation, [%args...]
//!   { ability_ref: @State, op_name: @get }
//!
//! // Output:
//! %marker = ability.evidence_lookup %evidence { ability_ref: @State }
//! %prompt = adt.struct_get %marker, 1       // prompt_tag
//! %op_idx = arith.const <hash(ability, op)>
//! %shift_value = %args (or null)
//! %cont_anyref = cast %continuation to anyref
//! %shift_info = adt.struct_new(%shift_value, %prompt, %op_idx, %cont_anyref)
//! %yr = adt.variant_new Shift(%shift_info) : YieldResult
//! func.return %yr
//! ```
//!
//! Unlike the old `cont.shift` lowering, no resume function extraction or
//! live variable analysis is needed — the continuation closure already
//! captures the entire remaining computation.

use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::{adt, arith, core, func};
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{BlockRef, OpRef, ValueRef};
use trunk_ir::rewrite::Module;
use trunk_ir::types::Attribute;

use tribute_ir::dialect::ability as arena_ability;

use crate::cont_to_yield_bubbling::types::YieldBubblingTypes;
use tribute_ir::dialect::ability::compute_op_idx;

/// Lower all `ability.perform` ops in the module to YieldResult::Shift.
pub fn lower_ability_perform(ctx: &mut IrContext, module: Module) {
    let types = YieldBubblingTypes::new(ctx);

    let func_ops: Vec<OpRef> = module.ops(ctx);
    for func_op_ref in func_ops {
        let Ok(func_op) = func::Func::from_op(ctx, func_op_ref) else {
            continue;
        };

        let body = func_op.body(ctx);
        let blocks: Vec<BlockRef> = ctx.region(body).blocks.to_vec();
        if blocks.is_empty() {
            continue;
        }

        // Find evidence value: first block arg with evidence type.
        let entry = blocks[0];
        let evidence_val = find_evidence_arg(ctx, entry);

        lower_performs_in_blocks(ctx, &blocks, evidence_val, &types);
    }
}

/// Find the return type of the enclosing `func.func`.
fn find_enclosing_func_return_type(
    ctx: &IrContext,
    block: BlockRef,
) -> Option<trunk_ir::refs::TypeRef> {
    let region = ctx.block(block).parent_region?;
    let parent_op = ctx.region(region).parent_op?;
    let func_op = func::Func::from_op(ctx, parent_op).ok()?;
    let func_ty = func_op.r#type(ctx);
    let td = ctx.types.get(func_ty);
    // core.func layout: params[0] = return type
    if td.dialect == trunk_ir::Symbol::new("core") && td.name == trunk_ir::Symbol::new("func") {
        td.params.first().copied()
    } else {
        None
    }
}

/// Check if a block belongs directly to a `func.func` body region.
///
/// Returns `true` if the block's parent region's parent op is `func.func`.
/// Returns `false` if the block is inside a nested region (e.g., `cont.push_prompt`
/// body, `scf.if` region, etc.).
fn is_in_func_body(ctx: &IrContext, block: BlockRef) -> bool {
    let Some(region) = ctx.block(block).parent_region else {
        return false;
    };
    let Some(parent_op) = ctx.region(region).parent_op else {
        return false;
    };
    func::Func::matches(ctx, parent_op)
}

/// Find the evidence parameter in a block's arguments.
fn find_evidence_arg(ctx: &IrContext, block: BlockRef) -> Option<ValueRef> {
    ctx.block_args(block)
        .iter()
        .find(|&&arg| arena_ability::is_evidence_type_ref(ctx, ctx.value_ty(arg)))
        .copied()
}

/// Lower all `ability.perform` ops in the given blocks.
fn lower_performs_in_blocks(
    ctx: &mut IrContext,
    blocks: &[BlockRef],
    evidence_val: Option<ValueRef>,
    types: &YieldBubblingTypes,
) {
    for &block in blocks {
        // Collect perform ops in this block (snapshot, since we'll mutate).
        let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();
        for op in ops {
            if arena_ability::Perform::matches(ctx, op) {
                lower_single_perform(ctx, block, op, evidence_val, types);
            }

            // Recurse into nested regions.
            let regions: Vec<_> = ctx.op(op).regions.to_vec();
            for region in regions {
                let inner_blocks: Vec<BlockRef> = ctx.region(region).blocks.to_vec();
                lower_performs_in_blocks(ctx, &inner_blocks, evidence_val, types);
            }
        }
    }
}

/// Lower a single `ability.perform` op to tail-call handler_dispatch.
///
/// Replaces ability.perform with:
/// 1. evidence_lookup → marker
/// 2. Extract handler_dispatch from marker (field 3)
/// 3. Compute op_idx
/// 4. Pack shift value
/// 5. Cast continuation to anyref
/// 6. func.call_indirect handler_dispatch(k, op_idx, value)
///    (This becomes a tail call after func_to_clif lowering)
fn lower_single_perform(
    ctx: &mut IrContext,
    block: BlockRef,
    op: OpRef,
    evidence_val: Option<ValueRef>,
    types: &YieldBubblingTypes,
) {
    let Ok(perform_op) = arena_ability::Perform::from_op(ctx, op) else {
        return;
    };

    let location = ctx.op(op).location;
    let ability_ref_type = perform_op.ability_ref(ctx);
    let op_name_sym = perform_op.op_name(ctx);

    // Operands: [continuation, ...values]
    let operands: Vec<ValueRef> = ctx.op_operands(op).to_vec();
    let continuation_val = operands[0];
    let value_operands: Vec<ValueRef> = operands[1..].to_vec();

    let t = types;
    let mut new_ops: Vec<OpRef> = Vec::new();

    // === 1. Evidence lookup → marker ===
    let marker_ty = arena_ability::marker_adt_type_ref(ctx);
    let marker_val = if let Some(ev) = evidence_val {
        let lookup = arena_ability::evidence_lookup(ctx, location, ev, marker_ty, ability_ref_type);
        new_ops.push(lookup.op_ref());
        lookup.result(ctx)
    } else {
        let null_op = adt::ref_null(ctx, location, marker_ty, marker_ty);
        new_ops.push(null_op.op_ref());
        null_op.result(ctx)
    };

    // === 2. Extract handler_dispatch from marker (field index 3) ===
    let handler_get = adt::struct_get(ctx, location, marker_val, t.anyref, marker_ty, 3);
    new_ops.push(handler_get.op_ref());
    let handler_dispatch_val = handler_get.result(ctx);

    // === 3. Compute op_idx ===
    let ability_data = ctx.types.get(ability_ref_type);
    let ability_name = match ability_data.attrs.get(&Symbol::new("name")) {
        Some(Attribute::Symbol(s)) => Some(*s),
        _ => None,
    };
    let op_idx = compute_op_idx(ability_name, Some(op_name_sym));
    let op_idx_const = arith::r#const(ctx, location, t.i32, Attribute::Int(op_idx as i128));
    new_ops.push(op_idx_const.op_ref());

    // === 4. Build shift value (pack args or null) ===
    let shift_value_val = if let Some(&sv) = value_operands.first() {
        let cast = core::unrealized_conversion_cast(ctx, location, sv, t.anyref);
        new_ops.push(cast.op_ref());
        cast.result(ctx)
    } else {
        let null_op = adt::ref_null(ctx, location, t.anyref, t.anyref);
        new_ops.push(null_op.op_ref());
        null_op.result(ctx)
    };

    // === 5. Cast continuation closure to anyref ===
    let cont_anyref = core::unrealized_conversion_cast(ctx, location, continuation_val, t.anyref);
    new_ops.push(cont_anyref.op_ref());

    // === 6. Decompose handler_dispatch closure {table_idx, env} and call ===
    // handler_dispatch is a closure struct (adt.struct{i32, anyref}{name=_closure}).
    // Extract the function pointer and environment, then call with env prepended.
    // Include evidence as arg[0] (after table_idx) so that resolve_evidence can
    // replace it with the correct extended evidence later in the pipeline.
    let closure_ty = crate::closure_lower::closure_struct_type_ref(ctx);
    let fn_ptr_get = adt::struct_get(ctx, location, handler_dispatch_val, t.i32, closure_ty, 0);
    new_ops.push(fn_ptr_get.op_ref());
    let fn_ptr = fn_ptr_get.result(ctx);

    let env_get = adt::struct_get(ctx, location, handler_dispatch_val, t.anyref, closure_ty, 1);
    new_ops.push(env_get.op_ref());
    let env_val = env_get.result(ctx);

    // Placeholder evidence: resolve_evidence will replace operands[1] with
    // the correct evidence value. Use the current evidence or null.
    let placeholder_ev = if let Some(ev) = evidence_val {
        ev
    } else {
        let null_ev = adt::ref_null(ctx, location, t.anyref, t.anyref);
        new_ops.push(null_ev.op_ref());
        null_ev.result(ctx)
    };

    let call_op = func::call_indirect(
        ctx,
        location,
        fn_ptr,
        vec![
            placeholder_ev,
            env_val,
            cont_anyref.result(ctx),
            op_idx_const.result(ctx),
            shift_value_val,
        ],
        t.anyref,
    );
    new_ops.push(call_op.op_ref());

    // === 7. Return the handler's result (cast to function return type) ===
    if is_in_func_body(ctx, block) {
        let result_val = call_op.result(ctx);
        // Find the enclosing func's return type to cast the anyref result.
        let func_ret_ty = find_enclosing_func_return_type(ctx, block);
        let casted = if let Some(ret_ty) = func_ret_ty {
            if ret_ty != t.anyref {
                let cast = core::unrealized_conversion_cast(ctx, location, result_val, ret_ty);
                new_ops.push(cast.op_ref());
                cast.result(ctx)
            } else {
                result_val
            }
        } else {
            result_val
        };
        let ret_op = func::r#return(ctx, location, [casted]);
        new_ops.push(ret_op.op_ref());
    }

    // Insert all new ops before the perform op, then remove it.
    let old_result = ctx.op_result(op, 0);
    let new_result = call_op.result(ctx);
    ctx.replace_all_uses(old_result, new_result);

    for new_op in &new_ops {
        ctx.insert_op_before(block, op, *new_op);
    }

    // Remove the perform op AND any dead code after it in the same block.
    // After lowering, the handler_dispatch call + func.return replaces the
    // perform op. Any ops that followed perform (e.g., a previous func.return
    // from the body closure) are now dead and must be removed to avoid
    // "block already filled" errors in Cranelift.
    let block_ops: Vec<OpRef> = ctx.block(block).ops.to_vec();
    let perform_idx = block_ops.iter().position(|&o| o == op);
    if let Some(idx) = perform_idx {
        // Remove perform and everything after it
        for &dead_op in &block_ops[idx..] {
            ctx.remove_op_from_block(block, dead_op);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::context::{BlockArgData, BlockData, RegionData};
    use trunk_ir::dialect::core as arena_core;
    use trunk_ir::refs::PathRef;
    use trunk_ir::types::{Location, TypeDataBuilder};
    use trunk_ir::{IrContext, OperationDataBuilder, Span};

    use tribute_ir::dialect::tribute_rt;

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

    fn make_ability_ref_type(ctx: &mut IrContext, name: &str) -> trunk_ir::TypeRef {
        ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ability_ref"))
                .attr("name", Attribute::Symbol(Symbol::from_dynamic(name)))
                .build(),
        )
    }

    #[test]
    fn test_lower_perform_basic() {
        let (mut ctx, loc) = test_ctx();
        let (module, module_block) = make_module(&mut ctx, loc);

        let anyref_ty = tribute_rt::anyref(&mut ctx).as_type_ref();
        let evidence_ty = arena_ability::evidence_adt_type_ref(&mut ctx);

        // Build: func.func @test_fn(%ev: Evidence) {
        //   %k = arith.const 0   (dummy continuation)
        //   %yr = ability.perform %k, [] { ability_ref: @State, op_name: @get }
        // }
        let func_entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![BlockArgData {
                ty: evidence_ty,
                attrs: Default::default(),
            }],
            ops: Default::default(),
            parent_region: None,
        });
        let _ev_val = ctx.block_arg(func_entry, 0);

        let dummy_k = arith::r#const(&mut ctx, loc, anyref_ty, Attribute::Int(0));
        ctx.push_op(func_entry, dummy_k.op_ref());
        let k_val = dummy_k.result(&ctx);

        let state_ref = make_ability_ref_type(&mut ctx, "State");
        let perform_op = arena_ability::perform(
            &mut ctx,
            loc,
            k_val,
            Vec::<ValueRef>::new(),
            anyref_ty,
            state_ref,
            Symbol::new("get"),
        );
        ctx.push_op(func_entry, perform_op.op_ref());

        let func_body = ctx.create_region(RegionData {
            location: loc,
            blocks: trunk_ir::smallvec::smallvec![func_entry],
            parent_op: None,
        });
        let yr_ty = {
            let types = YieldBubblingTypes::new(&mut ctx);
            types.yield_result
        };
        let func_ty = arena_core::func(&mut ctx, yr_ty, [evidence_ty], None).as_type_ref();
        let test_func = func::func(&mut ctx, loc, Symbol::new("test_fn"), func_ty, func_body);
        ctx.push_op(module_block, test_func.op_ref());

        // Run the pass.
        lower_ability_perform(&mut ctx, module);

        // Verify: the ability.perform should be replaced.
        let test_fn = func::Func::from_op(&ctx, module.ops(&ctx)[0]).unwrap();
        let body = test_fn.body(&ctx);
        let entry = ctx.region(body).blocks[0];
        let ops: Vec<OpRef> = ctx.block(entry).ops.to_vec();

        // Should NOT have ability.perform anymore.
        assert!(
            !ops.iter()
                .any(|&o| arena_ability::Perform::matches(&ctx, o)),
            "ability.perform should be replaced"
        );

        // Should have func.call_indirect (handler dispatch call).
        assert!(
            ops.iter().any(|&o| func::CallIndirect::matches(&ctx, o)),
            "should have func.call_indirect for handler dispatch"
        );

        // Should have func.return as the last op.
        let last_op = *ops.last().unwrap();
        assert!(
            func::Return::matches(&ctx, last_op),
            "last op should be func.return"
        );

        // Should have ability.evidence_lookup.
        assert!(
            ops.iter()
                .any(|&o| arena_ability::EvidenceLookup::matches(&ctx, o)),
            "should have ability.evidence_lookup"
        );
    }

    /// Integration test: closure.lambda + ability.perform → lowered IR.
    ///
    /// Simulates the CPS output of ast_to_ir for `State::get()`:
    /// 1. A function containing closure.lambda (continuation) + ability.perform
    /// 2. lower_closure_lambda extracts the lambda into a top-level function
    /// 3. lower_ability_perform converts perform → YieldResult::Shift
    #[test]
    fn test_cps_shift_path_integration() {
        let (mut ctx, loc) = test_ctx();
        let (module, module_block) = make_module(&mut ctx, loc);

        let anyref_ty = tribute_rt::anyref(&mut ctx).as_type_ref();
        let evidence_ty = arena_ability::evidence_adt_type_ref(&mut ctx);
        let i32_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());

        // Simulate CPS output for: fn foo() ->{State} Int { State::get() }
        //
        // func.func @foo(%ev: Evidence) {
        //   // continuation: fn(result) { func.return result }
        //   %k = closure.lambda [] {
        //     ^bb0(%result: i32):
        //       func.return %result
        //   }
        //   %yr = ability.perform %k, [] { ability_ref: @State, op_name: @get }
        //   func.return %yr
        // }

        // Build the closure.lambda body: ^bb0(%result: i32): func.return %result
        let lambda_entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![BlockArgData {
                ty: i32_ty,
                attrs: Default::default(),
            }],
            ops: Default::default(),
            parent_region: None,
        });
        let lambda_result = ctx.block_arg(lambda_entry, 0);
        let lambda_ret = func::r#return(&mut ctx, loc, [lambda_result]);
        ctx.push_op(lambda_entry, lambda_ret.op_ref());

        let lambda_body_region = ctx.create_region(RegionData {
            location: loc,
            blocks: trunk_ir::smallvec::smallvec![lambda_entry],
            parent_op: None,
        });

        // closure type
        let func_ty_inner = arena_core::func(&mut ctx, i32_ty, [i32_ty], None).as_type_ref();
        let closure_ty =
            tribute_ir::dialect::closure::closure(&mut ctx, func_ty_inner).as_type_ref();

        // Build the main function
        let func_entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![BlockArgData {
                ty: evidence_ty,
                attrs: Default::default(),
            }],
            ops: Default::default(),
            parent_region: None,
        });

        // closure.lambda [] { ... }
        let lambda_op = tribute_ir::dialect::closure::lambda(
            &mut ctx,
            loc,
            Vec::<ValueRef>::new(),
            closure_ty,
            lambda_body_region,
        );
        ctx.push_op(func_entry, lambda_op.op_ref());
        let k_val = lambda_op.result(&ctx);

        // ability.perform %k, [] { @State, @get }
        let state_ref = make_ability_ref_type(&mut ctx, "State");
        let perform_op = arena_ability::perform(
            &mut ctx,
            loc,
            k_val,
            Vec::<ValueRef>::new(),
            anyref_ty,
            state_ref,
            Symbol::new("get"),
        );
        ctx.push_op(func_entry, perform_op.op_ref());

        // func.return %yr
        let yr_val = perform_op.result(&ctx);
        let ret_op = func::r#return(&mut ctx, loc, [yr_val]);
        ctx.push_op(func_entry, ret_op.op_ref());

        let func_body = ctx.create_region(RegionData {
            location: loc,
            blocks: trunk_ir::smallvec::smallvec![func_entry],
            parent_op: None,
        });
        let yr_ty = YieldBubblingTypes::new(&mut ctx).yield_result;
        let func_ty = arena_core::func(&mut ctx, yr_ty, [evidence_ty], None).as_type_ref();
        let foo_func = func::func(&mut ctx, loc, Symbol::new("foo"), func_ty, func_body);
        ctx.push_op(module_block, foo_func.op_ref());

        // === Pass 1: lower_closure_lambda ===
        crate::lower_closure_lambda::lower_closure_lambda(&mut ctx, module);

        // Verify: module should now have 2 functions (foo + __clam_0)
        let ops_after_lambda = module.ops(&ctx);
        assert_eq!(
            ops_after_lambda.len(),
            2,
            "expected foo + lifted lambda function"
        );

        // foo should now have closure.new instead of closure.lambda
        let foo = func::Func::from_op(&ctx, ops_after_lambda[0]).unwrap();
        let foo_body = foo.body(&ctx);
        let foo_entry = ctx.region(foo_body).blocks[0];
        let foo_ops: Vec<OpRef> = ctx.block(foo_entry).ops.to_vec();
        assert!(
            foo_ops
                .iter()
                .any(|&o| tribute_ir::dialect::closure::New::matches(&ctx, o)),
            "foo should have closure.new after lower_closure_lambda"
        );
        assert!(
            foo_ops
                .iter()
                .any(|&o| arena_ability::Perform::matches(&ctx, o)),
            "foo should still have ability.perform"
        );

        // === Pass 2: lower_ability_perform ===
        lower_ability_perform(&mut ctx, module);

        // Verify: ability.perform should be replaced with YieldResult construction
        let ops_after_perform = module.ops(&ctx);
        let foo2 = func::Func::from_op(&ctx, ops_after_perform[0]).unwrap();
        let foo2_body = foo2.body(&ctx);
        let foo2_entry = ctx.region(foo2_body).blocks[0];
        let foo2_ops: Vec<OpRef> = ctx.block(foo2_entry).ops.to_vec();

        assert!(
            !foo2_ops
                .iter()
                .any(|&o| arena_ability::Perform::matches(&ctx, o)),
            "ability.perform should be gone after lowering"
        );
        assert!(
            foo2_ops
                .iter()
                .any(|&o| func::CallIndirect::matches(&ctx, o)),
            "should have func.call_indirect for handler dispatch"
        );

        // The lifted lambda function should exist as foo::__clam_0
        let lifted = func::Func::from_op(&ctx, ops_after_perform[1]).unwrap();
        let lifted_name = lifted.sym_name(&ctx).to_string();
        assert!(
            lifted_name.contains("__clam_"),
            "lifted function should be scope::__clam_N, got: {}",
            lifted_name
        );
    }
}
