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
use crate::cont_util::compute_op_idx;

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

        lower_performs_in_blocks(ctx, &blocks, evidence_val, &types, true);
    }
}

/// Find the evidence parameter in a block's arguments.
fn find_evidence_arg(ctx: &IrContext, block: BlockRef) -> Option<ValueRef> {
    let args = ctx.block_args(block);
    args.iter()
        .find(|&&arg| arena_ability::is_evidence_type_ref(ctx, ctx.value_ty(arg)))
        .copied()
}

/// Lower all `ability.perform` ops in the given blocks.
///
/// Only performs in tail position of a function body are lowered (the CPS
/// transform is expected to have placed them there). Performs found inside
/// nested scf/handler regions are skipped — they should not exist after CPS
/// conversion and indicate a pipeline bug if encountered.
fn lower_performs_in_blocks(
    ctx: &mut IrContext,
    blocks: &[BlockRef],
    evidence_val: Option<ValueRef>,
    types: &YieldBubblingTypes,
    in_function_body: bool,
) {
    for &block in blocks {
        // Collect perform ops in this block (snapshot, since we'll mutate).
        let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();
        for (i, op) in ops.iter().enumerate() {
            let op = *op;
            if arena_ability::Perform::matches(ctx, op) {
                let is_last_op = i == ops.len() - 1;
                assert!(
                    in_function_body && is_last_op,
                    "ability.perform must be in tail position of a function body; \
                     found in {} position of a {} region",
                    if is_last_op { "tail" } else { "non-tail" },
                    if in_function_body {
                        "function"
                    } else {
                        "non-function"
                    },
                );
                lower_single_perform(ctx, block, op, evidence_val, types);
            }

            // Recurse into nested regions (these are NOT function bodies).
            let regions: Vec<_> = ctx.op(op).regions.to_vec();
            for region in regions {
                let inner_blocks: Vec<BlockRef> = ctx.region(region).blocks.to_vec();
                lower_performs_in_blocks(ctx, &inner_blocks, evidence_val, types, false);
            }
        }
    }
}

/// Lower a single `ability.perform` op.
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
    // Destructure eagerly so the borrow on ctx is released before mutable use below.
    let (continuation_val, value_operand) = {
        let operands = ctx.op_operands(op);
        let (&cont, rest) = operands
            .split_first()
            .expect("malformed ability.perform: missing continuation operand");
        let val = match rest {
            [] => None,
            &[single] => Some(single),
            _ => panic!(
                "ability.perform with multiple payload operands is not yet supported (got {})",
                rest.len()
            ),
        };
        (cont, val)
    };

    let t = types;
    let mut new_ops: Vec<OpRef> = Vec::new();

    // === 1. Evidence lookup → marker ===
    let Some(ev) = evidence_val else {
        // No evidence parameter in the enclosing function — this is a bug.
        // Every function containing ability.perform must receive evidence.
        panic!(
            "ability.perform requires evidence parameter but enclosing function has none (at {:?})",
            location
        );
    };
    let marker_ty = arena_ability::marker_adt_type_ref(ctx);
    let lookup = arena_ability::evidence_lookup(ctx, location, ev, marker_ty, ability_ref_type);
    new_ops.push(lookup.op_ref());
    let marker_val = lookup.result(ctx);

    // === 2. Extract prompt tag from marker (field index 1) ===
    let prompt_get = adt::struct_get(ctx, location, marker_val, t.i32, marker_ty, 1);
    new_ops.push(prompt_get.op_ref());
    let prompt_val = prompt_get.result(ctx);

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
    let shift_value_val = if let Some(sv) = value_operand {
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

    // === 6. Build ShiftInfo ===
    let shift_info_op = adt::struct_new(
        ctx,
        location,
        vec![
            shift_value_val,
            prompt_val,
            op_idx_const.result(ctx),
            cont_anyref.result(ctx),
        ],
        t.shift_info,
        t.shift_info,
    );
    new_ops.push(shift_info_op.op_ref());

    // === 7. Build YieldResult::Shift ===
    let yr_op = adt::variant_new(
        ctx,
        location,
        [shift_info_op.result(ctx)],
        t.yield_result,
        t.yield_result,
        Symbol::new("Shift"),
    );
    new_ops.push(yr_op.op_ref());

    // === 8. func.return the YieldResult ===
    let ret_op = func::r#return(ctx, location, [yr_op.result(ctx)]);
    new_ops.push(ret_op.op_ref());

    // Insert all new ops before the perform op, then remove it.
    // Map the perform result to the YieldResult value.
    let old_result = ctx.op_result(op, 0);
    let new_result = yr_op.result(ctx);
    ctx.replace_all_uses(old_result, new_result);

    for new_op in &new_ops {
        ctx.insert_op_before(block, op, *new_op);
    }
    ctx.remove_op_from_block(block, op);
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
        let _i32_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());

        // Build: func.func @test_fn(%ev: Evidence) -> YieldResult {
        //   %k = ... (dummy closure value)
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

        // Create a dummy closure value via arith.const (just need a ValueRef)
        let dummy_k = arith::r#const(&mut ctx, loc, anyref_ty, Attribute::Int(0));
        ctx.push_op(func_entry, dummy_k.op_ref());
        let k_val = dummy_k.result(&ctx);

        // ability.perform %k, [] { ability_ref: @State, op_name: @get }
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

        // Should have adt.variant_new (YieldResult::Shift).
        assert!(
            ops.iter().any(|&o| adt::VariantNew::matches(&ctx, o)),
            "should have adt.variant_new for YieldResult::Shift"
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
}
