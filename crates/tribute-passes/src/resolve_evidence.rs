//! Evidence-based dispatch resolution pass.
//!
//! This pass transforms static `cont.shift` operations into evidence-based dispatch
//! using runtime function calls.
//!
//! ## Transformations
//!
//! ```text
//! // Before (static placeholder tag)
//! %result = cont.shift(%placeholder_tag, args...) { ability_ref, op_name }
//!
//! // After (dynamic evidence-based tag)
//! %marker = func.call @__tribute_evidence_lookup(%ev, ability_id)
//! %tag = adt.struct_get(%marker, 1)  // field 1 = prompt_tag
//! %result = cont.shift(%tag, args...) { ability_ref, op_name }
//! ```
//!
//! This pass must run AFTER `add_evidence_params` so that effectful functions
//! have evidence as their first parameter.

use std::collections::{HashMap, HashSet};

use tribute_ir::arena::dialect::ability as arena_ability;
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::adt as arena_adt;
use trunk_ir::dialect::arith;
use trunk_ir::dialect::cont as arena_cont;
use trunk_ir::dialect::core as arena_core;
use trunk_ir::dialect::func as arena_func;
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{BlockRef, OpRef, RegionRef, TypeRef, ValueRef};
use trunk_ir::rewrite::Module;
use trunk_ir::types::{Attribute, Location, TypeDataBuilder};

/// Sentinel value used for unresolved cont.shift tags.
/// When a shift is generated without an enclosing handler, this value is used
/// as a placeholder. The evidence pass should transform all such shifts.
pub const UNRESOLVED_SHIFT_TAG: u32 = u32::MAX;

// ============================================================================
// OpTable Registry
// ============================================================================

/// Registry for managing op_table_index assignments.
///
/// Only tracks handler count for index assignment; entry data is not retained
/// because the pipeline does not need to look up entries after registration.
#[derive(Debug, Default)]
struct OpTableRegistry {
    next_index: u32,
}

impl OpTableRegistry {
    fn new() -> Self {
        Self { next_index: 0 }
    }

    fn register(
        &mut self,
        _abilities: Vec<TypeRef>,
        _operations: Vec<(TypeRef, Symbol)>,
        _location: Location,
    ) -> u32 {
        let index = self.next_index;
        self.next_index += 1;
        index
    }
}

// ============================================================================
// Helper type constructors
// ============================================================================

fn i32_type_ref(ctx: &mut IrContext) -> TypeRef {
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build())
}

// ============================================================================
// Analysis helpers
// ============================================================================

/// Ensure runtime helper functions exist in the module.
fn ensure_runtime_functions(ctx: &mut IrContext, module: Module) {
    let ops = module.ops(ctx);

    let mut has_lookup = false;
    let mut has_extend = false;

    for op in &ops {
        if let Ok(func_op) = arena_func::Func::from_op(ctx, *op) {
            let name = func_op.sym_name(ctx);
            if name == Symbol::new("__tribute_evidence_lookup") {
                has_lookup = true;
            } else if name == Symbol::new("__tribute_evidence_extend") {
                has_extend = true;
            }
        }
    }

    if has_lookup && has_extend {
        return;
    }

    let Some(module_block) = module.first_block(ctx) else {
        return;
    };
    let loc = ctx.op(module.op()).location;

    let first_existing_op = ctx.block(module_block).ops.first().copied();

    if !has_lookup {
        let evidence_ty = arena_ability::evidence_adt_type_ref(ctx);
        let i32_ty = i32_type_ref(ctx);
        let marker_ty = arena_ability::marker_adt_type_ref(ctx);

        // fn __tribute_evidence_lookup(ev: Evidence, ability_id: i32) -> Marker
        let func_ty = arena_core::func(ctx, marker_ty, [evidence_ty, i32_ty], None).as_type_ref();

        // Body with unreachable
        let body_block = ctx.create_block(trunk_ir::context::BlockData {
            location: loc,
            args: vec![
                trunk_ir::context::BlockArgData {
                    ty: evidence_ty,
                    attrs: Default::default(),
                },
                trunk_ir::context::BlockArgData {
                    ty: i32_ty,
                    attrs: Default::default(),
                },
            ],
            ops: Default::default(),
            parent_region: None,
        });
        let unreachable_op = arena_func::unreachable(ctx, loc);
        ctx.push_op(body_block, unreachable_op.op_ref());
        let body = ctx.create_region(trunk_ir::context::RegionData {
            location: loc,
            blocks: trunk_ir::smallvec::smallvec![body_block],
            parent_op: None,
        });
        let func_op = arena_func::func(
            ctx,
            loc,
            Symbol::new("__tribute_evidence_lookup"),
            func_ty,
            body,
        );
        if let Some(first) = first_existing_op {
            ctx.insert_op_before(module_block, first, func_op.op_ref());
        } else {
            ctx.push_op(module_block, func_op.op_ref());
        }
    }

    if !has_extend {
        let evidence_ty = arena_ability::evidence_adt_type_ref(ctx);
        let marker_ty = arena_ability::marker_adt_type_ref(ctx);

        // fn __tribute_evidence_extend(ev: Evidence, marker: Marker) -> Evidence
        let func_ty =
            arena_core::func(ctx, evidence_ty, [evidence_ty, marker_ty], None).as_type_ref();

        let body_block = ctx.create_block(trunk_ir::context::BlockData {
            location: loc,
            args: vec![
                trunk_ir::context::BlockArgData {
                    ty: evidence_ty,
                    attrs: Default::default(),
                },
                trunk_ir::context::BlockArgData {
                    ty: marker_ty,
                    attrs: Default::default(),
                },
            ],
            ops: Default::default(),
            parent_region: None,
        });
        let unreachable_op = arena_func::unreachable(ctx, loc);
        ctx.push_op(body_block, unreachable_op.op_ref());
        let body = ctx.create_region(trunk_ir::context::RegionData {
            location: loc,
            blocks: trunk_ir::smallvec::smallvec![body_block],
            parent_op: None,
        });
        let func_op = arena_func::func(
            ctx,
            loc,
            Symbol::new("__tribute_evidence_extend"),
            func_ty,
            body,
        );
        // Insert at the beginning of the module
        let first_op = ctx.block(module_block).ops.first().copied();
        if let Some(first) = first_op {
            ctx.insert_op_before(module_block, first, func_op.op_ref());
        } else {
            ctx.push_op(module_block, func_op.op_ref());
        }
    }
}

/// Collect functions with evidence first parameter.
fn collect_functions_with_evidence(ctx: &IrContext, module: Module) -> HashSet<Symbol> {
    let mut fns_with_evidence = HashSet::new();
    for op in module.ops(ctx) {
        if let Ok(func_op) = arena_func::Func::from_op(ctx, op) {
            let func_ty = func_op.r#type(ctx);
            if has_evidence_first_param(ctx, func_ty) {
                fns_with_evidence.insert(func_op.sym_name(ctx));
            }
        }
    }
    fns_with_evidence
}

/// Check if a `core.func` type has evidence as its first parameter.
fn has_evidence_first_param(ctx: &IrContext, func_ty: TypeRef) -> bool {
    let Some(func) = arena_core::Func::from_type_ref(ctx, func_ty) else {
        return false;
    };
    let params = func.params(ctx);
    if params.is_empty() {
        return false;
    }
    arena_ability::is_evidence_type_ref(ctx, params[0])
}

/// Collect handler-root functions (contain push_prompt but no evidence param).
fn collect_handler_root_functions(
    ctx: &IrContext,
    module: Module,
    fns_with_evidence: &HashSet<Symbol>,
) -> HashSet<Symbol> {
    let mut handler_roots = HashSet::new();
    for op in module.ops(ctx) {
        if let Ok(func_op) = arena_func::Func::from_op(ctx, op) {
            let func_name = func_op.sym_name(ctx);
            if fns_with_evidence.contains(&func_name) {
                continue;
            }
            if region_contains_push_prompt(ctx, func_op.body(ctx)) {
                handler_roots.insert(func_name);
            }
        }
    }
    handler_roots
}

/// Check if a region contains `cont.push_prompt`.
fn region_contains_push_prompt(ctx: &IrContext, region: RegionRef) -> bool {
    for &block in ctx.region(region).blocks.iter() {
        for &op in ctx.block(block).ops.iter() {
            if arena_cont::PushPrompt::from_op(ctx, op).is_ok() {
                return true;
            }
            for &nested in ctx.op(op).regions.iter() {
                if region_contains_push_prompt(ctx, nested) {
                    return true;
                }
            }
        }
    }
    false
}

/// Collect handled abilities by tag from handler_dispatch ops in a block.
fn collect_handled_abilities_by_tag(
    ctx: &IrContext,
    block: BlockRef,
) -> HashMap<u32, Vec<TypeRef>> {
    let mut map = HashMap::new();
    for &op in ctx.block(block).ops.iter() {
        if let Ok(dispatch_op) = arena_cont::HandlerDispatch::from_op(ctx, op) {
            let tag = dispatch_op.tag(ctx);
            let mut abilities = Vec::new();
            let body = dispatch_op.body(ctx);
            let blocks = &ctx.region(body).blocks;
            if let Some(&first_block) = blocks.first() {
                for &child_op in ctx.block(first_block).ops.iter() {
                    if let Ok(suspend_op) = arena_cont::Suspend::from_op(ctx, child_op) {
                        let ability_ref = suspend_op.ability_ref(ctx);
                        if !abilities.contains(&ability_ref) {
                            abilities.push(ability_ref);
                        }
                    }
                }
            }
            map.insert(tag, abilities);
        }
    }
    map
}

/// Collect operations for a tag from handler_dispatch.
fn collect_operations_for_tag(
    ctx: &IrContext,
    block: BlockRef,
    target_tag: u32,
) -> Vec<(TypeRef, Symbol)> {
    let mut operations = Vec::new();
    for &op in ctx.block(block).ops.iter() {
        if let Ok(dispatch_op) = arena_cont::HandlerDispatch::from_op(ctx, op) {
            let tag = dispatch_op.tag(ctx);
            if tag != target_tag {
                continue;
            }
            let body = dispatch_op.body(ctx);
            let blocks = &ctx.region(body).blocks;
            if let Some(&first_block) = blocks.first() {
                for &child_op in ctx.block(first_block).ops.iter() {
                    if let Ok(suspend_op) = arena_cont::Suspend::from_op(ctx, child_op) {
                        let ability = suspend_op.ability_ref(ctx);
                        let name = suspend_op.op_name(ctx);
                        if !operations.contains(&(ability, name)) {
                            operations.push((ability, name));
                        }
                    }
                }
            }
            break;
        }
    }
    operations
}

/// Compute ability ID from a TypeRef.
fn compute_ability_id(ctx: &IrContext, ability_ref: TypeRef) -> u32 {
    let data = ctx.types.get(ability_ref);
    // ability_ref is core.ability_ref type
    let ability_name = match data.attrs.get(&Symbol::new("name")) {
        Some(Attribute::Symbol(s)) => *s,
        _ => panic!(
            "ICE: compute_ability_id: ability type has no name: {:?}",
            data
        ),
    };

    let mut hash: u32 = ability_name.with_str(|s| {
        let mut h: u32 = 0;
        for byte in s.bytes() {
            h = h.wrapping_mul(31).wrapping_add(byte as u32);
        }
        h
    });

    // Include type parameters
    for &param in data.params.iter() {
        hash = hash.wrapping_mul(37);
        hash = hash.wrapping_add(hash_type(ctx, param));
    }

    hash
}

/// Hash a type for ability ID computation.
fn hash_type(ctx: &IrContext, ty: TypeRef) -> u32 {
    let data = ctx.types.get(ty);
    let mut hash: u32 = 0;

    data.dialect.with_str(|s| {
        for byte in s.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u32);
        }
    });

    data.name.with_str(|s| {
        for byte in s.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u32);
        }
    });

    for &param in data.params.iter() {
        hash = hash.wrapping_mul(37);
        hash = hash.wrapping_add(hash_type(ctx, param));
    }

    hash
}

/// Validate ability ID uniqueness.
fn validate_ability_id_uniqueness(ctx: &IrContext, module: Module) {
    let mut seen: HashMap<u32, TypeRef> = HashMap::new();
    for op in module.ops(ctx) {
        if let Ok(func_op) = arena_func::Func::from_op(ctx, op) {
            collect_ability_ids_in_region(ctx, func_op.body(ctx), &mut seen);
        }
    }
}

fn collect_ability_ids_in_region(
    ctx: &IrContext,
    region: RegionRef,
    seen: &mut HashMap<u32, TypeRef>,
) {
    for &block in ctx.region(region).blocks.iter() {
        for &op in ctx.block(block).ops.iter() {
            if let Ok(shift_op) = arena_cont::Shift::from_op(ctx, op) {
                let ability_ref = shift_op.ability_ref(ctx);
                let id = compute_ability_id(ctx, ability_ref);
                if let Some(existing) = seen.get(&id) {
                    if *existing != ability_ref {
                        panic!("ICE: ability ID hash collision: both hash to {id:#010x}");
                    }
                } else {
                    seen.insert(id, ability_ref);
                }
            }
            for &nested in ctx.op(op).regions.iter() {
                collect_ability_ids_in_region(ctx, nested, seen);
            }
        }
    }
}

/// Validate no unresolved shifts remain.
fn validate_no_unresolved_shifts(ctx: &IrContext, module: Module) {
    for op in module.ops(ctx) {
        if let Ok(func_op) = arena_func::Func::from_op(ctx, op) {
            validate_no_unresolved_shifts_in_region(ctx, func_op.body(ctx));
        }
    }
}

fn validate_no_unresolved_shifts_in_region(ctx: &IrContext, region: RegionRef) {
    for &block in ctx.region(region).blocks.iter() {
        for &op in ctx.block(block).ops.iter() {
            if let Ok(shift_op) = arena_cont::Shift::from_op(ctx, op)
                && let trunk_ir::refs::ValueDef::OpResult(def_op, 0) =
                    ctx.value_def(shift_op.tag(ctx))
                && let Ok(const_op) = arith::Const::from_op(ctx, def_op)
                && let Attribute::IntBits(value) = const_op.value(ctx)
                && value == UNRESOLVED_SHIFT_TAG as u64
            {
                let ability_ref = shift_op.ability_ref(ctx);
                let op_name = shift_op.op_name(ctx);
                panic!(
                    "ICE: Unresolved cont.shift found after evidence pass.\n\
                     Ability: {:?}, Op: {:?}",
                    ability_ref, op_name
                );
            }
            for &nested in ctx.op(op).regions.iter() {
                validate_no_unresolved_shifts_in_region(ctx, nested);
            }
        }
    }
}

// ============================================================================
// Transform functions
// ============================================================================

/// Transform handler-root functions (contain push_prompt but no evidence param).
fn transform_handler_roots(
    ctx: &mut IrContext,
    module: Module,
    handler_root_fns: &HashSet<Symbol>,
    fns_with_evidence: &HashSet<Symbol>,
    registry: &mut OpTableRegistry,
) {
    let func_ops: Vec<OpRef> = module.ops(ctx);
    for func_op_ref in func_ops {
        let Ok(func_op) = arena_func::Func::from_op(ctx, func_op_ref) else {
            continue;
        };
        let func_name = func_op.sym_name(ctx);
        if !handler_root_fns.contains(&func_name) {
            continue;
        }

        let body = func_op.body(ctx);
        let blocks: Vec<BlockRef> = ctx.region(body).blocks.to_vec();
        let Some(&entry_block) = blocks.first() else {
            continue;
        };

        let loc = ctx.op(func_op_ref).location;
        let evidence_ty = arena_ability::evidence_adt_type_ref(ctx);
        let i32_ty = i32_type_ref(ctx);

        // Create empty evidence: arith.const 0 + adt.array_new
        let zero_const = arith::r#const(ctx, loc, i32_ty, Attribute::IntBits(0));
        let empty_evidence = arena_adt::array_new(
            ctx,
            loc,
            vec![ctx.op_result(zero_const.op_ref(), 0)],
            evidence_ty,
            evidence_ty,
        );
        let ev_value = ctx.op_result(empty_evidence.op_ref(), 0);

        // Insert at start of entry block
        let first_op = ctx.block(entry_block).ops.first().copied();
        if let Some(first) = first_op {
            ctx.insert_op_before(entry_block, first, zero_const.op_ref());
            ctx.insert_op_before(entry_block, first, empty_evidence.op_ref());
        } else {
            ctx.push_op(entry_block, zero_const.op_ref());
            ctx.push_op(entry_block, empty_evidence.op_ref());
        }

        // Transform the entry block (prepend_evidence=true: handler roots have no evidence param)
        let handled_by_tag = collect_handled_abilities_by_tag(ctx, entry_block);
        transform_shifts_in_block(
            ctx,
            entry_block,
            ev_value,
            &handled_by_tag,
            registry,
            fns_with_evidence,
            None,
            true, // prepend evidence to calls (no evidence param yet)
        );

        // Also transform remaining blocks
        for &block in blocks.iter().skip(1) {
            let handled_by_tag = collect_handled_abilities_by_tag(ctx, block);
            transform_shifts_in_block(
                ctx,
                block,
                ev_value,
                &handled_by_tag,
                registry,
                fns_with_evidence,
                None,
                true,
            );
        }
    }
}

/// Transform shifts in all functions that have evidence.
fn transform_shifts_in_module(
    ctx: &mut IrContext,
    module: Module,
    fns_with_evidence: &HashSet<Symbol>,
    registry: &mut OpTableRegistry,
) {
    let func_ops: Vec<OpRef> = module.ops(ctx);
    for func_op_ref in func_ops {
        let Ok(func_op) = arena_func::Func::from_op(ctx, func_op_ref) else {
            continue;
        };
        let func_name = func_op.sym_name(ctx);
        if !fns_with_evidence.contains(&func_name) {
            continue;
        }

        let body = func_op.body(ctx);
        let blocks: Vec<BlockRef> = ctx.region(body).blocks.to_vec();
        let Some(&entry_block) = blocks.first() else {
            continue;
        };

        // Get evidence from first block argument
        let args = ctx.block_args(entry_block);
        if args.is_empty() {
            continue;
        }
        let ev_value = args[0];

        // Pass fns_with_evidence so that inside push_prompt bodies (where evidence
        // changes via evidence_extend), func.call evidence arguments get updated.
        // At the top level, evidence is already correct (evidence_calls added it),
        // so the first_arg == ev_value check prevents unnecessary changes.
        // prepend_evidence=false: evidence_calls already added evidence as first arg.
        for &block in blocks.iter() {
            let handled_by_tag = collect_handled_abilities_by_tag(ctx, block);
            transform_shifts_in_block(
                ctx,
                block,
                ev_value,
                &handled_by_tag,
                registry,
                fns_with_evidence,
                None,
                false, // replace evidence (already present as first arg)
            );
        }
    }
}

/// Transform shifts in a single block.
///
/// This is the core transformation function. It processes operations in order,
/// handling push_prompt, shift, func.call, and func.call_indirect.
///
/// `ev_value` is the current evidence value in scope. For push_prompt bodies,
/// this is the extended evidence (after evidence_extend calls).
///
/// `prepend_evidence`: when true, evidence is PREPENDED to call args (for handler root
/// functions where evidence_calls couldn't add evidence). When false, evidence REPLACES
/// the first arg (for functions where evidence_calls already added it).
#[allow(clippy::too_many_arguments)]
fn transform_shifts_in_block(
    ctx: &mut IrContext,
    block: BlockRef,
    ev_value: ValueRef,
    handled_by_tag: &HashMap<u32, Vec<TypeRef>>,
    registry: &mut OpTableRegistry,
    fns_with_evidence: &HashSet<Symbol>,
    dispatch_block: Option<BlockRef>,
    prepend_evidence: bool,
) {
    let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();

    for op in ops {
        // Handle cont.push_prompt
        if let Ok(push_prompt_op) = arena_cont::PushPrompt::from_op(ctx, op) {
            let loc = ctx.op(op).location;
            let tag_attr = push_prompt_op.tag(ctx);
            let tag = match &tag_attr {
                Attribute::IntBits(v) => *v as u32,
                _ => continue,
            };

            let abilities = handled_by_tag.get(&tag).cloned().unwrap_or_default();

            if abilities.is_empty() {
                // No abilities - just recurse into regions
                let body_region = push_prompt_op.body(ctx);
                let handlers_region = push_prompt_op.handlers(ctx);
                transform_shifts_in_region(
                    ctx,
                    body_region,
                    ev_value,
                    registry,
                    fns_with_evidence,
                    prepend_evidence,
                );
                transform_shifts_in_region(
                    ctx,
                    handlers_region,
                    ev_value,
                    registry,
                    fns_with_evidence,
                    prepend_evidence,
                );
                continue;
            }

            // Generate evidence_extend calls for each ability
            let i32_ty = i32_type_ref(ctx);
            let prompt_tag_ty = arena_cont::prompt_tag(ctx).as_type_ref();
            let evidence_ty = arena_ability::evidence_adt_type_ref(ctx);
            let marker_ty = arena_ability::marker_adt_type_ref(ctx);

            let dispatch_blk = dispatch_block.unwrap_or(block);
            let operations = collect_operations_for_tag(ctx, dispatch_blk, tag);
            let op_table_idx = registry.register(abilities.clone(), operations, loc);

            // Create tag constant
            let tag_const = arith::r#const(ctx, loc, prompt_tag_ty, Attribute::IntBits(tag as u64));
            let tag_val = ctx.op_result(tag_const.op_ref(), 0);
            ctx.insert_op_before(block, op, tag_const.op_ref());

            // Create op_table_index constant
            let op_table_idx_const =
                arith::r#const(ctx, loc, i32_ty, Attribute::IntBits(op_table_idx as u64));
            let op_table_idx_val = ctx.op_result(op_table_idx_const.op_ref(), 0);
            ctx.insert_op_before(block, op, op_table_idx_const.op_ref());

            // Extend evidence for each ability
            let mut current_ev = ev_value;
            for &ability_ref in &abilities {
                let ability_id = compute_ability_id(ctx, ability_ref);

                let ability_id_const =
                    arith::r#const(ctx, loc, i32_ty, Attribute::IntBits(ability_id as u64));
                let ability_id_val = ctx.op_result(ability_id_const.op_ref(), 0);
                ctx.insert_op_before(block, op, ability_id_const.op_ref());

                // Create Marker struct: { ability_id, prompt_tag, op_table_index }
                let marker_struct = arena_adt::struct_new(
                    ctx,
                    loc,
                    vec![ability_id_val, tag_val, op_table_idx_val],
                    marker_ty,
                    marker_ty,
                );
                let marker_val = ctx.op_result(marker_struct.op_ref(), 0);
                ctx.insert_op_before(block, op, marker_struct.op_ref());

                // Call __tribute_evidence_extend
                let extend_call = arena_func::call(
                    ctx,
                    loc,
                    vec![current_ev, marker_val],
                    evidence_ty,
                    Symbol::new("__tribute_evidence_extend"),
                );
                current_ev = ctx.op_result(extend_call.op_ref(), 0);
                ctx.insert_op_before(block, op, extend_call.op_ref());
            }

            // Transform body/handlers with extended evidence.
            // Pass current_ev as the evidence value so inner transforms use it.
            let body_region = push_prompt_op.body(ctx);
            let handlers_region = push_prompt_op.handlers(ctx);
            transform_shifts_in_region(
                ctx,
                body_region,
                current_ev,
                registry,
                fns_with_evidence,
                prepend_evidence,
            );
            transform_shifts_in_region(
                ctx,
                handlers_region,
                current_ev,
                registry,
                fns_with_evidence,
                prepend_evidence,
            );
            continue;
        }

        // Handle cont.shift - replace tag with evidence lookup
        if let Ok(shift_op) = arena_cont::Shift::from_op(ctx, op) {
            let loc = ctx.op(op).location;
            let ability_ref = shift_op.ability_ref(ctx);
            let op_name = shift_op.op_name(ctx);
            let ability_id = compute_ability_id(ctx, ability_ref);

            let marker_ty = arena_ability::marker_adt_type_ref(ctx);
            let i32_ty = i32_type_ref(ctx);
            let prompt_tag_ty = arena_cont::prompt_tag(ctx).as_type_ref();

            // %ability_id_const = arith.const ability_id
            let ability_id_const =
                arith::r#const(ctx, loc, i32_ty, Attribute::IntBits(ability_id as u64));
            let ability_id_val = ctx.op_result(ability_id_const.op_ref(), 0);
            ctx.insert_op_before(block, op, ability_id_const.op_ref());

            // %marker = func.call @__tribute_evidence_lookup(%ev, %ability_id)
            let lookup_call = arena_func::call(
                ctx,
                loc,
                vec![ev_value, ability_id_val],
                marker_ty,
                Symbol::new("__tribute_evidence_lookup"),
            );
            let marker_val = ctx.op_result(lookup_call.op_ref(), 0);
            ctx.insert_op_before(block, op, lookup_call.op_ref());

            // %tag = adt.struct_get(%marker, field=1) -- prompt_tag
            let struct_get_tag =
                arena_adt::struct_get(ctx, loc, marker_val, prompt_tag_ty, marker_ty, 1);
            let tag_val = ctx.op_result(struct_get_tag.op_ref(), 0);
            ctx.insert_op_before(block, op, struct_get_tag.op_ref());

            let result_ty = ctx
                .op_result_types(op)
                .first()
                .copied()
                .unwrap_or_else(|| arena_core::nil(ctx).as_type_ref());

            // Transform handler region
            let handler_region = shift_op.handler(ctx);
            transform_shifts_in_region(
                ctx,
                handler_region,
                ev_value,
                registry,
                fns_with_evidence,
                prepend_evidence,
            );

            // Get value operands (skip old tag at index 0)
            let current_operands = ctx.op_operands(op).to_vec();
            let value_operands: Vec<ValueRef> = current_operands[1..].to_vec();

            // Detach handler region before creating new shift
            ctx.detach_region(handler_region);

            // Create new shift with resolved tag
            let new_shift = arena_cont::shift(
                ctx,
                loc,
                tag_val,
                value_operands,
                result_ty,
                ability_ref,
                op_name,
                None, // op_table_index is dynamic
                None, // op_offset
                handler_region,
            );

            // Replace old result uses
            let old_result = ctx.op_result(op, 0);
            let new_result = ctx.op_result(new_shift.op_ref(), 0);
            ctx.replace_all_uses(old_result, new_result);

            ctx.insert_op_before(block, op, new_shift.op_ref());
            ctx.remove_op_from_block(block, op);
            continue;
        }

        // Handle func.call to effectful functions: update evidence argument.
        if let Ok(call_op) = arena_func::Call::from_op(ctx, op) {
            let callee = call_op.callee(ctx);
            if fns_with_evidence.contains(&callee) {
                let current_operands = ctx.op_operands(op).to_vec();
                let first_arg = current_operands.first().copied();
                if first_arg != Some(ev_value) {
                    let loc = ctx.op(op).location;
                    let result_ty = ctx
                        .op_result_types(op)
                        .first()
                        .copied()
                        .unwrap_or_else(|| arena_core::nil(ctx).as_type_ref());

                    let new_args = if prepend_evidence {
                        // Handler root: evidence_calls couldn't add evidence,
                        // so PREPEND it before all existing args.
                        let mut args = vec![ev_value];
                        args.extend(current_operands.iter().copied());
                        args
                    } else {
                        // Functions with evidence: evidence_calls already added
                        // evidence as first arg. REPLACE old evidence with current.
                        let mut args = vec![ev_value];
                        args.extend(current_operands[1..].iter().copied());
                        args
                    };

                    let new_call = arena_func::call(ctx, loc, new_args, result_ty, callee);

                    if !ctx.op_results(op).is_empty() {
                        let old_result = ctx.op_result(op, 0);
                        let new_result = ctx.op_result(new_call.op_ref(), 0);
                        ctx.replace_all_uses(old_result, new_result);
                    }

                    ctx.insert_op_before(block, op, new_call.op_ref());
                    ctx.remove_op_from_block(block, op);
                    continue;
                }
            }
        }

        // Handle func.call_indirect: replace evidence argument (index 1)
        if arena_func::CallIndirect::from_op(ctx, op).is_ok() {
            let current_operands = ctx.op_operands(op).to_vec();
            if current_operands.len() >= 2 {
                let current_ev = current_operands[1];
                if current_ev != ev_value {
                    let loc = ctx.op(op).location;
                    let result_ty = ctx
                        .op_result_types(op)
                        .first()
                        .copied()
                        .unwrap_or_else(|| arena_core::nil(ctx).as_type_ref());
                    let table_idx = current_operands[0];
                    let mut new_args = vec![ev_value];
                    new_args.extend(current_operands[2..].iter().copied());

                    let new_call =
                        arena_func::call_indirect(ctx, loc, table_idx, new_args, result_ty);

                    if !ctx.op_results(op).is_empty() {
                        let old_result = ctx.op_result(op, 0);
                        let new_result = ctx.op_result(new_call.op_ref(), 0);
                        ctx.replace_all_uses(old_result, new_result);
                    }

                    ctx.insert_op_before(block, op, new_call.op_ref());
                    ctx.remove_op_from_block(block, op);
                    continue;
                }
            }
        }

        // Recursively transform nested regions
        let regions: Vec<RegionRef> = ctx.op(op).regions.to_vec();
        for region in regions {
            transform_shifts_in_region(
                ctx,
                region,
                ev_value,
                registry,
                fns_with_evidence,
                prepend_evidence,
            );
        }
    }
}

/// Transform shifts in a region.
fn transform_shifts_in_region(
    ctx: &mut IrContext,
    region: RegionRef,
    ev_value: ValueRef,
    registry: &mut OpTableRegistry,
    fns_with_evidence: &HashSet<Symbol>,
    prepend_evidence: bool,
) {
    let blocks: Vec<BlockRef> = ctx.region(region).blocks.to_vec();
    for block in blocks {
        let handled_by_tag = collect_handled_abilities_by_tag(ctx, block);
        transform_shifts_in_block(
            ctx,
            block,
            ev_value,
            &handled_by_tag,
            registry,
            fns_with_evidence,
            None,
            prepend_evidence,
        );
    }
}

// ============================================================================
// Entry point
// ============================================================================

/// Resolve evidence-based dispatch for `cont.shift` operations.
///
/// Transforms `cont.shift` with placeholder tags into `cont.shift` with
/// dynamically resolved tags via evidence lookup. This enables proper
/// handler dispatch at runtime.
pub fn resolve_evidence_dispatch(ctx: &mut IrContext, module: Module) {
    let mut registry = OpTableRegistry::new();

    // Ensure runtime helpers exist
    ensure_runtime_functions(ctx, module);

    // Validate ability ID uniqueness
    validate_ability_id_uniqueness(ctx, module);

    // Collect functions with evidence
    let fns_with_evidence = collect_functions_with_evidence(ctx, module);

    // Transform handler-root functions first
    let handler_root_fns = collect_handler_root_functions(ctx, module, &fns_with_evidence);
    if !handler_root_fns.is_empty() {
        transform_handler_roots(
            ctx,
            module,
            &handler_root_fns,
            &fns_with_evidence,
            &mut registry,
        );
    }

    // Transform shifts in functions with evidence
    if !fns_with_evidence.is_empty() {
        transform_shifts_in_module(ctx, module, &fns_with_evidence, &mut registry);
    }

    // Validate no unresolved shifts remain
    validate_no_unresolved_shifts(ctx, module);
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::types::TypeDataBuilder;

    #[test]
    fn test_compute_ability_id() {
        let mut ctx = IrContext::new();

        let state_ref = ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ability_ref"))
                .attr("name", Attribute::Symbol(Symbol::new("State")))
                .build(),
        );
        let console_ref = ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ability_ref"))
                .attr("name", Attribute::Symbol(Symbol::new("Console")))
                .build(),
        );

        let state_id = compute_ability_id(&ctx, state_ref);
        let console_id = compute_ability_id(&ctx, console_ref);

        // Same ability should have same ID (interning gives same TypeRef)
        let state_ref2 = ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ability_ref"))
                .attr("name", Attribute::Symbol(Symbol::new("State")))
                .build(),
        );
        let state_id2 = compute_ability_id(&ctx, state_ref2);
        assert_eq!(state_id, state_id2);

        // Different abilities should have different IDs
        assert_ne!(state_id, console_id);
    }

    #[test]
    fn test_compute_ability_id_with_type_params() {
        let mut ctx = IrContext::new();

        let i32_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());

        let state_i32 = ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ability_ref"))
                .attr("name", Attribute::Symbol(Symbol::new("State")))
                .param(i32_ty)
                .build(),
        );

        let state_no_params = ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ability_ref"))
                .attr("name", Attribute::Symbol(Symbol::new("State")))
                .build(),
        );

        let id_with_params = compute_ability_id(&ctx, state_i32);
        let id_no_params = compute_ability_id(&ctx, state_no_params);

        // Same ability name but different type params should produce different IDs
        assert_ne!(id_with_params, id_no_params);
    }

    fn test_location(ctx: &mut IrContext) -> Location {
        let path = ctx.paths.intern("test".to_owned());
        Location::new(path, trunk_ir::Span::new(0, 0))
    }

    #[test]
    fn test_op_table_registry_sequential_indices() {
        let mut ctx = IrContext::new();
        let mut registry = OpTableRegistry::new();
        let loc = test_location(&mut ctx);

        let idx0 = registry.register(vec![], vec![], loc);
        let idx1 = registry.register(vec![], vec![], loc);
        let idx2 = registry.register(vec![], vec![], loc);

        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(idx2, 2);
    }
}
