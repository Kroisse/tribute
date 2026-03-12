//! Push prompt body outlining for libmprompt backend.
//!
//! `mp_prompt` takes a function pointer, so the body of `cont.push_prompt`
//! must be outlined into a separate top-level `func.func`.
//!
//! Steps:
//! 1. Compute live-ins (values used inside but defined outside the body)
//! 2. Create an outlined function that unpacks live-ins from an env struct
//! 3. Replace the push_prompt with: build env struct -> call `__tribute_prompt`

use std::cell::Cell;
use std::collections::HashSet;

use tribute_ir::dialect::ability as arena_ability;
use trunk_ir::Symbol;
use trunk_ir::context::{BlockArgData, BlockData, IrContext, RegionData};
use trunk_ir::dialect::{
    adt as arena_adt, arith, cont as arena_cont, core as arena_core, func as arena_func,
    scf as arena_scf,
};
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{OpRef, RegionRef, TypeRef, ValueRef};
use trunk_ir::rewrite::{PatternRewriter, RewritePattern};
use trunk_ir::smallvec::smallvec;
use trunk_ir::types::{Attribute, TypeDataBuilder};

/// Pattern: Lower `cont.push_prompt` -> body outlining + `__tribute_prompt` call.
pub(crate) struct LowerPushPromptPattern {
    body_counter: Cell<u32>,
}

impl LowerPushPromptPattern {
    pub(crate) fn new() -> Self {
        Self {
            body_counter: Cell::new(0),
        }
    }
}

impl RewritePattern for LowerPushPromptPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(push_prompt) = arena_cont::PushPrompt::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let i32_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());
        let ptr_ty = arena_core::ptr(ctx).as_type_ref();

        let tag = push_prompt.tag(ctx);
        let body = push_prompt.body(ctx);

        // Compute live-ins for the body region
        let live_ins = compute_live_ins(ctx, body);

        // Generate unique name
        let body_idx = self.body_counter.get();
        self.body_counter.set(body_idx + 1);
        let body_name = format!("__prompt_body_{body_idx}");

        // Generate outlined body function and add to module
        let outlined_func = generate_outlined_body(ctx, &body_name, &live_ins, body, loc);
        rewriter.add_module_op(outlined_func);

        // Build the call site
        let env_val = if live_ins.is_empty() {
            let null = arith::r#const(ctx, loc, ptr_ty, Attribute::Int(0));
            rewriter.insert_op(null.op_ref());
            null.result(ctx)
        } else {
            let mut field_vals = Vec::new();
            let mut field_types = Vec::new();
            for &(value, ty) in &live_ins {
                if arena_ability::is_evidence_type_ref(ctx, ty) {
                    let cast = arena_core::unrealized_conversion_cast(ctx, loc, value, ptr_ty);
                    rewriter.insert_op(cast.op_ref());
                    field_vals.push(cast.result(ctx));
                    field_types.push(ptr_ty);
                } else if ty != ptr_ty {
                    let cast = arena_core::unrealized_conversion_cast(ctx, loc, value, ptr_ty);
                    rewriter.insert_op(cast.op_ref());
                    field_vals.push(cast.result(ctx));
                    field_types.push(ptr_ty);
                } else {
                    field_vals.push(value);
                    field_types.push(ptr_ty);
                }
            }

            let env_struct_ty = build_env_struct_type(ctx, &field_types);
            let struct_new =
                arena_adt::struct_new(ctx, loc, field_vals, env_struct_ty, env_struct_ty);
            rewriter.insert_op(struct_new.op_ref());

            let env_cast =
                arena_core::unrealized_conversion_cast(ctx, loc, struct_new.result(ctx), ptr_ty);
            rewriter.insert_op(env_cast.op_ref());
            env_cast.result(ctx)
        };

        // %body_fn = func.constant @__prompt_body_N
        let body_fn = arena_func::constant(ctx, loc, ptr_ty, Symbol::from_dynamic(&body_name));
        rewriter.insert_op(body_fn.op_ref());

        // %tag_val: use the runtime tag operand if present (set by resolve_evidence),
        // otherwise fall back to the static tag attribute.
        let tag_val = if !ctx.op_operands(op).is_empty() {
            // Runtime tag from resolve_evidence (first operand)
            ctx.op_operands(op)[0]
        } else {
            let tag_bits = match tag {
                Attribute::Int(bits) => bits,
                other => {
                    unreachable!("cont.push_prompt expected Int tag, got {other:?} at {loc:?}")
                }
            };
            let c = arith::r#const(ctx, loc, i32_ty, Attribute::Int(tag_bits));
            rewriter.insert_op(c.op_ref());
            c.result(ctx)
        };

        // %result = func.call @__tribute_prompt(%tag_val, %body_fn, %env)
        let prompt_call = arena_func::call(
            ctx,
            loc,
            [tag_val, body_fn.result(ctx), env_val],
            ptr_ty,
            Symbol::new("__tribute_prompt"),
        );

        // Cast result back if needed
        let result_types: Vec<_> = ctx.op_result_types(op).to_vec();
        let original_result_ty = result_types.first().copied();
        if let Some(result_ty) = original_result_ty
            && result_ty != ptr_ty
        {
            rewriter.insert_op(prompt_call.op_ref());
            let cast = arena_core::unrealized_conversion_cast(
                ctx,
                loc,
                prompt_call.result(ctx),
                result_ty,
            );
            rewriter.replace_op(cast.op_ref());
        } else {
            rewriter.replace_op(prompt_call.op_ref());
        }
        true
    }

    fn name(&self) -> &'static str {
        "LowerPushPromptPattern"
    }
}

/// Compute live-in values for a region.
///
/// A value is a live-in if it is used (as an operand) inside the region
/// but defined outside (not a block arg or operation result within the region,
/// including nested sub-regions).
pub(super) fn compute_live_ins(ctx: &IrContext, region: RegionRef) -> Vec<(ValueRef, TypeRef)> {
    let mut defined: HashSet<ValueRef> = HashSet::new();
    let mut used: Vec<ValueRef> = Vec::new();
    let mut seen_used: HashSet<ValueRef> = HashSet::new();

    // Collect all defined values in the region
    collect_defined_in_region(ctx, region, &mut defined);

    // Collect all used values
    let blocks = ctx.region(region).blocks.clone();
    for &block in &blocks {
        for &op in &ctx.block(block).ops.clone() {
            collect_used_values(ctx, op, &mut used, &mut seen_used);
        }
    }

    // Live-ins = used but not defined
    let mut live_ins: Vec<(ValueRef, TypeRef)> = Vec::new();
    let mut live_in_set: HashSet<ValueRef> = HashSet::new();

    for v in used {
        if !defined.contains(&v) && live_in_set.insert(v) {
            live_ins.push((v, ctx.value_ty(v)));
        }
    }

    live_ins
}

fn collect_defined_in_region(ctx: &IrContext, region: RegionRef, defined: &mut HashSet<ValueRef>) {
    let blocks = ctx.region(region).blocks.clone();
    for &block in &blocks {
        // Block arguments are defined
        for &arg in ctx.block_args(block) {
            defined.insert(arg);
        }

        // Operation results are defined
        let ops = ctx.block(block).ops.clone();
        for &op in &ops {
            for &result in ctx.op_results(op) {
                defined.insert(result);
            }
            // Recurse into nested regions
            let regions = ctx.op(op).regions.clone();
            for r in regions {
                collect_defined_in_region(ctx, r, defined);
            }
        }
    }
}

fn collect_used_values(
    ctx: &IrContext,
    op: OpRef,
    used: &mut Vec<ValueRef>,
    seen: &mut HashSet<ValueRef>,
) {
    for &v in ctx.op_operands(op) {
        if seen.insert(v) {
            used.push(v);
        }
    }

    let regions = ctx.op(op).regions.clone();
    for r in regions {
        let blocks = ctx.region(r).blocks.clone();
        for &block in &blocks {
            let ops = ctx.block(block).ops.clone();
            for &nested_op in &ops {
                collect_used_values(ctx, nested_op, used, seen);
            }
        }
    }
}

/// Generate a `func.func` for an outlined prompt body.
///
/// Uses explicit value remapping instead of `replace_all_uses` to avoid
/// corrupting values used outside the body region.
fn generate_outlined_body(
    ctx: &mut IrContext,
    name: &str,
    live_ins: &[(ValueRef, TypeRef)],
    body_region: RegionRef,
    loc: trunk_ir::types::Location,
) -> OpRef {
    use super::handler_dispatch::clone_op_into_block_with_remap;
    use std::collections::HashMap;

    let ptr_ty = arena_core::ptr(ctx).as_type_ref();

    // Create entry block with ptr parameter (the env struct)
    let entry_block = ctx.create_block(BlockData {
        location: loc,
        args: vec![BlockArgData {
            ty: ptr_ty,
            attrs: Default::default(),
        }],
        ops: smallvec![],
        parent_region: None,
    });
    let env_value = ctx.block_args(entry_block)[0];

    // Build value remap: live-in values -> extracted values from env struct
    let mut value_remap: HashMap<ValueRef, ValueRef> = HashMap::new();

    if !live_ins.is_empty() {
        let field_types: Vec<TypeRef> = live_ins
            .iter()
            .map(|&(_, ty)| {
                if arena_ability::is_evidence_type_ref(ctx, ty) {
                    ptr_ty
                } else {
                    ptr_ty
                }
            })
            .collect();
        let env_struct_ty = build_env_struct_type(ctx, &field_types);

        // Cast ptr -> struct type
        let env_cast = arena_core::unrealized_conversion_cast(ctx, loc, env_value, env_struct_ty);
        ctx.push_op(entry_block, env_cast.op_ref());
        let env_ref = env_cast.result(ctx);

        for (i, &(orig_value, orig_ty)) in live_ins.iter().enumerate() {
            let field_ty = field_types[i];
            let field = arena_adt::struct_get(ctx, loc, env_ref, field_ty, env_struct_ty, i as u32);
            ctx.push_op(entry_block, field.op_ref());

            let extracted = if orig_ty != field_ty {
                let cast =
                    arena_core::unrealized_conversion_cast(ctx, loc, field.result(ctx), orig_ty);
                ctx.push_op(entry_block, cast.op_ref());
                cast.result(ctx)
            } else {
                field.result(ctx)
            };

            value_remap.insert(orig_value, extracted);
        }
    }

    // Copy body operations with remapping
    let body_blocks = ctx.region(body_region).blocks.clone();
    let mut last_result: Option<ValueRef> = None;

    if let Some(&body_block) = body_blocks.first() {
        let body_ops: Vec<OpRef> = ctx.block(body_block).ops.clone().to_vec();
        for &op in &body_ops {
            if arena_scf::Yield::matches(ctx, op) {
                let operands: Vec<ValueRef> = ctx.op_operands(op).to_vec();
                last_result = operands
                    .first()
                    .map(|&v| value_remap.get(&v).copied().unwrap_or(v));
                continue;
            }
            if arena_func::Return::matches(ctx, op) {
                let operands: Vec<ValueRef> = ctx.op_operands(op).to_vec();
                last_result = operands
                    .first()
                    .map(|&v| value_remap.get(&v).copied().unwrap_or(v));
                continue;
            }

            // Clone op into entry block with value remapping
            clone_op_into_block_with_remap(ctx, entry_block, op, &value_remap);

            // Map old results -> new results for subsequent ops
            let new_ops = ctx.block(entry_block).ops.clone();
            if let Some(&new_op) = new_ops.last() {
                let old_results: Vec<ValueRef> = ctx.op_results(op).to_vec();
                let new_results: Vec<ValueRef> = ctx.op_results(new_op).to_vec();
                for (old_r, new_r) in old_results.into_iter().zip(new_results) {
                    value_remap.insert(old_r, new_r);
                }

                if !ctx.op_results(new_op).is_empty() {
                    last_result = Some(ctx.op_results(new_op)[0]);
                }
            }
        }
    }

    // Add func.return with result cast to ptr
    if let Some(result) = last_result {
        if ctx.value_ty(result) != ptr_ty {
            let cast = arena_core::unrealized_conversion_cast(ctx, loc, result, ptr_ty);
            ctx.push_op(entry_block, cast.op_ref());
            let ret = arena_func::r#return(ctx, loc, [cast.result(ctx)]);
            ctx.push_op(entry_block, ret.op_ref());
        } else {
            let ret = arena_func::r#return(ctx, loc, [result]);
            ctx.push_op(entry_block, ret.op_ref());
        }
    } else {
        let null = arith::r#const(ctx, loc, ptr_ty, Attribute::Int(0));
        ctx.push_op(entry_block, null.op_ref());
        let ret = arena_func::r#return(ctx, loc, [null.result(ctx)]);
        ctx.push_op(entry_block, ret.op_ref());
    }

    let body = ctx.create_region(RegionData {
        location: loc,
        blocks: smallvec![entry_block],
        parent_op: None,
    });

    let func_ty = arena_core::func(ctx, ptr_ty, [ptr_ty], None).as_type_ref();

    let func_op = arena_func::func(ctx, loc, Symbol::from_dynamic(name), func_ty, body);
    func_op.op_ref()
}

/// Build an env struct type with the given field types.
///
/// Matches the structure of `adt::struct_type` from the Salsa-based API:
/// - `name` attribute: struct name
/// - `fields` attribute: list of [name, type] pairs
/// - params: field types
fn build_env_struct_type(ctx: &mut IrContext, field_types: &[TypeRef]) -> TypeRef {
    // Build fields attribute: list of [Symbol(f0), Type(ty0)], [Symbol(f1), Type(ty1)], ...
    let fields_attr: Vec<Attribute> = field_types
        .iter()
        .enumerate()
        .map(|(i, &ty)| {
            Attribute::List(vec![
                Attribute::Symbol(Symbol::from_dynamic(&format!("f{i}"))),
                Attribute::Type(ty),
            ])
        })
        .collect();

    let mut builder = TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("struct"))
        .attr("name", Attribute::Symbol(Symbol::new("__prompt_env")))
        .attr("fields", Attribute::List(fields_attr));
    for &ty in field_types {
        builder = builder.param(ty);
    }
    ctx.types.intern(builder.build())
}
