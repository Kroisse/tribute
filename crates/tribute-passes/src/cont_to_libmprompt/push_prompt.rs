//! Push prompt body outlining for libmprompt backend.
//!
//! `mp_prompt` takes a function pointer, so the body of `cont.push_prompt`
//! must be outlined into a separate top-level `func.func`.
//!
//! Steps:
//! 1. Compute live-ins (values used inside but defined outside the body)
//! 2. Create an outlined function that unpacks live-ins from an env struct
//! 3. Replace the push_prompt with: build env struct → call `__tribute_prompt`

use std::cell::Cell;
use std::collections::{HashMap, HashSet};

use trunk_ir::dialect::core::{self};
use trunk_ir::dialect::func::{self};
use trunk_ir::dialect::{adt, arith, cont};
use trunk_ir::rewrite::{PatternRewriter, RewritePattern};
use trunk_ir::{
    Attribute, Block, BlockArg, DialectOp, DialectType, IdVec, Location, Operation, Region, Symbol,
    Type, Value, ValueDef,
};

use crate::cont_util::{rebuild_op_with_remap, remap_value};

// ============================================================================
// Pattern
// ============================================================================

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

impl<'db> RewritePattern<'db> for LowerPushPromptPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        let Ok(push_prompt) = cont::PushPrompt::from_operation(db, *op) else {
            return false;
        };

        let location = op.location(db);
        let i32_ty = core::I32::new(db).as_type();
        let ptr_ty = core::Ptr::new(db).as_type();
        let tag = push_prompt.tag(db);

        // Get the body region (already recursively transformed by applicator)
        let body = push_prompt.body(db);

        // Compute live-ins for the body region, using rewriter for type lookup
        let live_ins = compute_live_ins(db, &body, &|db, v| rewriter.get_value_type(db, v));

        // Generate unique name
        let body_idx = self.body_counter.get();
        self.body_counter.set(body_idx + 1);
        let body_name = format!("__prompt_body_{body_idx}");

        // Generate outlined body function and add to module
        let outlined_func = generate_outlined_body(db, &body_name, &live_ins, body, location);
        rewriter.add_module_op(outlined_func);

        // Build the call site
        let mut ops = Vec::new();

        // Build env struct from live-in values
        let env_val = if live_ins.is_empty() {
            // No live-ins: pass null ptr
            let null = arith::r#const(db, location, ptr_ty, Attribute::IntBits(0));
            ops.push(null.as_operation());
            null.result(db)
        } else {
            // Create an env struct with live-in values
            // First, cast all values to ptr
            let mut field_ptrs = Vec::new();
            for (value, ty) in &live_ins {
                let remapped = rewriter.lookup_value(*value);
                if *ty != ptr_ty {
                    let cast = core::unrealized_conversion_cast(db, location, remapped, ptr_ty);
                    ops.push(cast.as_operation());
                    field_ptrs.push(cast.as_operation().result(db, 0));
                } else {
                    field_ptrs.push(remapped);
                }
            }

            // Build env struct type
            let env_struct_ty = build_env_struct_type(db, live_ins.len());

            // adt.struct_new to create the env
            let struct_new =
                adt::struct_new(db, location, field_ptrs, env_struct_ty, env_struct_ty);
            ops.push(struct_new.as_operation());

            // Cast struct to ptr for FFI
            let env_cast =
                core::unrealized_conversion_cast(db, location, struct_new.result(db), ptr_ty);
            ops.push(env_cast.as_operation());
            env_cast.as_operation().result(db, 0)
        };

        // %body_fn = func.constant @__prompt_body_N
        let body_fn = func::constant(db, location, ptr_ty, Symbol::from_dynamic(&body_name));
        ops.push(body_fn.as_operation());

        // %tag_val = arith.const <tag>
        let tag_val = arith::r#const(db, location, i32_ty, Attribute::IntBits(tag as u64));
        ops.push(tag_val.as_operation());

        // %result = func.call @__tribute_prompt(%tag_val, %body_fn, %env)
        let prompt_call = func::call(
            db,
            location,
            vec![tag_val.result(db), body_fn.result(db), env_val],
            ptr_ty,
            Symbol::new("__tribute_prompt"),
        );
        ops.push(prompt_call.as_operation());

        // Cast ptr result back to the original result type if needed
        let original_result_ty = op.results(db).first().copied();
        if let Some(result_ty) = original_result_ty
            && result_ty != ptr_ty
        {
            let cast =
                core::unrealized_conversion_cast(db, location, prompt_call.result(db), result_ty);
            ops.push(cast.as_operation());
        }

        let last = ops.pop().unwrap();
        for o in ops {
            rewriter.insert_op(o);
        }
        rewriter.replace_op(last);
        true
    }
}

// ============================================================================
// Live-in analysis
// ============================================================================

/// Compute live-in values for a region.
///
/// A value is a live-in if it is used (as an operand) inside the region
/// but defined outside (not a block arg or operation result within the region,
/// including nested sub-regions).
///
/// The `external_type_lookup` is used to resolve types for values defined
/// outside the region (e.g., block arguments from enclosing scopes).
pub(super) fn compute_live_ins<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
    external_type_lookup: &dyn Fn(&'db dyn salsa::Database, Value<'db>) -> Option<Type<'db>>,
) -> Vec<(Value<'db>, Type<'db>)> {
    let mut defined: HashSet<Value<'db>> = HashSet::new();
    let mut used: Vec<Value<'db>> = Vec::new();
    let mut seen_used: HashSet<Value<'db>> = HashSet::new();

    // Collect all defined values in the region, including nested regions
    collect_defined_in_region(db, region, &mut defined);

    // Collect all used values (including in nested regions)
    for block in region.blocks(db).iter() {
        for op in block.operations(db).iter() {
            collect_used_values(db, op, &mut used, &mut seen_used);
        }
    }

    // Live-ins = used but not defined
    let mut live_ins: Vec<(Value<'db>, Type<'db>)> = Vec::new();
    let mut live_in_set: HashSet<Value<'db>> = HashSet::new();

    for v in used {
        if !defined.contains(&v) && live_in_set.insert(v) {
            // Try external type lookup first, then fall back to definition-based
            let ty = external_type_lookup(db, v).unwrap_or_else(|| get_value_type(db, v));
            live_ins.push((v, ty));
        }
    }

    live_ins
}

/// Recursively collect all values defined in a region (block args + op results),
/// including those in nested sub-regions.
fn collect_defined_in_region<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
    defined: &mut HashSet<Value<'db>>,
) {
    for block in region.blocks(db).iter() {
        // Block arguments are defined
        for (i, _) in block.args(db).iter().enumerate() {
            defined.insert(Value::new(db, ValueDef::BlockArg(block.id(db)), i));
        }

        // Operation results are defined
        for op in block.operations(db).iter() {
            for (i, _) in op.results(db).iter().enumerate() {
                defined.insert(op.result(db, i));
            }

            // Recurse into nested regions
            for nested_region in op.regions(db).iter() {
                collect_defined_in_region(db, nested_region, defined);
            }
        }
    }
}

/// Recursively collect all values used as operands.
fn collect_used_values<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    used: &mut Vec<Value<'db>>,
    seen: &mut HashSet<Value<'db>>,
) {
    for v in op.operands(db).iter() {
        if seen.insert(*v) {
            used.push(*v);
        }
    }

    // Recurse into nested regions
    for region in op.regions(db).iter() {
        for block in region.blocks(db).iter() {
            for nested_op in block.operations(db).iter() {
                collect_used_values(db, nested_op, used, seen);
            }
        }
    }
}

/// Get the type of a value from its definition.
fn get_value_type<'db>(db: &'db dyn salsa::Database, value: Value<'db>) -> Type<'db> {
    match value.def(db) {
        ValueDef::OpResult(defining_op) => {
            let idx = value.index(db);
            defining_op
                .results(db)
                .get(idx)
                .copied()
                .unwrap_or_else(|| core::Ptr::new(db).as_type())
        }
        ValueDef::BlockArg(_block_id) => {
            // Block argument type — we can't easily look it up without traversing
            // the entire module. Use ptr as the default type for FFI.
            core::Ptr::new(db).as_type()
        }
    }
}

// ============================================================================
// Outlined body generation
// ============================================================================

/// Generate a `func.func` operation for an outlined prompt body.
fn generate_outlined_body<'db>(
    db: &'db dyn salsa::Database,
    name: &str,
    live_ins: &[(Value<'db>, Type<'db>)],
    body_region: Region<'db>,
    location: Location<'db>,
) -> Operation<'db> {
    let ptr_ty = core::Ptr::new(db).as_type();

    // Use Vec<Operation> + Block::new pattern for raw remapped ops
    let entry_block_id = trunk_ir::BlockId::fresh();
    let env_value = Value::new(db, ValueDef::BlockArg(entry_block_id), 0);

    let mut ops: Vec<Operation<'db>> = Vec::new();
    let mut value_remap: HashMap<Value<'db>, Value<'db>> = HashMap::new();

    // Extract live-in values from env struct
    if !live_ins.is_empty() {
        let env_struct_ty = build_env_struct_type(db, live_ins.len());

        // Cast ptr → struct type
        let env_cast = core::unrealized_conversion_cast(db, location, env_value, env_struct_ty);
        ops.push(env_cast.as_operation());
        let env_ref = env_cast.as_operation().result(db, 0);

        for (i, (orig_value, orig_ty)) in live_ins.iter().enumerate() {
            let field = adt::struct_get(db, location, env_ref, ptr_ty, env_struct_ty, i as u64);
            ops.push(field.as_operation());

            let extracted = if *orig_ty != ptr_ty {
                let cast =
                    core::unrealized_conversion_cast(db, location, field.result(db), *orig_ty);
                ops.push(cast.as_operation());
                cast.as_operation().result(db, 0)
            } else {
                field.result(db)
            };

            value_remap.insert(*orig_value, extracted);
        }
    }

    // Copy body operations with remapping
    if let Some(body_block) = body_region.blocks(db).first() {
        assert!(
            body_block.args(db).is_empty(),
            "cont.push_prompt body region must have no block arguments, \
             but spec.body_region has {} args",
            body_block.args(db).len(),
        );

        let mut last_result: Option<Value<'db>> = None;

        for op in body_block.operations(db).iter() {
            // Skip scf.yield — we'll add func.return instead
            if trunk_ir::dialect::scf::Yield::from_operation(db, *op).is_ok() {
                let yielded = op
                    .operands(db)
                    .first()
                    .copied()
                    .map(|val| remap_value(val, &value_remap));
                if let Some(result) = yielded {
                    last_result = Some(result);
                }
                continue;
            }

            // Skip func.return — we'll add our own
            if func::Return::from_operation(db, *op).is_ok() {
                let ret_val = op
                    .operands(db)
                    .first()
                    .copied()
                    .map(|val| remap_value(val, &value_remap));
                if let Some(result) = ret_val {
                    last_result = Some(result);
                }
                continue;
            }

            let remapped_op = rebuild_op_with_remap(db, op, &value_remap);
            if remapped_op != *op {
                for (i, _) in op.results(db).iter().enumerate() {
                    value_remap.insert(op.result(db, i), remapped_op.result(db, i));
                }
            }
            ops.push(remapped_op);

            if !remapped_op.results(db).is_empty() {
                last_result = Some(remapped_op.result(db, 0));
            }
        }

        // Add func.return with the result cast to ptr
        if let Some(result) = last_result {
            if get_value_type(db, result) != ptr_ty {
                let cast = core::unrealized_conversion_cast(db, location, result, ptr_ty);
                ops.push(cast.as_operation());
                ops.push(
                    func::r#return(db, location, Some(cast.as_operation().result(db, 0)))
                        .as_operation(),
                );
            } else {
                ops.push(func::r#return(db, location, Some(result)).as_operation());
            }
        } else {
            let null = arith::r#const(db, location, ptr_ty, Attribute::IntBits(0));
            ops.push(null.as_operation());
            ops.push(func::r#return(db, location, Some(null.result(db))).as_operation());
        }
    }

    let entry_block = Block::new(
        db,
        entry_block_id,
        location,
        IdVec::from(vec![BlockArg::of_type(db, ptr_ty)]),
        IdVec::from(ops),
    );
    let body = Region::new(db, location, IdVec::from(vec![entry_block]));

    let func_ty = core::Func::new(db, IdVec::from(vec![ptr_ty]), ptr_ty);
    func::func(db, location, Symbol::from_dynamic(name), *func_ty, body).as_operation()
}

// ============================================================================
// Helpers
// ============================================================================

/// Build an env struct type with N ptr fields.
fn build_env_struct_type<'db>(db: &'db dyn salsa::Database, field_count: usize) -> Type<'db> {
    let ptr_ty = core::Ptr::new(db).as_type();
    let fields: Vec<(Symbol, Type<'db>)> = (0..field_count)
        .map(|i| (Symbol::from_dynamic(&format!("f{i}")), ptr_ty))
        .collect();
    adt::struct_type(db, Symbol::new("__prompt_env"), fields)
}
