//! Push prompt body outlining for libmprompt backend.
//!
//! `mp_prompt` takes a function pointer, so the body of `cont.push_prompt`
//! must be outlined into a separate top-level `func.func`.
//!
//! Steps:
//! 1. Compute live-ins (values used inside but defined outside the body)
//! 2. Create an outlined function that unpacks live-ins from an env struct
//! 3. Replace the push_prompt with: build env struct → call `__tribute_prompt`

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use trunk_ir::dialect::core::{self};
use trunk_ir::dialect::func::{self};
use trunk_ir::dialect::{adt, arith, cont};
use trunk_ir::rewrite::{OpAdaptor, RewritePattern, RewriteResult};
use trunk_ir::{
    Attribute, Block, BlockArg, DialectOp, DialectType, IdVec, Location, Operation, Region, Symbol,
    Type, Value, ValueDef,
};

// ============================================================================
// Types
// ============================================================================

/// Specification for an outlined prompt body function.
pub(super) struct OutlinedBodySpec<'db> {
    /// Function name (e.g., `__prompt_body_0`)
    pub(super) name: String,
    /// Live-in values and their types
    pub(super) live_ins: Vec<(Value<'db>, Type<'db>)>,
    /// The body region (already transformed by the applicator)
    pub(super) body_region: Region<'db>,
    /// Source location
    pub(super) location: Location<'db>,
}

/// Shared storage for outlined body specs.
pub(super) type OutlinedBodies<'db> = Rc<RefCell<Vec<OutlinedBodySpec<'db>>>>;

/// Shared counter for generating unique outlined body names.
pub(super) type BodyCounter = Rc<RefCell<u32>>;

// ============================================================================
// Pattern
// ============================================================================

pub(crate) struct LowerPushPromptPattern<'db> {
    pub(super) outlined_bodies: OutlinedBodies<'db>,
    pub(super) body_counter: BodyCounter,
}

impl<'db> RewritePattern<'db> for LowerPushPromptPattern<'db> {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(push_prompt) = cont::PushPrompt::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let location = op.location(db);
        let i32_ty = core::I32::new(db).as_type();
        let ptr_ty = core::Ptr::new(db).as_type();
        let tag = push_prompt.tag(db);

        // Get the body region (already recursively transformed by applicator)
        let body = push_prompt.body(db);

        // Compute live-ins for the body region, using adaptor for type lookup
        let live_ins = compute_live_ins(db, &body, &|db, v| adaptor.get_value_type(db, v));

        // Generate unique name
        let body_idx = {
            let mut counter = self.body_counter.borrow_mut();
            let idx = *counter;
            *counter += 1;
            idx
        };
        let body_name = format!("__prompt_body_{body_idx}");

        // Store outlined body spec
        self.outlined_bodies.borrow_mut().push(OutlinedBodySpec {
            name: body_name.clone(),
            live_ins: live_ins.clone(),
            body_region: body,
            location,
        });

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
                if *ty != ptr_ty {
                    let cast = core::unrealized_conversion_cast(db, location, *value, ptr_ty);
                    ops.push(cast.as_operation());
                    field_ptrs.push(cast.as_operation().result(db, 0));
                } else {
                    field_ptrs.push(*value);
                }
            }

            // Build env struct type
            let env_struct_ty = build_env_struct_type(db, live_ins.len());

            // adt.struct_new to create the env
            let struct_new = adt::struct_new(db, location, field_ptrs, env_struct_ty, ptr_ty);
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

        RewriteResult::expand(ops)
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
fn compute_live_ins<'db>(
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

/// Generate a `func.func` operation from an outlined body spec.
pub(super) fn generate_outlined_body<'db>(
    db: &'db dyn salsa::Database,
    spec: &OutlinedBodySpec<'db>,
) -> Operation<'db> {
    let location = spec.location;
    let ptr_ty = core::Ptr::new(db).as_type();

    // Use Vec<Operation> + Block::new pattern for raw remapped ops
    let entry_block_id = trunk_ir::BlockId::fresh();
    let env_value = Value::new(db, ValueDef::BlockArg(entry_block_id), 0);

    let mut ops: Vec<Operation<'db>> = Vec::new();
    let mut value_remap: HashMap<Value<'db>, Value<'db>> = HashMap::new();

    // Extract live-in values from env struct
    if !spec.live_ins.is_empty() {
        let env_struct_ty = build_env_struct_type(db, spec.live_ins.len());

        // Cast ptr → struct type
        let env_cast = core::unrealized_conversion_cast(db, location, env_value, env_struct_ty);
        ops.push(env_cast.as_operation());
        let env_ref = env_cast.as_operation().result(db, 0);

        for (i, (orig_value, orig_ty)) in spec.live_ins.iter().enumerate() {
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
    let body_region = &spec.body_region;
    if let Some(body_block) = body_region.blocks(db).first() {
        // Remap body block args too if they exist
        for (i, _) in body_block.args(db).iter().enumerate() {
            let orig = Value::new(db, ValueDef::BlockArg(body_block.id(db)), i);
            value_remap.entry(orig).or_insert(env_value);
        }

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
    func::func(
        db,
        location,
        Symbol::from_dynamic(&spec.name),
        *func_ty,
        body,
    )
    .as_operation()
}

/// Add outlined body functions to the module's first block.
pub(super) fn add_outlined_bodies<'db>(
    db: &'db dyn salsa::Database,
    module: trunk_ir::dialect::core::Module<'db>,
    specs: &[OutlinedBodySpec<'db>],
) -> trunk_ir::dialect::core::Module<'db> {
    if specs.is_empty() {
        return module;
    }

    let body = module.body(db);
    let mut blocks: Vec<Block<'db>> = body.blocks(db).iter().copied().collect();

    if let Some(block) = blocks.first_mut() {
        let mut ops: Vec<Operation<'db>> = block.operations(db).iter().copied().collect();

        for spec in specs {
            ops.push(generate_outlined_body(db, spec));
        }

        *block = Block::new(
            db,
            block.id(db),
            block.location(db),
            block.args(db).clone(),
            IdVec::from(ops),
        );
    }

    let new_body = Region::new(db, body.location(db), IdVec::from(blocks));
    trunk_ir::dialect::core::Module::create(db, module.location(db), module.name(db), new_body)
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

fn remap_value<'db>(v: Value<'db>, value_remap: &HashMap<Value<'db>, Value<'db>>) -> Value<'db> {
    let mut current = v;
    let mut steps = 0u32;
    while let Some(&remapped) = value_remap.get(&current) {
        current = remapped;
        steps += 1;
        assert!(
            steps < 1000,
            "cycle detected in value_remap after {steps} steps"
        );
    }
    current
}

fn rebuild_op_with_remap<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    value_remap: &HashMap<Value<'db>, Value<'db>>,
) -> Operation<'db> {
    let operands = op.operands(db);
    let remapped_operands: IdVec<Value<'db>> = operands
        .iter()
        .map(|v| remap_value(*v, value_remap))
        .collect::<Vec<_>>()
        .into();

    let regions = op.regions(db);
    let remapped_regions: IdVec<Region<'db>> = regions
        .iter()
        .map(|r| rebuild_region_with_remap(db, r, value_remap))
        .collect::<Vec<_>>()
        .into();

    if remapped_operands == *operands && remapped_regions == *regions {
        return *op;
    }

    Operation::new(
        db,
        op.location(db),
        op.dialect(db),
        op.name(db),
        remapped_operands,
        op.results(db).clone(),
        op.attributes(db).clone(),
        remapped_regions,
        op.successors(db).clone(),
    )
}

fn rebuild_region_with_remap<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
    value_remap: &HashMap<Value<'db>, Value<'db>>,
) -> Region<'db> {
    let new_blocks: Vec<Block<'db>> = region
        .blocks(db)
        .iter()
        .map(|block| {
            let new_ops: Vec<Operation<'db>> = block
                .operations(db)
                .iter()
                .map(|op| rebuild_op_with_remap(db, op, value_remap))
                .collect();
            Block::new(
                db,
                block.id(db),
                block.location(db),
                block.args(db).clone(),
                IdVec::from(new_ops),
            )
        })
        .collect();
    Region::new(db, region.location(db), IdVec::from(new_blocks))
}
