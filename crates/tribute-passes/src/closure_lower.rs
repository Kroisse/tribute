//! Lower closure operations in indirect calls.
//!
//! This pass transforms `func.call_indirect` operations when the callee
//! is a closure:
//!
//! Before:
//! ```text
//! %closure = closure.new @lifted_func, %env
//! %result = func.call_indirect %closure, %args...
//! ```
//!
//! After:
//! ```text
//! %closure = closure.new @lifted_func, %env
//! %funcref = closure.func %closure
//! %env = closure.env %closure
//! %result = func.call_indirect %funcref, %env, %args...
//! ```
//!
//! Uses `RewritePattern` + `PatternApplicator` for declarative transformation.

use std::collections::HashSet;

use tribute_ir::dialect::{closure, tribute_rt};
use trunk_ir::dialect::adt;
use trunk_ir::dialect::{cont, core, func, wasm};
use trunk_ir::rewrite::{
    ConversionTarget, OpAdaptor, PatternApplicator, RewritePattern, RewriteResult, TypeConverter,
};
use trunk_ir::{
    Attribute, Block, DialectOp, DialectType, IdVec, Operation, Region, Symbol, Type, Value,
    ValueDef,
};

use crate::evidence::collect_effectful_functions;

/// Create the unified closure struct type: `{ table_idx: i32, env: anyref }`.
///
/// All closures share the same struct type regardless of their specific function/env types.
/// This ensures consistent representation across the lowering pipeline.
fn closure_struct_type(db: &dyn salsa::Database) -> Type<'_> {
    let i32_ty = core::I32::new(db).as_type();
    let anyref_ty = tribute_rt::Any::new(db).as_type();
    adt::struct_type(
        db,
        Symbol::new("_closure"),
        vec![
            (Symbol::new("table_idx"), i32_ty),
            (Symbol::new("env"), anyref_ty),
        ],
    )
}

/// Lower closure operations in the module.
///
/// This pass has two phases:
///
/// Phase 1 (PatternApplicator):
/// 1. UpdateFuncSignaturePattern - updates function signatures: core.func params → closure.closure
/// 2. LowerClosureCallPattern - expands call_indirect to use closure.func/closure.env
/// 3. LowerClosureNewPattern - expands closure.new to func.constant + adt.struct_new
/// 4. LowerClosureFuncPattern - extracts i32 table index from struct (field 0)
/// 5. LowerClosureEnvPattern - extracts env from struct (field 1)
///
/// Phase 2 (Post-processing):
/// - Transform ALL closure calls to pass evidence from the enclosing function
#[salsa::tracked]
pub fn lower_closures<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> core::Module<'db> {
    // Collect effectful functions BEFORE lowering (while closure types are intact)
    let effectful_fns = collect_effectful_functions(db, &module);

    // Collect ALL closure calls (not just effectful) because all lifted lambdas
    // now have evidence parameters and need evidence passed to them.
    let all_closure_calls = collect_all_closure_calls(db, &module);

    let converter = TypeConverter::new()
        .add_conversion(|db, ty| {
            tribute_rt::Int::from_type(db, ty).map(|_| core::I32::new(db).as_type())
        })
        .add_conversion(|db, ty| {
            tribute_rt::Nat::from_type(db, ty).map(|_| core::I32::new(db).as_type())
        })
        .add_conversion(|db, ty| {
            tribute_rt::Bool::from_type(db, ty).map(|_| core::I::<1>::new(db).as_type())
        })
        .add_conversion(|db, ty| {
            tribute_rt::Float::from_type(db, ty).map(|_| core::F64::new(db).as_type())
        });

    // No specific conversion target - closure lowering is an optimization pass
    let target = ConversionTarget::new();
    let applicator = PatternApplicator::new(converter)
        // First, update function signatures: core.func params → closure.closure
        .add_pattern(UpdateFuncSignaturePattern)
        .add_pattern(LowerClosureCallPattern)
        .add_pattern(LowerClosureNewPattern)
        .add_pattern(LowerClosureFuncPattern)
        .add_pattern(LowerClosureEnvPattern);
    let module = applicator.apply_partial(db, module, target).module;

    // Phase 2: Transform closure calls to pass evidence
    // This is done after pattern application because we need to know which
    // calls are closure calls, and we need access to enclosing function context.
    transform_closure_calls_with_evidence(db, module, &effectful_fns, &all_closure_calls)
}

/// Collect call_indirect operations that call ANY closure (for evidence passing).
/// Since all lifted lambdas now have evidence parameters, we need to pass evidence
/// to ALL closure calls, not just effectful ones.
/// Returns a set of locations where closure calls occur.
fn collect_all_closure_calls<'db>(
    db: &'db dyn salsa::Database,
    module: &core::Module<'db>,
) -> HashSet<(usize, usize)> {
    let mut closure_calls = HashSet::new();
    let body = module.body(db);

    for block in body.blocks(db).iter() {
        for op in block.operations(db).iter() {
            // Process func.func operations
            if let Ok(func_op) = func::Func::from_operation(db, *op) {
                // Get function parameter types for BlockArg lookup
                let func_ty = func_op.r#type(db);
                let param_types = core::Func::from_type(db, func_ty)
                    .map(|ft| ft.params(db).clone())
                    .unwrap_or_default();

                collect_closure_calls_in_func(
                    db,
                    &func_op.body(db),
                    &param_types,
                    &mut closure_calls,
                );
            }
        }
    }

    closure_calls
}

fn collect_closure_calls_in_func<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
    param_types: &IdVec<Type<'db>>,
    closure_calls: &mut HashSet<(usize, usize)>,
) {
    for block in region.blocks(db).iter() {
        for op in block.operations(db).iter() {
            collect_closure_calls_in_op(db, op, param_types, closure_calls);
        }
    }
}

fn collect_closure_calls_in_op<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    param_types: &IdVec<Type<'db>>,
    closure_calls: &mut HashSet<(usize, usize)>,
) {
    // Check if this is a call_indirect with a closure callee
    if func::CallIndirect::from_operation(db, *op).is_ok() {
        let operands = op.operands(db);
        if let Some(&callee) = operands.first() {
            let is_closure = is_any_closure_value(db, callee, param_types);
            tracing::debug!(
                "collect_closure_calls: call_indirect at {:?}, callee def={:?}, is_closure={}",
                op.location(db).span,
                callee.def(db),
                is_closure
            );
            if is_closure {
                let loc = op.location(db);
                closure_calls.insert((loc.span.start, loc.span.end));
            }
        }
    }

    // Recurse into regions
    for region in op.regions(db).iter() {
        for block in region.blocks(db).iter() {
            for nested_op in block.operations(db).iter() {
                collect_closure_calls_in_op(db, nested_op, param_types, closure_calls);
            }
        }
    }
}

/// Check if a value is any closure (not just effectful).
/// Used for evidence passing since all lifted lambdas have evidence parameters.
fn is_any_closure_value<'db>(
    db: &'db dyn salsa::Database,
    value: Value<'db>,
    param_types: &IdVec<Type<'db>>,
) -> bool {
    // Get the type of the value
    let ty = match value.def(db) {
        ValueDef::OpResult(op) => {
            // Direct check: result of closure.new
            if closure::New::from_operation(db, op).is_ok() {
                tracing::debug!("is_any_closure_value: result of closure.new → true");
                return true;
            }
            // Type check from result
            op.results(db).get(value.index(db)).copied()
        }
        ValueDef::BlockArg(_) => {
            // For block args, look up type from param_types using value's index
            let idx = value.index(db);
            let ty = param_types.get(idx).copied();
            tracing::debug!(
                "is_any_closure_value: block arg idx={}, param_types.len()={}, ty={:?}",
                idx,
                param_types.len(),
                ty.map(|t| format!("{:?}", t))
            );
            ty
        }
    };

    let Some(ty) = ty else {
        tracing::debug!("is_any_closure_value: no type found");
        return false;
    };

    // Check if it's a closure type
    if closure::Closure::from_type(db, ty).is_some() {
        tracing::debug!("is_any_closure_value: closure.closure type → true");
        return true;
    }

    // Check if it's a core.func type (function parameter that could receive closures)
    if core::Func::from_type(db, ty).is_some() {
        tracing::debug!("is_any_closure_value: core.func type → true");
        return true;
    }

    // Check if it's a cont.continuation type (continuations are also closures)
    if cont::Continuation::from_type(db, ty).is_some() {
        tracing::debug!("is_any_closure_value: cont.continuation type → true");
        return true;
    }

    tracing::debug!("is_any_closure_value: not a closure type → false");
    false
}

/// Transform ALL closure calls to pass evidence.
///
/// After pattern application, closure calls look like:
/// ```text
/// %table_idx = adt.struct_get %closure, 0
/// %env = adt.struct_get %closure, 1
/// %result = func.call_indirect %table_idx, %env, %args...
/// ```
///
/// Since all lifted lambdas have evidence as their first parameter, we transform to:
/// ```text
/// %table_idx = adt.struct_get %closure, 0
/// %env = adt.struct_get %closure, 1
/// %result = func.call_indirect %table_idx, %evidence, %env, %args...
/// ```
fn transform_closure_calls_with_evidence<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
    effectful_fns: &HashSet<Symbol>,
    closure_calls: &HashSet<(usize, usize)>,
) -> core::Module<'db> {
    if closure_calls.is_empty() {
        return module;
    }

    let body = module.body(db);
    let new_blocks: IdVec<Block<'db>> = body
        .blocks(db)
        .iter()
        .map(|block| {
            let new_ops: IdVec<Operation<'db>> = block
                .operations(db)
                .iter()
                .map(|op| transform_func_for_closure_evidence(db, op, effectful_fns, closure_calls))
                .collect();
            Block::new(
                db,
                block.id(db),
                block.location(db),
                block.args(db).clone(),
                new_ops,
            )
        })
        .collect();

    let new_body = Region::new(db, body.location(db), new_blocks);
    core::Module::create(db, module.location(db), module.name(db), new_body)
}

/// Transform a func.func operation to pass evidence to closure calls.
///
/// For effectful functions (in effectful_fns), evidence is taken from the first block argument.
/// For non-effectful functions (e.g., lambdas in handler bodies), null evidence is created.
fn transform_func_for_closure_evidence<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    effectful_fns: &HashSet<Symbol>,
    closure_calls: &HashSet<(usize, usize)>,
) -> Operation<'db> {
    // Only process func.func operations
    let Ok(func_op) = func::Func::from_operation(db, *op) else {
        return *op;
    };

    let func_name = func_op.sym_name(db);
    let is_effectful = effectful_fns.contains(&func_name);

    let body = func_op.body(db);
    let blocks = body.blocks(db);
    let Some(entry_block) = blocks.first() else {
        return *op;
    };

    // For effectful functions, get evidence from first block argument
    // For non-effectful functions, we'll create null evidence later if needed
    let evidence_from_param = if is_effectful && !entry_block.args(db).is_empty() {
        Some(entry_block.arg(db, 0))
    } else {
        None
    };

    // Transform closure calls in the body
    let mut changed = false;
    let location = op.location(db);

    let new_blocks: IdVec<Block<'db>> = blocks
        .iter()
        .map(|block| {
            let (new_block, block_changed, _needs_null_ev) =
                transform_closure_calls_in_block_with_null(
                    db,
                    block,
                    evidence_from_param,
                    closure_calls,
                    location,
                );
            if block_changed {
                changed = true;
            }
            new_block
        })
        .collect();

    if !changed {
        return *op;
    }

    // Rebuild function with transformed body
    let new_body = Region::new(db, location, new_blocks);
    let func_ty = func_op.r#type(db);

    func::func(db, location, func_name, func_ty, new_body).as_operation()
}

/// Transform closure calls in a block, creating null evidence if needed for non-effectful functions.
fn transform_closure_calls_in_block_with_null<'db>(
    db: &'db dyn salsa::Database,
    block: &Block<'db>,
    evidence_from_param: Option<Value<'db>>,
    closure_calls: &HashSet<(usize, usize)>,
    func_location: trunk_ir::Location<'db>,
) -> (Block<'db>, bool, bool) {
    use std::collections::HashMap;

    let mut new_ops = Vec::new();
    let mut changed = false;
    let mut value_map: HashMap<Value<'db>, Value<'db>> = HashMap::new();
    let mut needs_null_ev = false;
    let mut null_ev_value: Option<Value<'db>> = None;

    // Helper to get or create null evidence
    let get_evidence = |new_ops: &mut Vec<Operation<'db>>,
                        null_ev_value: &mut Option<Value<'db>>,
                        needs_null_ev: &mut bool|
     -> Value<'db> {
        if let Some(ev) = evidence_from_param {
            ev
        } else if let Some(ev) = *null_ev_value {
            ev
        } else {
            // Create null evidence
            // Use anyref directly for wasm compatibility - ability.evidence_ptr
            // may not be converted properly by the wasm backend
            let anyref_ty = wasm::Anyref::new(db).as_type();
            let null_ev_op = adt::ref_null(db, func_location, anyref_ty, anyref_ty);
            let ev = null_ev_op.as_operation().result(db, 0);
            new_ops.insert(0, null_ev_op.as_operation()); // Prepend to block
            *null_ev_value = Some(ev);
            *needs_null_ev = true;
            ev
        }
    };

    for op in block.operations(db).iter() {
        // First, remap operands using the value map
        let remapped_operands: Vec<Value<'db>> = op
            .operands(db)
            .iter()
            .map(|v| *value_map.get(v).unwrap_or(v))
            .collect();

        // Check if this is a call_indirect that was a closure call
        if func::CallIndirect::from_operation(db, *op).is_ok() {
            let loc = op.location(db);
            let loc_key = (loc.span.start, loc.span.end);

            tracing::debug!(
                "transform: checking call_indirect at {:?}, closure_calls contains={}, closure_calls={:?}",
                loc_key,
                closure_calls.contains(&loc_key),
                closure_calls
            );

            if closure_calls.contains(&loc_key) {
                // This was a closure call - add evidence
                let evidence = get_evidence(&mut new_ops, &mut null_ev_value, &mut needs_null_ev);
                tracing::debug!(
                    "transform: adding evidence {:?} to call_indirect at {:?}, evidence_from_param={:?}",
                    evidence,
                    loc_key,
                    evidence_from_param
                );

                let result_ty = op
                    .results(db)
                    .first()
                    .copied()
                    .unwrap_or_else(|| core::Nil::new(db).as_type());

                // First operand is table_idx, rest are env + args
                let table_idx = remapped_operands[0];
                let rest_args: Vec<_> = remapped_operands[1..].to_vec();

                // Build new args: [evidence, env, args...]
                let mut new_args = vec![evidence];
                new_args.extend(rest_args);

                let new_call = func::call_indirect(db, loc, table_idx, new_args, result_ty);
                let new_call_op = new_call.as_operation();

                // Map old result to new result
                if !op.results(db).is_empty() {
                    let old_result = op.result(db, 0);
                    let new_result = new_call_op.result(db, 0);
                    value_map.insert(old_result, new_result);
                }

                new_ops.push(new_call_op);
                changed = true;
                continue;
            }
        }

        // Recursively transform nested regions
        let regions = op.regions(db);
        if !regions.is_empty() {
            let mut region_changed = false;
            let new_regions: IdVec<Region<'db>> = regions
                .iter()
                .map(|region| {
                    let (new_region, r_changed, _) = transform_closure_calls_in_region_with_null(
                        db,
                        region,
                        evidence_from_param,
                        closure_calls,
                        func_location,
                    );
                    if r_changed {
                        region_changed = true;
                    }
                    new_region
                })
                .collect();

            if region_changed {
                changed = true;
                let new_op = op
                    .modify(db)
                    .operands(IdVec::from(remapped_operands))
                    .regions(new_regions)
                    .build();
                for i in 0..op.results(db).len() {
                    let old_result = op.result(db, i);
                    let new_result = new_op.result(db, i);
                    value_map.insert(old_result, new_result);
                }
                new_ops.push(new_op);
                continue;
            }
        }

        // If operands were remapped, rebuild the operation
        let operands_changed = op
            .operands(db)
            .iter()
            .zip(remapped_operands.iter())
            .any(|(old, new)| old != new);

        if operands_changed {
            let new_op = op
                .modify(db)
                .operands(IdVec::from(remapped_operands))
                .build();
            for i in 0..op.results(db).len() {
                let old_result = op.result(db, i);
                let new_result = new_op.result(db, i);
                value_map.insert(old_result, new_result);
            }
            new_ops.push(new_op);
        } else {
            new_ops.push(*op);
        }
    }

    let new_block = Block::new(
        db,
        block.id(db),
        block.location(db),
        block.args(db).clone(),
        new_ops.into_iter().collect(),
    );

    (new_block, changed, needs_null_ev)
}

/// Transform closure calls in a region with null evidence support.
fn transform_closure_calls_in_region_with_null<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
    evidence_from_param: Option<Value<'db>>,
    closure_calls: &HashSet<(usize, usize)>,
    func_location: trunk_ir::Location<'db>,
) -> (Region<'db>, bool, bool) {
    let mut changed = false;
    let mut needs_null_ev = false;
    let new_blocks: IdVec<Block<'db>> = region
        .blocks(db)
        .iter()
        .map(|block| {
            let (new_block, block_changed, block_needs_null) =
                transform_closure_calls_in_block_with_null(
                    db,
                    block,
                    evidence_from_param,
                    closure_calls,
                    func_location,
                );
            if block_changed {
                changed = true;
            }
            if block_needs_null {
                needs_null_ev = true;
            }
            new_block
        })
        .collect();

    let new_region = Region::new(db, region.location(db), new_blocks);
    (new_region, changed, needs_null_ev)
}

/// Pattern: Update function signatures to convert `core.func` parameters to `closure.closure`.
///
/// This ensures that function parameters with function types accept closure structs,
/// since in Tribute all function values are represented as closures.
struct UpdateFuncSignaturePattern;

impl<'db> RewritePattern<'db> for UpdateFuncSignaturePattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        // Match: func.func
        let func_op = match func::Func::from_operation(db, *op) {
            Ok(f) => f,
            Err(_) => return RewriteResult::Unchanged,
        };

        let func_ty = func_op.r#type(db);
        let Some(func_type) = core::Func::from_type(db, func_ty) else {
            return RewriteResult::Unchanged;
        };

        let params = func_type.params(db);
        let result = func_type.result(db);
        let effect = func_type.effect(db);

        // Check if any parameter has core.func type (needs conversion to closure)
        let mut needs_update = false;
        let mut new_params: Vec<Type<'db>> = Vec::with_capacity(params.len());

        for param_ty in params.iter() {
            if core::Func::from_type(db, *param_ty).is_some() {
                // Convert core.func to closure.closure
                new_params.push(closure::Closure::new(db, *param_ty).as_type());
                needs_update = true;
            } else {
                new_params.push(*param_ty);
            }
        }

        if !needs_update {
            return RewriteResult::Unchanged;
        }

        // Create new function type with updated parameters
        let new_func_ty = if let Some(eff) = effect {
            core::Func::with_effect(db, new_params.into_iter().collect(), result, Some(eff))
        } else {
            core::Func::new(db, new_params.into_iter().collect(), result)
        };

        // Rebuild the function with the new type
        let new_op = op
            .modify(db)
            .attr("type", Attribute::Type(new_func_ty.as_type()))
            .build();

        RewriteResult::Replace(new_op)
    }
}

/// Pattern: Lower `closure.new` to `func.constant` + `adt.struct_new`.
struct LowerClosureNewPattern;

impl<'db> RewritePattern<'db> for LowerClosureNewPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        // Match: closure.new
        let closure_new = match closure::New::from_operation(db, *op) {
            Ok(c) => c,
            Err(_) => return RewriteResult::Unchanged,
        };

        let location = op.location(db);
        let func_ref = closure_new.func_ref(db);

        // Get the closure type to extract the inner function type
        let closure_result_ty = op
            .results(db)
            .first()
            .copied()
            .expect("closure.new should have a result");

        // Extract function type from closure.closure type
        let func_ty = closure::Closure::from_type(db, closure_result_ty)
            .map(|ct| ct.func_type(db))
            .unwrap_or_else(|| *core::Nil::new(db));

        // Get env from adaptor (remapped value)
        let env = adaptor
            .operand(0)
            .expect("closure.new requires env operand");

        // Generate: %funcref = func.constant @func_ref : func_type
        // func_ref is already a Symbol, use it directly
        let constant_op = func::constant(db, location, func_ty, func_ref);
        let funcref = constant_op.as_operation().result(db, 0);

        // Create closure struct type: adt.struct with (i32 table_idx, anyref env) fields
        let closure_struct_ty = closure_struct_type(db);

        // Generate: %closure = adt.struct_new(%funcref, %env) : closure_struct_type
        let struct_new_op = adt::struct_new(
            db,
            location,
            vec![funcref, env],
            closure_struct_ty,
            closure_struct_ty,
        );

        RewriteResult::Expand(vec![
            constant_op.as_operation(),
            struct_new_op.as_operation(),
        ])
    }
}

/// Pattern: Lower `func.call_indirect` on closure values.
///
/// Matches calls where the callee is a closure and expands to:
/// 1. Extract funcref via `closure.func`
/// 2. Extract env via `closure.env`
/// 3. Call with env as first argument
///
/// A value is considered a closure if:
/// - Its type is `closure.closure` (direct match), OR
/// - It's a result of `closure.new` operation, OR
/// - It's a block arg with a function-like type (heuristic for parameters)
struct LowerClosureCallPattern;

impl<'db> RewritePattern<'db> for LowerClosureCallPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        // Match: func.call_indirect
        if func::CallIndirect::from_operation(db, *op).is_err() {
            return RewriteResult::Unchanged;
        }

        // Get callee from adaptor (remapped value)
        let callee = adaptor
            .operand(0)
            .expect("call_indirect requires callee operand");

        // Check if callee is a closure using the adaptor's type information.
        // This is more accurate than the heuristic in is_closure_value because
        // the adaptor has access to converted types including block arg types.
        //
        // A value is considered a closure if:
        // 1. The type is explicitly closure.closure, OR
        // 2. The value is a result of closure.new operation, OR
        // 3. The value is a block arg with core.func type (function parameter that receives closures)
        //
        // NOTE: We do NOT treat core.func OP RESULTS as closures (except closure.new).
        // After this pass runs, closure.func generates struct_get with result type core.func,
        // and if we treated that as a closure, we'd get infinite expansion.
        //
        // Block args with core.func type ARE treated as closures because in Tribute,
        // function parameters with function types can receive closures at runtime.
        let callee_ty_opt = adaptor.operand_type(0);

        let (callee_is_closure, _func_ty) = if let Some(callee_ty) = callee_ty_opt {
            // Type is available - check if it's closure.closure
            // Check if it's an adt.struct with name "_closure" (already lowered closure.new)
            let is_closure_struct = adt::is_struct_type(db, callee_ty)
                && callee_ty.get_attr(db, adt::ATTR_NAME()).is_some_and(
                    |attr| matches!(attr, Attribute::Symbol(s) if *s == Symbol::new("_closure")),
                );

            if let Some(closure_ty) = closure::Closure::from_type(db, callee_ty) {
                // closure.closure → extract inner func_type
                (true, closure_ty.func_type(db))
            } else if is_closure_struct {
                // adt.struct with name "_closure" → already lowered closure
                // The function type needs to be extracted from the func.constant that was
                // stored in the struct's first field during LowerClosureNewPattern.
                // Try to extract func_type from the struct_new operation's first operand
                // which should be a func.constant with the actual function type
                if let Some(func_ty) = get_func_type_from_closure_struct(db, callee) {
                    (true, func_ty)
                } else {
                    // Failed to extract function type - skip closure handling
                    // to preserve the original type annotation
                    tracing::warn!(
                        "Failed to extract function type from closure struct, skipping closure handling"
                    );
                    (false, *core::Nil::new(db))
                }
            } else if core::Func::from_type(db, callee_ty).is_some() {
                // core.func type - check if callee is a block arg (function parameter)
                // Block args with core.func type should be treated as closures since
                // they can receive closure values at runtime.
                // Op results with core.func type should NOT be treated as closures
                // (they're raw funcrefs, e.g., from closure.func).
                let is_block_arg = matches!(callee.def(db), ValueDef::BlockArg(_));
                if is_block_arg {
                    // Block arg with function type - treat as closure
                    // The func_type is the callee_ty itself since it's already core.func
                    (true, callee_ty)
                } else {
                    // Op result with core.func - NOT a closure (avoid infinite expansion)
                    (false, *core::Nil::new(db))
                }
            } else {
                (false, *core::Nil::new(db))
            }
        } else {
            // No type info - fall back to structural check (result of closure.new)
            let is_closure = is_closure_value(db, callee);
            let func_ty = if is_closure {
                get_closure_func_type(db, callee)
            } else {
                *core::Nil::new(db)
            };
            (is_closure, func_ty)
        };

        if !callee_is_closure {
            return RewriteResult::Unchanged;
        }

        // Get location and other info
        let location = op.location(db);
        // Get args from adaptor (remapped values), skipping the callee (index 0)
        let args: Vec<_> = adaptor.operands().iter().skip(1).copied().collect();
        let result_ty = op
            .results(db)
            .first()
            .copied()
            .expect("call_indirect should have a result");

        // Get env type from the closure.new operation if available.
        // Use anyref for env since closure struct stores env as anyref.
        let anyref_ty = tribute_rt::Any::new(db).as_type();
        let i32_ty = core::I32::new(db).as_type();

        // Generate: %table_idx = closure.func %closure
        // Result type is i32 (function table index) for call_indirect via function table.
        let table_idx_op = closure::func(db, location, callee, i32_ty);
        let table_idx = table_idx_op.as_operation().result(db, 0);

        // Generate: %env = closure.env %closure
        let env_op = closure::env(db, location, callee, anyref_ty);
        let env = env_op.as_operation().result(db, 0);

        // Generate: %result = func.call_indirect %table_idx, %env, %args...
        // First operand is i32 table index, followed by env and other arguments.
        let mut new_args: Vec<Value<'db>> = vec![env];
        new_args.extend(args.iter().copied());

        let new_call = func::call_indirect(db, location, table_idx, new_args, result_ty);

        RewriteResult::Expand(vec![
            table_idx_op.as_operation(),
            env_op.as_operation(),
            new_call.as_operation(),
        ])
    }
}

/// Pattern: Lower `closure.func` to `adt.struct_get` field 0.
///
/// After closure.new is lowered to an adt.struct with (i32, anyref),
/// closure.func extracts the function table index (first field).
/// Returns i32 (function table index).
struct LowerClosureFuncPattern;

impl<'db> RewritePattern<'db> for LowerClosureFuncPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        // Match: closure.func
        if closure::Func::from_operation(db, *op).is_err() {
            return RewriteResult::Unchanged;
        }

        let location = op.location(db);
        // Get closure from adaptor (remapped value)
        let closure_value = adaptor
            .operand(0)
            .expect("closure.func requires closure operand");

        // The result type is now i32 (function table index), not funcref
        let i32_ty = core::I32::new(db).as_type();
        let struct_ty = closure_struct_type(db);

        // Generate: %table_idx = adt.struct_get %closure, 0
        // Parameter order: (db, location, operand, result_type, struct_type, field_idx)
        let get_op = adt::struct_get(
            db,
            location,
            closure_value,
            i32_ty, // Result is i32 (table index), not funcref
            struct_ty,
            0,
        );

        RewriteResult::Replace(get_op.as_operation())
    }
}

/// Pattern: Lower `closure.env` to `adt.struct_get` field 1.
///
/// After closure.new is lowered to an adt.struct with (i32, anyref),
/// closure.env extracts the environment (second field).
struct LowerClosureEnvPattern;

impl<'db> RewritePattern<'db> for LowerClosureEnvPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        // Match: closure.env
        if closure::Env::from_operation(db, *op).is_err() {
            return RewriteResult::Unchanged;
        }

        let location = op.location(db);
        // Get closure from adaptor (remapped value)
        let closure_value = adaptor
            .operand(0)
            .expect("closure.env requires closure operand");

        // Get the result type (env type)
        let result_ty = op
            .results(db)
            .first()
            .copied()
            .expect("closure.env should have a result");

        let struct_ty = closure_struct_type(db);

        // Generate: %env = adt.struct_get %closure, 1
        // Parameter order: (db, location, operand, result_type, struct_type, field_idx)
        let get_op = adt::struct_get(db, location, closure_value, result_ty, struct_ty, 1);

        RewriteResult::Replace(get_op.as_operation())
    }
}

/// Check if a value is a closure.
fn is_closure_value<'db>(db: &'db dyn salsa::Database, value: Value<'db>) -> bool {
    match value.def(db) {
        ValueDef::OpResult(op) => {
            // Direct check: result of closure.new
            if closure::New::from_operation(db, op).is_ok() {
                return true;
            }
            // Type check: has closure.closure type
            op.results(db)
                .get(value.index(db))
                .copied()
                .is_some_and(|ty| closure::Closure::from_type(db, ty).is_some())
        }
        ValueDef::BlockArg(_) => {
            // For block args, we cannot determine if they're closures without
            // additional type information. The caller (LowerClosureCallPattern)
            // should use the OpAdaptor to get the actual type.
            //
            // We return false here to be conservative. This means that block args
            // without explicit closure.closure type in the adaptor won't be
            // treated as closures - which is correct for funcref parameters.
            //
            // NOTE: If this breaks existing code that relies on the heuristic,
            // the fix is to ensure lambda_lift properly updates function signatures
            // to use closure.closure types for closure parameters.
            false
        }
    }
}

/// Get the function type from a closure value.
fn get_closure_func_type<'db>(db: &'db dyn salsa::Database, value: Value<'db>) -> Type<'db> {
    // Try to get closure.closure type and extract func_type
    if let Some(ty) = get_value_type(db, value) {
        // closure.closure → extract inner func_type
        if let Some(closure_ty) = closure::Closure::from_type(db, ty) {
            return closure_ty.func_type(db);
        }
        // core.func → the type itself is the function type
        if core::Func::from_type(db, ty).is_some() {
            return ty;
        }
        return ty;
    }

    // Fallback for block args: infer from closure.new if available
    if let ValueDef::OpResult(op) = value.def(db)
        && closure::New::from_operation(db, op).is_ok()
        && let Some(ty) = op.results(db).first().copied()
    {
        return closure::Closure::from_type(db, ty)
            .map(|ct| ct.func_type(db))
            .unwrap_or(ty);
    }

    // Last resort: use a placeholder type
    *core::Nil::new(db)
}

/// Get the type of a value from its definition site.
fn get_value_type<'db>(db: &'db dyn salsa::Database, value: Value<'db>) -> Option<Type<'db>> {
    match value.def(db) {
        ValueDef::OpResult(op) => op.results(db).get(value.index(db)).copied(),
        ValueDef::BlockArg(_) => {
            // For block args, we would need to track block arg types
            // For now, return None and let callers handle this case
            None
        }
    }
}

/// Get the function type from a closure struct value.
///
/// When LowerClosureNewPattern transforms `closure.new` into `adt.struct_new`,
/// the first operand is a `func.constant` that holds the function type.
/// This function traces back through the struct_new to extract that type.
fn get_func_type_from_closure_struct<'db>(
    db: &'db dyn salsa::Database,
    closure_value: Value<'db>,
) -> Option<Type<'db>> {
    // The closure_value should be the result of adt.struct_new
    let ValueDef::OpResult(struct_new_op) = closure_value.def(db) else {
        return None;
    };

    // Check if it's an adt.struct_new
    if adt::StructNew::from_operation(db, struct_new_op).is_err() {
        return None;
    }

    // First operand should be the funcref (from func.constant)
    let operands = struct_new_op.operands(db);
    let funcref_value = operands.first()?;

    // The funcref should be the result of func.constant
    let ValueDef::OpResult(constant_op) = funcref_value.def(db) else {
        return None;
    };

    // Verify it's a func.constant and get the type
    func::Constant::from_operation(db, constant_op)
        .ok()
        .and_then(|_| constant_op.results(db).first().copied())
}
