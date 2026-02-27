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

use tribute_ir::arena::dialect::closure as arena_closure;
use tribute_ir::dialect::{closure, tribute_rt};
use trunk_ir::arena::context::IrContext;
use trunk_ir::arena::dialect::adt as arena_adt;
use trunk_ir::arena::dialect::func as arena_func;
use trunk_ir::arena::ops::ArenaDialectOp;
use trunk_ir::arena::refs::{OpRef, TypeRef, ValueRef};
use trunk_ir::arena::rewrite::{
    ArenaModule, ArenaRewritePattern, ArenaTypeConverter,
    PatternApplicator as ArenaPatternApplicator, PatternRewriter as ArenaPatternRewriter,
};
use trunk_ir::arena::types::{Attribute as ArenaAttribute, TypeDataBuilder};
use trunk_ir::dialect::adt;
use trunk_ir::dialect::{cont, core, func};
use trunk_ir::rewrite::{
    ConversionTarget, PatternApplicator, PatternRewriter, RewritePattern, TypeConverter,
};
use trunk_ir::{
    Attribute, Block, DialectOp, DialectType, IdVec, Operation, Region, Symbol, Type, Value,
    ValueDef,
};

use crate::evidence::{collect_effectful_functions, collect_effectful_functions_arena};

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
            // Create null evidence using the canonical evidence ADT type
            let evidence_ty = tribute_ir::dialect::ability::evidence_adt_type(db);
            let null_ev_op = adt::ref_null(db, func_location, evidence_ty, evidence_ty);
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
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        // Match: func.func
        let func_op = match func::Func::from_operation(db, *op) {
            Ok(f) => f,
            Err(_) => return false,
        };

        let func_ty = func_op.r#type(db);
        let Some(func_type) = core::Func::from_type(db, func_ty) else {
            return false;
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
            return false;
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

        rewriter.replace_op(new_op);
        true
    }
}

/// Pattern: Lower `closure.new` to `func.constant` + `adt.struct_new`.
struct LowerClosureNewPattern;

impl<'db> RewritePattern<'db> for LowerClosureNewPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        // Match: closure.new
        let closure_new = match closure::New::from_operation(db, *op) {
            Ok(c) => c,
            Err(_) => return false,
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

        // Get env from rewriter (remapped value)
        let env = rewriter
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

        rewriter.insert_op(constant_op.as_operation());
        rewriter.replace_op(struct_new_op.as_operation());
        true
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
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        // Match: func.call_indirect
        if func::CallIndirect::from_operation(db, *op).is_err() {
            return false;
        }

        // Get callee from rewriter (remapped value)
        let callee = rewriter
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
        let callee_ty_opt = rewriter.operand_type(0);

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
            return false;
        }

        // Get location and other info
        let location = op.location(db);
        // Get args from rewriter (remapped values), skipping the callee (index 0)
        let args: Vec<_> = rewriter.operands().iter().skip(1).copied().collect();
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

        rewriter.insert_op(table_idx_op.as_operation());
        rewriter.insert_op(env_op.as_operation());
        rewriter.replace_op(new_call.as_operation());
        true
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
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        // Match: closure.func
        if closure::Func::from_operation(db, *op).is_err() {
            return false;
        }

        let location = op.location(db);
        // Get closure from rewriter (remapped value)
        let closure_value = rewriter
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

        rewriter.replace_op(get_op.as_operation());
        true
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
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        // Match: closure.env
        if closure::Env::from_operation(db, *op).is_err() {
            return false;
        }

        let location = op.location(db);
        // Get closure from rewriter (remapped value)
        let closure_value = rewriter
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

        rewriter.replace_op(get_op.as_operation());
        true
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
            // should use the PatternRewriter to get the actual type.
            //
            // We return false here to be conservative. This means that block args
            // without explicit closure.closure type in the rewriter won't be
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

// ============================================================================
// Arena-based closure lowering implementation
// ============================================================================

/// Create the unified closure struct type in arena: `{ table_idx: i32, env: anyref }`.
fn closure_struct_type_ref(ctx: &mut IrContext) -> TypeRef {
    let i32_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());
    let anyref_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("tribute_rt"), Symbol::new("any")).build());
    ctx.types.intern(
        TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("struct"))
            .param(i32_ty)
            .param(anyref_ty)
            .attr("name", ArenaAttribute::Symbol(Symbol::new("_closure")))
            .build(),
    )
}

/// Create an `i32` type ref in arena.
fn i32_type_ref(ctx: &mut IrContext) -> TypeRef {
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build())
}

/// Create a `tribute_rt.any` (anyref) type ref in arena.
fn anyref_type_ref(ctx: &mut IrContext) -> TypeRef {
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("tribute_rt"), Symbol::new("any")).build())
}

/// Create a `core.nil` type ref in arena.
fn nil_type_ref(ctx: &mut IrContext) -> TypeRef {
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("nil")).build())
}

/// Check if a TypeRef is a closure.closure type.
fn is_closure_type_ref(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.types.get(ty);
    data.dialect == Symbol::new("closure") && data.name == Symbol::new("closure")
}

/// Check if a TypeRef is a core.func type.
fn is_core_func_type_ref(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.types.get(ty);
    data.dialect == Symbol::new("core") && data.name == Symbol::new("func")
}

/// Check if a TypeRef is a cont.continuation type.
fn is_continuation_type_ref(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.types.get(ty);
    data.dialect == Symbol::new("cont") && data.name == Symbol::new("continuation")
}

/// Check if a TypeRef is an adt.struct with name "_closure".
fn is_closure_struct_type_ref(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.types.get(ty);
    if data.dialect != Symbol::new("adt") || data.name != Symbol::new("struct") {
        return false;
    }
    matches!(
        data.attrs.get(&Symbol::new("name")),
        Some(ArenaAttribute::Symbol(s)) if *s == Symbol::new("_closure")
    )
}

/// Extract the inner func type from a closure.closure TypeRef.
fn extract_closure_func_type(ctx: &IrContext, ty: TypeRef) -> Option<TypeRef> {
    let data = ctx.types.get(ty);
    if data.dialect != Symbol::new("closure") || data.name != Symbol::new("closure") {
        return None;
    }
    // closure.closure type has the inner func type as first param
    data.params.first().copied()
}

/// Get the type of an arena value.
fn arena_value_type(ctx: &IrContext, value: ValueRef) -> TypeRef {
    ctx.value_ty(value)
}

/// Check if an arena value is a closure value (for pre-lowering collection).
fn is_any_closure_value_arena(ctx: &IrContext, value: ValueRef) -> bool {
    use trunk_ir::arena::refs::ValueDef as ArenaValueDef;

    let ty = arena_value_type(ctx, value);

    // Direct check for closure.new result
    if let ArenaValueDef::OpResult(op, _) = ctx.value_def(value) {
        if arena_closure::New::from_op(ctx, op).is_ok() {
            return true;
        }
    }

    // Check type
    if is_closure_type_ref(ctx, ty) {
        return true;
    }
    if is_core_func_type_ref(ctx, ty) {
        return true;
    }
    if is_continuation_type_ref(ctx, ty) {
        return true;
    }
    false
}

/// Collect ALL closure call_indirect operations as OpRefs.
/// Collect all closure calls by location span (stable across Phase 1 rewrites).
///
/// We use `(span.start, span.end)` instead of `OpRef` because Phase 1 pattern
/// application destroys original ops and creates new ones with different OpRefs.
fn collect_all_closure_calls_arena(
    ctx: &IrContext,
    module: ArenaModule,
) -> HashSet<(usize, usize)> {
    let mut closure_calls = HashSet::new();
    for op in module.ops(ctx) {
        if let Ok(func_op) = arena_func::Func::from_op(ctx, op) {
            let body = func_op.body(ctx);
            collect_closure_calls_in_region_arena(ctx, body, &mut closure_calls);
        }
    }
    closure_calls
}

fn collect_closure_calls_in_region_arena(
    ctx: &IrContext,
    region: trunk_ir::arena::refs::RegionRef,
    closure_calls: &mut HashSet<(usize, usize)>,
) {
    for &block in ctx.region(region).blocks.iter() {
        for &op in ctx.block(block).ops.iter() {
            collect_closure_calls_in_op_arena(ctx, op, closure_calls);
        }
    }
}

fn collect_closure_calls_in_op_arena(
    ctx: &IrContext,
    op: OpRef,
    closure_calls: &mut HashSet<(usize, usize)>,
) {
    // Check if this is a call_indirect with a closure callee
    if arena_func::CallIndirect::from_op(ctx, op).is_ok() {
        let operands = ctx.op_operands(op);
        if let Some(&callee) = operands.first() {
            if is_any_closure_value_arena(ctx, callee) {
                let loc = ctx.op(op).location;
                closure_calls.insert((loc.span.start, loc.span.end));
            }
        }
    }

    // Recurse into regions
    for &region in ctx.op(op).regions.iter() {
        collect_closure_calls_in_region_arena(ctx, region, closure_calls);
    }
}

// ============================================================================
// Arena Rewrite Patterns
// ============================================================================

/// Arena pattern: Update function signatures to convert `core.func` params to `closure.closure`.
struct UpdateFuncSignatureArena;

impl ArenaRewritePattern for UpdateFuncSignatureArena {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut ArenaPatternRewriter<'_>,
    ) -> bool {
        let Ok(func_op) = arena_func::Func::from_op(ctx, op) else {
            return false;
        };

        let func_ty = func_op.r#type(ctx);
        let func_data = ctx.types.get(func_ty);
        if func_data.dialect != Symbol::new("core") || func_data.name != Symbol::new("func") {
            return false;
        }

        // params[0] = return, params[1..] = param types
        if func_data.params.is_empty() {
            return false;
        }

        // Clone data we need before mutating ctx.types
        let params: Vec<TypeRef> = func_data.params.to_vec();
        let effect_attr = func_data.attrs.get(&Symbol::new("effect")).cloned();

        let mut needs_update = false;
        let mut new_params = Vec::with_capacity(params.len());
        new_params.push(params[0]); // return type

        for &param_ty in &params[1..] {
            if is_core_func_type_ref(ctx, param_ty) {
                // Convert core.func to closure.closure wrapping the func type
                let closure_ty = ctx.types.intern(
                    TypeDataBuilder::new(Symbol::new("closure"), Symbol::new("closure"))
                        .param(param_ty)
                        .build(),
                );
                new_params.push(closure_ty);
                needs_update = true;
            } else {
                new_params.push(param_ty);
            }
        }

        if !needs_update {
            return false;
        }

        // Build new func type preserving effect attribute
        let mut builder = TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func"))
            .params(new_params.into_iter());
        if let Some(eff) = effect_attr {
            builder = builder.attr("effect", eff);
        }
        let new_func_ty = ctx.types.intern(builder.build());

        // Rebuild the function with new type
        let func_name = func_op.sym_name(ctx);
        let body = func_op.body(ctx);
        let loc = ctx.op(op).location;
        ctx.detach_region(body);
        let new_op = arena_func::func(ctx, loc, func_name, new_func_ty, body).op_ref();
        rewriter.replace_op(new_op);
        true
    }

    fn name(&self) -> &'static str {
        "UpdateFuncSignatureArena"
    }
}

/// Arena pattern: Lower `closure.new` to `func.constant` + `adt.struct_new`.
struct LowerClosureNewArena;

impl ArenaRewritePattern for LowerClosureNewArena {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut ArenaPatternRewriter<'_>,
    ) -> bool {
        let Ok(closure_new) = arena_closure::New::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let func_ref = closure_new.func_ref(ctx);
        let env = closure_new.env(ctx);

        // Extract function type from closure.closure result type
        let result_ty = ctx.op_result_types(op)[0];
        let func_ty =
            extract_closure_func_type(ctx, result_ty).unwrap_or_else(|| nil_type_ref(ctx));

        // Generate: %funcref = func.constant @func_ref : func_type
        let constant_op = arena_func::constant(ctx, loc, func_ty, func_ref);
        let funcref = ctx.op_result(constant_op.op_ref(), 0);

        // Generate: %closure = adt.struct_new(%funcref, %env) : closure_struct_type
        let struct_ty = closure_struct_type_ref(ctx);
        let struct_new_op =
            arena_adt::struct_new(ctx, loc, vec![funcref, env], struct_ty, struct_ty);

        rewriter.insert_op(constant_op.op_ref());
        rewriter.replace_op(struct_new_op.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "LowerClosureNewArena"
    }
}

/// Arena pattern: Lower `func.call_indirect` on closure values.
struct LowerClosureCallArena;

impl ArenaRewritePattern for LowerClosureCallArena {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut ArenaPatternRewriter<'_>,
    ) -> bool {
        if arena_func::CallIndirect::from_op(ctx, op).is_err() {
            return false;
        }

        let operands = ctx.op_operands(op);
        if operands.is_empty() {
            return false;
        }
        let callee = operands[0];
        let callee_ty = arena_value_type(ctx, callee);

        // Determine if callee is a closure
        let callee_is_closure = if is_closure_type_ref(ctx, callee_ty) {
            true
        } else if is_closure_struct_type_ref(ctx, callee_ty) {
            // Already lowered closure struct
            true
        } else if is_core_func_type_ref(ctx, callee_ty) {
            // core.func: only treat as closure if it's a block arg
            matches!(
                ctx.value_def(callee),
                trunk_ir::arena::refs::ValueDef::BlockArg(_, _)
            )
        } else {
            // Fallback: check if result of closure.new
            if let trunk_ir::arena::refs::ValueDef::OpResult(def_op, _) = ctx.value_def(callee) {
                arena_closure::New::from_op(ctx, def_op).is_ok()
            } else {
                false
            }
        };

        if !callee_is_closure {
            return false;
        }

        let loc = ctx.op(op).location;
        let args: Vec<ValueRef> = operands[1..].to_vec();
        let result_ty = ctx.op_result_types(op)[0];

        let i32_ty = i32_type_ref(ctx);
        let anyref_ty = anyref_type_ref(ctx);

        // Generate: %table_idx = closure.func %closure
        let table_idx_op = arena_closure::func(ctx, loc, callee, i32_ty);
        let table_idx = ctx.op_result(table_idx_op.op_ref(), 0);

        // Generate: %env = closure.env %closure
        let env_op = arena_closure::env(ctx, loc, callee, anyref_ty);
        let env = ctx.op_result(env_op.op_ref(), 0);

        // Generate: %result = func.call_indirect %table_idx, [%env, %args...]
        let mut new_args = vec![env];
        new_args.extend(args);
        let new_call = arena_func::call_indirect(ctx, loc, table_idx, new_args, result_ty);

        rewriter.insert_op(table_idx_op.op_ref());
        rewriter.insert_op(env_op.op_ref());
        rewriter.replace_op(new_call.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "LowerClosureCallArena"
    }
}

/// Arena pattern: Lower `closure.func` to `adt.struct_get` field 0.
struct LowerClosureFuncArena;

impl ArenaRewritePattern for LowerClosureFuncArena {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut ArenaPatternRewriter<'_>,
    ) -> bool {
        if arena_closure::Func::from_op(ctx, op).is_err() {
            return false;
        }

        let loc = ctx.op(op).location;
        let closure_value = ctx.op_operands(op)[0];
        let i32_ty = i32_type_ref(ctx);
        let struct_ty = closure_struct_type_ref(ctx);

        let get_op = arena_adt::struct_get(ctx, loc, closure_value, i32_ty, struct_ty, 0);
        rewriter.replace_op(get_op.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "LowerClosureFuncArena"
    }
}

/// Arena pattern: Lower `closure.env` to `adt.struct_get` field 1.
struct LowerClosureEnvArena;

impl ArenaRewritePattern for LowerClosureEnvArena {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut ArenaPatternRewriter<'_>,
    ) -> bool {
        if arena_closure::Env::from_op(ctx, op).is_err() {
            return false;
        }

        let loc = ctx.op(op).location;
        let closure_value = ctx.op_operands(op)[0];
        let result_ty = ctx.op_result_types(op)[0];
        let struct_ty = closure_struct_type_ref(ctx);

        let get_op = arena_adt::struct_get(ctx, loc, closure_value, result_ty, struct_ty, 1);
        rewriter.replace_op(get_op.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "LowerClosureEnvArena"
    }
}

// ============================================================================
// Phase 2: Evidence passing for closure calls (arena)
// ============================================================================

/// Transform closure calls to pass evidence in arena IR.
///
/// After pattern application, closure calls have been expanded. Now we insert
/// evidence as the first argument to all closure call_indirect operations.
fn transform_closure_calls_with_evidence_arena(
    ctx: &mut IrContext,
    module: ArenaModule,
    effectful_fns: &HashSet<Symbol>,
    closure_calls: &HashSet<(usize, usize)>,
) {
    if closure_calls.is_empty() {
        return;
    }

    let func_ops: Vec<OpRef> = module.ops(ctx);
    for func_op_ref in func_ops {
        let Ok(func_op) = arena_func::Func::from_op(ctx, func_op_ref) else {
            continue;
        };

        let func_name = func_op.sym_name(ctx);
        let is_effectful = effectful_fns.contains(&func_name);

        let body = func_op.body(ctx);
        let blocks: Vec<_> = ctx.region(body).blocks.to_vec();
        if blocks.is_empty() {
            continue;
        }

        // For effectful functions, get evidence from first block argument
        let evidence_from_param = if is_effectful {
            let entry = blocks[0];
            let args = ctx.block_args(entry);
            if !args.is_empty() {
                let ev = args[0];
                if tribute_ir::arena::dialect::ability::is_evidence_type_ref(ctx, ctx.value_ty(ev))
                {
                    Some(ev)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        let loc = ctx.op(func_op_ref).location;
        transform_closure_calls_in_region_arena(ctx, body, evidence_from_param, closure_calls, loc);
    }
}

/// Transform closure calls in a region, inserting evidence arguments.
fn transform_closure_calls_in_region_arena(
    ctx: &mut IrContext,
    region: trunk_ir::arena::refs::RegionRef,
    evidence_from_param: Option<ValueRef>,
    closure_calls: &HashSet<(usize, usize)>,
    func_location: trunk_ir::arena::types::Location,
) {
    let blocks: Vec<_> = ctx.region(region).blocks.to_vec();
    for block in blocks {
        transform_closure_calls_in_block_arena(
            ctx,
            block,
            evidence_from_param,
            closure_calls,
            func_location,
        );
    }
}

/// Transform closure calls in a block, inserting evidence arguments.
fn transform_closure_calls_in_block_arena(
    ctx: &mut IrContext,
    block: trunk_ir::arena::refs::BlockRef,
    evidence_from_param: Option<ValueRef>,
    closure_calls: &HashSet<(usize, usize)>,
    func_location: trunk_ir::arena::types::Location,
) {
    // We need to track null evidence creation (lazy)
    let mut null_ev_value: Option<ValueRef> = None;

    // Collect ops to process (snapshot since we'll mutate)
    let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();

    for op in ops {
        // Process nested regions first
        let regions: Vec<_> = ctx.op(op).regions.to_vec();
        for region in regions {
            transform_closure_calls_in_region_arena(
                ctx,
                region,
                evidence_from_param,
                closure_calls,
                func_location,
            );
        }

        // Check if this is a closure call_indirect that needs evidence
        if arena_func::CallIndirect::from_op(ctx, op).is_err() {
            continue;
        }

        let loc_key = {
            let loc = ctx.op(op).location;
            (loc.span.start, loc.span.end)
        };
        if !closure_calls.contains(&loc_key) {
            continue;
        }

        let evidence = if let Some(ev) = evidence_from_param {
            ev
        } else {
            // Create null evidence lazily
            if null_ev_value.is_none() {
                let evidence_ty = tribute_ir::arena::dialect::ability::evidence_adt_type_ref(ctx);
                let null_op = arena_adt::ref_null(ctx, func_location, evidence_ty, evidence_ty);
                let ev = ctx.op_result(null_op.op_ref(), 0);
                // Insert null evidence at the beginning of the block
                let first_op_in_block = ctx.block(block).ops.first().copied();
                if let Some(first_op) = first_op_in_block {
                    ctx.insert_op_before(block, first_op, null_op.op_ref());
                } else {
                    ctx.push_op(block, null_op.op_ref());
                }
                null_ev_value = Some(ev);
            }
            null_ev_value.unwrap()
        };

        // Transform: add evidence as first argument after table_idx
        let operands = ctx.op_operands(op).to_vec();
        let result_ty = ctx.op_result_types(op)[0];
        let loc = ctx.op(op).location;

        // operands[0] = table_idx, operands[1..] = env + args
        let table_idx = operands[0];
        let rest_args: Vec<ValueRef> = operands[1..].to_vec();

        // Build new args: [evidence, env, args...]
        let mut new_args = vec![evidence];
        new_args.extend(rest_args);

        let new_call = arena_func::call_indirect(ctx, loc, table_idx, new_args, result_ty);

        // Replace old result uses with new result
        let old_result = ctx.op_result(op, 0);
        let new_result = ctx.op_result(new_call.op_ref(), 0);
        ctx.replace_all_uses(old_result, new_result);

        // Insert new call and remove old one
        ctx.insert_op_before(block, op, new_call.op_ref());
        ctx.remove_op_from_block(block, op);
    }
}

// ============================================================================
// Arena entry point
// ============================================================================

/// Lower closures using arena IR.
///
/// This is the arena equivalent of `lower_closures`. It performs:
/// - Phase 1: Pattern-based closure lowering (5 patterns)
/// - Phase 2: Evidence passing for closure calls
pub fn lower_closures_arena(ctx: &mut IrContext, module: ArenaModule) {
    // Collect effectful functions BEFORE lowering (while closure types are intact)
    let effectful_fns = collect_effectful_functions_arena(ctx, module);

    // Collect ALL closure calls before pattern application
    let all_closure_calls = collect_all_closure_calls_arena(ctx, module);

    // Phase 1: Pattern application
    let applicator = ArenaPatternApplicator::new(ArenaTypeConverter::new())
        .add_pattern(UpdateFuncSignatureArena)
        .add_pattern(LowerClosureCallArena)
        .add_pattern(LowerClosureNewArena)
        .add_pattern(LowerClosureFuncArena)
        .add_pattern(LowerClosureEnvArena);
    applicator.apply_partial(ctx, module);

    // Phase 2: Evidence passing for closure calls
    transform_closure_calls_with_evidence_arena(ctx, module, &effectful_fns, &all_closure_calls);
}
