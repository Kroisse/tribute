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

use tribute_ir::dialect::{adt, closure};
use trunk_ir::dialect::{core, func, wasm};
use trunk_ir::rewrite::{
    OpAdaptor, PatternApplicator, RewritePattern, RewriteResult, TypeConverter,
};
use trunk_ir::{Attribute, DialectOp, DialectType, Operation, Type, Value, ValueDef};

/// Lower closure operations in the module.
///
/// Pattern ordering is important:
/// 1. LowerClosureCallPattern - expands call_indirect to use closure.func/closure.env
/// 2. LowerClosureNewPattern - expands closure.new to func.constant + adt.struct_new
/// 3. LowerClosureFuncPattern - extracts funcref from struct (field 0)
/// 4. LowerClosureEnvPattern - extracts env from struct (field 1)
#[salsa::tracked]
pub fn lower_closures<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> core::Module<'db> {
    let applicator = PatternApplicator::new(TypeConverter::new())
        // First, update function signatures: core.func params → closure.closure
        .add_pattern(UpdateFuncSignaturePattern)
        .add_pattern(LowerClosureCallPattern)
        .add_pattern(LowerClosureNewPattern)
        .add_pattern(LowerClosureFuncPattern)
        .add_pattern(LowerClosureEnvPattern);
    applicator.apply(db, module).module
}

/// Pattern: Update function signatures to convert `core.func` parameters to `closure.closure`.
///
/// This ensures that function parameters with function types accept closure structs,
/// since in Tribute all function values are represented as closures.
struct UpdateFuncSignaturePattern;

impl RewritePattern for UpdateFuncSignaturePattern {
    fn match_and_rewrite<'db>(
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

impl RewritePattern for LowerClosureNewPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
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

        let env = closure_new.env(db);

        // Generate: %funcref = func.constant @func_ref : func_type
        // func_ref is already a Symbol, use it directly
        let constant_op = func::constant(db, location, func_ty, func_ref);
        let funcref = constant_op.as_operation().result(db, 0);

        // Create closure struct type: adt.struct with (funcref, anyref) fields
        // Use abstract funcref and anyref types for uniformity - all closures share
        // the same struct type regardless of their specific function/env types.
        let funcref_ty = wasm::Funcref::new(db).as_type();
        let anyref_ty = wasm::Anyref::new(db).as_type();
        let closure_struct_ty = adt::struct_type(
            db,
            trunk_ir::Symbol::new("_closure"),
            vec![
                (trunk_ir::Symbol::new("funcref"), funcref_ty),
                (trunk_ir::Symbol::new("env"), anyref_ty),
            ],
        );

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

impl RewritePattern for LowerClosureCallPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        // Match: func.call_indirect
        let call = match func::CallIndirect::from_operation(db, *op) {
            Ok(c) => c,
            Err(_) => return RewriteResult::Unchanged,
        };

        let callee = call.callee(db);

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
        let (callee_is_closure, func_ty) = if let Some(callee_ty) = adaptor.operand_type(0) {
            // Type is available - check if it's closure.closure
            if let Some(closure_ty) = closure::Closure::from_type(db, callee_ty) {
                // closure.closure → extract inner func_type
                (true, closure_ty.func_type(db))
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
        let args = call.args(db);
        let result_ty = op
            .results(db)
            .first()
            .copied()
            .expect("call_indirect should have a result");

        // Get env type from the closure.new operation if available.
        // Use anyref for env since closure struct stores env as anyref.
        let anyref_ty = wasm::Anyref::new(db).as_type();

        // Generate: %funcref = closure.func %closure
        // Keep func_ty (core.func) to preserve function signature for call_indirect.
        let funcref_op = closure::func(db, location, callee, func_ty);
        let funcref = funcref_op.as_operation().result(db, 0);

        // Generate: %env = closure.env %closure
        let env_op = closure::env(db, location, callee, anyref_ty);
        let env = env_op.as_operation().result(db, 0);

        // Generate: %result = func.call_indirect %funcref, %env, %args...
        let mut new_args: Vec<Value<'db>> = vec![env];
        new_args.extend(args.iter().copied());

        let new_call = func::call_indirect(db, location, funcref, new_args, result_ty);

        RewriteResult::Expand(vec![
            funcref_op.as_operation(),
            env_op.as_operation(),
            new_call.as_operation(),
        ])
    }
}

/// Pattern: Lower `closure.func` to `adt.struct_get` field 0.
///
/// After closure.new is lowered to an adt.struct with (funcref, env),
/// closure.func extracts the funcref (first field).
struct LowerClosureFuncPattern;

impl RewritePattern for LowerClosureFuncPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        // Match: closure.func
        let closure_func = match closure::Func::from_operation(db, *op) {
            Ok(f) => f,
            Err(_) => return RewriteResult::Unchanged,
        };

        let location = op.location(db);
        let closure_value = closure_func.closure(db);

        // Get the result type (function type)
        let result_ty = op
            .results(db)
            .first()
            .copied()
            .expect("closure.func should have a result");

        // Use the unified _closure struct type (funcref, anyref)
        let funcref_ty = wasm::Funcref::new(db).as_type();
        let anyref_ty = wasm::Anyref::new(db).as_type();
        let struct_ty = adt::struct_type(
            db,
            trunk_ir::Symbol::new("_closure"),
            vec![
                (trunk_ir::Symbol::new("funcref"), funcref_ty),
                (trunk_ir::Symbol::new("env"), anyref_ty),
            ],
        );

        // Generate: %funcref = adt.struct_get %closure, 0
        // Parameter order: (db, location, operand, result_type, struct_type, field_idx)
        let get_op = adt::struct_get(
            db,
            location,
            closure_value,
            result_ty,
            struct_ty,
            Attribute::IntBits(0),
        );

        RewriteResult::Replace(get_op.as_operation())
    }
}

/// Pattern: Lower `closure.env` to `adt.struct_get` field 1.
///
/// After closure.new is lowered to an adt.struct with (funcref, env),
/// closure.env extracts the env (second field).
struct LowerClosureEnvPattern;

impl RewritePattern for LowerClosureEnvPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        // Match: closure.env
        let closure_env = match closure::Env::from_operation(db, *op) {
            Ok(e) => e,
            Err(_) => return RewriteResult::Unchanged,
        };

        let location = op.location(db);
        let closure_value = closure_env.closure(db);

        // Get the result type (env type)
        let result_ty = op
            .results(db)
            .first()
            .copied()
            .expect("closure.env should have a result");

        // Use the unified _closure struct type (funcref, anyref)
        let funcref_ty = wasm::Funcref::new(db).as_type();
        let anyref_ty = wasm::Anyref::new(db).as_type();
        let struct_ty = adt::struct_type(
            db,
            trunk_ir::Symbol::new("_closure"),
            vec![
                (trunk_ir::Symbol::new("funcref"), funcref_ty),
                (trunk_ir::Symbol::new("env"), anyref_ty),
            ],
        );

        // Generate: %env = adt.struct_get %closure, 1
        // Parameter order: (db, location, operand, result_type, struct_type, field_idx)
        let get_op = adt::struct_get(
            db,
            location,
            closure_value,
            result_ty,
            struct_ty,
            Attribute::IntBits(1),
        );

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

/// Get the env type from a closure value by inspecting the closure.new operation.
#[allow(dead_code)]
fn get_env_type_from_closure<'db>(
    db: &'db dyn salsa::Database,
    closure_value: Value<'db>,
) -> Type<'db> {
    if let ValueDef::OpResult(op) = closure_value.def(db)
        && let Ok(closure_new) = closure::New::from_operation(db, op)
    {
        let env_value = closure_new.env(db);
        if let Some(env_ty) = get_value_type(db, env_value) {
            return env_ty;
        }
    }

    // Fallback: return nil type
    *core::Nil::new(db)
}
