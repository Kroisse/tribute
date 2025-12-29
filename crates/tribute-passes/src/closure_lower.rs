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

use trunk_ir::dialect::{closure, core, func};
use trunk_ir::rewrite::{PatternApplicator, RewritePattern, RewriteResult};
use trunk_ir::{DialectOp, DialectType, Operation, Type, Value, ValueDef};

/// Lower closure operations in the module.
#[salsa::tracked]
pub fn lower_closures<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> core::Module<'db> {
    let applicator = PatternApplicator::new().add_pattern(LowerClosureCallPattern);
    applicator.apply(db, module).module
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
    ) -> RewriteResult<'db> {
        // Match: func.call_indirect
        let call = match func::CallIndirect::from_operation(db, *op) {
            Ok(c) => c,
            Err(_) => return RewriteResult::Unchanged,
        };

        let callee = call.callee(db);

        // Check if callee is a closure
        if !is_closure_value(db, callee) {
            return RewriteResult::Unchanged;
        }

        // Get the function type for the extracted funcref
        let func_ty = get_closure_func_type(db, callee);

        // Get location and other info
        let location = op.location(db);
        let args = call.args(db);
        let result_ty = op
            .results(db)
            .first()
            .copied()
            .expect("call_indirect should have a result");

        // Get env type from the closure.new operation if available
        let env_ty = get_env_type_from_closure(db, callee);

        // Generate: %funcref = closure.func %closure
        let funcref_op = closure::func(db, location, callee, func_ty);
        let funcref = funcref_op.as_operation().result(db, 0);

        // Generate: %env = closure.env %closure
        let env_op = closure::env(db, location, callee, env_ty);
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
            // For block args (function parameters), we use a heuristic:
            // If the parameter has a function-like type, treat it as a closure.
            // This handles cases like `fn apply(f, x) { f(x) }` where f is
            // passed a closure from the caller.
            //
            // TODO: A more precise approach would be to track which block args
            // receive closures at call sites, or update function signatures
            // during lambda_lift to use closure.closure types.
            true
        }
    }
}

/// Get the function type from a closure value.
fn get_closure_func_type<'db>(db: &'db dyn salsa::Database, value: Value<'db>) -> Type<'db> {
    // Try to get closure.closure type and extract func_type
    if let Some(ty) = get_value_type(db, value) {
        return closure::Closure::from_type(db, ty)
            .map(|ct| ct.func_type(db))
            .unwrap_or(ty);
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
