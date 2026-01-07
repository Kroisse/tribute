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
use trunk_ir::dialect::{core, func};
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
        .add_pattern(LowerClosureCallPattern)
        .add_pattern(LowerClosureNewPattern)
        .add_pattern(LowerClosureFuncPattern)
        .add_pattern(LowerClosureEnvPattern);
    applicator.apply(db, module).module
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
        let env_ty = get_value_type(db, env).unwrap_or_else(|| *core::Nil::new(db));

        // Generate: %funcref = func.constant @func_ref : func_type
        // func_ref is already a Symbol, use it directly
        let constant_op = func::constant(db, location, func_ty, func_ref);
        let funcref = constant_op.as_operation().result(db, 0);

        // Create closure struct type: adt.struct with (funcref, env) fields
        let closure_struct_ty = adt::struct_type(
            db,
            trunk_ir::Symbol::new("_closure"),
            vec![
                (trunk_ir::Symbol::new("funcref"), func_ty),
                (trunk_ir::Symbol::new("env"), env_ty),
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
        _adaptor: &OpAdaptor<'db, '_>,
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

        // Get the struct type from the closure value
        let struct_ty = get_value_type(db, closure_value).unwrap_or_else(|| *core::Nil::new(db));

        // Generate: %funcref = adt.struct_get %closure, 0
        let get_op = adt::struct_get(
            db,
            location,
            closure_value,
            struct_ty,
            result_ty,
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

        // Get the struct type from the closure value
        let struct_ty = get_value_type(db, closure_value).unwrap_or_else(|| *core::Nil::new(db));

        // Generate: %env = adt.struct_get %closure, 1
        let get_op = adt::struct_get(
            db,
            location,
            closure_value,
            struct_ty,
            result_ty,
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
