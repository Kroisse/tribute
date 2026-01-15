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

use tribute_ir::dialect::{adt, closure, tribute_rt};
use trunk_ir::dialect::{core, func};
use trunk_ir::rewrite::{
    OpAdaptor, PatternApplicator, RewritePattern, RewriteResult, TypeConverter,
};
use trunk_ir::{Attribute, DialectOp, DialectType, Operation, Symbol, Type, Value, ValueDef};

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
/// Pattern ordering is important:
/// 1. LowerClosureCallPattern - expands call_indirect to use closure.func/closure.env
/// 2. LowerClosureNewPattern - expands closure.new to func.constant + adt.struct_new
/// 3. LowerClosureFuncPattern - extracts i32 table index from struct (field 0)
/// 4. LowerClosureEnvPattern - extracts env from struct (field 1)
#[salsa::tracked]
pub fn lower_closures<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> core::Module<'db> {
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

    let applicator = PatternApplicator::new(converter)
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
            Attribute::IntBits(0),
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
