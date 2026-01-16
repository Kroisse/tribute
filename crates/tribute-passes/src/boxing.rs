//! Boxing insertion pass for polymorphic function calls.
//!
//! This pass inserts explicit `tribute_rt.box_*` and `tribute_rt.unbox_*` operations
//! at call sites where polymorphic parameters or results need boxing/unboxing.
//!
//! ## Problem
//!
//! When calling a generic function like `identity(x: a) -> a` with a concrete type,
//! the value needs to be boxed to anyref for the polymorphic parameter, and unboxed
//! after the call if a concrete type is expected.
//!
//! ## Example
//!
//! Before:
//! ```text
//! %result = func.call @identity(%x) -> tribute.type_var
//! ```
//!
//! After:
//! ```text
//! %boxed = tribute_rt.box_int(%x)
//! %result_any = func.call @identity(%boxed) -> tribute_rt.any
//! %result = tribute_rt.unbox_int(%result_any)
//! ```
//!
//! This makes boxing explicit in the IR, removing the need for emit-time type inference.

use std::collections::HashMap;

use tracing::debug;
use tribute_ir::dialect::{closure, tribute, tribute_rt};
use trunk_ir::dialect::core::{self, F64 as CoreF64, I32 as CoreI32, Module};
use trunk_ir::dialect::func;
use trunk_ir::rewrite::{
    ConversionTarget, OpAdaptor, PatternApplicator, RewritePattern, RewriteResult, TypeConverter,
};
use trunk_ir::{Attribute, DialectOp, DialectType, Location, Operation, Symbol, Type, Value};

// =============================================================================
// Helper Functions
// =============================================================================

/// Collect function types from a module.
fn collect_func_types<'db>(
    db: &'db dyn salsa::Database,
    module: &Module<'db>,
) -> HashMap<Symbol, core::Func<'db>> {
    let mut func_types = HashMap::new();
    let body = module.body(db);

    for block in body.blocks(db).iter() {
        for op in block.operations(db).iter() {
            if let Ok(func_op) = func::Func::from_operation(db, *op) {
                let name = func_op.sym_name(db);
                let func_ty = func_op.r#type(db);
                if let Some(func_type) = core::Func::from_type(db, func_ty) {
                    debug!(
                        "boxing: collected function {} with type {:?}",
                        name, func_ty
                    );
                    func_types.insert(name, func_type);
                }
            }
        }
    }

    func_types
}

/// Get the ID of a type_var (if it is one).
fn get_type_var_id<'db>(db: &'db dyn salsa::Database, ty: Type<'db>) -> Option<u64> {
    if !tribute::is_type_var(db, ty) {
        return None;
    }

    let attrs = ty.attrs(db);
    match attrs.get(&Symbol::new("id")) {
        Some(Attribute::IntBits(id)) => Some(*id),
        _ => None,
    }
}

/// Categorize value type for boxing/unboxing purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ValueCategory {
    /// Signed integer (tribute_rt.int or core.i32)
    Int,
    /// Unsigned natural number (tribute_rt.nat)
    Nat,
    /// Boolean (tribute_rt.bool)
    Bool,
    /// Float (tribute_rt.float or core.f64)
    Float,
    /// Reference type (no boxing needed)
    Reference,
}

/// Categorize a type for boxing/unboxing.
fn categorize_type<'db>(db: &'db dyn salsa::Database, ty: Type<'db>) -> ValueCategory {
    if tribute_rt::is_int(db, ty) || CoreI32::from_type(db, ty).is_some() {
        ValueCategory::Int
    } else if tribute_rt::is_nat(db, ty) {
        ValueCategory::Nat
    } else if tribute_rt::is_bool(db, ty) {
        ValueCategory::Bool
    } else if tribute_rt::is_float(db, ty) || CoreF64::from_type(db, ty).is_some() {
        ValueCategory::Float
    } else {
        ValueCategory::Reference
    }
}

/// Create a boxing operation for a value with known type.
/// Returns (box_op, boxed_value) or None if boxing not applicable.
fn create_box_op<'db>(
    db: &'db dyn salsa::Database,
    value: Value<'db>,
    value_ty: Type<'db>,
    location: Location<'db>,
) -> Option<(Operation<'db>, Value<'db>)> {
    let any_ty = *tribute_rt::Any::new(db);

    match categorize_type(db, value_ty) {
        ValueCategory::Int => {
            let box_op = tribute_rt::box_int(db, location, value, any_ty);
            let boxed = box_op.as_operation().result(db, 0);
            Some((box_op.as_operation(), boxed))
        }
        ValueCategory::Nat => {
            let box_op = tribute_rt::box_nat(db, location, value, any_ty);
            let boxed = box_op.as_operation().result(db, 0);
            Some((box_op.as_operation(), boxed))
        }
        ValueCategory::Bool => {
            let box_op = tribute_rt::box_bool(db, location, value, any_ty);
            let boxed = box_op.as_operation().result(db, 0);
            Some((box_op.as_operation(), boxed))
        }
        ValueCategory::Float => {
            let box_op = tribute_rt::box_float(db, location, value, any_ty);
            let boxed = box_op.as_operation().result(db, 0);
            Some((box_op.as_operation(), boxed))
        }
        ValueCategory::Reference => {
            // Reference types don't need boxing (already subtypes of anyref)
            None
        }
    }
}

/// Create unboxing operations for a value.
/// Returns (ops, unboxed_value) or None if unboxing not applicable.
///
/// This only emits `tribute_rt.unbox_*` operations. The ref_cast operations
/// are handled by tribute_rt_to_wasm.rs during lowering.
fn create_unbox_op<'db>(
    db: &'db dyn salsa::Database,
    value: Value<'db>,
    target_ty: Type<'db>,
    location: Location<'db>,
) -> Option<(Vec<Operation<'db>>, Value<'db>)> {
    match categorize_type(db, target_ty) {
        ValueCategory::Int => {
            // Emit unbox_int (signed) - tribute_rt_to_wasm will add ref_cast to i31ref
            let unbox_op = tribute_rt::unbox_int(db, location, value, target_ty);
            let unboxed = unbox_op.as_operation().result(db, 0);
            Some((vec![unbox_op.as_operation()], unboxed))
        }
        ValueCategory::Nat => {
            // Emit unbox_nat (unsigned) - tribute_rt_to_wasm will add ref_cast to i31ref
            let unbox_op = tribute_rt::unbox_nat(db, location, value, target_ty);
            let unboxed = unbox_op.as_operation().result(db, 0);
            Some((vec![unbox_op.as_operation()], unboxed))
        }
        ValueCategory::Bool => {
            // Emit unbox_bool - tribute_rt_to_wasm will add ref_cast to i31ref
            let unbox_op = tribute_rt::unbox_bool(db, location, value, target_ty);
            let unboxed = unbox_op.as_operation().result(db, 0);
            Some((vec![unbox_op.as_operation()], unboxed))
        }
        ValueCategory::Float => {
            // Emit unbox_float - tribute_rt_to_wasm will add ref_cast to BoxedF64
            let unbox_op = tribute_rt::unbox_float(db, location, value, target_ty);
            let unboxed = unbox_op.as_operation().result(db, 0);
            Some((vec![unbox_op.as_operation()], unboxed))
        }
        ValueCategory::Reference => {
            // Reference types don't need unboxing
            None
        }
    }
}

// =============================================================================
// Rewrite Patterns
// =============================================================================

/// Pattern for boxing/unboxing at `func.call` sites.
///
/// Transforms polymorphic function calls by:
/// 1. Boxing concrete arguments passed to type_var parameters
/// 2. Changing result type to tribute_rt.any if return is type_var
/// 3. Unboxing the result to the expected concrete type
struct BoxCallPattern<'db> {
    func_types: HashMap<Symbol, core::Func<'db>>,
}

impl<'db> RewritePattern<'db> for BoxCallPattern<'db> {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(call_op) = func::Call::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let callee = call_op.callee(db);
        let Some(func_type) = self.func_types.get(&callee) else {
            return RewriteResult::Unchanged;
        };

        let param_types = func_type.params(db);
        let return_ty = func_type.result(db);

        let operands = adaptor.operands();
        let location = op.location(db);

        // Check if any boxing is needed
        let needs_boxing = param_types.iter().any(|ty| tribute::is_type_var(db, *ty));
        let needs_unboxing = tribute::is_type_var(db, return_ty);

        if !needs_boxing && !needs_unboxing {
            return RewriteResult::Unchanged;
        }

        let mut result_ops: Vec<Operation<'db>> = Vec::new();
        let mut new_operands: Vec<Value<'db>> = Vec::new();

        // Track the concrete types of arguments for type_var positions
        // This will be used to infer the unbox type when return is type_var
        let mut type_var_to_concrete: HashMap<u64, Type<'db>> = HashMap::new();

        // Verify operand count matches parameter count
        if operands.len() != param_types.len() {
            debug!(
                "boxing: operand count mismatch for call to {}: {} operands, {} params",
                callee,
                operands.len(),
                param_types.len()
            );
            return RewriteResult::Unchanged;
        }

        // Insert boxing for each operand that needs it
        for (i, param_ty) in param_types.iter().enumerate() {
            let operand = operands[i];

            if tribute::is_type_var(db, *param_ty) {
                // Get the concrete type of this operand (works for both OpResults and BlockArgs)
                let operand_ty = adaptor.get_value_type(db, operand);

                // Track for later unboxing inference
                if let Some(ty) = operand_ty
                    && let Some(type_var_id) = get_type_var_id(db, *param_ty)
                {
                    type_var_to_concrete.insert(type_var_id, ty);
                }

                // Need to box this operand
                if let Some(ty) = operand_ty
                    && let Some((box_op, boxed_val)) = create_box_op(db, operand, ty, location)
                {
                    result_ops.push(box_op);
                    new_operands.push(boxed_val);
                } else {
                    // Can't determine type to box, pass through
                    new_operands.push(operand);
                }
            } else {
                new_operands.push(operand);
            }
        }

        // Create the call with potentially boxed operands
        let call_result_ty = if needs_unboxing {
            // Change result type to tribute_rt.any
            *tribute_rt::Any::new(db)
        } else {
            op.results(db).first().copied().unwrap_or(return_ty)
        };

        let new_call = func::call(db, location, new_operands, call_result_ty, callee);
        let new_call_op = new_call.as_operation();
        result_ops.push(new_call_op);

        // Insert unboxing if needed
        if needs_unboxing {
            let call_result = new_call_op.result(db, 0);

            // Get the expected concrete type from:
            // 1. The call result type if it's not a type_var (typeck resolved it)
            // 2. Or infer from the argument type if return type_var matches a param type_var
            let original_result_ty = op.results(db).first().copied();

            debug!(
                "boxing: needs_unboxing for call to {}, original_result_ty: {:?}",
                callee, original_result_ty
            );

            // Try to find concrete type for unboxing
            let unbox_ty = if let Some(orig_ty) = original_result_ty {
                if !tribute::is_type_var(db, orig_ty) {
                    // typeck resolved it to a concrete type
                    Some(orig_ty)
                } else if let Some(type_var_id) = get_type_var_id(db, return_ty) {
                    // Return type is a type_var, look up the concrete type from arguments
                    type_var_to_concrete.get(&type_var_id).copied()
                } else {
                    None
                }
            } else {
                None
            };

            debug!("boxing: inferred unbox_ty: {:?}", unbox_ty);

            if let Some(target_ty) = unbox_ty {
                // Insert unbox (which may include ref_cast + unbox)
                if let Some((unbox_ops, _unboxed_val)) =
                    create_unbox_op(db, call_result, target_ty, location)
                {
                    debug!(
                        "boxing: inserted {} unbox ops for type {:?}",
                        unbox_ops.len(),
                        target_ty
                    );
                    result_ops.extend(unbox_ops);
                }
            }
        }

        RewriteResult::Expand(result_ops)
    }

    fn name(&self) -> &'static str {
        "BoxCallPattern"
    }
}

/// Pattern for boxing/unboxing at `func.call_indirect` sites.
///
/// Transforms polymorphic indirect function calls by:
/// 1. Getting the function type from the callee (core.func or closure.closure)
/// 2. Boxing concrete arguments passed to type_var parameters
/// 3. Changing result type to tribute_rt.any if return is type_var
/// 4. Unboxing the result to the expected concrete type
struct BoxCallIndirectPattern;

impl<'db> RewritePattern<'db> for BoxCallIndirectPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        // Match func.call_indirect
        if func::CallIndirect::from_operation(db, *op).is_err() {
            return RewriteResult::Unchanged;
        }

        let operands = adaptor.operands();
        if operands.is_empty() {
            return RewriteResult::Unchanged;
        }

        // Get callee (first operand) and its type
        let callee = operands[0];
        let Some(callee_ty) = adaptor.operand_type(0) else {
            return RewriteResult::Unchanged;
        };

        // Extract function type from callee
        // - If core.func, use directly
        // - If closure.closure, extract inner func_type
        let func_type = if let Some(func_ty) = core::Func::from_type(db, callee_ty) {
            func_ty
        } else if let Some(closure_ty) = closure::Closure::from_type(db, callee_ty) {
            let inner_ty = closure_ty.func_type(db);
            let Some(func_ty) = core::Func::from_type(db, inner_ty) else {
                return RewriteResult::Unchanged;
            };
            func_ty
        } else {
            // Callee type is neither core.func nor closure.closure
            debug!(
                "boxing: call_indirect callee has unexpected type {}.{}",
                callee_ty.dialect(db),
                callee_ty.name(db)
            );
            return RewriteResult::Unchanged;
        };

        let param_types = func_type.params(db);
        let return_ty = func_type.result(db);

        // Check if any boxing is needed
        let needs_boxing = param_types.iter().any(|ty| tribute::is_type_var(db, *ty));
        let needs_unboxing = tribute::is_type_var(db, return_ty);

        if !needs_boxing && !needs_unboxing {
            return RewriteResult::Unchanged;
        }

        let location = op.location(db);
        let args = &operands[1..]; // Arguments (excluding callee)

        // Verify argument count matches parameter count
        if args.len() != param_types.len() {
            debug!(
                "boxing: call_indirect operand count mismatch: {} args, {} params",
                args.len(),
                param_types.len()
            );
            return RewriteResult::Unchanged;
        }

        let mut result_ops: Vec<Operation<'db>> = Vec::new();
        let mut new_args: Vec<Value<'db>> = Vec::new();

        // Track the concrete types of arguments for type_var positions
        let mut type_var_to_concrete: HashMap<u64, Type<'db>> = HashMap::new();

        // Insert boxing for each argument that needs it
        for (i, param_ty) in param_types.iter().enumerate() {
            let arg = args[i];

            if tribute::is_type_var(db, *param_ty) {
                // Get the concrete type of this argument
                // Note: adaptor.operand_type uses 0-based index including callee,
                // so arg at index i corresponds to operand index i+1
                let arg_ty = adaptor.operand_type(i + 1);

                // Track for later unboxing inference
                if let Some(ty) = arg_ty
                    && let Some(type_var_id) = get_type_var_id(db, *param_ty)
                {
                    type_var_to_concrete.insert(type_var_id, ty);
                }

                // Need to box this argument
                if let Some(ty) = arg_ty
                    && let Some((box_op, boxed_val)) = create_box_op(db, arg, ty, location)
                {
                    result_ops.push(box_op);
                    new_args.push(boxed_val);
                } else {
                    // Can't determine type to box, pass through
                    new_args.push(arg);
                }
            } else {
                new_args.push(arg);
            }
        }

        // Determine call result type
        let call_result_ty = if needs_unboxing {
            *tribute_rt::Any::new(db)
        } else {
            op.results(db).first().copied().unwrap_or(return_ty)
        };

        // Create new call_indirect with boxed arguments
        let new_call = func::call_indirect(db, location, callee, new_args, call_result_ty);
        let new_call_op = new_call.as_operation();
        result_ops.push(new_call_op);

        // Insert unboxing if needed
        if needs_unboxing {
            let call_result = new_call_op.result(db, 0);

            // Get the expected concrete type
            let original_result_ty = op.results(db).first().copied();

            debug!(
                "boxing: call_indirect needs_unboxing, original_result_ty: {:?}",
                original_result_ty
            );

            let unbox_ty = if let Some(orig_ty) = original_result_ty {
                if !tribute::is_type_var(db, orig_ty) {
                    // typeck resolved it to a concrete type
                    Some(orig_ty)
                } else if let Some(type_var_id) = get_type_var_id(db, return_ty) {
                    // Return type is a type_var, look up concrete type from arguments
                    type_var_to_concrete.get(&type_var_id).copied()
                } else {
                    None
                }
            } else {
                None
            };

            debug!("boxing: call_indirect inferred unbox_ty: {:?}", unbox_ty);

            if let Some(target_ty) = unbox_ty
                && let Some((unbox_ops, _unboxed_val)) =
                    create_unbox_op(db, call_result, target_ty, location)
            {
                debug!(
                    "boxing: call_indirect inserted {} unbox ops for type {:?}",
                    unbox_ops.len(),
                    target_ty
                );
                result_ops.extend(unbox_ops);
            }
        }

        RewriteResult::Expand(result_ops)
    }

    fn name(&self) -> &'static str {
        "BoxCallIndirectPattern"
    }
}

// =============================================================================
// Public API
// =============================================================================

/// Insert explicit boxing/unboxing operations for polymorphic function calls.
pub fn insert_boxing<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    // First pass: collect function signatures
    let func_types = collect_func_types(db, &module);

    // Second pass: apply boxing patterns
    // No specific conversion target - this is an optimization pass, not a dialect lowering
    let target = ConversionTarget::new();
    PatternApplicator::new(TypeConverter::new())
        .add_pattern(BoxCallPattern { func_types })
        .add_pattern(BoxCallIndirectPattern)
        .apply_partial(db, module, target)
        .module
}

#[cfg(test)]
mod tests {
    // TODO: Add tests for boxing insertion
}
