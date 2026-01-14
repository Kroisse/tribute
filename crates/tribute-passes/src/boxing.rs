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
use tribute_ir::dialect::{tribute, tribute_rt};
use trunk_ir::dialect::core::{self, F64 as CoreF64, I32 as CoreI32, Module};
use trunk_ir::dialect::func;
use trunk_ir::rewrite::{
    OpAdaptor, PatternApplicator, RewritePattern, RewriteResult, TypeConverter,
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

/// Get the type of a value from its definition.
fn get_value_type<'db>(db: &'db dyn salsa::Database, value: Value<'db>) -> Option<Type<'db>> {
    match value.def(db) {
        trunk_ir::ValueDef::OpResult(def_op) => def_op.results(db).get(value.index(db)).copied(),
        trunk_ir::ValueDef::BlockArg(_block_id) => {
            // Block argument types need context - handled by OpAdaptor
            None
        }
    }
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

/// Create a boxing operation for a value.
/// Returns (box_op, boxed_value) or None if boxing not applicable.
fn create_box_op<'db>(
    db: &'db dyn salsa::Database,
    value: Value<'db>,
    location: Location<'db>,
) -> Option<(Operation<'db>, Value<'db>)> {
    let value_ty = get_value_type(db, value)?;

    let any_ty = *tribute_rt::Any::new(db);

    // Check for tribute_rt types AND core types
    let is_int_like = tribute_rt::is_int(db, value_ty)
        || tribute_rt::is_nat(db, value_ty)
        || tribute_rt::is_bool(db, value_ty)
        || CoreI32::from_type(db, value_ty).is_some();
    let is_float_like =
        tribute_rt::is_float(db, value_ty) || CoreF64::from_type(db, value_ty).is_some();

    if is_int_like {
        let box_op = tribute_rt::box_int(db, location, value, any_ty);
        let boxed = box_op.as_operation().result(db, 0);
        Some((box_op.as_operation(), boxed))
    } else if is_float_like {
        let box_op = tribute_rt::box_float(db, location, value, any_ty);
        let boxed = box_op.as_operation().result(db, 0);
        Some((box_op.as_operation(), boxed))
    } else {
        // Reference types don't need boxing (already subtypes of anyref)
        None
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
    // Check for tribute_rt types AND core types
    let is_int_like = tribute_rt::is_int(db, target_ty)
        || tribute_rt::is_nat(db, target_ty)
        || tribute_rt::is_bool(db, target_ty)
        || CoreI32::from_type(db, target_ty).is_some();
    let is_float_like =
        tribute_rt::is_float(db, target_ty) || CoreF64::from_type(db, target_ty).is_some();

    if is_int_like {
        // Emit unbox_int - tribute_rt_to_wasm will add ref_cast to i31ref
        let unbox_op = tribute_rt::unbox_int(db, location, value, target_ty);
        let unboxed = unbox_op.as_operation().result(db, 0);
        Some((vec![unbox_op.as_operation()], unboxed))
    } else if is_float_like {
        // Emit unbox_float - tribute_rt_to_wasm will add ref_cast to BoxedF64
        let unbox_op = tribute_rt::unbox_float(db, location, value, target_ty);
        let unboxed = unbox_op.as_operation().result(db, 0);
        Some((vec![unbox_op.as_operation()], unboxed))
    } else {
        // Reference types don't need unboxing
        None
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

        // Insert boxing for each operand that needs it
        for (i, param_ty) in param_types.iter().enumerate() {
            let Some(operand) = operands.get(i).copied() else {
                continue;
            };

            if tribute::is_type_var(db, *param_ty) {
                // Get the concrete type of this operand for later unboxing inference
                if let Some(operand_ty) = adaptor.get_value_type(db, operand)
                    && let Some(type_var_id) = get_type_var_id(db, *param_ty)
                {
                    type_var_to_concrete.insert(type_var_id, operand_ty);
                }

                // Need to box this operand
                if let Some((box_op, boxed_val)) = create_box_op(db, operand, location) {
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

// =============================================================================
// Public API
// =============================================================================

/// Insert explicit boxing/unboxing operations for polymorphic function calls.
pub fn insert_boxing<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    // First pass: collect function signatures
    let func_types = collect_func_types(db, &module);

    // Second pass: apply boxing patterns
    PatternApplicator::new(TypeConverter::new())
        .add_pattern(BoxCallPattern { func_types })
        .apply(db, module)
        .module
}

#[cfg(test)]
mod tests {
    // TODO: Add tests for boxing insertion
}
