//! Concretize types in wasm dialect operations.
//!
//! This pass resolves placeholder types (`tribute.type_var`) to concrete types
//! before the emit phase. This allows emit to be a simple 1:1 translation
//! without runtime type inference.
//!
//! ## What this pass does
//!
//! **Replace `type_var` in operation results** with concrete types:
//! - `wasm.call`: Use callee's return type from function signature
//! - `wasm.call_indirect`: Use enclosing function's return type as hint
//! - `wasm.if`/`wasm.block`/`wasm.loop`: Infer from branch result types
//!
//! Note: Boxing/unboxing of values at polymorphic call sites is currently
//! handled in the emit phase (`value_emission.rs`). Future work may move
//! this logic here as explicit IR operations.

use std::collections::HashMap;

use tracing::debug;
use tribute_ir::ModulePathExt;
use tribute_ir::dialect::{adt, closure, tribute};
use trunk_ir::Attribute;
use trunk_ir::dialect::{cont, core, wasm};
use trunk_ir::rewrite::{OpAdaptor, PatternApplicator, RewritePattern, RewriteResult};
use trunk_ir::{DialectOp, DialectType, IdVec, Operation, Region, Symbol, Type, Value, ValueDef};

use crate::type_converter::wasm_type_converter;

/// Concretize types in wasm operations.
///
/// This pass runs after `tribute_rt_to_wasm` and before emit.
pub fn lower<'db>(db: &'db dyn salsa::Database, module: core::Module<'db>) -> core::Module<'db> {
    // First pass: collect function signatures for type lookup
    let func_return_types = collect_func_return_types(db, module);

    let applicator = PatternApplicator::new(wasm_type_converter())
        .add_pattern(CallResultTypePattern {
            func_return_types: func_return_types.clone(),
        })
        .add_pattern(CallIndirectResultTypePattern {
            func_return_types: func_return_types.clone(),
        })
        .add_pattern(StructGetResultTypePattern)
        .add_pattern(IfResultTypePattern)
        .add_pattern(BlockResultTypePattern)
        .add_pattern(LoopResultTypePattern);
    applicator.apply(db, module).module
}

/// Collect function return types from the module.
fn collect_func_return_types<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> HashMap<Symbol, Type<'db>> {
    let mut func_return_types = HashMap::new();
    collect_func_return_types_from_region(db, module.body(db), &mut func_return_types);
    func_return_types
}

/// Recursively collect function return types from a region, including nested modules.
fn collect_func_return_types_from_region<'db>(
    db: &'db dyn salsa::Database,
    region: Region<'db>,
    func_return_types: &mut HashMap<Symbol, Type<'db>>,
) {
    for block in region.blocks(db).iter() {
        for op in block.operations(db).iter() {
            // Check for wasm.func operations
            if let Ok(func_op) = wasm::Func::from_operation(db, *op) {
                let sym_name = func_op.sym_name(db);
                let func_ty = func_op.r#type(db);

                // Extract return type from function type
                if let Some(func) = core::Func::from_type(db, func_ty) {
                    let return_ty = func.result(db);
                    debug!(
                        "wasm_type_concrete: registered function {} -> {}.{}",
                        sym_name,
                        return_ty.dialect(db),
                        return_ty.name(db)
                    );
                    func_return_types.insert(sym_name, return_ty);
                }
            }

            // Recursively collect from nested modules
            if core::Module::matches(db, *op) {
                for nested_region in op.regions(db).iter() {
                    collect_func_return_types_from_region(db, *nested_region, func_return_types);
                }
            }
        }
    }
}

/// Pattern to concretize result types of wasm.call operations.
///
/// If a call's result type is `tribute.type_var`, replace it with
/// the callee's declared return type.
struct CallResultTypePattern<'db> {
    /// Map of function name -> return type.
    func_return_types: HashMap<Symbol, Type<'db>>,
}

impl<'db> RewritePattern<'db> for CallResultTypePattern<'db> {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        // Only handle wasm.call operations
        let Ok(call_op) = wasm::Call::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        // Look up the callee's return type
        // First try qualified name (e.g., "Point::x"), then fall back to last segment (e.g., "x")
        let callee = call_op.callee(db);
        let return_ty = if let Some(&ty) = self.func_return_types.get(&callee) {
            ty
        } else if let Some(&ty) = self.func_return_types.get(&callee.last_segment()) {
            ty
        } else {
            debug!(
                "wasm_type_concrete: callee {} not found in func_return_types",
                callee
            );
            return RewriteResult::Unchanged;
        };

        // If callee returns type_var (generic function), try to infer concrete type from operands
        let concrete_return_ty = if tribute::is_type_var(db, return_ty) {
            // Try to infer from first operand (works for identity-like functions)
            if let Some(&operand) = op.operands(db).first() {
                if let Some(operand_ty) = infer_operand_type(db, operand) {
                    if !tribute::is_type_var(db, operand_ty)
                        && !tribute::is_placeholder_type(db, operand_ty)
                    {
                        debug!(
                            "wasm_type_concrete: callee {} returns type_var, inferring from operand: {}.{}",
                            callee,
                            operand_ty.dialect(db),
                            operand_ty.name(db)
                        );
                        operand_ty
                    } else {
                        debug!(
                            "wasm_type_concrete: callee {} returns type_var, operand also placeholder",
                            callee
                        );
                        return RewriteResult::Unchanged;
                    }
                } else {
                    debug!(
                        "wasm_type_concrete: callee {} returns type_var, cannot infer operand type",
                        callee
                    );
                    return RewriteResult::Unchanged;
                }
            } else {
                debug!(
                    "wasm_type_concrete: callee {} returns type_var, no operands to infer from",
                    callee
                );
                return RewriteResult::Unchanged;
            }
        } else {
            return_ty
        };

        // Try to concretize results
        let Some(new_results) = concretize_results(db, op.results(db), concrete_return_ty) else {
            return RewriteResult::Unchanged;
        };

        debug!(
            "wasm_type_concrete: concretizing wasm.call {} result(s) to {}.{}",
            callee,
            concrete_return_ty.dialect(db),
            concrete_return_ty.name(db)
        );

        let new_op = op.modify(db).results(new_results).build();
        RewriteResult::Replace(new_op)
    }
}

/// Pattern to concretize result types of wasm.call_indirect operations.
///
/// If a call_indirect's result type is `tribute.type_var`, try to infer it from:
/// 1. The callee's function type (if it's a known funcref)
/// 2. For `wasm.ref_func` callees, look up the referenced function's return type
struct CallIndirectResultTypePattern<'db> {
    /// Map of function name -> return type.
    func_return_types: HashMap<Symbol, Type<'db>>,
}

impl<'db> RewritePattern<'db> for CallIndirectResultTypePattern<'db> {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        // Only handle wasm.call_indirect operations
        if !wasm::CallIndirect::matches(db, *op) {
            return RewriteResult::Unchanged;
        }

        // The callee is the last operand (funcref)
        let operands = op.operands(db);
        let Some(&callee_val) = operands.last() else {
            return RewriteResult::Unchanged;
        };

        // Try to infer type from callee
        if let Some(concrete_ty) = infer_type_from_callee(db, callee_val, &self.func_return_types)
            && !tribute::is_type_var(db, concrete_ty)
        {
            // Try to concretize results
            let Some(new_results) = concretize_results(db, op.results(db), concrete_ty) else {
                return RewriteResult::Unchanged;
            };

            debug!(
                "wasm_type_concrete: concretizing wasm.call_indirect result(s) to {}.{}",
                concrete_ty.dialect(db),
                concrete_ty.name(db)
            );

            let new_op = op.modify(db).results(new_results).build();
            return RewriteResult::Replace(new_op);
        }

        // Log diagnostic info about why we couldn't infer
        if let ValueDef::OpResult(def_op) = callee_val.def(db) {
            let callee_ty = def_op.results(db).get(callee_val.index(db)).copied();
            debug!(
                "wasm_type_concrete: cannot concretize wasm.call_indirect - callee from {}.{}, type: {:?}",
                def_op.dialect(db),
                def_op.name(db),
                callee_ty.map(|t| format!("{}.{}", t.dialect(db), t.name(db)))
            );
        }

        RewriteResult::Unchanged
    }
}

/// Try to infer the type of an operand value.
///
/// Handles cases like:
/// - Operation results: get the type from the operation's result
/// - Block arguments: get the type from the block argument type
fn infer_operand_type<'db>(db: &'db dyn salsa::Database, value: Value<'db>) -> Option<Type<'db>> {
    match value.def(db) {
        ValueDef::OpResult(def_op) => {
            let index = value.index(db);
            def_op.results(db).get(index).copied()
        }
        ValueDef::BlockArg(block_id) => {
            // Block arguments don't store their types directly in this context,
            // so we return None and let the caller handle it
            let _ = block_id;
            None
        }
    }
}

/// Try to infer the return type from a callee value.
///
/// Handles cases like:
/// - wasm.ref_func: look up the referenced function's return type
/// - Values with core.func type: extract return type from the type
fn infer_type_from_callee<'db>(
    db: &'db dyn salsa::Database,
    callee: Value<'db>,
    func_return_types: &HashMap<Symbol, Type<'db>>,
) -> Option<Type<'db>> {
    match callee.def(db) {
        ValueDef::OpResult(def_op) => {
            // Check if it's a wasm.ref_func operation
            if let Ok(ref_func) = wasm::RefFunc::from_operation(db, def_op) {
                let func_name = ref_func.func_name(db);
                if let Some(&return_ty) = func_return_types.get(&func_name) {
                    return Some(return_ty);
                }
            }

            // Try to get type from the operation's result
            let index = callee.index(db);
            if let Some(callee_ty) = def_op.results(db).get(index).copied() {
                // If it's a function type, extract return type
                if let Some(func_ty) = core::Func::from_type(db, callee_ty) {
                    return Some(func_ty.result(db));
                }
                // If it's a continuation type, extract result type
                if let Some(cont_ty) = cont::Continuation::from_type(db, callee_ty) {
                    return Some(cont_ty.result(db));
                }
                // If it's a closure type, extract return type from the wrapped function type
                if let Some(closure_ty) = closure::Closure::from_type(db, callee_ty) {
                    let func_type = closure_ty.func_type(db);
                    if let Some(func_ty) = core::Func::from_type(db, func_type) {
                        return Some(func_ty.result(db));
                    }
                }
            }

            None
        }
        ValueDef::BlockArg(_) => None,
    }
}

/// Pattern to concretize result types of wasm.if operations.
///
/// If an if's result type is `tribute.type_var`, try to infer it from
/// the yield operations in its then/else branches.
struct IfResultTypePattern;

impl<'db> RewritePattern<'db> for IfResultTypePattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        // Only handle wasm.if operations
        if !wasm::If::matches(db, *op) {
            return RewriteResult::Unchanged;
        }

        // Try to infer concrete type from regions
        let Some(concrete_ty) = infer_type_from_regions(db, op.regions(db)) else {
            return RewriteResult::Unchanged;
        };

        // Try to concretize results
        let Some(new_results) = concretize_results(db, op.results(db), concrete_ty) else {
            return RewriteResult::Unchanged;
        };

        debug!(
            "wasm_type_concrete: concretizing wasm.if result(s) to {}.{}",
            concrete_ty.dialect(db),
            concrete_ty.name(db)
        );

        let new_op = op.modify(db).results(new_results).build();
        RewriteResult::Replace(new_op)
    }
}

/// Pattern to concretize result types of wasm.block operations.
struct BlockResultTypePattern;

impl<'db> RewritePattern<'db> for BlockResultTypePattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        // Only handle wasm.block operations
        if !wasm::Block::matches(db, *op) {
            return RewriteResult::Unchanged;
        }

        // Try to infer concrete type from the body region
        let Some(concrete_ty) = infer_type_from_regions(db, op.regions(db)) else {
            return RewriteResult::Unchanged;
        };

        // Try to concretize results
        let Some(new_results) = concretize_results(db, op.results(db), concrete_ty) else {
            return RewriteResult::Unchanged;
        };

        debug!(
            "wasm_type_concrete: concretizing wasm.block result(s) to {}.{}",
            concrete_ty.dialect(db),
            concrete_ty.name(db)
        );

        let new_op = op.modify(db).results(new_results).build();
        RewriteResult::Replace(new_op)
    }
}

/// Pattern to concretize result types of wasm.loop operations.
struct LoopResultTypePattern;

impl<'db> RewritePattern<'db> for LoopResultTypePattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        // Only handle wasm.loop operations
        if !wasm::Loop::matches(db, *op) {
            return RewriteResult::Unchanged;
        }

        // Try to infer concrete type from the body region
        let Some(concrete_ty) = infer_type_from_regions(db, op.regions(db)) else {
            return RewriteResult::Unchanged;
        };

        // Try to concretize results
        let Some(new_results) = concretize_results(db, op.results(db), concrete_ty) else {
            return RewriteResult::Unchanged;
        };

        debug!(
            "wasm_type_concrete: concretizing wasm.loop result(s) to {}.{}",
            concrete_ty.dialect(db),
            concrete_ty.name(db)
        );

        let new_op = op.modify(db).results(new_results).build();
        RewriteResult::Replace(new_op)
    }
}

/// Pattern to concretize result types of wasm.struct_get operations.
///
/// If a struct_get's result type is `tribute.type_var`, try to infer it from
/// the operand's struct type and field index.
struct StructGetResultTypePattern;

impl<'db> RewritePattern<'db> for StructGetResultTypePattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        // Only handle wasm.struct_get operations
        if !wasm::StructGet::matches(db, *op) {
            return RewriteResult::Unchanged;
        }

        // Check if result is already concrete
        let results = op.results(db);
        let has_type_var = results.iter().any(|ty| tribute::is_type_var(db, *ty));
        if !has_type_var {
            return RewriteResult::Unchanged;
        }

        // Get field index from field_idx attribute
        let attrs = op.attributes(db);
        let field_idx = match attrs.get(&Symbol::new("field_idx")) {
            Some(Attribute::IntBits(idx)) => *idx as usize,
            // Fallback to field attribute (from adt.struct_get/variant_get)
            _ => match attrs.get(&Symbol::new("field")) {
                Some(Attribute::IntBits(idx)) => *idx as usize,
                _ => return RewriteResult::Unchanged,
            },
        };

        // Get operand type (the struct reference)
        let Some(ref_type) = adaptor.operand_type(0) else {
            return RewriteResult::Unchanged;
        };

        // Try to get field type from the struct type
        let field_type = get_field_type_from_struct(db, ref_type, field_idx)
            // Try type attribute as fallback
            .or_else(|| {
                if let Some(Attribute::Type(struct_ty)) = attrs.get(&Symbol::new("type")) {
                    get_field_type_from_struct(db, *struct_ty, field_idx)
                } else {
                    None
                }
            });

        let Some(concrete_ty) = field_type else {
            debug!(
                "wasm_type_concrete: cannot infer struct_get field type for field_idx {}, ref_type={}.{}",
                field_idx,
                ref_type.dialect(db),
                ref_type.name(db)
            );
            return RewriteResult::Unchanged;
        };

        // Skip if inferred type is also a type_var
        if tribute::is_type_var(db, concrete_ty) {
            return RewriteResult::Unchanged;
        }

        // Concretize results
        let Some(new_results) = concretize_results(db, results, concrete_ty) else {
            return RewriteResult::Unchanged;
        };

        debug!(
            "wasm_type_concrete: concretizing wasm.struct_get result to {}.{}",
            concrete_ty.dialect(db),
            concrete_ty.name(db)
        );

        let new_op = op.modify(db).results(new_results).build();
        RewriteResult::Replace(new_op)
    }
}

/// Get field type from a struct or variant instance type by field index.
fn get_field_type_from_struct<'db>(
    db: &'db dyn salsa::Database,
    struct_ty: Type<'db>,
    field_idx: usize,
) -> Option<Type<'db>> {
    // 1. Check if it's a variant instance type (from adt.variant_cast)
    if adt::is_variant_instance_type(db, struct_ty) {
        // Get base enum type and tag
        let base_enum = adt::get_base_enum(db, struct_ty)?;
        let tag = adt::get_variant_tag(db, struct_ty)?;

        // Get variants from base enum
        let variants = adt::get_enum_variants(db, base_enum)?;

        // Find the variant with matching tag
        for (variant_name, field_types) in variants {
            if variant_name == tag {
                return field_types.get(field_idx).copied();
            }
        }
        return None;
    }

    // 2. Check if it's a direct adt.struct type
    if adt::is_struct_type(db, struct_ty) {
        let fields_attr = struct_ty.get_attr(db, adt::ATTR_FIELDS())?;
        let Attribute::List(fields) = fields_attr else {
            return None;
        };

        // Each field is [name, type]
        let field = fields.get(field_idx)?;
        let Attribute::List(field_parts) = field else {
            return None;
        };

        // Get type from [name, type] pair
        if field_parts.len() >= 2
            && let Attribute::Type(ty) = &field_parts[1]
        {
            return Some(*ty);
        }
        return None;
    }

    // 3. Check variant_fields attribute (legacy path)
    if let Some(field_types) = adt::get_variant_field_types(db, struct_ty) {
        return field_types.get(field_idx).copied();
    }

    None
}

// ============================================================================
// Helper functions
// ============================================================================

/// Replace all `type_var` results with a concrete type.
///
/// Returns the modified results, or None if no changes were needed.
fn concretize_results<'db>(
    db: &'db dyn salsa::Database,
    results: &IdVec<Type<'db>>,
    concrete_ty: Type<'db>,
) -> Option<IdVec<Type<'db>>> {
    // Check if any result is a type_var
    let has_type_var = results.iter().any(|ty| tribute::is_type_var(db, *ty));
    if !has_type_var {
        return None;
    }

    // Replace all type_var results with the concrete type
    let new_results: IdVec<Type<'db>> = results
        .iter()
        .map(|ty| {
            if tribute::is_type_var(db, *ty) {
                concrete_ty
            } else {
                *ty
            }
        })
        .collect();

    Some(new_results)
}

/// Try to infer a concrete type from regions by looking at yield operations.
///
/// For control flow operations like `wasm.if`, all branches should yield the same type.
/// This function validates type agreement across regions and returns None if they disagree.
fn infer_type_from_regions<'db>(
    db: &'db dyn salsa::Database,
    regions: &IdVec<Region<'db>>,
) -> Option<Type<'db>> {
    let mut found: Option<Type<'db>> = None;

    for region in regions.iter() {
        let Some(ty) = infer_type_from_region(db, region) else {
            continue;
        };

        // Skip placeholder types (type_var, unresolved type, error_type) - we want concrete types
        if tribute::is_placeholder_type(db, ty) {
            continue;
        }

        match found {
            None => found = Some(ty),
            Some(prev) if prev == ty => {
                // Types agree, continue
            }
            Some(prev) => {
                // Type disagreement across regions - this indicates a type error
                // that should have been caught earlier in the pipeline.
                debug_assert!(
                    false,
                    "infer_type_from_regions: type disagreement between branches: {}.{} vs {}.{}",
                    prev.dialect(db),
                    prev.name(db),
                    ty.dialect(db),
                    ty.name(db)
                );
                // In release builds, return None to avoid making assumptions
                return None;
            }
        }
    }

    found
}

/// Try to infer a concrete type from a region's yield operations.
fn infer_type_from_region<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
) -> Option<Type<'db>> {
    for block in region.blocks(db).iter() {
        for op in block.operations(db).iter() {
            // Look for wasm.yield operations
            if let Ok(yield_op) = wasm::Yield::from_operation(db, *op) {
                let yielded_value = yield_op.value(db);
                if let Some(ty) = get_value_type(db, yielded_value)
                    && !tribute::is_placeholder_type(db, ty)
                {
                    return Some(ty);
                }
            }
        }
    }
    None
}

/// Get the type of a value from its definition.
fn get_value_type<'db>(db: &'db dyn salsa::Database, value: Value<'db>) -> Option<Type<'db>> {
    match value.def(db) {
        ValueDef::OpResult(def_op) => {
            let index = value.index(db);
            def_op.results(db).get(index).copied()
        }
        ValueDef::BlockArg(_block_id) => {
            // For block arguments, we'd need the block's argument types
            // which we don't easily have access to here.
            // Return None for now - these cases are less common.
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::{Block, BlockId, Location, PathId, Region, Span, idvec};

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    #[salsa::tracked]
    fn make_and_lower_module_with_type_var_call(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();
        let type_var = tribute::type_var_with_id(db, 0);

        // Create a function that returns i32
        let func_ty = core::Func::new(db, idvec![], i32_ty).as_type();
        let func_body_block = Block::new(db, BlockId::fresh(), location, idvec![], idvec![]);
        let func_body = Region::new(db, location, idvec![func_body_block]);
        let func_op = wasm::func(db, location, Symbol::new("callee"), func_ty, func_body);

        // Create a call to that function with type_var result
        let call_op = wasm::call(db, location, None, Some(type_var), Symbol::new("callee"));

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![func_op.as_operation(), call_op.as_operation()],
        );
        let region = Region::new(db, location, idvec![block]);
        let module = core::Module::create(db, location, "test".into(), region);

        // Lower the module within the tracked function
        lower(db, module)
    }

    #[salsa_test]
    fn test_call_result_type_concretization(db: &salsa::DatabaseImpl) {
        let lowered = make_and_lower_module_with_type_var_call(db);

        // Find the call operation
        let block = lowered.body(db).blocks(db).first().unwrap();
        let call_op = block
            .operations(db)
            .iter()
            .find(|op| op.dialect(db) == wasm::DIALECT_NAME() && op.name(db) == wasm::CALL())
            .expect("call operation not found");

        // Check that result type is now i32, not type_var
        let result_ty = call_op.results(db).first().copied().unwrap();
        assert!(
            !tribute::is_type_var(db, result_ty),
            "result type should be concrete, not type_var"
        );
        assert!(
            core::I32::from_type(db, result_ty).is_some(),
            "result type should be i32"
        );
    }
}
