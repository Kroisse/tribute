//! Concretize types in wasm dialect operations.
//!
//! This pass resolves placeholder types to concrete types before the emit phase.
//! This allows emit to be a simple 1:1 translation without runtime type inference.
//!
//! ## What this pass does
//!
//! **Replace placeholder types in operation results** with concrete types:
//! - `wasm.call`: Use callee's return type from function signature
//! - `wasm.call_indirect`: Infer from callee's function type
//! - `wasm.if`/`wasm.block`/`wasm.loop`: Infer from branch result types
//! - `wasm.struct_get`: Infer from struct field type
//!
//! ## Current Status
//!
//! Type variables are now resolved at the AST level and converted to `tribute_rt.any`
//! or concrete types in `ast_to_ir`. As a result, the type_var-specific patterns
//! have been removed. This pass now primarily handles placeholder type resolution
//! for unresolved type references.

use std::collections::HashMap;

use tracing::debug;
use tribute_ir::ModulePathExt;
use tribute_ir::dialect::{closure, tribute, tribute_rt};
use trunk_ir::Attribute;
use trunk_ir::dialect::adt;
use trunk_ir::dialect::{cont, core, wasm};
use trunk_ir::rewrite::{
    ConversionTarget, OpAdaptor, PatternApplicator, RewritePattern, RewriteResult,
};
use trunk_ir::{DialectOp, DialectType, IdVec, Operation, Region, Symbol, Type, Value, ValueDef};

use super::type_converter::wasm_type_converter;

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
    // No specific conversion target - type concretization is an optimization pass
    let target = ConversionTarget::new();
    applicator.apply_partial(db, module, target).module
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
/// If a call's result type is a placeholder, replace it with
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

        // Check if result type needs concretization
        let results = op.results(db);
        if !results
            .iter()
            .any(|ty| tribute::is_placeholder_type(db, *ty))
        {
            return RewriteResult::Unchanged;
        }

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

        // Skip if the return type is also a placeholder (generic function not yet resolved)
        if tribute::is_placeholder_type(db, return_ty) {
            debug!(
                "wasm_type_concrete: callee {} returns placeholder type",
                callee
            );
            return RewriteResult::Unchanged;
        }

        // Try to concretize results
        let Some(new_results) = concretize_results(db, results, return_ty) else {
            return RewriteResult::Unchanged;
        };

        debug!(
            "wasm_type_concrete: concretizing wasm.call {} result(s) to {}.{}",
            callee,
            return_ty.dialect(db),
            return_ty.name(db)
        );

        let new_op = op.modify(db).results(new_results).build();
        RewriteResult::Replace(new_op)
    }
}

/// Pattern to concretize result types of wasm.call_indirect operations.
///
/// If a call_indirect's result type is a placeholder, try to infer it from
/// the callee's function type (wasm.ref_func, core.func, continuation, closure).
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

        // Check if result type needs concretization
        let results = op.results(db);
        if !results
            .iter()
            .any(|ty| tribute::is_placeholder_type(db, *ty))
        {
            return RewriteResult::Unchanged;
        }

        // The callee is the first operand (funcref) in our IR convention
        // Note: WebAssembly stack order differs from IR operand order
        let operands = op.operands(db);
        let Some(&callee_val) = operands.first() else {
            return RewriteResult::Unchanged;
        };

        // Try to infer type from callee
        let Some(concrete_ty) = infer_type_from_callee(db, callee_val, &self.func_return_types)
        else {
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
            return RewriteResult::Unchanged;
        };

        // Skip if inferred type is also a placeholder
        if tribute::is_placeholder_type(db, concrete_ty) {
            debug!("wasm_type_concrete: wasm.call_indirect callee returns placeholder type");
            return RewriteResult::Unchanged;
        }

        // Try to concretize results
        let Some(new_results) = concretize_results(db, results, concrete_ty) else {
            return RewriteResult::Unchanged;
        };

        debug!(
            "wasm_type_concrete: concretizing wasm.call_indirect result(s) to {}.{}",
            concrete_ty.dialect(db),
            concrete_ty.name(db)
        );

        let new_op = op.modify(db).results(new_results).build();
        RewriteResult::Replace(new_op)
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
/// If an if's result type is a placeholder, try to infer it from
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

        // If result type is already nil (void), branches use terminators (break/continue)
        // and don't yield values, so skip type inference
        if let Some(result_ty) = op.results(db).first()
            && core::Nil::from_type(db, *result_ty).is_some()
        {
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
/// If a struct_get's result type is a placeholder, try to infer it from
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
        let has_placeholder = results
            .iter()
            .any(|ty| tribute::is_placeholder_type(db, *ty));
        if !has_placeholder {
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

        // Skip if inferred type is also a placeholder
        if tribute::is_placeholder_type(db, concrete_ty) {
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

/// Replace all placeholder results with a concrete type.
///
/// Returns the modified results, or None if no changes were needed.
fn concretize_results<'db>(
    db: &'db dyn salsa::Database,
    results: &IdVec<Type<'db>>,
    concrete_ty: Type<'db>,
) -> Option<IdVec<Type<'db>>> {
    // Check if any result is a placeholder
    let has_placeholder = results
        .iter()
        .any(|ty| tribute::is_placeholder_type(db, *ty));
    if !has_placeholder {
        return None;
    }

    // Replace all placeholder results with the concrete type
    let new_results: IdVec<Type<'db>> = results
        .iter()
        .map(|ty| {
            if tribute::is_placeholder_type(db, *ty) {
                concrete_ty
            } else {
                *ty
            }
        })
        .collect();

    Some(new_results)
}

/// Check if two types are both reference-like and will be lowered to the same WASM type.
///
/// This handles cases where different dialects represent the same underlying type:
/// - `core.ptr`, `tribute_rt.any`, `wasm.anyref` all map to `anyref` in WASM
fn are_reference_compatible<'db>(
    db: &'db dyn salsa::Database,
    ty1: Type<'db>,
    ty2: Type<'db>,
) -> bool {
    fn is_anyref_like<'db>(db: &'db dyn salsa::Database, ty: Type<'db>) -> bool {
        // core.ptr maps to anyref
        if core::Ptr::from_type(db, ty).is_some() {
            return true;
        }
        // tribute_rt.any maps to anyref
        if tribute_rt::is_any(db, ty) {
            return true;
        }
        // wasm.anyref
        if wasm::Anyref::from_type(db, ty).is_some() {
            return true;
        }
        // wasm.structref is a subtype of anyref
        if wasm::Structref::from_type(db, ty).is_some() {
            return true;
        }
        false
    }

    is_anyref_like(db, ty1) && is_anyref_like(db, ty2)
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

        // Skip placeholder types - we want concrete types
        if tribute::is_placeholder_type(db, ty) {
            continue;
        }

        match found {
            None => {
                found = Some(ty);
            }
            Some(prev) if prev == ty => {
                // Types agree, continue
            }
            Some(prev) if are_reference_compatible(db, prev, ty) => {
                // Both types are reference-like and will be lowered to anyref
                // Keep the first one
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
    // Tests removed as they depended on tribute::type_var_with_id which no longer exists.
    // Type variables are now resolved at AST level before IR generation.
}
