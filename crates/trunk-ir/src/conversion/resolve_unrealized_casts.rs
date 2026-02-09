//! Cast resolution pass for eliminating `unrealized_conversion_cast` operations.
//!
//! After dialect conversion, `unrealized_conversion_cast` operations may remain
//! in the IR as placeholders for type mismatches. This pass resolves them by:
//! 1. Finding all `unrealized_conversion_cast` operations
//! 2. Using the TypeConverter's materialization functions to generate actual conversion code
//! 3. Replacing the casts with the materialized operations
//!
//! If any casts cannot be resolved, the pass returns an error listing the unresolved casts.

use crate::dialect::core::{self, Module};
use crate::rewrite::TypeConverter;
use crate::{Block, IdVec, Location, Operation, Region, Type, Value};

/// Error when cast resolution fails.
#[derive(Debug, Clone)]
pub struct UnresolvedCastError<'db> {
    /// List of casts that could not be resolved.
    pub unresolved: Vec<UnresolvedCast<'db>>,
}

/// Information about an unresolved cast.
#[derive(Debug, Clone)]
pub struct UnresolvedCast<'db> {
    /// Location of the cast operation.
    pub location: Location<'db>,
    /// Source type (type of the input value).
    pub from_type: Type<'db>,
    /// Target type (type of the cast result).
    pub to_type: Type<'db>,
}

impl std::fmt::Display for UnresolvedCastError<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Failed to resolve {} unrealized_conversion_cast operation(s):",
            self.unresolved.len()
        )?;
        for cast in &self.unresolved {
            writeln!(
                f,
                "  - {:?} -> {:?} at {:?}",
                cast.from_type, cast.to_type, cast.location
            )?;
        }
        Ok(())
    }
}

impl std::error::Error for UnresolvedCastError<'_> {}

/// Result of resolving casts in a module.
#[derive(Debug)]
pub struct ResolveResult<'db> {
    /// The transformed module with casts resolved.
    pub module: Module<'db>,
    /// Number of casts that were resolved.
    pub resolved_count: usize,
    /// Casts that could not be resolved (empty if all were resolved).
    pub unresolved: Vec<UnresolvedCast<'db>>,
}

/// Resolve all `unrealized_conversion_cast` operations in a module.
///
/// Uses the provided TypeConverter's materialization functions to generate
/// actual conversion operations.
///
/// Always returns a `ResolveResult` with the partially-resolved module.
/// Check `result.unresolved` to see if any casts remain unresolved.
///
/// # Example
///
/// ```ignore
/// let type_converter = TypeConverter::new()
///     .add_materialization(|db, loc, value, from_ty, to_ty| {
///         // Generate actual conversion operations
///         MaterializeResult::single(my_cast_op)
///     });
///
/// let result = resolve_unrealized_casts(db, module, &type_converter);
/// assert_eq!(result.resolved_count, 2);
/// assert!(result.unresolved.is_empty());
/// ```
pub fn resolve_unrealized_casts<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
    type_converter: &TypeConverter,
) -> ResolveResult<'db> {
    tracing::debug!("resolve_unrealized_casts: starting resolution for module");
    let mut resolver = CastResolver::new(db, type_converter);

    // First, collect block argument types from the entire module
    resolver.collect_block_arg_types(&module.body(db));
    tracing::debug!(
        "resolve_unrealized_casts: collected {} block arg types",
        resolver.block_arg_types.len()
    );

    let new_body = resolver.resolve_region(&module.body(db));
    tracing::debug!(
        "resolve_unrealized_casts: resolved {} casts, {} unresolved",
        resolver.resolved_count,
        resolver.unresolved.len()
    );

    let new_module = Module::create(db, module.location(db), module.name(db), new_body);
    ResolveResult {
        module: new_module,
        resolved_count: resolver.resolved_count,
        unresolved: resolver.unresolved,
    }
}

/// Internal resolver state.
struct CastResolver<'db, 'a> {
    db: &'db dyn salsa::Database,
    type_converter: &'a TypeConverter,
    /// Value mappings from cast results to materialized values.
    value_map: std::collections::HashMap<Value<'db>, Value<'db>>,
    /// Maps block arguments to their types.
    /// Key is (BlockId, argument index).
    block_arg_types: std::collections::HashMap<(crate::BlockId, usize), Type<'db>>,
    /// Unresolved casts.
    unresolved: Vec<UnresolvedCast<'db>>,
    /// Count of successfully resolved casts.
    resolved_count: usize,
}

impl<'db, 'a> CastResolver<'db, 'a> {
    fn new(db: &'db dyn salsa::Database, type_converter: &'a TypeConverter) -> Self {
        Self {
            db,
            type_converter,
            value_map: std::collections::HashMap::new(),
            block_arg_types: std::collections::HashMap::new(),
            unresolved: Vec::new(),
            resolved_count: 0,
        }
    }

    /// Collect block argument types from a region (recursively).
    fn collect_block_arg_types(&mut self, region: &Region<'db>) {
        for block in region.blocks(self.db).iter() {
            let block_id = block.id(self.db);
            for (idx, arg) in block.args(self.db).iter().enumerate() {
                self.block_arg_types
                    .insert((block_id, idx), arg.ty(self.db));
            }
            // Recurse into nested regions
            for op in block.operations(self.db).iter() {
                for nested_region in op.regions(self.db).iter() {
                    self.collect_block_arg_types(nested_region);
                }
            }
        }
    }

    fn resolve_region(&mut self, region: &Region<'db>) -> Region<'db> {
        let new_blocks: IdVec<Block<'db>> = region
            .blocks(self.db)
            .iter()
            .map(|block| self.resolve_block(block))
            .collect();
        Region::new(self.db, region.location(self.db), new_blocks)
    }

    fn resolve_block(&mut self, block: &Block<'db>) -> Block<'db> {
        let mut new_ops = Vec::new();

        for op in block.operations(self.db).iter() {
            let resolved_ops = self.resolve_operation(op);
            new_ops.extend(resolved_ops);
        }

        Block::new(
            self.db,
            block.id(self.db),
            block.location(self.db),
            block.args(self.db).clone(),
            new_ops.into_iter().collect(),
        )
    }

    fn resolve_operation(&mut self, op: &Operation<'db>) -> Vec<Operation<'db>> {
        // First, remap operands from previously resolved casts
        let remapped_op = self.remap_operands(op);

        // Map old results to remapped results BEFORE resolving casts
        // This ensures that subsequent operations referencing old results
        // will be remapped to the new operation's results
        if remapped_op != *op {
            self.map_results(op, &remapped_op);
        }

        // Check if this is an unrealized_conversion_cast
        if remapped_op.dialect(self.db) == core::DIALECT_NAME()
            && remapped_op.name(self.db) == core::UNREALIZED_CONVERSION_CAST()
        {
            return self.try_resolve_cast(&remapped_op);
        }

        // Not a cast - recursively process nested regions
        let regions = remapped_op.regions(self.db);
        let final_op = if regions.is_empty() {
            remapped_op
        } else {
            let new_regions: IdVec<Region<'db>> = regions
                .iter()
                .map(|region| self.resolve_region(region))
                .collect();
            remapped_op.modify(self.db).regions(new_regions).build()
        };

        // If the operation changed (operands remapped or regions updated),
        // map old results to new results so subsequent operations can find them
        if final_op != *op {
            self.map_results(op, &final_op);
        }

        vec![final_op]
    }

    /// Map old operation results to new operation results.
    fn map_results(&mut self, old_op: &Operation<'db>, new_op: &Operation<'db>) {
        let old_results = old_op.results(self.db);
        let new_results = new_op.results(self.db);
        let count = old_results.len().min(new_results.len());
        for i in 0..count {
            let old_val = old_op.result(self.db, i);
            let new_val = new_op.result(self.db, i);
            if old_val != new_val {
                self.value_map.insert(old_val, new_val);
            }
        }
    }

    fn try_resolve_cast(&mut self, cast_op: &Operation<'db>) -> Vec<Operation<'db>> {
        let operands = cast_op.operands(self.db);
        let results = cast_op.results(self.db);

        // Get the input value and types
        let input_value = operands[0];
        let original_to_type = results[0]; // Target type is stored in the result type

        // Convert the target type if needed (e.g., core.array -> wasm.arrayref)
        let to_type = self
            .type_converter
            .convert_type(self.db, original_to_type)
            .unwrap_or(original_to_type);

        // Get the source type from the input value
        let from_type = self.get_value_type(input_value);
        let Some(from_type) = from_type else {
            // Can't determine source type - keep the cast as-is and mark as unresolved
            self.unresolved.push(UnresolvedCast {
                location: cast_op.location(self.db),
                from_type: to_type, // Use target type as placeholder
                to_type,
            });
            return vec![*cast_op];
        };

        // If types are the same, just map the value directly (no-op conversion)
        if from_type == to_type {
            let cast_result = cast_op.result(self.db, 0);
            self.value_map.insert(cast_result, input_value);
            self.resolved_count += 1;
            return vec![];
        }

        // Try to materialize the conversion
        let location = cast_op.location(self.db);
        let mat_result =
            self.type_converter
                .materialize(self.db, location, input_value, from_type, to_type);

        match mat_result {
            Some(crate::rewrite::MaterializeResult::NoOp) => {
                // No IR needed - just map the value
                let cast_result = cast_op.result(self.db, 0);
                self.value_map.insert(cast_result, input_value);
                self.resolved_count += 1;
                vec![]
            }
            Some(crate::rewrite::MaterializeResult::Ops(ops)) => {
                // Use the materialized operations
                if let Some(last_op) = ops.last() {
                    let cast_result = cast_op.result(self.db, 0);
                    let materialized_result = last_op.result(self.db, 0);
                    self.value_map.insert(cast_result, materialized_result);
                }
                self.resolved_count += 1;
                ops.into_vec()
            }
            Some(crate::rewrite::MaterializeResult::Skip) | None => {
                // Could not materialize - keep the cast and mark as unresolved
                tracing::debug!(
                    "resolve_unrealized_casts: FAILED to materialize {}.{} -> {}.{}",
                    from_type.dialect(self.db),
                    from_type.name(self.db),
                    to_type.dialect(self.db),
                    to_type.name(self.db),
                );
                self.unresolved.push(UnresolvedCast {
                    location,
                    from_type,
                    to_type,
                });
                vec![*cast_op]
            }
        }
    }

    fn get_value_type(&self, value: Value<'db>) -> Option<Type<'db>> {
        use crate::ValueDef;
        match value.def(self.db) {
            ValueDef::OpResult(op) => op.results(self.db).get(value.index(self.db)).copied(),
            ValueDef::BlockArg(block_id) => {
                // Look up the type from our pre-collected block arg types map
                self.block_arg_types
                    .get(&(block_id, value.index(self.db)))
                    .copied()
            }
        }
    }

    fn remap_operands(&self, op: &Operation<'db>) -> Operation<'db> {
        let operands = op.operands(self.db);
        let mut new_operands: IdVec<Value<'db>> = IdVec::new();
        let mut changed = false;

        for &operand in operands.iter() {
            let mapped = self.lookup(operand);
            new_operands.push(mapped);
            if mapped != operand {
                changed = true;
            }
        }

        if !changed {
            return *op;
        }

        op.modify(self.db).operands(new_operands).build()
    }

    fn lookup(&self, value: Value<'db>) -> Value<'db> {
        let mut current = value;
        let mut visited = std::collections::HashSet::new();
        while let Some(&mapped) = self.value_map.get(&current) {
            if !visited.insert(current) {
                break; // Cycle detected
            }
            current = mapped;
        }
        current
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialect::{arith, core};
    use crate::ops::DialectOp;
    use crate::types::DialectType;
    use crate::{Attribute, BlockId, PathId, Span, Symbol, idvec};
    use salsa_test_macros::salsa_test;

    /// Create a module with unrealized_conversion_cast operations.
    #[salsa::tracked]
    fn make_module_with_casts(db: &dyn salsa::Database) -> Module<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let i32_ty = core::I32::new(db).as_type();
        let i64_ty = core::I64::new(db).as_type();

        // Create: const(42) : i32
        let const_op = arith::r#const(db, location, i32_ty, Attribute::IntBits(42));

        // Create: unrealized_conversion_cast(const_result) : i64
        let cast_op = core::unrealized_conversion_cast(db, location, const_op.result(db), i64_ty);

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![const_op.as_operation(), cast_op.as_operation()],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, Symbol::new("test"), region)
    }

    /// Result of running resolve_unrealized_casts: (is_ok, resolved_count, unresolved_count)
    #[salsa::tracked]
    fn run_resolve_no_materializer<'db>(
        db: &'db dyn salsa::Database,
        module: Module<'db>,
    ) -> (bool, usize, usize) {
        let type_converter = TypeConverter::new();
        let result = resolve_unrealized_casts(db, module, &type_converter);
        if result.unresolved.is_empty() {
            (true, result.resolved_count, 0)
        } else {
            (false, result.resolved_count, result.unresolved.len())
        }
    }

    #[salsa_test]
    fn test_resolve_unrealized_casts_no_materializer(db: &salsa::DatabaseImpl) {
        let module = make_module_with_casts(db);
        let (is_ok, _, unresolved_count) = run_resolve_no_materializer(db, module);

        // Should fail because no materializer is registered
        assert!(!is_ok);
        assert_eq!(unresolved_count, 1);
    }

    #[salsa::tracked]
    fn run_resolve_with_noop<'db>(
        db: &'db dyn salsa::Database,
        module: Module<'db>,
    ) -> (bool, usize, usize, Module<'db>) {
        let type_converter =
            TypeConverter::new().add_materialization(|_db, _loc, _value, _from_ty, _to_ty| {
                crate::rewrite::MaterializeResult::NoOp
            });
        let result = resolve_unrealized_casts(db, module, &type_converter);
        let is_ok = result.unresolved.is_empty();
        (
            is_ok,
            result.resolved_count,
            result.unresolved.len(),
            result.module,
        )
    }

    #[salsa_test]
    fn test_resolve_unrealized_casts_with_noop_materializer(db: &salsa::DatabaseImpl) {
        let module = make_module_with_casts(db);
        let (is_ok, resolved_count, _, result_module) = run_resolve_with_noop(db, module);

        assert!(is_ok);
        assert_eq!(resolved_count, 1);

        // Cast should be removed
        let ops = result_module.body(db).blocks(db)[0].operations(db);
        assert_eq!(ops.len(), 1, "Cast should be removed, only const remains");
        assert_eq!(ops[0].name(db), arith::CONST());
    }

    /// Create a module with same-type cast (i32 -> i32).
    #[salsa::tracked]
    fn make_module_same_type_cast(db: &dyn salsa::Database) -> Module<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let i32_ty = core::I32::new(db).as_type();

        let const_op = arith::r#const(db, location, i32_ty, Attribute::IntBits(42));
        let cast_op = core::unrealized_conversion_cast(db, location, const_op.result(db), i32_ty);

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![const_op.as_operation(), cast_op.as_operation()],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, Symbol::new("test"), region)
    }

    #[salsa_test]
    fn test_resolve_unrealized_casts_same_type_noop(db: &salsa::DatabaseImpl) {
        let module = make_module_same_type_cast(db);
        let (is_ok, resolved_count, _) = run_resolve_no_materializer(db, module);

        // Even without a materializer, same-type casts should be resolved
        assert!(is_ok);
        assert_eq!(resolved_count, 1);
    }
}
