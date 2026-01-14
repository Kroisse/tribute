//! IR validation for wasm backend.
//!
//! This module validates that IR is fully typed before emission.
//! Placeholder types (like `tribute.type_var`) should be resolved
//! by earlier passes before reaching the emit phase.

use std::collections::HashMap;

use tracing::warn;
use tribute_ir::dialect::tribute;
use trunk_ir::dialect::core::Module;
use trunk_ir::{Operation, Region, Symbol, Type};

/// Statistics about unresolved types found during validation.
#[derive(Debug, Default)]
pub struct ValidationStats {
    /// Number of unresolved type variables in operation results.
    pub type_var_in_results: usize,
    /// Number of operations with unresolved types (deduplicated).
    pub operations_with_type_var: usize,
    /// Counts by operation type (dialect.name -> count).
    pub counts_by_op: HashMap<(Symbol, Symbol), usize>,
}

impl ValidationStats {
    /// Returns true if any issues were found.
    pub fn has_issues(&self) -> bool {
        self.type_var_in_results > 0
    }

    /// Log a summary of validation results.
    pub fn log_summary(&self) {
        if self.has_issues() {
            // Log breakdown by operation type
            let mut breakdown: Vec<_> = self.counts_by_op.iter().collect();
            breakdown.sort_by(|a, b| b.1.cmp(a.1)); // Sort by count descending

            let breakdown_str: Vec<String> = breakdown
                .iter()
                .map(|((dialect, name), count)| {
                    dialect.with_str(|d| name.with_str(|n| format!("{}.{}: {}", d, n, count)))
                })
                .collect();

            warn!(
                "IR validation found {} unresolved type variable(s) in {} operation(s). \
                 Breakdown: [{}]. These should be resolved before emit phase.",
                self.type_var_in_results,
                self.operations_with_type_var,
                breakdown_str.join(", ")
            );
        }
    }
}

/// Validate that a module's IR is fully typed before emission.
///
/// This function walks all operations in the module and checks for
/// unresolved type variables (`tribute.type_var`). If found, warnings
/// are logged but compilation continues (for now).
///
/// In the future, this will return errors instead of warnings.
pub fn validate_wasm_types<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
) -> ValidationStats {
    let mut stats = ValidationStats::default();

    let body = module.body(db);
    validate_region(db, body, &mut stats);

    stats.log_summary();
    stats
}

/// Validate a region recursively.
fn validate_region<'db>(
    db: &'db dyn salsa::Database,
    region: Region<'db>,
    stats: &mut ValidationStats,
) {
    for block in region.blocks(db).iter() {
        for op in block.operations(db).iter() {
            validate_operation(db, op, stats);
        }
    }
}

/// Validate a single operation.
fn validate_operation<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    stats: &mut ValidationStats,
) {
    let mut op_has_type_var = false;

    // Check result types
    for (idx, result_ty) in op.results(db).iter().enumerate() {
        if is_unresolved_type(db, *result_ty) {
            warn!(
                "Unresolved type variable in result {} of {}.{}: {:?}",
                idx,
                op.dialect(db),
                op.name(db),
                format_type(db, *result_ty)
            );
            stats.type_var_in_results += 1;
            op_has_type_var = true;
        }
    }

    if op_has_type_var {
        stats.operations_with_type_var += 1;
        // Track by operation type
        let key = (op.dialect(db), op.name(db));
        *stats.counts_by_op.entry(key).or_insert(0) += 1;
    }

    // Recursively validate nested regions
    for region in op.regions(db).iter() {
        validate_region(db, *region, stats);
    }
}

/// Check if a type is an unresolved placeholder type.
fn is_unresolved_type<'db>(db: &'db dyn salsa::Database, ty: Type<'db>) -> bool {
    tribute::is_type_var(db, ty)
}

/// Format a type for diagnostic output.
fn format_type<'db>(db: &'db dyn salsa::Database, ty: Type<'db>) -> String {
    ty.dialect(db)
        .with_str(|d| ty.name(db).with_str(|n| format!("{}.{}", d, n)))
}

// Unit tests omitted - validation is tested via integration tests
// by compiling actual .trb files and checking log output.
