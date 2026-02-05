//! Boxing pass - DEPRECATED
//!
//! This pass has been removed. Boxing/unboxing is now handled via
//! `unrealized_conversion_cast` operations inserted during `ast_to_ir`,
//! which are later materialized by `resolve_casts` using the TypeConverter
//! infrastructure.
//!
//! See:
//! - `crates/tribute-front/src/ast_to_ir/lower.rs`: `cast_if_needed()`
//! - `crates/tribute-passes/src/type_converter.rs`: boxing materializers (lines 100-176)
//! - `trunk-ir/src/conversion.rs`: `resolve_unrealized_casts()`

use trunk_ir::dialect::core::Module;

/// DEPRECATED: Boxing is now handled via unrealized_conversion_cast.
///
/// This function is a no-op and exists only for compatibility.
/// It will be removed in a future refactoring.
#[deprecated(
    since = "0.1.0",
    note = "Boxing is now handled via unrealized_conversion_cast in ast_to_ir"
)]
pub fn insert_boxing<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    let _ = db;
    module
}
