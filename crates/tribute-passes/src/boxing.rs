//! Boxing pass - removed.
//!
//! Boxing/unboxing is now handled via `unrealized_conversion_cast` operations
//! inserted during `ast_to_ir`, which are later materialized by `resolve_casts`
//! using the TypeConverter infrastructure.
//!
//! See:
//! - `crates/tribute-front/src/ast_to_ir/lower.rs`: `cast_if_needed()`
//! - `crates/tribute-passes/src/type_converter.rs`: boxing materializers in `generic_type_converter_arena()`
//! - `crates/trunk-ir/src/conversion/resolve_unrealized_casts.rs`: `resolve_unrealized_casts_arena()`
