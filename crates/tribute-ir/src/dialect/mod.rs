//! Tribute language-specific dialect modules.

// === High-level (AST/HIR level) ===
pub mod tribute;
pub mod tribute_pat;

// === Runtime types and boxing ===
pub mod tribute_rt;

// === Effect system (runtime ops, lowered to cont) ===
pub mod ability;

// === Continuation implementation ===
pub mod trampoline;

// === Closures ===
pub mod closure;

// === Data structures ===
pub mod list;

// Re-export adt from trunk-ir for backwards compatibility
pub use trunk_ir::dialect::adt;
