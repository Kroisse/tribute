//! Tribute language-specific dialect modules.

// === High-level (AST/HIR level) ===
pub mod tribute;
pub mod tribute_pat;

// === Runtime types and boxing ===
pub mod tribute_rt;

// === Effect system (runtime ops, lowered to cont) ===
pub mod ability;

// === Type system ===
pub mod adt;

// === Closures ===
pub mod closure;

// === Data structures ===
pub mod list;
