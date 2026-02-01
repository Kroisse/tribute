//! Tribute language-specific dialect modules.

// === High-level (AST/HIR level) ===
pub mod tribute;

// === Runtime types and boxing ===
pub mod tribute_rt;

// === Effect system (runtime ops, lowered to cont) ===
pub mod ability;

// === Closures ===
pub mod closure;
