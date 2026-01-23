//! WebAssembly lowering plan metadata.
//!
//! Tracks module-level planning decisions during wasm lowering:
//! - Memory allocation planning
//! - Function exports
//!
//! Note: WASI imports and data segments are now handled by intrinsic_to_wasm and const_to_wasm passes.

use trunk_ir::Type;

/// Linear memory planning.
///
/// Tracks memory initialization and export decisions.
#[derive(Default)]
pub struct MemoryPlan {
    /// Whether a memory section has been defined in the module.
    pub has_memory: bool,
    /// Whether memory has been exported.
    pub has_exported_memory: bool,
    /// Whether any memory is needed by the module.
    pub needs_memory: bool,
}

impl MemoryPlan {
    /// Create a new memory plan.
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate required pages for the given end offset.
    pub fn required_pages(&self, end_offset: u32) -> u32 {
        std::cmp::max(1, end_offset.div_ceil(0x10000))
    }
}

/// Main function export tracking.
///
/// Tracks whether the main function was encountered and what type it returns.
#[derive(Default)]
pub struct MainExports<'db> {
    /// Whether the main function was encountered during lowering.
    pub saw_main: bool,
    /// The return type of the main function, if any.
    /// When main is effectful, this will be the Step type.
    pub main_result_type: Option<Type<'db>>,
    /// The original return type of main before Step conversion.
    /// This is used by the trampoline to properly unbox the result.
    pub original_result_type: Option<Type<'db>>,
    /// Whether main has been exported.
    pub main_exported: bool,
}

impl<'db> MainExports<'db> {
    /// Create a new main exports tracker.
    pub fn new() -> Self {
        Self::default()
    }
}
