//! WebAssembly lowering plan metadata.
//!
//! Tracks module-level planning decisions during wasm lowering:
//! - Memory allocation planning
//! - Function exports
//!
//! Note: WASI imports and data segments are now handled by intrinsic_to_wasm and const_to_wasm passes.

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
