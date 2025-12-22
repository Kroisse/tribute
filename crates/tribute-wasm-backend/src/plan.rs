//! WebAssembly lowering plan metadata.
//!
//! Tracks module-level planning decisions during wasm lowering:
//! - WASI imports needed
//! - Memory allocation planning
//! - Function exports
//! - Data segments and literal allocations

use std::collections::HashMap;

use trunk_ir::{Type, Value};

/// WASI import planning.
///
/// Tracks whether WASI functions need to be imported during lowering.
#[derive(Default)]
pub(crate) struct WasiPlan {
    pub(crate) needs_fd_write: bool,
}

impl WasiPlan {
    /// Create a new WASI plan.
    pub(crate) fn new() -> Self {
        Self::default()
    }
}

/// Linear memory planning.
///
/// Tracks memory initialization and export decisions.
#[derive(Default)]
pub(crate) struct MemoryPlan {
    /// Whether a memory section has been defined in the module.
    pub(crate) has_memory: bool,
    /// Whether memory has been exported.
    pub(crate) has_exported_memory: bool,
    /// Whether any memory is needed by the module.
    pub(crate) needs_memory: bool,
}

impl MemoryPlan {
    /// Create a new memory plan.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Calculate required pages for the given end offset.
    pub(crate) fn required_pages(&self, end_offset: u32) -> u32 {
        std::cmp::max(1, end_offset.div_ceil(0x10000))
    }
}

/// Main function export tracking.
///
/// Tracks whether the main function was encountered and what type it returns.
#[derive(Default)]
pub(crate) struct MainExports<'db> {
    /// Whether the main function was encountered during lowering.
    pub(crate) saw_main: bool,
    /// The return type of the main function, if any.
    pub(crate) main_result_type: Option<Type<'db>>,
    /// Whether main has been exported.
    pub(crate) main_exported: bool,
}

impl<'db> MainExports<'db> {
    /// Create a new main exports tracker.
    pub(crate) fn new() -> Self {
        Self::default()
    }
}

/// Data segment allocation.
///
/// Manages linear memory data segments (string literals, WASI metadata, etc.)
/// and computes their offsets in the data section.
#[derive(Default)]
pub(crate) struct DataSegments<'db> {
    /// Next available offset for allocation.
    next_offset: u32,
    /// Data segments: (offset, bytes) pairs.
    segments: Vec<(u32, Vec<u8>)>,
    /// Mapping from literal values to (offset, length).
    literal_data: HashMap<Value<'db>, (u32, u32)>,
    /// Cached iovec structure offsets: (ptr, len) -> offset.
    iovec_offsets: HashMap<(u32, u32), u32>,
    /// Cached nwritten output location.
    nwritten_offset: Option<u32>,
}

impl<'db> DataSegments<'db> {
    /// Create a new data segments allocator.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Get the current end offset (total bytes allocated so far).
    pub(crate) fn end_offset(&self) -> u32 {
        self.next_offset
    }

    /// Allocate bytes in a data segment and return (offset, length).
    pub(crate) fn allocate_bytes(&mut self, bytes: Vec<u8>) -> (u32, u32) {
        let offset = Self::align_to(self.next_offset, 4);
        let len = bytes.len() as u32;
        self.segments.push((offset, bytes));
        self.next_offset = offset + len;
        (offset, len)
    }

    /// Record a literal value and its data segment location.
    pub(crate) fn record_literal(&mut self, value: Value<'db>, offset: u32, len: u32) {
        self.literal_data.insert(value, (offset, len));
    }

    /// Look up a recorded literal's location.
    pub(crate) fn literal_for(&self, value: Value<'db>) -> Option<(u32, u32)> {
        self.literal_data.get(&value).copied()
    }

    /// Ensure an iovec structure exists and return its offset.
    pub(crate) fn ensure_iovec(&mut self, ptr: u32, len: u32) -> u32 {
        if let Some(&offset) = self.iovec_offsets.get(&(ptr, len)) {
            return offset;
        }
        let mut bytes = Vec::with_capacity(8);
        bytes.extend_from_slice(&ptr.to_le_bytes());
        bytes.extend_from_slice(&len.to_le_bytes());
        let (offset, _) = self.allocate_bytes(bytes);
        self.iovec_offsets.insert((ptr, len), offset);
        offset
    }

    /// Ensure the nwritten output location exists and return its offset.
    pub(crate) fn ensure_nwritten(&mut self) -> u32 {
        if let Some(offset) = self.nwritten_offset {
            return offset;
        }
        let (offset, _) = self.allocate_bytes(vec![0, 0, 0, 0]);
        self.nwritten_offset = Some(offset);
        offset
    }

    /// Take all collected data segments.
    pub(crate) fn take_segments(&mut self) -> Vec<(u32, Vec<u8>)> {
        std::mem::take(&mut self.segments)
    }

    /// Align a value to the given power-of-2 alignment.
    fn align_to(value: u32, align: u32) -> u32 {
        if align == 0 {
            return value;
        }
        value.div_ceil(align) * align
    }
}
