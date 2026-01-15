//! Shared constants for WASM backend.

/// Global variable indices for yield/trampoline state.
///
/// These globals are created in `lower_wasm.rs` and accessed in `trampoline_to_wasm.rs`.
/// The order must be kept consistent between these two modules.
pub mod yield_globals {
    /// `$yield_state`: i32 (0 = normal, 1 = yielding)
    pub const STATE_IDX: u32 = 0;
    /// `$yield_tag`: i32 (prompt tag being yielded to)
    pub const TAG_IDX: u32 = 1;
    /// `$yield_cont`: anyref (captured continuation)
    pub const CONT_IDX: u32 = 2;
    /// `$yield_op_idx`: i32 (operation index within ability)
    pub const OP_IDX: u32 = 3;
}
