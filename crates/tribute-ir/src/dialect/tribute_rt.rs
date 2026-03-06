//! Tribute runtime dialect — operation registrations and constants.
//!
//! Arena dialect definitions are in `arena/dialect/tribute_rt.rs`.

// === RC Header Layout ===

/// RC header size in bytes: 4 bytes refcount + 4 bytes rtti_idx = 8 bytes.
///
/// All heap-allocated objects are prefixed with this header. The allocation
/// functions receive `payload_size + RC_HEADER_SIZE` and return a raw pointer.
/// Callers store the header at the raw pointer and use `raw_ptr + RC_HEADER_SIZE`
/// as the payload pointer.
pub const RC_HEADER_SIZE: u64 = 8;

// === Pure operation registrations ===
// Boxing operations are pure (no side effects)

inventory::submit! { trunk_ir::op_interface::PureOps::register("tribute_rt", "box_int") }
inventory::submit! { trunk_ir::op_interface::PureOps::register("tribute_rt", "unbox_int") }
inventory::submit! { trunk_ir::op_interface::PureOps::register("tribute_rt", "box_nat") }
inventory::submit! { trunk_ir::op_interface::PureOps::register("tribute_rt", "unbox_nat") }
inventory::submit! { trunk_ir::op_interface::PureOps::register("tribute_rt", "box_float") }
inventory::submit! { trunk_ir::op_interface::PureOps::register("tribute_rt", "unbox_float") }
inventory::submit! { trunk_ir::op_interface::PureOps::register("tribute_rt", "box_bool") }
inventory::submit! { trunk_ir::op_interface::PureOps::register("tribute_rt", "unbox_bool") }
