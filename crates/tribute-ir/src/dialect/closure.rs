//! Closure dialect — operation registrations only.
//!
//! Arena dialect definitions are in `arena/dialect/closure.rs`.

// === Pure operation registrations ===
// All closure operations are pure

inventory::submit! { trunk_ir::op_interface::PureOps::register("closure", "new") }
inventory::submit! { trunk_ir::op_interface::PureOps::register("closure", "func") }
inventory::submit! { trunk_ir::op_interface::PureOps::register("closure", "env") }
