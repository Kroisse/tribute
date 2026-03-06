//! Ability dialect — operation registrations only.
//!
//! Arena dialect definitions are in `arena/dialect/ability.rs`.

// === Pure operation registrations ===

inventory::submit! { trunk_ir::op_interface::PureOps::register("ability", "evidence_lookup") }
inventory::submit! { trunk_ir::op_interface::PureOps::register("ability", "evidence_extend") }

// handler_table and handler_entry are isolated (contain regions)
inventory::submit! { trunk_ir::op_interface::IsolatedFromAboveOps::register("ability", "handler_table") }
inventory::submit! { trunk_ir::op_interface::IsolatedFromAboveOps::register("ability", "handler_entry") }
