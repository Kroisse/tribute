//! Ability dialect operations and types.
//!
//! This dialect provides evidence-based handler dispatch operations and types
//! for the ability (algebraic effect) system.
//!
//! ## Design
//!
//! This dialect contains the evidence operations for dynamic handler dispatch:
//! - `ability.evidence_lookup`: look up a marker in the evidence
//! - `ability.evidence_extend`: extend evidence with a new marker
//!
//! Handler lowering converts ability operations to `cont.push_prompt` + runtime calls.
//!
//! ## Future Direction
//!
//! The current `evidence_ptr` and `marker` opaque types are planned to be replaced
//! with standard ADT types (struct/array) to simplify the type system:
//!
//! ```text
//! type Evidence = Array(Marker)
//!
//! struct Marker {
//!     ability_id: i32,
//!     prompt_tag: i32,
//!     op_table_index: i32,
//! }
//! ```
//!
//! The `marker_prompt` operation has been replaced by `adt.struct_get(marker, 1)` (field index 1).
//! See `new-plans/implementation.md` for details.

use std::fmt::Write;

use trunk_ir::dialect;

dialect! {
    mod ability {
        // === Evidence operations ===

        /// `ability.evidence_lookup` operation: looks up a marker in the evidence.
        ///
        /// Finds the marker for a specific ability in the evidence structure.
        /// The marker contains the prompt tag for this handler.
        ///
        /// Returns a marker that can be used to get the prompt tag.
        #[attr(ability_ref: Type)]
        fn evidence_lookup(evidence) -> result;

        /// `ability.evidence_extend` operation: creates new evidence with an additional marker.
        ///
        /// Called when installing a handler. Creates a new evidence by adding
        /// a marker for the handled ability to the existing evidence.
        ///
        /// `prompt_tag` is the fresh prompt tag for the new handler.
        #[attr(ability_ref: Type, prompt_tag: any)]
        fn evidence_extend(evidence) -> result;

        /// `ability.marker_prompt` operation: extracts the prompt tag from a marker.
        ///
        /// Returns the prompt tag associated with the marker, which is used
        /// for `cont.shift` to capture the correct continuation.
        ///
        /// **Deprecated**: This operation will be replaced by `adt.struct_get(marker, 1)`
        /// (field index 1 = prompt_tag) once Evidence/Marker types are migrated to standard ADT types.
        fn marker_prompt(marker) -> result;

        // === Handler Table Operations ===

        /// `ability.handler_table` operation: defines handler dispatch table.
        ///
        /// This operation is emitted after resolve_evidence pass to capture
        /// the handler dispatch table structure. It is later lowered to
        /// wasm.table + wasm.elem in the WASM backend.
        ///
        /// The `entries` region contains `ability.handler_entry` operations,
        /// one per registered handler.
        ///
        /// ## Attributes
        /// - `max_ops_per_handler`: Maximum number of operations per handler (for table sizing)
        ///
        /// ## Example IR
        /// ```text
        /// ability.handler_table { max_ops_per_handler = 8 }
        ///   entries {
        ///     ability.handler_entry { tag = 0, op_count = 2 } { ... }
        ///     ability.handler_entry { tag = 1, op_count = 1 } { ... }
        ///   }
        /// ```
        #[attr(max_ops_per_handler: u32)]
        fn handler_table() {
            #[region(entries)] {}
        };

        /// `ability.handler_entry` operation: single handler's dispatch entries.
        ///
        /// Each entry corresponds to a handler (push_prompt) and contains
        /// function references for each operation it handles.
        ///
        /// The `funcs` region contains func.constant operations for each
        /// handler operation function.
        ///
        /// ## Attributes
        /// - `tag`: The handler's unique tag (op_table_index)
        /// - `op_count`: Number of operations in this handler
        ///
        /// ## Example IR
        /// ```text
        /// ability.handler_entry { tag = 0, op_count = 2 }
        ///   funcs {
        ///     func.constant @__handler_0_op_0
        ///     func.constant @__handler_0_op_1
        ///   }
        /// ```
        #[attr(tag: u32, op_count: u32)]
        fn handler_entry() {
            #[region(funcs)] {}
        };

        // === Types ===

        /// `ability.evidence_ptr` type: pointer to evidence struct.
        ///
        /// Evidence is a runtime structure containing ability markers for
        /// dynamic handler dispatch. Passed as first argument to effectful functions.
        ///
        /// **Deprecated**: Will be replaced by `Array(Marker)` - a sorted array of
        /// marker structs. See `new-plans/implementation.md` for the new design.
        type evidence_ptr;

        /// `ability.marker` type: marker within evidence.
        ///
        /// Contains the prompt tag and handler information for a specific ability.
        /// Used for evidence-based handler dispatch.
        ///
        /// **Deprecated**: Will be replaced by a standard struct type:
        /// ```text
        /// struct Marker { ability_id: i32, prompt_tag: i32, op_table_index: i32 }
        /// ```
        type marker;
    }
}

// Re-export cont::PromptTag for backward compatibility
pub use trunk_ir::dialect::cont::PromptTag;

// === Printable interface registrations ===

use trunk_ir::type_interface::{PrintContext, Printable};

// evidence_ptr -> "Evidence"
inventory::submit! { Printable::implement("ability", "evidence_ptr", |_, _, f: &mut PrintContext<'_, '_>| f.write_str("Evidence")) }

// marker -> "Marker"
inventory::submit! { Printable::implement("ability", "marker", |_, _, f: &mut PrintContext<'_, '_>| f.write_str("Marker")) }

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::dialect::{arith, cont, core};
    use trunk_ir::type_interface::print_type;
    use trunk_ir::{Attribute, DialectOp, DialectType, Location, PathId, Span, Symbol};

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    #[salsa_test]
    fn test_evidence_ptr_type(db: &salsa::DatabaseImpl) {
        let evidence_ty = EvidencePtr::new(db);

        assert_eq!(evidence_ty.as_type().dialect(db), DIALECT_NAME());
        assert_eq!(evidence_ty.as_type().name(db), EVIDENCE_PTR());
    }

    #[salsa_test]
    fn test_evidence_ptr_printable(db: &salsa::DatabaseImpl) {
        let evidence_ty = EvidencePtr::new(db);

        let printed = print_type(db, evidence_ty.as_type());
        assert_eq!(printed, "Evidence");
    }

    #[salsa_test]
    fn test_marker_type(db: &salsa::DatabaseImpl) {
        let marker_ty = Marker::new(db);

        assert_eq!(marker_ty.as_type().dialect(db), DIALECT_NAME());
        assert_eq!(marker_ty.as_type().name(db), MARKER());
        assert_eq!(print_type(db, marker_ty.as_type()), "Marker");
    }

    #[salsa_test]
    fn test_prompt_tag_type(db: &salsa::DatabaseImpl) {
        // PromptTag is re-exported from cont dialect
        let prompt_tag_ty = PromptTag::new(db);

        assert_eq!(prompt_tag_ty.as_type().dialect(db), cont::DIALECT_NAME());
        assert_eq!(prompt_tag_ty.as_type().name(db), cont::PROMPT_TAG());
        assert_eq!(print_type(db, prompt_tag_ty.as_type()), "PromptTag");
    }

    #[salsa::tracked]
    fn evidence_lookup_test(db: &dyn salsa::Database) -> (Symbol, Symbol) {
        let location = test_location(db);
        let evidence_ty = EvidencePtr::new(db).as_type();
        let marker_ty = Marker::new(db).as_type();
        let ability_ref = core::AbilityRefType::simple(db, Symbol::new("State")).as_type();

        let evidence_val =
            arith::r#const(db, location, evidence_ty, Attribute::IntBits(0)).result(db);
        let op = evidence_lookup(db, location, evidence_val, marker_ty, ability_ref);

        let adapted = EvidenceLookup::from_operation(db, op.as_operation()).unwrap();
        (
            adapted.ability_ref(db).dialect(db),
            adapted.as_operation().name(db),
        )
    }

    #[salsa_test]
    fn test_evidence_lookup_operation(db: &salsa::DatabaseImpl) {
        let (dialect, name) = evidence_lookup_test(db);
        assert_eq!(dialect, Symbol::new("core"));
        assert_eq!(name, EVIDENCE_LOOKUP());
    }

    #[salsa::tracked]
    fn evidence_extend_test(db: &dyn salsa::Database) -> Symbol {
        let location = test_location(db);
        let evidence_ty = EvidencePtr::new(db).as_type();
        let ability_ref = core::AbilityRefType::simple(db, Symbol::new("Console")).as_type();

        let evidence_val =
            arith::r#const(db, location, evidence_ty, Attribute::IntBits(0)).result(db);
        let op = evidence_extend(
            db,
            location,
            evidence_val,
            evidence_ty,
            ability_ref,
            Attribute::IntBits(42),
        );

        let adapted = EvidenceExtend::from_operation(db, op.as_operation()).unwrap();
        adapted.as_operation().name(db)
    }

    #[salsa_test]
    fn test_evidence_extend_operation(db: &salsa::DatabaseImpl) {
        let name = evidence_extend_test(db);
        assert_eq!(name, EVIDENCE_EXTEND());
    }

    #[salsa::tracked]
    fn marker_prompt_test(db: &dyn salsa::Database) -> Symbol {
        let location = test_location(db);
        let marker_ty = Marker::new(db).as_type();
        let prompt_tag_ty = PromptTag::new(db).as_type();

        let marker_val = arith::r#const(db, location, marker_ty, Attribute::IntBits(0)).result(db);
        let op = marker_prompt(db, location, marker_val, prompt_tag_ty);

        MarkerPrompt::from_operation(db, op.as_operation())
            .unwrap()
            .as_operation()
            .name(db)
    }

    #[salsa_test]
    fn test_marker_prompt_operation(db: &salsa::DatabaseImpl) {
        let name = marker_prompt_test(db);
        assert_eq!(name, MARKER_PROMPT());
    }

    #[salsa::tracked]
    fn handler_table_test(db: &dyn salsa::Database) -> (Symbol, u32) {
        use trunk_ir::{Block, BlockId, IdVec, Region};

        let location = test_location(db);

        // Create empty entries region
        let entry_block = Block::new(db, BlockId::fresh(), location, IdVec::new(), IdVec::new());
        let entries_region = Region::new(db, location, IdVec::from(vec![entry_block]));

        let op = handler_table(db, location, 8, entries_region);
        let adapted = HandlerTable::from_operation(db, op.as_operation()).unwrap();
        (
            adapted.as_operation().name(db),
            adapted.max_ops_per_handler(db),
        )
    }

    #[salsa_test]
    fn test_handler_table_operation(db: &salsa::DatabaseImpl) {
        let (name, max_ops) = handler_table_test(db);
        assert_eq!(name, HANDLER_TABLE());
        assert_eq!(max_ops, 8);
    }

    #[salsa::tracked]
    fn handler_entry_test(db: &dyn salsa::Database) -> (Symbol, u32, u32) {
        use trunk_ir::{Block, BlockId, IdVec, Region};

        let location = test_location(db);

        // Create empty funcs region
        let funcs_block = Block::new(db, BlockId::fresh(), location, IdVec::new(), IdVec::new());
        let funcs_region = Region::new(db, location, IdVec::from(vec![funcs_block]));

        let op = handler_entry(db, location, 0, 2, funcs_region);
        let adapted = HandlerEntry::from_operation(db, op.as_operation()).unwrap();
        (
            adapted.as_operation().name(db),
            adapted.tag(db),
            adapted.op_count(db),
        )
    }

    #[salsa_test]
    fn test_handler_entry_operation(db: &salsa::DatabaseImpl) {
        let (name, tag, op_count) = handler_entry_test(db);
        assert_eq!(name, HANDLER_ENTRY());
        assert_eq!(tag, 0);
        assert_eq!(op_count, 2);
    }
}
