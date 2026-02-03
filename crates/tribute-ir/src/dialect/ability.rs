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
//! ## Types
//!
//! Evidence and Marker are represented using standard ADT types:
//!
//! ```text
//! type Evidence = core.array(Marker)  // Array of markers for O(log n) lookup
//!
//! struct _Marker {                    // adt.struct type
//!     ability_id: i32,                // Hash of ability name
//!     prompt_tag: i32,                // For cont.shift
//!     op_table_index: i32,            // Handler dispatch table index
//! }
//! ```
//!
//! Use `evidence_adt_type()` and `marker_adt_type()` to create these types.
//! Use `is_evidence_type()` and `is_marker_type()` to check type compatibility.
//!
//! See `new-plans/implementation.md` for details.

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
    }
}

// Re-export cont::PromptTag for backward compatibility
pub use trunk_ir::dialect::cont::PromptTag;

// === ADT Type Functions ===
//
// These functions provide the canonical ADT representations for evidence-based
// handler dispatch. They are used by various passes to ensure consistent type
// identity across the compilation pipeline.

use trunk_ir::dialect::{adt, core};
use trunk_ir::{DialectType, Symbol, Type};

/// Get the canonical Marker ADT type for evidence-based dispatch.
///
/// Layout:
/// ```text
/// struct _Marker {
///     ability_id: i32,      // Hash of ability name for fast lookup
///     prompt_tag: i32,      // Prompt tag for cont.shift
///     op_table_index: i32,  // Index into handler dispatch table
/// }
/// ```
///
/// This type is used in the evidence array for dynamic handler lookup.
pub fn marker_adt_type(db: &dyn salsa::Database) -> Type<'_> {
    let i32_ty = core::I32::new(db).as_type();

    adt::struct_type(
        db,
        "_Marker",
        vec![
            (Symbol::new("ability_id"), i32_ty),
            (Symbol::new("prompt_tag"), i32_ty),
            (Symbol::new("op_table_index"), i32_ty),
        ],
    )
}

/// Get the canonical Evidence ADT type (array of Marker).
///
/// Evidence is a sorted array of Marker structs for O(log n) ability lookup.
/// At the WASM level, this is represented as `wasm.arrayref`.
///
/// ```text
/// type Evidence = Array(Marker)
/// ```
pub fn evidence_adt_type(db: &dyn salsa::Database) -> Type<'_> {
    let marker_ty = marker_adt_type(db);
    core::Array::new(db, marker_ty).as_type()
}

/// Check if a type is the evidence ADT type (`core.array(Marker)`).
pub fn is_evidence_type(db: &dyn salsa::Database, ty: Type<'_>) -> bool {
    // Evidence is core.array(Marker)
    if let Some(array_ty) = core::Array::from_type(db, ty) {
        is_marker_type(db, array_ty.element(db))
    } else {
        false
    }
}

/// Check if a type is the marker ADT type (`adt.struct("_Marker", ...)`).
pub fn is_marker_type(db: &dyn salsa::Database, ty: Type<'_>) -> bool {
    // Check if it's an adt.struct with name "_Marker"
    if !adt::is_struct_type(db, ty) {
        return false;
    }

    if let Some(name) = adt::get_type_name(db, ty) {
        name == Symbol::new("_Marker")
    } else {
        false
    }
}

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

    // === ADT Type Function Tests ===

    #[salsa_test]
    fn test_marker_adt_type(db: &salsa::DatabaseImpl) {
        let marker_ty = marker_adt_type(db);

        // Should be an adt.struct
        assert!(adt::is_struct_type(db, marker_ty));

        // Should have name "_Marker"
        let name = adt::get_type_name(db, marker_ty);
        assert_eq!(name, Some(Symbol::new("_Marker")));

        // Should have 3 i32 fields
        let fields = adt::get_struct_fields(db, marker_ty).expect("should have fields");
        assert_eq!(fields.len(), 3);

        let i32_ty = core::I32::new(db).as_type();
        assert_eq!(fields[0], (Symbol::new("ability_id"), i32_ty));
        assert_eq!(fields[1], (Symbol::new("prompt_tag"), i32_ty));
        assert_eq!(fields[2], (Symbol::new("op_table_index"), i32_ty));
    }

    #[salsa_test]
    fn test_evidence_adt_type(db: &salsa::DatabaseImpl) {
        let evidence_ty = evidence_adt_type(db);

        // Should be a core.array type
        let array = core::Array::from_type(db, evidence_ty);
        assert!(array.is_some());

        // Element type should be the Marker ADT
        let element_ty = array.unwrap().element(db);
        assert!(is_marker_type(db, element_ty));
    }

    #[salsa_test]
    fn test_is_marker_type(db: &salsa::DatabaseImpl) {
        let marker_ty = marker_adt_type(db);
        assert!(is_marker_type(db, marker_ty));

        // Non-marker struct should return false
        let other_struct = adt::struct_type(
            db,
            "OtherStruct",
            vec![(Symbol::new("field"), core::I32::new(db).as_type())],
        );
        assert!(!is_marker_type(db, other_struct));

        // Non-struct type should return false
        let i32_ty = core::I32::new(db).as_type();
        assert!(!is_marker_type(db, i32_ty));
    }

    #[salsa_test]
    fn test_is_evidence_type(db: &salsa::DatabaseImpl) {
        let evidence_ty = evidence_adt_type(db);
        assert!(is_evidence_type(db, evidence_ty));

        // Array of non-marker should return false
        let other_array = core::Array::new(db, core::I32::new(db).as_type()).as_type();
        assert!(!is_evidence_type(db, other_array));

        // Non-array type should return false
        let i32_ty = core::I32::new(db).as_type();
        assert!(!is_evidence_type(db, i32_ty));
    }

    // === PromptTag Re-export Test ===

    #[salsa_test]
    fn test_prompt_tag_type(db: &salsa::DatabaseImpl) {
        // PromptTag is re-exported from cont dialect
        let prompt_tag_ty = PromptTag::new(db);

        assert_eq!(prompt_tag_ty.as_type().dialect(db), cont::DIALECT_NAME());
        assert_eq!(prompt_tag_ty.as_type().name(db), cont::PROMPT_TAG());
        assert_eq!(print_type(db, prompt_tag_ty.as_type()), "PromptTag");
    }

    // === Operation Tests ===

    #[salsa::tracked]
    fn evidence_lookup_test(db: &dyn salsa::Database) -> (Symbol, Symbol) {
        let location = test_location(db);
        let evidence_ty = evidence_adt_type(db);
        let marker_ty = marker_adt_type(db);
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
        let evidence_ty = evidence_adt_type(db);
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
