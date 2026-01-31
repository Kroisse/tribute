//! Ability dialect operations and types.
//!
//! This dialect represents ability (algebraic effect) operations that get
//! lowered to the `cont` dialect for continuation-based control flow.
//!
//! ## Design
//!
//! Ability declarations (`tribute.ability_def`, `tribute.op_def`) are in the tribute dialect.
//! This dialect contains the runtime operations:
//! - `ability.perform`: invoke an ability operation
//! - `ability.resume`: resume a captured continuation
//! - `ability.abort`: discard a continuation
//!
//! Handler pattern matching uses fused `tribute.handle`:
//!
//! ```text
//! // Source: handle expr { ... }
//! // Lowers to:
//! tribute.handle { expr } {
//!     tribute.arm("{result}") { ... }
//!     tribute.arm("{State::get() -> k}") { ... }
//! }
//! ```

use std::fmt::Write;

use trunk_ir::dialect;

dialect! {
    mod ability {
        // === Operations ===

        /// `ability.perform` operation: performs an ability operation.
        ///
        /// Invokes an operation from an ability, capturing the current continuation
        /// until a handler is found. Returns when resumed by a handler.
        ///
        /// The `ability_ref` attribute is a `Type` (specifically `core.ability_ref`)
        /// to support parameterized abilities like `State(Int)`.
        #[attr(ability_ref: Type, op: Symbol)]
        fn perform(#[rest] args) -> result;

        /// `ability.resume` operation: resumes a captured continuation.
        ///
        /// Continues execution from where `perform` was called, providing a value.
        /// The continuation is consumed (linear type).
        fn resume(continuation, value) -> result;

        /// `ability.abort` operation: discards a continuation without resuming.
        ///
        /// Satisfies the linear type requirement for continuations by explicitly
        /// dropping them. Used when a handler doesn't want to continue execution
        /// (e.g., `Fail::fail` handler returning `None`).
        fn abort(continuation);

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
        fn marker_prompt(marker) -> result;

        // === Types ===

        /// `ability.evidence_ptr` type: pointer to evidence struct.
        ///
        /// Evidence is a runtime structure containing ability markers for
        /// dynamic handler dispatch. Passed as first argument to effectful functions.
        ///
        /// See `new-plans/implementation.md` for the evidence passing design.
        type evidence_ptr;

        /// `ability.marker` type: marker within evidence.
        ///
        /// Contains the prompt tag and handler information for a specific ability.
        /// Used for evidence-based handler dispatch.
        type marker;

        /// `ability.prompt_tag` type: prompt tag for delimited control.
        ///
        /// A unique identifier that connects `cont.push_prompt` with `cont.shift`.
        type prompt_tag;
    }
}

// === Printable interface registrations ===

use trunk_ir::type_interface::{PrintContext, Printable};

// evidence_ptr -> "Evidence"
inventory::submit! { Printable::implement("ability", "evidence_ptr", |_, _, f: &mut PrintContext<'_, '_>| f.write_str("Evidence")) }

// marker -> "Marker"
inventory::submit! { Printable::implement("ability", "marker", |_, _, f: &mut PrintContext<'_, '_>| f.write_str("Marker")) }

// prompt_tag -> "PromptTag"
inventory::submit! { Printable::implement("ability", "prompt_tag", |_, _, f: &mut PrintContext<'_, '_>| f.write_str("PromptTag")) }

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::dialect::{arith, core};
    use trunk_ir::type_interface::print_type;
    use trunk_ir::{Attribute, DialectOp, DialectType, IdVec, Location, PathId, Span, Symbol};

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

    #[salsa::tracked]
    fn perform_test(db: &dyn salsa::Database) -> (Symbol, Symbol) {
        let location = test_location(db);
        let int_ty = core::I32::new(db).as_type();
        let ability_ref = core::AbilityRefType::simple(db, Symbol::new("State")).as_type();

        let arg = arith::r#const(db, location, int_ty, Attribute::IntBits(0)).result(db);
        let op = perform(
            db,
            location,
            IdVec::from(vec![arg]),
            int_ty,
            ability_ref,
            Symbol::new("get"),
        );

        let adapted = Perform::from_operation(db, op.as_operation()).unwrap();
        (adapted.ability_ref(db).dialect(db), adapted.op(db))
    }

    #[salsa_test]
    fn test_perform_operation(db: &salsa::DatabaseImpl) {
        let (dialect, op_name) = perform_test(db);
        assert_eq!(dialect, Symbol::new("core"));
        assert_eq!(op_name, Symbol::new("get"));
    }

    #[salsa::tracked]
    fn resume_test(db: &dyn salsa::Database) -> (Symbol, Symbol) {
        let location = test_location(db);
        let int_ty = core::I32::new(db).as_type();

        let cont = arith::r#const(db, location, int_ty, Attribute::IntBits(0)).result(db);
        let value = arith::r#const(db, location, int_ty, Attribute::IntBits(42)).result(db);

        let op = resume(db, location, cont, value, int_ty);
        let adapted = Resume::from_operation(db, op.as_operation()).unwrap();
        (
            adapted.as_operation().dialect(db),
            adapted.as_operation().name(db),
        )
    }

    #[salsa_test]
    fn test_resume_operation(db: &salsa::DatabaseImpl) {
        let (dialect, name) = resume_test(db);
        assert_eq!(dialect, DIALECT_NAME());
        assert_eq!(name, RESUME());
    }

    #[salsa::tracked]
    fn abort_test(db: &dyn salsa::Database) -> (Symbol, Symbol) {
        let location = test_location(db);
        let int_ty = core::I32::new(db).as_type();

        let cont = arith::r#const(db, location, int_ty, Attribute::IntBits(0)).result(db);
        let op = abort(db, location, cont);

        let adapted = Abort::from_operation(db, op.as_operation()).unwrap();
        (
            adapted.as_operation().dialect(db),
            adapted.as_operation().name(db),
        )
    }

    #[salsa_test]
    fn test_abort_operation(db: &salsa::DatabaseImpl) {
        let (dialect, name) = abort_test(db);
        assert_eq!(dialect, DIALECT_NAME());
        assert_eq!(name, ABORT());
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
        let prompt_tag_ty = PromptTag::new(db);

        assert_eq!(prompt_tag_ty.as_type().dialect(db), DIALECT_NAME());
        assert_eq!(prompt_tag_ty.as_type().name(db), PROMPT_TAG());
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
}
