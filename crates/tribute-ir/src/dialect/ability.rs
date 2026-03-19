//! Ability dialect — evidence-based handler dispatch.

#[trunk_ir::dialect]
mod ability {
    #[attr(ability_ref: Type)]
    fn evidence_lookup(evidence: ()) -> result {}

    #[attr(ability_ref: Type, prompt_tag: any)]
    fn evidence_extend(evidence: ()) -> result {}

    #[attr(max_ops_per_handler: u32)]
    fn handler_table() {
        #[region(entries)]
        {}
    }

    #[attr(tag: u32, op_count: u32)]
    fn handler_entry() {
        #[region(funcs)]
        {}
    }

    /// Perform an ability operation with an explicit continuation closure.
    ///
    /// CPS replacement for `cont.shift`. The continuation closure captures
    /// the rest of the computation after the effect point.
    ///
    /// ```text
    /// %yr = ability.perform %continuation, [%args...]
    ///   { ability_ref: @State, op_name: @get }
    /// ```
    ///
    /// Lowered to: evidence lookup → ShiftInfo construction → YieldResult::Shift.
    #[attr(ability_ref: Type, op_name: Symbol)]
    fn perform(continuation: (), #[rest] values: ()) -> result {}

    /// Handler dispatch loop over a YieldResult value.
    ///
    /// CPS replacement for `cont.handler_dispatch`. Matches the prompt tag
    /// and dispatches to the appropriate handler arm in the body region.
    ///
    /// The `handler_fn` operand is a closure `(k, op_idx, value) -> void`
    /// that dispatches to the appropriate handler arm. It is stored in the
    /// Marker's `handler_dispatch` field by `resolve_evidence` for use by
    /// the tail-call-based CPS path in `lower_ability_perform`.
    ///
    /// ```text
    /// %result = ability.handle_dispatch %yield_result, %handler_fn
    ///   { tag: 0, result_type: anyref }
    ///   body { ... handler arms ... }
    /// ```
    #[attr(tag: u32, result_type: Type)]
    fn handle_dispatch(value: (), handler_fn: ()) -> result {
        #[region(body)]
        {}
    }

    fn done() {
        #[region(body)]
        {}
    }

    #[attr(ability_ref: Type, op_name: Symbol)]
    fn suspend() {
        #[region(body)]
        {}
    }

    /// Tail-resumptive yield: like `suspend` but guarantees no continuation capture.
    #[attr(ability_ref: Type, op_name: Symbol)]
    fn r#yield() {
        #[region(body)]
        {}
    }

    fn resume(continuation: (), value: ()) -> result {}
}

// === Hash-Based Dispatch ===

/// Compute operation index using hash-based dispatch.
///
/// Computes a stable, handler-independent index from ability name and
/// operation name. Both shift sites and handler dispatch use this function,
/// ensuring they always agree on the op index regardless of handler
/// registration order.
pub fn compute_op_idx(ability_ref: Option<Symbol>, op_name: Option<Symbol>) -> u32 {
    use std::hash::{Hash, Hasher};

    let mut hasher = rustc_hash::FxHasher::default();
    ability_ref.hash(&mut hasher);
    op_name.hash(&mut hasher);

    (hasher.finish() % 0x7FFFFFFF) as u32
}

// === Pure operation registrations ===

inventory::submit! { trunk_ir::op_interface::PureOps::register("ability", "evidence_lookup") }
inventory::submit! { trunk_ir::op_interface::PureOps::register("ability", "evidence_extend") }

// handler_table and handler_entry are isolated (contain regions)
inventory::submit! { trunk_ir::op_interface::IsolatedFromAboveOps::register("ability", "handler_table") }
inventory::submit! { trunk_ir::op_interface::IsolatedFromAboveOps::register("ability", "handler_entry") }

// === ADT Type Functions ===

use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::core as arena_core;
use trunk_ir::refs::TypeRef;
use trunk_ir::types::{Attribute, TypeDataBuilder};

/// Get the canonical Marker ADT type for evidence-based dispatch.
///
/// Layout:
/// ```text
/// struct _Marker {
///     ability_id: i32,
///     prompt_tag: i32,
///     tr_dispatch_fn: ptr,
///     handler_dispatch: ptr,
/// }
/// ```
///
/// `tr_dispatch_fn` is a pointer to a tail-resumptive dispatch function
/// `(op_idx: i32, shift_value: ptr) -> ptr`, or null if the handler is
/// not fully tail-resumptive.
///
/// `handler_dispatch` is a pointer to the full CPS handler dispatch closure
/// `(k: ptr, op_idx: i32, value: ptr) -> void`, or null if not using
/// full CPS. Used by the tail-call-based effect handling path.
pub fn marker_adt_type_ref(ctx: &mut IrContext) -> TypeRef {
    let i32_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());
    let ptr_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ptr")).build());

    let fields_attr = Attribute::List(vec![
        Attribute::List(vec![
            Attribute::Symbol(Symbol::new("ability_id")),
            Attribute::Type(i32_ty),
        ]),
        Attribute::List(vec![
            Attribute::Symbol(Symbol::new("prompt_tag")),
            Attribute::Type(i32_ty),
        ]),
        Attribute::List(vec![
            Attribute::Symbol(Symbol::new("tr_dispatch_fn")),
            Attribute::Type(ptr_ty),
        ]),
        Attribute::List(vec![
            Attribute::Symbol(Symbol::new("handler_dispatch")),
            Attribute::Type(ptr_ty),
        ]),
    ]);

    ctx.types.intern(
        TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("struct"))
            .attr("name", Attribute::Symbol(Symbol::new("_Marker")))
            .attr("fields", fields_attr)
            .build(),
    )
}

/// Get the canonical Evidence ADT type — `core.array(Marker)`.
pub fn evidence_adt_type_ref(ctx: &mut IrContext) -> TypeRef {
    let marker_ty = marker_adt_type_ref(ctx);
    arena_core::array(ctx, marker_ty).as_type_ref()
}

/// Check if a type is the marker ADT type (`adt.struct("_Marker", ...)`).
pub fn is_marker_type_ref(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.types.get(ty);
    if data.dialect != Symbol::new("adt") || data.name != Symbol::new("struct") {
        return false;
    }
    matches!(
        data.attrs.get(&Symbol::new("name")),
        Some(Attribute::Symbol(s)) if *s == Symbol::new("_Marker")
    )
}

/// Check if a type is the evidence ADT type (`core.array(Marker)`).
pub fn is_evidence_type_ref(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.types.get(ty);
    if data.dialect != Symbol::new("core") || data.name != Symbol::new("array") {
        return false;
    }
    if data.params.len() != 1 {
        return false;
    }
    is_marker_type_ref(ctx, data.params[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_marker_adt_type_ref() {
        let mut ctx = IrContext::new();
        let marker_ty = marker_adt_type_ref(&mut ctx);

        assert!(is_marker_type_ref(&ctx, marker_ty));

        // Should be an adt.struct
        let data = ctx.types.get(marker_ty);
        assert_eq!(data.dialect, Symbol::new("adt"));
        assert_eq!(data.name, Symbol::new("struct"));

        // Should have name "_Marker"
        assert_eq!(
            data.attrs.get(&Symbol::new("name")),
            Some(&Attribute::Symbol(Symbol::new("_Marker")))
        );

        // Should have 4 fields
        let fields = data.attrs.get(&Symbol::new("fields")).unwrap();
        match fields {
            Attribute::List(list) => assert_eq!(list.len(), 4),
            _ => panic!("expected list attribute for fields"),
        }
    }

    #[test]
    fn test_evidence_adt_type_ref() {
        let mut ctx = IrContext::new();
        let evidence_ty = evidence_adt_type_ref(&mut ctx);

        assert!(is_evidence_type_ref(&ctx, evidence_ty));

        // Should be a core.array type
        let data = ctx.types.get(evidence_ty);
        assert_eq!(data.dialect, Symbol::new("core"));
        assert_eq!(data.name, Symbol::new("array"));
        assert_eq!(data.params.len(), 1);

        // Element type should be the Marker ADT
        assert!(is_marker_type_ref(&ctx, data.params[0]));
    }

    #[test]
    fn test_is_marker_type_ref_negative() {
        let mut ctx = IrContext::new();

        // Non-marker struct should return false
        let other_struct = ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("struct"))
                .attr("name", Attribute::Symbol(Symbol::new("OtherStruct")))
                .build(),
        );
        assert!(!is_marker_type_ref(&ctx, other_struct));

        // Non-struct type should return false
        let i32_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());
        assert!(!is_marker_type_ref(&ctx, i32_ty));
    }

    #[test]
    fn test_is_evidence_type_ref_negative() {
        let mut ctx = IrContext::new();

        // Array of non-marker should return false
        let i32_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());
        let other_array = arena_core::array(&mut ctx, i32_ty).as_type_ref();
        assert!(!is_evidence_type_ref(&ctx, other_array));

        // Non-array type should return false
        assert!(!is_evidence_type_ref(&ctx, i32_ty));
    }

    #[test]
    fn test_type_deduplication() {
        let mut ctx = IrContext::new();
        let marker1 = marker_adt_type_ref(&mut ctx);
        let marker2 = marker_adt_type_ref(&mut ctx);
        assert_eq!(marker1, marker2, "marker types should be deduplicated");

        let ev1 = evidence_adt_type_ref(&mut ctx);
        let ev2 = evidence_adt_type_ref(&mut ctx);
        assert_eq!(ev1, ev2, "evidence types should be deduplicated");
    }
}
