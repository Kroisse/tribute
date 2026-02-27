//! Arena-based ability dialect.

#[trunk_ir::arena_dialect]
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
}

// === Arena ADT Type Functions ===
//
// Arena equivalents of the Salsa type helpers in `crate::dialect::ability`.
// These construct canonical Marker and Evidence types using the arena type interner.

use trunk_ir::arena::context::IrContext;
use trunk_ir::arena::refs::TypeRef;
use trunk_ir::arena::types::{Attribute, TypeDataBuilder};
use trunk_ir::ir::Symbol;

/// Get the canonical Marker ADT type for evidence-based dispatch (arena version).
///
/// Layout:
/// ```text
/// struct _Marker {
///     ability_id: i32,
///     prompt_tag: i32,
///     op_table_index: i32,
/// }
/// ```
pub fn marker_adt_type_ref(ctx: &mut IrContext) -> TypeRef {
    let i32_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());

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
            Attribute::Symbol(Symbol::new("op_table_index")),
            Attribute::Type(i32_ty),
        ]),
    ]);

    ctx.types.intern(
        TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("struct"))
            .attr("name", Attribute::Symbol(Symbol::new("_Marker")))
            .attr("fields", fields_attr)
            .build(),
    )
}

/// Get the canonical Evidence ADT type — `core.array(Marker)` (arena version).
pub fn evidence_adt_type_ref(ctx: &mut IrContext) -> TypeRef {
    let marker_ty = marker_adt_type_ref(ctx);
    ctx.types.intern(
        TypeDataBuilder::new(Symbol::new("core"), Symbol::new("array"))
            .param(marker_ty)
            .build(),
    )
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

        // Should have 3 fields
        let fields = data.attrs.get(&Symbol::new("fields")).unwrap();
        match fields {
            Attribute::List(list) => assert_eq!(list.len(), 3),
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
        let other_array = ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("core"), Symbol::new("array"))
                .param(i32_ty)
                .build(),
        );
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
