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
    /// The continuation closure captures the rest of the computation
    /// after the effect point.
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
    /// Matches the prompt tag and dispatches to the appropriate handler
    /// arm in the body region.
    ///
    /// The `handler_fn` operand is a closure `(k, op_idx, value) -> void`
    /// that dispatches to the appropriate handler arm. It is stored in the
    /// Marker's `handler_dispatch` field by `resolve_evidence` for use by
    /// the tail-call-based CPS path in `lower_ability_perform`.
    ///
    /// The `tr_dispatch_fn` operand is a closure `(op_idx, value) -> anyref`
    /// for tail-resumptive (`fn`) operations only. It is stored in the
    /// Marker's `tr_dispatch_fn` field. May be a null constant if there
    /// are no `fn` handlers.
    ///
    /// ```text
    /// %result = ability.handle_dispatch %yield_result, %handler_fn, %tr_dispatch_fn
    ///   { tag: 0, result_type: anyref }
    ///   body { ... handler arms ... }
    /// ```
    #[attr(tag: u32, result_type: Type)]
    fn handle_dispatch(value: (), handler_fn: (), tr_dispatch_fn: ()) -> result {
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

    /// Direct call to a `fn` (tail-resumptive) ability operation.
    ///
    /// Unlike `ability.perform`, this does not take a continuation closure.
    /// The result flows inline — no CPS transformation is needed.
    ///
    /// ```text
    /// %result = ability.call %args...
    ///   { ability_ref = @State, op_name = @get }
    /// ```
    ///
    /// Lowered to: evidence lookup → tr_dispatch_fn(op_idx, value) → result.
    #[attr(ability_ref: Type, op_name: Symbol)]
    fn call(#[rest] values: ()) -> result {}
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

/// Compute the stable runtime ability ID for an ability reference type.
pub fn compute_ability_id(ctx: &IrContext, ability_ref: TypeRef) -> u32 {
    let data = ctx.types.get(ability_ref);
    let ability_name = match data.attrs.get(&Symbol::new("name")) {
        Some(Attribute::Symbol(s)) => *s,
        _ => panic!(
            "ICE: compute_ability_id: ability type has no name: {:?}",
            data
        ),
    };

    let mut hash: u32 = ability_name.with_str(|s| {
        let mut h: u32 = 0;
        for byte in s.bytes() {
            h = h.wrapping_mul(31).wrapping_add(byte as u32);
        }
        h
    });

    for &param in data.params.iter() {
        hash = hash.wrapping_mul(37);
        hash = hash.wrapping_add(hash_type(ctx, param));
    }

    hash
}

fn hash_type(ctx: &IrContext, ty: TypeRef) -> u32 {
    let data = ctx.types.get(ty);
    let mut hash: u32 = 0;

    data.dialect.with_str(|s| {
        for byte in s.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u32);
        }
    });

    data.name.with_str(|s| {
        for byte in s.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u32);
        }
    });

    for &param in data.params.iter() {
        hash = hash.wrapping_mul(37);
        hash = hash.wrapping_add(hash_type(ctx, param));
    }

    hash
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

/// Canonical field identifiers for the `_Marker` ADT used by ability evidence.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkerField {
    AbilityId = 0,
    PromptTag = 1,
    TrDispatchFn = 2,
    HandlerDispatch = 3,
}

impl MarkerField {
    pub const fn index(self) -> u32 {
        self as u32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkerFieldType {
    I32,
    Ptr,
}

impl MarkerFieldType {
    fn type_ref(self, ctx: &mut IrContext) -> TypeRef {
        let dialect = Symbol::new("core");
        let name = match self {
            Self::I32 => Symbol::new("i32"),
            Self::Ptr => Symbol::new("ptr"),
        };
        ctx.types
            .intern(TypeDataBuilder::new(dialect, name).build())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MarkerFieldSpec {
    pub field: MarkerField,
    pub symbol_name: &'static str,
    pub field_type: MarkerFieldType,
}

/// Canonical field layout for the `_Marker` ADT.
pub const MARKER_FIELDS: [MarkerFieldSpec; 4] = [
    MarkerFieldSpec {
        field: MarkerField::AbilityId,
        symbol_name: "ability_id",
        field_type: MarkerFieldType::I32,
    },
    MarkerFieldSpec {
        field: MarkerField::PromptTag,
        symbol_name: "prompt_tag",
        field_type: MarkerFieldType::I32,
    },
    MarkerFieldSpec {
        field: MarkerField::TrDispatchFn,
        symbol_name: "tr_dispatch_fn",
        field_type: MarkerFieldType::Ptr,
    },
    MarkerFieldSpec {
        field: MarkerField::HandlerDispatch,
        symbol_name: "handler_dispatch",
        field_type: MarkerFieldType::Ptr,
    },
];

impl MarkerFieldSpec {
    fn type_ref(self, ctx: &mut IrContext) -> TypeRef {
        self.field_type.type_ref(ctx)
    }
}

impl MarkerField {
    pub fn spec(self) -> &'static MarkerFieldSpec {
        &MARKER_FIELDS[self.index() as usize]
    }

    pub fn symbol_name(self) -> &'static str {
        self.spec().symbol_name
    }

    pub fn field_type(self) -> MarkerFieldType {
        self.spec().field_type
    }
}

pub const MARKER_FIELD_COUNT: usize = MARKER_FIELDS.len();

const _: () = {
    let mut idx = 0;
    while idx < MARKER_FIELDS.len() {
        assert!(MARKER_FIELDS[idx].field.index() as usize == idx);
        idx += 1;
    }
};

/// Runtime ABI symbols used by native and WASM evidence lowering.
pub mod evidence_abi {
    pub const EMPTY: &str = "__tribute_evidence_empty";
    pub const LOOKUP: &str = "__tribute_evidence_lookup";
    pub const EXTEND: &str = "__tribute_evidence_extend";
    pub const LOOKUP_TR: &str = "__tribute_evidence_lookup_tr";
    pub const LOOKUP_HANDLER: &str = "__tribute_evidence_lookup_handler";
}

pub fn evidence_runtime_symbols() -> [Symbol; 5] {
    [
        Symbol::new(evidence_abi::EMPTY),
        Symbol::new(evidence_abi::LOOKUP),
        Symbol::new(evidence_abi::EXTEND),
        Symbol::new(evidence_abi::LOOKUP_TR),
        Symbol::new(evidence_abi::LOOKUP_HANDLER),
    ]
}

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
    let fields_attr = Attribute::List(
        MARKER_FIELDS
            .into_iter()
            .map(|spec| {
                Attribute::List(vec![
                    Attribute::Symbol(Symbol::new(spec.symbol_name)),
                    Attribute::Type(spec.type_ref(ctx)),
                ])
            })
            .collect(),
    );

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

        // Should have the canonical field layout.
        let fields = data.attrs.get(&Symbol::new("fields")).unwrap();
        match fields {
            Attribute::List(list) => {
                assert_eq!(list.len(), MARKER_FIELD_COUNT);
                for (idx, spec) in MARKER_FIELDS.into_iter().enumerate() {
                    assert_eq!(spec.field.index() as usize, idx);
                    let Attribute::List(field_attr) = &list[idx] else {
                        panic!("expected list attribute for field {idx}");
                    };
                    assert_eq!(
                        field_attr.first(),
                        Some(&Attribute::Symbol(Symbol::new(spec.symbol_name)))
                    );
                }
            }
            _ => panic!("expected list attribute for fields"),
        }
    }

    #[test]
    fn test_marker_field_indices_are_canonical() {
        assert_eq!(MarkerField::AbilityId.index(), 0);
        assert_eq!(MarkerField::PromptTag.index(), 1);
        assert_eq!(MarkerField::TrDispatchFn.index(), 2);
        assert_eq!(MarkerField::HandlerDispatch.index(), 3);
    }

    #[test]
    fn test_marker_field_specs_are_canonical() {
        assert_eq!(MARKER_FIELDS.len(), 4);
        assert_eq!(
            MARKER_FIELDS,
            [
                MarkerFieldSpec {
                    field: MarkerField::AbilityId,
                    symbol_name: "ability_id",
                    field_type: MarkerFieldType::I32,
                },
                MarkerFieldSpec {
                    field: MarkerField::PromptTag,
                    symbol_name: "prompt_tag",
                    field_type: MarkerFieldType::I32,
                },
                MarkerFieldSpec {
                    field: MarkerField::TrDispatchFn,
                    symbol_name: "tr_dispatch_fn",
                    field_type: MarkerFieldType::Ptr,
                },
                MarkerFieldSpec {
                    field: MarkerField::HandlerDispatch,
                    symbol_name: "handler_dispatch",
                    field_type: MarkerFieldType::Ptr,
                },
            ]
        );
    }

    #[test]
    fn test_evidence_runtime_symbols_are_canonical() {
        assert_eq!(
            evidence_runtime_symbols(),
            [
                Symbol::new(evidence_abi::EMPTY),
                Symbol::new(evidence_abi::LOOKUP),
                Symbol::new(evidence_abi::EXTEND),
                Symbol::new(evidence_abi::LOOKUP_TR),
                Symbol::new(evidence_abi::LOOKUP_HANDLER),
            ]
        );
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
