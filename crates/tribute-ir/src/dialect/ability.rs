//! Ability dialect — evidence-based handler dispatch.

#[trunk_ir::dialect]
mod ability {

    /// Perform an ability operation with explicit evidence and a
    /// ContinuationFrame-carrying continuation closure.
    ///
    /// The continuation closure captures the rest of the computation
    /// after the effect point.
    ///
    /// ```text
    /// ability.perform %evidence, %dispatch, %resume, [%args...]
    ///   { ability_ref: @State, op_name: @get }
    /// ```
    ///
    /// This final form is resultless and lowers to `effect.dispatch_cps`.
    fn perform(
        ability_ref: Attr<Type>,
        op_name: Attr<Symbol>,
        evidence: Value<_>,
        dispatch: Value<_>,
        resume: Value<_>,
        values: Variadic<_>,
    ) {
    }

    /// Resultless proper-tail handler delimiter emitted by
    /// `tribute_control_to_cps`.
    ///
    /// `ability_refs` is ordered to match pairs in `dispatchers`. Each pair is
    /// `(tr_dispatch_fn, handler_dispatch)`, using typed reject closures when a
    /// handled ability has no operation of that kind. `resolve_evidence`
    /// allocates one runtime-unique prompt identity for the delimiter and
    /// shares it across every pair. The body entry block receives the extended
    /// evidence. Every body path ends in a proper tail transfer or
    /// `func.unreachable`.
    fn handle_dispatch(
        ability_refs: Attr<_>,
        evidence: Value<_>,
        prompt_tag: Value<_>,
        dispatchers: Variadic<_>,
    ) {
        #[region(body)]
        {}
    }

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
    fn call(ability_ref: Attr<Type>, op_name: Attr<Symbol>, values: Variadic<_>) -> Value<_> {}
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

/// Canonical target-independent payload product for one ability operation.
///
/// A product is used even for zero and one operand so the effect ABI never
/// needs a null or another in-band sentinel for an empty payload.
pub fn operation_payload_type_ref(
    ctx: &mut trunk_ir::IrContext,
    ability_ref: trunk_ir::TypeRef,
    op_name: Symbol,
    fields: impl IntoIterator<Item = trunk_ir::TypeRef>,
) -> trunk_ir::TypeRef {
    use trunk_ir::types::{Attribute, TypeDataBuilder};

    let ability_name = ctx.get_type(ability_ref).attrs.get_symbol("name");
    let op_idx = compute_op_idx(ability_name, Some(op_name));
    let fields = fields
        .into_iter()
        .enumerate()
        .map(|(index, ty)| {
            Attribute::List(vec![
                Attribute::Symbol(Symbol::from_dynamic(&format!("arg{index}"))),
                Attribute::Type(ty),
            ])
        })
        .collect();
    ctx.intern_type(
        TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("struct"))
            .attr(
                "name",
                Attribute::Symbol(Symbol::from_dynamic(&format!(
                    "__tribute_ability_payload_{op_idx:08x}"
                ))),
            )
            .attr("fields", Attribute::List(fields))
            .build(),
    )
}

/// Compute the stable runtime ability ID for an ability reference type.
pub fn compute_ability_id(ctx: &IrContext, ability_ref: TypeRef) -> u32 {
    use std::hash::{Hash, Hasher};

    let data = ctx.get_type(ability_ref);
    let name = match ability_name(ctx, ability_ref) {
        Some(s) => s,
        _ => panic!(
            "ICE: compute_ability_id: ability type has no name: {:?}",
            data
        ),
    };

    let mut hasher = rustc_hash::FxHasher::default();
    name.hash(&mut hasher);
    data.params.len().hash(&mut hasher);

    for &param in data.params.iter() {
        hash_type(ctx, param).hash(&mut hasher);
    }

    hasher.finish() as u32
}

/// Return the source-level ability name attached to an ability reference type.
pub fn ability_name(ctx: &IrContext, ability_ref: TypeRef) -> Option<Symbol> {
    ctx.get_type(ability_ref).attrs.get_symbol("name")
}

/// Build an `arith.const` for the stable runtime ability ID.
pub fn ability_id_const(
    ctx: &mut IrContext,
    loc: Location,
    i32_ty: TypeRef,
    ability_ref: TypeRef,
) -> arith::Const {
    let ability_id = compute_ability_id(ctx, ability_ref);
    arith::Const::operands()
        .value(Attribute::Int(ability_id as i128))
        .results(i32_ty)
        .build(ctx, loc)
}

fn hash_type(ctx: &IrContext, ty: TypeRef) -> u32 {
    use std::hash::{Hash, Hasher};

    let data = ctx.get_type(ty);
    let mut hasher = rustc_hash::FxHasher::default();
    data.dialect.hash(&mut hasher);
    data.name.hash(&mut hasher);
    data.params.len().hash(&mut hasher);

    for &param in data.params.iter() {
        hash_type(ctx, param).hash(&mut hasher);
    }

    hasher.finish() as u32
}

// === Pure operation registrations ===

use trunk_ir::op_interface::{CallableExitModel, CallableExitOps, ControlFlowInterfaceError};

impl CallableExitModel for Perform {
    fn verify_callable_exit(
        &self,
        ctx: &trunk_ir::IrContext,
    ) -> Result<(), ControlFlowInterfaceError> {
        let data = ctx.op(self.op_ref());
        if data.regions.is_empty()
            && ctx.op_operands(self.op_ref()).len() >= 3
            && data.attributes.get_type("ability_ref").is_some()
            && data.attributes.get_symbol("op_name").is_some()
        {
            Ok(())
        } else {
            Err(ControlFlowInterfaceError::new(
                "ability.perform CallableExit has an invalid final CPS shape",
            ))
        }
    }
}

impl CallableExitModel for HandleDispatch {
    fn allows_nested_regions(&self, _ctx: &trunk_ir::IrContext) -> bool {
        true
    }

    fn verify_callable_exit(
        &self,
        ctx: &trunk_ir::IrContext,
    ) -> Result<(), ControlFlowInterfaceError> {
        let data = ctx.op(self.op_ref());
        if data.regions.len() == 1
            && ctx.region(data.regions[0]).blocks.len() == 1
            && ctx.op_operands(self.op_ref()).len() >= 2
            && matches!(
                data.attributes.get("ability_refs"),
                Some(trunk_ir::types::Attribute::List(ability_refs))
                    if ability_refs.iter().all(|ability_ref| matches!(
                        ability_ref,
                        trunk_ir::types::Attribute::Type(_)
                    ))
            )
        {
            Ok(())
        } else {
            Err(ControlFlowInterfaceError::new(
                "ability.handle_dispatch CallableExit has an invalid final delimiter shape",
            ))
        }
    }
}

inventory::submit! { CallableExitOps::register::<Perform>() }
inventory::submit! { CallableExitOps::register::<HandleDispatch>() }

// === ADT Type Functions ===

use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::arith;
use trunk_ir::dialect::core;
use trunk_ir::refs::TypeRef;
use trunk_ir::types::{Attribute, Location, TypeDataBuilder};

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
        ctx.intern_type(TypeDataBuilder::new(dialect, name).build())
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
/// Dispatch fields store erased closure references. Tail-resumptive operations
/// return their source result; general CPS dispatch uses the exact resultless
/// dispatch ABI. Shared lowering installs typed reject closures for missing kinds.
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

    ctx.intern_type(
        TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("struct"))
            .attr("name", Attribute::Symbol(Symbol::new("_Marker")))
            .attr("fields", fields_attr)
            .build(),
    )
}

/// Get the canonical Evidence ADT type — `core.array(Marker)`.
pub fn evidence_adt_type_ref(ctx: &mut IrContext) -> TypeRef {
    let marker_ty = marker_adt_type_ref(ctx);
    core::array(ctx, marker_ty).as_type_ref()
}

/// Check if a type is the marker ADT type (`adt.struct("_Marker", ...)`).
pub fn is_marker_type_ref(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.get_type(ty);
    if data.dialect != Symbol::new("adt") || data.name != Symbol::new("struct") {
        return false;
    }
    data.attrs.get_symbol("name") == Some(Symbol::new("_Marker"))
}

/// Check if a type is the evidence ADT type (`core.array(Marker)`).
pub fn is_evidence_type_ref(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.get_type(ty);
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
    use trunk_ir::op_interface::CallableExitOps;
    use trunk_ir::ops::DialectOp;

    #[test]
    fn test_marker_adt_type_ref() {
        let mut ctx = IrContext::new();
        let marker_ty = marker_adt_type_ref(&mut ctx);

        assert!(is_marker_type_ref(&ctx, marker_ty));

        // Should be an adt.struct
        let data = ctx.get_type(marker_ty);
        assert_eq!(data.dialect, Symbol::new("adt"));
        assert_eq!(data.name, Symbol::new("struct"));

        // Should have name "_Marker"
        assert_eq!(data.attrs.get_symbol("name"), Some(Symbol::new("_Marker")));

        // Should have the canonical field layout.
        let fields = data.attrs.get("fields").unwrap();
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
        let data = ctx.get_type(evidence_ty);
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
        let other_struct = ctx.intern_type(
            TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("struct"))
                .attr("name", Attribute::Symbol(Symbol::new("OtherStruct")))
                .build(),
        );
        assert!(!is_marker_type_ref(&ctx, other_struct));

        // Non-struct type should return false
        let i32_ty = ctx.intern_type(TypeDataBuilder::new("core", "i32").build());
        assert!(!is_marker_type_ref(&ctx, i32_ty));
    }

    #[test]
    fn test_is_evidence_type_ref_negative() {
        let mut ctx = IrContext::new();

        // Array of non-marker should return false
        let i32_ty = ctx.intern_type(TypeDataBuilder::new("core", "i32").build());
        let other_array = core::array(&mut ctx, i32_ty).as_type_ref();
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

    #[test]
    fn callable_exit_rejects_malformed_ability_refs_attribute() {
        for ability_refs in ["@not_a_list", "[1]"] {
            let mut ctx = IrContext::new();
            let module = trunk_ir::parser::parse_test_module(
                &mut ctx,
                &format!(
                    r#"core.module @test {{
  func.func @main(%evidence: core.ptr, %prompt: core.i32) {{
    ability.handle_dispatch %evidence, %prompt {{ability_refs = {ability_refs}}} {{
      func.unreachable
    }}
  }}
}}"#
                ),
            );
            let function = trunk_ir::dialect::func::Func::from_op(&ctx, module.ops(&ctx)[0])
                .expect("function");
            let handle = ctx.block(ctx.region(function.body(&ctx)).blocks[0]).ops[0];

            assert!(CallableExitOps::exits_callable(&ctx, handle).is_err());
        }
    }
}
