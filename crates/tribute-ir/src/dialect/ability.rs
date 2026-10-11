//! Ability dialect — evidence-based handler dispatch.

mod frames;

pub use frames::{CpsClosure, HandlerBinding, NotNever, OperationKind};

use super::closure::Closure;
use super::tribute_control::EvidenceStep;

#[trunk_ir::dialect]
mod ability {

    /// One ability instance the module names, and its operations.
    ///
    /// `sym_name` is the instance key. The `operations` region holds one
    /// `ability.operation` per operation of the instance.
    fn decl(sym_name: Attr<String>) {
        #[region(operations)]
        {}
    }

    /// One operation of the enclosing `ability.decl`, with the parameter and
    /// result types of that instance. `kind` is `fn` or `op`.
    fn operation(
        op_name: Attr<String>,
        kind: Attr<String>,
        param_types: Attr<[Type]>,
        result_type: Attr<Type>,
    ) {
    }

    /// Resultless proper-tail handler delimiter emitted by
    /// `lower_continuation_frames`.
    ///
    /// `ability_refs` is ordered to match `dispatchers`: one `tr_dispatch_fn`
    /// per handled ability instance, which rejects when the instance has no
    /// `fn` operation handled here. `resolve_evidence` allocates one
    /// runtime-unique prompt identity for the delimiter and shares it across
    /// every instance. The body entry block receives the extended evidence. Every body path ends in a proper tail transfer or
    /// `func.unreachable`.
    fn handle_dispatch(
        ability_refs: Attr<[SymbolRef]>,
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
    ///   { ability_ref = @State, op_name = "get" }
    /// ```
    ///
    /// Lowered to: evidence lookup → tr_dispatch_fn(op_idx, value) → result.
    fn call(ability_ref: Attr<SymbolRef>, op_name: Attr<String>, values: Variadic<_>) -> Value<_> {}

    // === Abstract continuation frames ===
    //
    // `tribute_control_to_cps` builds continuations through these operations
    // and `lower_continuation_frames` expands them into frame layouts and
    // closures.

    /// Opaque `ContinuationFrame<Result>`: where a CPS computation delivers
    /// its value and sends the operations it performs.
    struct Frame<Result>;

    /// Frame that enters `continuation` with a value, wrapping `outer`.
    ///
    /// `continuation` is `(Evidence, outer frame, value) -> core.never`. When
    /// the frame is resumed, `evidence_plan` selects the evidence passed to
    /// the resumed inner computation. The produced frame's type is not
    /// inferred: it is built from the continuation's value type.
    #[verify]
    fn suffix_frame<F: Frame, C, G: Frame>(
        evidence_plan: Option<Attr<[EvidenceStep]>>,
        evidence: Value<Evidence>,
        outer: Value<F>,
        continuation: Value<C>,
    ) -> Value<G>
    where
        C: CpsClosure<Inputs = (Evidence, F, G::Result)>,
    {
    }

    /// Deliver `value` to the frame's completion.
    fn exit<F: Frame>(frame: Value<F>, value: Value<F::Result>) {}

    /// Install one layer of a handle around `body`, exiting to `exit`.
    ///
    /// `arms` holds one handler-arm closure per `handlers` entry, in the
    /// same order. The body receives the extended evidence and the frame of
    /// the installed layer, whose completion is `completion`.
    #[verify]
    fn handle<F: Frame>(
        handlers: Attr<[HandlerBinding]>,
        evidence_plan: Option<Attr<[EvidenceStep]>>,
        evidence: Value<Evidence>,
        exit: Value<F>,
        completion: Value<CpsClosure>,
        arms: Variadic<Closure>,
    ) {
        #[region(body)]
        {}
    }

    /// Perform a general operation through the frame's dispatcher.
    ///
    /// `resumption` is `(Evidence, frame, operation result) -> core.never`,
    /// the rest of the computation with no one-shot check. Expanding the
    /// operation adds the check.
    fn perform<F: Frame, C>(
        ability_ref: Attr<SymbolRef>,
        op_name: Attr<String>,
        evidence: Value<Evidence>,
        frame: Value<F>,
        resumption: Value<C>,
        values: Variadic<_>,
    ) where
        C: CpsClosure<Inputs = (Evidence, F, impl NotNever)>,
    {
    }

    /// Perform an operation returning `core.never` through the frame's
    /// dispatcher, with no resumption.
    fn abort<F: Frame>(
        ability_ref: Attr<SymbolRef>,
        op_name: Attr<String>,
        evidence: Value<Evidence>,
        frame: Value<F>,
        values: Variadic<_>,
    ) {
    }

    /// Run the CPS computation `body` to completion in a flow that is not
    /// Cps, and yield its answer.
    ///
    /// `body` is `(Evidence, frame) -> core.never` and the result is the
    /// answer of that frame. `evidence` is the flow's evidence, or
    /// `effect.initial_evidence` in a `Direct` flow. Frame expansion lowers
    /// the operation to `effect.delimit`.
    fn delimit<C, F: Frame>(body: Value<C>, evidence: Value<Evidence>) -> Value<F::Result>
    where
        C: CpsClosure<Inputs = (Evidence, F)>,
    {
    }
}

// === Hash-Based Dispatch ===

/// The index of operation `op_name` of the ability instance `ability`.
///
/// It is a hash of the instance's symbol and the operation name, so perform
/// sites and handler dispatch agree without consulting the declaration.
pub fn compute_op_idx(ability: &SymbolPath, op_name: &str) -> u32 {
    use std::hash::{Hash, Hasher};

    let mut hasher = rustc_hash::FxHasher::default();
    ability.to_string().hash(&mut hasher);
    op_name.hash(&mut hasher);

    (hasher.finish() % 0x7FFFFFFF) as u32
}

/// Canonical target-independent payload product for one ability operation.
///
/// A product is used even for zero and one operand so the effect ABI never
/// needs a null or another in-band sentinel for an empty payload.
pub fn operation_payload_type_ref(
    ctx: &mut trunk_ir::IrContext,
    ability: &SymbolPath,
    op_name: StringRef,
    fields: impl IntoIterator<Item = trunk_ir::TypeRef>,
) -> trunk_ir::TypeRef {
    let op_idx = compute_op_idx(ability, ctx.str(op_name));
    let fields = fields
        .into_iter()
        .enumerate()
        .map(|(index, ty)| (format!("arg{index}"), ty));
    crate::dialect::adt::struct_type(
        ctx,
        format!("__tribute_ability_payload_{op_idx:08x}"),
        fields,
        trunk_ir::types::AttributeMap::new(),
    )
    .as_type_ref()
}

/// The runtime id of the ability instance `ability`: a hash of its symbol,
/// which is the instance key the frontend derives from the source arguments.
pub fn compute_ability_id(ability: &SymbolPath) -> u32 {
    use std::hash::{Hash, Hasher};

    let mut hasher = rustc_hash::FxHasher::default();
    ability.to_string().hash(&mut hasher);

    // Negative identifiers are the row tail slots (`tail_slot_id`).
    (hasher.finish() as u32) >> 1
}

/// The marker slot that holds the evidence of a callable's row tail `index`.
///
/// A row tail slot is a marker whose `outer` field is the tail's evidence.
/// Its identifier is negative, so it never equals an ability's.
pub fn tail_slot_id(index: u32) -> i32 {
    let index = i32::try_from(index).expect("row tail index fits in i32");
    -1 - index
}

/// Remove the `ability.decl` definitions of `module`, once nothing names an
/// ability instance any more.
pub fn remove_declarations(ctx: &mut IrContext, module: trunk_ir::rewrite::Module) {
    for op in module.ops_snapshot(ctx) {
        if <Decl as trunk_ir::ops::DialectOp>::matches(ctx, op) {
            trunk_ir::rewrite::helpers::erase_op(ctx, op);
        }
    }
}

/// Build an `arith.const` for the stable runtime ability ID.
pub fn ability_id_const(
    ctx: &mut IrContext,
    loc: Location,
    i32_ty: TypeRef,
    ability: &SymbolPath,
) -> arith::Const {
    let ability_id = compute_ability_id(ability);
    arith::Const::operands()
        .value(Attribute::Int(ability_id as i128))
        .results(i32_ty)
        .build(ctx, loc)
}

// === Pure operation registrations ===

use trunk_ir::attr_kind::{SymbolRef, Type};
use trunk_ir::op_interface::{CallableExitModel, CallableExitOps, ControlFlowInterfaceError};

impl CallableExitModel for Perform {
    fn verify_callable_exit(
        &self,
        ctx: &trunk_ir::IrContext,
    ) -> Result<(), ControlFlowInterfaceError> {
        let data = ctx.op(self.op_ref());
        if !ctx.op_has_regions(self.op_ref())
            && ctx.op_operands(self.op_ref()).len() >= 3
            && data.attributes.get_symbol_ref("ability_ref").is_some()
            && data.attributes.get_string_ref("op_name").is_some()
        {
            Ok(())
        } else {
            Err(ControlFlowInterfaceError::new(
                "ability.perform CallableExit has an invalid CPS shape",
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
        if ctx.op_region_count(self.op_ref()) == 1
            && ctx
                .op_region(self.op_ref(), 0)
                .is_some_and(|region| ctx.region(region).blocks.len() == 1)
            && ctx.op_operands(self.op_ref()).len() >= 2
            && matches!(
                data.attributes.get("ability_refs"),
                Some(trunk_ir::types::Attribute::List(ability_refs))
                    if ability_refs.iter().all(|ability_ref| matches!(
                        ability_ref,
                        trunk_ir::types::Attribute::SymbolRef(_)
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

impl CallableExitModel for Exit {}

impl CallableExitModel for Abort {}

impl CallableExitModel for Handle {
    fn allows_nested_regions(&self, _ctx: &trunk_ir::IrContext) -> bool {
        true
    }
}

inventory::submit! { CallableExitOps::register::<Perform>() }
inventory::submit! { CallableExitOps::register::<HandleDispatch>() }
inventory::submit! { CallableExitOps::register::<Exit>() }
inventory::submit! { CallableExitOps::register::<Abort>() }
inventory::submit! { CallableExitOps::register::<Handle>() }

// === ADT Type Functions ===

use crate::runtime_layout;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::arith;
use trunk_ir::refs::TypeRef;
use trunk_ir::types::{Attribute, Location, StringRef, TypeDataBuilder};
use trunk_ir::{Symbol, SymbolPath};

/// Canonical field identifiers for the `_Marker` ADT used by ability evidence.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkerField {
    AbilityId = 0,
    PromptTag = 1,
    TrDispatchFn = 2,
    Shadowed = 3,
    Outer = 4,
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
pub const MARKER_FIELDS: [MarkerFieldSpec; 5] = [
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
        field: MarkerField::Shadowed,
        symbol_name: "shadowed",
        field_type: MarkerFieldType::Ptr,
    },
    MarkerFieldSpec {
        field: MarkerField::Outer,
        symbol_name: "outer",
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
    pub const MASK: &str = "__tribute_evidence_mask";
    pub const DUP: &str = "__tribute_evidence_dup";
    pub const OUTER: &str = "__tribute_evidence_outer";
    pub const LOOKUP_TR: &str = "__tribute_evidence_lookup_tr";
    pub const TAIL: &str = "__tribute_evidence_tail";
    pub const WITH_TAIL: &str = "__tribute_evidence_with_tail";
    pub const PUSH: &str = "__tribute_evidence_push";
}

pub fn evidence_runtime_symbols() -> [Symbol; 10] {
    [
        Symbol::new(evidence_abi::EMPTY),
        Symbol::new(evidence_abi::LOOKUP),
        Symbol::new(evidence_abi::EXTEND),
        Symbol::new(evidence_abi::MASK),
        Symbol::new(evidence_abi::DUP),
        Symbol::new(evidence_abi::OUTER),
        Symbol::new(evidence_abi::LOOKUP_TR),
        Symbol::new(evidence_abi::TAIL),
        Symbol::new(evidence_abi::WITH_TAIL),
        Symbol::new(evidence_abi::PUSH),
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
///     shadowed: ptr,
///     outer: ptr,
/// }
/// ```
///
/// `shadowed` is the marker of the same ability this one shadows, or null.
/// `outer` is the evidence the handler was installed on.
/// `tr_dispatch_fn` stores an erased closure reference whose tail-resumptive
/// operations return their source result. General operations read only the
/// prompt and dispatch through the continuation frame.
pub fn marker_adt_type_ref(ctx: &mut IrContext) -> TypeRef {
    let fields: Vec<_> = MARKER_FIELDS
        .into_iter()
        .map(|spec| (spec.symbol_name, spec.type_ref(ctx)))
        .collect();
    let mut attrs = trunk_ir::types::AttributeMap::new();
    attrs.insert(
        runtime_layout::LAYOUT_ATTR,
        ctx.string_attr(runtime_layout::EVIDENCE_MARKER),
    );
    crate::dialect::adt::struct_type(ctx, "_Marker", fields, attrs).as_type_ref()
}

/// Get the canonical Evidence ADT type — `core.array<Marker>` carrying the
/// evidence runtime layout identifier.
pub fn evidence_adt_type_ref(ctx: &mut IrContext) -> TypeRef {
    let marker_ty = marker_adt_type_ref(ctx);
    let layout = ctx.string_attr(runtime_layout::EVIDENCE);
    ctx.intern_type(
        TypeDataBuilder::new(Symbol::new("core"), Symbol::new("array"))
            .param(marker_ty)
            .attr(runtime_layout::LAYOUT_ATTR, layout)
            .build(),
    )
}

/// Whether a type is the evidence marker layout, identified by its runtime
/// layout attribute.
pub fn is_marker_type_ref(ctx: &IrContext, ty: TypeRef) -> bool {
    runtime_layout::has_runtime_layout(ctx, ty, runtime_layout::EVIDENCE_MARKER)
}

/// Whether a type is the evidence array layout, identified by its runtime
/// layout attribute.
pub fn is_evidence_type_ref(ctx: &IrContext, ty: TypeRef) -> bool {
    runtime_layout::has_runtime_layout(ctx, ty, runtime_layout::EVIDENCE)
}

/// The evidence type, identified by its runtime layout.
pub struct Evidence;

impl trunk_ir::type_constraint::TypeConstraint for Evidence {
    const DESC: &'static trunk_ir::type_constraint::ConstraintDesc =
        &trunk_ir::type_constraint::ConstraintDesc {
            name: "Evidence",
            exact: false,
            projections: &[],
            matches: is_evidence_type_ref,
            project: |_, _, _| None,
            fixed: None,
        };
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::dialect::core;

    #[test]
    fn evidence_layouts_are_identified_by_layout_not_name() {
        let mut ctx = IrContext::new();
        let name_attr = ctx.string_attr("_Marker");
        let named_marker = ctx.intern_type(
            TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("struct"))
                .attr("name", name_attr)
                .build(),
        );
        let plain_array = core::array(&mut ctx, named_marker).as_type_ref();
        assert!(!is_marker_type_ref(&ctx, named_marker));
        assert!(!is_evidence_type_ref(&ctx, plain_array));

        let marker = marker_adt_type_ref(&mut ctx);
        let evidence = evidence_adt_type_ref(&mut ctx);
        assert!(is_marker_type_ref(&ctx, marker));
        assert!(is_evidence_type_ref(&ctx, evidence));
        assert_eq!(ctx.get_type(evidence).params.as_slice(), [marker]);
    }
    use trunk_ir::op_interface::CallableExitOps;
    use trunk_ir::ops::{DialectOp, DialectType};

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
        assert_eq!(data.attrs.get_str(&ctx, "name"), Some("_Marker"));

        // Should have the canonical field layout.
        let marker = crate::dialect::adt::Struct::from_type_ref(&ctx, marker_ty).unwrap();
        assert_eq!(marker.field_count(&ctx), MARKER_FIELD_COUNT);
        for (idx, spec) in MARKER_FIELDS.into_iter().enumerate() {
            assert_eq!(spec.field.index() as usize, idx);
            assert_eq!(marker.field_name(&ctx, idx), Some(spec.symbol_name));
        }
    }

    #[test]
    fn test_marker_field_indices_are_canonical() {
        assert_eq!(MarkerField::AbilityId.index(), 0);
        assert_eq!(MarkerField::PromptTag.index(), 1);
        assert_eq!(MarkerField::TrDispatchFn.index(), 2);
        assert_eq!(MarkerField::Shadowed.index(), 3);
        assert_eq!(MarkerField::Outer.index(), 4);
    }

    #[test]
    fn test_marker_field_specs_are_canonical() {
        assert_eq!(MARKER_FIELDS.len(), 5);
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
                    field: MarkerField::Shadowed,
                    symbol_name: "shadowed",
                    field_type: MarkerFieldType::Ptr,
                },
                MarkerFieldSpec {
                    field: MarkerField::Outer,
                    symbol_name: "outer",
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
                Symbol::new(evidence_abi::MASK),
                Symbol::new(evidence_abi::DUP),
                Symbol::new(evidence_abi::OUTER),
                Symbol::new(evidence_abi::LOOKUP_TR),
                Symbol::new(evidence_abi::TAIL),
                Symbol::new(evidence_abi::WITH_TAIL),
                Symbol::new(evidence_abi::PUSH),
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
        let name_attr = ctx.string_attr("OtherStruct");
        let other_struct = ctx.intern_type(
            TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("struct"))
                .attr("name", name_attr)
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
