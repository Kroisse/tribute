//! Ability dialect — evidence-based handler dispatch.

mod frames;
mod syntax;

pub use crate::dialect::tribute_control::{HandlerBinding, OperationKind};
pub use frames::{CpsClosure, NotNever};
pub use syntax::{Dispatchers, HandlerArms};

use super::closure::Closure;
use super::tribute_control::EvidenceStep;
use trunk_ir::op_flag::{NoTerminator, Terminator};

#[trunk_ir::dialect]
mod ability {

    /// One ability instance the module names, and its operations.
    ///
    /// ```text
    /// ability.decl @State {
    ///   ability.operation {kind = "op", op_name = "get", param_types = [], result_type = ResultType}
    ///   ability.operation {kind = "op", op_name = "set", param_types = [ValueType], result_type = core.nil}
    /// }
    /// ```
    ///
    /// - **Role:** a symbol definition that declares one ability instance the
    ///   module uses, with all of its operations. The frontend emits exactly
    ///   one into the module body for each instance that an operation or
    ///   evidence plan in the module names. The declaration is part of the
    ///   module, not a table outside the IR, so textual IR alone carries every
    ///   fact verification needs.
    /// - **Unit:** a declaration is per **instance**, not per ability. The
    ///   operation signatures are logical types instantiated with the
    ///   instance's type arguments. Logical types do not represent source type
    ///   variables, so there are no generic declarations.
    /// - **Name:** `sym_name` is the instance key. It is built from the
    ///   checked source arguments and distinguishes different source arguments
    ///   with the same lowered representation. The run-time ability id is
    ///   computed from the same key.
    /// - **References:** every place that names an ability instance is a
    ///   symbol reference to this definition: `perform`, handler arms, the
    ///   `ability` of `ability.*` and `effect.*` operations, the `abilities`
    ///   of `ability.handle_dispatch`, and evidence plan elements. No type
    ///   denotes an instance, and an instance's type arguments do not appear
    ///   in the IR. Value type conversion cannot change an instance's
    ///   identity.
    /// - **Region:** one `operations` region whose block holds only
    ///   `ability.operation`. An `ability.operation` has no `sym_name`.
    ///   Different instances declare the same operation names, so operations
    ///   do not enter the symbol namespace; they are found within the
    ///   declaration by `op_name`.
    /// - **Verification:** `op_name` does not repeat within one declaration.
    ///   The uniqueness of symbol definitions guarantees one declaration per
    ///   instance. `perform` and handler arms must agree with the kind,
    ///   parameter types, and result type of the declaration their `ability`
    ///   and `op_name` name.
    /// - **Lifetime:** a declaration stays in the module while an operation
    ///   naming the instance remains. The target's evidence lowering consumes
    ///   the last reference and removes the declaration. A declaration is not
    ///   a callable definition, so global DCE does not remove it.
    #[flags(NoTerminator)]
    #[asm_format(symbol(sym_name) attr_dict region(operations))]
    fn decl(sym_name: Attr<String>) {
        #[region(operations)]
        {}
    }

    /// One operation of the enclosing `ability.decl`, with the parameter and
    /// result types of that instance.
    ///
    /// Its attributes are `op_name: String`, `kind: String` (`fn` or `op`),
    /// `param_types: [Type]`, and `result_type: Type`: the parameter types in
    /// source declaration order and the result type. These types are at
    /// module level, so they cannot have a row tail.
    #[asm_format(attr_dict_bare)]
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
    /// `abilities` is ordered to match `dispatchers`: one `tr_dispatch_fn`
    /// per handled ability instance, which rejects when the instance has no
    /// `fn` operation handled here. `resolve_evidence` allocates one
    /// runtime-unique prompt identity for the delimiter and shares it across
    /// every instance. The body entry block receives the extended evidence. Every body path ends in a proper tail transfer or
    /// `func.unreachable`.
    ///
    /// ```text
    /// ability.handle_dispatch %evidence, %prompt_tag [@State: %dispatcher0, @Console: %dispatcher1] {
    /// ^body(%ev: Evidence):
    ///   ...
    /// }
    /// ```
    ///
    /// The format writes each ability with its dispatcher. The verifier
    /// checks that there is one dispatcher per ability.
    #[verify]
    #[asm_format(evidence "," prompt_tag custom(Dispatchers(abilities, dispatchers)) attr_dict region(body))]
    #[flags(Terminator)]
    #[implements(CallableExit)]
    fn handle_dispatch(
        abilities: Attr<[SymbolRef]>,
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
    /// %result = ability.call @State @get(%args...) : (ArgType, ...) -> ResultType
    /// ```
    ///
    /// Lowered to: evidence lookup → tr_dispatch_fn(op_idx, value) → result.
    #[asm_format(ability symbol(op_name) "(" values ")" attr_dict ":" functional_type(values, result))]
    fn call(ability: Attr<SymbolRef>, op_name: Attr<String>, values: Variadic<_>) -> Value<_> {}

    // === Abstract continuation frames ===
    //
    // `tribute_control_to_cps` builds continuations through these operations
    // and `lower_continuation_frames` expands them into frame layouts and
    // closures.

    /// Opaque `ContinuationFrame<Result>`: where a CPS computation delivers
    /// its value and sends the operations it performs.
    ///
    /// `ability.frame<R>` is an opaque type whose type parameter `result` is
    /// the answer type `R`. The frame parameter of a `Cps` callable's
    /// `CallableAbi`, completions, and resumptions use it. No operation
    /// extracts its components.
    struct Frame<Result>;

    /// Frame that enters `continuation` with a value, wrapping `outer`.
    ///
    /// ```text
    /// %f = ability.suffix_frame %ev, %outer, %k {evidence_plan = [...]} : !ability.frame<V>
    /// ```
    ///
    /// `evidence` is Evidence, `outer` is an `ability.frame<R>`, and
    /// `continuation` is a `Completion<V, R>`, the `Cps` closure
    /// `(Evidence, ability.frame<R>, V) -> core.never`; the result is an
    /// `ability.frame<V>`. When the frame is resumed, the optional
    /// `evidence_plan` selects the evidence passed to the resumed inner
    /// computation, in the syntax of `design/tribute-dialects.md`, "Evidence 선택 속성".
    /// A frame for a structured branch has none. The produced frame's type is
    /// not inferred: it is built from the continuation's value type.
    #[verify]
    #[asm_format(evidence "," outer "," continuation attr_dict_bare ":" type(result))]
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
    ///
    /// ```text
    /// ability.exit %f, %value
    /// ```
    ///
    /// A resultless terminator that takes an `ability.frame<R>` and one value
    /// of type `R`. It appears only in a `Cps` callable, a completion, or a
    /// generated continuation body.
    #[flags(Terminator)]
    #[implements(CallableExit)]
    #[asm_format(frame "," value attr_dict_bare)]
    fn exit<F: Frame>(frame: Value<F>, value: Value<F::Result>) {}

    /// Install one layer of a handle around `body`, exiting to `exit`.
    ///
    /// `arms` holds one handler-arm closure per `handlers` entry, in the
    /// same order. The body receives the extended evidence and the frame of
    /// the installed layer, whose completion is `completion`.
    ///
    /// ```text
    /// ability.handle %outer_ev, %exit, %completion [
    ///   op @State @get : core.i32 = %get,
    ///   fn @State @peek : core.i32 = %peek
    /// ] {
    /// ^body(%ev: Evidence, %f: !ability.frame<M>):
    ///   ...
    /// }
    /// ```
    ///
    /// The format writes each binding with its arm, as
    /// `kind @Ability @op_name : operation_result_type = %arm`.
    ///
    /// A resultless terminator with one region. `evidence` is the Evidence
    /// where the handle is installed, `exit` is an `ability.frame<R>`, and
    /// `completion` is a `Completion<M, R>`, the `Cps` closure
    /// `(Evidence, ability.frame<R>, M) -> core.never`. Each `handlers`
    /// element is a dictionary with `ability` (SymbolRef), `op_name`
    /// (String), `kind` (`"fn"` or `"op"`), and `operation_result_type`
    /// (Type), and one `(ability, op_name)` appears at most once. With `I`
    /// the `operation_result_type` and `M` the value type of the body frame,
    /// a `kind = "op"` arm is the following `Cps` closure, where
    /// `ResumeExact<I, M>` is `(Evidence, ability.frame<M>, I) -> core.never`;
    /// the token is present only when `I` is not `core.never`:
    ///
    /// ```text
    /// (Evidence, ability.frame<R>, operation arguments..., [ResumeExact<I, M>])
    ///   -> core.never
    /// ```
    ///
    /// The token resumes the computation from the perform site to the end of
    /// the body, without this handle and without `completion`. It takes the
    /// evidence and the `ability.frame<M>` its caller supplies, and the
    /// computation runs with them.
    ///
    /// A `kind = "fn"` arm is the `EvidenceDirect` closure
    /// `(Evidence, operation arguments...) -> I`. A closure type records no
    /// convention, so verification checks each callable's shape, and the
    /// handle enters an arm through the convention its `kind` names. The optional
    /// `evidence_plan` holds only `mask`. The body entry block receives the
    /// extended Evidence and an `ability.frame<M>`, and every path ends in a
    /// proper tail transfer or `func.unreachable`.
    #[verify]
    #[asm_format(evidence "," exit "," completion custom(HandlerArms(handlers, arms)) attr_dict region(body))]
    #[flags(Terminator)]
    #[implements(CallableExit)]
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
    ///
    /// ```text
    /// ability.perform @State @get(%arg0, ...) [%ev, %f, %resumption]
    ///   : (ArgType, ...) -> ()
    /// ```
    ///
    /// A resultless terminator that takes Evidence, an `ability.frame<R>`, a
    /// raw `ResumeExact<I, R>`, and the operation arguments. `I` is not
    /// `core.never`.
    #[asm_format(ability symbol(op_name) "(" values ")" "[" evidence "," frame "," resumption "]" attr_dict ":" functional_type(values))]
    #[flags(Terminator)]
    #[implements(CallableExit)]
    fn perform<F: Frame, C>(
        ability: Attr<SymbolRef>,
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
    ///
    /// ```text
    /// ability.abort @Fail @fail(%arg0, ...) [%ev, %f] : (ArgType, ...) -> ()
    /// ```
    ///
    /// The shape of `ability.perform` without the resumption, used only for
    /// source `op -> Never`.
    #[asm_format(ability symbol(op_name) "(" values ")" "[" evidence "," frame "]" attr_dict ":" functional_type(values))]
    #[flags(Terminator)]
    #[implements(CallableExit)]
    fn abort<F: Frame>(
        ability: Attr<SymbolRef>,
        op_name: Attr<String>,
        evidence: Value<Evidence>,
        frame: Value<F>,
        values: Variadic<_>,
    ) {
    }

    /// Run the CPS computation `body` to completion in a flow that is not
    /// Cps, and yield its answer.
    ///
    /// ```text
    /// %r = ability.delimit %body, %ev : R
    /// ```
    ///
    /// The value delimiter of `design/cps-effects.md`, "값 delimiter".
    /// `body` is the `Cps` closure `(Evidence, ability.frame<R>) -> core.never`
    /// and the result is one `R`. `evidence` is the flow's evidence, or the
    /// result of `effect.initial_evidence` in a `Direct` flow, which takes no
    /// evidence. It is not a terminator and appears only in a flow that is
    /// not `Cps`. Frame expansion lowers it to `effect.delimit` with the same
    /// operands and result.
    #[asm_format(body "," evidence attr_dict_bare ":" type(result))]
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

/// Canonical target-independent payload product of an ability operation's
/// operands of types `fields`: a literal `adt.struct`.
///
/// A product is used even for zero and one operand so the effect ABI never
/// needs a null or another in-band sentinel for an empty payload.
pub fn operation_payload_type_ref(
    ctx: &mut trunk_ir::IrContext,
    fields: impl IntoIterator<Item = trunk_ir::TypeRef>,
) -> trunk_ir::TypeRef {
    crate::dialect::adt::Struct::literal(ctx, fields, trunk_ir::types::AttributeMap::new())
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

/// Build an `arith.constant` for the stable runtime ability ID.
pub fn ability_id_const(
    ctx: &mut IrContext,
    loc: Location,
    i32_ty: TypeRef,
    ability: &SymbolPath,
) -> arith::Constant {
    let ability_id = compute_ability_id(ability);
    arith::Constant::operands()
        .value(Attribute::Int(ability_id as i128))
        .results(i32_ty)
        .build(ctx, loc)
}

use trunk_ir::attr_kind::{SymbolRef, Type};
use trunk_ir::op_interface::{CallableExit, ControlFlowInterfaceError};

impl CallableExit for Perform {
    fn verify_callable_exit(
        &self,
        ctx: &trunk_ir::IrContext,
    ) -> Result<(), ControlFlowInterfaceError> {
        let attrs = ctx.op_attrs(self.op_ref());
        if !ctx.op_has_regions(self.op_ref())
            && ctx.op_operands(self.op_ref()).len() >= 3
            && attrs.get_symbol_ref("ability").is_some()
            && attrs.get_string_ref("op_name").is_some()
        {
            Ok(())
        } else {
            Err(ControlFlowInterfaceError::new(
                "ability.perform CallableExit has an invalid CPS shape",
            ))
        }
    }
}

impl trunk_ir::ops::Verify for HandleDispatch {
    /// One dispatcher per ability.
    fn verify(&self, ctx: &trunk_ir::IrContext) -> Result<(), trunk_ir::ops::VerificationError> {
        let abilities = self.abilities(ctx).len();
        let dispatchers = self.dispatchers(ctx).len();
        if abilities != dispatchers {
            return Err(format!("{abilities} abilities but {dispatchers} dispatchers").into());
        }
        Ok(())
    }
}

impl CallableExit for HandleDispatch {
    fn allows_nested_regions(&self, _ctx: &trunk_ir::IrContext) -> bool {
        true
    }

    fn verify_callable_exit(
        &self,
        ctx: &trunk_ir::IrContext,
    ) -> Result<(), ControlFlowInterfaceError> {
        if ctx.op_region_count(self.op_ref()) == 1
            && ctx
                .op_region(self.op_ref(), 0)
                .is_some_and(|region| ctx.region(region).blocks.len() == 1)
            && ctx.op_operands(self.op_ref()).len() >= 2
            && matches!(
                ctx.op_attrs(self.op_ref()).get("abilities"),
                Some(trunk_ir::types::Attribute::List(abilities))
                    if abilities.iter().all(|ability| matches!(
                        ability,
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

impl CallableExit for Exit {}

impl CallableExit for Abort {}

impl CallableExit for Handle {
    fn allows_nested_regions(&self, _ctx: &trunk_ir::IrContext) -> bool {
        true
    }
}

// === ADT Type Functions ===

use trunk_ir::SymbolPath;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::arith;
use trunk_ir::ops::DialectType;
use trunk_ir::refs::TypeRef;
use trunk_ir::types::{Attribute, Location};

/// The evidence marker: the layout of the prelude's
/// `std::runtime::evidence::Marker` the root module `module` publishes in
/// `tribute.types` (`design/cps-effects.md`).
///
/// # Panics
///
/// If `module` publishes no marker.
pub fn marker_type(ctx: &IrContext, module: trunk_ir::rewrite::Module) -> TypeRef {
    crate::runtime_items::runtime_type(ctx, module, crate::runtime_items::evidence::MARKER)
        .expect("the root module publishes the evidence marker in tribute.types")
}

/// The evidence type of the root module `module`: `Slots(Marker)`, an
/// `adt.array` of its [`marker_type`].
pub fn evidence_type(ctx: &mut IrContext, module: trunk_ir::rewrite::Module) -> TypeRef {
    let marker_ty = marker_type(ctx, module);
    crate::dialect::adt::Array::params(marker_ty)
        .intern(ctx)
        .as_type_ref()
}

/// Whether `ty` is a value of the evidence marker of the root module
/// `module`: its layout is the [`marker_type`].
pub fn is_marker_type(ctx: &IrContext, module: trunk_ir::rewrite::Module, ty: TypeRef) -> bool {
    let marker =
        crate::runtime_items::runtime_type(ctx, module, crate::runtime_items::evidence::MARKER);
    marker.is_some() && crate::dialect::adt::layout_of(ctx, ty) == marker
}

/// Whether `ty` is the evidence type of the root module `module`: an
/// `adt.array` of its markers.
pub fn is_evidence_type(ctx: &IrContext, module: trunk_ir::rewrite::Module, ty: TypeRef) -> bool {
    let marker =
        crate::runtime_items::runtime_type(ctx, module, crate::runtime_items::evidence::MARKER);
    marker.is_some_and(|marker| is_evidence_of(ctx, marker, ty))
}

/// Whether `ty` is an `adt.array` of values of the marker layout `marker`,
/// for a holder of the [`marker_type`] that no longer holds its module.
pub fn is_evidence_of(ctx: &IrContext, marker: TypeRef, ty: TypeRef) -> bool {
    crate::dialect::adt::Array::from_type_ref(ctx, ty).is_some_and(|array| {
        crate::dialect::adt::layout_of(ctx, array.element(ctx)) == Some(marker)
    })
}

/// Whether `ty` has the shape of the evidence type: an `adt.array` of
/// values of a nominal struct. Which struct is the marker is the root
/// module's, which [`is_evidence_type`] checks.
pub fn is_evidence_shaped(ctx: &IrContext, ty: TypeRef) -> bool {
    crate::dialect::adt::Array::from_type_ref(ctx, ty).is_some_and(|array| {
        let element = array.element(ctx);
        crate::dialect::adt::layout_of(ctx, element)
            .and_then(|layout| crate::dialect::adt::Struct::from_type_ref(ctx, layout))
            .is_some_and(|layout| layout.is_nominal(ctx))
    })
}

/// A value of the evidence type, by its shape: the schema cannot see the
/// root module, so the passes that hold it check the exact marker
/// (`design/cps-effects.md`).
pub struct Evidence;

impl trunk_ir::type_constraint::TypeConstraint for Evidence {
    const DESC: &'static trunk_ir::type_constraint::ConstraintDesc =
        &trunk_ir::type_constraint::ConstraintDesc {
            name: "Evidence",
            exact: false,
            projections: &[],
            matches: is_evidence_shaped,
            project: |_, _, _| None,
            fixed: None,
        };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evidence_is_identified_by_the_published_marker() {
        let mut ctx = IrContext::new();
        let module = trunk_ir::parser::parse_test_module(
            &mut ctx,
            r#"core.module @test {tribute.types = {"std::runtime::evidence::Marker" = adt.struct<"std::runtime::evidence::Marker">}} {
  !Marker = adt.struct<"std::runtime::evidence::Marker", (core.i32)>
  !Lookalike = adt.struct<"_Marker", (core.i32)>
}"#,
        );
        let marker = marker_type(&ctx, module);
        assert_eq!(Some(marker), ctx.type_alias_by_text("Marker"));
        let evidence = evidence_type(&mut ctx, module);
        assert!(is_marker_type(&ctx, module, marker));
        assert!(is_evidence_type(&ctx, module, evidence));
        assert!(is_evidence_shaped(&ctx, evidence));
        assert_eq!(evidence, evidence_type(&mut ctx, module));

        // Another struct of the same shape is not the marker, although the
        // schema's shape check cannot tell them apart.
        let lookalike = ctx.type_alias_by_text("Lookalike").unwrap();
        let lookalike_array = crate::dialect::adt::Array::params(lookalike)
            .intern(&mut ctx)
            .as_type_ref();
        assert!(!is_marker_type(&ctx, module, lookalike));
        assert!(!is_evidence_type(&ctx, module, lookalike_array));
        assert!(is_evidence_shaped(&ctx, lookalike_array));
        assert!(!is_evidence_type(&ctx, module, marker));
    }

    use trunk_ir::op_interface::DynCallableExit;
    use trunk_ir::ops::DialectOp;

    #[test]
    fn callable_exit_rejects_malformed_abilities_attribute() {
        for abilities in ["@not_a_list", "[1]"] {
            let mut ctx = IrContext::new();
            let module = trunk_ir::parser::parse_test_module(
                &mut ctx,
                &format!(
                    r#"core.module @test {{
  func.func @main(%evidence: core.ptr, %prompt: core.i32) {{
    "ability.handle_dispatch"(%evidence, %prompt) ({{
      func.unreachable
    }}) {{abilities = {abilities}}}
  }}
}}"#
                ),
            );
            let function = trunk_ir::dialect::func::Func::from_op(&ctx, module.ops(&ctx)[0])
                .expect("function");
            let handle = ctx.block(ctx.region(function.body(&ctx)).blocks[0]).ops[0];

            assert!(DynCallableExit::prove(&ctx, handle).is_err());
        }
    }

    #[test]
    fn handle_dispatch_writes_each_ability_with_its_dispatcher() {
        let input = r#"core.module @test {
  func.func @main(%0: core.ptr, %1: core.i32, %2: core.ptr, %3: core.ptr) {
      ability.handle_dispatch %0, %1 [@State: %2, @Console: %3] {
          func.unreachable
      }
  }
}
"#;
        let mut ctx = IrContext::new();
        let module = trunk_ir::parser::parse_test_module(&mut ctx, input);
        let function =
            trunk_ir::dialect::func::Func::from_op(&ctx, module.ops(&ctx)[0]).expect("function");
        let entry = ctx.region(function.body(&ctx)).blocks[0];
        let handle = HandleDispatch::from_op(&ctx, ctx.block(entry).ops[0]).expect("handle");
        let abilities: Vec<_> = handle
            .abilities(&ctx)
            .map(|ability| ability.to_string())
            .collect();
        assert_eq!(abilities, ["State", "Console"]);
        assert_eq!(
            handle.dispatchers(&ctx),
            [ctx.block_arg(entry, 2), ctx.block_arg(entry, 3)]
        );
        assert_eq!(module.op().display(&ctx).to_string(), input);
    }
}
