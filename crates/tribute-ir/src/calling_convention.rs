//! Tribute calling conventions and the IR that records them.

use std::fmt;
use std::str::FromStr;

use trunk_ir::context::IrContext;
use trunk_ir::dialect::{core, func};
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{OpRef, TypeRef};
use trunk_ir::types::Attribute;

use crate::dialect::closure;

/// Exact attribute key of a convention on a callable type or operation.
pub const CALLING_CONVENTION_ATTR: &str = "tribute.calling_convention";

/// The ABI strength required to call a function.
///
/// Ordering is significant: composing requirements selects the stronger
/// convention with [`CallingConvention::join`].
///
/// It displays as, and parses from, its [keyword](Self::keyword).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CallingConvention {
    /// Pure function: source parameters and source result only.
    Direct,
    /// Tail-resumptive effect: evidence parameter, direct source result.
    EvidenceDirect,
    /// General control effect: evidence and a typed ContinuationFrame.
    Cps,
}

impl CallingConvention {
    /// Compose two requirements by selecting the stronger convention.
    pub fn join(self, other: Self) -> Self {
        self.max(other)
    }

    /// Whether the convention carries an evidence parameter.
    pub fn needs_evidence(self) -> bool {
        self >= Self::EvidenceDirect
    }

    /// Whether the convention carries a typed ContinuationFrame.
    pub fn needs_continuation_frame(self) -> bool {
        self == Self::Cps
    }

    /// The environment slot of a closure entry with this convention, which
    /// the convention alone determines.
    pub fn closure_environment_index(self) -> usize {
        usize::from(self.needs_evidence())
    }

    /// Every convention, in convention table order.
    pub const ALL: [Self; 3] = [Self::Direct, Self::EvidenceDirect, Self::Cps];

    /// The position of this convention's entry in a closure's convention
    /// table (`new-plans/ir.md`, "Convention entry").
    pub fn slot(self) -> u32 {
        match self {
            Self::Direct => 0,
            Self::EvidenceDirect => 1,
            Self::Cps => 2,
        }
    }

    /// The keyword that spells this convention, both as the stored
    /// attribute value and in the `convention(..)` syntax.
    pub fn keyword(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::EvidenceDirect => "evidence_direct",
            Self::Cps => "cps",
        }
    }

    /// The attribute value that stores this convention.
    pub(crate) fn to_attribute(self, ctx: &mut IrContext) -> Attribute {
        ctx.string_attr(self.keyword())
    }

    /// The convention a stored attribute value spells; `None` for any value
    /// that is not one of the keywords.
    pub(crate) fn from_attribute(ctx: &IrContext, value: &Attribute) -> Option<Self> {
        match value {
            Attribute::String(keyword) => ctx.str(*keyword).parse().ok(),
            _ => None,
        }
    }
}

/// `Attr<CallingConvention>`: a keyword string. The schema checks only that
/// the value is a string; an operation declaring it verifies the keyword.
impl trunk_ir::attr_kind::AttrKind for CallingConvention {
    const KIND: trunk_ir::op_schema::AttributeKind = trunk_ir::op_schema::AttributeKind::String;
    type Out<'ctx> = CallingConvention;
    type In = CallingConvention;

    fn read<'ctx>(ctx: &'ctx IrContext, attr: &'ctx Attribute) -> CallingConvention {
        Self::from_attribute(ctx, attr).expect("unverified calling convention keyword")
    }

    fn write(ctx: &mut IrContext, value: CallingConvention) -> Attribute {
        value.to_attribute(ctx)
    }
}

impl fmt::Display for CallingConvention {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.keyword())
    }
}

/// A string that is not a calling-convention keyword.
#[derive(Clone, Copy, Debug, PartialEq, Eq, derive_more::Display, derive_more::Error)]
#[display("unknown calling convention")]
pub struct UnknownConvention;

impl FromStr for CallingConvention {
    type Err = UnknownConvention;

    fn from_str(keyword: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|convention| convention.keyword() == keyword)
            .ok_or(UnknownConvention)
    }
}

/// A convention an operation records that is not a valid one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, derive_more::Display, derive_more::Error)]
pub enum ConventionError {
    /// The recorded value is not one of the convention keywords.
    #[display("malformed calling-convention metadata")]
    Malformed,
    /// The operation does not implement [`ConventionBearing`].
    #[display("calling-convention metadata on an operation that cannot carry it")]
    Unregistered,
}

/// A physical operation that records the calling convention of the callable
/// it defines or transfers to, from CPS legalization until the last reader
/// inside the representation/ABI boundary consumes it
/// (`new-plans/cps-effects.md`, "CPS 변환의 호출 대상 형상").
///
/// Only these operations carry a convention, and passes read, write, and
/// consume it only through this interface. A `closure.lambda` declares the
/// convention of each body in its own `conventions` attribute instead.
#[trunk_ir::op_interface]
pub trait ConventionBearing: DialectOp {
    /// The recorded convention; `Ok(None)` when the operation records none.
    fn calling_convention(
        &self,
        ctx: &IrContext,
    ) -> Result<Option<CallingConvention>, ConventionError> {
        read_convention(ctx, self.op_ref())
    }

    fn set_calling_convention(&self, ctx: &mut IrContext, convention: CallingConvention) {
        let value = convention.to_attribute(ctx);
        ctx.op_attrs_mut(self.op_ref())
            .insert(CALLING_CONVENTION_ATTR, value);
    }

    /// Consume the recorded convention.
    fn remove_calling_convention(&self, ctx: &mut IrContext) {
        ctx.op_attrs_mut(self.op_ref())
            .remove(CALLING_CONVENTION_ATTR);
    }
}

impl DynConventionBearing {
    /// The convention `op` records, rejecting a malformed value and a
    /// convention on an operation that cannot carry one.
    pub fn exact(ctx: &IrContext, op: OpRef) -> Result<Option<CallingConvention>, ConventionError> {
        match Self::get(ctx, op) {
            Some(bearing) => bearing.calling_convention(ctx),
            None if ctx.op_attrs(op).contains_key(CALLING_CONVENTION_ATTR) => {
                Err(ConventionError::Unregistered)
            }
            None => Ok(None),
        }
    }

    /// The valid convention `op` records, if any.
    pub fn of(ctx: &IrContext, op: OpRef) -> Option<CallingConvention> {
        Self::exact(ctx, op).ok().flatten()
    }

    /// Record `convention` on `op`.
    ///
    /// # Panics
    ///
    /// If `op` does not implement [`ConventionBearing`].
    pub fn set(ctx: &mut IrContext, op: OpRef, convention: CallingConvention) {
        Self::get(ctx, op)
            .expect("only a ConventionBearing operation records a calling convention")
            .set_calling_convention(ctx, convention);
    }
}

fn read_convention(
    ctx: &IrContext,
    op: OpRef,
) -> Result<Option<CallingConvention>, ConventionError> {
    let Some(value) = ctx.op_attrs(op).get(CALLING_CONVENTION_ATTR) else {
        return Ok(None);
    };
    CallingConvention::from_attribute(ctx, value)
        .map(Some)
        .ok_or(ConventionError::Malformed)
}

impl ConventionBearing for func::Func {}
impl ConventionBearing for func::Call {}
impl ConventionBearing for func::TailCall {}
impl ConventionBearing for func::CallIndirect {}
impl ConventionBearing for func::TailCallIndirect {}

/// A calling-convention violation of one operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConventionViolation {
    pub op: OpRef,
    pub error: ConventionError,
}

/// Every operation under `root` whose convention is malformed or sits on an
/// operation that cannot carry one (`new-plans/cps-effects.md`, "CPS 변환의
/// 호출 대상 형상").
pub fn verify_conventions(ctx: &IrContext, root: OpRef) -> Vec<ConventionViolation> {
    let mut violations = Vec::new();
    let _ = trunk_ir::walk::walk_op::<()>(ctx, root, &mut |op| {
        if let Err(error) = DynConventionBearing::exact(ctx, op) {
            violations.push(ConventionViolation { op, error });
        }
        std::ops::ControlFlow::Continue(trunk_ir::walk::WalkAction::Advance)
    });
    violations
}

trunk_ir::register_interfaces!(
    func::Func: ConventionBearing;
    func::Call: ConventionBearing;
    func::CallIndirect: ConventionBearing;
    func::TailCall: ConventionBearing;
    func::TailCallIndirect: ConventionBearing
);

/// Result-indexed CPS completion target `Done<R>`.
pub fn cps_done_type(ctx: &mut IrContext, result: TypeRef) -> TypeRef {
    let never = core::Never::params().intern(ctx).as_type_ref();
    let function = func::func_sig(ctx, [result], [never]).as_type_ref();
    physical_closure_type(ctx, function)
}

/// Strict suffix continuation `Completion<X, R> = (Evidence, ContinuationFrame<R>, X) -> never`.
pub fn cps_completion_type(
    ctx: &mut IrContext,
    evidence: TypeRef,
    value: TypeRef,
    frame: TypeRef,
) -> TypeRef {
    let never = core::Never::params().intern(ctx).as_type_ref();
    let function = func::func_sig(ctx, [evidence, frame, value], [never]).as_type_ref();
    physical_closure_type(ctx, function)
}

/// Exact operation resumption `ResumeExact<I, R> = (Evidence, ContinuationFrame<R>, I) -> never`.
pub fn cps_resume_exact_type(
    ctx: &mut IrContext,
    evidence: TypeRef,
    input: TypeRef,
    frame: TypeRef,
) -> TypeRef {
    cps_completion_type(ctx, evidence, input, frame)
}

/// Erased-input resumption `Resume<R> = (Evidence, ContinuationFrame<R>, anyref) -> never`.
pub fn cps_resume_type(
    ctx: &mut IrContext,
    evidence: TypeRef,
    frame: TypeRef,
    anyref: TypeRef,
) -> TypeRef {
    cps_completion_type(ctx, evidence, anyref, frame)
}

/// Result-indexed general-operation dispatcher.
pub fn cps_dispatch_type(
    ctx: &mut IrContext,
    evidence: TypeRef,
    frame: TypeRef,
    anyref: TypeRef,
    i32_type: TypeRef,
) -> TypeRef {
    let never = core::Never::params().intern(ctx).as_type_ref();
    let resume = cps_resume_type(ctx, evidence, frame, anyref);
    let function = func::func_sig(
        ctx,
        [evidence, resume, i32_type, i32_type, i32_type, anyref],
        [never],
    )
    .as_type_ref();
    physical_closure_type(ctx, function)
}

/// The `Cps` entry, without the environment, of a generated continuation
/// `closure`. Only a generated continuation's callable is its entry; a source
/// callable value's is not (`new-plans/cps-effects.md`, "Closure type의
/// callable").
pub fn cps_closure_function_type(ctx: &IrContext, closure: TypeRef) -> Option<TypeRef> {
    physical_closure_function_type(ctx, closure, CallingConvention::Cps)
}

/// The entry, without the environment, of a generated closure entered
/// through `convention`, when it is a `closure.closure` whose callable has
/// the environment slot that `convention` places. A source callable value's
/// closure type is not its entry, so its transfers carry their own signature.
pub fn physical_closure_function_type(
    ctx: &IrContext,
    closure: TypeRef,
    convention: CallingConvention,
) -> Option<TypeRef> {
    let function = closure::Closure::from_type_ref(ctx, closure)?.func_type(ctx);
    let callable = func::FuncSig::from_type_ref(ctx, function)?;
    (convention.closure_environment_index() <= callable.inputs(ctx).len()).then_some(function)
}

/// The closure type of the callable `function`.
pub fn physical_closure_type(ctx: &mut IrContext, function: TypeRef) -> TypeRef {
    closure::Closure::params(function).intern(ctx).as_type_ref()
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::types::TypeDataBuilder;
    use trunk_ir::walk::{WalkAction, walk_op};

    const CONVENTIONS: &str = r#"core.module @test {
  func.func @direct() -> core.nil attributes {tribute.calling_convention = "direct"} {
    func.return
  }
  func.func @untagged() -> core.nil {
    func.return
  }
  func.func @malformed() -> core.nil attributes {tribute.calling_convention = 3} {
    func.return
  }
  func.func @wrong_kind() -> core.nil attributes {tribute.calling_convention = 2} {
    func.return
  }
  func.func @calls() -> core.nil attributes {tribute.calling_convention = "evidence_direct"} {
    %r = func.call {callee = @direct, tribute.calling_convention = "direct"} : core.nil
    %c = arith.const {value = 0, tribute.calling_convention = "direct"} : core.i32
    func.return
  }
}"#;

    fn ops(ctx: &IrContext, module: trunk_ir::rewrite::Module) -> Vec<OpRef> {
        let mut ops = Vec::new();
        let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
            ops.push(op);
            std::ops::ControlFlow::Continue(WalkAction::Advance)
        });
        ops
    }

    fn function(ctx: &IrContext, module: trunk_ir::rewrite::Module, name: &str) -> OpRef {
        ops(ctx, module)
            .into_iter()
            .find(|&op| {
                func::Func::from_op(ctx, op).is_ok_and(|function| function.sym_name(ctx) == name)
            })
            .expect("fixture function")
    }

    fn named(ctx: &IrContext, module: trunk_ir::rewrite::Module, name: &str) -> OpRef {
        ops(ctx, module)
            .into_iter()
            .find(|&op| ctx.op(op).name.is(ctx, name))
            .expect("fixture operation")
    }

    #[test]
    fn convention_bearing_operations_are_registered() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @f(%x: core.i32) -> core.nil {
    %r = func.call %x {callee = @f} : core.nil
    func.tail_call %x {callee = @f}
  }
}"#,
        );
        assert!(DynConventionBearing::is_implemented(
            &ctx,
            function(&ctx, module, "f")
        ));
        assert!(DynConventionBearing::is_implemented(
            &ctx,
            named(&ctx, module, "func.call")
        ));
        assert!(DynConventionBearing::is_implemented(
            &ctx,
            named(&ctx, module, "func.tail_call")
        ));
        assert!(!DynConventionBearing::is_implemented(
            &ctx,
            named(&ctx, module, "core.module")
        ));
        let location = ctx.op(module.op()).location;
        for name in [func::CallIndirect::NAME, func::TailCallIndirect::NAME] {
            let op = trunk_ir::OperationDataBuilder::new(location, name).build(&mut ctx);
            let op = ctx.create_op(op);
            assert!(DynConventionBearing::is_implemented(&ctx, op), "{name:?}");
        }
    }

    fn violations(input: &str) -> Vec<(String, ConventionError)> {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        verify_conventions(&ctx, module.op())
            .into_iter()
            .map(|violation| {
                (
                    ctx.op(violation.op).name.display(&ctx).to_string(),
                    violation.error,
                )
            })
            .collect()
    }

    const CLOSURES: &str = r#"core.module @test {
  !fn = func.func_sig<(core.i32) -> core.i32>
  !closure = closure.closure<!fn>
  func.func @callee(%x: core.i32) -> core.i32 attributes {tribute.calling_convention = "direct"} {
    func.return %x
  }
  func.func @agrees(%x: core.i32) -> core.i32 attributes {tribute.calling_convention = "direct"} {
    %c = closure.new {entries = [@callee, @callee, @callee]} : !closure
    %r = func.call_indirect %c, %x {signature = !fn, tribute.calling_convention = "direct"} : core.i32
    func.return %r
  }
}"#;

    #[test]
    fn conventions_on_closure_operations_verify() {
        assert_eq!(violations(CLOSURES), []);
    }

    #[test]
    fn a_convention_on_an_operation_that_cannot_carry_one_is_rejected() {
        assert_eq!(
            violations(CONVENTIONS),
            [
                ("func.func".to_owned(), ConventionError::Malformed),
                ("func.func".to_owned(), ConventionError::Malformed),
                ("arith.const".to_owned(), ConventionError::Unregistered),
            ]
        );
    }

    #[test]
    fn reads_distinguish_absent_malformed_and_unregistered_conventions() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, CONVENTIONS);

        assert_eq!(
            DynConventionBearing::exact(&ctx, function(&ctx, module, "direct")),
            Ok(Some(CallingConvention::Direct))
        );
        assert_eq!(
            DynConventionBearing::exact(&ctx, function(&ctx, module, "calls")),
            Ok(Some(CallingConvention::EvidenceDirect))
        );
        assert_eq!(
            DynConventionBearing::exact(&ctx, named(&ctx, module, "func.call")),
            Ok(Some(CallingConvention::Direct))
        );
        assert_eq!(
            DynConventionBearing::exact(&ctx, function(&ctx, module, "untagged")),
            Ok(None)
        );
        for malformed in ["malformed", "wrong_kind"] {
            let op = function(&ctx, module, malformed);
            assert_eq!(
                DynConventionBearing::exact(&ctx, op),
                Err(ConventionError::Malformed)
            );
            assert_eq!(DynConventionBearing::of(&ctx, op), None);
        }
        let constant = named(&ctx, module, "arith.const");
        assert_eq!(
            DynConventionBearing::exact(&ctx, constant),
            Err(ConventionError::Unregistered)
        );
        assert_eq!(DynConventionBearing::of(&ctx, constant), None);
        assert_eq!(
            DynConventionBearing::exact(&ctx, named(&ctx, module, "func.return")),
            Ok(None)
        );
    }

    #[test]
    fn set_and_remove_round_trip_through_the_interface() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, CONVENTIONS);
        let untagged = function(&ctx, module, "untagged");

        DynConventionBearing::set(&mut ctx, untagged, CallingConvention::Cps);
        assert_eq!(
            DynConventionBearing::exact(&ctx, untagged),
            Ok(Some(CallingConvention::Cps))
        );

        let bearing = DynConventionBearing::get(&ctx, untagged).unwrap();
        bearing.remove_calling_convention(&mut ctx);
        assert_eq!(DynConventionBearing::exact(&ctx, untagged), Ok(None));
        assert!(!ctx.op_attrs(untagged).contains_key(CALLING_CONVENTION_ATTR));
    }

    #[test]
    #[should_panic(expected = "only a ConventionBearing operation")]
    fn setting_a_convention_on_an_unregistered_operation_panics() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, CONVENTIONS);
        let constant = named(&ctx, module, "arith.const");
        DynConventionBearing::set(&mut ctx, constant, CallingConvention::Direct);
    }

    #[test]
    fn keywords_are_the_only_stored_form() {
        let mut ctx = IrContext::new();
        for convention in [
            CallingConvention::Direct,
            CallingConvention::EvidenceDirect,
            CallingConvention::Cps,
        ] {
            let value = convention.to_attribute(&mut ctx);
            assert_eq!(
                CallingConvention::from_attribute(&ctx, &value),
                Some(convention)
            );
            assert_eq!(convention.to_string().parse(), Ok(convention));
        }

        assert!("Cps".parse::<CallingConvention>().is_err());
        let unknown = ctx.string_attr("tail");
        assert_eq!(CallingConvention::from_attribute(&ctx, &unknown), None);
        assert_eq!(
            CallingConvention::from_attribute(&ctx, &Attribute::Int(2)),
            None
        );
    }

    #[test]
    fn result_indexed_cps_closure_builders_preserve_exact_types() {
        let mut ctx = IrContext::new();
        let evidence = TypeDataBuilder::new("ability.evidence").intern(&mut ctx);
        let i32_ty = TypeDataBuilder::new("core.i32").intern(&mut ctx);
        let anyref = TypeDataBuilder::new("tribute_rt.anyref").intern(&mut ctx);
        let frame = TypeDataBuilder::new("test.frame").intern(&mut ctx);
        let done = cps_done_type(&mut ctx, i32_ty);
        let dispatch = cps_dispatch_type(&mut ctx, evidence, frame, anyref, i32_ty);
        let completion = cps_completion_type(&mut ctx, evidence, i32_ty, frame);
        let resume_exact = cps_resume_exact_type(&mut ctx, evidence, i32_ty, frame);
        let resume = cps_resume_type(&mut ctx, evidence, frame, anyref);
        let never = core::Never::params().intern(&mut ctx).as_type_ref();

        for (closure, expected) in [
            (done, vec![i32_ty, never]),
            (completion, vec![evidence, frame, i32_ty, never]),
            (resume_exact, vec![evidence, frame, i32_ty, never]),
            (resume, vec![evidence, frame, anyref, never]),
            (
                dispatch,
                vec![evidence, resume, i32_ty, i32_ty, i32_ty, anyref, never],
            ),
        ] {
            let function = cps_closure_function_type(&ctx, closure).expect("exact CPS closure");
            assert_eq!(
                ctx.get_type(function).params.as_slice(),
                expected.as_slice()
            );
            assert!(ctx.get_type(closure).attrs.is_empty());
        }
    }

    #[test]
    fn cps_closure_readers_reject_a_callable_without_an_environment_slot() {
        let mut ctx = IrContext::new();
        let never = core::Never::params().intern(&mut ctx).as_type_ref();
        let i32_ty = TypeDataBuilder::new("core.i32").intern(&mut ctx);
        let nullary = func::func_sig(&mut ctx, [], [never]).as_type_ref();
        let unary = func::func_sig(&mut ctx, [i32_ty], [never]).as_type_ref();
        let nullary = physical_closure_type(&mut ctx, nullary);
        let unary_closure = physical_closure_type(&mut ctx, unary);

        assert_eq!(cps_closure_function_type(&ctx, nullary), None);
        assert_eq!(cps_closure_function_type(&ctx, unary_closure), Some(unary));
        assert_eq!(
            physical_closure_function_type(&ctx, nullary, CallingConvention::Direct),
            Some(ctx.get_type(nullary).params[0])
        );
        assert_eq!(cps_closure_function_type(&ctx, i32_ty), None);
    }
}
