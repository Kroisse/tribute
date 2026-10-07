//! Verification of the abstract continuation frame operations.

use rustc_hash::FxHashSet as HashSet;
use trunk_ir::dialect::core::Never;
use trunk_ir::dialect::func;
use trunk_ir::ops::DialectType;
use trunk_ir::refs::{TypeRef, ValueRef};
use trunk_ir::types::{Attribute, AttributeMap, StringRef};
use trunk_ir::{IrContext, Symbol};

use super::{Frame, Handle, SuffixFrame, is_evidence_type_ref};
use crate::dialect::tribute_control::{
    CALLING_CONVENTION_ATTR, CallingConvention, verify_evidence_plan,
};

/// Declared kind of a handled operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OperationKind {
    /// `fn`: tail-resumptive; the arm returns the operation result.
    Fn,
    /// `op`: general; the arm receives resume tokens unless the operation
    /// result is `core.never`.
    Op,
}

impl OperationKind {
    pub fn keyword(self) -> &'static str {
        match self {
            Self::Fn => "fn",
            Self::Op => "op",
        }
    }

    fn from_keyword(keyword: &str) -> Option<Self> {
        match keyword {
            "fn" => Some(Self::Fn),
            "op" => Some(Self::Op),
            _ => None,
        }
    }
}

/// One element of `ability.handle`'s `handlers`: the operation an arm handles.
///
/// Written `{ability_ref = !State, op_name = "get", kind = "op",
/// operation_result_type = core.i32}`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HandlerBinding {
    pub ability_ref: TypeRef,
    pub op_name: StringRef,
    pub kind: OperationKind,
    pub operation_result_type: TypeRef,
}

impl HandlerBinding {
    /// Decode one binding, or say why it is malformed.
    pub fn from_attribute(ctx: &IrContext, attr: &Attribute) -> Result<Self, String> {
        let Attribute::Dict(entries) = attr else {
            return Err("handlers element must be a dictionary".into());
        };
        if entries.len() != 4 {
            return Err(format!(
                "handlers element must have exactly ability_ref, op_name, kind, and \
                 operation_result_type, found {} entries",
                entries.len()
            ));
        }
        let ability_ref = match entries.get("ability_ref") {
            Some(Attribute::Type(ty)) if is_ability_ref(ctx, *ty) => *ty,
            _ => return Err("handlers ability_ref must be a core.ability_ref type".into()),
        };
        let op_name = match entries.get("op_name") {
            Some(Attribute::String(name)) => *name,
            _ => return Err("handlers op_name must be a string".into()),
        };
        let kind = match entries.get("kind") {
            Some(Attribute::String(kind)) => OperationKind::from_keyword(ctx.str(*kind))
                .ok_or_else(|| "handlers kind must be fn or op".to_string())?,
            _ => return Err("handlers kind must be fn or op".into()),
        };
        let operation_result_type = match entries.get("operation_result_type") {
            Some(Attribute::Type(ty)) => *ty,
            _ => return Err("handlers operation_result_type must be a type".into()),
        };
        Ok(Self {
            ability_ref,
            op_name,
            kind,
            operation_result_type,
        })
    }

    pub fn to_attribute(self, ctx: &mut IrContext) -> Attribute {
        let kind = ctx.string_attr(self.kind.keyword());
        let mut entries = AttributeMap::new();
        entries.insert(
            Symbol::new("ability_ref"),
            Attribute::Type(self.ability_ref),
        );
        entries.insert(Symbol::new("op_name"), Attribute::String(self.op_name));
        entries.insert(Symbol::new("kind"), kind);
        entries.insert(
            Symbol::new("operation_result_type"),
            Attribute::Type(self.operation_result_type),
        );
        Attribute::Dict(entries)
    }

    /// Whether the arm receives resume tokens.
    pub fn is_resumptive(self, ctx: &IrContext) -> bool {
        self.kind == OperationKind::Op && !is_never(ctx, self.operation_result_type)
    }
}

impl trunk_ir::attr_kind::AttrKind for HandlerBinding {
    const KIND: trunk_ir::op_schema::AttributeKind =
        trunk_ir::op_schema::AttributeKind::Dict(&trunk_ir::op_schema::AttributeKind::Any);
    type Out<'ctx> = HandlerBinding;
    type In = HandlerBinding;

    fn read<'ctx>(ctx: &'ctx IrContext, attr: &'ctx Attribute) -> HandlerBinding {
        Self::from_attribute(ctx, attr)
            .unwrap_or_else(|error| panic!("unverified handlers element: {error}"))
    }

    fn write(ctx: &mut IrContext, value: HandlerBinding) -> Attribute {
        value.to_attribute(ctx)
    }
}

fn is_ability_ref(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.get_type(ty);
    data.dialect == Symbol::new("core") && data.name == Symbol::new("ability_ref")
}

fn is_never(ctx: &IrContext, ty: TypeRef) -> bool {
    Never::from_type_ref(ctx, ty).is_some()
}

/// The convention and function signature of a convention-proven closure type.
fn closure_signature(ctx: &IrContext, ty: TypeRef) -> Option<(CallingConvention, func::FuncSig)> {
    let data = ctx.get_type(ty);
    if data.dialect != Symbol::new("closure") || data.name != Symbol::new("closure") {
        return None;
    }
    let convention = match data.attrs.get(CALLING_CONVENTION_ATTR) {
        Some(Attribute::Int(code)) => CallingConvention::try_from(*code).ok()?,
        _ => return None,
    };
    let [function] = data.params.as_slice() else {
        return None;
    };
    Some((convention, func::FuncSig::from_type_ref(ctx, *function)?))
}

/// Check that `value` is a `Cps` closure `(Evidence, frame, rest...) -> core.never`
/// and return the inputs after the frame.
fn cps_closure_rest<'a>(
    ctx: &'a IrContext,
    value: ValueRef,
    frame: TypeRef,
    role: &str,
) -> Result<&'a [TypeRef], String> {
    let Some((convention, signature)) = closure_signature(ctx, ctx.value_ty(value)) else {
        return Err(format!(
            "{role} must be a closure with a calling convention"
        ));
    };
    if convention != CallingConvention::Cps {
        return Err(format!("{role} must use the cps convention"));
    }
    let results = signature.results(ctx);
    if !matches!(results, [result] if is_never(ctx, *result)) {
        return Err(format!("{role} must return core.never"));
    }
    match signature.inputs(ctx) {
        [evidence, input_frame, rest @ ..]
            if is_evidence_type_ref(ctx, *evidence) && *input_frame == frame =>
        {
            Ok(rest)
        }
        _ => Err(format!(
            "{role} must take the evidence and the frame of the answer first"
        )),
    }
}

/// The answer type of an `ability.frame`.
fn frame_result(ctx: &IrContext, ty: TypeRef) -> Option<TypeRef> {
    Frame::from_type_ref(ctx, ty).map(|frame| frame.result(ctx))
}

impl trunk_ir::ops::Verify for SuffixFrame {
    fn verify(self, ctx: &IrContext) -> Result<(), String> {
        let op = self.op_ref();
        verify_evidence_plan(ctx, op, false)?;
        let outer = ctx.value_ty(self.outer(ctx));
        let value_type = frame_result(ctx, self.result_ty(ctx))
            .ok_or("ability.suffix_frame must produce an ability.frame")?;
        let rest = cps_closure_rest(
            ctx,
            self.continuation(ctx),
            outer,
            "ability.suffix_frame continuation",
        )?;
        if rest != [value_type] {
            return Err(
                "ability.suffix_frame continuation must take the value of the produced frame"
                    .into(),
            );
        }
        Ok(())
    }
}

impl trunk_ir::ops::Verify for Handle {
    fn verify(self, ctx: &IrContext) -> Result<(), String> {
        let op = self.op_ref();
        verify_evidence_plan(ctx, op, true)?;
        let Some(Attribute::List(items)) = ctx.op(op).attributes.get("handlers") else {
            return Err("ability.handle requires a handlers list".into());
        };
        let bindings = items
            .iter()
            .map(|item| HandlerBinding::from_attribute(ctx, item))
            .collect::<Result<Vec<_>, _>>()?;
        let mut seen = HashSet::default();
        for binding in &bindings {
            if !seen.insert((binding.ability_ref, binding.op_name)) {
                return Err(format!(
                    "ability.handle binds operation {} twice",
                    ctx.str(binding.op_name)
                ));
            }
        }
        let arms = self.arms(ctx);
        if arms.len() != bindings.len() {
            return Err(format!(
                "ability.handle has {} handlers but {} arms",
                bindings.len(),
                arms.len()
            ));
        }

        let exit = ctx.value_ty(self.exit(ctx));
        let body_frame = verify_body(ctx, self)?;
        let body_type = frame_result(ctx, body_frame).expect("body frame was checked");
        let rest = cps_closure_rest(ctx, self.completion(ctx), exit, "ability.handle completion")?;
        if rest != [body_type] {
            return Err("ability.handle completion must take the value of the body frame".into());
        }

        for (binding, &arm) in bindings.iter().zip(arms) {
            verify_arm(ctx, binding, arm, exit)?;
        }
        Ok(())
    }
}

/// Check the body entry block and return its frame type.
fn verify_body(ctx: &IrContext, handle: Handle) -> Result<TypeRef, String> {
    let body = ctx
        .op_region(handle.op_ref(), 0)
        .ok_or("ability.handle requires a body region")?;
    let entry = ctx
        .region(body)
        .blocks
        .first()
        .copied()
        .ok_or("ability.handle body must have an entry block")?;
    let evidence = ctx.value_ty(handle.evidence(ctx));
    match ctx.block_args(entry) {
        [body_evidence, frame]
            if ctx.value_ty(*body_evidence) == evidence
                && frame_result(ctx, ctx.value_ty(*frame)).is_some() =>
        {
            Ok(ctx.value_ty(*frame))
        }
        _ => Err("ability.handle body must take the extended evidence and an ability.frame".into()),
    }
}

fn verify_arm(
    ctx: &IrContext,
    binding: &HandlerBinding,
    arm: ValueRef,
    exit: TypeRef,
) -> Result<(), String> {
    let name = ctx.str(binding.op_name);
    match binding.kind {
        OperationKind::Fn => {
            let Some((convention, signature)) = closure_signature(ctx, ctx.value_ty(arm)) else {
                return Err(format!(
                    "arm of fn operation {name} must be a closure with a calling convention"
                ));
            };
            if convention != CallingConvention::EvidenceDirect {
                return Err(format!(
                    "arm of fn operation {name} must use the evidence_direct convention"
                ));
            }
            if !matches!(signature.inputs(ctx), [evidence, ..] if is_evidence_type_ref(ctx, *evidence))
            {
                return Err(format!(
                    "arm of fn operation {name} must take the evidence first"
                ));
            }
            if signature.results(ctx) != [binding.operation_result_type] {
                return Err(format!(
                    "arm of fn operation {name} must return the operation result"
                ));
            }
        }
        OperationKind::Op => {
            let role = format!("arm of op operation {name}");
            let rest = cps_closure_rest(ctx, arm, exit, &role)?;
            if binding.is_resumptive(ctx) {
                let tokens_match = rest.len() >= 2
                    && rest[rest.len() - 2..].iter().all(|&token| {
                        is_resume_exact(ctx, token, exit, binding.operation_result_type)
                    });
                if !tokens_match {
                    return Err(format!(
                        "{role} must end with two resumptions of the operation result \
                         into the handle answer"
                    ));
                }
            } else if rest.iter().any(|&input| {
                closure_signature(ctx, input).is_some_and(|(_, signature)| {
                    matches!(signature.inputs(ctx), [_, frame, _] if *frame == exit)
                })
            }) {
                return Err(format!(
                    "{role} returns core.never and must not take a resumption"
                ));
            }
        }
    }
    Ok(())
}

/// Whether `ty` is `ResumeExact<input, R>`: a `Cps` closure
/// `(Evidence, frame, input) -> core.never`.
fn is_resume_exact(ctx: &IrContext, ty: TypeRef, frame: TypeRef, input: TypeRef) -> bool {
    closure_signature(ctx, ty).is_some_and(|(convention, signature)| {
        convention == CallingConvention::Cps
            && matches!(signature.results(ctx), [result] if is_never(ctx, *result))
            && matches!(
                signature.inputs(ctx),
                [evidence, input_frame, value]
                    if is_evidence_type_ref(ctx, *evidence)
                        && *input_frame == frame
                        && *value == input
            )
    })
}

#[cfg(test)]
mod tests {
    use trunk_ir::IrContext;
    use trunk_ir::parser::parse_module;
    use trunk_ir::printer::print_module;
    use trunk_ir::rewrite::Module;
    use trunk_ir::validation::validate_operation_verifiers;

    const TYPES: &str = r#"  !marker = adt.struct<_Marker(ability_id: core.i32, prompt_tag: core.i32, tr_dispatch_fn: core.ptr, shadowed: core.ptr, outer: core.ptr), {layout = "evidence_marker"}>
  !ev = core.array<!marker, {layout = "evidence"}>
  !state = core.ability_ref<{name = "State"}>
  !frame_i32 = ability.frame<core.i32>
  !frame_nil = ability.frame<core.nil>
  !completion = closure.closure<func.func_sig<(!ev, !frame_i32, core.nil) -> core.never>, {tribute.calling_convention = 2, tribute.closure_environment_index = 0}>
  !resume = closure.closure<func.func_sig<(!ev, !frame_i32, core.i32) -> core.never>, {tribute.calling_convention = 2, tribute.closure_environment_index = 0}>
  !op_arm = closure.closure<func.func_sig<(!ev, !frame_i32, core.i32, !resume, !resume) -> core.never>, {tribute.calling_convention = 2, tribute.closure_environment_index = 1}>
  !never_arm = closure.closure<func.func_sig<(!ev, !frame_i32, core.i32) -> core.never>, {tribute.calling_convention = 2, tribute.closure_environment_index = 1}>
  !fn_arm = closure.closure<func.func_sig<(!ev) -> core.i32>, {tribute.calling_convention = 1, tribute.closure_environment_index = 1}>"#;

    fn module(params: &str, body: &str) -> String {
        format!(
            r#"core.module @test {{
{TYPES}
  func.func @run(%ev: !ev, %exit: !frame_i32{params}) -> core.never {{
{body}
  }}
}}"#
        )
    }

    fn parse(text: &str) -> (IrContext, Module) {
        let mut ctx = IrContext::new();
        let root = parse_module(&mut ctx, text)
            .unwrap_or_else(|error| panic!("failed to parse:\n{error}\n\n{text}"));
        let module = Module::new(&ctx, root).expect("core.module");
        (ctx, module)
    }

    /// The verifier messages for `text`, empty when it verifies.
    fn errors(text: &str) -> String {
        let (ctx, module) = parse(text);
        let result = validate_operation_verifiers(&ctx, module);
        if result.is_ok() {
            String::new()
        } else {
            result.to_string()
        }
    }

    fn binding(kind: &str, op_name: &str, result: &str) -> String {
        format!(
            r#"{{ability_ref = !state, kind = "{kind}", op_name = "{op_name}", operation_result_type = {result}}}"#
        )
    }

    fn handle(handlers: &[String], arms: &str, attrs: &str) -> String {
        format!(
            r#"    ability.handle %ev, %exit, %done{arms} {{handlers = [{}]{attrs}}} {{
      ^body(%inner: !ev, %frame: !frame_nil):
        %unit = core.nil_value : core.nil
        ability.exit %frame, %unit
    }}"#,
            handlers.join(", ")
        )
    }

    const ARMS: &str = ", %get: !op_arm, %fail: !never_arm, %peek: !fn_arm, %done: !completion";

    fn valid_handle() -> String {
        module(
            ARMS,
            &handle(
                &[
                    binding("op", "get", "core.i32"),
                    binding("op", "fail", "core.never"),
                    binding("fn", "peek", "core.i32"),
                ],
                ", %get, %fail, %peek",
                "",
            ),
        )
    }

    #[test]
    fn frame_operations_round_trip_and_verify() {
        let text = module(
            ", %k: !completion, %arg: core.i32",
            r#"    %frame = ability.suffix_frame %ev, %exit, %k {evidence_plan = [{mask = !state}]} : !frame_nil
    ability.abort %ev, %exit, %arg {ability_ref = !state, op_name = "fail"}"#,
        );
        let (ctx, module) = parse(&text);
        let result = validate_operation_verifiers(&ctx, module);
        assert!(result.is_ok(), "{result}");

        let printed = print_module(&ctx, module.op());
        let (reparsed, reparsed_module) = parse(&printed);
        assert_eq!(printed, print_module(&reparsed, reparsed_module.op()));
        assert_eq!(errors(&valid_handle()), "");
    }

    #[test]
    fn exit_value_must_match_the_frame_answer() {
        let text = module(", %value: core.nil", "    ability.exit %exit, %value");
        assert!(!errors(&text).is_empty());
    }

    #[test]
    fn malformed_frames_and_handles_are_rejected() {
        let wrong_value = module(
            ", %k: !completion",
            "    %frame = ability.suffix_frame %ev, %exit, %k : !frame_i32\n    func.unreachable",
        );
        let wrong_outer = module(
            ", %k: !completion, %other: !frame_nil",
            "    %frame = ability.suffix_frame %ev, %other, %k : !frame_nil\n    func.unreachable",
        );
        let direct_continuation = module(
            ", %k: !fn_arm",
            "    %frame = ability.suffix_frame %ev, %exit, %k : !frame_nil\n    func.unreachable",
        );
        let cases = [
            (
                "suffix frame of another value",
                wrong_value,
                "continuation must take the value of the produced frame",
            ),
            (
                "suffix frame around another frame",
                wrong_outer,
                "must take the evidence and the frame of the answer first",
            ),
            (
                "direct continuation",
                direct_continuation,
                "continuation must use the cps convention",
            ),
            (
                "duplicate binding",
                module(
                    ARMS,
                    &handle(
                        &[binding("op", "get", "core.i32"), binding("op", "get", "core.i32")],
                        ", %get, %get",
                        "",
                    ),
                ),
                "binds operation get twice",
            ),
            (
                "missing arm",
                module(ARMS, &handle(&[binding("op", "get", "core.i32")], "", "")),
                "has 1 handlers but 0 arms",
            ),
            (
                "fn binding with a cps arm",
                module(ARMS, &handle(&[binding("fn", "get", "core.i32")], ", %get", "")),
                "must use the evidence_direct convention",
            ),
            (
                "fn arm with the wrong result",
                module(ARMS, &handle(&[binding("fn", "peek", "core.nil")], ", %peek", "")),
                "must return the operation result",
            ),
            (
                "op binding with a fn arm",
                module(ARMS, &handle(&[binding("op", "peek", "core.i32")], ", %peek", "")),
                "must use the cps convention",
            ),
            (
                "resumptive arm without tokens",
                module(ARMS, &handle(&[binding("op", "fail", "core.i32")], ", %fail", "")),
                "must end with two resumptions",
            ),
            (
                "never arm with tokens",
                module(ARMS, &handle(&[binding("op", "get", "core.never")], ", %get", "")),
                "must not take a resumption",
            ),
            (
                "non-mask handle selection",
                module(
                    ARMS,
                    &handle(
                        &[binding("op", "get", "core.i32")],
                        ", %get",
                        ", evidence_plan = [{dup = !state}]",
                    ),
                ),
                "evidence_plan",
            ),
            (
                "malformed binding",
                module(
                    ARMS,
                    &handle(
                        &[r#"{ability_ref = !state, kind = "ctl", op_name = "get", operation_result_type = core.i32}"#.to_string()],
                        ", %get",
                        "",
                    ),
                ),
                "handlers kind must be fn or op",
            ),
        ];
        for (name, text, expected) in cases {
            let error = errors(&text);
            assert!(
                error.contains(expected),
                "{name}: expected {expected:?}, got {error:?}"
            );
        }
    }

    #[test]
    fn handler_binding_round_trips_through_its_attribute() {
        let mut ctx = IrContext::new();
        let name = ctx.intern_str("State");
        let state = ctx.intern_type(
            trunk_ir::types::TypeDataBuilder::new("core", "ability_ref")
                .attr("name", trunk_ir::types::Attribute::String(name))
                .build(),
        );
        let never = ctx.intern_type(trunk_ir::types::TypeDataBuilder::new("core", "never").build());
        let i32_ty = ctx.intern_type(trunk_ir::types::TypeDataBuilder::new("core", "i32").build());
        let op_name = ctx.intern_str("get");
        for (kind, result, resumptive) in [
            (super::OperationKind::Op, i32_ty, true),
            (super::OperationKind::Op, never, false),
            (super::OperationKind::Fn, i32_ty, false),
        ] {
            let binding = super::HandlerBinding {
                ability_ref: state,
                op_name,
                kind,
                operation_result_type: result,
            };
            let attr =
                <super::HandlerBinding as trunk_ir::attr_kind::AttrKind>::write(&mut ctx, binding);
            let read = <super::HandlerBinding as trunk_ir::attr_kind::AttrKind>::read(&ctx, &attr);
            assert_eq!(read, binding);
            assert_eq!(read.is_resumptive(&ctx), resumptive);
        }
    }

    #[test]
    fn malformed_bindings_closures_and_bodies_are_rejected() {
        const EXTRA: &str = r#"  !plain = closure.closure<func.func_sig<(!ev, !frame_i32, core.nil) -> core.never>>
  !returning = closure.closure<func.func_sig<(!ev, !frame_i32, core.nil) -> core.i32>, {tribute.calling_convention = 2, tribute.closure_environment_index = 0}>
  !no_evidence = closure.closure<func.func_sig<(core.i32) -> core.i32>, {tribute.calling_convention = 1, tribute.closure_environment_index = 1}>
  !bad_code = closure.closure<func.func_sig<(!ev) -> core.i32>, {tribute.calling_convention = 7, tribute.closure_environment_index = 1}>"#;
        let with_extra = |params: &str, body: &str| {
            module(params, body).replacen(TYPES, &format!("{TYPES}\n{EXTRA}"), 1)
        };
        let extra_arms = ", %plain: !plain, %returning: !returning, %no_evidence: !no_evidence, %bad_code: !bad_code, %value: core.i32";
        let arms = format!("{ARMS}{extra_arms}");
        let one = |kind: &str, result: &str, arm: &str| {
            with_extra(&arms, &handle(&[binding(kind, "get", result)], arm, ""))
        };
        let raw_binding =
            |entries: &str| with_extra(&arms, &handle(&[format!("{{{entries}}}")], ", %get", ""));
        let body = |args: &str| {
            with_extra(
                &arms,
                &format!(
                    r#"    ability.handle %ev, %exit, %done {{handlers = []}} {{
      ^body({args}):
        func.unreachable
    }}"#
                ),
            )
        };
        let cases = [
            (
                "plain closure arm",
                one("op", "core.i32", ", %plain"),
                "must be a closure with a calling convention",
            ),
            (
                "unknown convention",
                one("fn", "core.i32", ", %bad_code"),
                "must be a closure with a calling convention",
            ),
            (
                "non-closure fn arm",
                one("fn", "core.i32", ", %value"),
                "must be a closure with a calling convention",
            ),
            (
                "returning cps arm",
                one("op", "core.i32", ", %returning"),
                "must return core.never",
            ),
            (
                "fn arm without evidence",
                one("fn", "core.i32", ", %no_evidence"),
                "must take the evidence first",
            ),
            (
                "completion of another value",
                with_extra(
                    &arms.replace("%done: !completion", "%done: !resume"),
                    &handle(&[], "", ""),
                ),
                "completion must take the value of the body frame",
            ),
            (
                "body without frame",
                body("%inner: !ev"),
                "must take the extended evidence and an ability.frame",
            ),
            (
                "body with another evidence",
                body("%inner: core.i32, %frame: !frame_nil"),
                "must take the extended evidence",
            ),
            (
                "binding with an extra entry",
                raw_binding(
                    r#"ability_ref = !state, extra = 1, kind = "op", op_name = "get", operation_result_type = core.i32"#,
                ),
                "must have exactly",
            ),
            (
                "binding without an ability",
                raw_binding(
                    r#"ability_ref = core.i32, kind = "op", op_name = "get", operation_result_type = core.i32"#,
                ),
                "ability_ref must be a core.ability_ref",
            ),
            (
                "binding with a numeric name",
                raw_binding(
                    r#"ability_ref = !state, kind = "op", op_name = 1, operation_result_type = core.i32"#,
                ),
                "op_name must be a string",
            ),
            (
                "binding with a numeric kind",
                raw_binding(
                    r#"ability_ref = !state, kind = 1, op_name = "get", operation_result_type = core.i32"#,
                ),
                "kind must be fn or op",
            ),
            (
                "binding without a result type",
                raw_binding(
                    r#"ability_ref = !state, kind = "op", op_name = "get", operation_result_type = 1"#,
                ),
                "operation_result_type must be a type",
            ),
            (
                "binding that is not a dictionary",
                with_extra(&arms, &handle(&["1".to_string()], ", %get", "")),
                "must be a [Dict<any>] attribute",
            ),
        ];
        for (name, text, expected) in cases {
            let error = errors(&text);
            assert!(
                error.contains(expected),
                "{name}: expected {expected:?}, got {error:?}"
            );
        }
    }
}
