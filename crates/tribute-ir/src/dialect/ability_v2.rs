//! Direct-style ability elaboration dialect.
//!
//! `ability_v2` holds handler scopes and ability-operation invocations after
//! the ability declarations they refer to have been interpreted, and before
//! CPS legalization builds any continuation. Callables stay in
//! `tribute_control`: handler arms and completions are `tribute_control.lambda`
//! helpers that a scope binds by operand.

use rustc_hash::FxHashSet as HashSet;
use trunk_ir::attr_kind::Type;
use trunk_ir::dialect::core::Never;
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{OpRef, TypeRef, ValueRef};
use trunk_ir::types::{Attribute, AttributeMap, StringRef};
use trunk_ir::{IrContext, Symbol};

use super::tribute_control::{
    self, CallingConvention, EvidenceStep, FuncSig, ResumeToken, verify_evidence_plan,
};

#[trunk_ir::dialect]
mod ability_v2 {
    /// Install one handler for the operations `handlers` names around `body`.
    ///
    /// `helpers` holds one handler-arm helper per `handlers` entry, in the
    /// same order, and `completion` receives the value `body` yields. All of
    /// them run on the evidence the scope is installed on.
    #[verify]
    fn scope(
        handlers: Attr<[HandlerBinding]>,
        evidence_plan: Option<Attr<[EvidenceStep]>>,
        completion: Value<FuncSig>,
        helpers: Variadic<FuncSig>,
    ) -> Value<_> {
        #[region(body)]
        {}
    }

    /// Terminator of a scope body; its operand goes to the scope completion.
    #[verify]
    fn r#yield(value: Value<_>) {}

    /// InvokeOp a `fn` operation. The handler result flows inline.
    #[verify]
    fn call_fn(ability_ref: Attr<Type>, op_name: Attr<String>, args: Variadic<_>) -> Value<_> {}

    /// InvokeOp a general `op` operation. The handler may resume it unless its
    /// result is `core.never`.
    #[verify]
    fn invoke_op(ability_ref: Attr<Type>, op_name: Attr<String>, args: Variadic<_>) -> Value<_> {}

    /// Resume the computation a resumptive helper received the token of.
    #[verify]
    fn resume<T: ResumeToken>(
        evidence_plan: Option<Attr<[EvidenceStep]>>,
        resume_token: Value<T>,
        value: Value<T::Input>,
    ) -> Value<T::Answer> {
    }
}

/// Operation kind of a handled operation, as declared by its ability.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OperationKind {
    /// `fn`: tail-resumptive; the helper returns the operation result.
    Fn,
    /// `op`: general; the helper receives a resume token unless the
    /// operation result is `Never`.
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

/// One element of a scope's `handlers`: the operation a helper handles.
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

impl trunk_ir::ops::Verify for Scope {
    fn verify(self, ctx: &IrContext) -> Result<(), String> {
        let op = self.op_ref();
        verify_evidence_plan(ctx, op, true)?;
        let answer = self.result_ty(ctx);

        let Some(Attribute::List(items)) = ctx.op(op).attributes.get("handlers") else {
            return Err("ability_v2.scope requires a handlers list".into());
        };
        let bindings = items
            .iter()
            .map(|item| HandlerBinding::from_attribute(ctx, item))
            .collect::<Result<Vec<_>, _>>()?;
        let mut seen = HashSet::default();
        for binding in &bindings {
            if !seen.insert((binding.ability_ref, binding.op_name)) {
                return Err(format!(
                    "ability_v2.scope binds operation {} twice",
                    ctx.str(binding.op_name)
                ));
            }
        }
        let helpers = self.helpers(ctx);
        if helpers.len() != bindings.len() {
            return Err(format!(
                "ability_v2.scope has {} handlers but {} helpers",
                bindings.len(),
                helpers.len()
            ));
        }

        let body_type = verify_body(ctx, op)?;

        let completion = self.completion(ctx);
        let completion_sig = helper_signature(ctx, op, completion, "completion")?;
        if completion_sig.inputs(ctx) != [body_type] || completion_sig.result(ctx) != answer {
            return Err(
                "ability_v2.scope completion must take the body value and return the scope answer"
                    .into(),
            );
        }

        for (binding, &helper) in bindings.iter().zip(helpers) {
            let signature = helper_signature(ctx, op, helper, "helper")?;
            verify_helper(ctx, binding, signature, answer)?;
        }
        Ok(())
    }
}

/// Check the body shape and return the type of the value it yields.
fn verify_body(ctx: &IrContext, op: OpRef) -> Result<TypeRef, String> {
    let body = ctx
        .op_region(op, 0)
        .ok_or("ability_v2.scope requires a body region")?;
    let [block] = ctx.region(body).blocks[..] else {
        return Err("ability_v2.scope body must be a single block".into());
    };
    if !ctx.block(block).args.is_empty() {
        return Err("ability_v2.scope body takes no block arguments".into());
    }
    let terminator = ctx
        .block(block)
        .ops
        .last()
        .copied()
        .and_then(|last| Yield::from_op(ctx, last).ok())
        .ok_or("ability_v2.scope body must end with ability_v2.yield")?;
    Ok(ctx.value_ty(terminator.value(ctx)))
}

/// Read a helper operand's signature and check that the scope owns it alone.
fn helper_signature(
    ctx: &IrContext,
    scope: OpRef,
    value: ValueRef,
    role: &str,
) -> Result<FuncSig, String> {
    let defined_by_lambda = match ctx.value_def(value) {
        trunk_ir::refs::ValueDef::OpResult(def, _) => tribute_control::Lambda::matches(ctx, def),
        trunk_ir::refs::ValueDef::BlockArg(..) => false,
    };
    if !defined_by_lambda {
        return Err(format!(
            "ability_v2.scope {role} must be a tribute_control.lambda"
        ));
    }
    if ctx.uses(value).iter().any(|use_| use_.user != scope) || ctx.uses(value).len() != 1 {
        return Err(format!(
            "ability_v2.scope {role} must have the scope as its only use"
        ));
    }
    FuncSig::from_type_ref(ctx, ctx.value_ty(value))
        .ok_or_else(|| format!("ability_v2.scope {role} must be a tribute_control.func_sig"))
}

fn verify_helper(
    ctx: &IrContext,
    binding: &HandlerBinding,
    signature: FuncSig,
    answer: TypeRef,
) -> Result<(), String> {
    let name = ctx.str(binding.op_name);
    let inputs = signature.inputs(ctx);
    let convention = signature.convention(ctx);
    match binding.kind {
        OperationKind::Fn => {
            if convention == CallingConvention::Cps {
                return Err(format!(
                    "helper of fn operation {name} must not use the cps convention"
                ));
            }
            if signature.result(ctx) != binding.operation_result_type {
                return Err(format!(
                    "helper of fn operation {name} must return the operation result"
                ));
            }
            if inputs
                .iter()
                .any(|&input| ResumeToken::from_type_ref(ctx, input).is_some())
            {
                return Err(format!(
                    "helper of fn operation {name} must not take a resume token"
                ));
            }
        }
        OperationKind::Op => {
            if convention != CallingConvention::Cps {
                return Err(format!(
                    "helper of op operation {name} must use the cps convention"
                ));
            }
            if signature.result(ctx) != answer {
                return Err(format!(
                    "helper of op operation {name} must return the scope answer"
                ));
            }
            let token = inputs
                .last()
                .and_then(|&input| tribute_control::resume_token_parts(ctx, input));
            if is_never(ctx, binding.operation_result_type) {
                if inputs
                    .iter()
                    .any(|&input| ResumeToken::from_type_ref(ctx, input).is_some())
                {
                    return Err(format!(
                        "helper of Never operation {name} must not take a resume token"
                    ));
                }
            } else if token != Some((binding.operation_result_type, answer)) {
                return Err(format!(
                    "helper of op operation {name} must end with \
                     resume_token<operation result, scope answer>"
                ));
            }
        }
    }
    Ok(())
}

impl trunk_ir::ops::Verify for Yield {
    fn verify(self, ctx: &IrContext) -> Result<(), String> {
        let op = self.op_ref();
        let owner = ctx
            .op(op)
            .parent_block
            .and_then(|block| ctx.block(block).parent_region)
            .and_then(|region| ctx.region(region).parent_op);
        if !owner.is_some_and(|owner| Scope::matches(ctx, owner)) {
            return Err("ability_v2.yield must terminate an ability_v2.scope body".into());
        }
        Ok(())
    }
}

fn verify_invocation(
    ctx: &IrContext,
    op: OpRef,
    required: CallingConvention,
) -> Result<(), String> {
    let data = ctx.op(op);
    if !data
        .attributes
        .get_type("ability_ref")
        .is_some_and(|ty| is_ability_ref(ctx, ty))
    {
        return Err("ability_ref must be a core.ability_ref type".into());
    }
    let returns_never = ctx
        .op_result_types(op)
        .first()
        .is_some_and(|&ty| is_never(ctx, ty));
    if returns_never && CallFn::matches(ctx, op) {
        return Err("an operation returning core.never must use ability_v2.invoke_op".into());
    }
    match enclosing_convention(ctx, op) {
        Some(convention) if convention >= required => Ok(()),
        _ => Err(format!(
            "ability_v2.{} requires an enclosing callable with at least the {required:?} \
             convention",
            ctx.op(op).name
        )),
    }
}

/// The convention of the nearest `tribute_control.func` or `lambda` around
/// `op`.
fn enclosing_convention(ctx: &IrContext, op: OpRef) -> Option<CallingConvention> {
    let mut current = op;
    loop {
        current = ctx
            .op(current)
            .parent_block
            .and_then(|block| ctx.block(block).parent_region)
            .and_then(|region| ctx.region(region).parent_op)?;
        let callable = if let Ok(func) = tribute_control::Func::from_op(ctx, current) {
            ctx.op(func.op_ref()).attributes.get_type("type")
        } else if tribute_control::Lambda::matches(ctx, current) {
            ctx.op_result_types(current).first().copied()
        } else {
            continue;
        };
        return tribute_control::func_sig_convention(ctx, callable?);
    }
}

impl trunk_ir::ops::Verify for CallFn {
    fn verify(self, ctx: &IrContext) -> Result<(), String> {
        verify_invocation(ctx, self.op_ref(), CallingConvention::EvidenceDirect)
    }
}

impl trunk_ir::ops::Verify for InvokeOp {
    fn verify(self, ctx: &IrContext) -> Result<(), String> {
        verify_invocation(ctx, self.op_ref(), CallingConvention::Cps)
    }
}

impl trunk_ir::ops::Verify for Resume {
    fn verify(self, ctx: &IrContext) -> Result<(), String> {
        verify_evidence_plan(ctx, self.op_ref(), false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_module;
    use trunk_ir::printer::print_module;
    use trunk_ir::rewrite::Module;
    use trunk_ir::validation::validate_operation_verifiers;

    const STATE: &str = r#"core.ability_ref<{name = "State"}>"#;

    /// A module whose `@run` body is `body`, with `State` helpers bound.
    fn module_text(scope_attrs: &str, helpers: &str, body: &str) -> String {
        format!(
            r#"core.module @test {{
  tribute_control.func @run(%value: core.i32) -> core.i32 convention(cps) {{
{helpers}
    %answer = ability_v2.scope %done{scope_attrs} : core.i32 {{
{body}
    }}
    tribute_control.return %answer
  }}
}}"#
        )
    }

    const DONE: &str = r#"    %done = tribute_control.lambda(%completed: core.i32) -> core.i32 convention(direct) captures [] {
      tribute_control.return %completed
    }"#;

    fn get_helper() -> String {
        format!(
            r#"{DONE}
    %get = tribute_control.lambda(%token: tribute_control.resume_token<core.i32, core.i32>) -> core.i32 convention(cps) captures [%value] {{
      %resumed = ability_v2.resume %token, %value : core.i32
      tribute_control.return %resumed
    }}"#
        )
    }

    fn get_binding(kind: &str) -> String {
        format!(
            r#", %get {{handlers = [{{ability_ref = {STATE}, kind = "{kind}", op_name = "get", operation_result_type = core.i32}}]}}"#
        )
    }

    fn invoke_body(op: &str) -> String {
        format!(
            r#"      %got = ability_v2.{op} {{ability_ref = {STATE}, op_name = "get"}} : core.i32
      ability_v2.yield %got"#
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

    #[test]
    fn scope_with_resumptive_helper_round_trips_and_verifies() {
        let text = module_text(&get_binding("op"), &get_helper(), &invoke_body("invoke_op"));
        let (ctx, module) = parse(&text);
        let result = validate_operation_verifiers(&ctx, module);
        assert!(result.is_ok(), "{result}");

        let printed = print_module(&ctx, module.op());
        let (reparsed, reparsed_module) = parse(&printed);
        assert_eq!(printed, print_module(&reparsed, reparsed_module.op()));
        for expected in [
            "ability_v2.scope",
            "ability_v2.invoke_op",
            "ability_v2.resume",
            "ability_v2.yield",
        ] {
            assert!(printed.contains(expected), "missing {expected}\n{printed}");
        }
    }

    const NEVER_HELPER: &str = r#"    %fail = tribute_control.lambda(%reason: core.i32) -> core.i32 convention(cps) captures [] {
      tribute_control.return %reason
    }"#;

    fn never_binding() -> String {
        format!(
            r#", %fail {{handlers = [{{ability_ref = {STATE}, kind = "op", op_name = "fail", operation_result_type = core.never}}]}}"#
        )
    }

    #[test]
    fn never_helper_takes_no_token_and_returns_the_answer() {
        let body = format!(
            r#"      %never = ability_v2.invoke_op %value {{ability_ref = {STATE}, op_name = "fail"}} : core.never
      ability_v2.yield %value"#
        );
        let text = module_text(&never_binding(), &format!("{DONE}\n{NEVER_HELPER}"), &body);
        assert_eq!(errors(&text), "");
    }

    #[test]
    fn fn_helper_returns_the_operation_result_without_cps() {
        let helper = format!(
            r#"{DONE}
    %get = tribute_control.lambda() -> core.i32 convention(evidence_direct) captures [%value] {{
      tribute_control.return %value
    }}"#
        );
        let text = module_text(&get_binding("fn"), &helper, &invoke_body("call_fn"));
        assert_eq!(errors(&text), "");
    }

    #[test]
    fn malformed_scopes_and_invocations_are_rejected() {
        let op_body = invoke_body("invoke_op");
        let done_only = DONE.to_string();
        let cases: Vec<(&str, String, &str)> = vec![
            (
                "fn binding with a cps helper",
                module_text(
                    &get_binding("fn"),
                    &get_helper(),
                    &invoke_body("call_fn"),
                ),
                "must not use the cps convention",
            ),
            (
                "op helper without its resume token",
                module_text(
                    &get_binding("op"),
                    &format!(
                        r#"{DONE}
    %get = tribute_control.lambda() -> core.i32 convention(cps) captures [%value] {{
      tribute_control.return %value
    }}"#
                    ),
                    &op_body,
                ),
                "must end with resume_token<operation result, scope answer>",
            ),
            (
                "Never helper with a resume token",
                module_text(
                    &never_binding()
                        .replace("\"fail\"", "\"get\"")
                        .replace("%fail", "%get"),
                    &get_helper(),
                    &op_body,
                ),
                "helper of Never operation get must not take a resume token",
            ),
            (
                "helper count differs from handlers",
                module_text(
                    &format!(
                        r#" {{handlers = [{{ability_ref = {STATE}, kind = "op", op_name = "get", operation_result_type = core.i32}}]}}"#
                    ),
                    &done_only,
                    &op_body,
                ),
                "has 1 handlers but 0 helpers",
            ),
            (
                "operation bound twice",
                module_text(
                    &format!(
                        r#", %get, %get {{handlers = [{{ability_ref = {STATE}, kind = "op", op_name = "get", operation_result_type = core.i32}}, {{ability_ref = {STATE}, kind = "op", op_name = "get", operation_result_type = core.i32}}]}}"#
                    ),
                    &get_helper(),
                    &op_body,
                ),
                "binds operation get twice",
            ),
            (
                "helper with another use",
                module_text(
                    &get_binding("op"),
                    &format!(
                        "{}\n    %alias = tribute_control.call_indirect %get : core.i32",
                        get_helper()
                    ),
                    &op_body,
                ),
                "must have the scope as its only use",
            ),
            (
                "completion of another type",
                module_text(
                    &get_binding("op"),
                    &get_helper().replace("%completed: core.i32", "%completed: core.i64"),
                    &op_body,
                ),
                "completion must take the body value and return the scope answer",
            ),
            (
                "body without a yield",
                module_text(
                    &get_binding("op"),
                    &get_helper(),
                    &format!(
                        r#"      %got = ability_v2.invoke_op {{ability_ref = {STATE}, op_name = "get"}} : core.i32
      tribute_control.return %got"#
                    ),
                ),
                "body must end with ability_v2.yield",
            ),
            (
                "yield outside a scope body",
                module_text(
                    &get_binding("op"),
                    &format!(
                        r#"{DONE}
    %get = tribute_control.lambda(%token: tribute_control.resume_token<core.i32, core.i32>) -> core.i32 convention(cps) captures [%value] {{
      ability_v2.yield %value
    }}"#
                    ),
                    &op_body,
                ),
                "ability_v2.yield must terminate an ability_v2.scope body",
            ),
            (
                "invoke_op in an evidence-direct callable",
                module_text(&get_binding("op"), &get_helper(), &op_body).replace(
                    "-> core.i32 convention(cps) {",
                    "-> core.i32 convention(evidence_direct) {",
                ),
                "ability_v2.invoke_op requires an enclosing callable with at least the Cps convention",
            ),
            (
                "invocation of a non-ability",
                module_text(
                    &get_binding("op"),
                    &get_helper(),
                    &op_body.replace(STATE, "core.i32"),
                ),
                "ability_ref must be a core.ability_ref type",
            ),
            (
                "fn helper returning another type",
                module_text(
                    &get_binding("fn"),
                    &format!(
                        r#"{DONE}
    %get = tribute_control.lambda() -> core.i64 convention(evidence_direct) captures [] {{
      %wide = arith.const {{value = 0}} : core.i64
      tribute_control.return %wide
    }}"#
                    ),
                    &invoke_body("call_fn"),
                ),
                "helper of fn operation get must return the operation result",
            ),
            (
                "fn helper with a resume token",
                module_text(
                    &get_binding("fn"),
                    &get_helper().replace("convention(cps) captures [%value]", "convention(evidence_direct) captures [%value]"),
                    &invoke_body("call_fn"),
                ),
                "helper of fn operation get must not take a resume token",
            ),
            (
                "op helper without the cps convention",
                module_text(
                    &get_binding("op"),
                    &get_helper().replace("convention(cps) captures [%value]", "convention(evidence_direct) captures [%value]"),
                    &op_body,
                ),
                "helper of op operation get must use the cps convention",
            ),
            (
                "op helper returning another type",
                module_text(
                    &get_binding("op"),
                    &format!(
                        r#"{DONE}
    %get = tribute_control.lambda(%token: tribute_control.resume_token<core.i32, core.i32>) -> core.i64 convention(cps) captures [] {{
      %wide = arith.const {{value = 0}} : core.i64
      tribute_control.return %wide
    }}"#
                    ),
                    &op_body,
                ),
                "helper of op operation get must return the scope answer",
            ),
            (
                "helper that is not a lambda",
                module_text(&get_binding("op"), DONE, &op_body)
                    .replace("@run(%value: core.i32)", "@run(%value: core.i32, %get: tribute_control.func_sig<(tribute_control.resume_token<core.i32, core.i32>) -> core.i32, {tribute.calling_convention = 2}>)"),
                "ability_v2.scope helper must be a tribute_control.lambda",
            ),
            (
                "call_fn of a Never operation",
                module_text(
                    &never_binding(),
                    &format!("{DONE}\n{NEVER_HELPER}"),
                    &format!(
                        r#"      %never = ability_v2.call_fn %value {{ability_ref = {STATE}, op_name = "fail"}} : core.never
      ability_v2.yield %value"#
                    ),
                ),
                "an operation returning core.never must use ability_v2.invoke_op",
            ),
            (
                "scope evidence_plan that duplicates",
                module_text(
                    &get_binding("op").replace(
                        "]}",
                        &format!("], evidence_plan = [{{dup = {STATE}}}]}}"),
                    ),
                    &get_helper(),
                    &op_body,
                ),
                "a handle's evidence_plan may only mask",
            ),
            (
                "binding with an unknown kind",
                module_text(&get_binding("pure"), &get_helper(), &op_body),
                "handlers kind must be fn or op",
            ),
            (
                "binding without an operation result type",
                module_text(
                    &get_binding("op").replace(", operation_result_type = core.i32", ""),
                    &get_helper(),
                    &op_body,
                ),
                "handlers element must have exactly",
            ),
            (
                "binding of a non-ability",
                module_text(
                    &get_binding("op").replace(STATE, "core.i32"),
                    &get_helper(),
                    &op_body,
                ),
                "handlers ability_ref must be a core.ability_ref type",
            ),
        ];
        for (name, text, expected) in cases {
            let errors = errors(&text);
            assert!(
                errors.contains(expected),
                "{name}: missing `{expected}` in:\n{errors}\n\n{text}"
            );
        }
    }

    #[test]
    fn handler_binding_round_trips_through_its_attribute() {
        let mut ctx = IrContext::new();
        let name = ctx.string_attr("State");
        let ability_ref = ctx.intern_type(
            trunk_ir::types::TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ability_ref"))
                .attr("name", name)
                .build(),
        );
        let i32_ty = ctx.intern_type(trunk_ir::types::TypeDataBuilder::new("core", "i32").build());
        let binding = HandlerBinding {
            ability_ref,
            op_name: ctx.intern_str("get"),
            kind: OperationKind::Fn,
            operation_result_type: i32_ty,
        };
        let attr = binding.to_attribute(&mut ctx);
        assert_eq!(HandlerBinding::from_attribute(&ctx, &attr), Ok(binding));
        assert_eq!(OperationKind::Op.keyword(), "op");
        assert_eq!(
            HandlerBinding::from_attribute(&ctx, &Attribute::Int(0)),
            Err("handlers element must be a dictionary".to_string())
        );
    }
}
