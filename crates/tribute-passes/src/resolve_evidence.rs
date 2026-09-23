//! Evidence-based dispatch resolution pass.
//!
//! Shared CPS legalization has already established callable signatures and
//! explicit evidence operands. This pass resolves handler prompt identities and
//! replaces each delimiter body evidence argument with its extended evidence.

use std::error::Error;
use std::fmt;

use tribute_ir::dialect::ability;
use tribute_ir::dialect::effect;
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::core;
use trunk_ir::dialect::func;
use trunk_ir::ops::DialectOp;
use trunk_ir::pass::{Pass, PassRunResult};
use trunk_ir::refs::{OpRef, RegionRef, TypeRef, ValueDef, ValueRef};
use trunk_ir::rewrite::{Module, erase_op};
use trunk_ir::types::{Attribute, TypeDataBuilder};

// ============================================================================
// Helper type constructors
// ============================================================================

fn i32_type_ref(ctx: &mut IrContext) -> TypeRef {
    ctx.intern_type(TypeDataBuilder::new("core", "i32").build())
}

#[derive(Debug)]
pub(crate) struct ResolveEvidenceError {
    op: OpRef,
    source_location: String,
    message: String,
}

impl ResolveEvidenceError {
    pub(crate) fn op(&self) -> OpRef {
        self.op
    }
}

impl fmt::Display for ResolveEvidenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} at {}: {}",
            self.op, self.source_location, self.message
        )
    }
}

impl Error for ResolveEvidenceError {}

struct FinalHandleDispatchShape {
    evidence: ValueRef,
    prompt_tag: ValueRef,
    dispatcher_pairs: Vec<(TypeRef, ValueRef, ValueRef)>,
    body_evidence: ValueRef,
}

fn final_handle_dispatch_shape(
    ctx: &IrContext,
    op: OpRef,
) -> Result<FinalHandleDispatchShape, ResolveEvidenceError> {
    let data = ctx.op(op);
    let error = |message: String| ResolveEvidenceError {
        op,
        source_location: format!(
            "{}:{}:{}",
            ctx.paths().get(data.location.path),
            data.location.span.start,
            data.location.span.end
        ),
        message,
    };
    let operands = ctx.op_operands(op);
    if !ctx.op_result_types(op).is_empty() {
        return Err(error(
            "final ability.handle_dispatch must be resultless".into(),
        ));
    }
    let Some((&evidence, rest)) = operands.split_first() else {
        return Err(error(
            "final ability.handle_dispatch requires an evidence operand".into(),
        ));
    };
    let Some((&prompt_tag, dispatchers)) = rest.split_first() else {
        return Err(error(
            "final ability.handle_dispatch requires a prompt-tag operand".into(),
        ));
    };
    let prompt_ty = ctx.get_type(ctx.value_ty(prompt_tag));
    if prompt_ty.dialect != Symbol::new("core") || prompt_ty.name != Symbol::new("i32") {
        return Err(error(
            "final ability.handle_dispatch prompt-tag operand must have type core.i32".into(),
        ));
    }
    if !dispatchers.len().is_multiple_of(2) {
        return Err(error(
            "final ability.handle_dispatch dispatcher operands must form exact pairs".into(),
        ));
    }
    let Some(Attribute::List(ability_refs)) = data.attributes.get("ability_refs") else {
        return Err(error(
            "final ability.handle_dispatch requires an ability_refs list".into(),
        ));
    };
    if ability_refs.len() * 2 != dispatchers.len() {
        return Err(error(format!(
            "ability_refs cardinality {} does not match {} dispatcher operands",
            ability_refs.len(),
            dispatchers.len()
        )));
    }
    let mut dispatcher_pairs = Vec::with_capacity(ability_refs.len());
    for (ability_ref, pair) in ability_refs.iter().zip(dispatchers.as_chunks::<2>().0) {
        let Attribute::Type(ability_ref) = ability_ref else {
            return Err(error("every ability_refs entry must be a type".into()));
        };
        let ty = ctx.get_type(*ability_ref);
        if ty.dialect != Symbol::new("core") || ty.name != Symbol::new("ability_ref") {
            return Err(error(
                "every ability_refs entry must be a core.ability_ref type".into(),
            ));
        }
        dispatcher_pairs.push((*ability_ref, pair[0], pair[1]));
    }
    let [body] = data.regions.as_slice() else {
        return Err(error(
            "final ability.handle_dispatch requires exactly one body region".into(),
        ));
    };
    let [body_block] = ctx.region(*body).blocks.as_slice() else {
        return Err(error(
            "final ability.handle_dispatch body must have exactly one block".into(),
        ));
    };
    let [body_evidence] = ctx.block_args(*body_block) else {
        return Err(error(
            "final ability.handle_dispatch body must have one evidence argument".into(),
        ));
    };
    if ctx.value_ty(*body_evidence) != ctx.value_ty(evidence)
        || !ability::is_evidence_type_ref(ctx, ctx.value_ty(evidence))
    {
        return Err(error(
            "final ability.handle_dispatch evidence operand and body argument must have the evidence type"
                .into(),
        ));
    }
    let Some(&terminator) = ctx.block(*body_block).ops.last() else {
        return Err(error(
            "final ability.handle_dispatch body must end in a proper tail transfer".into(),
        ));
    };
    if !trunk_ir::validation::is_proper_tail_terminator(ctx, terminator) {
        return Err(error(
            "final ability.handle_dispatch body must end in a proper tail transfer".into(),
        ));
    }
    Ok(FinalHandleDispatchShape {
        evidence,
        prompt_tag,
        dispatcher_pairs,
        body_evidence: *body_evidence,
    })
}

pub(crate) fn validate_final_handle_dispatches(
    ctx: &IrContext,
    module: Module,
) -> Result<(), ResolveEvidenceError> {
    fn visit(ctx: &IrContext, op: OpRef) -> Result<(), ResolveEvidenceError> {
        if ability::HandleDispatch::matches(ctx, op) {
            final_handle_dispatch_shape(ctx, op)?;
        }
        for region in ctx.op(op).regions.iter().copied() {
            for block in ctx.region(region).blocks.iter().copied() {
                for child in ctx.block(block).ops.iter().copied() {
                    visit(ctx, child)?;
                }
            }
        }
        Ok(())
    }
    visit(ctx, module.op())
}

// ============================================================================
// Analysis helpers
// ============================================================================

/// Ensure the runtime prompt allocator is declared.
fn ensure_prompt_tag_runtime(ctx: &mut IrContext, module: Module) {
    let has_next_tag = module.ops(ctx).into_iter().any(|op| {
        func::Func::from_op(ctx, op)
            .is_ok_and(|function| function.sym_name(ctx) == Symbol::new("__tribute_next_tag"))
    });
    let Some(module_block) = module.first_block(ctx) else {
        return;
    };
    let loc = ctx.op(module.op()).location;

    if !has_next_tag {
        let i32_ty = i32_type_ref(ctx);

        // fn __tribute_next_tag() -> i32
        let func_ty = func::func_sig(ctx, std::iter::empty(), [i32_ty]).as_type_ref();

        let data =
            trunk_ir::OperationDataBuilder::new(loc, Symbol::new("func"), Symbol::new("func"))
                .attr(
                    "sym_name",
                    Attribute::Symbol(Symbol::new("__tribute_next_tag")),
                )
                .attr("type", Attribute::Type(func_ty))
                .attr("abi", Attribute::String("C".to_owned()))
                .build(ctx);
        let func_op = ctx.create_op(data);
        let first_op = ctx.block(module_block).ops.first().copied();
        if let Some(first) = first_op {
            ctx.insert_op_before(module_block, first, func_op);
        } else {
            ctx.push_op(module_block, func_op);
        }
    }
}

/// Resolve each explicit delimiter without changing callable signatures or calls.
fn resolve_delimiters(
    ctx: &mut IrContext,
    module: Module,
    region: RegionRef,
) -> Result<(), ResolveEvidenceError> {
    let blocks = ctx.region(region).blocks.to_vec();
    for block in blocks {
        let ops = ctx.block(block).ops.to_vec();
        for op in ops {
            if ability::HandleDispatch::from_op(ctx, op).is_ok() {
                let location = ctx.op(op).location;
                let shape = final_handle_dispatch_shape(ctx, op)?;
                let mut current_ev = shape.evidence;
                let mut prompt_tag = shape.prompt_tag;
                if let ValueDef::OpResult(prompt_op, _) = ctx.value_def(prompt_tag)
                    && effect::FreshPromptTag::from_op(ctx, prompt_op).is_ok()
                {
                    ensure_prompt_tag_runtime(ctx, module);
                    let i32_ty = i32_type_ref(ctx);
                    let prompt = func::call(
                        ctx,
                        location,
                        std::iter::empty::<ValueRef>(),
                        [i32_ty],
                        Symbol::new("__tribute_next_tag"),
                    );
                    let resolved = prompt.result(ctx);
                    ctx.insert_op_before(block, op, prompt.op_ref());
                    ctx.replace_all_uses(prompt_tag, resolved);
                    erase_op(ctx, prompt_op);
                    prompt_tag = resolved;
                }
                let evidence_ty = ability::evidence_adt_type_ref(ctx);
                for (ability_ref, tr_dispatch, handler_dispatch) in shape.dispatcher_pairs {
                    let extend = effect::extend(
                        ctx,
                        location,
                        current_ev,
                        prompt_tag,
                        tr_dispatch,
                        handler_dispatch,
                        evidence_ty,
                        ability_ref,
                    );
                    current_ev = extend.result(ctx);
                    ctx.insert_op_before(block, op, extend.op_ref());
                }

                ctx.replace_all_uses(shape.body_evidence, current_ev);
            }
            let regions = ctx.op(op).regions.to_vec();
            for region in regions {
                resolve_delimiters(ctx, module, region)?;
            }
        }
    }
    Ok(())
}

/// Resolve runtime prompt identities and explicit handler evidence extensions.
pub(crate) fn resolve_evidence_dispatch(
    ctx: &mut IrContext,
    module: Module,
) -> Result<(), ResolveEvidenceError> {
    validate_final_handle_dispatches(ctx, module)?;
    if let Some(body) = module.body(ctx) {
        resolve_delimiters(ctx, module, body)?;
    }
    Ok(())
}

/// PassManager-friendly wrapper for [`resolve_evidence_dispatch`].
pub struct ResolveEvidenceDispatch;

impl Pass for ResolveEvidenceDispatch {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "resolve-evidence-dispatch"
    }

    fn run(&mut self, ctx: &mut IrContext, target: core::Module) -> PassRunResult {
        resolve_evidence_dispatch(ctx, target.into()).map_err(Into::into)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::ControlFlow;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;
    use trunk_ir::types::TypeDataBuilder;
    use trunk_ir::walk::{WalkAction, walk_op};

    #[test]
    fn test_compute_ability_id() {
        let mut ctx = IrContext::new();

        let state_ref = ctx.intern_type(
            TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ability_ref"))
                .attr("name", Attribute::Symbol(Symbol::new("State")))
                .build(),
        );
        let console_ref = ctx.intern_type(
            TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ability_ref"))
                .attr("name", Attribute::Symbol(Symbol::new("Console")))
                .build(),
        );

        let state_id = ability::compute_ability_id(&ctx, state_ref);
        let console_id = ability::compute_ability_id(&ctx, console_ref);

        // Same ability should have same ID (interning gives same TypeRef)
        let state_ref2 = ctx.intern_type(
            TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ability_ref"))
                .attr("name", Attribute::Symbol(Symbol::new("State")))
                .build(),
        );
        let state_id2 = ability::compute_ability_id(&ctx, state_ref2);
        assert_eq!(state_id, state_id2);

        // Different abilities should have different IDs
        assert_ne!(state_id, console_id);
    }

    #[test]
    fn test_compute_ability_id_with_type_params() {
        let mut ctx = IrContext::new();

        let i32_ty = ctx.intern_type(TypeDataBuilder::new("core", "i32").build());

        let state_i32 = ctx.intern_type(
            TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ability_ref"))
                .attr("name", Attribute::Symbol(Symbol::new("State")))
                .param(i32_ty)
                .build(),
        );

        let state_no_params = ctx.intern_type(
            TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ability_ref"))
                .attr("name", Attribute::Symbol(Symbol::new("State")))
                .build(),
        );

        let id_with_params = ability::compute_ability_id(&ctx, state_i32);
        let id_no_params = ability::compute_ability_id(&ctx, state_no_params);

        // Same ability name but different type params should produce different IDs
        assert_ne!(id_with_params, id_no_params);
    }

    fn final_dispatch_fixture(operation: &str) -> String {
        format!(
            r#"core.module @test {{
  !marker = adt.struct() {{fields = [[@ability_id, core.i32], [@prompt_tag, core.i32], [@tr_dispatch_fn, core.ptr], [@handler_dispatch, core.ptr]], name = @_Marker}}
  !evidence = core.array(!marker)
  func.func @test(%ev: !evidence, %prompt: core.i32, %tr: core.ptr, %handler: core.ptr, %tr2: core.ptr, %handler2: core.ptr) -> core.never {{
    {operation}
  }}
}}"#
        )
    }

    #[test]
    fn final_handle_dispatch_extends_each_ability_pair_and_lowers_resultlessly() {
        let input = final_dispatch_fixture(
            r#"ability.handle_dispatch %ev, %prompt, %tr, %handler, %tr2, %handler2 {ability_refs = [core.ability_ref() {name = @State}, core.ability_ref() {name = @Console}]} {
      ^body(%inner: !evidence):
        func.unreachable
    }"#,
        );
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, &input);
        resolve_evidence_dispatch(&mut ctx, module).unwrap();
        let resolved = print_module(&ctx, module.op());
        assert_eq!(resolved.matches("effect.extend").count(), 2);
        for name in [ability::evidence_abi::LOOKUP, ability::evidence_abi::EXTEND] {
            assert!(
                module.ops(&ctx).into_iter().all(|op| {
                    ctx.op(op).attributes.get_symbol("sym_name") != Some(Symbol::new(name))
                }),
                "shared resolution must not fabricate target helper {name}"
            );
        }
        assert_eq!(resolved.matches("func.call").count(), 0);
        assert!(!resolved.contains("__tribute_next_tag"));
        assert!(resolved.contains("ability.handle_dispatch"));
        let mut extensions = Vec::new();
        let _ = walk_op::<()>(&ctx, module.op(), &mut |op| {
            if effect::Extend::from_op(&ctx, op).is_ok() {
                extensions.push(op);
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
        assert_eq!(extensions.len(), 2);
        assert_eq!(
            ctx.op_operands(extensions[0])[1],
            ctx.op_operands(extensions[1])[1],
            "all abilities in one delimiter must share one runtime prompt tag"
        );
        crate::lower_handle_dispatch::lower_handle_dispatch(&mut ctx, module).unwrap();
        let lowered = print_module(&ctx, module.op());
        assert!(!lowered.contains("ability."));
        assert!(!lowered.contains("__tribute_cps_control"));
        assert!(lowered.contains("func.unreachable"));

        let mut reparsed = IrContext::new();
        parse_test_module(&mut reparsed, &lowered);
    }

    #[test]
    fn final_handle_dispatch_materializes_a_fresh_prompt_tag_once() {
        let input = final_dispatch_fixture(
            r#"%fresh = effect.fresh_prompt_tag : core.i32
    ability.handle_dispatch %ev, %fresh, %tr, %handler {ability_refs = [core.ability_ref() {name = @State}]} {
      ^body(%inner: !evidence):
        func.unreachable
    }"#,
        );
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, &input);

        resolve_evidence_dispatch(&mut ctx, module).unwrap();

        let resolved = print_module(&ctx, module.op());
        assert!(!resolved.contains("effect.fresh_prompt_tag"), "{resolved}");
        assert_eq!(
            resolved
                .matches("func.call {callee = @__tribute_next_tag}")
                .count(),
            1,
            "{resolved}"
        );
        let next_tag = module
            .ops(&ctx)
            .iter()
            .copied()
            .find(|&op| {
                ctx.op(op).attributes.get_symbol("sym_name")
                    == Some(Symbol::new("__tribute_next_tag"))
            })
            .expect("runtime tag declaration");
        assert_eq!(
            trunk_ir::callable::classify_callable_body(&ctx, next_tag),
            Ok(trunk_ir::callable::CallableBody::Declaration)
        );
        assert_eq!(ctx.op(next_tag).attributes.get_str("abi"), Some("C"));
        assert_eq!(resolved.matches("func.call").count(), 1, "{resolved}");
    }

    #[test]
    fn malformed_final_handle_dispatches_fail_before_mutation() {
        let malformed = [
            (
                r#"%result = ability.handle_dispatch %ev, %prompt {ability_refs = []} : core.i32 {
      ^body(%inner: !evidence):
        func.unreachable
    }"#,
                "must be resultless",
            ),
            (
                r#"ability.handle_dispatch {ability_refs = []} {
      ^body(%inner: !evidence):
        func.unreachable
    }"#,
                "requires an evidence operand",
            ),
            (
                r#"ability.handle_dispatch %ev, %prompt, %tr {ability_refs = []} {
      ^body(%inner: !evidence):
        func.unreachable
    }"#,
                "must form exact pairs",
            ),
            (
                r#"ability.handle_dispatch %ev {
      ^body(%inner: !evidence):
        func.unreachable
    }"#,
                "requires a prompt-tag operand",
            ),
            (
                r#"ability.handle_dispatch %ev, %handler {ability_refs = []} {
      ^body(%inner: !evidence):
        func.unreachable
    }"#,
                "prompt-tag operand must have type core.i32",
            ),
            (
                r#"ability.handle_dispatch %ev, %prompt {
      ^body(%inner: !evidence):
        func.unreachable
    }"#,
                "requires an ability_refs list",
            ),
            (
                r#"ability.handle_dispatch %ev, %prompt, %tr, %handler {ability_refs = []} {
      ^body(%inner: !evidence):
        func.unreachable
    }"#,
                "cardinality",
            ),
            (
                r#"ability.handle_dispatch %ev, %prompt, %tr, %handler {ability_refs = [1]} {
      ^body(%inner: !evidence):
        func.unreachable
    }"#,
                "entry must be a type",
            ),
            (
                r#"ability.handle_dispatch %ev, %prompt, %tr, %handler {ability_refs = [core.i32]} {
      ^body(%inner: !evidence):
        func.unreachable
    }"#,
                "core.ability_ref type",
            ),
            (
                r#"ability.handle_dispatch %prompt, %prompt {ability_refs = []} {
      ^body(%inner: core.i32):
        func.unreachable
    }"#,
                "must have the evidence type",
            ),
            (
                r#"ability.handle_dispatch %ev, %prompt {ability_refs = []} {
    }"#,
                "exactly one block",
            ),
            (
                r#"ability.handle_dispatch %ev, %prompt {ability_refs = []} {
      ^first(%inner: !evidence):
        func.unreachable
      ^second:
        func.unreachable
    }"#,
                "exactly one block",
            ),
            (
                r#"ability.handle_dispatch %ev, %prompt {ability_refs = []} {
      ^body:
        func.unreachable
    }"#,
                "one evidence argument",
            ),
            (
                r#"ability.handle_dispatch %ev, %prompt {ability_refs = []} {
      ^body(%inner: !evidence):
    }"#,
                "must end in a proper tail transfer",
            ),
            (
                r#"ability.handle_dispatch %ev, %prompt {ability_refs = []} {
      ^body(%inner: !evidence):
        func.return
    }"#,
                "proper tail transfer",
            ),
        ];
        for (operation, expected) in malformed {
            let input = final_dispatch_fixture(operation);
            let mut ctx = IrContext::new();
            let module = parse_test_module(&mut ctx, &input);
            let before = print_module(&ctx, module.op());
            let error = resolve_evidence_dispatch(&mut ctx, module).unwrap_err();
            assert!(error.to_string().contains(expected), "{error}");
            assert!(error.to_string().contains("textual-ir:0:0"), "{error}");
            assert_eq!(print_module(&ctx, module.op()), before);
        }
    }

    #[test]
    fn structured_final_delimiter_is_a_valid_proper_tail_body() {
        let input = final_dispatch_fixture(
            r#"ability.handle_dispatch %ev, %prompt {ability_refs = []} {
      ^body(%inner: !evidence):
        %choice = arith.const {value = 0} : core.i32
        scf.switch %choice {
          scf.default {
            func.unreachable
          }
        }
    }"#,
        );
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, &input);
        validate_final_handle_dispatches(&ctx, module).unwrap();
    }

    #[test]
    fn final_delimiter_accepts_resultless_effect_cps_dispatch() {
        let input = final_dispatch_fixture(
            r#"ability.handle_dispatch %ev, %prompt {ability_refs = []} {
      ^body(%inner: !evidence):
        effect.dispatch_cps %inner, %tr, %handler, %tr2 {ability_ref = core.ability_ref() {name = @State}, op_name = @get}
    }"#,
        );
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, &input);
        validate_final_handle_dispatches(&ctx, module).unwrap();
    }

    #[test]
    fn bodyless_declarations_are_preserved_during_evidence_resolution() {
        let input = r#"core.module @test {
  !marker = adt.struct() {fields = [[@ability_id, core.i32], [@prompt_tag, core.i32], [@tr_dispatch_fn, core.ptr], [@handler_dispatch, core.ptr]], name = @_Marker}
  !evidence = core.array(!marker)
  func.func @plain_external() -> core.i32
  func.func @evidence_external(%ev: !evidence) -> !marker
  func.func @body(%ev: !evidence, %prompt: core.i32) -> core.never {
    ability.handle_dispatch %ev, %prompt {ability_refs = []} {
      ^body(%inner: !evidence):
        func.tail_call %inner {callee = @finish}
    }
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        resolve_evidence_dispatch(&mut ctx, module).unwrap();

        let resolved = print_module(&ctx, module.op());
        assert!(
            resolved.contains("func.func @plain_external() -> core.i32\n"),
            "plain external declaration changed or disappeared:\n{resolved}"
        );
        assert!(
            resolved.contains("func.func @evidence_external(%arg0: !evidence) -> !marker\n"),
            "evidence-bearing external declaration changed or disappeared:\n{resolved}"
        );
        let body = module
            .ops(&ctx)
            .into_iter()
            .find(|&op| ctx.op(op).attributes.get_symbol("sym_name") == Some(Symbol::new("body")))
            .unwrap();
        let entry = ctx.region(ctx.op(body).regions[0]).blocks[0];
        let outer_evidence = ctx.block_args(entry)[0];
        let delimiter = ctx.block(entry).ops[0];
        let inner = ctx.region(ctx.op(delimiter).regions[0]).blocks[0];
        let tail = ctx.block(inner).ops[0];
        assert_eq!(ctx.op_operands(tail), &[outer_evidence]);
    }
}
