//! Expansion of the abstract continuation frame surface that
//! `tribute_control_to_cps` emits.
//!
//! `ability.frame<R>` becomes the nominal frame reference of `R`, and
//! `ability.suffix_frame` and `ability.exit` expand into the done adapter,
//! dispatch adapter factory, and frame struct, or the transfer to the frame's
//! `Done<R>`. `ability.handle` expands into the handle layer's factories and
//! its `ability.handle_dispatch` delimiter, and `ability.perform` and
//! `ability.abort` into `effect.dispatch_cps` through the frame's dispatcher.

use std::cell::Cell;
use std::ops::ControlFlow;

use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;
use tribute_ir::continuation_frame;
use tribute_ir::dialect::{ability, adt};
use trunk_ir::Symbol;
use trunk_ir::analysis::AnalysisCache;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::{core, func};
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::pass::{Pass, PassRunResult};
use trunk_ir::refs::{BlockRef, OpRef, TypeRef};
use trunk_ir::rewrite::{
    Module, PatternApplicator, PatternRewriter, RewritePattern, TypeConverter,
};
use trunk_ir::types::Location;
use trunk_ir::walk::{WalkAction, walk_op};

use crate::closure_lower::{TypeSubstitution, substitute_module_types_keeping_casts};
use crate::cps_builders::{FrameTypes, helper_symbol};
use crate::tribute_control_to_cps::TributeControlToCpsError;

mod handle_layer;
mod perform;
mod suffix_layer;

/// Pass-manager wrapper of [`lower_continuation_frames`].
pub struct LowerContinuationFrames;

impl Pass for LowerContinuationFrames {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "lower-continuation-frames"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: core::Module,
        _analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        lower_continuation_frames(ctx, target.into()).map_err(|error| Box::new(error) as _)
    }
}

/// Replace the abstract frame surface of `module` with the frame layouts
/// `tribute_control_to_cps` registered.
pub fn lower_continuation_frames(
    ctx: &mut IrContext,
    module: Module,
) -> Result<(), TributeControlToCpsError> {
    let location = ctx.op(module.op()).location;
    let module_block = module
        .body(ctx)
        .and_then(|body| ctx.region(body).blocks.first().copied())
        .ok_or_else(|| TributeControlToCpsError::post_at(location, "module has no body block"))?;

    let replacements = frame_references(ctx);
    substitute_module_types_keeping_casts(ctx, module, |_, ty| replacements.get(&ty).copied());

    PatternApplicator::new(TypeConverter::new())
        .add_pattern(ExpandFrameOperations {
            frames: frame_types(ctx, location)?,
            next_helper: Cell::new(next_helper_index(ctx, module_block)),
        })
        .apply_partial(ctx, module);
    reject_abstract_frames(ctx, module)
}

/// Expands the abstract frame operations. A failure leaves the operation for
/// [`reject_abstract_frames`] to report.
struct ExpandFrameOperations {
    frames: HashMap<TypeRef, FrameTypes>,
    next_helper: Cell<u32>,
}

impl RewritePattern for ExpandFrameOperations {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if let Ok(suffix) = ability::SuffixFrame::from_op(ctx, op) {
            self.expand_suffix_frame(ctx, suffix, rewriter).is_some()
        } else if let Ok(exit) = ability::Exit::from_op(ctx, op) {
            self.expand_exit(ctx, exit, rewriter).is_some()
        } else if let Ok(handle) = ability::Handle::from_op(ctx, op) {
            self.expand_handle(ctx, handle, rewriter).is_some()
        } else if let Ok(perform) = ability::Perform::from_op(ctx, op) {
            let resumption = Some(perform.resumption(ctx));
            let values = perform.values(ctx).to_vec();
            self.expand_dispatch(ctx, op, perform.frame(ctx), resumption, &values, rewriter)
                .is_some()
        } else if let Ok(abort) = ability::Abort::from_op(ctx, op) {
            let values = abort.values(ctx).to_vec();
            self.expand_dispatch(ctx, op, abort.frame(ctx), None, &values, rewriter)
                .is_some()
        } else {
            false
        }
    }
}

impl ExpandFrameOperations {
    fn frame_of(&self, ctx: &IrContext, frame: TypeRef) -> Option<FrameTypes> {
        let answer = continuation_frame::result_type(ctx, frame)?;
        self.frames.get(&answer).copied()
    }

    fn fresh_helper(&self, prefix: &str) -> Symbol {
        helper_symbol(prefix, self.next_helper.replace(self.next_helper.get() + 1))
    }
}

/// Hand the operations built into `block` to the rewriter, to be inserted
/// where the replaced operation was.
fn detach_into(ctx: &mut IrContext, block: BlockRef, rewriter: &mut PatternRewriter<'_>) {
    for built in ctx.block(block).ops.clone() {
        ctx.remove_op_from_block(block, built);
        rewriter.insert_op(built);
    }
}

/// The nominal reference of the layout of each `ability.frame<R>`.
///
/// The layout is named by the answer type after its own frames are
/// converted, so an answer that mentions frames resolves them first.
fn frame_references(ctx: &mut IrContext) -> HashMap<TypeRef, TypeRef> {
    let names = frame_layouts(ctx);
    let mut resolved = HashMap::default();
    for &answer in names.keys() {
        resolve_frame(ctx, answer, &names, &mut resolved);
    }
    resolved
        .into_iter()
        .map(|(answer, reference)| (ability::frame(ctx, answer).as_type_ref(), reference))
        .collect()
}

fn resolve_frame(
    ctx: &mut IrContext,
    answer: TypeRef,
    names: &HashMap<TypeRef, String>,
    resolved: &mut HashMap<TypeRef, TypeRef>,
) -> Option<TypeRef> {
    if let Some(&reference) = resolved.get(&answer) {
        return Some(reference);
    }
    let name = names.get(&answer)?;
    let converted = TypeSubstitution::new(ctx, |ctx, ty| {
        let inner = ability::Frame::from_type_ref(ctx, ty)?.result(ctx);
        resolve_frame(ctx, inner, names, &mut *resolved)
    })
    .convert_type(answer);
    let reference = continuation_frame::ref_type(ctx, name.clone(), converted);
    resolved.insert(answer, reference);
    Some(reference)
}

/// The module's frame layouts: answer type to layout name.
fn frame_layouts(ctx: &IrContext) -> HashMap<TypeRef, String> {
    ctx.type_aliases()
        .iter()
        .filter(|(name, _)| name.as_str().starts_with(continuation_frame::NAME_PREFIX))
        .filter_map(|(name, layout)| {
            let answer = continuation_frame::result_type(ctx, *layout)?;
            Some((answer, name.to_string()))
        })
        .collect()
}

/// The frame types of each registered layout.
fn frame_types(
    ctx: &mut IrContext,
    location: Location,
) -> Result<HashMap<TypeRef, FrameTypes>, TributeControlToCpsError> {
    let mut frames = HashMap::default();
    for (answer, name) in frame_layouts(ctx) {
        let layout = ctx
            .type_alias_by_text(&name)
            .expect("a listed frame layout is registered");
        let fields = adt::Struct::from_type_ref(ctx, layout)
            .map(|layout| layout.fields(ctx).collect::<Vec<_>>())
            .unwrap_or_default();
        let [("done", done), ("dispatch", dispatch)] = fields[..] else {
            return Err(TributeControlToCpsError::post_at(
                location,
                format!("continuation frame layout {name} must hold done and dispatch"),
            ));
        };
        let reference = continuation_frame::ref_type(ctx, name, answer);
        frames.insert(
            answer,
            FrameTypes {
                answer,
                reference,
                layout,
                done,
                dispatch,
            },
        );
    }
    Ok(frames)
}

fn type_mentions_frame(ctx: &IrContext, ty: TypeRef, seen: &mut HashSet<TypeRef>) -> bool {
    if !seen.insert(ty) {
        return false;
    }
    if ability::Frame::matches(ctx, ty) {
        return true;
    }
    let data = ctx.get_type(ty);
    let mut mentioned = data
        .params
        .iter()
        .any(|param| type_mentions_frame(ctx, *param, seen));
    for (_, value) in data.attrs.iter() {
        value.visit_types(&mut |inner| {
            mentioned = mentioned || type_mentions_frame(ctx, inner, seen);
        });
    }
    mentioned
}

/// An index past every `__tribute_<prefix>_<index>` helper of the module, so
/// helpers made here cannot collide with those of `tribute_control_to_cps`.
fn next_helper_index(ctx: &IrContext, module_block: BlockRef) -> u32 {
    ctx.block(module_block)
        .ops
        .iter()
        .filter_map(|op| func::Func::from_op(ctx, *op).ok())
        .filter_map(|function| {
            let name = function.sym_name(ctx);
            name.strip_prefix("__tribute_")?
                .rsplit('_')
                .next()?
                .parse::<u32>()
                .ok()
        })
        .map(|index| index + 1)
        .max()
        .unwrap_or(0)
}

/// Fail if an abstract frame operation or type survived the expansion, on any
/// surface the type substitution rewrites: aliases, operation attributes and
/// results, and block arguments with their attributes.
fn reject_abstract_frames(ctx: &IrContext, module: Module) -> Result<(), TributeControlToCpsError> {
    let mut seen = HashSet::default();
    let mut mentions_frame = |ctx: &IrContext, ty: TypeRef| type_mentions_frame(ctx, ty, &mut seen);
    let survived = |location| {
        TributeControlToCpsError::post_at(
            location,
            "an abstract continuation frame survived lower_continuation_frames",
        )
    };
    if ctx
        .type_aliases()
        .iter()
        .any(|(_, ty)| mentions_frame(ctx, *ty))
    {
        return Err(survived(ctx.op(module.op()).location));
    }
    let mut failure = None;
    let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
        let mut mentions = ctx
            .op_result_types(op)
            .iter()
            .any(|ty| mentions_frame(ctx, *ty));
        for (_, value) in ctx.op(op).attributes.iter() {
            value.visit_types(&mut |ty| mentions = mentions || mentions_frame(ctx, ty));
        }
        for region in ctx.op_regions(op) {
            for block in ctx.region(region).blocks.iter() {
                for arg in &ctx.block(*block).args {
                    mentions = mentions || mentions_frame(ctx, arg.ty);
                    for (_, value) in arg.attrs.iter() {
                        value.visit_types(&mut |ty| mentions = mentions || mentions_frame(ctx, ty));
                    }
                }
            }
        }
        let operation = ability::SuffixFrame::matches(ctx, op)
            || ability::Exit::matches(ctx, op)
            || ability::Handle::matches(ctx, op)
            || ability::Perform::matches(ctx, op)
            || ability::Abort::matches(ctx, op);
        if (operation || mentions) && failure.is_none() {
            failure = Some(survived(ctx.op(op).location));
        }
        ControlFlow::Continue(WalkAction::Advance)
    });
    failure.map_or(Ok(()), Err)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tribute_ir::dialect::{effect, tribute_rt};
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::types::Attribute;

    const MODULE: &str = "core.module @m {
  func.func @f(%x: core.i32) -> core.i32 { func.return %x }
}";

    fn parse(text: &str) -> (IrContext, Module) {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, text);
        (ctx, module)
    }

    fn unregistered_frame(ctx: &mut IrContext) -> TypeRef {
        let answer = ctx.intern_type(trunk_ir::types::TypeDataBuilder::new("core", "i32").build());
        ability::frame(ctx, answer).as_type_ref()
    }

    #[test]
    fn a_module_without_frames_lowers() {
        let (mut ctx, module) = parse(MODULE);
        assert!(lower_continuation_frames(&mut ctx, module).is_ok());
    }

    #[test]
    fn an_abstract_frame_in_a_type_alias_is_rejected() {
        let (mut ctx, module) = parse(MODULE);
        let frame = unregistered_frame(&mut ctx);
        ctx.register_type_alias("orphan".into(), frame);
        assert!(lower_continuation_frames(&mut ctx, module).is_err());
    }

    #[test]
    fn an_abstract_frame_in_a_block_argument_attribute_is_rejected() {
        let (mut ctx, module) = parse(MODULE);
        let frame = unregistered_frame(&mut ctx);
        let function = module.ops(&ctx)[0];
        let body = ctx.op_region(function, 0).expect("function body");
        let block = ctx.region(body).blocks[0];
        ctx.block_mut(block).args[0]
            .attrs
            .insert("frame", Attribute::Type(frame));
        assert!(lower_continuation_frames(&mut ctx, module).is_err());
    }

    #[test]
    fn a_handle_over_an_unregistered_frame_is_rejected() {
        let (mut ctx, module) = parse(
            r#"core.module @m {
  !marker = adt.struct<_Marker(ability_id: core.i32, prompt_tag: core.i32, tr_dispatch_fn: core.ptr, shadowed: core.ptr, outer: core.ptr), {layout = "evidence_marker"}>
  !ev = core.array<!marker, {layout = "evidence"}>
  !frame = ability.frame<core.i32>
  !completion = closure.closure<func.func_sig<(!ev, !frame, core.i32) -> core.never>, {tribute.calling_convention = 2, tribute.closure_environment_index = 0}>
  func.func @run(%ev: !ev, %exit: !frame, %done: !completion) -> core.never {
    ability.handle %ev, %exit, %done {handlers = []} {
      ^body(%inner: !ev, %frame: !frame):
        func.unreachable
    }
  }
}"#,
        );
        assert!(lower_continuation_frames(&mut ctx, module).is_err());
    }

    const TYPES: &str = r#"  !marker = adt.struct<_Marker(ability_id: core.i32, prompt_tag: core.i32, tr_dispatch_fn: core.ptr, shadowed: core.ptr, outer: core.ptr), {layout = "evidence_marker"}>
  !ev = core.array<!marker, {layout = "evidence"}>
  !state = core.ability_ref<{name = "State"}>
  !frame = ability.frame<core.i32>
  !resume = closure.closure<func.func_sig<(!ev, !frame, core.i32) -> core.never>, {tribute.calling_convention = 2, tribute.closure_environment_index = 0}>
  !nested = ability.frame<!resume>"#;

    /// Parse a module over `TYPES` whose `@run` has `params` and `body`, and
    /// register the layouts of `!frame` and `!nested` as the CPS pass does.
    fn frame_module(params: &str, body: &str) -> (IrContext, Module) {
        let (mut ctx, module) = parse(&format!(
            "core.module @m {{\n{TYPES}\n  func.func @run(%ev: !ev{params}) -> core.never {{\n{body}\n  }}\n}}"
        ));
        let evidence = ability::evidence_adt_type_ref(&mut ctx);
        let anyref = tribute_rt::anyref(&mut ctx).as_type_ref();
        let i32_type =
            ctx.intern_type(trunk_ir::types::TypeDataBuilder::new("core", "i32").build());
        let inner = ability::frame(&mut ctx, i32_type).as_type_ref();
        let resume = tribute_core::calling_convention::cps_resume_exact_type(
            &mut ctx, evidence, i32_type, inner,
        );
        for (index, answer) in [i32_type, resume].into_iter().enumerate() {
            let name = format!("{}{index}", continuation_frame::NAME_PREFIX);
            let frame = ability::frame(&mut ctx, answer).as_type_ref();
            let done = tribute_core::calling_convention::cps_done_type(&mut ctx, answer);
            let dispatch = tribute_core::calling_convention::cps_dispatch_type(
                &mut ctx, evidence, frame, anyref, i32_type,
            );
            let name_ref = ctx.intern_str(&name);
            let layout =
                continuation_frame::layout_type(&mut ctx, name_ref, answer, done, dispatch);
            ctx.register_type_alias(Symbol::new(&name), layout);
        }
        (ctx, module)
    }

    fn lowered(params: &str, body: &str) -> String {
        let (mut ctx, module) = frame_module(params, body);
        lower_continuation_frames(&mut ctx, module).expect("the frames lower");
        let printed = trunk_ir::printer::print_module(&ctx, module.op());
        assert!(!printed.contains("ability.frame"), "{printed}");
        printed
    }

    #[test]
    fn a_perform_dispatches_through_the_frame_with_a_one_shot_resumption() {
        let printed = lowered(
            ", %f: !frame, %k: !resume, %arg: core.i32",
            r#"    ability.perform %ev, %f, %k, %arg {ability_ref = !state, op_name = "set"}"#,
        );
        assert!(!printed.contains("ability.perform"), "{printed}");
        assert_eq!(
            printed.matches("effect.dispatch_cps").count(),
            1,
            "{printed}"
        );
        assert!(printed.contains("answer_type = core.i32"), "{printed}");
        assert!(printed.contains("__tribute_one_shot_state_0"), "{printed}");
        // The wrapper marks the state consumed before it enters the raw
        // resumption, and traps when it is already consumed.
        assert_eq!(printed.matches("adt.struct_set").count(), 1, "{printed}");
        assert_eq!(printed.matches("func.unreachable").count(), 1, "{printed}");
        assert_eq!(
            printed.matches("func.tail_call_indirect").count(),
            1,
            "{printed}"
        );
    }

    #[test]
    fn an_abort_dispatches_with_a_resumption_that_captures_nothing() {
        let (mut ctx, module) = frame_module(
            ", %f: !frame, %arg: core.i32",
            r#"    ability.abort %ev, %f, %arg {ability_ref = !state, op_name = "fail"}"#,
        );
        lower_continuation_frames(&mut ctx, module).unwrap();
        let printed = trunk_ir::printer::print_module(&ctx, module.op());
        assert!(!printed.contains("ability.abort"), "{printed}");
        assert!(!printed.contains("one_shot_state"), "{printed}");
        let mut dispatches = Vec::new();
        let _ = walk_op::<()>(&ctx, module.op(), &mut |op| {
            dispatches.extend(effect::DispatchCps::from_op(&ctx, op).ok());
            ControlFlow::Continue(WalkAction::Advance)
        });
        let [dispatch] = dispatches[..] else {
            panic!("expected one effect.dispatch_cps: {printed}");
        };
        let trunk_ir::ValueDef::OpResult(reject, _) = ctx.value_def(dispatch.resume(&ctx)) else {
            panic!("the resumption is built in place: {printed}");
        };
        assert!(ctx.op_operands(reject).is_empty(), "{printed}");
        let body = ctx.region(ctx.op_region(reject, 0).unwrap()).blocks[0];
        assert!(func::Unreachable::matches(&ctx, ctx.block(body).ops[0]));
    }

    #[test]
    fn a_frame_whose_answer_mentions_a_frame_lowers_both() {
        let printed = lowered(
            ", %outer: !nested, %k: !resume",
            "    ability.exit %outer, %k",
        );
        assert!(!printed.contains("ability.exit"), "{printed}");
        assert_eq!(
            printed.matches("func.tail_call_indirect").count(),
            1,
            "{printed}"
        );
        // The outer layout is named by its answer after the inner frame in
        // it became a reference to the inner layout.
        assert!(
            printed.contains(&format!("{}1", continuation_frame::NAME_PREFIX)),
            "{printed}"
        );
    }

    #[test]
    fn frame_operations_over_an_unregistered_frame_are_rejected() {
        const OTHER: &str = "ability.frame<core.i64>";
        let cases = [
            (
                format!(", %f: {OTHER}, %v: core.i64"),
                "    ability.exit %f, %v".to_string(),
            ),
            (
                format!(", %f: {OTHER}, %arg: core.i32"),
                r#"    ability.abort %ev, %f, %arg {ability_ref = !state, op_name = "fail"}"#
                    .to_string(),
            ),
        ];
        for (params, body) in cases {
            let (mut ctx, module) = frame_module(&params, &body);
            assert!(
                lower_continuation_frames(&mut ctx, module).is_err(),
                "{body}"
            );
        }
    }

    #[test]
    fn a_perform_without_an_exact_resumption_is_rejected() {
        let (mut ctx, module) = frame_module(
            ", %f: !frame, %k: core.i32",
            r#"    ability.perform %ev, %f, %k {ability_ref = !state, op_name = "get"}"#,
        );
        assert!(lower_continuation_frames(&mut ctx, module).is_err());
    }

    #[test]
    fn a_handle_with_fewer_handlers_than_arms_is_rejected() {
        let (mut ctx, module) = frame_module(
            ", %exit: !frame, %done: !resume",
            r#"    ability.handle %ev, %exit, %done, %done {handlers = []} {
      ^body(%inner: !ev, %frame: !frame):
        func.unreachable
    }"#,
        );
        assert!(lower_continuation_frames(&mut ctx, module).is_err());
    }
}
