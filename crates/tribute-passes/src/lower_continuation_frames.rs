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
use tribute_core::calling_convention::{cps_dispatch_type, cps_done_type};
use tribute_ir::continuation_frame;
use tribute_ir::dialect::{ability, tribute_rt};
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
use trunk_ir::types::TypeDataBuilder;
use trunk_ir::walk::{WalkAction, walk_op};

use crate::closure_lower::{TypeSubstitution, substitute_module_types_keeping_casts};
use crate::cps_builders::helper_symbol;
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

/// Replace the abstract frame surface of `module` with nominal frame layouts
/// and the operations that build and read them.
pub fn lower_continuation_frames(
    ctx: &mut IrContext,
    module: Module,
) -> Result<(), TributeControlToCpsError> {
    let location = ctx.op(module.op()).location;
    let module_block = module
        .body(ctx)
        .and_then(|body| ctx.region(body).blocks.first().copied())
        .ok_or_else(|| TributeControlToCpsError::post_at(location, "module has no body block"))?;

    let layouts = frame_layouts(ctx, module);
    let replacements: HashMap<_, _> = layouts
        .iter()
        .map(|(frame, types)| (*frame, types.reference))
        .collect();
    substitute_module_types_keeping_casts(ctx, module, |_, ty| replacements.get(&ty).copied());

    PatternApplicator::new(TypeConverter::new())
        .add_pattern(ExpandFrameOperations {
            frames: layouts
                .into_iter()
                .map(|(_, types)| (types.answer, types))
                .collect(),
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

/// The types of the layout of one `ability.frame<R>`.
#[derive(Clone, Copy)]
struct FrameTypes {
    /// The answer type `R` after the frames it mentions became references.
    answer: TypeRef,
    reference: TypeRef,
    layout: TypeRef,
    done: TypeRef,
    dispatch: TypeRef,
}

/// Choose and register the layout of each `ability.frame<R>` of the module,
/// keyed by that abstract type. Layouts are numbered in the order a walk of
/// the module first meets their frames.
///
/// A layout holds the answer type after its own frames are converted, so an
/// answer that mentions frames resolves them first.
fn frame_layouts(ctx: &mut IrContext, module: Module) -> Vec<(TypeRef, FrameTypes)> {
    let answers = frame_answers(ctx, module);
    let names: HashMap<TypeRef, String> = answers
        .iter()
        .enumerate()
        .map(|(index, answer)| {
            let name = format!("{}{index}", continuation_frame::NAME_PREFIX);
            (*answer, name)
        })
        .collect();
    let evidence = ability::evidence_adt_type_ref(ctx);
    let anyref = tribute_rt::anyref(ctx).as_type_ref();
    let i32_type = ctx.intern_type(TypeDataBuilder::new("core", "i32").build());
    let mut resolved = HashMap::default();
    let mut layouts = Vec::with_capacity(answers.len());
    for source_answer in answers {
        let reference = resolve_frame(ctx, source_answer, &names, &mut resolved)
            .expect("every collected answer has a name");
        let answer = continuation_frame::result_type(ctx, reference)
            .expect("a frame reference records its answer");
        let name = &names[&source_answer];
        let done = cps_done_type(ctx, answer);
        let dispatch = cps_dispatch_type(ctx, evidence, reference, anyref, i32_type);
        let name_ref = ctx.intern_str(name);
        let layout = continuation_frame::layout_type(ctx, name_ref, answer, done, dispatch);
        ctx.register_type_alias(Symbol::new(name), layout);
        let types = FrameTypes {
            answer,
            reference,
            layout,
            done,
            dispatch,
        };
        layouts.push((ability::frame(ctx, source_answer).as_type_ref(), types));
    }
    layouts
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

/// The answer types of the module's `ability.frame<R>` types, in the order a
/// walk of the module first meets them: each operation's results, attributes,
/// and block arguments, then the type aliases.
fn frame_answers(ctx: &IrContext, module: Module) -> Vec<TypeRef> {
    let mut seen = HashSet::default();
    let mut answers = Vec::new();
    for_each_module_type(ctx, module, &mut |ty| {
        collect_frame_answers(ctx, ty, &mut seen, &mut answers);
    });
    answers
}

fn collect_frame_answers(
    ctx: &IrContext,
    ty: TypeRef,
    seen: &mut HashSet<TypeRef>,
    answers: &mut Vec<TypeRef>,
) {
    if !seen.insert(ty) {
        return;
    }
    if let Some(frame) = ability::Frame::from_type_ref(ctx, ty) {
        answers.push(frame.result(ctx));
    }
    let data = ctx.get_type(ty);
    for param in data.params.iter() {
        collect_frame_answers(ctx, *param, seen, answers);
    }
    for (_, value) in data.attrs.iter() {
        value.visit_types(&mut |inner| collect_frame_answers(ctx, inner, seen, answers));
    }
}

/// Visit every type on a surface the type substitution rewrites: operation
/// results and attributes, block arguments with their attributes, and type
/// aliases.
fn for_each_module_type(ctx: &IrContext, module: Module, visit: &mut impl FnMut(TypeRef)) {
    let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
        for ty in ctx.op_result_types(op) {
            visit(*ty);
        }
        for (_, value) in ctx.op(op).attributes.iter() {
            value.visit_types(visit);
        }
        for region in ctx.op_regions(op) {
            for block in ctx.region(region).blocks.iter() {
                for arg in &ctx.block(*block).args {
                    visit(arg.ty);
                    for (_, value) in arg.attrs.iter() {
                        value.visit_types(visit);
                    }
                }
            }
        }
        ControlFlow::Continue(WalkAction::Advance)
    });
    for (_, ty) in ctx.type_aliases() {
        visit(*ty);
    }
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
    use tribute_ir::dialect::effect;
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

    fn frame_of_answer(ctx: &mut IrContext, answer: &str) -> TypeRef {
        let answer = ctx.intern_type(TypeDataBuilder::new("core", answer).build());
        ability::frame(ctx, answer).as_type_ref()
    }

    fn layout_names(ctx: &IrContext) -> Vec<String> {
        ctx.type_aliases()
            .iter()
            .map(|(name, _)| name.to_string())
            .filter(|name| name.starts_with(continuation_frame::NAME_PREFIX))
            .collect()
    }

    #[test]
    fn a_module_without_frames_lowers() {
        let (mut ctx, module) = parse(MODULE);
        assert!(lower_continuation_frames(&mut ctx, module).is_ok());
        assert!(layout_names(&ctx).is_empty());
    }

    #[test]
    fn a_frame_in_a_type_alias_gets_a_layout() {
        let (mut ctx, module) = parse(MODULE);
        let frame = frame_of_answer(&mut ctx, "i32");
        ctx.register_type_alias("orphan".into(), frame);
        lower_continuation_frames(&mut ctx, module).unwrap();
        let orphan = ctx.type_alias_by_text("orphan").unwrap();
        assert!(continuation_frame::result_type(&ctx, orphan).is_some());
        assert_eq!(layout_names(&ctx).len(), 1);
    }

    #[test]
    fn a_frame_in_a_block_argument_attribute_gets_a_layout() {
        let (mut ctx, module) = parse(MODULE);
        let frame = frame_of_answer(&mut ctx, "i32");
        let function = module.ops(&ctx)[0];
        let body = ctx.op_region(function, 0).expect("function body");
        let block = ctx.region(body).blocks[0];
        ctx.block_mut(block).args[0]
            .attrs
            .insert("frame", Attribute::Type(frame));
        lower_continuation_frames(&mut ctx, module).unwrap();
        let converted = ctx.block(block).args[0].attrs.get_type("frame").unwrap();
        assert!(continuation_frame::result_type(&ctx, converted).is_some());
    }

    #[test]
    fn layouts_are_numbered_in_module_walk_order() {
        // A `TypeRef` is an interner index, which changes with the types
        // interned before the pass and must not decide the numbering.
        let answers = |unrelated_types: usize, first: &str, second: &str| {
            let mut ctx = IrContext::new();
            for index in 0..unrelated_types {
                let name = ctx.string_attr(&format!("unrelated{index}"));
                ctx.intern_type(
                    TypeDataBuilder::new("test", "unrelated")
                        .attr("name", name)
                        .build(),
                );
            }
            frame_of_answer(&mut ctx, second);
            let module = parse_test_module(
                &mut ctx,
                &format!(
                    "core.module @m {{
  func.func @f(%a: ability.frame<core.{first}>, %b: ability.frame<core.{second}>) -> core.never {{
    func.unreachable
  }}
}}"
                ),
            );
            lower_continuation_frames(&mut ctx, module).unwrap();
            [0, 1].map(|index| {
                let name = format!("{}{index}", continuation_frame::NAME_PREFIX);
                let layout = ctx
                    .type_alias_by_text(&name)
                    .expect("the layout is registered");
                let answer = continuation_frame::result_type(&ctx, layout).unwrap();
                ctx.get_type(answer).name.to_string()
            })
        };
        assert_eq!(answers(0, "i32", "i64"), ["i32", "i64"]);
        assert_eq!(answers(17, "i32", "i64"), ["i32", "i64"]);
        assert_eq!(answers(0, "i64", "i32"), ["i64", "i32"]);
    }

    const TYPES: &str = r#"  !marker = adt.struct<_Marker(ability_id: core.i32, prompt_tag: core.i32, tr_dispatch_fn: core.ptr, shadowed: core.ptr, outer: core.ptr), {layout = "evidence_marker"}>
  !ev = core.array<!marker, {layout = "evidence"}>
  !state = core.ability_ref<{name = "State"}>
  !frame = ability.frame<core.i32>
  !resume = closure.closure<func.func_sig<(!ev, !frame, core.i32) -> core.never>, {tribute.calling_convention = 2, tribute.closure_environment_index = 0}>
  !nested = ability.frame<!resume>"#;

    /// Parse a module over `TYPES` whose `@run` has `params` and `body`.
    fn frame_module(params: &str, body: &str) -> (IrContext, Module) {
        parse(&format!(
            "core.module @m {{\n{TYPES}\n  func.func @run(%ev: !ev{params}) -> core.never {{\n{body}\n  }}\n}}"
        ))
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
        // Both frames get a layout, and `lowered` checked that the outer one
        // no longer mentions the inner frame as an abstract type.
        for index in 0..2 {
            let name = format!("!{}{index} = ", continuation_frame::NAME_PREFIX);
            assert!(printed.contains(&name), "{printed}");
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
