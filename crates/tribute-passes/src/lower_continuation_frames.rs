//! Expansion of the abstract continuation frame surface that
//! `tribute_control_to_cps` emits.
//!
//! `ability.frame<R>` becomes the nominal frame reference of `R`, and
//! `ability.suffix_frame` and `ability.exit` expand into the done adapter,
//! dispatch adapter factory, and frame struct, or the transfer to the frame's
//! `Done<R>`.

use std::cell::RefCell;
use std::ops::ControlFlow;
use std::rc::Rc;

use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;
use tribute_ir::continuation_frame;
use tribute_ir::dialect::{ability, adt};
use trunk_ir::analysis::AnalysisCache;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::{core, func};
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::pass::{Pass, PassRunResult};
use trunk_ir::refs::{BlockRef, OpRef, TypeRef};
use trunk_ir::rewrite::{
    Module, PatternApplicator, PatternRewriter, RewritePattern, TypeConverter, erase_op,
};
use trunk_ir::walk::{WalkAction, walk_op};

use crate::closure_lower::substitute_module_types_keeping_casts;
use crate::tribute_control_to_cps::{FrameExpander, FrameTypes, TributeControlToCpsError};

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

    select_frame_layouts(ctx, module);
    let expander = Rc::new(RefCell::new(FrameExpander::new(
        frame_types(ctx, location)?,
        next_helper_index(ctx, module_block),
    )));
    let failure = Rc::new(RefCell::new(None));
    PatternApplicator::new(TypeConverter::new())
        .add_pattern(ExpandSuffixFrame {
            expander: expander.clone(),
            module_block,
            failure: failure.clone(),
        })
        .add_pattern(ExpandExit {
            expander,
            module_block,
            failure: failure.clone(),
        })
        .apply_partial(ctx, module);
    if let Some(error) = failure.take() {
        return Err(error);
    }
    reject_abstract_frames(ctx, module)
}

/// Replaces `ability.suffix_frame` with the frame it builds.
struct ExpandSuffixFrame {
    expander: Rc<RefCell<FrameExpander>>,
    module_block: BlockRef,
    failure: Rc<RefCell<Option<TributeControlToCpsError>>>,
}

impl RewritePattern for ExpandSuffixFrame {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(suffix) = ability::SuffixFrame::from_op(ctx, op) else {
            return false;
        };
        match self
            .expander
            .borrow_mut()
            .suffix_frame(ctx, self.module_block, suffix)
        {
            Ok(expansion) => {
                for built in expansion.body {
                    rewriter.insert_op(built);
                }
                for helper in expansion.helpers {
                    rewriter.add_module_op(helper);
                }
                rewriter.erase_op(expansion.frame.into_iter().collect());
                true
            }
            Err(error) => {
                self.failure.borrow_mut().get_or_insert(error);
                false
            }
        }
    }
}

/// Replaces `ability.exit` with the transfer to the frame's `Done<R>`.
struct ExpandExit {
    expander: Rc<RefCell<FrameExpander>>,
    module_block: BlockRef,
    failure: Rc<RefCell<Option<TributeControlToCpsError>>>,
}

impl RewritePattern for ExpandExit {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(exit) = ability::Exit::from_op(ctx, op) else {
            return false;
        };
        match self
            .expander
            .borrow_mut()
            .exit(ctx, self.module_block, exit)
        {
            Ok(mut expansion) => {
                let transfer = expansion.body.pop().expect("an exit ends in a transfer");
                rewriter.replace_with_prefix(expansion.body, transfer);
                for helper in expansion.helpers {
                    rewriter.add_module_op(helper);
                }
                true
            }
            Err(error) => {
                self.failure.borrow_mut().get_or_insert(error);
                false
            }
        }
    }
}

/// Replace `ability.frame<R>` with the nominal frame reference of `R`.
///
/// A frame whose answer type mentions another frame waits for the inner one,
/// since the layout is named by the substituted answer type. Casts that were
/// identities before the substitution stay.
fn select_frame_layouts(ctx: &mut IrContext, module: Module) {
    let identities = identity_casts(ctx, module);
    loop {
        let layouts = frame_layouts(ctx);
        let mut replaced = false;
        substitute_module_types_keeping_casts(ctx, module, |ctx, ty| {
            let answer = ability::Frame::from_type_ref(ctx, ty)?.result(ctx);
            if type_mentions_frame(ctx, answer, &mut HashSet::default()) {
                return None;
            }
            let name = layouts.get(&answer)?;
            replaced = true;
            Some(continuation_frame::ref_type(ctx, name.clone(), answer))
        });
        if !replaced {
            break;
        }
    }
    let mut casts = Vec::new();
    let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
        if !identities.contains(&op)
            && let Ok(cast) = core::UnrealizedConversionCast::from_op(ctx, op)
            && ctx.value_ty(cast.value(ctx)) == ctx.value_ty(cast.result(ctx))
        {
            casts.push(cast);
        }
        ControlFlow::Continue(WalkAction::Advance)
    });
    for cast in casts {
        let (input, result) = (cast.value(ctx), cast.result(ctx));
        ctx.replace_all_uses(result, input);
        erase_op(ctx, cast.op_ref());
    }
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
    location: trunk_ir::types::Location,
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
                abstract_frame: reference,
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

fn identity_casts(ctx: &IrContext, module: Module) -> HashSet<OpRef> {
    let mut casts = HashSet::default();
    let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
        if let Ok(cast) = core::UnrealizedConversionCast::from_op(ctx, op)
            && ctx.value_ty(cast.value(ctx)) == ctx.value_ty(cast.result(ctx))
        {
            casts.insert(op);
        }
        ControlFlow::Continue(WalkAction::Advance)
    });
    casts
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

/// Fail if an abstract frame operation or type survived the expansion.
fn reject_abstract_frames(ctx: &IrContext, module: Module) -> Result<(), TributeControlToCpsError> {
    let mut failure = None;
    let mut seen = HashSet::default();
    let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
        let mut mentions = ctx
            .op_result_types(op)
            .iter()
            .any(|ty| type_mentions_frame(ctx, *ty, &mut seen));
        for (_, value) in ctx.op(op).attributes.iter() {
            value.visit_types(&mut |ty| {
                mentions = mentions || type_mentions_frame(ctx, ty, &mut seen);
            });
        }
        for region in ctx.op_regions(op) {
            for block in ctx.region(region).blocks.iter() {
                mentions = mentions
                    || ctx
                        .block_args(*block)
                        .iter()
                        .any(|arg| type_mentions_frame(ctx, ctx.value_ty(*arg), &mut seen));
            }
        }
        let operation = ability::SuffixFrame::matches(ctx, op) || ability::Exit::matches(ctx, op);
        if (operation || mentions) && failure.is_none() {
            failure = Some(TributeControlToCpsError::post_at(
                ctx.op(op).location,
                "an abstract continuation frame survived lower_continuation_frames",
            ));
        }
        ControlFlow::Continue(WalkAction::Advance)
    });
    failure.map_or(Ok(()), Err)
}
