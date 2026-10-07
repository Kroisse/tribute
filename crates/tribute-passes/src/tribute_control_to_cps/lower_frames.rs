//! Expansion of the abstract continuation frame surface into the nominal
//! frame layouts, done adapters, and dispatch adapters.

use super::*;
use crate::closure_lower::substitute_module_types_keeping_casts;
use trunk_ir::walk::{WalkAction, walk_op};

/// Replace the abstract frame surface of `module` with the frame layouts
/// `tribute_control_to_cps` registered.
pub(crate) fn lower_continuation_frames(
    ctx: &mut IrContext,
    module: Module,
) -> Result<(), TributeControlToCpsError> {
    let module_location = ctx.op(module.op()).location;
    let module_block = module
        .body(ctx)
        .and_then(|body| ctx.region(body).blocks.first().copied())
        .ok_or_else(|| {
            TributeControlToCpsError::post_at(module_location, "module has no body block")
        })?;
    let identity_casts = identity_casts(ctx, module);

    loop {
        let layouts = frame_layouts(ctx);
        let mut replaced = false;
        substitute_module_types_keeping_casts(ctx, module, |ctx, ty| {
            let frame = ability::Frame::from_type_ref(ctx, ty)?;
            let answer = frame.result(ctx);
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
    erase_new_identity_casts(ctx, module, &identity_casts);

    let mut converter = Converter::new(ctx, module_block, HashMap::default());
    for (answer, name) in frame_layouts(converter.ctx) {
        let layout = converter
            .ctx
            .type_alias_by_text(&name)
            .expect("frame layout alias was listed");
        let fields = adt::Struct::from_type_ref(converter.ctx, layout)
            .map(|layout| layout.fields(converter.ctx).collect::<Vec<_>>())
            .unwrap_or_default();
        let [("done", done), ("dispatch", dispatch)] = fields[..] else {
            return Err(TributeControlToCpsError::post_at(
                module_location,
                format!("continuation frame layout {name} must hold done and dispatch"),
            ));
        };
        let reference = continuation_frame::ref_type(converter.ctx, name.clone(), answer);
        converter.frames.insert(
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
    converter.helper_index = next_helper_index(converter.ctx, module_block);

    let mut ops = Vec::new();
    let _ = walk_op::<()>(converter.ctx, module.op(), &mut |op| {
        ops.push(op);
        ControlFlow::Continue(WalkAction::Advance)
    });
    for op in ops {
        let suffix = ability::SuffixFrame::from_op(converter.ctx, op).ok();
        let exit = ability::Exit::from_op(converter.ctx, op).ok();
        if suffix.is_none() && exit.is_none() {
            continue;
        }
        let location = converter.ctx.op(op).location;
        let block = converter
            .ctx
            .op(op)
            .parent_block
            .expect("frame operation is in a block");
        let scratch = converter.make_block(location, &[]);
        let top = top_level_ancestor(converter.ctx, module_block, op);
        let known = converter.ctx.block(module_block).ops.len();
        let replacement = match (suffix, exit) {
            (Some(suffix), _) => Some(converter.expand_suffix_frame(scratch, suffix)?),
            (_, Some(exit)) => {
                converter.expand_exit(scratch, exit)?;
                None
            }
            _ => unreachable!("one frame operation matched"),
        };
        let helpers = converter.ctx.block(module_block).ops[known..].to_vec();
        for helper in helpers {
            converter.ctx.remove_op_from_block(module_block, helper);
            converter.ctx.insert_op_before(module_block, top, helper);
        }
        for expanded in converter.ctx.block(scratch).ops.clone() {
            converter.ctx.remove_op_from_block(scratch, expanded);
            converter.ctx.insert_op_before(block, op, expanded);
        }
        if let Some(frame) = replacement {
            let old = converter.ctx.op_result(op, 0);
            converter.ctx.replace_all_uses(old, frame);
        }
        trunk_ir::rewrite::erase_op(converter.ctx, op);
    }
    reject_abstract_frames(ctx, module)
}

/// The module's frame layouts: result type to layout name.
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

/// Remove the casts the frame substitution turned into identities, keeping
/// those that were identities before it.
fn erase_new_identity_casts(ctx: &mut IrContext, module: Module, kept: &HashSet<OpRef>) {
    let mut casts = Vec::new();
    let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
        if !kept.contains(&op)
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
        trunk_ir::rewrite::erase_op(ctx, cast.op_ref());
    }
}

/// The operation of `module_block` that contains `op`.
fn top_level_ancestor(ctx: &IrContext, module_block: BlockRef, op: OpRef) -> OpRef {
    let mut current = op;
    loop {
        let block = ctx
            .op(current)
            .parent_block
            .expect("operation is inside the module");
        if block == module_block {
            return current;
        }
        let region = ctx
            .block(block)
            .parent_region
            .expect("block is inside a region");
        current = ctx
            .region(region)
            .parent_op
            .expect("region belongs to an operation");
    }
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

fn reject_abstract_frames(ctx: &IrContext, module: Module) -> Result<(), TributeControlToCpsError> {
    let mut failure = None;
    let mut seen = HashSet::default();
    let mut remaining = |ctx: &IrContext, op: OpRef, what: &str| {
        failure.get_or_insert_with(|| {
            TributeControlToCpsError::post_at(
                ctx.op(op).location,
                format!("{what} survived lower_continuation_frames"),
            )
        });
    };
    let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
        if ability::SuffixFrame::matches(ctx, op) || ability::Exit::matches(ctx, op) {
            remaining(ctx, op, "an abstract frame operation");
        }
        let mut mentions = ctx
            .op_result_types(op)
            .iter()
            .any(|ty| type_mentions_frame(ctx, *ty, &mut seen));
        for (_, value) in ctx.op(op).attributes.iter() {
            value.visit_types(&mut |ty| {
                mentions = mentions || type_mentions_frame(ctx, ty, &mut seen)
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
        if mentions {
            remaining(ctx, op, "an ability.frame type");
        }
        ControlFlow::Continue(WalkAction::Advance)
    });
    failure.map_or(Ok(()), Err)
}
