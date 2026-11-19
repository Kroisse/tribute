//! Builders that `tribute_control_to_cps` and `lower_continuation_frames`
//! share.

use tribute_ir::calling_convention::cps_closure_function_type;
use tribute_ir::calling_convention::{CallingConvention, ConventionBearing};
use tribute_ir::dialect::{closure, tribute_control};
use trunk_ir::Symbol;
use trunk_ir::context::{BlockArgData, BlockData, IrContext, RegionData};
use trunk_ir::dialect::{core, func};
use trunk_ir::ops::DialectType;
use trunk_ir::refs::{BlockRef, OpRef, RegionRef, TypeRef, ValueRef};
use trunk_ir::region_utils::values_defined_above;
use trunk_ir::types::{Attribute, AttributeMap, Location};

use crate::tribute_control_to_cps::TributeControlToCpsError;

/// The entry block of a closure's entry function.
pub(crate) struct ClosureEntryBlock {
    pub(crate) block: BlockRef,
    /// The entry's formals without the environment, in their own order.
    pub(crate) formals: Vec<ValueRef>,
    /// The captures the entry reads from its environment.
    pub(crate) captures: Vec<ValueRef>,
}

impl ClosureEntryBlock {
    /// Make the entry block of a `convention` entry function of a closure
    /// of type `closure_ty`: `formals` with the environment, the closure
    /// itself marked `bind_name = "__env"`, interposed at the convention's
    /// environment index, and a `closure.unpack` of `capture_types` when
    /// there are captures (`design/cps-effects.md`, "CPS 변환의 호출 대상
    /// 형상").
    pub(crate) fn new(
        ctx: &mut IrContext,
        location: Location,
        convention: CallingConvention,
        closure_ty: TypeRef,
        formals: Vec<BlockArgData>,
        capture_types: &[TypeRef],
    ) -> Self {
        let environment_index = convention.closure_environment_index();
        let mut environment_attrs = AttributeMap::new();
        let name = ctx.string_attr("__env");
        environment_attrs.view_mut(ctx).insert("bind_name", name);
        let mut args = formals;
        args.insert(
            environment_index,
            BlockArgData {
                ty: closure_ty,
                attrs: environment_attrs,
            },
        );
        let block = ctx.create_block(BlockData {
            location,
            args,
            ops: Default::default(),
            parent_region: None,
        });
        let mut formals = ctx.block_args(block).to_vec();
        let environment = formals.remove(environment_index);
        let captures = if capture_types.is_empty() {
            Vec::new()
        } else {
            let unpack = closure::Unpack::operands(environment)
                .results(capture_types.to_vec())
                .build(ctx, location);
            ctx.push_op(block, unpack.op_ref());
            unpack.results(ctx).to_vec()
        };
        Self {
            block,
            formals,
            captures,
        }
    }
}

/// The type of a `convention` entry function of a closure of type
/// `closure_ty` whose entry signature without the environment is
/// `signature`.
pub(crate) fn closure_entry_type(
    ctx: &mut IrContext,
    signature: func::FuncSig,
    convention: CallingConvention,
    closure_ty: TypeRef,
) -> TypeRef {
    signature
        .rebuild(ctx, |inputs, _| {
            inputs.insert(
                convention.closure_environment_index(),
                (closure_ty, AttributeMap::new()),
            );
        })
        .as_type_ref()
}

/// Put a selection on the operation that passes the evidence it selects.
pub(crate) fn set_evidence_plan(ctx: &mut IrContext, target: OpRef, plan: Option<Attribute>) {
    if let Some(plan) = plan {
        ctx.op_attrs_mut(target)
            .insert(tribute_control::EVIDENCE_PLAN_ATTR, plan);
    }
}

pub(crate) fn make_block(ctx: &mut IrContext, location: Location, types: &[TypeRef]) -> BlockRef {
    ctx.create_block(BlockData {
        location,
        args: types
            .iter()
            .copied()
            .map(|ty| BlockArgData {
                ty,
                attrs: AttributeMap::new(),
            })
            .collect(),
        ops: Default::default(),
        parent_region: None,
    })
}

pub(crate) fn single_block_region(
    ctx: &mut IrContext,
    location: Location,
    block: BlockRef,
) -> RegionRef {
    ctx.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![block],
        parent_op: None,
    })
}

/// The name of the `index`th helper function of kind `prefix`.
pub(crate) fn helper_symbol(prefix: &str, index: u32) -> Symbol {
    Symbol::new(&format!("__tribute_{prefix}_{index}"))
}

/// Build the closure of `region`, capturing the values it uses from outside
/// in their order of first use.
pub(crate) fn closure_over(
    ctx: &mut IrContext,
    location: Location,
    region: RegionRef,
    closure_type: TypeRef,
    convention: CallingConvention,
) -> closure::Lambda {
    let signature = closure::Closure::from_type_ref(ctx, closure_type)
        .expect("a generated continuation is a closure")
        .func_type(ctx);
    closure::Lambda::operands(values_defined_above(ctx, &[region]))
        .signatures([signature])
        .conventions([convention])
        .results(closure_type)
        .regions([region])
        .build(ctx, location)
}

/// Emit a final CPS tail transfer into a generated continuation, whose
/// closure type's callable is its `Cps` entry. This must not reconstruct a
/// signature from physical operands: closure extraction interposes an
/// environment later and preserves this contract on the resulting indirect
/// transfer.
pub(crate) fn emit_cps_tail_call_indirect(
    ctx: &mut IrContext,
    block: BlockRef,
    location: Location,
    callee: ValueRef,
    args: impl IntoIterator<Item = ValueRef>,
) -> Result<OpRef, TributeControlToCpsError> {
    let closure_type = ctx.value_ty(callee);
    let signature = cps_closure_function_type(ctx, closure_type).ok_or_else(|| {
        TributeControlToCpsError::post_at(
            location,
            "CPS indirect tail callee has no exact closure contract",
        )
    })?;
    emit_cps_entry_transfer(ctx, block, location, callee, signature, args)
}

/// Emit a final CPS tail transfer into the `Cps` entry `signature` of the
/// closure `callee`, the environment not yet interposed.
pub(crate) fn emit_cps_entry_transfer(
    ctx: &mut IrContext,
    block: BlockRef,
    location: Location,
    callee: ValueRef,
    signature: TypeRef,
    args: impl IntoIterator<Item = ValueRef>,
) -> Result<OpRef, TributeControlToCpsError> {
    let args = args.into_iter().collect::<Vec<_>>();
    let callable = func::FuncSig::from_type_ref(ctx, signature).ok_or_else(|| {
        TributeControlToCpsError::post_at(
            location,
            "CPS indirect tail callee contract is not func.func_sig",
        )
    })?;
    let never = core::Never::params().intern(ctx).as_type_ref();
    if callable.results(ctx) != [never]
        || callable.inputs(ctx).len() != args.len()
        || callable
            .inputs(ctx)
            .iter()
            .zip(&args)
            .any(|(expected, actual)| *expected != ctx.value_ty(*actual))
    {
        return Err(TributeControlToCpsError::post_at(
            location,
            "CPS indirect tail operands differ from the exact closure contract",
        ));
    }
    let tail = func::TailCallIndirect::operands(callee, args)
        .signature(signature)
        .build(ctx, location);
    tail.set_calling_convention(ctx, CallingConvention::Cps);
    ctx.push_op(block, tail.op_ref());
    Ok(tail.op_ref())
}
