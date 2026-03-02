//! Native lowering passes for Tribute.
//!
//! This module contains Tribute-specific passes that lower high-level Tribute IR
//! to native (Cranelift) dialect operations.
//!
//! ## Passes
//!
//! - `entrypoint`: Generate C ABI `main` wrapper for native binaries
//! - `type_converter`: Native type converter for IR-level type transformations
//! - `adt_rc_header`: Lower `adt.struct_new` to clif alloc + RC header init + field stores
//! - `tribute_rt_to_clif`: Lower `tribute_rt.box_*`/`unbox_*` to clif alloc + load/store
//! - `rc_insertion`: Insert `tribute_rt.retain`/`release` for reference counting
//! - `rc_lowering`: Lower `tribute_rt.retain`/`release` to inline `clif.*` ops

pub mod adt_rc_header;
pub mod cont_rc;
pub mod entrypoint;
pub mod evidence;
pub mod rc_insertion;
pub mod rc_lowering;
pub mod rtti;
pub mod tribute_rt_to_clif;
pub mod type_converter;

use std::collections::BTreeMap;
use trunk_ir::Symbol;
use trunk_ir::arena::context::{BlockArgData, BlockData, IrContext, RegionData};
use trunk_ir::arena::dialect::func as arena_func;
use trunk_ir::arena::refs::{OpRef, TypeRef};
use trunk_ir::arena::types::{
    Attribute as ArenaAttribute, Location as ArenaLocation, TypeDataBuilder,
};
use trunk_ir::smallvec::smallvec;

/// Build an extern `func.func` with an unreachable body and `abi = "C"`.
///
/// This is the common pattern for declaring external runtime functions
/// that are linked at native code generation time.
pub(crate) fn build_extern_func(
    ctx: &mut IrContext,
    loc: ArenaLocation,
    name: &str,
    params: &[TypeRef],
    result: TypeRef,
) -> OpRef {
    let func_ty = ctx.types.intern({
        let mut builder = TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func"));
        builder = builder.param(result);
        for &p in params {
            builder = builder.param(p);
        }
        builder.build()
    });

    let args: Vec<BlockArgData> = params
        .iter()
        .map(|&ty| BlockArgData {
            ty,
            attrs: BTreeMap::new(),
        })
        .collect();

    let unreachable_op = arena_func::unreachable(ctx, loc);
    let entry = ctx.create_block(BlockData {
        location: loc,
        args,
        ops: smallvec![],
        parent_region: None,
    });
    ctx.push_op(entry, unreachable_op.op_ref());

    let body = ctx.create_region(RegionData {
        location: loc,
        blocks: smallvec![entry],
        parent_op: None,
    });

    let func_op = arena_func::func(ctx, loc, Symbol::from_dynamic(name), func_ty, body);
    ctx.op_mut(func_op.op_ref())
        .attributes
        .insert(Symbol::new("abi"), ArenaAttribute::String("C".to_string()));

    func_op.op_ref()
}
