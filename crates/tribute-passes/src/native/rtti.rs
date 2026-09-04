//! RTTI (Runtime Type Information) pass for the native backend.
//!
//! This pass consumes the validated typed ownership plan, assigns each planned
//! allocation type a unique `rtti_idx`, and generates per-type release
//! functions that recursively release typed managed-reference fields before
//! deallocating the aggregate itself.
//!
//! ## RTTI Index Layout
//!
//! | Index | Type | Release |
//! |-------|------|---------|
//! | 0 | Nil | shallow |
//! | 1 | Bool | fixed 12-byte release |
//! | 2 | Nat | fixed 12-byte release |
//! | 3 | Int | fixed 12-byte release |
//! | 4 | Float | fixed 16-byte release |
//! | 5 | Rune | shallow |
//! | 6 | Bytes | shallow |
//! | 7 | Array | generic (future) |
//! | 8-31 | reserved | — |
//! | 32+ | user structs | per-type deep release |
//!
//! ## Pipeline Position
//!
//! Runs before `adt_to_clif` (Phase 1.9) so that `adt_to_clif` can use the
//! `RttiMap` to store correct `rtti_idx` values in allocation headers.

use std::collections::HashMap;

use trunk_ir::Symbol;
use trunk_ir::TypeDataBuilder;
use trunk_ir::adt_layout::{
    compute_enum_layout, compute_struct_layout, get_enum_variants, get_struct_fields,
};
use trunk_ir::context::{BlockArgData, BlockData, IrContext, RegionData};
use trunk_ir::dialect::clif;
use trunk_ir::dialect::core;
use trunk_ir::location::Span;
use trunk_ir::ops::DialectOp;
use trunk_ir::rewrite::{Module, TypeConverter};
use trunk_ir::smallvec::smallvec;
use trunk_ir::types::Location;
use trunk_ir::{BlockRef, OpRef, RegionRef, TypeRef, ValueRef};

use tribute_ir::dialect::tribute_rt;

use super::ownership_plan::{ManagedFieldBitmap, RttiTypePlan};

/// Commonly used CLIF primitive types, pre-interned for convenience.
struct ClifTypes {
    ptr: TypeRef,
    nil: TypeRef,
    i64: TypeRef,
    i32: TypeRef,
    i8: TypeRef,
}

impl ClifTypes {
    fn intern(ctx: &mut IrContext) -> Self {
        let mk = |ctx: &mut IrContext, name: &'static str| {
            ctx.types
                .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new(name)).build())
        };
        Self {
            ptr: mk(ctx, "ptr"),
            nil: mk(ctx, "nil"),
            i64: mk(ctx, "i64"),
            i32: mk(ctx, "i32"),
            i8: mk(ctx, "i8"),
        }
    }
}

/// First index for user-defined struct types.
pub const RTTI_USER_START: u32 = 32;

/// Reserved RTTI indices for built-in types.
pub const RTTI_NIL: u32 = 0;
pub const RTTI_BOOL: u32 = 1;
pub const RTTI_NAT: u32 = 2;
pub const RTTI_INT: u32 = 3;
pub const RTTI_FLOAT: u32 = 4;

const PRIMITIVE_I32_ALLOC_SIZE: u64 = 12;
const PRIMITIVE_F64_ALLOC_SIZE: u64 = 16;

/// Name prefix for per-type release functions.
pub const RELEASE_FN_PREFIX: &str = "__tribute_release_";

/// Name of the runtime deallocation function.
const DEALLOC_FN: &str = "__tribute_dealloc";

/// Mapping from TypeRef to RTTI indices.
#[derive(Debug, Clone, Default)]
pub struct RttiMap {
    pub type_to_idx: HashMap<TypeRef, u32>,
    next_idx: u32,
}

impl RttiMap {
    pub fn new() -> Self {
        Self {
            type_to_idx: HashMap::new(),
            next_idx: RTTI_USER_START,
        }
    }

    pub fn get_or_insert(&mut self, ty: TypeRef) -> u32 {
        if let Some(&idx) = self.type_to_idx.get(&ty) {
            return idx;
        }
        let idx = self.next_idx;
        self.next_idx += 1;
        self.type_to_idx.insert(ty, idx);
        idx
    }

    pub fn get(&self, ty: &TypeRef) -> Option<u32> {
        self.type_to_idx.get(ty).copied()
    }
}

/// Consume the typed RTTI plan, assign indices, and generate release functions.
pub fn generate_rtti(
    ctx: &mut IrContext,
    module: Module,
    type_converter: &TypeConverter,
    rtti_types: &[RttiTypePlan],
) -> RttiMap {
    let mut rtti_map = RttiMap::new();
    let primitive_releases = primitive_release_entries(ctx, module);

    // The allocation order and managed-field classification were validated
    // while semantic types were intact. RTTI must not rediscover either from
    // converted pointer shape.
    for entry in rtti_types {
        rtti_map.get_or_insert(entry.ty);
    }

    if rtti_map.type_to_idx.is_empty() && primitive_releases.is_empty() {
        return rtti_map;
    }

    // Phase 2: Generate per-type release functions and append to module
    let Some(module_block) = module.first_block(ctx) else {
        return rtti_map;
    };

    let loc = Location::new(ctx.paths.intern("<rtti>".to_string()), Span::new(0, 0));

    // `anyref` and `intref` have no static nominal allocation layout. Their
    // release action carries a dynamic-size signal, resolved by the header
    // RTTI index before deallocation. Used primitive slots must therefore own
    // exact release functions instead of falling through with zero.
    for (rtti_idx, alloc_size) in primitive_releases {
        let func_op = generate_fixed_release_function(ctx, rtti_idx, alloc_size, loc);
        ctx.push_op(module_block, func_op);
    }

    // Sort by rtti_idx for deterministic output
    let mut entries: Vec<_> = rtti_map
        .type_to_idx
        .iter()
        .map(|(&ty, &idx)| (ty, idx))
        .collect();
    entries.sort_by_key(|(_, idx)| *idx);

    for (ty, rtti_idx) in entries {
        let field_plan = rtti_types
            .iter()
            .find(|entry| entry.ty == ty)
            .expect("RTTI map was built from ownership plan");
        let func_op = match &field_plan.fields {
            ManagedFieldBitmap::Enum(fields) => {
                generate_release_function_for_enum(ctx, ty, rtti_idx, type_converter, fields, loc)
            }
            ManagedFieldBitmap::Struct(fields) => {
                generate_release_function_for_struct(ctx, ty, rtti_idx, type_converter, fields, loc)
            }
        };
        ctx.push_op(module_block, func_op);
    }

    rtti_map
}

/// Find primitive boxing operations while their semantic operation identity is
/// still present. This selects only fixed reserved RTTI entries; it neither
/// discovers ownership nor follows physical pointer definitions.
fn primitive_release_entries(ctx: &IrContext, module: Module) -> Vec<(u32, u64)> {
    let mut used = [false; 4];
    if let Some(body) = module.body(ctx) {
        collect_primitive_boxes(ctx, body, &mut used);
    }

    [
        (RTTI_BOOL, PRIMITIVE_I32_ALLOC_SIZE, 0),
        (RTTI_NAT, PRIMITIVE_I32_ALLOC_SIZE, 1),
        (RTTI_INT, PRIMITIVE_I32_ALLOC_SIZE, 2),
        (RTTI_FLOAT, PRIMITIVE_F64_ALLOC_SIZE, 3),
    ]
    .into_iter()
    .filter_map(|(rtti_idx, alloc_size, used_index)| {
        used[used_index].then_some((rtti_idx, alloc_size))
    })
    .collect()
}

fn collect_primitive_boxes(ctx: &IrContext, region: RegionRef, used: &mut [bool; 4]) {
    for &block in &ctx.region(region).blocks {
        for &op in &ctx.block(block).ops {
            if tribute_rt::BoxBool::from_op(ctx, op).is_ok() {
                used[0] = true;
            } else if tribute_rt::BoxNat::from_op(ctx, op).is_ok() {
                used[1] = true;
            } else if tribute_rt::BoxInt::from_op(ctx, op).is_ok() {
                used[2] = true;
            } else if tribute_rt::BoxFloat::from_op(ctx, op).is_ok() {
                used[3] = true;
            }
            for &nested in &ctx.op(op).regions {
                collect_primitive_boxes(ctx, nested, used);
            }
        }
    }
}

/// Generate a reserved primitive release function with an exact total
/// allocation size, including the RC header.
fn generate_fixed_release_function(
    ctx: &mut IrContext,
    rtti_idx: u32,
    alloc_size: u64,
    loc: Location,
) -> OpRef {
    let tys = ClifTypes::intern(ctx);
    let func_ty = core::func(ctx, [tys.ptr], [tys.nil]).as_type_ref();
    let entry_block = ctx.create_block(BlockData {
        location: loc,
        args: vec![BlockArgData {
            ty: tys.ptr,
            attrs: Default::default(),
        }],
        ops: smallvec![],
        parent_region: None,
    });
    let payload_ptr = ctx.block_arg(entry_block, 0);
    gen_dealloc_and_return_with_size(
        ctx,
        loc,
        entry_block,
        payload_ptr,
        alloc_size,
        tys.ptr,
        tys.nil,
        tys.i64,
    );
    let body = ctx.create_region(RegionData {
        location: loc,
        blocks: smallvec![entry_block],
        parent_op: None,
    });
    clif::func(
        ctx,
        loc,
        Symbol::from_dynamic(&format!("{RELEASE_FN_PREFIX}{rtti_idx}")),
        func_ty,
        body,
    )
    .op_ref()
}

/// Generate release function for a struct type.
fn generate_release_function_for_struct(
    ctx: &mut IrContext,
    struct_ty: TypeRef,
    rtti_idx: u32,
    type_converter: &TypeConverter,
    managed_fields: &[bool],
    loc: Location,
) -> OpRef {
    let fields = get_struct_fields(ctx, struct_ty)
        .expect("struct type registered in RttiMap must have fields");
    let layout = compute_struct_layout(ctx, struct_ty, type_converter)
        .expect("struct type registered in RttiMap must have a valid layout");

    let tys = ClifTypes::intern(ctx);
    let ptr_ty = tys.ptr;
    let nil_ty = tys.nil;
    let i64_ty = tys.i64;
    let i8_ty = tys.i8;

    let func_name = format!("{}{}", RELEASE_FN_PREFIX, rtti_idx);

    // Function type: (core.ptr) -> core.nil
    let func_ty = core::func(ctx, [ptr_ty], [nil_ty]).as_type_ref();

    assert_eq!(fields.len(), managed_fields.len());
    let managed_field_offsets: Vec<i32> = fields
        .iter()
        .enumerate()
        .filter_map(|(i, _)| managed_fields[i].then_some(layout.field_offsets[i] as i32))
        .collect();

    // Build entry block with payload_ptr argument
    let entry_block = ctx.create_block(BlockData {
        location: loc,
        args: vec![BlockArgData {
            ty: ptr_ty,
            attrs: Default::default(),
        }],
        ops: smallvec![],
        parent_region: None,
    });
    let payload_ptr = ctx.block_arg(entry_block, 0);

    // Build dealloc block
    let dealloc_block = ctx.create_block(BlockData {
        location: loc,
        args: vec![],
        ops: smallvec![],
        parent_region: None,
    });
    gen_dealloc_and_return(
        ctx,
        loc,
        dealloc_block,
        payload_ptr,
        &layout,
        ptr_ty,
        nil_ty,
        i64_ty,
    );

    if managed_field_offsets.is_empty() {
        // No managed fields: entry block IS the dealloc block
        // Move dealloc ops to entry block
        let dealloc_ops: Vec<OpRef> = ctx.block(dealloc_block).ops.to_vec();
        for op in dealloc_ops {
            ctx.remove_op_from_block(dealloc_block, op);
            ctx.push_op(entry_block, op);
        }

        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry_block],
            parent_op: None,
        });

        let func_op = clif::func(ctx, loc, Symbol::from_dynamic(&func_name), func_ty, body);
        return func_op.op_ref();
    }

    // Build null-guarded field check/release blocks backwards
    let mut blocks_after_entry: Vec<BlockRef> = vec![dealloc_block];
    let mut next_block = dealloc_block;

    for &offset in managed_field_offsets.iter().rev() {
        // Release block: load field, release, jump to next
        let release_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let reload = clif::load(ctx, loc, payload_ptr, ptr_ty, offset);
        ctx.push_op(release_block, reload.op_ref());
        let release = tribute_rt::release(ctx, loc, reload.result(ctx), 0);
        ctx.push_op(release_block, release.op_ref());
        let jump = clif::jump(ctx, loc, [], next_block);
        ctx.push_op(release_block, jump.op_ref());

        // Check block: load field, null check, branch
        let check_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let load = clif::load(ctx, loc, payload_ptr, ptr_ty, offset);
        ctx.push_op(check_block, load.op_ref());
        let null_const = clif::iconst(ctx, loc, ptr_ty, 0);
        ctx.push_op(check_block, null_const.op_ref());
        let is_null = clif::icmp(
            ctx,
            loc,
            load.result(ctx),
            null_const.result(ctx),
            i8_ty,
            Symbol::new("eq"),
        );
        ctx.push_op(check_block, is_null.op_ref());
        let brif = clif::brif(ctx, loc, is_null.result(ctx), next_block, release_block);
        ctx.push_op(check_block, brif.op_ref());

        blocks_after_entry.push(release_block);
        blocks_after_entry.push(check_block);
        next_block = check_block;
    }

    // Entry block gets the ops of the first check block
    let first_check = blocks_after_entry.pop().unwrap();
    let first_check_ops: Vec<OpRef> = ctx.block(first_check).ops.to_vec();
    for op in first_check_ops {
        ctx.remove_op_from_block(first_check, op);
        ctx.push_op(entry_block, op);
    }

    blocks_after_entry.reverse();
    let mut all_blocks: Vec<BlockRef> = vec![entry_block];
    all_blocks.extend(blocks_after_entry);

    let body = ctx.create_region(RegionData {
        location: loc,
        blocks: all_blocks.into(),
        parent_op: None,
    });

    let func_op = clif::func(ctx, loc, Symbol::from_dynamic(&func_name), func_ty, body);
    func_op.op_ref()
}

/// Emit dealloc + return ops into a block.
#[allow(clippy::too_many_arguments)]
fn gen_dealloc_and_return(
    ctx: &mut IrContext,
    loc: Location,
    block: BlockRef,
    payload_ptr: ValueRef,
    layout: &trunk_ir::adt_layout::StructLayout,
    ptr_ty: TypeRef,
    nil_ty: TypeRef,
    i64_ty: TypeRef,
) {
    use tribute_ir::dialect::tribute_rt::RC_HEADER_SIZE;

    gen_dealloc_and_return_with_size(
        ctx,
        loc,
        block,
        payload_ptr,
        layout.total_size as u64 + RC_HEADER_SIZE,
        ptr_ty,
        nil_ty,
        i64_ty,
    );
}

#[allow(clippy::too_many_arguments)]
fn gen_dealloc_and_return_with_size(
    ctx: &mut IrContext,
    loc: Location,
    block: BlockRef,
    payload_ptr: ValueRef,
    alloc_size: u64,
    ptr_ty: TypeRef,
    nil_ty: TypeRef,
    i64_ty: TypeRef,
) {
    use tribute_ir::dialect::tribute_rt::RC_HEADER_SIZE;

    let hdr_sz = clif::iconst(ctx, loc, i64_ty, RC_HEADER_SIZE as i64);
    ctx.push_op(block, hdr_sz.op_ref());
    let raw_ptr = clif::isub(ctx, loc, payload_ptr, hdr_sz.result(ctx), ptr_ty);
    ctx.push_op(block, raw_ptr.op_ref());

    let size_op = clif::iconst(ctx, loc, i64_ty, alloc_size as i64);
    ctx.push_op(block, size_op.op_ref());

    let dealloc_call = clif::call(
        ctx,
        loc,
        [raw_ptr.result(ctx), size_op.result(ctx)],
        nil_ty,
        Symbol::new(DEALLOC_FN),
    );
    ctx.push_op(block, dealloc_call.op_ref());

    let ret_op = clif::r#return(ctx, loc, []);
    ctx.push_op(block, ret_op.op_ref());
}

/// Build an `adt.struct` type with named fields (for testing and internal use).
#[cfg(test)]
pub(crate) fn make_struct_type(ctx: &mut IrContext, fields: &[(&'static str, TypeRef)]) -> TypeRef {
    use trunk_ir::types::Attribute as A;
    let fields_list: Vec<A> = fields
        .iter()
        .map(|(name, ty)| A::List(vec![A::Symbol(Symbol::new(name)), A::Type(*ty)]))
        .collect();
    ctx.types.intern(
        TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("struct"))
            .attr(Symbol::new("fields"), A::List(fields_list))
            .build(),
    )
}

/// Generate release function for an enum type.
fn generate_release_function_for_enum(
    ctx: &mut IrContext,
    enum_ty: TypeRef,
    rtti_idx: u32,
    type_converter: &TypeConverter,
    managed_variants: &[Vec<bool>],
    loc: Location,
) -> OpRef {
    let layout = compute_enum_layout(ctx, enum_ty, type_converter)
        .expect("enum type registered in RttiMap must have a valid layout");
    let variants = get_enum_variants(ctx, enum_ty).unwrap_or_default();

    let tys = ClifTypes::intern(ctx);
    let ptr_ty = tys.ptr;
    let nil_ty = tys.nil;
    let i64_ty = tys.i64;
    let i32_ty = tys.i32;
    let i8_ty = tys.i8;

    let func_name = format!("{}{}", RELEASE_FN_PREFIX, rtti_idx);
    let func_ty = core::func(ctx, [ptr_ty], [nil_ty]).as_type_ref();

    let entry_block = ctx.create_block(BlockData {
        location: loc,
        args: vec![BlockArgData {
            ty: ptr_ty,
            attrs: Default::default(),
        }],
        ops: smallvec![],
        parent_region: None,
    });
    let payload_ptr = ctx.block_arg(entry_block, 0);

    // Collect variants with managed fields.
    struct VariantRelease {
        tag_value: u32,
        managed_field_offsets: Vec<i32>,
    }
    let mut variants_with_ptrs: Vec<VariantRelease> = Vec::new();

    assert_eq!(variants.len(), managed_variants.len());
    for (variant_idx, (_variant_name, field_types)) in variants.iter().enumerate() {
        let variant_layout = &layout.variant_layouts[variant_idx];
        assert_eq!(field_types.len(), managed_variants[variant_idx].len());
        let managed_field_offsets: Vec<i32> = field_types
            .iter()
            .enumerate()
            .filter_map(|(field_idx, _)| {
                managed_variants[variant_idx][field_idx].then_some(
                    (layout.fields_offset + variant_layout.field_offsets[field_idx]) as i32,
                )
            })
            .collect();

        if !managed_field_offsets.is_empty() {
            variants_with_ptrs.push(VariantRelease {
                tag_value: variant_layout.tag_value,
                managed_field_offsets,
            });
        }
    }

    // Build dealloc block
    let dealloc_block = ctx.create_block(BlockData {
        location: loc,
        args: vec![],
        ops: smallvec![],
        parent_region: None,
    });
    {
        use tribute_ir::dialect::tribute_rt::RC_HEADER_SIZE;

        let hdr_sz = clif::iconst(ctx, loc, i64_ty, RC_HEADER_SIZE as i64);
        ctx.push_op(dealloc_block, hdr_sz.op_ref());
        let raw_ptr = clif::isub(ctx, loc, payload_ptr, hdr_sz.result(ctx), ptr_ty);
        ctx.push_op(dealloc_block, raw_ptr.op_ref());

        let alloc_size = layout.total_size as u64 + RC_HEADER_SIZE;
        let size_op = clif::iconst(ctx, loc, i64_ty, alloc_size as i64);
        ctx.push_op(dealloc_block, size_op.op_ref());

        let dealloc_call = clif::call(
            ctx,
            loc,
            [raw_ptr.result(ctx), size_op.result(ctx)],
            nil_ty,
            Symbol::new(DEALLOC_FN),
        );
        ctx.push_op(dealloc_block, dealloc_call.op_ref());

        let ret_op = clif::r#return(ctx, loc, []);
        ctx.push_op(dealloc_block, ret_op.op_ref());
    }

    if variants_with_ptrs.is_empty() {
        // No managed fields: entry jumps straight to dealloc
        let jump = clif::jump(ctx, loc, [], dealloc_block);
        ctx.push_op(entry_block, jump.op_ref());

        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry_block, dealloc_block],
            parent_op: None,
        });
        let func_op = clif::func(ctx, loc, Symbol::from_dynamic(&func_name), func_ty, body);
        return func_op.op_ref();
    }

    // Build null-guarded release block chains for each variant.
    // Each variant gets a chain of check→release blocks (like struct release),
    // with the final block jumping to dealloc_block.
    let mut release_entry_blocks: Vec<BlockRef> = Vec::new();
    let mut extra_blocks: Vec<BlockRef> = Vec::new();

    for vr in &variants_with_ptrs {
        // Build chain backwards from dealloc_block
        let mut next_block = dealloc_block;

        for &offset in vr.managed_field_offsets.iter().rev() {
            // Release block: load field, release, jump to next
            let rel_block = ctx.create_block(BlockData {
                location: loc,
                args: vec![],
                ops: smallvec![],
                parent_region: None,
            });
            let reload = clif::load(ctx, loc, payload_ptr, ptr_ty, offset);
            ctx.push_op(rel_block, reload.op_ref());
            let release_op = tribute_rt::release(ctx, loc, reload.result(ctx), 0);
            ctx.push_op(rel_block, release_op.op_ref());
            let jump = clif::jump(ctx, loc, [], next_block);
            ctx.push_op(rel_block, jump.op_ref());

            // Check block: load field, null check, branch
            let chk_block = ctx.create_block(BlockData {
                location: loc,
                args: vec![],
                ops: smallvec![],
                parent_region: None,
            });
            let load_op = clif::load(ctx, loc, payload_ptr, ptr_ty, offset);
            ctx.push_op(chk_block, load_op.op_ref());
            let null_const = clif::iconst(ctx, loc, ptr_ty, 0);
            ctx.push_op(chk_block, null_const.op_ref());
            let is_null = clif::icmp(
                ctx,
                loc,
                load_op.result(ctx),
                null_const.result(ctx),
                i8_ty,
                Symbol::new("eq"),
            );
            ctx.push_op(chk_block, is_null.op_ref());
            let brif = clif::brif(ctx, loc, is_null.result(ctx), next_block, rel_block);
            ctx.push_op(chk_block, brif.op_ref());

            extra_blocks.push(rel_block);
            extra_blocks.push(chk_block);
            next_block = chk_block;
        }

        // The first check block is this variant's entry point
        release_entry_blocks.push(next_block);
    }
    // Replace release_blocks with release_entry_blocks for tag dispatch
    let release_blocks = release_entry_blocks;

    // Build check blocks for variants_with_ptrs[1..] in reverse
    let mut check_blocks: Vec<BlockRef> = Vec::new();
    let num_variants = variants_with_ptrs.len();

    // Load tag in entry block
    let tag_load = clif::load(ctx, loc, payload_ptr, i32_ty, 0);
    ctx.push_op(entry_block, tag_load.op_ref());
    let tag_val = tag_load.result(ctx);

    let mut next_else_block = dealloc_block;
    for i in (1..num_variants).rev() {
        let vr = &variants_with_ptrs[i];
        let check_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let expected = clif::iconst(ctx, loc, i32_ty, vr.tag_value as i64);
        ctx.push_op(check_block, expected.op_ref());
        let cmp_op = clif::icmp(
            ctx,
            loc,
            tag_val,
            expected.result(ctx),
            i8_ty,
            Symbol::new("eq"),
        );
        ctx.push_op(check_block, cmp_op.op_ref());
        let brif_op = clif::brif(
            ctx,
            loc,
            cmp_op.result(ctx),
            release_blocks[i],
            next_else_block,
        );
        ctx.push_op(check_block, brif_op.op_ref());

        next_else_block = check_block;
        check_blocks.push(check_block);
    }
    check_blocks.reverse();

    // Entry block: check first variant
    let first_vr = &variants_with_ptrs[0];
    let expected = clif::iconst(ctx, loc, i32_ty, first_vr.tag_value as i64);
    ctx.push_op(entry_block, expected.op_ref());
    let cmp_op = clif::icmp(
        ctx,
        loc,
        tag_val,
        expected.result(ctx),
        i8_ty,
        Symbol::new("eq"),
    );
    ctx.push_op(entry_block, cmp_op.op_ref());
    let brif_op = clif::brif(
        ctx,
        loc,
        cmp_op.result(ctx),
        release_blocks[0],
        next_else_block,
    );
    ctx.push_op(entry_block, brif_op.op_ref());

    // Assemble blocks: entry, tag check_blocks, variant null-check/release blocks, dealloc.
    // Filter extra_blocks to exclude release_entry_blocks (already in release_blocks)
    // to avoid duplicate BlockRef entries in the region.
    let mut all_blocks: Vec<BlockRef> = vec![entry_block];
    all_blocks.extend(check_blocks);
    all_blocks.extend(&release_blocks);
    for &block in &extra_blocks {
        if !release_blocks.contains(&block) {
            all_blocks.push(block);
        }
    }
    all_blocks.push(dealloc_block);

    let body = ctx.create_region(RegionData {
        location: loc,
        blocks: all_blocks.into(),
        parent_op: None,
    });
    let func_op = clif::func(ctx, loc, Symbol::from_dynamic(&func_name), func_ty, body);
    func_op.op_ref()
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::Span;
    use trunk_ir::context::{BlockArgData, BlockData, IrContext, OperationDataBuilder};
    use trunk_ir::dialect::func;
    use trunk_ir::printer::print_module;
    use trunk_ir::rewrite::Module;
    use trunk_ir::types::Attribute;

    fn rtti_plan(ctx: &IrContext, module: Module) -> Vec<RttiTypePlan> {
        let plan = crate::native::ownership_plan::build_native_ownership_plan(ctx, module)
            .expect("typed ownership plan");
        plan.remap_rtti_types(ctx, module, &[])
            .expect("exact RTTI identities")
    }

    fn test_ctx() -> (IrContext, Location) {
        let mut ctx = IrContext::new();
        let path = ctx.paths.intern("file:///test.trb".to_owned());
        let loc = Location::new(path, Span::new(0, 0));
        (ctx, loc)
    }

    fn intern_ty(ctx: &mut IrContext, dialect: &'static str, name: &'static str) -> TypeRef {
        ctx.types
            .intern(TypeDataBuilder::new(Symbol::new(dialect), Symbol::new(name)).build())
    }

    /// Build a module containing a function that creates a struct via adt.struct_new.
    fn build_struct_new_module(
        ctx: &mut IrContext,
        loc: Location,
        struct_ty: TypeRef,
        field_types: &[TypeRef],
    ) -> Module {
        // Build function type: (field_types...) -> struct_ty
        let func_ty = core::func(ctx, field_types.iter().copied(), [struct_ty]).as_type_ref();

        // Create entry block with field arguments
        let args: Vec<BlockArgData> = field_types
            .iter()
            .map(|&ty| BlockArgData {
                ty,
                attrs: Default::default(),
            })
            .collect();

        let entry = ctx.create_block(BlockData {
            location: loc,
            args,
            ops: smallvec![],
            parent_region: None,
        });

        // adt.struct_new with field args as operands
        let field_vals: Vec<_> = (0..field_types.len())
            .map(|i| ctx.block_arg(entry, i as u32))
            .collect();

        let struct_new_data =
            OperationDataBuilder::new(loc, Symbol::new("adt"), Symbol::new("struct_new"))
                .operands(field_vals)
                .result(struct_ty)
                .attr("type", Attribute::Type(struct_ty))
                .build(ctx);
        let struct_new_ref = ctx.create_op(struct_new_data);
        let struct_result = ctx.op_result(struct_new_ref, 0);
        ctx.push_op(entry, struct_new_ref);

        let ret = func::r#return(ctx, loc, [struct_result]);
        ctx.push_op(entry, ret.op_ref());

        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let func_op = func::func(ctx, loc, Symbol::new("create_struct"), func_ty, body);

        // Build module
        let module_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        ctx.push_op(module_block, func_op.op_ref());

        let module_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![module_block],
            parent_op: None,
        });

        let module_data =
            OperationDataBuilder::new(loc, Symbol::new("core"), Symbol::new("module"))
                .attr("sym_name", Attribute::Symbol(Symbol::new("test")))
                .region(module_region)
                .build(ctx);
        let module_op = ctx.create_op(module_data);

        Module::new(ctx, module_op).expect("valid arena module")
    }

    #[test]
    fn test_rtti_map_assignment() {
        let mut ctx = IrContext::new();
        let ty1 = intern_ty(&mut ctx, "test", "t1");
        let ty2 = intern_ty(&mut ctx, "test", "t2");

        let mut rtti = RttiMap::new();
        assert_eq!(rtti.get_or_insert(ty1), 32);
        assert_eq!(rtti.get_or_insert(ty2), 33);
        // Idempotent
        assert_eq!(rtti.get_or_insert(ty1), 32);
    }

    #[test]
    fn test_release_fn_prefix() {
        assert_eq!(RELEASE_FN_PREFIX, "__tribute_release_");
    }

    #[test]
    fn test_no_structs_noop() {
        let mut ctx = IrContext::new();
        let ir = r#"core.module @test {
  func.func @f(%0: core.i32) -> core.i32 {
    func.return %0
  }
}"#;
        let module = trunk_ir::parser::parse_test_module(&mut ctx, ir);
        let (tc, _) = crate::native::type_converter::native_type_converter(&mut ctx);
        let plan = rtti_plan(&ctx, module);
        let rtti = generate_rtti(&mut ctx, module, &tc, &plan);
        assert!(rtti.type_to_idx.is_empty());
    }

    #[test]
    fn boxed_primitives_receive_exact_reserved_release_functions() {
        let mut ctx = IrContext::new();
        let ir = r#"core.module @test {
  func.func @f(%0: core.i32, %1: core.f64) -> core.nil {
    %2 = tribute_rt.box_int %0 : tribute_rt.anyref
    %3 = tribute_rt.box_float %1 : tribute_rt.anyref
    func.return
  }
}"#;
        let module = trunk_ir::parser::parse_test_module(&mut ctx, ir);
        let (tc, _) = crate::native::type_converter::native_type_converter(&mut ctx);
        let plan = rtti_plan(&ctx, module);
        let rtti = generate_rtti(&mut ctx, module, &tc, &plan);
        assert!(rtti.type_to_idx.is_empty());

        let output = print_module(&ctx, module.op());
        let int_release = output
            .split("clif.func {sym_name = @__tribute_release_3")
            .nth(1)
            .expect("boxed Int must have a reserved RTTI release entry");
        assert!(int_release.contains("value = 12"));
        assert!(int_release.contains("callee = @__tribute_dealloc"));
        let float_release = output
            .split("clif.func {sym_name = @__tribute_release_4")
            .nth(1)
            .expect("boxed Float must have a reserved RTTI release entry");
        assert!(float_release.contains("value = 16"));
        assert!(float_release.contains("callee = @__tribute_dealloc"));
    }

    #[test]
    fn test_struct_no_ptr_fields() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = intern_ty(&mut ctx, "core", "i32");

        // Point(x: i32, y: i32) - no managed fields
        let point_ty = make_struct_type(&mut ctx, &[("x", i32_ty), ("y", i32_ty)]);
        let module = build_struct_new_module(&mut ctx, loc, point_ty, &[i32_ty, i32_ty]);

        let (tc, _) = crate::native::type_converter::native_type_converter(&mut ctx);
        let plan = rtti_plan(&ctx, module);
        let _rtti = generate_rtti(&mut ctx, module, &tc, &plan);

        let output = print_module(&ctx, module.op());
        insta::assert_snapshot!(output);
    }

    #[test]
    fn test_struct_with_ptr_fields() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = intern_ty(&mut ctx, "core", "i32");
        let managed_ty = intern_ty(&mut ctx, "tribute_rt", "anyref");

        // Node(value: i32, next: anyref) has one typed managed field.
        let node_ty = make_struct_type(&mut ctx, &[("value", i32_ty), ("next", managed_ty)]);
        let module = build_struct_new_module(&mut ctx, loc, node_ty, &[i32_ty, managed_ty]);

        let (tc, _) = crate::native::type_converter::native_type_converter(&mut ctx);
        let plan = rtti_plan(&ctx, module);
        let _rtti = generate_rtti(&mut ctx, module, &tc, &plan);

        let output = print_module(&ctx, module.op());
        insta::assert_snapshot!(output);
    }

    #[test]
    fn test_multiple_struct_types() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = intern_ty(&mut ctx, "core", "i32");
        let ptr_ty = intern_ty(&mut ctx, "core", "ptr");

        let point_ty = make_struct_type(&mut ctx, &[("x", i32_ty), ("y", i32_ty)]);
        let node_ty = make_struct_type(&mut ctx, &[("value", i32_ty), ("next", ptr_ty)]);

        // Build module with two struct_new ops
        let func_ty = core::func(&mut ctx, [i32_ty, i32_ty, ptr_ty], [node_ty]).as_type_ref();

        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![
                BlockArgData {
                    ty: i32_ty,
                    attrs: Default::default(),
                },
                BlockArgData {
                    ty: i32_ty,
                    attrs: Default::default(),
                },
                BlockArgData {
                    ty: ptr_ty,
                    attrs: Default::default(),
                },
            ],
            ops: smallvec![],
            parent_region: None,
        });

        let x = ctx.block_arg(entry, 0);
        let y = ctx.block_arg(entry, 1);
        let next = ctx.block_arg(entry, 2);

        // First struct_new: Point(x, y)
        let sn1 = OperationDataBuilder::new(loc, Symbol::new("adt"), Symbol::new("struct_new"))
            .operands([x, y])
            .result(point_ty)
            .attr("type", Attribute::Type(point_ty))
            .build(&mut ctx);
        let sn1_ref = ctx.create_op(sn1);
        ctx.push_op(entry, sn1_ref);

        // Second struct_new: Node(x, next)
        let sn2 = OperationDataBuilder::new(loc, Symbol::new("adt"), Symbol::new("struct_new"))
            .operands([x, next])
            .result(node_ty)
            .attr("type", Attribute::Type(node_ty))
            .build(&mut ctx);
        let sn2_ref = ctx.create_op(sn2);
        let sn2_result = ctx.op_result(sn2_ref, 0);
        ctx.push_op(entry, sn2_ref);

        let ret = func::r#return(&mut ctx, loc, [sn2_result]);
        ctx.push_op(entry, ret.op_ref());

        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let func_op = func::func(&mut ctx, loc, Symbol::new("create"), func_ty, body);

        let module_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        ctx.push_op(module_block, func_op.op_ref());

        let module_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![module_block],
            parent_op: None,
        });
        let module_data =
            OperationDataBuilder::new(loc, Symbol::new("core"), Symbol::new("module"))
                .attr("sym_name", Attribute::Symbol(Symbol::new("test")))
                .region(module_region)
                .build(&mut ctx);
        let module_op = ctx.create_op(module_data);
        let module = Module::new(&ctx, module_op).expect("valid");

        let (tc, _) = crate::native::type_converter::native_type_converter(&mut ctx);
        let plan = rtti_plan(&ctx, module);
        let rtti = generate_rtti(&mut ctx, module, &tc, &plan);

        // Both struct types should be registered
        assert!(rtti.type_to_idx.contains_key(&point_ty));
        assert!(rtti.type_to_idx.contains_key(&node_ty));

        // Should generate both release functions
        let output = print_module(&ctx, module.op());
        let point_idx = rtti.type_to_idx[&point_ty];
        let node_idx = rtti.type_to_idx[&node_ty];
        assert!(output.contains(&format!("__tribute_release_{point_idx}")));
        assert!(output.contains(&format!("__tribute_release_{node_idx}")));
    }

    #[test]
    fn test_closure_struct_skips_func_ptr() {
        let (mut ctx, loc) = test_ctx();
        let ptr_ty = intern_ty(&mut ctx, "core", "ptr");
        let managed_ty = intern_ty(&mut ctx, "tribute_rt", "anyref");

        // The raw code pointer is unmanaged; the typed environment is managed.
        let closure_ty = make_struct_type(&mut ctx, &[("func_ptr", ptr_ty), ("env", managed_ty)]);
        let module = build_struct_new_module(&mut ctx, loc, closure_ty, &[ptr_ty, managed_ty]);

        let (tc, _) = crate::native::type_converter::native_type_converter(&mut ctx);
        let plan = rtti_plan(&ctx, module);
        let _rtti = generate_rtti(&mut ctx, module, &tc, &plan);

        let output = print_module(&ctx, module.op());
        // The release function should only release the env field (not func_ptr)
        // Count tribute_rt.release ops in the release function
        let release_count = output.matches("tribute_rt.release").count();
        assert_eq!(
            release_count, 1,
            "only env should be released, not func_ptr"
        );
    }
}
