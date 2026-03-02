//! RTTI (Runtime Type Information) pass for the native backend.
//!
//! This pass scans the module for struct types used in `adt.struct_new` operations,
//! assigns each a unique `rtti_idx`, and generates per-type release functions that
//! recursively release pointer fields before deallocating the struct itself.
//!
//! ## RTTI Index Layout
//!
//! | Index | Type | Release |
//! |-------|------|---------|
//! | 0 | Nil | shallow |
//! | 1 | Bool | shallow |
//! | 2 | Nat | shallow |
//! | 3 | Int | shallow |
//! | 4 | Float | shallow |
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

use std::collections::{BTreeMap, HashMap};

use trunk_ir::Symbol;
use trunk_ir::adt_layout::{
    compute_enum_layout_arena, compute_struct_layout_arena, get_enum_variants_arena,
    get_struct_fields_arena,
};
use trunk_ir::arena::TypeDataBuilder;
use trunk_ir::arena::context::{BlockArgData, BlockData, IrContext, RegionData};
use trunk_ir::arena::dialect::adt as arena_adt;
use trunk_ir::arena::dialect::clif as arena_clif;
use trunk_ir::arena::ops::ArenaDialectOp;
use trunk_ir::arena::rewrite::{ArenaModule, ArenaTypeConverter};
use trunk_ir::arena::types::Location as ArenaLocation;
use trunk_ir::arena::walk::WalkAction;
use trunk_ir::arena::{BlockRef, OpRef, TypeRef, ValueRef};
use trunk_ir::location::Span;
use trunk_ir::smallvec::smallvec;

use tribute_ir::arena::dialect::tribute_rt as arena_tribute_rt;

/// First index for user-defined struct types.
pub const RTTI_USER_START: u32 = 32;

/// Reserved RTTI indices for built-in types.
pub const RTTI_NIL: u32 = 0;
pub const RTTI_BOOL: u32 = 1;
pub const RTTI_NAT: u32 = 2;
pub const RTTI_INT: u32 = 3;
pub const RTTI_FLOAT: u32 = 4;

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

/// Run the RTTI pass: scan for struct types, assign indices, generate release functions.
pub fn generate_rtti(
    ctx: &mut IrContext,
    module: ArenaModule,
    type_converter: &ArenaTypeConverter,
) -> RttiMap {
    let mut rtti_map = RttiMap::new();

    // Phase 1: Scan module for adt.struct_new and adt.variant_new operations
    collect_types(ctx, module, &mut rtti_map);

    if rtti_map.type_to_idx.is_empty() {
        return rtti_map;
    }

    // Phase 2: Generate per-type release functions and append to module
    let Some(module_block) = module.first_block(ctx) else {
        return rtti_map;
    };

    let loc = ArenaLocation::new(ctx.paths.intern("<rtti>".to_string()), Span::new(0, 0));

    // Sort by rtti_idx for deterministic output
    let mut entries: Vec<_> = rtti_map
        .type_to_idx
        .iter()
        .map(|(&ty, &idx)| (ty, idx))
        .collect();
    entries.sort_by_key(|(_, idx)| *idx);

    for (ty, rtti_idx) in entries {
        let is_enum = get_enum_variants_arena(ctx, ty).is_some();
        let func_op = if is_enum {
            generate_release_function_for_enum(ctx, ty, rtti_idx, type_converter, loc)
        } else {
            generate_release_function_for_struct(ctx, ty, rtti_idx, type_converter, loc)
        };
        ctx.push_op(module_block, func_op);
    }

    rtti_map
}

/// Scan module for adt.struct_new and adt.variant_new operations.
fn collect_types(ctx: &IrContext, module: ArenaModule, rtti_map: &mut RttiMap) {
    let body = module.body(ctx);
    use std::ops::ControlFlow;
    let Some(body) = body else { return };
    let _ = trunk_ir::arena::walk::walk_region::<()>(ctx, body, &mut |op| {
        let op_data = ctx.op(op);
        let dialect = op_data.dialect;
        let name = op_data.name;

        if dialect == Symbol::new("adt") && name == Symbol::new("struct_new") {
            if let Ok(struct_new) = arena_adt::StructNew::from_op(ctx, op) {
                let struct_ty = struct_new.r#type(ctx);
                if get_struct_fields_arena(ctx, struct_ty).is_some() {
                    rtti_map.get_or_insert(struct_ty);
                }
            }
        } else if dialect == Symbol::new("adt")
            && name == Symbol::new("variant_new")
            && let Ok(variant_new) = arena_adt::VariantNew::from_op(ctx, op)
        {
            let enum_ty = variant_new.r#type(ctx);
            if get_enum_variants_arena(ctx, enum_ty).is_some() {
                rtti_map.get_or_insert(enum_ty);
            }
        }

        ControlFlow::Continue(WalkAction::Advance)
    });
}

/// Generate release function for a struct type.
fn generate_release_function_for_struct(
    ctx: &mut IrContext,
    struct_ty: TypeRef,
    rtti_idx: u32,
    type_converter: &ArenaTypeConverter,
    loc: ArenaLocation,
) -> OpRef {
    let fields = get_struct_fields_arena(ctx, struct_ty)
        .expect("struct type registered in RttiMap must have fields");
    let layout = compute_struct_layout_arena(ctx, struct_ty, type_converter)
        .expect("struct type registered in RttiMap must have a valid layout");

    let ptr_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ptr")).build());
    let nil_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("nil")).build());
    let i64_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i64")).build());
    let i8_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i8")).build());

    let func_name = format!("{}{}", RELEASE_FN_PREFIX, rtti_idx);

    // Function type: (core.ptr) -> core.nil
    let func_ty = ctx.types.intern(
        TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func"))
            .param(nil_ty)
            .param(ptr_ty)
            .build(),
    );

    // Collect pointer field offsets (skip func_ptr fields)
    let func_ptr_sym = Symbol::new("func_ptr");
    let ptr_field_offsets: Vec<i32> = fields
        .iter()
        .enumerate()
        .filter_map(|(i, (name, field_ty))| {
            if *name == func_ptr_sym {
                return None;
            }
            let native_ty = type_converter
                .convert_type(ctx, *field_ty)
                .unwrap_or(*field_ty);
            let native_data = ctx.types.get(native_ty);
            if native_data.dialect == Symbol::new("core") && native_data.name == Symbol::new("ptr")
            {
                Some(layout.field_offsets[i] as i32)
            } else {
                None
            }
        })
        .collect();

    // Build entry block with payload_ptr argument
    let entry_block = ctx.create_block(BlockData {
        location: loc,
        args: vec![BlockArgData {
            ty: ptr_ty,
            attrs: BTreeMap::new(),
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

    if ptr_field_offsets.is_empty() {
        // No pointer fields: entry block IS the dealloc block
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

        let func_op = arena_clif::func(ctx, loc, Symbol::from_dynamic(&func_name), func_ty, body);
        return func_op.op_ref();
    }

    // Build null-guarded field check/release blocks backwards
    let mut blocks_after_entry: Vec<BlockRef> = vec![dealloc_block];
    let mut next_block = dealloc_block;

    for &offset in ptr_field_offsets.iter().rev() {
        // Release block: load field, release, jump to next
        let release_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let reload = arena_clif::load(ctx, loc, payload_ptr, ptr_ty, offset);
        ctx.push_op(release_block, reload.op_ref());
        let release = arena_tribute_rt::release(ctx, loc, reload.result(ctx), 0);
        ctx.push_op(release_block, release.op_ref());
        let jump = arena_clif::jump(ctx, loc, [], next_block);
        ctx.push_op(release_block, jump.op_ref());

        // Check block: load field, null check, branch
        let check_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let load = arena_clif::load(ctx, loc, payload_ptr, ptr_ty, offset);
        ctx.push_op(check_block, load.op_ref());
        let null_const = arena_clif::iconst(ctx, loc, ptr_ty, 0);
        ctx.push_op(check_block, null_const.op_ref());
        let is_null = arena_clif::icmp(
            ctx,
            loc,
            load.result(ctx),
            null_const.result(ctx),
            i8_ty,
            Symbol::new("eq"),
        );
        ctx.push_op(check_block, is_null.op_ref());
        let brif = arena_clif::brif(ctx, loc, is_null.result(ctx), next_block, release_block);
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

    let func_op = arena_clif::func(ctx, loc, Symbol::from_dynamic(&func_name), func_ty, body);
    func_op.op_ref()
}

/// Emit dealloc + return ops into a block.
#[allow(clippy::too_many_arguments)]
fn gen_dealloc_and_return(
    ctx: &mut IrContext,
    loc: ArenaLocation,
    block: BlockRef,
    payload_ptr: ValueRef,
    layout: &trunk_ir::adt_layout::StructLayout,
    ptr_ty: TypeRef,
    nil_ty: TypeRef,
    i64_ty: TypeRef,
) {
    use tribute_ir::dialect::tribute_rt::RC_HEADER_SIZE;

    let hdr_sz = arena_clif::iconst(ctx, loc, i64_ty, RC_HEADER_SIZE as i64);
    ctx.push_op(block, hdr_sz.op_ref());
    let raw_ptr = arena_clif::isub(ctx, loc, payload_ptr, hdr_sz.result(ctx), ptr_ty);
    ctx.push_op(block, raw_ptr.op_ref());

    let alloc_size = layout.total_size as u64 + RC_HEADER_SIZE;
    let size_op = arena_clif::iconst(ctx, loc, i64_ty, alloc_size as i64);
    ctx.push_op(block, size_op.op_ref());

    let dealloc_call = arena_clif::call(
        ctx,
        loc,
        [raw_ptr.result(ctx), size_op.result(ctx)],
        nil_ty,
        Symbol::new(DEALLOC_FN),
    );
    ctx.push_op(block, dealloc_call.op_ref());

    let ret_op = arena_clif::r#return(ctx, loc, []);
    ctx.push_op(block, ret_op.op_ref());
}

/// Generate release function for an enum type.
fn generate_release_function_for_enum(
    ctx: &mut IrContext,
    enum_ty: TypeRef,
    rtti_idx: u32,
    type_converter: &ArenaTypeConverter,
    loc: ArenaLocation,
) -> OpRef {
    let layout = compute_enum_layout_arena(ctx, enum_ty, type_converter)
        .expect("enum type registered in RttiMap must have a valid layout");
    let variants = get_enum_variants_arena(ctx, enum_ty).unwrap_or_default();

    let ptr_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ptr")).build());
    let nil_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("nil")).build());
    let i64_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i64")).build());
    let i32_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());
    let i1_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i1")).build());

    let func_name = format!("{}{}", RELEASE_FN_PREFIX, rtti_idx);
    let func_ty = ctx.types.intern(
        TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func"))
            .param(nil_ty)
            .param(ptr_ty)
            .build(),
    );

    let entry_block = ctx.create_block(BlockData {
        location: loc,
        args: vec![BlockArgData {
            ty: ptr_ty,
            attrs: BTreeMap::new(),
        }],
        ops: smallvec![],
        parent_region: None,
    });
    let payload_ptr = ctx.block_arg(entry_block, 0);

    // Collect variants with pointer fields
    struct VariantRelease {
        tag_value: u32,
        ptr_field_offsets: Vec<i32>,
    }
    let mut variants_with_ptrs: Vec<VariantRelease> = Vec::new();

    for (variant_idx, (_variant_name, field_types)) in variants.iter().enumerate() {
        let variant_layout = &layout.variant_layouts[variant_idx];
        let ptr_field_offsets: Vec<i32> = field_types
            .iter()
            .enumerate()
            .filter_map(|(field_idx, field_ty)| {
                let native_ty = type_converter
                    .convert_type(ctx, *field_ty)
                    .unwrap_or(*field_ty);
                let native_data = ctx.types.get(native_ty);
                if native_data.dialect == Symbol::new("core")
                    && native_data.name == Symbol::new("ptr")
                {
                    Some((layout.fields_offset + variant_layout.field_offsets[field_idx]) as i32)
                } else {
                    None
                }
            })
            .collect();

        if !ptr_field_offsets.is_empty() {
            variants_with_ptrs.push(VariantRelease {
                tag_value: variant_layout.tag_value,
                ptr_field_offsets,
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

        let hdr_sz = arena_clif::iconst(ctx, loc, i64_ty, RC_HEADER_SIZE as i64);
        ctx.push_op(dealloc_block, hdr_sz.op_ref());
        let raw_ptr = arena_clif::isub(ctx, loc, payload_ptr, hdr_sz.result(ctx), ptr_ty);
        ctx.push_op(dealloc_block, raw_ptr.op_ref());

        let alloc_size = layout.total_size as u64 + RC_HEADER_SIZE;
        let size_op = arena_clif::iconst(ctx, loc, i64_ty, alloc_size as i64);
        ctx.push_op(dealloc_block, size_op.op_ref());

        let dealloc_call = arena_clif::call(
            ctx,
            loc,
            [raw_ptr.result(ctx), size_op.result(ctx)],
            nil_ty,
            Symbol::new(DEALLOC_FN),
        );
        ctx.push_op(dealloc_block, dealloc_call.op_ref());

        let ret_op = arena_clif::r#return(ctx, loc, []);
        ctx.push_op(dealloc_block, ret_op.op_ref());
    }

    if variants_with_ptrs.is_empty() {
        // No pointer fields: entry jumps straight to dealloc
        let jump = arena_clif::jump(ctx, loc, [], dealloc_block);
        ctx.push_op(entry_block, jump.op_ref());

        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry_block, dealloc_block],
            parent_op: None,
        });
        let func_op = arena_clif::func(ctx, loc, Symbol::from_dynamic(&func_name), func_ty, body);
        return func_op.op_ref();
    }

    // Build release blocks for each variant
    let mut release_blocks: Vec<BlockRef> = Vec::new();
    for vr in &variants_with_ptrs {
        let release_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        for &offset in &vr.ptr_field_offsets {
            let load_op = arena_clif::load(ctx, loc, payload_ptr, ptr_ty, offset);
            ctx.push_op(release_block, load_op.op_ref());
            let release_op = arena_tribute_rt::release(ctx, loc, load_op.result(ctx), 0);
            ctx.push_op(release_block, release_op.op_ref());
        }
        let jump_to_dealloc = arena_clif::jump(ctx, loc, [], dealloc_block);
        ctx.push_op(release_block, jump_to_dealloc.op_ref());
        release_blocks.push(release_block);
    }

    // Build check blocks for variants_with_ptrs[1..] in reverse
    let mut check_blocks: Vec<BlockRef> = Vec::new();
    let num_variants = variants_with_ptrs.len();

    // Load tag in entry block
    let tag_load = arena_clif::load(ctx, loc, payload_ptr, i32_ty, 0);
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
        let expected = arena_clif::iconst(ctx, loc, i32_ty, vr.tag_value as i64);
        ctx.push_op(check_block, expected.op_ref());
        let cmp_op = arena_clif::icmp(
            ctx,
            loc,
            tag_val,
            expected.result(ctx),
            i1_ty,
            Symbol::new("eq"),
        );
        ctx.push_op(check_block, cmp_op.op_ref());
        let brif_op = arena_clif::brif(
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
    let expected = arena_clif::iconst(ctx, loc, i32_ty, first_vr.tag_value as i64);
    ctx.push_op(entry_block, expected.op_ref());
    let cmp_op = arena_clif::icmp(
        ctx,
        loc,
        tag_val,
        expected.result(ctx),
        i1_ty,
        Symbol::new("eq"),
    );
    ctx.push_op(entry_block, cmp_op.op_ref());
    let brif_op = arena_clif::brif(
        ctx,
        loc,
        cmp_op.result(ctx),
        release_blocks[0],
        next_else_block,
    );
    ctx.push_op(entry_block, brif_op.op_ref());

    // Assemble blocks: entry, check_blocks, release_blocks, dealloc
    let mut all_blocks: Vec<BlockRef> = vec![entry_block];
    all_blocks.extend(check_blocks);
    all_blocks.extend(release_blocks);
    all_blocks.push(dealloc_block);

    let body = ctx.create_region(RegionData {
        location: loc,
        blocks: all_blocks.into(),
        parent_op: None,
    });
    let func_op = arena_clif::func(ctx, loc, Symbol::from_dynamic(&func_name), func_ty, body);
    func_op.op_ref()
}
