//! RTTI (Runtime Type Information) pass for the native backend.
//!
//! Native ownership planning declares each planned allocation type with its
//! `rtti_idx` and managed fields as a `tribute_rtti.layout` operation
//! ([`declare_rtti_layouts`]). This pass reads those declarations and
//! generates per-type release functions that recursively release typed
//! managed-reference fields before deallocating the aggregate itself.
//!
//! ## RTTI Index Layout
//!
//! | Index | Type | Release |
//! |-------|------|---------|
//! | 0 | no release function (e.g. runtime-allocated `Bytes`) | shallow |
//! | 1 | Bool | fixed 12-byte release |
//! | 2 | Nat | fixed 12-byte release |
//! | 3 | Int | fixed 12-byte release |
//! | 4 | Float | fixed 16-byte release |
//! | 5+ | declared allocation layouts | per-type deep release |
//!
//! Indices are private to one compiled program: the table and
//! `__tribute_deep_release` interpret them within the module, and only index 0
//! is shared with the runtime. Growing the reserved range therefore needs no
//! compatibility step; user indices simply start after it.
//!
//! ## Pipeline Position
//!
//! Runs before `adt_rc_header` (Phase 1.95), which stores the declared
//! `rtti_idx` values in allocation headers and then erases the declarations.

use std::collections::{HashMap, HashSet};
use std::ops::ControlFlow;

use trunk_ir::Symbol;
use trunk_ir::TypeDataBuilder;
use trunk_ir::adt_layout::{
    compute_enum_layout, compute_struct_layout, get_enum_variants, get_struct_fields,
};
use trunk_ir::context::{BlockArgData, BlockData, IrContext, RegionData};
use trunk_ir::dialect::{adt, clif};
use trunk_ir::location::Span;
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::rewrite::{Module, TypeConverter};
use trunk_ir::smallvec::smallvec;
use trunk_ir::types::Location;
use trunk_ir::{BlockRef, OpRef, RegionRef, TypeRef, ValueRef};

use tribute_ir::dialect::{tribute_rt, tribute_rtti};

use super::ownership_plan::{ManagedFieldBitmap, RttiTypePlan};
use trunk_ir::walk::{WalkAction, walk_region};

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
            ctx.intern_type(TypeDataBuilder::new("core", name).build())
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

/// Reserved RTTI indices. Index 0, which the runtime also writes, has no
/// release function.
pub const RTTI_NIL: u32 = 0;
pub const RTTI_BOOL: u32 = 1;
pub const RTTI_NAT: u32 = 2;
pub const RTTI_INT: u32 = 3;
pub const RTTI_FLOAT: u32 = 4;

/// First index for declared allocation layouts, right after the last
/// reserved index.
pub const RTTI_USER_START: u32 = RTTI_FLOAT + 1;

const PRIMITIVE_I32_ALLOC_SIZE: u64 = 12;
const PRIMITIVE_F64_ALLOC_SIZE: u64 = 16;

/// Name prefix for per-type release functions.
pub const RELEASE_FN_PREFIX: &str = "__tribute_release_";

/// Name of the runtime deallocation function.
const DEALLOC_FN: &str = "__tribute_dealloc";

/// Name of the data object mapping each RTTI index to its release function.
pub const RTTI_TABLE: &str = "__tribute_rtti_table";

/// Name of the function that releases an allocation through its RTTI entry.
pub const DEEP_RELEASE_FN: &str = "__tribute_deep_release";

/// Width of an RTTI table entry: one native function pointer.
const RTTI_TABLE_ENTRY_SIZE: u32 = 8;

/// Trap code for a dynamically sized release without an RTTI release entry.
const UNRESOLVED_DYNAMIC_RELEASE_TRAP: &str = "unresolved_dynamic_release";

/// A native RTTI declaration that contradicts the module it declares.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RttiError(String);

impl std::fmt::Display for RttiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "native RTTI layout declarations: {}", self.0)
    }
}

impl std::error::Error for RttiError {}

/// Declare the planned RTTI layouts as `tribute_rtti.layout` operations.
///
/// Each layout receives the index `RTTI_USER_START` plus its position in the
/// plan, which keeps the plan's allocation order.
pub fn declare_rtti_layouts(ctx: &mut IrContext, module: Module, rtti_types: &[RttiTypePlan]) {
    let Some(module_block) = module.first_block(ctx) else {
        return;
    };
    let location = ctx.op(module.op()).location;
    for (position, entry) in rtti_types.iter().enumerate() {
        let index = RTTI_USER_START + u32::try_from(position).expect("RTTI index fits u32");
        let layout = tribute_rtti::Layout::declare(ctx, location, entry.ty, index, &entry.fields);
        ctx.push_op(module_block, layout.op_ref());
    }
}

/// The `tribute_rtti.layout` declarations of a module, in module order.
pub fn declared_rtti_layouts(ctx: &IrContext, module: Module) -> Vec<tribute_rtti::Layout> {
    module
        .ops(ctx)
        .iter()
        .copied()
        .filter_map(|op| tribute_rtti::Layout::from_op(ctx, op).ok())
        .collect()
}

/// The declared RTTI index of each allocation layout.
pub fn declared_rtti_indices(ctx: &IrContext, module: Module) -> HashMap<TypeRef, u32> {
    declared_rtti_layouts(ctx, module)
        .into_iter()
        .map(|layout| (layout.r#type(ctx), layout.index(ctx)))
        .collect()
}

/// Check that the declarations name each allocation layout of the module
/// exactly once, under distinct user indices.
fn validate_declarations(
    ctx: &IrContext,
    module: Module,
    layouts: &[tribute_rtti::Layout],
) -> Result<(), RttiError> {
    let mut declared = HashSet::new();
    let mut indices = HashSet::new();
    for layout in layouts {
        if !declared.insert(layout.r#type(ctx)) {
            return Err(RttiError("a layout is declared more than once".into()));
        }
        let index = layout.index(ctx);
        if index < RTTI_USER_START || !indices.insert(index) {
            return Err(RttiError(format!(
                "index {index} is reserved or declared more than once"
            )));
        }
    }

    let mut allocated = HashSet::new();
    if let Some(body) = module.body(ctx) {
        let _ = walk_region::<()>(ctx, body, &mut |op| {
            let ty = adt::StructNew::from_op(ctx, op)
                .ok()
                .map(|new| new.r#type(ctx))
                .or_else(|| {
                    adt::VariantNew::from_op(ctx, op)
                        .ok()
                        .map(|new| new.r#type(ctx))
                });
            if let Some(ty) = ty {
                allocated.insert(ty);
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
    }
    if allocated != declared {
        return Err(RttiError(
            "allocation layout identities differ from the declared layouts".into(),
        ));
    }
    Ok(())
}

/// Generate a release function for every declared RTTI layout and every used
/// primitive slot, the RTTI table that maps each index to its release
/// function, and `__tribute_deep_release`, which dispatches through it.
pub fn generate_rtti(
    ctx: &mut IrContext,
    module: Module,
    type_converter: &TypeConverter,
) -> Result<(), RttiError> {
    let mut layouts = declared_rtti_layouts(ctx, module);
    validate_declarations(ctx, module, &layouts)?;
    let primitive_releases = primitive_release_entries(ctx, module);

    // Phase 2: Generate per-type release functions and append to module
    let Some(module_block) = module.first_block(ctx) else {
        return Ok(());
    };

    let loc = Location::new(ctx.intern_path("<rtti>"), Span::new(0, 0));
    let mut release_indices = Vec::new();

    // `anyref` and `intref` have no static nominal allocation layout. Their
    // release action carries a dynamic-size signal, resolved by the header
    // RTTI index before deallocation. Used primitive slots must therefore own
    // exact release functions instead of falling through with zero.
    for (rtti_idx, alloc_size) in primitive_releases {
        let func_op = generate_fixed_release_function(ctx, rtti_idx, alloc_size, loc);
        ctx.push_op(module_block, func_op);
        release_indices.push(rtti_idx);
    }

    // Sort by rtti_idx for deterministic output
    layouts.sort_by_key(|layout| layout.index(ctx));

    for layout in layouts {
        let ty = layout.r#type(ctx);
        let rtti_idx = layout.index(ctx);
        release_indices.push(rtti_idx);
        let func_op = match &layout.managed_fields(ctx) {
            ManagedFieldBitmap::Enum(fields) => {
                generate_release_function_for_enum(ctx, ty, rtti_idx, type_converter, fields, loc)
            }
            ManagedFieldBitmap::Struct(fields) => {
                generate_release_function_for_struct(ctx, ty, rtti_idx, type_converter, fields, loc)
            }
        };
        ctx.push_op(module_block, func_op);
    }

    let has_table = !release_indices.is_empty();
    if has_table {
        let table = generate_rtti_table(ctx, &release_indices, loc);
        ctx.push_op(module_block, table);
    }
    let deep_release = generate_deep_release_function(ctx, has_table, loc);
    ctx.push_op(module_block, deep_release);

    Ok(())
}

/// Declare the RTTI table: one pointer-sized entry per index up to the
/// largest release index, holding that index's release function or null.
fn generate_rtti_table(ctx: &mut IrContext, release_indices: &[u32], loc: Location) -> OpRef {
    let max_idx = *release_indices.iter().max().expect("a release index");
    let entries = max_idx as usize + 1;
    let relocs = ctx.create_block(BlockData {
        location: loc,
        args: vec![],
        ops: smallvec![],
        parent_region: None,
    });
    for &idx in release_indices {
        let reloc = clif::FuncReloc::operands()
            .offset(idx * RTTI_TABLE_ENTRY_SIZE)
            .func(Symbol::from_dynamic(&format!("{RELEASE_FN_PREFIX}{idx}")))
            .build(ctx, loc);
        ctx.push_op(relocs, reloc.op_ref());
    }
    let relocs = ctx.create_region(RegionData {
        location: loc,
        blocks: smallvec![relocs],
        parent_op: None,
    });
    // Zero bytes rather than zero-initialized data, so the table lives in a
    // data section: macOS linkers reject relocations in zero-fill sections.
    clif::Data::operands()
        .sym_name(Symbol::new(RTTI_TABLE))
        .bytes(vec![0u8; entries * RTTI_TABLE_ENTRY_SIZE as usize].into())
        .align(RTTI_TABLE_ENTRY_SIZE)
        .regions(relocs)
        .build(ctx, loc)
        .op_ref()
}

/// Build `__tribute_deep_release(payload_ptr, alloc_size)`.
///
/// ```text
/// entry(payload_ptr, alloc_size):
///   raw_ptr = payload_ptr - RC_HEADER_SIZE
///   [with a table]
///   release_fn = load ptr from rtti_table[load i32 from raw_ptr + 4]
///   release_fn == null ? goto shallow : goto deep
/// shallow:
///   alloc_size == 0 ? trap : __tribute_dealloc(raw_ptr, alloc_size)
/// deep:
///   call_indirect release_fn(payload_ptr)
/// ```
///
/// A zero size is a dynamic-size signal that only an RTTI release entry can
/// resolve, so a shallow release of it traps instead of leaking.
fn generate_deep_release_function(ctx: &mut IrContext, has_table: bool, loc: Location) -> OpRef {
    let tys = ClifTypes::intern(ctx);
    let new_block = |ctx: &mut IrContext, args: Vec<TypeRef>| {
        ctx.create_block(BlockData {
            location: loc,
            args: args
                .into_iter()
                .map(|ty| BlockArgData {
                    ty,
                    attrs: Default::default(),
                })
                .collect(),
            ops: smallvec![],
            parent_region: None,
        })
    };
    let entry = new_block(ctx, vec![tys.ptr, tys.i64]);
    let shallow = new_block(ctx, vec![]);
    let dealloc = new_block(ctx, vec![]);
    let unresolved = new_block(ctx, vec![]);
    let payload_ptr = ctx.block_arg(entry, 0);
    let alloc_size = ctx.block_arg(entry, 1);

    let push = |ctx: &mut IrContext, block: BlockRef, op: OpRef| ctx.push_op(block, op);
    let iconst = |ctx: &mut IrContext, block: BlockRef, value: i64, ty: TypeRef| {
        let op = clif::Iconst::operands()
            .value(value)
            .results(ty)
            .build(ctx, loc);
        ctx.push_op(block, op.op_ref());
        op.result(ctx)
    };

    let header = iconst(
        ctx,
        entry,
        i64::from(tribute_rt::RC_HEADER_SIZE as u32),
        tys.i64,
    );
    let raw_ptr = clif::Isub::operands(payload_ptr, header)
        .results(tys.ptr)
        .build(ctx, loc);
    push(ctx, entry, raw_ptr.op_ref());
    let raw_ptr = raw_ptr.result(ctx);

    let mut blocks = vec![entry];
    if has_table {
        let deep = new_block(ctx, vec![]);
        let rtti_idx = clif::Load::operands(raw_ptr)
            .offset(tribute_rt::RTTI_IDX_OFFSET as i32)
            .results(tys.i32)
            .build(ctx, loc);
        push(ctx, entry, rtti_idx.op_ref());
        let rtti_idx = clif::Uextend::operands(rtti_idx.result(ctx))
            .results(tys.i64)
            .build(ctx, loc);
        push(ctx, entry, rtti_idx.op_ref());
        let entry_size = iconst(ctx, entry, i64::from(RTTI_TABLE_ENTRY_SIZE), tys.i64);
        let entry_offset = clif::Imul::operands(rtti_idx.result(ctx), entry_size)
            .results(tys.i64)
            .build(ctx, loc);
        push(ctx, entry, entry_offset.op_ref());
        let table = clif::SymbolAddr::operands()
            .sym(Symbol::new(RTTI_TABLE))
            .results(tys.ptr)
            .build(ctx, loc);
        push(ctx, entry, table.op_ref());
        let entry_addr = clif::Iadd::operands(table.result(ctx), entry_offset.result(ctx))
            .results(tys.ptr)
            .build(ctx, loc);
        push(ctx, entry, entry_addr.op_ref());
        let release_fn = clif::Load::operands(entry_addr.result(ctx))
            .offset(0)
            .results(tys.ptr)
            .build(ctx, loc);
        push(ctx, entry, release_fn.op_ref());
        let null = iconst(ctx, entry, 0, tys.ptr);
        let is_null = clif::Icmp::operands(release_fn.result(ctx), null)
            .cond("eq")
            .results(tys.i8)
            .build(ctx, loc);
        push(ctx, entry, is_null.op_ref());
        let branch = clif::Brif::operands(is_null.result(ctx))
            .successors(shallow, deep)
            .build(ctx, loc);
        push(ctx, entry, branch.op_ref());

        let release_sig = clif::func_sig(ctx, [tys.ptr], [tys.nil]).as_type_ref();
        let call = clif::CallIndirect::operands(release_fn.result(ctx), [payload_ptr])
            .sig(release_sig)
            .results([tys.nil])
            .build(ctx, loc);
        push(ctx, deep, call.op_ref());
        let ret = clif::Return::operands([]).build(ctx, loc);
        push(ctx, deep, ret.op_ref());
        blocks.push(shallow);
        blocks.push(deep);
    } else {
        let jump = clif::Jump::operands([]).successors(shallow).build(ctx, loc);
        push(ctx, entry, jump.op_ref());
        blocks.push(shallow);
    }

    let zero = iconst(ctx, shallow, 0, tys.i64);
    let is_dynamic = clif::Icmp::operands(alloc_size, zero)
        .cond("eq")
        .results(tys.i8)
        .build(ctx, loc);
    push(ctx, shallow, is_dynamic.op_ref());
    let branch = clif::Brif::operands(is_dynamic.result(ctx))
        .successors(unresolved, dealloc)
        .build(ctx, loc);
    push(ctx, shallow, branch.op_ref());

    let call = clif::Call::operands([raw_ptr, alloc_size])
        .callee(Symbol::new(DEALLOC_FN))
        .results([tys.nil])
        .build(ctx, loc);
    push(ctx, dealloc, call.op_ref());
    let ret = clif::Return::operands([]).build(ctx, loc);
    push(ctx, dealloc, ret.op_ref());

    let trap = clif::Trap::operands()
        .code(UNRESOLVED_DYNAMIC_RELEASE_TRAP)
        .build(ctx, loc);
    push(ctx, unresolved, trap.op_ref());
    blocks.push(dealloc);
    blocks.push(unresolved);

    let body = ctx.create_region(RegionData {
        location: loc,
        blocks: blocks.into_iter().collect(),
        parent_op: None,
    });
    let func_ty = clif::func_sig(ctx, [tys.ptr, tys.i64], [tys.nil]).as_type_ref();
    clif::Func::operands()
        .sym_name(Symbol::new(DEEP_RELEASE_FN))
        .r#type(func_ty)
        .regions(body)
        .build(ctx, loc)
        .op_ref()
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
            for nested in ctx.op_regions(op) {
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
    let func_ty = clif::func_sig(ctx, [tys.ptr], [tys.nil]).as_type_ref();
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
    clif::Func::operands()
        .sym_name(Symbol::from_dynamic(&format!(
            "{RELEASE_FN_PREFIX}{rtti_idx}"
        )))
        .r#type(func_ty)
        .regions(body)
        .build(ctx, loc)
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
        .expect("struct type declared as an RTTI layout must have fields");
    let layout = compute_struct_layout(ctx, struct_ty, type_converter)
        .expect("struct type declared as an RTTI layout must have a valid layout");

    let tys = ClifTypes::intern(ctx);
    let ptr_ty = tys.ptr;
    let nil_ty = tys.nil;
    let i64_ty = tys.i64;
    let i8_ty = tys.i8;

    let func_name = format!("{}{}", RELEASE_FN_PREFIX, rtti_idx);

    // Function type: (core.ptr) -> core.nil
    let func_ty = clif::func_sig(ctx, [ptr_ty], [nil_ty]).as_type_ref();

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
        let dealloc_ops = ctx.block(dealloc_block).ops.clone();
        for op in dealloc_ops {
            ctx.remove_op_from_block(dealloc_block, op);
            ctx.push_op(entry_block, op);
        }

        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry_block],
            parent_op: None,
        });

        let func_op = clif::Func::operands()
            .sym_name(Symbol::from_dynamic(&func_name))
            .r#type(func_ty)
            .regions(body)
            .build(ctx, loc);
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
        let reload = clif::Load::operands(payload_ptr)
            .offset(offset)
            .results(ptr_ty)
            .build(ctx, loc);
        ctx.push_op(release_block, reload.op_ref());
        let release = tribute_rt::Release::operands(reload.result(ctx))
            .alloc_size(0)
            .build(ctx, loc);
        ctx.push_op(release_block, release.op_ref());
        let jump = clif::Jump::operands([])
            .successors(next_block)
            .build(ctx, loc);
        ctx.push_op(release_block, jump.op_ref());

        // Check block: load field, null check, branch
        let check_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let load = clif::Load::operands(payload_ptr)
            .offset(offset)
            .results(ptr_ty)
            .build(ctx, loc);
        ctx.push_op(check_block, load.op_ref());
        let null_const = clif::Iconst::operands()
            .value(0)
            .results(ptr_ty)
            .build(ctx, loc);
        ctx.push_op(check_block, null_const.op_ref());
        let is_null = clif::Icmp::operands(load.result(ctx), null_const.result(ctx))
            .cond("eq")
            .results(i8_ty)
            .build(ctx, loc);
        ctx.push_op(check_block, is_null.op_ref());
        let brif = clif::Brif::operands(is_null.result(ctx))
            .successors(next_block, release_block)
            .build(ctx, loc);
        ctx.push_op(check_block, brif.op_ref());

        blocks_after_entry.push(release_block);
        blocks_after_entry.push(check_block);
        next_block = check_block;
    }

    // Entry block gets the ops of the first check block
    let first_check = blocks_after_entry.pop().unwrap();
    let first_check_ops = ctx.block(first_check).ops.clone();
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

    let func_op = clif::Func::operands()
        .sym_name(Symbol::from_dynamic(&func_name))
        .r#type(func_ty)
        .regions(body)
        .build(ctx, loc);
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

    let hdr_sz = clif::Iconst::operands()
        .value(RC_HEADER_SIZE as i64)
        .results(i64_ty)
        .build(ctx, loc);
    ctx.push_op(block, hdr_sz.op_ref());
    let raw_ptr = clif::Isub::operands(payload_ptr, hdr_sz.result(ctx))
        .results(ptr_ty)
        .build(ctx, loc);
    ctx.push_op(block, raw_ptr.op_ref());

    let size_op = clif::Iconst::operands()
        .value(alloc_size as i64)
        .results(i64_ty)
        .build(ctx, loc);
    ctx.push_op(block, size_op.op_ref());

    let dealloc_call = clif::Call::operands([raw_ptr.result(ctx), size_op.result(ctx)])
        .callee(Symbol::new(DEALLOC_FN))
        .results([nil_ty])
        .build(ctx, loc);
    ctx.push_op(block, dealloc_call.op_ref());

    let ret_op = clif::Return::operands([]).build(ctx, loc);
    ctx.push_op(block, ret_op.op_ref());
}

/// Build an `adt.struct` type with named fields (for testing and internal use).
#[cfg(test)]
pub(crate) fn make_struct_type(ctx: &mut IrContext, fields: &[(&'static str, TypeRef)]) -> TypeRef {
    let fields = fields.iter().map(|(name, ty)| (Symbol::new(name), *ty));
    trunk_ir::dialect::adt::struct_type(ctx, "Test", fields, trunk_ir::types::AttributeMap::new())
        .as_type_ref()
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
        .expect("enum type declared as an RTTI layout must have a valid layout");
    let variants = get_enum_variants(ctx, enum_ty).unwrap_or_default();

    let tys = ClifTypes::intern(ctx);
    let ptr_ty = tys.ptr;
    let nil_ty = tys.nil;
    let i64_ty = tys.i64;
    let i32_ty = tys.i32;
    let i8_ty = tys.i8;

    let func_name = format!("{}{}", RELEASE_FN_PREFIX, rtti_idx);
    let func_ty = clif::func_sig(ctx, [ptr_ty], [nil_ty]).as_type_ref();

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

        let hdr_sz = clif::Iconst::operands()
            .value(RC_HEADER_SIZE as i64)
            .results(i64_ty)
            .build(ctx, loc);
        ctx.push_op(dealloc_block, hdr_sz.op_ref());
        let raw_ptr = clif::Isub::operands(payload_ptr, hdr_sz.result(ctx))
            .results(ptr_ty)
            .build(ctx, loc);
        ctx.push_op(dealloc_block, raw_ptr.op_ref());

        let alloc_size = layout.total_size as u64 + RC_HEADER_SIZE;
        let size_op = clif::Iconst::operands()
            .value(alloc_size as i64)
            .results(i64_ty)
            .build(ctx, loc);
        ctx.push_op(dealloc_block, size_op.op_ref());

        let dealloc_call = clif::Call::operands([raw_ptr.result(ctx), size_op.result(ctx)])
            .callee(Symbol::new(DEALLOC_FN))
            .results([nil_ty])
            .build(ctx, loc);
        ctx.push_op(dealloc_block, dealloc_call.op_ref());

        let ret_op = clif::Return::operands([]).build(ctx, loc);
        ctx.push_op(dealloc_block, ret_op.op_ref());
    }

    if variants_with_ptrs.is_empty() {
        // No managed fields: entry jumps straight to dealloc
        let jump = clif::Jump::operands([])
            .successors(dealloc_block)
            .build(ctx, loc);
        ctx.push_op(entry_block, jump.op_ref());

        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry_block, dealloc_block],
            parent_op: None,
        });
        let func_op = clif::Func::operands()
            .sym_name(Symbol::from_dynamic(&func_name))
            .r#type(func_ty)
            .regions(body)
            .build(ctx, loc);
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
            let reload = clif::Load::operands(payload_ptr)
                .offset(offset)
                .results(ptr_ty)
                .build(ctx, loc);
            ctx.push_op(rel_block, reload.op_ref());
            let release_op = tribute_rt::Release::operands(reload.result(ctx))
                .alloc_size(0)
                .build(ctx, loc);
            ctx.push_op(rel_block, release_op.op_ref());
            let jump = clif::Jump::operands([])
                .successors(next_block)
                .build(ctx, loc);
            ctx.push_op(rel_block, jump.op_ref());

            // Check block: load field, null check, branch
            let chk_block = ctx.create_block(BlockData {
                location: loc,
                args: vec![],
                ops: smallvec![],
                parent_region: None,
            });
            let load_op = clif::Load::operands(payload_ptr)
                .offset(offset)
                .results(ptr_ty)
                .build(ctx, loc);
            ctx.push_op(chk_block, load_op.op_ref());
            let null_const = clif::Iconst::operands()
                .value(0)
                .results(ptr_ty)
                .build(ctx, loc);
            ctx.push_op(chk_block, null_const.op_ref());
            let is_null = clif::Icmp::operands(load_op.result(ctx), null_const.result(ctx))
                .cond("eq")
                .results(i8_ty)
                .build(ctx, loc);
            ctx.push_op(chk_block, is_null.op_ref());
            let brif = clif::Brif::operands(is_null.result(ctx))
                .successors(next_block, rel_block)
                .build(ctx, loc);
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
    let tag_load = clif::Load::operands(payload_ptr)
        .offset(0)
        .results(i32_ty)
        .build(ctx, loc);
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
        let expected = clif::Iconst::operands()
            .value(vr.tag_value as i64)
            .results(i32_ty)
            .build(ctx, loc);
        ctx.push_op(check_block, expected.op_ref());
        let cmp_op = clif::Icmp::operands(tag_val, expected.result(ctx))
            .cond("eq")
            .results(i8_ty)
            .build(ctx, loc);
        ctx.push_op(check_block, cmp_op.op_ref());
        let brif_op = clif::Brif::operands(cmp_op.result(ctx))
            .successors(release_blocks[i], next_else_block)
            .build(ctx, loc);
        ctx.push_op(check_block, brif_op.op_ref());

        next_else_block = check_block;
        check_blocks.push(check_block);
    }
    check_blocks.reverse();

    // Entry block: check first variant
    let first_vr = &variants_with_ptrs[0];
    let expected = clif::Iconst::operands()
        .value(first_vr.tag_value as i64)
        .results(i32_ty)
        .build(ctx, loc);
    ctx.push_op(entry_block, expected.op_ref());
    let cmp_op = clif::Icmp::operands(tag_val, expected.result(ctx))
        .cond("eq")
        .results(i8_ty)
        .build(ctx, loc);
    ctx.push_op(entry_block, cmp_op.op_ref());
    let brif_op = clif::Brif::operands(cmp_op.result(ctx))
        .successors(release_blocks[0], next_else_block)
        .build(ctx, loc);
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
    let func_op = clif::Func::operands()
        .sym_name(Symbol::from_dynamic(&func_name))
        .r#type(func_ty)
        .regions(body)
        .build(ctx, loc);
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

    fn declare_planned_layouts(ctx: &mut IrContext, module: Module) {
        let plan = crate::native::ownership_plan::build_native_ownership_plan(
            ctx,
            module,
            crate::native::ownership_plan::NativeOwnershipPlanOptions::production(),
            &mut Default::default(),
        )
        .expect("typed ownership plan");
        declare_rtti_layouts(ctx, module, plan.rtti_types());
    }

    fn test_ctx() -> (IrContext, Location) {
        let mut ctx = IrContext::new();
        let path = ctx.intern_path("file:///test.trb");
        let loc = Location::new(path, Span::new(0, 0));
        (ctx, loc)
    }

    fn intern_ty(ctx: &mut IrContext, dialect: &'static str, name: &'static str) -> TypeRef {
        ctx.intern_type(TypeDataBuilder::new(dialect, name).build())
    }

    /// Build a module containing a function that creates a struct via adt.struct_new.
    fn build_struct_new_module(
        ctx: &mut IrContext,
        loc: Location,
        struct_ty: TypeRef,
        field_types: &[TypeRef],
    ) -> Module {
        // Build function type: (field_types...) -> struct_ty
        let func_ty = func::func_sig(ctx, field_types.iter().copied(), [struct_ty]).as_type_ref();

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

        let ret = func::Return::operands([struct_result]).build(ctx, loc);
        ctx.push_op(entry, ret.op_ref());

        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let func_op = func::Func::operands()
            .sym_name(Symbol::new("create_struct"))
            .r#type(func_ty)
            .regions(body)
            .build(ctx, loc);

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
    fn declarations_receive_user_indices_in_plan_order() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = intern_ty(&mut ctx, "core", "i32");
        let point_ty = make_struct_type(&mut ctx, &[("x", i32_ty)]);
        let module = build_struct_new_module(&mut ctx, loc, point_ty, &[i32_ty]);

        declare_planned_layouts(&mut ctx, module);

        assert_eq!(
            declared_rtti_indices(&ctx, module),
            HashMap::from([(point_ty, RTTI_USER_START)])
        );
    }

    #[test]
    fn generation_rejects_an_undeclared_allocation_layout() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = intern_ty(&mut ctx, "core", "i32");
        let point_ty = make_struct_type(&mut ctx, &[("x", i32_ty)]);
        let module = build_struct_new_module(&mut ctx, loc, point_ty, &[i32_ty]);
        let (tc, _) = crate::native::type_converter::native_type_converter(&mut ctx);

        let error = generate_rtti(&mut ctx, module, &tc).expect_err("undeclared layout");

        assert!(
            error
                .to_string()
                .contains("differ from the declared layouts"),
            "{error}"
        );
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
        declare_planned_layouts(&mut ctx, module);
        generate_rtti(&mut ctx, module, &tc).expect("declared layouts");
        assert!(declared_rtti_indices(&ctx, module).is_empty());
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
        declare_planned_layouts(&mut ctx, module);
        generate_rtti(&mut ctx, module, &tc).expect("declared layouts");
        assert!(declared_rtti_indices(&ctx, module).is_empty());

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
        declare_planned_layouts(&mut ctx, module);
        generate_rtti(&mut ctx, module, &tc).expect("declared layouts");

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
        declare_planned_layouts(&mut ctx, module);
        generate_rtti(&mut ctx, module, &tc).expect("declared layouts");

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
        let func_ty = func::func_sig(&mut ctx, [i32_ty, i32_ty, ptr_ty], [node_ty]).as_type_ref();

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

        let ret = func::Return::operands([sn2_result]).build(&mut ctx, loc);
        ctx.push_op(entry, ret.op_ref());

        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let func_op = func::Func::operands()
            .sym_name(Symbol::new("create"))
            .r#type(func_ty)
            .regions(body)
            .build(&mut ctx, loc);

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
        declare_planned_layouts(&mut ctx, module);
        generate_rtti(&mut ctx, module, &tc).expect("declared layouts");

        // Both struct types should be registered
        let indices = declared_rtti_indices(&ctx, module);
        assert!(indices.contains_key(&point_ty));
        assert!(indices.contains_key(&node_ty));

        // Should generate both release functions
        let output = print_module(&ctx, module.op());
        let point_idx = indices[&point_ty];
        let node_idx = indices[&node_ty];
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
        declare_planned_layouts(&mut ctx, module);
        generate_rtti(&mut ctx, module, &tc).expect("declared layouts");

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
