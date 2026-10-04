//! RTTI (Runtime Type Information) pass for the native backend.
//!
//! Native ownership planning declares each runtime type descriptor, a struct
//! allocation layout or one variant of an enum allocation layout, with its
//! `rtti_idx` and field kinds as a `tribute_rtti.layout` operation
//! ([`declare_rtti_layouts`]). This pass reads those declarations and
//! generates per-descriptor release functions that recursively release typed
//! managed-reference fields before deallocating the aggregate itself, and the
//! descriptor records ([`super::descriptor_records`]) that describe each index.
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
//! | 5+ | declared struct and variant descriptors | per-descriptor deep release |
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

use tribute_ir::dialect::adt::layout::{
    compute_enum_layout, compute_struct_layout, find_variant_layout,
};
use tribute_ir::dialect::tribute_rtti::FieldKind;
use trunk_ir::TypeDataBuilder;
use trunk_ir::context::{BlockArgData, BlockData, IrContext, RegionData};
use trunk_ir::dialect::clif;
use trunk_ir::location::Span;
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::rewrite::{Module, TypeConverter};
use trunk_ir::smallvec::smallvec;
use trunk_ir::types::Location;
use trunk_ir::{BlockRef, OpRef, RegionRef, StringRef, TypeRef, ValueRef};
use trunk_ir::{Symbol, SymbolPath};

use tribute_ir::dialect::{tribute_rt, tribute_rtti};

use super::descriptor_records::{self, DescriptorRecord};
use super::ownership_plan::RttiTypePlan;
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

pub use super::descriptor_records::RTTI_TABLE;

/// Name of the function that releases an allocation through its RTTI entry.
pub const DEEP_RELEASE_FN: &str = "__tribute_deep_release";

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
        let layout =
            tribute_rtti::Layout::declare(ctx, location, entry.ty, entry.tag, index, &entry.fields);
        ctx.push_op(module_block, layout.op_ref());
    }
}

/// Check that the declarations name each allocation descriptor of the module
/// exactly once, under distinct user indices.
fn validate_declarations(
    ctx: &IrContext,
    module: Module,
    layouts: &[tribute_rtti::Layout],
) -> Result<(), RttiError> {
    let mut declared = HashSet::new();
    let mut indices = HashSet::new();
    for layout in layouts {
        if !declared.insert((layout.r#type(ctx), layout.tag_ref(ctx))) {
            return Err(RttiError("a descriptor is declared more than once".into()));
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
            if let Some(descriptor) = tribute_rtti::allocation_descriptor(ctx, op) {
                allocated.insert(descriptor);
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
    }
    if allocated != declared {
        return Err(RttiError(
            "allocation descriptors differ from the declared layouts".into(),
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
    let mut layouts = tribute_rtti::Layout::declared(ctx, module);
    validate_declarations(ctx, module, &layouts)?;
    let primitive_releases = primitive_release_entries(ctx, module);

    // Phase 2: Generate per-type release functions and append to module
    let Some(module_block) = module.first_block(ctx) else {
        return Ok(());
    };

    let loc = Location::new(ctx.intern_path("<rtti>"), Span::new(0, 0));
    let mut release_fns = HashMap::new();

    // `anyref` and `intref` have no static nominal allocation layout. Their
    // release action carries a dynamic-size signal, resolved by the header
    // RTTI index before deallocation. Used primitive slots must therefore own
    // exact release functions instead of falling through with zero.
    for (rtti_idx, alloc_size) in primitive_releases {
        let func_op = generate_fixed_release_function(ctx, rtti_idx, alloc_size, loc);
        ctx.push_op(module_block, func_op);
        release_fns.insert(rtti_idx, release_fn_symbol(rtti_idx));
    }

    // Sort by rtti_idx for deterministic output
    layouts.sort_by_key(|layout| layout.index(ctx));

    let mut records = Vec::with_capacity(layouts.len());
    for layout in layouts {
        let ty = layout.r#type(ctx);
        let tag = layout.tag_ref(ctx);
        let rtti_idx = layout.index(ctx);
        let fields = layout.field_kinds(ctx);
        release_fns.insert(rtti_idx, release_fn_symbol(rtti_idx));
        let release = descriptor_release(ctx, ty, tag, &fields, type_converter);
        let func_op = generate_release_function(
            ctx,
            rtti_idx,
            &release.managed_offsets,
            release.alloc_size,
            loc,
        );
        ctx.push_op(module_block, func_op);
        records.push((rtti_idx, DescriptorRecord::of_layout(ctx, ty, tag, fields)));
    }
    descriptor_records::generate(ctx, module_block, records, &release_fns, loc);

    let deep_release = generate_deep_release_function(ctx, loc);
    ctx.push_op(module_block, deep_release);

    Ok(())
}

/// The release function of an RTTI index.
fn release_fn_symbol(rtti_idx: u32) -> SymbolPath {
    SymbolPath::from(format!("{RELEASE_FN_PREFIX}{rtti_idx}").as_str())
}

/// Build `__tribute_deep_release(payload_ptr, alloc_size)`.
///
/// ```text
/// entry(payload_ptr, alloc_size):
///   raw_ptr = payload_ptr - RC_HEADER_SIZE
///   release_fn = load ptr from __tribute_rtti[load i32 from raw_ptr + 4].release_fn
///   release_fn == null ? goto shallow : goto deep
/// shallow:
///   alloc_size == 0 ? trap : __tribute_dealloc(raw_ptr, alloc_size)
/// deep:
///   call_indirect release_fn(payload_ptr)
/// ```
///
/// A zero size is a dynamic-size signal that only an RTTI release entry can
/// resolve, so a shallow release of it traps instead of leaking.
fn generate_deep_release_function(ctx: &mut IrContext, loc: Location) -> OpRef {
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
    let entry_size = iconst(ctx, entry, descriptor_records::RECORD_SIZE as i64, tys.i64);
    let entry_offset = clif::Imul::operands(rtti_idx.result(ctx), entry_size)
        .results(tys.i64)
        .build(ctx, loc);
    push(ctx, entry, entry_offset.op_ref());
    let table = clif::SymbolAddr::operands()
        .sym(SymbolPath::from(RTTI_TABLE))
        .results(tys.ptr)
        .build(ctx, loc);
    push(ctx, entry, table.op_ref());
    let entry_addr = clif::Iadd::operands(table.result(ctx), entry_offset.result(ctx))
        .results(tys.ptr)
        .build(ctx, loc);
    push(ctx, entry, entry_addr.op_ref());
    let release_fn = clif::Load::operands(entry_addr.result(ctx))
        .offset(descriptor_records::RELEASE_FN_OFFSET as i32)
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
        .callee(SymbolPath::from(DEALLOC_FN))
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

/// What releasing one descriptor's allocation frees.
struct DescriptorRelease {
    /// Payload offsets of the released fields.
    managed_offsets: Vec<i32>,
    /// Total allocation size, including the RC header.
    alloc_size: u64,
}

/// The released field offsets and allocation size of the descriptor
/// `(ty, tag)`. A variant allocation has the size of its whole enum layout.
fn descriptor_release(
    ctx: &IrContext,
    ty: TypeRef,
    tag: Option<StringRef>,
    fields: &[FieldKind],
    type_converter: &TypeConverter,
) -> DescriptorRelease {
    use tribute_ir::dialect::tribute_rt::RC_HEADER_SIZE;

    let (offsets, total_size) = match tag {
        None => {
            let layout = compute_struct_layout(ctx, ty, type_converter)
                .expect("struct type declared as an RTTI layout must have a valid layout");
            (layout.field_offsets, layout.total_size)
        }
        Some(tag) => {
            let layout = compute_enum_layout(ctx, ty, type_converter)
                .expect("enum type declared as an RTTI layout must have a valid layout");
            let variant = find_variant_layout(&layout, tag)
                .expect("declared RTTI variant must exist in its enum layout");
            let offsets = variant
                .field_offsets
                .iter()
                .map(|offset| layout.fields_offset + offset)
                .collect();
            (offsets, layout.total_size)
        }
    };
    assert_eq!(offsets.len(), fields.len());
    DescriptorRelease {
        managed_offsets: offsets
            .into_iter()
            .zip(fields)
            .filter_map(|(offset, kind)| kind.is_released().then_some(offset as i32))
            .collect(),
        alloc_size: u64::from(total_size) + RC_HEADER_SIZE,
    }
}

/// Generate the release function of one descriptor: release each managed
/// field that is not null, then deallocate.
fn generate_release_function(
    ctx: &mut IrContext,
    rtti_idx: u32,
    managed_field_offsets: &[i32],
    alloc_size: u64,
    loc: Location,
) -> OpRef {
    let tys = ClifTypes::intern(ctx);
    let ptr_ty = tys.ptr;
    let nil_ty = tys.nil;
    let i64_ty = tys.i64;
    let i8_ty = tys.i8;

    let func_name = format!("{}{}", RELEASE_FN_PREFIX, rtti_idx);

    // Function type: (core.ptr) -> core.nil
    let func_ty = clif::func_sig(ctx, [ptr_ty], [nil_ty]).as_type_ref();

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
    gen_dealloc_and_return_with_size(
        ctx,
        loc,
        dealloc_block,
        payload_ptr,
        alloc_size,
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
        .callee(SymbolPath::from(DEALLOC_FN))
        .results([nil_ty])
        .build(ctx, loc);
    ctx.push_op(block, dealloc_call.op_ref());

    let ret_op = clif::Return::operands([]).build(ctx, loc);
    ctx.push_op(block, ret_op.op_ref());
}

/// Build an `adt.struct` type with named fields (for testing and internal use).
#[cfg(test)]
pub(crate) fn make_struct_type(ctx: &mut IrContext, fields: &[(&'static str, TypeRef)]) -> TypeRef {
    let fields = fields.iter().map(|&(name, ty)| (name, ty));
    tribute_ir::dialect::adt::struct_type(ctx, "Test", fields, trunk_ir::types::AttributeMap::new())
        .as_type_ref()
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
                .attr("sym_name", Attribute::String(ctx.intern_str("test")))
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
            tribute_rtti::Layout::declared_indices(&ctx, module),
            HashMap::from([((point_ty, None), RTTI_USER_START)])
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
        assert!(tribute_rtti::Layout::declared_indices(&ctx, module).is_empty());
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
        assert!(tribute_rtti::Layout::declared_indices(&ctx, module).is_empty());

        let output = print_module(&ctx, module.op());
        let int_release = output
            .split("clif.func {sym_name = \"__tribute_release_3\"")
            .nth(1)
            .expect("boxed Int must have a reserved RTTI release entry");
        assert!(int_release.contains("value = 12"));
        assert!(int_release.contains("callee = @__tribute_dealloc"));
        let float_release = output
            .split("clif.func {sym_name = \"__tribute_release_4\"")
            .nth(1)
            .expect("boxed Float must have a reserved RTTI release entry");
        assert!(float_release.contains("value = 16"));
        assert!(float_release.contains("callee = @__tribute_dealloc"));
    }

    #[test]
    fn each_variant_releases_its_own_fields_and_frees_the_enum_allocation() {
        use tribute_ir::dialect::tribute_rt::RC_HEADER_SIZE;

        let mut ctx = IrContext::new();
        let module = trunk_ir::parser::parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !Choice = adt.enum<{name = "Choice", variants = [["None", []], ["Pair", [core.i32, tribute_rt.anyref]]]}>
  func.func @f(%n: core.i32, %v: tribute_rt.anyref) -> core.nil {
    %none = adt.variant_new {type = !Choice, tag = "None"} : !Choice
    %pair = adt.variant_new %n, %v {type = !Choice, tag = "Pair"} : !Choice
    func.return
  }
}"#,
        );
        let (tc, _) = crate::native::type_converter::native_type_converter(&mut ctx);
        declare_planned_layouts(&mut ctx, module);
        generate_rtti(&mut ctx, module, &tc).expect("declared layouts");

        let choice = ctx.type_alias_by_text("Choice").expect("Choice alias");
        let layout = compute_enum_layout(&ctx, choice, &tc).expect("enum layout");
        let alloc_size = i64::from(layout.total_size) + RC_HEADER_SIZE as i64;
        let pair = ctx.intern_str("Pair");
        let pair_field =
            layout.fields_offset + find_variant_layout(&layout, pair).unwrap().field_offsets[1];
        let none = ctx.intern_str("None");
        let indices = tribute_rtti::Layout::declared_indices(&ctx, module);
        assert_eq!(indices.len(), 2, "one descriptor per allocated variant");

        let release = |index: u32| {
            let symbol = format!("{RELEASE_FN_PREFIX}{index}");
            module
                .ops(&ctx)
                .iter()
                .find_map(|&op| {
                    clif::Func::from_op(&ctx, op)
                        .ok()
                        .filter(|function| function.sym_name(&ctx) == symbol.as_str())
                })
                .expect("descriptor release function")
        };
        let summary = |index: u32| {
            let mut released = Vec::new();
            let mut sizes = Vec::new();
            let _ = trunk_ir::walk::walk_op::<()>(&ctx, release(index).op_ref(), &mut |op| {
                if let Ok(release) = tribute_rt::Release::from_op(&ctx, op)
                    && let trunk_ir::ValueDef::OpResult(load, _) = ctx.value_def(release.ptr(&ctx))
                {
                    released.push(clif::Load::from_op(&ctx, load).unwrap().offset(&ctx));
                }
                if let Ok(call) = clif::Call::from_op(&ctx, op)
                    && call.callee(&ctx) == DEALLOC_FN
                    && let trunk_ir::ValueDef::OpResult(size, _) =
                        ctx.value_def(ctx.op_operands(op)[1])
                {
                    sizes.push(clif::Iconst::from_op(&ctx, size).unwrap().value(&ctx));
                }
                ControlFlow::Continue(WalkAction::Advance)
            });
            (released, sizes)
        };

        assert_eq!(
            summary(indices[&(choice, Some(none))]),
            (vec![], vec![alloc_size])
        );
        assert_eq!(
            summary(indices[&(choice, Some(pair))]),
            (vec![pair_field as i32], vec![alloc_size])
        );
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
                .attr("sym_name", Attribute::String(ctx.intern_str("test")))
                .region(module_region)
                .build(&mut ctx);
        let module_op = ctx.create_op(module_data);
        let module = Module::new(&ctx, module_op).expect("valid");

        let (tc, _) = crate::native::type_converter::native_type_converter(&mut ctx);
        declare_planned_layouts(&mut ctx, module);
        generate_rtti(&mut ctx, module, &tc).expect("declared layouts");

        // Both struct types should be registered
        let indices = tribute_rtti::Layout::declared_indices(&ctx, module);
        assert!(indices.contains_key(&(point_ty, None)));
        assert!(indices.contains_key(&(node_ty, None)));

        // Should generate both release functions
        let output = print_module(&ctx, module.op());
        let point_idx = indices[&(point_ty, None)];
        let node_idx = indices[&(node_ty, None)];
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
