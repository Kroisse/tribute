//! Lower the `adt` array operations to `clif` on the native target.
//!
//! An array is an RC object whose payload is `[len: i64] [elements...]`
//! ([`array_layout`](super::array_layout)). Construction allocates it with the
//! RTTI index its `tribute_rtti.layout` declares, so [`adt_rc_header`] runs
//! these patterns in its applicator before it erases the declarations.
//!
//! - `adt.array_of` allocates its constant size and stores each element.
//! - `adt.array_new` and `adt.array_splice` call a helper generated once per
//!   descriptor, `__tribute_array_new_{idx}` and
//!   `__tribute_array_splice_{idx}`, whose loops fill or copy the elements.
//!   When the declared element kind is released, the helper retains each
//!   element it fills or copies; the operation owns those acquisitions.
//!   `adt.array_splice` then stores its `elements` in the gap the helper left.
//! - `adt.array_get`, `adt.array_set`, and `adt.array_len` load the length and
//!   access the element at `payload + 8 + index × stride`.
//!
//! Every index is checked against the length with an unsigned comparison
//! before the access, and a length whose allocation size overflows traps.
//! The checks are `clif.trapnz`, so the accesses stay straight-line.
//!
//! [`adt_rc_header`]: super::adt_rc_header

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::ops::ControlFlow;

use tribute_ir::dialect::adt;
use tribute_ir::dialect::{tribute_rt, tribute_rtti};
use trunk_ir::SymbolPath;
use trunk_ir::context::{BlockArgData, BlockData, IrContext, RegionData};
use trunk_ir::dialect::{clif, core};
use trunk_ir::location::Span;
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{BlockRef, OpRef, TypeRef, ValueRef};
use trunk_ir::rewrite::{Module, PatternRewriter, RewritePattern, TypeConverter};
use trunk_ir::smallvec::smallvec;
use trunk_ir::types::Location;
use trunk_ir::walk::{WalkAction, walk_region};

use super::adt_rc_header::rc_allocation;
use super::array_layout::{ELEMENTS_OFFSET, FIXED_ALLOC_SIZE, LEN_OFFSET, element_stride};

/// Name prefix of the helpers that build an `adt.array_new` result.
const ARRAY_NEW_FN_PREFIX: &str = "__tribute_array_new_";
/// Name prefix of the helpers that copy the kept ranges of an
/// `adt.array_splice` source.
const ARRAY_SPLICE_FN_PREFIX: &str = "__tribute_array_splice_";

/// Trap code of an element access at or past the length.
const INDEX_OUT_OF_BOUNDS_TRAP: &str = "array_index_out_of_bounds";
/// Trap code of a splice range outside `start ≤ end ≤ len`.
const SPLICE_OUT_OF_BOUNDS_TRAP: &str = "array_splice_out_of_bounds";
/// Trap code of a length whose allocation size does not fit.
const SIZE_OVERFLOW_TRAP: &str = "array_size_overflow";

/// The native layout of one declared array descriptor.
#[derive(Clone, Copy)]
struct ArrayDescriptor {
    index: u32,
    /// Whether the elements are owned references the array retains.
    released: bool,
    /// The native type of an element.
    element: TypeRef,
    stride: u32,
}

impl ArrayDescriptor {
    /// The longest length whose allocation size fits a signed 64-bit size.
    fn max_len(self) -> i64 {
        ((i64::MAX as u64 - FIXED_ALLOC_SIZE) / u64::from(self.stride)) as i64
    }
}

/// The native element type and stride of an array type.
#[derive(Clone, Copy)]
struct ElementLayout {
    element: TypeRef,
    stride: u32,
}

impl ElementLayout {
    /// The layout of `array_ty`'s elements. A released element converts to
    /// a pointer, so its native size is the released stride as well.
    fn of(ctx: &IrContext, array_ty: TypeRef, type_converter: &TypeConverter) -> Option<Self> {
        let element = adt::Array::from_type_ref(ctx, array_ty)?.element(ctx);
        Some(Self {
            element: type_converter.convert_type_or_identity(ctx, element),
            stride: element_stride(ctx, element, false, type_converter),
        })
    }
}

/// The patterns that lower the array operations of one module, which
/// [`prepare`] builds while the RTTI layouts are still declared.
pub(super) struct ArrayPatterns {
    descriptors: HashMap<TypeRef, ArrayDescriptor>,
    /// The element layout of each `adt.array_get`, `adt.array_set`, and
    /// `adt.array_len`, read before construction rewrites erase the array
    /// type of their operands.
    accesses: HashMap<OpRef, ElementLayout>,
}

/// Read the declared array descriptors, record the element layout of every
/// array access, and add the `adt.array_new` and `adt.array_splice` helpers
/// the module uses.
pub(super) fn prepare(
    ctx: &mut IrContext,
    module: Module,
    type_converter: &TypeConverter,
) -> ArrayPatterns {
    let mut descriptors = HashMap::default();
    for layout in tribute_rtti::Layout::declared(ctx, module) {
        let ty = layout.r#type(ctx);
        let Some(array) = adt::Array::from_type_ref(ctx, ty) else {
            continue;
        };
        let [kind] = layout.field_kinds(ctx)[..] else {
            panic!("an array descriptor declares the kind of its element");
        };
        let element = array.element(ctx);
        let released = kind.is_released();
        let native = if released {
            core::Ptr::params().intern(ctx).as_type_ref()
        } else {
            type_converter.convert_type_or_identity(ctx, element)
        };
        descriptors.insert(
            ty,
            ArrayDescriptor {
                index: layout.index(ctx),
                released,
                element: native,
                stride: element_stride(ctx, element, released, type_converter),
            },
        );
    }

    let mut accesses = HashMap::default();
    let mut helpers = Vec::new();
    let mut seen = HashSet::default();
    if let Some(body) = module.body(ctx) {
        let _ = walk_region::<()>(ctx, body, &mut |op| {
            let array = if let Ok(get) = adt::ArrayGet::from_op(ctx, op) {
                Some(get.r#ref(ctx))
            } else if let Ok(set) = adt::ArraySet::from_op(ctx, op) {
                Some(set.r#ref(ctx))
            } else if let Ok(len) = adt::ArrayLen::from_op(ctx, op) {
                Some(len.r#ref(ctx))
            } else {
                None
            };
            if let Some(array) = array {
                let array_ty = ctx.value_ty(array);
                let layout = ElementLayout::of(ctx, array_ty, type_converter)
                    .expect("an array access reads an adt.array operand");
                accesses.insert(op, layout);
            }
            let helper = if let Ok(new) = adt::ArrayNew::from_op(ctx, op) {
                Some((Helper::New, new.r#type(ctx)))
            } else if let Ok(splice) = adt::ArraySplice::from_op(ctx, op) {
                Some((Helper::Splice, splice.r#type(ctx)))
            } else {
                None
            };
            if let Some(helper) = helper
                && seen.insert(helper)
            {
                helpers.push(helper);
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
    }

    if let Some(module_block) = module.first_block(ctx) {
        let loc = Location::new(ctx.intern_path("<array>"), Span::new(0, 0));
        for (helper, array_ty) in helpers {
            let descriptor = *descriptors
                .get(&array_ty)
                .expect("an array construction has a declared descriptor");
            let func = match helper {
                Helper::New => array_new_function(ctx, descriptor, loc),
                Helper::Splice => array_splice_function(ctx, descriptor, loc),
            };
            ctx.push_op(module_block, func);
        }
    }

    ArrayPatterns {
        descriptors,
        accesses,
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Helper {
    New,
    Splice,
}

fn helper_symbol(helper: Helper, descriptor: ArrayDescriptor) -> SymbolPath {
    let prefix = match helper {
        Helper::New => ARRAY_NEW_FN_PREFIX,
        Helper::Splice => ARRAY_SPLICE_FN_PREFIX,
    };
    SymbolPath::new([format!("{prefix}{}", descriptor.index).as_str()])
}

/// Appends `clif` operations to a list.
struct Emitter<'a> {
    loc: Location,
    ops: &'a mut Vec<OpRef>,
}

impl Emitter<'_> {
    fn push<O: DialectOp>(&mut self, op: O) -> O {
        self.ops.push(op.op_ref());
        op
    }

    fn i64(&mut self, ctx: &mut IrContext, value: i64) -> ValueRef {
        let i64_ty = core::I64::type_ref(ctx);
        let op = clif::Iconst::operands()
            .value(value)
            .results(i64_ty)
            .build(ctx, self.loc);
        self.push(op).result(ctx)
    }

    fn iadd(&mut self, ctx: &mut IrContext, lhs: ValueRef, rhs: ValueRef, ty: TypeRef) -> ValueRef {
        let op = clif::Iadd::operands(lhs, rhs)
            .results(ty)
            .build(ctx, self.loc);
        self.push(op).result(ctx)
    }

    fn isub(&mut self, ctx: &mut IrContext, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        let i64_ty = core::I64::type_ref(ctx);
        let op = clif::Isub::operands(lhs, rhs)
            .results(i64_ty)
            .build(ctx, self.loc);
        self.push(op).result(ctx)
    }

    fn imul(&mut self, ctx: &mut IrContext, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        let i64_ty = core::I64::type_ref(ctx);
        let op = clif::Imul::operands(lhs, rhs)
            .results(i64_ty)
            .build(ctx, self.loc);
        self.push(op).result(ctx)
    }

    fn icmp(
        &mut self,
        ctx: &mut IrContext,
        cond: &'static str,
        lhs: ValueRef,
        rhs: ValueRef,
    ) -> ValueRef {
        let i8_ty = core::I8::type_ref(ctx);
        let op = clif::Icmp::operands(lhs, rhs)
            .cond(cond)
            .results(i8_ty)
            .build(ctx, self.loc);
        self.push(op).result(ctx)
    }

    /// Trap with `code` when `cond lhs, rhs` holds.
    fn trap_if(
        &mut self,
        ctx: &mut IrContext,
        cond: &'static str,
        lhs: ValueRef,
        rhs: ValueRef,
        code: &'static str,
    ) {
        let failed = self.icmp(ctx, cond, lhs, rhs);
        let trap = clif::Trapnz::operands(failed)
            .code(code)
            .build(ctx, self.loc);
        self.push(trap);
    }

    fn load(&mut self, ctx: &mut IrContext, addr: ValueRef, offset: i32, ty: TypeRef) -> ValueRef {
        let op = clif::Load::operands(addr)
            .offset(offset)
            .results(ty)
            .build(ctx, self.loc);
        self.push(op).result(ctx)
    }

    fn store(&mut self, ctx: &mut IrContext, value: ValueRef, addr: ValueRef, offset: i32) {
        let op = clif::Store::operands(value, addr)
            .offset(offset)
            .build(ctx, self.loc);
        self.push(op);
    }

    fn retain(&mut self, ctx: &mut IrContext, value: ValueRef) {
        let ptr_ty = core::Ptr::params().intern(ctx).as_type_ref();
        let op = tribute_rt::Retain::operands(value)
            .results(ptr_ty)
            .build(ctx, self.loc);
        self.push(op);
    }

    /// The address of element `index` less [`ELEMENTS_OFFSET`].
    fn element_base(
        &mut self,
        ctx: &mut IrContext,
        payload: ValueRef,
        index: ValueRef,
        stride: u32,
    ) -> ValueRef {
        let stride = self.i64(ctx, i64::from(stride));
        let offset = self.imul(ctx, index, stride);
        let ptr_ty = core::Ptr::params().intern(ctx).as_type_ref();
        self.iadd(ctx, payload, offset, ptr_ty)
    }

    /// Load the length of `array` and trap unless `index < len`.
    fn check_index(&mut self, ctx: &mut IrContext, array: ValueRef, index: ValueRef) {
        let i64_ty = core::I64::type_ref(ctx);
        let len = self.load(ctx, array, LEN_OFFSET, i64_ty);
        self.trap_if(ctx, "uge", index, len, INDEX_OUT_OF_BOUNDS_TRAP);
    }

    /// Allocate an array of `len` elements of `descriptor` and store `len`,
    /// trapping when its allocation size does not fit.
    fn allocate(
        &mut self,
        ctx: &mut IrContext,
        len: ValueRef,
        descriptor: ArrayDescriptor,
    ) -> ValueRef {
        let max_len = self.i64(ctx, descriptor.max_len());
        self.trap_if(ctx, "ugt", len, max_len, SIZE_OVERFLOW_TRAP);
        let stride = self.i64(ctx, i64::from(descriptor.stride));
        let elements = self.imul(ctx, len, stride);
        let fixed = self.i64(ctx, FIXED_ALLOC_SIZE as i64);
        let i64_ty = core::I64::type_ref(ctx);
        let size = self.iadd(ctx, fixed, elements, i64_ty);
        let payload = rc_allocation(ctx, self.loc, size, descriptor.index, self.ops);
        self.store(ctx, len, payload, LEN_OFFSET);
        payload
    }
}

impl RewritePattern for ArrayPatterns {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let mut ops = Vec::new();
        let mut emit = Emitter {
            loc: ctx.op(op).location,
            ops: &mut ops,
        };
        let result = if let Ok(of) = adt::ArrayOf::from_op(ctx, op) {
            let descriptor = self.descriptor(of.r#type(ctx));
            let elements = of.elements(ctx).to_vec();
            let size = FIXED_ALLOC_SIZE + elements.len() as u64 * u64::from(descriptor.stride);
            let size = emit.i64(ctx, size as i64);
            let payload = rc_allocation(ctx, emit.loc, size, descriptor.index, emit.ops);
            let len = emit.i64(ctx, elements.len() as i64);
            emit.store(ctx, len, payload, LEN_OFFSET);
            for (position, element) in elements.into_iter().enumerate() {
                let offset = ELEMENTS_OFFSET as usize + position * descriptor.stride as usize;
                let offset = i32::try_from(offset).expect("array_of element offset fits i32");
                emit.store(ctx, element, payload, offset);
            }
            Some(payload)
        } else if let Ok(new) = adt::ArrayNew::from_op(ctx, op) {
            let descriptor = self.descriptor(new.r#type(ctx));
            let ptr_ty = core::Ptr::params().intern(ctx).as_type_ref();
            let call = clif::Call::operands([new.length(ctx), new.initial(ctx)])
                .callee(helper_symbol(Helper::New, descriptor))
                .results([ptr_ty])
                .build(ctx, emit.loc);
            Some(emit.push(call).results(ctx)[0])
        } else if let Ok(splice) = adt::ArraySplice::from_op(ctx, op) {
            let descriptor = self.descriptor(splice.r#type(ctx));
            let start = splice.start(ctx);
            let elements = splice.elements(ctx).to_vec();
            let count = emit.i64(ctx, elements.len() as i64);
            let ptr_ty = core::Ptr::params().intern(ctx).as_type_ref();
            let call = clif::Call::operands([splice.source(ctx), start, splice.end(ctx), count])
                .callee(helper_symbol(Helper::Splice, descriptor))
                .results([ptr_ty])
                .build(ctx, emit.loc);
            let payload = emit.push(call).results(ctx)[0];
            if !elements.is_empty() {
                let base = emit.element_base(ctx, payload, start, descriptor.stride);
                for (position, element) in elements.into_iter().enumerate() {
                    let offset = ELEMENTS_OFFSET as usize + position * descriptor.stride as usize;
                    let offset = i32::try_from(offset).expect("splice element offset fits i32");
                    emit.store(ctx, element, base, offset);
                }
            }
            Some(payload)
        } else if let Ok(get) = adt::ArrayGet::from_op(ctx, op) {
            let layout = self.access(op);
            let (array, index) = (get.r#ref(ctx), get.index(ctx));
            emit.check_index(ctx, array, index);
            let base = emit.element_base(ctx, array, index, layout.stride);
            Some(emit.load(ctx, base, ELEMENTS_OFFSET, layout.element))
        } else if let Ok(set) = adt::ArraySet::from_op(ctx, op) {
            let layout = self.access(op);
            let (array, index) = (set.r#ref(ctx), set.index(ctx));
            emit.check_index(ctx, array, index);
            let base = emit.element_base(ctx, array, index, layout.stride);
            emit.store(ctx, set.value(ctx), base, ELEMENTS_OFFSET);
            None
        } else if let Ok(len) = adt::ArrayLen::from_op(ctx, op) {
            let i64_ty = core::I64::type_ref(ctx);
            Some(emit.load(ctx, len.r#ref(ctx), LEN_OFFSET, i64_ty))
        } else {
            return false;
        };
        for op in ops {
            rewriter.insert_op(op);
        }
        rewriter.erase_op(result.into_iter().collect());
        true
    }
}

impl ArrayPatterns {
    fn descriptor(&self, array_ty: TypeRef) -> ArrayDescriptor {
        *self
            .descriptors
            .get(&array_ty)
            .expect("an array construction has a declared tribute_rtti.layout")
    }

    fn access(&self, op: OpRef) -> ElementLayout {
        *self
            .accesses
            .get(&op)
            .expect("array accesses are recorded before rewriting")
    }
}

fn new_block(ctx: &mut IrContext, loc: Location, args: &[TypeRef]) -> BlockRef {
    ctx.create_block(BlockData {
        location: loc,
        args: args
            .iter()
            .map(|&ty| BlockArgData {
                ty,
                attrs: Default::default(),
            })
            .collect(),
        ops: smallvec![],
        parent_region: None,
    })
}

fn fill_block(ctx: &mut IrContext, block: BlockRef, ops: Vec<OpRef>) {
    for op in ops {
        ctx.push_op(block, op);
    }
}

fn jump(ctx: &mut IrContext, loc: Location, args: Vec<ValueRef>, dest: BlockRef) -> OpRef {
    clif::Jump::operands(args)
        .successors(dest)
        .build(ctx, loc)
        .op_ref()
}

/// Build a loop over `index` in `[from, to)` that runs `body` and continues
/// to `exit`. Returns the jump that enters it and the loop's blocks.
fn counted_loop(
    ctx: &mut IrContext,
    loc: Location,
    from: ValueRef,
    to: ValueRef,
    exit: BlockRef,
    body: impl FnOnce(&mut IrContext, &mut Emitter<'_>, ValueRef),
) -> (OpRef, [BlockRef; 2]) {
    let i64_ty = core::I64::type_ref(ctx);
    let head = new_block(ctx, loc, &[i64_ty]);
    let body_block = new_block(ctx, loc, &[]);
    let index = ctx.block_arg(head, 0);

    let mut ops = Vec::new();
    let mut emit = Emitter { loc, ops: &mut ops };
    let more = emit.icmp(ctx, "ult", index, to);
    let branch = clif::Brif::operands(more)
        .successors(body_block, exit)
        .build(ctx, loc);
    emit.push(branch);
    fill_block(ctx, head, ops);

    let mut ops = Vec::new();
    let mut emit = Emitter { loc, ops: &mut ops };
    body(ctx, &mut emit, index);
    let one = emit.i64(ctx, 1);
    let next = emit.iadd(ctx, index, one, i64_ty);
    ops.push(jump(ctx, loc, vec![next], head));
    fill_block(ctx, body_block, ops);

    (jump(ctx, loc, vec![from], head), [head, body_block])
}

fn helper_function(
    ctx: &mut IrContext,
    loc: Location,
    name: SymbolPath,
    params: &[TypeRef],
    blocks: Vec<BlockRef>,
) -> OpRef {
    let ptr_ty = core::Ptr::params().intern(ctx).as_type_ref();
    let func_ty = clif::func_sig(ctx, params.iter().copied(), [ptr_ty]).as_type_ref();
    let region = ctx.create_region(RegionData {
        location: loc,
        blocks: blocks.into(),
        parent_op: None,
    });
    let name = name.leaf().expect("helper names are simple").to_string();
    clif::Func::operands()
        .sym_name(name)
        .r#type(func_ty)
        .regions(region)
        .build(ctx, loc)
        .op_ref()
}

/// Build `__tribute_array_new_{idx}(len, initial) -> payload`, which stores
/// `initial` in every element and retains it once per element when the
/// elements are released.
fn array_new_function(ctx: &mut IrContext, descriptor: ArrayDescriptor, loc: Location) -> OpRef {
    let i64_ty = core::I64::type_ref(ctx);
    let params = [i64_ty, descriptor.element];
    let entry = new_block(ctx, loc, &params);
    let done = new_block(ctx, loc, &[]);
    let len = ctx.block_arg(entry, 0);
    let initial = ctx.block_arg(entry, 1);

    let mut ops = Vec::new();
    let mut emit = Emitter { loc, ops: &mut ops };
    let payload = emit.allocate(ctx, len, descriptor);
    let zero = emit.i64(ctx, 0);
    let (enter, [head, body]) = counted_loop(ctx, loc, zero, len, done, |ctx, emit, index| {
        let base = emit.element_base(ctx, payload, index, descriptor.stride);
        emit.store(ctx, initial, base, ELEMENTS_OFFSET);
        if descriptor.released {
            emit.retain(ctx, initial);
        }
    });
    ops.push(enter);
    fill_block(ctx, entry, ops);

    let ret = clif::Return::operands([payload]).build(ctx, loc);
    ctx.push_op(done, ret.op_ref());

    helper_function(
        ctx,
        loc,
        helper_symbol(Helper::New, descriptor),
        &params,
        vec![entry, head, body, done],
    )
}

/// Build `__tribute_array_splice_{idx}(source, start, end, count) -> payload`.
///
/// It traps unless `start ≤ end ≤ len`, allocates `len - (end - start) +
/// count` elements, and copies `source[0, start)` to the front and
/// `source[end, len)` to the back, retaining each copied element when the
/// elements are released. The `count` elements at `start` are left for the
/// caller to store.
fn array_splice_function(ctx: &mut IrContext, descriptor: ArrayDescriptor, loc: Location) -> OpRef {
    let i64_ty = core::I64::type_ref(ctx);
    let ptr_ty = core::Ptr::params().intern(ctx).as_type_ref();
    let params = [ptr_ty, i64_ty, i64_ty, i64_ty];
    let entry = new_block(ctx, loc, &params);
    let suffix = new_block(ctx, loc, &[]);
    let done = new_block(ctx, loc, &[]);
    let [source, start, end, count] = [0, 1, 2, 3].map(|i| ctx.block_arg(entry, i));

    let mut ops = Vec::new();
    let mut emit = Emitter { loc, ops: &mut ops };
    let len = emit.load(ctx, source, LEN_OFFSET, i64_ty);
    emit.trap_if(ctx, "ugt", start, end, SPLICE_OUT_OF_BOUNDS_TRAP);
    emit.trap_if(ctx, "ugt", end, len, SPLICE_OUT_OF_BOUNDS_TRAP);
    let removed = emit.isub(ctx, end, start);
    let kept = emit.isub(ctx, len, removed);
    let new_len = emit.iadd(ctx, kept, count, i64_ty);
    // `kept + count` wraps only past every representable length.
    emit.trap_if(ctx, "ult", new_len, kept, SIZE_OVERFLOW_TRAP);
    let payload = emit.allocate(ctx, new_len, descriptor);
    let zero = emit.i64(ctx, 0);

    let copy = |ctx: &mut IrContext, emit: &mut Emitter<'_>, from: ValueRef, to: ValueRef| {
        let element = emit.element_base(ctx, source, from, descriptor.stride);
        let element = emit.load(ctx, element, ELEMENTS_OFFSET, descriptor.element);
        let slot = emit.element_base(ctx, payload, to, descriptor.stride);
        emit.store(ctx, element, slot, ELEMENTS_OFFSET);
        if descriptor.released {
            emit.retain(ctx, element);
        }
    };

    let (enter_prefix, prefix_blocks) =
        counted_loop(ctx, loc, zero, start, suffix, |ctx, emit, index| {
            copy(ctx, emit, index, index)
        });
    ops.push(enter_prefix);
    fill_block(ctx, entry, ops);

    // Element `index` of the suffix moves by `count - removed`.
    let mut ops = Vec::new();
    let (enter_suffix, suffix_blocks) =
        counted_loop(ctx, loc, end, len, done, |ctx, emit, index| {
            let shifted = emit.isub(ctx, index, removed);
            let to = emit.iadd(ctx, shifted, count, i64_ty);
            copy(ctx, emit, index, to)
        });
    ops.push(enter_suffix);
    fill_block(ctx, suffix, ops);

    let ret = clif::Return::operands([payload]).build(ctx, loc);
    ctx.push_op(done, ret.op_ref());

    let mut blocks = vec![entry];
    blocks.extend(prefix_blocks);
    blocks.push(suffix);
    blocks.extend(suffix_blocks);
    blocks.push(done);
    helper_function(
        ctx,
        loc,
        helper_symbol(Helper::Splice, descriptor),
        &params,
        blocks,
    )
}

#[cfg(test)]
mod tests {
    use trunk_ir::context::IrContext;
    use trunk_ir::dialect::core;
    use trunk_ir::ops::DialectOp;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::pass::PassManager;
    use trunk_ir::printer::print_module;

    use crate::native::adt_rc_header::AdtRcHeader;
    use crate::native::clif_lowering::{CfToClif, FuncToClif};
    use crate::native::descriptor_to_clif::DescriptorToClif;
    use crate::native::ownership_lowering::LowerNativeOwnership;
    use crate::native::ownership_plan::NativeOwnershipPlanOptions;
    use crate::native::rtti::GenerateRtti;

    /// Run the native lowering through RC header lowering and print the
    /// functions whose names `keep` accepts.
    fn lower(ir: &str, keep: impl Fn(&str) -> bool) -> String {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, ir);
        let mut pm = PassManager::new();
        pm.with_debug_verifier();
        pm.add_pass(LowerNativeOwnership {
            options: NativeOwnershipPlanOptions::production(),
        })
        .add_pass(DescriptorToClif)
        .add_pass(FuncToClif)
        .add_pass(CfToClif)
        .add_pass(GenerateRtti)
        .add_pass(AdtRcHeader);
        let target = core::Module::from_op(&ctx, module.op()).expect("core.module");
        pm.run(&mut ctx, target, &mut Default::default())
            .expect("array lowering");

        let mut kept = String::new();
        let mut keeping = false;
        for line in print_module(&ctx, module.op()).lines() {
            if let Some(header) = line.strip_prefix("  clif.func {sym_name = \"") {
                keeping = keep(header.split('"').next().unwrap());
            }
            if keeping {
                kept.push_str(line);
                kept.push('\n');
            }
            if line == "  }" {
                keeping = false;
            }
        }
        kept
    }

    fn is_array_function(name: &str) -> bool {
        !name.starts_with("__tribute_") || name.starts_with("__tribute_array_")
    }

    #[test]
    fn managed_array_construction_allocates_and_retains_in_helpers() {
        let lowered = lower(
            r#"core.module @test {
  !Node = adt.struct<Node(value: core.i32)>
  !NodeRef = adt.typeref<{name = "Node"}>
  !Nodes = adt.array<!NodeRef>
  func.func @build(%a: !NodeRef, %b: !NodeRef, %at: core.index) -> !Nodes {
    %pair = adt.array_of %a, %b {type = !Nodes} : !Nodes
    %filled = adt.array_new %at, %a {type = !Nodes} : !Nodes
    %again = adt.array_new %at, %b {type = !Nodes} : !Nodes
    %spliced = adt.array_splice %pair, %at, %at, %b {type = !Nodes} : !Nodes
    func.return %spliced
  }
}"#,
            is_array_function,
        );
        // One helper of each kind serves every operation on the descriptor.
        assert_eq!(
            lowered.matches("sym_name = \"__tribute_array_new_").count(),
            1
        );
        insta::assert_snapshot!(lowered);
    }

    #[test]
    fn scalar_array_helpers_copy_without_retaining() {
        let lowered = lower(
            r#"core.module @test {
  !Ints = adt.array<core.i32>
  func.func @build(%value: core.i32, %count: core.index) -> !Ints {
    %filled = adt.array_new %count, %value {type = !Ints} : !Ints
    %zero = arith.const {value = 0} : core.index
    %trimmed = adt.array_splice %filled, %zero, %count {type = !Ints} : !Ints
    func.return %trimmed
  }
}"#,
            |name| name.starts_with("__tribute_array_"),
        );
        assert!(!lowered.contains("tribute_rt.retain"), "{lowered}");
        assert!(
            lowered.contains("clif.func_sig<(core.i64, core.i32) -> core.ptr>"),
            "{lowered}"
        );
        // An `i32` element is four bytes wide.
        assert!(
            lowered.contains("clif.iconst {value = 4} : core.i64"),
            "{lowered}"
        );
    }

    #[test]
    fn element_accesses_check_the_index_against_the_length() {
        let lowered = lower(
            r#"core.module @test {
  !Ints = adt.array<core.i32>
  func.func @access(%ints: !Ints, %at: core.index, %value: core.i32) -> core.i32 {
    adt.array_set %ints, %at, %value
    %read = adt.array_get %ints, %at : core.i32
    %len = adt.array_len %ints : core.index
    func.return %read
  }
}"#,
            |name| name == "access",
        );
        insta::assert_snapshot!(lowered);
    }
}
