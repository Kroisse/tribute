//! Lower closure operations in indirect calls.
//!
//! This pass transforms `func.call_indirect` operations when the callee
//! is a closure. The closure itself is the entry function's environment:
//!
//! Before:
//! ```text
//! %closure = closure.new %captures... {entries = [@lifted_func]}
//! %result = func.call_indirect %closure, %args...
//! ```
//!
//! After:
//! ```text
//! %closure = closure.new %captures... {entries = [@lifted_func]}
//! %entry = closure.entry %closure {index = 0}
//! %result = func.call_indirect %entry, %closure, %args...
//! ```
//!
//! `closure.new` then becomes one allocation of its convention table
//! reference and captures, and `closure.entry` a read of that reference and
//! of the table's entry.
//!
//! Uses `RewritePattern` + `PatternApplicator` for declarative transformation.

use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;
use std::ops::ControlFlow;
use std::sync::Arc;

use tribute_ir::calling_convention::{CallingConvention, ConventionBearing, DynConventionBearing};
use tribute_ir::dialect::adt;
use tribute_ir::dialect::closure;
use tribute_ir::dialect::tribute_rt;
use tribute_ir::runtime_layout;
use trunk_ir::analysis::AnalysisCache;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::core;
use trunk_ir::dialect::func;
use trunk_ir::op_interface::DynIndirectCallLike;
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::pass::{Pass, PassRunResult};
use trunk_ir::refs::{OpRef, TypeRef, ValueRef};
use trunk_ir::rewrite::{
    ConversionTarget, Module, PatternApplicator, PatternRewriter, RewritePattern, TypeConverter,
};
use trunk_ir::symbol_table::SymbolTable;
use trunk_ir::types::{Attribute, AttributeMap, TypeDataBuilder};
use trunk_ir::walk::{WalkAction, walk_op, walk_region};

/// The closure base: `{ entries: tribute_rt.convention_table_ref }`. Every
/// closure value has this storage
/// type, and a capture-free closure is allocated with it.
pub fn closure_struct_type_ref(ctx: &mut IrContext) -> TypeRef {
    runtime_layout::closure_base(ctx)
}

/// The closure layout holding `captures` after the base's convention table
/// field, with
/// the base as its `closure_base`. Equal capture types give one layout.
pub fn closure_captures_type_ref(ctx: &mut IrContext, captures: &[TypeRef]) -> TypeRef {
    let base = closure_struct_type_ref(ctx);
    closure_captures_layout(ctx, base, captures)
}

/// The closure base whose convention table field has the representation
/// `entries`.
pub(crate) fn closure_base_layout(ctx: &mut IrContext, entries: TypeRef) -> TypeRef {
    runtime_layout::closure_base_layout(ctx, entries)
}

/// The closure layout of `base` holding `captures` after the base's fields.
pub(crate) fn closure_captures_layout(
    ctx: &mut IrContext,
    base: TypeRef,
    captures: &[TypeRef],
) -> TypeRef {
    let mut fields = adt::Struct::from_type_ref(ctx, base)
        .expect("closure base is an adt.struct")
        .fields(ctx)
        .map(|(name, ty)| (name.to_owned(), ty))
        .collect::<Vec<_>>();
    fields.extend(
        captures
            .iter()
            .enumerate()
            .map(|(index, &ty)| (format!("_{index}"), ty)),
    );
    let mut attrs = AttributeMap::new();
    let layout = ctx.string_attr(runtime_layout::CLOSURE_CAPTURES);
    let mut view = attrs.view_mut(ctx);
    view.insert(runtime_layout::LAYOUT_ATTR, layout);
    view.insert(runtime_layout::CLOSURE_BASE_ATTR, Attribute::Type(base));
    adt::Struct::from_fields(ctx, "_closure_captures", fields, attrs).as_type_ref()
}

/// Whether `ty` is the closure base storage layout, identified by its
/// runtime layout attribute.
pub(crate) fn is_closure_struct_type_ref(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.get_type(ty);
    data.name.is(ctx, "adt.struct")
        && runtime_layout::has_runtime_layout(ctx, ty, runtime_layout::CLOSURE)
}

/// Whether `ty` is a closure storage layout, the base or a capture layout,
/// identified by its runtime layout attribute.
pub(crate) fn is_closure_layout(ctx: &IrContext, ty: TypeRef) -> bool {
    is_closure_struct_type_ref(ctx, ty)
        || (ctx.get_type(ty).name.is(ctx, "adt.struct")
            && runtime_layout::has_runtime_layout(ctx, ty, runtime_layout::CLOSURE_CAPTURES))
}

// ============================================================================
// Rewrite Patterns
// ============================================================================

/// Lower `closure.new` to `tribute_rt.convention_table` + `adt.struct_new`.
///
/// Every entry must name a function; a closure with an unresolved entry is
/// left for the boundary exit to reject. Until storage finalization,
/// remaining uses keep the semantic closure type through an unrealized cast
/// of the pack.
struct LowerClosureNewArena {
    functions: Arc<SymbolTable>,
}

impl RewritePattern for LowerClosureNewArena {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(closure_new) = closure::New::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let entries: Vec<_> = closure_new.entries(ctx).cloned().collect();
        let resolved = entries.iter().all(|entry| {
            self.functions
                .resolve(ctx, op, entry)
                .is_some_and(|target| func::Func::matches(ctx, target))
        });
        if !resolved {
            return false;
        }
        let captures = closure_new.captures(ctx).to_vec();
        let result_ty = ctx.op_result_types(op)[0];

        // %table = tribute_rt.convention_table {entries = [...]}
        let table_ty = tribute_rt::ConventionTableRef::params()
            .intern(ctx)
            .as_type_ref();
        let table_op = tribute_rt::ConventionTable::operands()
            .entries(entries)
            .results(table_ty)
            .build(ctx, loc);
        let table = table_op.result(ctx);
        rewriter.insert_op(table_op.op_ref());

        // %pack = adt.struct_new(%table, %captures...) : <closure layout>
        let struct_ty = if captures.is_empty() {
            closure_struct_type_ref(ctx)
        } else {
            let capture_types: Vec<_> = captures.iter().map(|&v| ctx.value_ty(v)).collect();
            closure_captures_type_ref(ctx, &capture_types)
        };
        let struct_new_op = adt::StructNew::operands(std::iter::once(table).chain(captures))
            .r#type(struct_ty)
            .results(struct_ty)
            .build(ctx, loc);
        // %closure = core.unrealized_conversion_cast %pack : closure.closure<...>
        let cast = core::UnrealizedConversionCast::operands(struct_new_op.result(ctx))
            .results(result_ty)
            .build(ctx, loc);

        rewriter.insert_op(struct_new_op.op_ref());
        rewriter.replace_op(cast.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "LowerClosureNewArena"
    }
}

/// Lower `closure.unpack` to a cast of the closure to its capture layout and
/// reads of the capture fields after the base's.
///
/// The cast keeps the closure's exact type as its source until storage
/// finalization makes it the base-to-capture-layout downcast that targets
/// materialize.
struct LowerClosureUnpackArena;

impl RewritePattern for LowerClosureUnpackArena {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(unpack) = closure::Unpack::from_op(ctx, op) else {
            return false;
        };
        let loc = ctx.op(op).location;
        let capture_types = ctx.op_result_types(op).to_vec();
        let layout = closure_captures_type_ref(ctx, &capture_types);
        let base = closure_struct_type_ref(ctx);
        let base_fields = adt::Struct::from_type_ref(ctx, base)
            .expect("closure base is an adt.struct")
            .field_count(ctx);

        let cast = core::UnrealizedConversionCast::operands(unpack.closure(ctx))
            .results(layout)
            .build(ctx, loc);
        rewriter.insert_op(cast.op_ref());
        let mut captures = Vec::with_capacity(capture_types.len());
        for (index, &ty) in capture_types.iter().enumerate() {
            let get = adt::StructGet::operands(cast.result(ctx))
                .r#type(layout)
                .field((base_fields + index) as u32)
                .results(ty)
                .build(ctx, loc);
            rewriter.insert_op(get.op_ref());
            captures.push(get.result(ctx));
        }
        rewriter.erase_op(captures);
        true
    }

    fn name(&self) -> &'static str {
        "LowerClosureUnpackArena"
    }
}

/// Lower `func.call_indirect` on closure values.
struct LowerClosureCallArena;

impl RewritePattern for LowerClosureCallArena {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if func::CallIndirect::from_op(ctx, op).is_err() {
            return false;
        }

        let operands = ctx.op_operands(op);
        if operands.is_empty() {
            return false;
        }
        let callee = operands[0];
        // Only a closure carries an environment; a `func.func_sig` callee
        // is a plain function pointer.
        if physical_closure_type_for_callee(ctx, callee).is_none() {
            return false;
        }

        let loc = ctx.op(op).location;
        let args: Vec<ValueRef> = operands[1..].to_vec();
        // An ordinary closure call has one result, or none when a value
        // delimiter enters its Cps body.
        let caller_results = ctx.op_result_types(op).to_vec();

        let i32_ty = TypeDataBuilder::new("core.i32").intern(ctx);
        let Some(convention) = DynConventionBearing::of(ctx, op) else {
            return false;
        };
        if caller_results.len() != usize::from(convention != CallingConvention::Cps) {
            return false;
        }
        let Some(contract) =
            exact_physical_call_contract(ctx, op, callee, convention, &args, &caller_results)
        else {
            return false;
        };

        // Generate: %table_idx = closure.entry %closure {index = <slot>}
        let table_idx_op = closure::Entry::operands(callee)
            .index(convention.slot())
            .results(i32_ty)
            .build(ctx, loc);
        let table_idx = ctx.op_result(table_idx_op.op_ref(), 0);

        let mut new_args = args;
        for &(index, expected) in &contract.argument_casts {
            let cast = core::UnrealizedConversionCast::operands(new_args[index])
                .results(expected)
                .build(ctx, loc);
            rewriter.insert_op(cast.op_ref());
            new_args[index] = cast.result(ctx);
        }
        new_args.insert(contract.environment_index, callee);
        let new_call = func::CallIndirect::operands(table_idx, new_args)
            .signature(contract.signature)
            .build(ctx, loc);
        copy_indirect_call_attributes(ctx, op, new_call);

        rewriter.insert_op(table_idx_op.op_ref());

        rewriter.replace_op(new_call.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "LowerClosureCallArena"
    }
}

/// Lower a closure-valued proper tail transfer: a Cps
/// transfer or a source `become` between value-returning callables. Untagged
/// ordinary direct calls are handled separately.
struct LowerClosureTailCallArena;

impl RewritePattern for LowerClosureTailCallArena {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if func::TailCallIndirect::from_op(ctx, op).is_err() {
            return false;
        }
        let operands = ctx.op_operands(op).to_vec();
        let Some((&callee, args)) = operands.split_first() else {
            return false;
        };
        let Some(convention) = DynConventionBearing::of(ctx, op) else {
            return false;
        };
        if physical_closure_type_for_callee(ctx, callee).is_none() {
            return false;
        }

        let location = ctx.op(op).location;
        let i32_ty = TypeDataBuilder::new("core.i32").intern(ctx);
        let Some(results) = exact_tail_results(ctx, op, convention) else {
            return false;
        };
        let Some(contract) =
            exact_physical_call_contract(ctx, op, callee, convention, args, &results)
        else {
            return false;
        };
        let func_ref = closure::Entry::operands(callee)
            .index(convention.slot())
            .results(i32_ty)
            .build(ctx, location);
        let mut args = args.to_vec();
        for (index, expected) in contract.argument_casts {
            let cast = core::UnrealizedConversionCast::operands(args[index])
                .results(expected)
                .build(ctx, location);
            rewriter.insert_op(cast.op_ref());
            args[index] = cast.result(ctx);
        }
        args.insert(contract.environment_index, callee);
        let tail = func::TailCallIndirect::operands(func_ref.result(ctx), args)
            .signature(contract.signature)
            .build(ctx, location);
        copy_indirect_call_attributes(ctx, op, tail);

        rewriter.insert_op(func_ref.op_ref());
        rewriter.replace_op(tail.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "LowerClosureTailCallArena"
    }
}

/// Copy source metadata without replacing the physical indirect-call contract.
///
/// The semantic calling convention is consumed here: the lowered call's
/// physical signature carries the contract it was validated against.
fn copy_indirect_call_attributes(
    ctx: &mut IrContext,
    source: OpRef,
    destination: impl ConventionBearing,
) {
    let mut attributes = ctx.op(source).attributes.clone();
    func::remove_indirect_call_signature(ctx, &mut attributes);
    ctx.op_attrs_mut(destination.op_ref()).extend(attributes);
    destination.remove_calling_convention(ctx);
}

pub(crate) fn physical_closure_type_for_callee(
    ctx: &IrContext,
    callee: ValueRef,
) -> Option<TypeRef> {
    let ty = ctx.value_ty(callee);
    closure::Closure::matches(ctx, ty).then_some(ty)
}

/// The results of a closure tail transfer, when its entry's results are the
/// enclosing function's: a Cps transfer leaves a Cps function, and a value
/// transfer (source `become`) returns the callee's result as its own.
fn exact_tail_results(
    ctx: &IrContext,
    op: OpRef,
    convention: CallingConvention,
) -> Option<Vec<TypeRef>> {
    let exact = DynIndirectCallLike::get(ctx, op).and_then(|call| call.exact_signature(ctx))?;
    let results = func::FuncSig::from_type_ref(ctx, exact)?.results(ctx);
    let mut parent = ctx.op(op).parent_block?;
    loop {
        let owner = ctx.region(ctx.block(parent).parent_region?).parent_op?;
        if let Ok(function) = func::Func::from_op(ctx, owner) {
            let caller = func::FuncSig::from_type_ref(ctx, function.r#type(ctx))?;
            let caller_is_cps =
                DynConventionBearing::of(ctx, owner) == Some(CallingConvention::Cps);
            return ((convention == CallingConvention::Cps) == caller_is_cps
                && caller.results(ctx) == results)
                .then(|| results.to_vec());
        }
        parent = ctx.op(owner).parent_block?;
    }
}

struct PhysicalCallContract {
    environment_index: usize,
    signature: TypeRef,
    argument_casts: Vec<(usize, TypeRef)>,
}

/// The physical contract of the transfer `op` into the closure `callee`,
/// whose environment is the closure itself. The transfer's exact signature
/// is the entry ABI; the closure type's callable is not.
fn exact_physical_call_contract(
    ctx: &mut IrContext,
    op: OpRef,
    callee: ValueRef,
    convention: CallingConvention,
    args: &[ValueRef],
    results: &[TypeRef],
) -> Option<PhysicalCallContract> {
    let closure_ty = physical_closure_type_for_callee(ctx, callee)?;
    let environment = closure_ty;
    let function = DynIndirectCallLike::get(ctx, op).and_then(|call| call.exact_signature(ctx))?;
    let callable = func::FuncSig::from_type_ref(ctx, function)?;
    if callable.results(ctx) != results || callable.inputs(ctx).len() != args.len() {
        return None;
    }
    let mut casts = Vec::new();
    for (index, (argument, expected)) in args.iter().zip(callable.inputs(ctx)).enumerate() {
        let actual = ctx.value_ty(*argument);
        if actual != *expected {
            if !is_closure_struct_type_ref(ctx, actual) && !closure::Closure::matches(ctx, actual) {
                return None;
            }
            if closure::Closure::matches(ctx, *expected) {
                if physical_closure_type_for_callee(ctx, *argument) != Some(*expected) {
                    return None;
                }
            } else if *expected != environment {
                return None;
            }
            casts.push((index, *expected));
        }
    }
    let environment_index = convention.closure_environment_index();
    if environment_index > args.len() {
        return None;
    }
    let environment_attrs = crate::target_abi::physical_parameter_attrs(ctx, convention);
    let signature = callable.rebuild(ctx, |inputs, _| {
        inputs.insert(environment_index, (environment, environment_attrs));
    });
    Some(PhysicalCallContract {
        environment_index,
        signature: signature.as_type_ref(),
        argument_casts: casts,
    })
}

/// Lower `closure.entry` to a read of the base's convention table field
/// (`adt.struct_get` field 0) and `tribute_rt.convention_entry`.
struct LowerClosureEntryArena;

impl RewritePattern for LowerClosureEntryArena {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(entry) = closure::Entry::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let index = entry.index(ctx);
        let closure_value = entry.closure(ctx);
        let i32_ty = TypeDataBuilder::new("core.i32").intern(ctx);
        let table_ty = tribute_rt::ConventionTableRef::params()
            .intern(ctx)
            .as_type_ref();
        let struct_ty = closure_struct_type_ref(ctx);

        let get_op = adt::StructGet::operands(closure_value)
            .r#type(struct_ty)
            .field(0)
            .results(table_ty)
            .build(ctx, loc);
        let read = tribute_rt::ConventionEntry::operands(get_op.result(ctx))
            .index(index)
            .results(i32_ty)
            .build(ctx, loc);
        rewriter.insert_op(get_op.op_ref());
        rewriter.replace_op(read.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "LowerClosureEntryArena"
    }
}

fn tagged_closure_transfers_are_legal(ctx: &mut IrContext, func_op: func::Func) -> bool {
    let Some(body) = ctx.op_region(func_op.op_ref(), 0) else {
        return true;
    };
    let mut transfers = Vec::new();
    let _ = walk_region::<()>(ctx, body, &mut |op| {
        if func::Func::matches(ctx, op) {
            return ControlFlow::Continue(WalkAction::Skip);
        }
        if let Some(call) = DynIndirectCallLike::get(ctx, op) {
            transfers.push(call);
        }
        ControlFlow::Continue(WalkAction::Advance)
    });
    if transfers.is_empty() {
        return true;
    }

    transfers.into_iter().all(|call| {
        let op = call.op();
        let callee = call.indirect_callee(ctx);
        let args = call.indirect_arguments(ctx).map(<[_]>::to_vec);
        let (Some(callee), Some(args)) = (callee, args) else {
            return false;
        };
        if physical_closure_type_for_callee(ctx, callee).is_none() {
            return true;
        }
        let Some(convention) = DynConventionBearing::of(ctx, op) else {
            return false;
        };
        if func::CallIndirect::matches(ctx, op) {
            // A value delimiter enters its Cps body by an ordinary call
            // with no result; every other closure call has one.
            let results = ctx.op_result_types(op).to_vec();
            let expected = usize::from(convention != CallingConvention::Cps);
            results.len() == expected
                && exact_physical_call_contract(ctx, op, callee, convention, &args, &results)
                    .is_some()
        } else {
            let Some(results) = exact_tail_results(ctx, op, convention) else {
                return false;
            };
            exact_physical_call_contract(ctx, op, callee, convention, &args, &results).is_some()
        }
    })
}

/// Lower every already-prepared function body in a module to closure storage.
///
/// Each batch discovers the functions not yet lowered, validates all of them
/// before rewriting any body, and then lowers the whole batch against one
/// symbol table. The module is rescanned after each batch so functions
/// introduced by an earlier transformation are lowered once as well.
/// Processing each function operation at most once keeps this traversal
/// bounded without relying on function names or target-specific pipeline
/// ordering.
pub fn lower_prepared_closures(ctx: &mut IrContext, module: Module) -> PassRunResult {
    let mut lowered = HashSet::default();

    loop {
        let mut discovered = Vec::new();
        let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
            if let Ok(func_op) = func::Func::from_op(ctx, op)
                && lowered.insert(op)
            {
                discovered.push(func_op);
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
        if discovered.is_empty() {
            break;
        }
        // Rebuilt per batch so functions introduced by lowering resolve too.
        let functions = Arc::new(SymbolTable::collect(ctx, module));

        // Validate the batch as a whole before rewriting any body.
        for &function in &discovered {
            validate_closure_transfers(ctx, function)?;
            validate_closure_constructions(ctx, function, &functions)?;
        }
        for func_op in discovered.into_iter().rev() {
            rewrite_validated_closures_in_func(ctx, func_op, functions.clone());
            // Closure lowering is the last reader of a function's convention.
            func_op.remove_calling_convention(ctx);
        }
    }
    Ok(())
}

fn validate_closure_transfers(ctx: &mut IrContext, func_op: func::Func) -> PassRunResult {
    if tagged_closure_transfers_are_legal(ctx, func_op) {
        Ok(())
    } else {
        Err("closure lowering: exact caller/callee/indirect result contract mismatch".into())
    }
}

/// Check each `closure.new` in `func_op` against its entry functions: each
/// entry function's environment parameter has the closure's exact type, and
/// its `closure.unpack` yields exactly the captured types. A capture-free
/// entry function has no `closure.unpack`.
fn validate_closure_constructions(
    ctx: &IrContext,
    func_op: func::Func,
    functions: &SymbolTable,
) -> PassRunResult {
    let Some(body) = ctx.op_region(func_op.op_ref(), 0) else {
        return Ok(());
    };
    let mut constructions = Vec::new();
    let _ = walk_region::<()>(ctx, body, &mut |op| {
        if func::Func::matches(ctx, op) {
            return ControlFlow::Continue(WalkAction::Skip);
        }
        if let Ok(new) = closure::New::from_op(ctx, op) {
            constructions.push(new);
        }
        ControlFlow::Continue(WalkAction::Advance)
    });
    for new in constructions {
        let closure_ty = ctx.value_ty(new.result(ctx));
        let captured: Vec<_> = new.captures(ctx).iter().map(|&v| ctx.value_ty(v)).collect();
        let entries: Vec<_> = new.entries(ctx).collect();
        if entries.len() != CallingConvention::ALL.len() {
            return Err("closure lowering: closure.new must name one entry per convention".into());
        }
        for (entry, slot) in entries.into_iter().zip(CallingConvention::ALL) {
            // An unresolved entry function is left for the boundary exit to reject.
            let Some(target) = functions
                .resolve(ctx, new.op_ref(), entry)
                .and_then(|target| func::Func::from_op(ctx, target).ok())
            else {
                continue;
            };
            if DynConventionBearing::of(ctx, target.op_ref()) != Some(slot) {
                return Err(
                    "closure lowering: entry function convention differs from its slot".into(),
                );
            }
            validate_entry_function(
                ctx,
                target,
                closure_ty,
                slot.closure_environment_index(),
                &captured,
            )?;
        }
    }
    Ok(())
}

/// Check one entry function of a closure of type `closure_ty` holding
/// captures of the types `captured`, whose environment is at
/// `environment_index`.
fn validate_entry_function(
    ctx: &IrContext,
    target: func::Func,
    closure_ty: TypeRef,
    environment_index: usize,
    captured: &[TypeRef],
) -> PassRunResult {
    let environment = func::FuncSig::from_type_ref(ctx, target.r#type(ctx))
        .and_then(|signature| signature.inputs(ctx).get(environment_index).copied());
    if environment != Some(closure_ty) {
        return Err(
            "closure lowering: entry function environment differs from its closure type".into(),
        );
    }
    let mut unpacked = Vec::new();
    if let Some(code_body) = ctx.op_region(target.op_ref(), 0) {
        let _ = walk_region::<()>(ctx, code_body, &mut |op| {
            if func::Func::matches(ctx, op) {
                return ControlFlow::Continue(WalkAction::Skip);
            }
            if closure::Unpack::matches(ctx, op) {
                unpacked.push(ctx.op_result_types(op).to_vec());
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
    }
    let consistent = if captured.is_empty() {
        unpacked.is_empty()
    } else {
        unpacked.iter().all(|types| types == captured)
    };
    if !consistent {
        return Err(
            "closure lowering: closure.unpack differs from the captures of closure.new".into(),
        );
    }
    Ok(())
}

fn rewrite_validated_closures_in_func(
    ctx: &mut IrContext,
    func_op: func::Func,
    functions: Arc<SymbolTable>,
) {
    if !ctx.op_has_regions(func_op.op_ref()) {
        return;
    }
    let applicator = PatternApplicator::new(TypeConverter::new())
        .with_target(
            ConversionTarget::new()
                .legal_op("func.func")
                .recursive_legal_op("func.func"),
        )
        .add_pattern(LowerClosureCallArena)
        .add_pattern(LowerClosureTailCallArena)
        .add_pattern(LowerClosureNewArena { functions })
        .add_pattern(LowerClosureEntryArena)
        .add_pattern(LowerClosureUnpackArena);
    applicator.apply_partial(ctx, func_op);
}

/// Select the canonical `_closure` storage type after all semantic consumers
/// have validated the exact callable type.
///
/// The plan covers aliases, nested type parameters, type-bearing operation
/// attributes, results, and block arguments before applying any update. The
/// casts that kept lowered closure packs at their semantic closure type become
/// identities and are removed.
pub(crate) fn finalize_closure_storage_layout(ctx: &mut IrContext, module: Module) {
    let closure_struct = closure_struct_type_ref(ctx);
    substitute_module_types(ctx, module, move |ctx, ty| {
        closure::Closure::matches(ctx, ty).then_some(closure_struct)
    });
}

/// Select the canonical `_closure` storage type after all semantic consumers
/// have validated the exact callable type.
///
/// The plan covers aliases, nested type parameters, type-bearing operation
/// attributes, results, and block arguments before applying any update. The
/// casts that kept lowered closure packs at their semantic closure type become
/// identities and are removed.
pub struct FinalizeClosureStorageLayout;

impl Pass for FinalizeClosureStorageLayout {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "finalize-closure-storage-layout"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: core::Module,
        _analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        finalize_closure_storage_layout(ctx, target.into());
        Ok(())
    }
}

/// Replace every type `substitute` maps, wherever it occurs in `module`:
/// aliases, nested type parameters and attributes, type-bearing operation
/// attributes, results, and block arguments. The plan is complete before any
/// update applies. Casts that the replacement turns into identities are
/// removed.
pub(crate) fn substitute_module_types(
    ctx: &mut IrContext,
    module: Module,
    substitute: impl FnMut(&mut IrContext, TypeRef) -> Option<TypeRef>,
) {
    substitute_module_types_keeping_casts(ctx, module, substitute);
    erase_identity_casts(ctx, module);
}

/// [`substitute_module_types`] without removing the casts the replacement
/// turns into identities.
pub(crate) fn substitute_module_types_keeping_casts(
    ctx: &mut IrContext,
    module: Module,
    substitute: impl FnMut(&mut IrContext, TypeRef) -> Option<TypeRef>,
) {
    let ops = collect_ops(ctx, module.op());
    let aliases = ctx.type_aliases().to_vec();
    let mut physicalizer = TypeSubstitution::new(ctx, substitute);
    let mut alias_updates = Vec::new();
    let mut attribute_updates = Vec::new();
    let mut result_updates = Vec::new();
    let mut block_arg_updates = Vec::new();
    let mut block_attribute_updates = Vec::new();

    for (name, ty) in aliases {
        let converted = physicalizer.convert_type(ty);
        if converted != ty {
            alias_updates.push((name, converted));
        }
    }
    for op in ops {
        for (name, value) in physicalizer.ctx.op(op).attributes.clone() {
            let converted = physicalizer.convert_attribute(value.clone());
            if converted != value {
                attribute_updates.push((op, name, converted));
            }
        }
        for (index, ty) in physicalizer
            .ctx
            .op_result_types(op)
            .to_vec()
            .into_iter()
            .enumerate()
        {
            let converted = physicalizer.convert_type(ty);
            if converted != ty {
                result_updates.push((op, index as u32, converted));
            }
        }
        for region in physicalizer
            .ctx
            .op_regions(op)
            .collect::<trunk_ir::RegionList>()
        {
            for block in physicalizer.ctx.region(region).blocks.clone() {
                let args = physicalizer.ctx.block(block).args.to_vec();
                for (index, argument) in args.into_iter().enumerate() {
                    let mut attrs = argument.attrs.clone();
                    for (name, value) in argument.attrs.iter() {
                        let value = physicalizer.convert_attribute(value.clone());
                        attrs.view_mut(physicalizer.ctx).insert(name, value);
                    }
                    if attrs != argument.attrs {
                        block_attribute_updates.push((block, index, attrs));
                    }
                    let ty = argument.ty;
                    let converted = physicalizer.convert_type(ty);
                    if converted != ty {
                        block_arg_updates.push((block, index as u32, converted));
                    }
                }
            }
        }
    }

    drop(physicalizer);
    for (name, ty) in alias_updates {
        ctx.register_type_alias(name, ty);
    }
    for (op, name, value) in attribute_updates {
        ctx.op_attrs_mut(op).insert(name, value);
    }
    for (op, index, ty) in result_updates {
        ctx.set_op_result_type(op, index, ty);
    }
    for (block, index, ty) in block_arg_updates {
        ctx.set_block_arg_type(block, index, ty);
    }
    for (block, index, attrs) in block_attribute_updates {
        ctx.block_mut(block).args[index].attrs = attrs;
    }
}

/// Remove unrealized casts whose source already has the declared type.
fn erase_identity_casts(ctx: &mut IrContext, module: Module) {
    for op in collect_ops(ctx, module.op()) {
        let Ok(cast) = core::UnrealizedConversionCast::from_op(ctx, op) else {
            continue;
        };
        let (input, result) = (cast.value(ctx), cast.result(ctx));
        if ctx.value_ty(input) == ctx.value_ty(result) {
            ctx.replace_all_uses(result, input);
            trunk_ir::rewrite::erase_op(ctx, op);
        }
    }
}

pub(crate) struct TypeSubstitution<'a, F> {
    ctx: &'a mut IrContext,
    substitute: F,
    cache: HashMap<TypeRef, TypeRef>,
    visiting: HashSet<TypeRef>,
}

impl<'a, F: FnMut(&mut IrContext, TypeRef) -> Option<TypeRef>> TypeSubstitution<'a, F> {
    pub(crate) fn new(ctx: &'a mut IrContext, substitute: F) -> Self {
        Self {
            ctx,
            substitute,
            cache: HashMap::default(),
            visiting: HashSet::default(),
        }
    }

    pub(crate) fn convert_type(&mut self, ty: TypeRef) -> TypeRef {
        if let Some(&converted) = self.cache.get(&ty) {
            return converted;
        }
        if let Some(replacement) = (self.substitute)(self.ctx, ty) {
            self.cache.insert(ty, replacement);
            return replacement;
        }
        if !self.visiting.insert(ty) {
            return ty;
        }

        let data = self.ctx.get_type(ty).clone();
        let mut converted = data.clone();
        for parameter in &mut converted.params {
            *parameter = self.convert_type(*parameter);
        }
        let attributes: Vec<_> = data
            .attrs
            .iter()
            .map(|(name, value)| (name, self.convert_attribute(value.clone())))
            .collect();
        converted.attrs.clear();
        converted.attrs.view_mut(self.ctx).extend(attributes);
        let converted = if converted == data {
            ty
        } else {
            self.ctx.intern_type(converted)
        };
        self.visiting.remove(&ty);
        self.cache.insert(ty, converted);
        converted
    }

    fn convert_attribute(&mut self, attribute: Attribute) -> Attribute {
        attribute.map_types(|ty| self.convert_type(ty))
    }
}

fn collect_ops(ctx: &IrContext, root: OpRef) -> Vec<OpRef> {
    let mut ops = Vec::new();
    let _ = walk_op::<()>(ctx, root, &mut |op| {
        ops.push(op);
        ControlFlow::Continue(WalkAction::Advance)
    });
    ops
}

/// PassManager-friendly closure lowering for a module prepared earlier in the
/// pipeline.
pub struct LowerPreparedClosures;

impl Pass for LowerPreparedClosures {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "lower-prepared-closures"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: core::Module,
        _analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        lower_prepared_closures(ctx, target.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;
    use trunk_ir::walk::{WalkAction, walk_op};

    #[test]
    fn closure_layout_is_identified_by_its_layout_attribute_only() {
        let mut ctx = IrContext::new();
        parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !Named = adt.struct<_closure(func_ptr: core.i32, env: tribute_rt.anyref)>
  !Layout = adt.struct<Other(code: core.i32), {layout = "closure"}>
}"#,
        );
        let alias = |ctx: &IrContext, name: &str| {
            ctx.type_aliases()
                .iter()
                .find_map(|(alias, ty)| (*alias == name).then_some(*ty))
                .unwrap()
        };
        let named = alias(&ctx, "Named");
        let layout = alias(&ctx, "Layout");

        assert!(!is_closure_struct_type_ref(&ctx, named));
        assert!(is_closure_struct_type_ref(&ctx, layout));
        let canonical = closure_struct_type_ref(&mut ctx);
        assert!(is_closure_struct_type_ref(&ctx, canonical));
        assert_ne!(canonical, named);
    }

    #[test]
    fn closure_layout_identifier_round_trips_through_textual_ir() {
        let mut ctx = IrContext::new();
        let canonical = closure_struct_type_ref(&mut ctx);
        let printed = format!(
            "core.module @test {{\n  !C = {}\n}}",
            trunk_ir::printer::print_type(&ctx, canonical)
        );
        assert!(printed.contains("layout = \"closure\""), "{printed}");

        let mut reparsed = IrContext::new();
        parse_test_module(&mut reparsed, &printed);
        let reparsed_ty = reparsed
            .type_aliases()
            .iter()
            .find_map(|(alias, ty)| (*alias == "C").then_some(*ty))
            .unwrap();
        assert_eq!(reparsed_ty, closure_struct_type_ref(&mut reparsed));
    }

    fn evidence_type_str() -> &'static str {
        "adt.array<adt.typeref<{name = \"std::runtime::evidence::Marker\"}>>"
    }

    fn closure_test_module(ctx: &mut IrContext) -> Module {
        let ev_ty = evidence_type_str();
        parse_test_module(
            ctx,
            &format!(
                r#"core.module @test {{
  !closure = closure.closure<func.func_sig<({ev_ty}, tribute_rt.anyref) -> tribute_rt.anyref>>

  func.func @callee(%ev: {ev_ty}, %env: !closure, %arg: tribute_rt.anyref) -> tribute_rt.anyref attributes {{tribute.calling_convention = "evidence_direct"}} {{
      func.return %arg
  }}
  func.func @trap_direct(%env: !closure, %ev: {ev_ty}, %arg: tribute_rt.anyref) -> tribute_rt.anyref attributes {{tribute.calling_convention = "direct"}} {{
      func.unreachable
  }}
  func.func @trap_cps(%ev: {ev_ty}, %env: !closure, %arg: tribute_rt.anyref) -> core.never attributes {{tribute.calling_convention = "cps"}} {{
      func.unreachable
  }}

  func.func @selected(%ev: {ev_ty}, %payload: tribute_rt.anyref) -> tribute_rt.anyref {{
      %closure = closure.new {{entries = [@trap_direct, @callee, @trap_cps]}} : !closure
      %result = func.call_indirect %closure, %ev, %payload {{tribute.calling_convention = "evidence_direct", signature = func.func_sig<({ev_ty}, tribute_rt.anyref) -> tribute_rt.anyref>}} : tribute_rt.anyref
      func.return %result
  }}

  func.func @untouched(%ev: {ev_ty}, %payload: tribute_rt.anyref) -> tribute_rt.anyref {{
      %closure = closure.new {{entries = [@trap_direct, @callee, @trap_cps]}} : !closure
      %result = func.call_indirect %closure, %ev, %payload {{tribute.calling_convention = "evidence_direct", signature = func.func_sig<({ev_ty}, tribute_rt.anyref) -> tribute_rt.anyref>}} : tribute_rt.anyref
      func.return %result
  }}
}}"#
            ),
        )
    }

    fn func_by_name(ctx: &IrContext, module: Module, name: &'static str) -> func::Func {
        module
            .ops(ctx)
            .iter()
            .copied()
            .filter_map(|op| func::Func::from_op(ctx, op).ok())
            .find(|func_op| func_op.sym_name(ctx) == name)
            .expect("test function should exist")
    }

    fn func_by_name_recursive(ctx: &IrContext, module: Module, name: &'static str) -> func::Func {
        let mut found = None;
        let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
            if let Ok(func_op) = func::Func::from_op(ctx, op)
                && func_op.sym_name(ctx) == name
            {
                found = Some(func_op);
                return ControlFlow::Break(());
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
        found.expect("test function should exist")
    }

    fn call_indirect_operands_in_func(ctx: &IrContext, func_op: func::Func) -> Vec<Vec<ValueRef>> {
        let mut calls = Vec::new();
        for &block in &ctx.region(func_op.body(ctx)).blocks {
            for &op in &ctx.block(block).ops {
                if func::CallIndirect::from_op(ctx, op).is_ok() {
                    calls.push(ctx.op_operands(op).to_vec());
                }
            }
        }
        calls
    }

    fn entry_evidence_arg(ctx: &IrContext, func_op: func::Func) -> ValueRef {
        let entry = ctx.region(func_op.body(ctx)).blocks[0];
        ctx.block_args(entry)[0]
    }

    fn nested_closure_test_module(ctx: &mut IrContext) -> Module {
        let ev_ty = evidence_type_str();
        parse_test_module(
            ctx,
            &format!(
                r#"core.module @test {{
  !closure = closure.closure<func.func_sig<({ev_ty}, tribute_rt.anyref) -> tribute_rt.anyref>>

  func.func @callee(%ev: {ev_ty}, %env: !closure, %arg: tribute_rt.anyref) -> tribute_rt.anyref attributes {{tribute.calling_convention = "evidence_direct"}} {{
      func.return %arg
  }}
  func.func @trap_direct(%env: !closure, %ev: {ev_ty}, %arg: tribute_rt.anyref) -> tribute_rt.anyref attributes {{tribute.calling_convention = "direct"}} {{
      func.unreachable
  }}
  func.func @trap_cps(%ev: {ev_ty}, %env: !closure, %arg: tribute_rt.anyref) -> core.never attributes {{tribute.calling_convention = "cps"}} {{
      func.unreachable
  }}

  func.func @outer(%outer_ev: {ev_ty}, %payload: tribute_rt.anyref) -> tribute_rt.anyref {{
      func.func @inner(%inner_ev: {ev_ty}, %inner_payload: tribute_rt.anyref) -> tribute_rt.anyref {{
          %closure = closure.new {{entries = [@trap_direct, @callee, @trap_cps]}} : !closure
          %result = func.call_indirect %closure, %inner_ev, %inner_payload {{tribute.calling_convention = "evidence_direct", signature = func.func_sig<({ev_ty}, tribute_rt.anyref) -> tribute_rt.anyref>}} : tribute_rt.anyref
          func.return %result
      }}
      func.return %payload
  }}
}}"#
            ),
        )
    }

    fn assert_module_is_structurally_valid(ctx: &IrContext, module: Module) {
        let use_chains = trunk_ir::validation::validate_use_chains(ctx, module);
        assert!(
            use_chains.is_ok(),
            "invalid use chains after lowering:\n{use_chains}"
        );

        let operation_verifiers = trunk_ir::validation::validate_operation_verifiers(ctx, module);
        assert!(
            operation_verifiers.is_ok(),
            "invalid operations after lowering:\n{operation_verifiers}"
        );
    }

    #[test]
    fn module_entrypoint_still_prepares_and_lowers_all_functions() {
        let mut ctx = IrContext::new();
        let module = closure_test_module(&mut ctx);

        lower_prepared_closures(&mut ctx, module).unwrap();

        let ir = print_module(&ctx, module.op());
        assert!(
            !ir.contains("closure.new"),
            "module entrypoint should lower closure.new:\n{ir}"
        );
        assert!(
            !ir.contains("closure.func") && !ir.contains("closure.unpack"),
            "module entrypoint should lower closure accessors:\n{ir}"
        );
        assert_eq!(
            ir.matches("signature").count(),
            2,
            "each lowered indirect call must retain an exact signature:\n{ir}"
        );

        for name in ["selected", "untouched"] {
            let func_op = func_by_name(&ctx, module, name);
            let calls = call_indirect_operands_in_func(&ctx, func_op);
            assert_eq!(
                calls.len(),
                1,
                "{name} should have one lowered indirect call"
            );
            assert_eq!(
                calls[0][1],
                entry_evidence_arg(&ctx, func_op),
                "{name} should pass the enclosing function's evidence argument immediately after table index"
            );
        }
    }

    #[test]
    fn module_entrypoint_lowers_nested_function_closures() {
        let mut ctx = IrContext::new();
        let module = nested_closure_test_module(&mut ctx);

        lower_prepared_closures(&mut ctx, module).unwrap();
        assert_module_is_structurally_valid(&ctx, module);

        let inner = func_by_name_recursive(&ctx, module, "inner");
        let inner_ir = print_module(&ctx, inner.op_ref());
        assert!(
            !inner_ir.contains("closure.new"),
            "module lowering must revisit nested functions:\n{inner_ir}"
        );
        assert!(
            !inner_ir.contains("closure.func") && !inner_ir.contains("closure.unpack"),
            "nested closure accessors must be fully lowered:\n{inner_ir}"
        );
        let inner_calls = call_indirect_operands_in_func(&ctx, inner);
        assert_eq!(inner_calls.len(), 1);
        assert_eq!(
            inner_calls[0][1],
            entry_evidence_arg(&ctx, inner),
            "nested function must use its own evidence argument"
        );
    }

    #[test]
    fn prepared_module_rejects_invalid_transfer_before_rewriting_other_functions() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
            !Cps = closure.closure<func.func_sig<(core.i32) -> core.never>>
            func.func @invalid(%callee: !Cps, %value: core.i32) -> core.never attributes {tribute.calling_convention = "cps"} {
                func.tail_call_indirect %callee, %value
            }
            func.func @otherwise_lowerable(%callee: !Cps) -> core.i32 {
                %code = closure.func %callee : core.i32
                func.return %code
            }
        }"#,
        );
        let before = print_module(&ctx, module.op());
        let ops = collect_ops(&ctx, module.op());
        assert!(lower_prepared_closures(&mut ctx, module).is_err());
        assert_eq!(print_module(&ctx, module.op()), before);
        assert_eq!(collect_ops(&ctx, module.op()), ops);
    }

    #[test]
    fn ordinary_closure_call_result_counts_are_checked_before_mutation() {
        for (signature_results, call, supported) in [
            (
                "core.i32",
                "func.call_indirect %callee {tribute.calling_convention = \"direct\"}",
                false,
            ),
            (
                "core.i32",
                "%first, %extra = func.call_indirect %callee {tribute.calling_convention = \"direct\"} : core.i32, core.i32",
                false,
            ),
            (
                "()",
                "func.call_indirect %callee {tribute.calling_convention = \"direct\"}",
                false,
            ),
            (
                "core.i32",
                "%result = func.call_indirect %callee {tribute.calling_convention = \"direct\"} : core.i32",
                true,
            ),
        ] {
            // The transfer carries its entry signature; the closure type's
            // callable is not one.
            let call = call.replace("{tribute", "{signature = !Entry, tribute");
            let mut ctx = IrContext::new();
            let module = parse_test_module(
                &mut ctx,
                &format!(
                    r#"core.module @test {{
            !Entry = func.func_sig<() -> {signature_results}>
            !Callback = closure.closure<func.func_sig<() -> core.i32>>
            func.func @invalid(%callee: !Callback) -> core.i32 {{
                {call}
                %zero = arith.constant {{value = 0}} : core.i32
                func.return %zero
            }}
            func.func @otherwise_lowerable(%callee: !Callback) -> core.i32 {{
                %code = closure.func %callee : core.i32
                func.return %code
            }}
        }}"#,
                ),
            );
            let before = print_module(&ctx, module.op());
            let ops = collect_ops(&ctx, module.op());
            let invalid = func_by_name(&ctx, module, "invalid");
            if supported {
                lower_prepared_closures(&mut ctx, module).unwrap();
                assert_module_is_structurally_valid(&ctx, module);
                let call = collect_ops(&ctx, invalid.op_ref())
                    .into_iter()
                    .find(|&op| func::CallIndirect::matches(&ctx, op))
                    .unwrap();
                assert_eq!(ctx.op_result_types(call).len(), 1);
                assert_eq!(ctx.op_operands(call).len(), 2);
                assert!(print_module(&ctx, invalid.op_ref()).contains("signature"));
                continue;
            }
            let error = lower_prepared_closures(&mut ctx, module).unwrap_err();
            assert_eq!(
                error.to_string(),
                "closure lowering: exact caller/callee/indirect result contract mismatch"
            );
            assert_eq!(print_module(&ctx, module.op()), before);
            assert_eq!(collect_ops(&ctx, module.op()), ops);
            // The rewrite pattern must also decline unsupported arities safely.
            rewrite_validated_closures_in_func(&mut ctx, invalid, Default::default());
            assert_eq!(print_module(&ctx, module.op()), before);
            assert_eq!(collect_ops(&ctx, module.op()), ops);
        }
    }

    #[test]
    fn prepared_module_worklist_lowers_nested_function_closures() {
        let mut ctx = IrContext::new();
        let module = nested_closure_test_module(&mut ctx);

        let core_module = core::Module::from_op(&ctx, module.op()).unwrap();
        let mut pass = LowerPreparedClosures;
        pass.run(&mut ctx, core_module, &mut Default::default())
            .unwrap();
        assert_module_is_structurally_valid(&ctx, module);

        let inner = func_by_name_recursive(&ctx, module, "inner");
        let inner_ir = print_module(&ctx, inner.op_ref());
        assert!(
            !inner_ir.contains("closure.new"),
            "prepared module worklist must lower nested functions:\n{inner_ir}"
        );
    }

    #[test]
    fn storage_finalization_rewrites_every_type_surface_but_not_pack_provenance() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !closure = closure.closure<func.func_sig<(core.i32) -> core.i32>>
  !nested = core.tuple<!closure>
  func.func @run(%callback: !closure) -> !nested {
    %created = closure.new {entries = [@callback]} : !closure
    %packed = core.tuple_pack %created : !nested
    func.return %packed
  }
}"#,
        );

        let run = func_by_name(&ctx, module, "run");
        let pack = ctx
            .block(ctx.region(run.body(&ctx)).blocks[0])
            .ops
            .iter()
            .copied()
            .find(|op| closure::New::from_op(&ctx, *op).is_ok())
            .expect("test module must create a closure");

        finalize_closure_storage_layout(&mut ctx, module);

        let physical = closure_struct_type_ref(&mut ctx);
        let signature = func::FuncSig::from_type_ref(&ctx, run.r#type(&ctx)).unwrap();
        assert_eq!(signature.inputs(&ctx), [physical]);
        assert_eq!(
            ctx.value_ty(ctx.block_args(ctx.region(run.body(&ctx)).blocks[0])[0]),
            physical
        );
        assert_eq!(ctx.op_result_types(pack), [physical]);
        let printed = print_module(&ctx, module.op());
        assert!(printed.contains("!closure = adt.struct"), "{printed}");
        assert!(
            printed.contains("!nested = core.tuple<!closure>"),
            "{printed}"
        );
    }

    fn check_tagged_dispatch_tail(physical: bool) {
        let mut ctx = IrContext::new();
        let ev = evidence_type_str();
        let module = parse_test_module(
            &mut ctx,
            &format!(
                r#"core.module @test {{
  !cps = closure.closure<func.func_sig<({ev}, core.i32, core.i32, core.i32) -> core.never>>
  func.func @run(%callee: !cps, %evidence: {ev}, %done: core.i32, %dispatch: core.i32, %value: core.i32) -> core.never attributes {{tribute.calling_convention = "cps"}} {{
    func.tail_call_indirect %callee, %evidence, %done, %dispatch, %value {{tribute.calling_convention = "cps", signature = func.func_sig<({ev}, core.i32, core.i32, core.i32) -> core.never>}}
  }}
}}"#
            ),
        );
        if physical {
            crate::target_abi::lower_cps_signatures_to_physical(&mut ctx, module).unwrap();
        }
        let run = func_by_name(&ctx, module, "run");
        let entry_args = ctx.block_args(ctx.region(run.body(&ctx)).blocks[0]);
        let callee = entry_args[0];
        let evidence = entry_args[1];
        let done = entry_args[2];
        let dispatch = entry_args[3];
        let value = entry_args[4];

        lower_prepared_closures(&mut ctx, module).unwrap();

        let tail = ctx
            .block(ctx.region(run.body(&ctx)).blocks[0])
            .ops
            .iter()
            .copied()
            .find(|op| func::TailCallIndirect::from_op(&ctx, *op).is_ok())
            .unwrap();
        let signature = DynIndirectCallLike::get(&ctx, tail)
            .and_then(|call| call.exact_signature(&ctx))
            .unwrap();
        let callable = func::FuncSig::from_type_ref(&ctx, signature).unwrap();
        assert_eq!(callable.results(&ctx).is_empty(), physical);
        assert_eq!(ctx.op_operands(tail).len(), 6);
        assert_eq!(ctx.op_operands(tail)[1], evidence);
        // The closure itself is its function's environment.
        assert_eq!(ctx.op_operands(tail)[2], callee);
        assert_eq!(ctx.op_operands(tail)[3], done);
        assert_eq!(ctx.op_operands(tail)[4], dispatch);
        assert_eq!(ctx.op_operands(tail)[5], value);
        assert_eq!(
            callable.inputs(&ctx),
            ctx.op_operands(tail)[1..]
                .iter()
                .map(|&operand| ctx.value_ty(operand))
                .collect::<Vec<_>>(),
            "the physical signature must include the inserted environment operand"
        );
    }

    #[test]
    fn logical_dispatch_tail_preserves_never_results() {
        check_tagged_dispatch_tail(false);
    }

    #[test]
    fn physical_dispatch_tail_preserves_empty_results() {
        check_tagged_dispatch_tail(true);
    }

    #[test]
    fn metadata_less_tagged_closure_transfer_leaves_function_unchanged() {
        let mut ctx = IrContext::new();
        let evidence = evidence_type_str();
        let module = parse_test_module(
            &mut ctx,
            &format!(
                r#"core.module @test {{
  !cps = closure.closure<func.func_sig<({evidence}, core.i32) -> core.never>>
  func.func @callee(%evidence: {evidence}, %env: !cps, %done: core.i32) -> core.never attributes {{tribute.calling_convention = "cps"}} {{
    func.unreachable
  }}
  func.func @run(%evidence: {evidence}, %done: core.i32) -> core.never attributes {{tribute.calling_convention = "cps"}} {{
    %callee = closure.new {{entries = [@callee]}} : !cps
    func.tail_call_indirect %callee, %evidence, %done
  }}
}}"#
            ),
        );
        let before = print_module(&ctx, module.op());

        assert!(lower_prepared_closures(&mut ctx, module).is_err());

        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn bodyless_declaration_remains_unchanged() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @external(%value: core.i32) -> core.i32
}"#,
        );
        let before = print_module(&ctx, module.op());

        lower_prepared_closures(&mut ctx, module).unwrap();

        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn raw_function_constant_indirect_call_remains_unchanged() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @external(%value: core.i32) -> core.i32
  func.func @caller(%value: core.i32) -> core.i32 {
    %callee = func.constant {func_ref = @external} : func.func_sig<(core.i32) -> core.i32>
    %result = func.call_indirect %callee, %value : core.i32
    func.return %result
  }
}"#,
        );
        let before = print_module(&ctx, module.op());

        lower_prepared_closures(&mut ctx, module).unwrap();

        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn function_pointer_argument_callee_is_not_a_closure() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @caller(%callee: func.func_sig<(core.i32) -> core.i32>, %value: core.i32) -> core.i32 attributes {tribute.calling_convention = "direct"} {
    %result = func.call_indirect %callee, %value {signature = func.func_sig<(core.i32) -> core.i32>, tribute.calling_convention = "direct"} : core.i32
    func.return %result
  }
}"#,
        );
        lower_prepared_closures(&mut ctx, module).unwrap();

        // Closure lowering leaves a non-closure transfer alone and consumes
        // only the enclosing function's convention.
        let caller = func::Func::from_op(&ctx, module.ops(&ctx)[0]).unwrap();
        assert_eq!(DynConventionBearing::of(&ctx, caller.op_ref()), None);
        let call = collect_ops(&ctx, caller.op_ref())
            .into_iter()
            .find(|&op| func::CallIndirect::matches(&ctx, op))
            .unwrap();
        assert_eq!(
            DynConventionBearing::of(&ctx, call),
            Some(CallingConvention::Direct)
        );
    }

    #[test]
    fn a_capturing_closure_is_one_allocation_read_through_its_capture_layout() {
        let mut ctx = IrContext::new();
        let evidence = evidence_type_str();
        let module = parse_test_module(
            &mut ctx,
            &format!(
                r#"core.module @test {{
  !closure = closure.closure<func.func_sig<(core.i64) -> core.i64>>
  func.func @code(%env: !closure, %value: core.i64) -> core.i64 attributes {{tribute.calling_convention = "direct"}} {{
    %first, %second = closure.unpack %env : core.i64, core.i32
    func.return %first
  }}
  func.func @trap_evidence_direct(%evidence: {evidence}, %env: !closure, %value: core.i64) -> core.i64 attributes {{tribute.calling_convention = "evidence_direct"}} {{
    func.unreachable
  }}
  func.func @trap_cps(%evidence: {evidence}, %env: !closure, %value: core.i64) -> core.never attributes {{tribute.calling_convention = "cps"}} {{
    func.unreachable
  }}
  func.func @make(%first: core.i64, %second: core.i32) -> !closure {{
    %created = closure.new %first, %second {{entries = [@code, @trap_evidence_direct, @trap_cps]}} : !closure
    func.return %created
  }}
}}"#
            ),
        );
        lower_prepared_closures(&mut ctx, module).unwrap();
        finalize_closure_storage_layout(&mut ctx, module);

        let [i32_ty, i64_ty] = [core::I32::type_ref(&mut ctx), core::I64::type_ref(&mut ctx)];
        let layout = closure_captures_type_ref(&mut ctx, &[i64_ty, i32_ty]);
        let base = closure_struct_type_ref(&mut ctx);
        let table_ty = tribute_rt::ConventionTableRef::params()
            .intern(&mut ctx)
            .as_type_ref();
        assert_eq!(
            layout,
            closure_captures_type_ref(&mut ctx, &[i64_ty, i32_ty])
        );
        assert_eq!(
            ctx.type_attrs(layout)
                .get_type(runtime_layout::CLOSURE_BASE_ATTR),
            Some(base)
        );
        itertools::assert_equal(
            adt::Struct::from_type_ref(&ctx, layout)
                .unwrap()
                .fields(&ctx),
            [("entries", table_ty), ("_0", i64_ty), ("_1", i32_ty)],
        );

        let make = func_by_name(&ctx, module, "make");
        let allocations: Vec<_> = collect_ops(&ctx, make.op_ref())
            .into_iter()
            .filter_map(|op| adt::StructNew::from_op(&ctx, op).ok())
            .collect();
        let [allocation] = allocations[..] else {
            panic!("one allocation per closure");
        };
        assert_eq!(allocation.r#type(&ctx), layout);
        assert_eq!(allocation.fields(&ctx).len(), 3);

        // The entry function downcasts its base environment once, then reads
        // each capture after the code field.
        let code = func_by_name(&ctx, module, "code");
        let entry = ctx.region(code.body(&ctx)).blocks[0];
        let environment = ctx.block_arg(entry, 0);
        assert_eq!(ctx.value_ty(environment), base);
        let reads: Vec<_> = collect_ops(&ctx, code.op_ref())
            .into_iter()
            .filter_map(|op| adt::StructGet::from_op(&ctx, op).ok())
            .map(|get| (get.r#type(&ctx), get.field(&ctx)))
            .collect();
        assert_eq!(reads, [(layout, 1), (layout, 2)]);
    }

    #[test]
    fn lowered_closures_reference_their_convention_tables_and_leave_no_casts_after_finalization() {
        let mut ctx = IrContext::new();
        let module = closure_test_module(&mut ctx);

        lower_prepared_closures(&mut ctx, module).unwrap();
        let mut tables = Vec::new();
        let _ = walk_op::<()>(&ctx, module.op(), &mut |op| {
            if let Ok(table) = tribute_rt::ConventionTable::from_op(&ctx, op) {
                tables.push(table);
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
        assert!(!tables.is_empty());
        for table in tables {
            for name in table.entries(&ctx) {
                assert!(
                    module
                        .ops(&ctx)
                        .iter()
                        .filter_map(|&op| func::Func::from_op(&ctx, op).ok())
                        .any(|function| name == [function.sym_name(&ctx)]),
                    "convention table entry must name a function"
                );
            }
        }

        finalize_closure_storage_layout(&mut ctx, module);
        let ir = print_module(&ctx, module.op());
        assert!(!ir.contains("core.unrealized_conversion_cast"), "{ir}");
        assert!(!ir.contains("closure.closure"), "{ir}");
    }

    #[test]
    fn closure_targets_resolve_by_root_qualified_path() {
        let source = |reference: &str| {
            format!(
                r#"core.module @test {{
  !closure = closure.closure<func.func_sig<(core.i32) -> core.i32>>
  core.module @left {{
    func.func @helper(%env: !closure, %value: core.i32) -> core.i32 attributes {{tribute.calling_convention = "direct"}} {{
      func.return %value
    }}
  }}
  core.module @right {{
    func.func @helper(%env: !closure, %value: core.i64) -> core.i32 {{
      %zero = arith.const {{value = 0}} : core.i32
      func.return %zero
    }}
  }}
  func.func @trap_evidence_direct(%evidence: core.i32, %env: !closure, %value: core.i32) -> core.i32 attributes {{tribute.calling_convention = "evidence_direct"}} {{
    func.unreachable
  }}
  func.func @trap_cps(%evidence: core.i32, %env: !closure, %value: core.i32) -> core.never attributes {{tribute.calling_convention = "cps"}} {{
    func.unreachable
  }}
  func.func @make() -> !closure {{
    %created = closure.new {{entries = [{reference}, @trap_evidence_direct, @trap_cps]}} : !closure
    func.return %created
  }}
}}"#
            )
        };

        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, &source("@left::@helper"));
        lower_prepared_closures(&mut ctx, module).unwrap();
        let ir = print_module(&ctx, module.op());
        assert!(!ir.contains("closure.new"), "{ir}");
        assert!(
            ir.contains(r#"tribute_rt.convention_table {entries = [@left::@helper, @trap_evidence_direct, @trap_cps]} : tribute_rt.convention_table_ref"#),
            "{ir}"
        );

        // A bare name does not resolve relative to any nested module.
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, &source("@helper"));
        lower_prepared_closures(&mut ctx, module).unwrap();
        let ir = print_module(&ctx, module.op());
        assert!(ir.contains("closure.new"), "{ir}");
        assert!(!ir.contains("tribute_rt.convention_table"), "{ir}");
    }

    #[test]
    fn differently_typed_tagged_closure_pack_fails_closed() {
        let mut ctx = IrContext::new();
        let evidence = evidence_type_str();
        let module = parse_test_module(
            &mut ctx,
            &format!(
                r#"core.module @test {{
  !_closure = adt.struct<_closure(func_ptr: core.i32, env: tribute_rt.anyref)>
  !expected = closure.closure<func.func_sig<({evidence}, core.i32, core.i32) -> core.never>>
  !actual = closure.closure<func.func_sig<({evidence}, core.i32, core.i1) -> core.never>>
  !outer = closure.closure<func.func_sig<({evidence}, core.i32, !expected) -> core.never>>

  func.func @actual_fn(%evidence: {evidence}, %done: core.i32, %value: core.i1) -> core.never attributes {{tribute.calling_convention = "cps"}} {{
    func.unreachable
  }}
  func.func @run(%callee: !outer, %evidence: {evidence}, %done: core.i32) -> core.never attributes {{tribute.calling_convention = "cps"}} {{
    %function = func.constant {{func_ref = @actual_fn}} : func.func_sig<({evidence}, core.i32, core.i1) -> core.never>
    %environment = adt.ref_null {{type = tribute_rt.anyref}} : tribute_rt.anyref
    %pack = adt.struct_new %function, %environment {{type = !_closure}} : !_closure
    %argument = core.unrealized_conversion_cast %pack : !actual
    func.tail_call_indirect %callee, %evidence, %done, %argument {{tribute.calling_convention = "cps"}}
  }}
}}"#
            ),
        );
        let before = print_module(&ctx, module.op());

        assert!(lower_prepared_closures(&mut ctx, module).is_err());

        assert_eq!(print_module(&ctx, module.op()), before);
    }
}
