//! The convention tables that `closure.new` names: one entry function per
//! calling convention, at its slot. A slot the closure does not implement
//! holds a shared adapter or trap entry (`new-plans/ir.md`, "Convention
//! entry").

use rustc_hash::FxHashMap as HashMap;
use tribute_ir::callable_abi::CallableAbi;
use tribute_ir::calling_convention::{CallingConvention, ConventionBearing, DynConventionBearing};
use tribute_ir::dialect::{ability, closure};
use trunk_ir::context::IrContext;
use trunk_ir::dialect::{core, func};
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{BlockRef, TypeRef};
use trunk_ir::types::Location;
use trunk_ir::{Symbol, SymbolPath};

use crate::cps_builders::{helper_symbol, make_block, single_block_region};
use crate::lower_continuation_frames::frame_reference;

const TRAP_ENTRY: &str = "trap_entry";
const ENTRY_ADAPTER: &str = "entry_adapter";

/// How the entries of a table are laid out where the pipeline builds it.
pub(crate) trait EntryLayout {
    /// The signature of the entry through `convention` of a closure of type
    /// `closure_ty`, with the closure as its environment.
    fn entry_signature(
        &self,
        ctx: &mut IrContext,
        closure_ty: TypeRef,
        convention: CallingConvention,
    ) -> TypeRef;
}

/// Logical entries before frame expansion, whose `Cps` entry takes an
/// `ability.frame<R>`.
pub(crate) struct AbstractFrames;

impl EntryLayout for AbstractFrames {
    fn entry_signature(
        &self,
        ctx: &mut IrContext,
        closure_ty: TypeRef,
        convention: CallingConvention,
    ) -> TypeRef {
        logical_entry_signature(ctx, closure_ty, convention, |ctx, answer| {
            ability::Frame::params(answer).intern(ctx).as_type_ref()
        })
    }
}

/// Logical entries after frame expansion, whose `Cps` entry takes the frame
/// reference of its answer.
pub(crate) struct ExpandedFrames;

impl EntryLayout for ExpandedFrames {
    fn entry_signature(
        &self,
        ctx: &mut IrContext,
        closure_ty: TypeRef,
        convention: CallingConvention,
    ) -> TypeRef {
        logical_entry_signature(ctx, closure_ty, convention, frame_reference)
    }
}

/// The convention tables of the closures a pass makes, and the adapters and
/// trap entries that fill their slots (`new-plans/ir.md`, "Convention
/// entry"). Both are shared in each symbol table block by what determines
/// them.
pub(crate) struct ConventionTables<L> {
    layout: L,
    scopes: HashMap<BlockRef, Scope>,
}

/// The adapters and trap entries one symbol table block defines.
struct Scope {
    traps: HashMap<(TypeRef, CallingConvention), Symbol>,
    /// Adapters by their signature, convention, and the convention of the
    /// entry they call.
    adapters: HashMap<(TypeRef, CallingConvention, CallingConvention), Symbol>,
    next_trap: u32,
    next_adapter: u32,
}

/// The index of a helper `__tribute_{prefix}_N` named `name`.
fn helper_index(name: &str, prefix: &str) -> Option<u32> {
    name.strip_prefix("__tribute_")?
        .strip_prefix(prefix)?
        .strip_prefix('_')?
        .parse()
        .ok()
}

impl Scope {
    /// The helpers `block` already defines. A trap entry is reused by its
    /// signature and convention; an adapter made by an earlier pass only
    /// reserves its name.
    fn read(ctx: &IrContext, block: BlockRef) -> Self {
        let mut scope = Self {
            traps: HashMap::default(),
            adapters: HashMap::default(),
            next_trap: 0,
            next_adapter: 0,
        };
        for &op in &ctx.block(block).ops {
            let Ok(function) = func::Func::from_op(ctx, op) else {
                continue;
            };
            let name = function.sym_name(ctx);
            if let Some(index) = helper_index(name, ENTRY_ADAPTER) {
                scope.next_adapter = scope.next_adapter.max(index + 1);
                continue;
            }
            let (Some(index), Some(convention)) = (
                helper_index(name, TRAP_ENTRY),
                DynConventionBearing::of(ctx, op),
            ) else {
                continue;
            };
            scope
                .traps
                .insert((function.r#type(ctx), convention), Symbol::new(name));
            scope.next_trap = scope.next_trap.max(index + 1);
        }
        scope
    }
}

impl<L: EntryLayout> ConventionTables<L> {
    pub(crate) fn new(layout: L) -> Self {
        Self {
            layout,
            scopes: HashMap::default(),
        }
    }

    /// The table of a closure of type `closure_ty` whose slots `entries`
    /// fills, each with the entry function of its convention. A trap entry
    /// defined in `block`, the symbol table block that resolves the
    /// closure's entries, fills each other slot.
    pub(crate) fn table(
        &mut self,
        ctx: &mut IrContext,
        block: BlockRef,
        location: Location,
        closure_ty: TypeRef,
        entries: impl IntoIterator<Item = (CallingConvention, SymbolPath)>,
    ) -> [SymbolPath; 3] {
        let mut slots: [Option<SymbolPath>; 3] = Default::default();
        for (convention, entry) in entries {
            slots[convention.slot() as usize] = Some(entry);
        }
        let mut slots = slots.into_iter();
        CallingConvention::ALL.map(|convention| {
            slots.next().flatten().unwrap_or_else(|| {
                SymbolPath::new([self.trap(ctx, block, location, closure_ty, convention)])
            })
        })
    }

    fn scope(&mut self, ctx: &IrContext, block: BlockRef) -> &mut Scope {
        self.scopes
            .entry(block)
            .or_insert_with(|| Scope::read(ctx, block))
    }

    fn trap(
        &mut self,
        ctx: &mut IrContext,
        block: BlockRef,
        location: Location,
        closure_ty: TypeRef,
        convention: CallingConvention,
    ) -> Symbol {
        let signature = self.layout.entry_signature(ctx, closure_ty, convention);
        let scope = self.scope(ctx, block);
        if let Some(name) = scope.traps.get(&(signature, convention)) {
            return name.clone();
        }
        let name = helper_symbol(TRAP_ENTRY, scope.next_trap);
        scope.next_trap += 1;
        scope.traps.insert((signature, convention), name.clone());
        let body = make_block(ctx, location, &entry_inputs(ctx, signature));
        let unreachable = func::Unreachable::operands().build(ctx, location);
        ctx.push_op(body, unreachable.op_ref());
        define(ctx, block, location, &name, signature, body, convention);
        name
    }
}

impl ConventionTables<AbstractFrames> {
    /// The adapters of a source closure of type `closure_ty` whose body
    /// implements `body`: one for each stronger convention, defined in
    /// `block`. A generated closure, whose closure type's callable is its own
    /// entry, has none.
    pub(crate) fn adapters(
        &mut self,
        ctx: &mut IrContext,
        block: BlockRef,
        location: Location,
        closure_ty: TypeRef,
        body: CallingConvention,
    ) -> Vec<(CallingConvention, Symbol)> {
        CallingConvention::ALL
            .into_iter()
            .filter(|&convention| convention > body)
            .map(|convention| {
                let adapter = self.adapter(ctx, block, location, closure_ty, convention, body);
                (convention, adapter)
            })
            .collect()
    }

    /// The adapter entry through `convention` that enters its closure again
    /// through `body`, a weaker convention. It drops the evidence `body` does
    /// not take. A `Cps` adapter exits to its frame with the result, and a
    /// value adapter tail calls the body's entry.
    fn adapter(
        &mut self,
        ctx: &mut IrContext,
        block: BlockRef,
        location: Location,
        closure_ty: TypeRef,
        convention: CallingConvention,
        body: CallingConvention,
    ) -> Symbol {
        let signature = self.layout.entry_signature(ctx, closure_ty, convention);
        let key = (signature, convention, body);
        let scope = self.scope(ctx, block);
        if let Some(name) = scope.adapters.get(&key) {
            return name.clone();
        }
        let name = helper_symbol(ENTRY_ADAPTER, scope.next_adapter);
        scope.next_adapter += 1;
        scope.adapters.insert(key, name.clone());

        // The transfer's signature is the body entry's without the
        // environment, which the closure lowering interposes again.
        let body_entry = self.layout.entry_signature(ctx, closure_ty, body);
        let transfer_signature = func::FuncSig::from_type_ref(ctx, body_entry)
            .expect("an entry signature is a func.func_sig")
            .rebuild(ctx, |inputs, _| {
                inputs.remove(body.closure_environment_index());
            })
            .as_type_ref();
        let entry = make_block(ctx, location, &entry_inputs(ctx, signature));
        let mut params = ctx.block_args(entry).to_vec();
        let closure = params.remove(convention.closure_environment_index());
        let evidence = convention.needs_evidence().then(|| params[0]);
        let frame = convention
            .needs_continuation_frame()
            .then(|| params[usize::from(convention.needs_evidence())]);
        let source = usize::from(convention.needs_evidence())
            + usize::from(convention.needs_continuation_frame());
        let args: Vec<_> = evidence
            .filter(|_| body.needs_evidence())
            .into_iter()
            .chain(params[source..].iter().copied())
            .collect();
        if let Some(frame) = frame {
            let call = func::CallIndirect::operands(closure, args)
                .signature(transfer_signature)
                .build(ctx, location);
            call.set_calling_convention(ctx, body);
            ctx.push_op(entry, call.op_ref());
            let exit = match ctx.op_results(call.op_ref()).first().copied() {
                Some(result) => ability::Exit::operands(frame, result)
                    .build(ctx, location)
                    .op_ref(),
                None => func::Unreachable::operands().build(ctx, location).op_ref(),
            };
            ctx.push_op(entry, exit);
        } else {
            let tail = func::TailCallIndirect::operands(closure, args)
                .signature(transfer_signature)
                .build(ctx, location);
            tail.set_calling_convention(ctx, body);
            ctx.push_op(entry, tail.op_ref());
        }
        define(ctx, block, location, &name, signature, entry, convention);
        name
    }
}

fn entry_inputs(ctx: &IrContext, signature: TypeRef) -> Vec<TypeRef> {
    func::FuncSig::from_type_ref(ctx, signature)
        .expect("an entry signature is a func.func_sig")
        .inputs(ctx)
        .to_vec()
}

/// Define the entry function `name` of `signature` and `convention` with
/// the body `entry` at the end of `block`.
fn define(
    ctx: &mut IrContext,
    block: BlockRef,
    location: Location,
    name: &Symbol,
    signature: TypeRef,
    entry: BlockRef,
    convention: CallingConvention,
) {
    let body = single_block_region(ctx, location, entry);
    let function = func::Func::operands()
        .sym_name(ctx.intern_symbol_text(name))
        .r#type(signature)
        .regions(body)
        .build(ctx, location);
    function.set_calling_convention(ctx, convention);
    ctx.push_op(block, function.op_ref());
}

/// The logical signature of the entry through `convention` of a closure of
/// type `closure_ty`: its callable's parameters and result laid out by that
/// convention, with the closure as the environment and `frame(R)` as the
/// continuation frame of a result `R`.
fn logical_entry_signature(
    ctx: &mut IrContext,
    closure_ty: TypeRef,
    convention: CallingConvention,
    frame: impl FnOnce(&mut IrContext, TypeRef) -> TypeRef,
) -> TypeRef {
    let callable = closure::Closure::from_type_ref(ctx, closure_ty)
        .and_then(|closure| func::FuncSig::from_type_ref(ctx, closure.func_type(ctx)))
        .expect("a closure type holds a func.func_sig");
    let never = core::Never::params().intern(ctx).as_type_ref();
    let result = callable.single_result(ctx).unwrap_or(never);
    let abi = CallableAbi::new(convention, callable.inputs(ctx).to_vec(), result);
    let evidence = ability::evidence_adt_type_ref(ctx);
    let frame = if convention.needs_continuation_frame() {
        frame(ctx, result)
    } else {
        never
    };
    let params = abi.interpose_environment(&abi.lowered_params(evidence, frame), closure_ty);
    // A `Cps` entry's result is `core.never`; a value entry of a callable
    // that never returns, such as a generated continuation's, has none.
    let results: &[TypeRef] = if convention == CallingConvention::Cps {
        &[never]
    } else if result == never {
        &[]
    } else {
        &[result]
    };
    func::func_sig(ctx, params, results.iter().copied()).as_type_ref()
}

#[cfg(test)]
mod tests {
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::symbol_table::SymbolTable;

    use super::*;
    use crate::lower_closure_lambda::lower_closure_lambda;

    /// The closure tables of `module`'s `closure.new`s, each entry with the
    /// convention of its function and its environment index.
    fn tables(
        ctx: &IrContext,
        module: trunk_ir::rewrite::Module,
    ) -> Vec<Vec<(String, CallingConvention, usize)>> {
        let symbols = SymbolTable::collect(ctx, module);
        let mut tables = Vec::new();
        let _ = trunk_ir::walk::walk_op::<()>(ctx, module.op(), &mut |op| {
            if let Ok(new) = closure::New::from_op(ctx, op) {
                let closure_ty = ctx.op_result_types(op)[0];
                tables.push(
                    new.entries(ctx)
                        .map(|entry| {
                            let function = symbols
                                .resolve(ctx, op, entry)
                                .and_then(|function| func::Func::from_op(ctx, function).ok())
                                .expect("every entry resolves to a function");
                            let environment =
                                func::FuncSig::from_type_ref(ctx, function.r#type(ctx))
                                    .expect("an entry has a signature")
                                    .inputs(ctx)
                                    .iter()
                                    .position(|&input| input == closure_ty)
                                    .expect("an entry takes its closure");
                            (
                                function.sym_name(ctx).to_owned(),
                                DynConventionBearing::of(ctx, function.op_ref())
                                    .expect("an entry records its convention"),
                                environment,
                            )
                        })
                        .collect(),
                );
            }
            std::ops::ControlFlow::Continue(trunk_ir::walk::WalkAction::Advance)
        });
        tables
    }

    /// The convention of the indirect transfer by which the adapter
    /// `function` enters its own environment, the closure at `environment`.
    fn adapter_transfer(
        ctx: &IrContext,
        function: &str,
        module: trunk_ir::rewrite::Module,
        environment: usize,
    ) -> CallingConvention {
        let function = module
            .ops(ctx)
            .iter()
            .find_map(|&op| {
                func::Func::from_op(ctx, op)
                    .ok()
                    .filter(|candidate| candidate.sym_name(ctx) == function)
            })
            .expect("the adapter is defined in the module");
        let entry = ctx.region(function.body(ctx)).blocks[0];
        let closure = ctx.block_args(entry)[environment];
        let transfer = ctx
            .block(entry)
            .ops
            .iter()
            .copied()
            .find(|&op| {
                func::CallIndirect::matches(ctx, op) || func::TailCallIndirect::matches(ctx, op)
            })
            .expect("an adapter enters its closure");
        assert_eq!(ctx.op_operands(transfer)[0], closure);
        let signature = ctx
            .op_attrs(transfer)
            .get_type("signature")
            .and_then(|signature| func::FuncSig::from_type_ref(ctx, signature))
            .expect("the transfer carries its signature");
        assert!(
            ctx.op_operands(transfer)[1..]
                .iter()
                .map(|&arg| ctx.value_ty(arg))
                .eq(signature.inputs(ctx).iter().copied()),
            "the transfer passes its signature's arguments"
        );
        DynConventionBearing::of(ctx, transfer).expect("the transfer records its convention")
    }

    /// A lambda without adapters, such as a generated closure, has trap
    /// entries in every other slot.
    #[test]
    fn a_closure_fills_the_slots_it_does_not_implement_with_shared_trap_entries() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @main(%y: core.i32) -> core.i32 attributes {tribute.calling_convention = "direct"} {
    %f = closure.lambda(%x: core.i32) -> core.i32 {tribute.calling_convention = "direct"} [%y] {
      func.return %x
    }
    %g = closure.lambda(%x: core.i32) -> core.i32 {tribute.calling_convention = "direct"} [%y] {
      func.return %y
    }
    func.return %y
  }
}"#,
        );
        lower_closure_lambda(&mut ctx, module);
        let tables = tables(&ctx, module);
        let [first, second] = &tables[..] else {
            panic!("two closures: {tables:?}");
        };
        // The Direct slot holds each lambda's own entry, with the closure
        // first; the other slots hold trap entries of their convention,
        // shared by the two closures of one type.
        assert_eq!(first[0].1, CallingConvention::Direct);
        assert_eq!(first[0].2, 0);
        assert_ne!(first[0].0, second[0].0);
        assert_eq!(first[1..], second[1..]);
        let traps: Vec<_> = first[1..]
            .iter()
            .map(|(name, convention, environment)| {
                (
                    name.starts_with("__tribute_trap_entry_"),
                    *convention,
                    *environment,
                )
            })
            .collect();
        assert_eq!(
            traps,
            [
                (true, CallingConvention::EvidenceDirect, 1),
                (true, CallingConvention::Cps, 1),
            ]
        );
    }

    /// A source closure's stronger slots hold adapters that enter it again
    /// through its own slot, and its weaker slots hold trap entries.
    #[test]
    fn a_source_closure_fills_its_stronger_slots_with_adapters() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  tribute_cps.func @direct(%y: core.i32) -> core.i32 convention(direct) {
    %f = tribute_cps.lambda(%x: core.i32) -> core.i32 captures [%y] entries {
      direct {
        %sum = arith.addi %x, %y : core.i32
        tribute_control.return %sum
      }
    }
    %g = tribute_cps.lambda(%x: core.i32) -> core.i32 captures [] entries {
      direct {
        tribute_control.return %x
      }
    }
    %a = tribute_cps.call_indirect %f, %y {convention = "direct"} : core.i32
    %b = tribute_cps.call_indirect %g, %a {convention = "direct"} : core.i32
    tribute_control.return %b
  }
  tribute_cps.func @evidence(%y: core.i32) -> core.i32 convention(evidence_direct) {
    %h = tribute_cps.lambda(%x: core.i32) -> core.i32 captures [] entries {
      evidence_direct {
        tribute_control.return %x
      }
    }
    %r = tribute_cps.call_indirect %h, %y {convention = "evidence_direct"} : core.i32
    tribute_control.return %r
  }
}"#,
        );
        crate::tribute_control_to_cps::tribute_control_to_cps(
            &mut ctx,
            module,
            &[],
            &mut Default::default(),
        )
        .unwrap();
        lower_closure_lambda(&mut ctx, module);
        let tables = tables(&ctx, module);
        let [first, second, third] = &tables[..] else {
            panic!("three closures: {tables:?}");
        };
        // Closures of one type share their adapters.
        assert_eq!(first[1..], second[1..]);
        let slots = |table: &[(String, CallingConvention, usize)]| -> Vec<_> {
            table
                .iter()
                .map(|(name, convention, environment)| {
                    let transfer = name
                        .starts_with("__tribute_entry_adapter_")
                        .then(|| adapter_transfer(&ctx, name, module, *environment));
                    (
                        name.starts_with("__tribute_trap_entry_"),
                        transfer,
                        *convention,
                        *environment,
                    )
                })
                .collect()
        };
        use CallingConvention::*;
        assert_eq!(
            slots(first),
            [
                (false, None, Direct, 0),
                (false, Some(Direct), EvidenceDirect, 1),
                (false, Some(Direct), Cps, 1),
            ]
        );
        assert_eq!(
            slots(third),
            [
                (true, None, Direct, 0),
                (false, None, EvidenceDirect, 1),
                (false, Some(EvidenceDirect), Cps, 1),
            ]
        );
    }
}
