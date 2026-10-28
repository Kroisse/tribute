//! The convention tables that `closure.new` names: one entry function per
//! calling convention, at its slot, with a shared trap entry in every slot
//! the closure does not implement (`new-plans/ir.md`, "Convention entry").

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

const TRAP_ENTRY: &str = "trap_entry";

/// The trap entries of each symbol table block, shared there by signature
/// and convention. A trap entry traps as soon as it is entered.
#[derive(Default)]
pub(crate) struct TrapEntries {
    scopes: HashMap<BlockRef, Scope>,
}

/// The trap entries one symbol table block defines.
struct Scope {
    known: HashMap<(TypeRef, CallingConvention), Symbol>,
    next: u32,
}

impl Scope {
    fn read(ctx: &IrContext, block: BlockRef) -> Self {
        let mut known = HashMap::default();
        let mut next = 0;
        for &op in &ctx.block(block).ops {
            let Ok(function) = func::Func::from_op(ctx, op) else {
                continue;
            };
            let name = function.sym_name(ctx);
            let index = name
                .strip_prefix("__tribute_")
                .and_then(|name| name.strip_prefix(TRAP_ENTRY))
                .and_then(|name| name.strip_prefix('_'))
                .and_then(|index| index.parse::<u32>().ok());
            let (Some(index), Some(convention)) = (index, DynConventionBearing::of(ctx, op)) else {
                continue;
            };
            known.insert((function.r#type(ctx), convention), Symbol::new(name));
            next = next.max(index + 1);
        }
        Self { known, next }
    }
}

impl TrapEntries {
    /// The entries of a closure that implements only `convention`, with
    /// `entry`: `entry` at its slot, and in each other slot a trap entry of
    /// the signature `signature` gives that convention, defined in `block`,
    /// the symbol table block that resolves the closure's entries.
    pub(crate) fn table(
        &mut self,
        ctx: &mut IrContext,
        block: BlockRef,
        location: Location,
        convention: CallingConvention,
        entry: SymbolPath,
        mut signature: impl FnMut(&mut IrContext, CallingConvention) -> TypeRef,
    ) -> [SymbolPath; 3] {
        CallingConvention::ALL.map(|slot| {
            if slot == convention {
                entry.clone()
            } else {
                let signature = signature(ctx, slot);
                SymbolPath::new([self.trap(ctx, block, location, signature, slot)])
            }
        })
    }

    fn trap(
        &mut self,
        ctx: &mut IrContext,
        block: BlockRef,
        location: Location,
        signature: TypeRef,
        convention: CallingConvention,
    ) -> Symbol {
        let scope = self
            .scopes
            .entry(block)
            .or_insert_with(|| Scope::read(ctx, block));
        if let Some(name) = scope.known.get(&(signature, convention)) {
            return name.clone();
        }
        let name = helper_symbol(TRAP_ENTRY, scope.next);
        scope.next += 1;
        scope.known.insert((signature, convention), name.clone());
        let inputs = func::FuncSig::from_type_ref(ctx, signature)
            .expect("a trap entry signature is a func.func_sig")
            .inputs(ctx)
            .to_vec();
        let body = make_block(ctx, location, &inputs);
        let unreachable = func::Unreachable::operands().build(ctx, location);
        ctx.push_op(body, unreachable.op_ref());
        let body = single_block_region(ctx, location, body);
        let function = func::Func::operands()
            .sym_name(ctx.intern_symbol_text(&name))
            .r#type(signature)
            .regions(body)
            .build(ctx, location);
        function.set_calling_convention(ctx, convention);
        ctx.push_op(block, function.op_ref());
        name
    }
}

/// The logical signature of the entry through `convention` of a closure of
/// type `closure_ty`: its callable's parameters and result laid out by that
/// convention, with the closure as the environment and `frame(R)` as the
/// continuation frame of a result `R`. Before physicalization it is the
/// signature of a trap entry in that slot.
pub(crate) fn logical_entry_signature(
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
}
