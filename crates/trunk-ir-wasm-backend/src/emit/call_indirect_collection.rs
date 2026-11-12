//! Call indirect type collection for wasm backend emission.
//!
//! This module handles the collection of function types used in call_indirect
//! operations and ref_func declarations.

use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;

use trunk_ir::dialect::wasm as wasm_dialect;
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{RegionRef, TypeRef};
use trunk_ir::smallvec::SmallVec;
use trunk_ir::{IrContext, Module, SymbolPath};

use crate::errors::CompilationResult;
use crate::stack_effect::{is_structured, structured_result_types};

use super::helpers;

/// The function types the type section declares beyond the functions' own.
#[derive(Debug, Default)]
pub(crate) struct IndexedFuncTypes {
    /// Each added `wasm.func_sig` with its type index, in index order.
    pub(crate) types: Vec<(u32, TypeRef)>,
    /// The type index of each multi-value block type, keyed by its results.
    pub(crate) block_types: HashMap<BlockResults, u32>,
}

/// The results of a structured operation's block type.
pub(crate) type BlockResults = SmallVec<[TypeRef; 2]>;

/// Register the exact physical signatures carried by indirect calls, and the
/// function type `() -> (results)` of each block type with several results.
///
/// The type section uses the retained signature as its key, including calls
/// with zero or multiple results. Erased operands cannot reconstruct it. A
/// block type shares the index of a signature with the same inputs and
/// results (`design/wasm-backend.md`, "구조적 제어의 block type").
pub(crate) fn collect_call_indirect_types(
    ctx: &mut IrContext,
    module: Module,
    type_idx_by_type: &mut HashMap<TypeRef, u32>,
    gc_type_count: usize,
    func_type_count: usize,
) -> CompilationResult<IndexedFuncTypes> {
    /// A function type to index, in the order the module first uses it.
    enum Needed {
        Signature(TypeRef),
        Block(BlockResults),
    }

    fn collect_from_region(
        ctx: &IrContext,
        region: RegionRef,
        needed: &mut Vec<Needed>,
    ) -> CompilationResult<()> {
        for &block in &ctx.region(region).blocks {
            for &op in &ctx.block(block).ops {
                for nested in ctx.op_regions(op) {
                    collect_from_region(ctx, nested, needed)?;
                }
                if wasm_dialect::ReturnCallIndirect::matches(ctx, op) {
                    needed.push(Needed::Signature(
                        helpers::exact_return_call_indirect_signature(ctx, op)?,
                    ));
                } else if wasm_dialect::CallIndirect::matches(ctx, op) {
                    needed.push(Needed::Signature(helpers::exact_call_indirect_signature(
                        ctx, op,
                    )?));
                } else if is_structured(ctx, op) {
                    let results = structured_result_types(ctx, op);
                    if results.len() > 1 {
                        needed.push(Needed::Block(results));
                    }
                }
            }
        }
        Ok(())
    }

    let mut needed = Vec::new();
    collect_from_region(ctx, module.body(ctx).unwrap(), &mut needed)?;

    let mut next_type_idx = (gc_type_count + func_type_count) as u32;
    let mut indexed = IndexedFuncTypes::default();
    let mut staged_indices = type_idx_by_type.clone();
    for needed in needed {
        let (signature, block) = match needed {
            Needed::Signature(signature) => (signature, None),
            Needed::Block(results) => {
                if indexed.block_types.contains_key(&results) {
                    continue;
                }
                let signature = wasm_dialect::func_sig(ctx, [], results.iter().copied());
                (signature.as_type_ref(), Some(results))
            }
        };
        let index = *staged_indices.entry(signature).or_insert_with(|| {
            let index = next_type_idx;
            next_type_idx += 1;
            indexed.types.push((index, signature));
            index
        });
        if let Some(results) = block {
            indexed.block_types.insert(results, index);
        }
    }
    *type_idx_by_type = staged_indices;
    Ok(indexed)
}

/// Collect function names referenced via wasm.ref_func.
///
/// These functions need to be declared in a declarative elem segment.
pub(crate) fn collect_ref_funcs(ctx: &IrContext, module: Module) -> HashSet<SymbolPath> {
    fn collect_from_region(
        ctx: &IrContext,
        region_ref: RegionRef,
        ref_funcs: &mut HashSet<SymbolPath>,
    ) {
        for &block_ref in &ctx.region(region_ref).blocks {
            for &op in &ctx.block(block_ref).ops {
                // Recursively process nested regions
                for nested in ctx.op_regions(op) {
                    collect_from_region(ctx, nested, ref_funcs);
                }

                // Check if this is a ref_func
                if let Ok(ref_func_op) = wasm_dialect::RefFunc::from_op(ctx, op) {
                    ref_funcs.insert(ref_func_op.func_name(ctx).clone());
                }
            }
        }
    }

    let mut ref_funcs = HashSet::default();
    let body = module.body(ctx).unwrap();
    collect_from_region(ctx, body, &mut ref_funcs);
    ref_funcs
}

/// Check if the module contains any table-based indirect transfer.
pub(crate) fn has_call_indirect(ctx: &IrContext, module: Module) -> bool {
    fn check_region(ctx: &IrContext, region_ref: RegionRef) -> bool {
        for &block_ref in &ctx.region(region_ref).blocks {
            for &op in &ctx.block(block_ref).ops {
                // Check nested regions first
                for nested in ctx.op_regions(op) {
                    if check_region(ctx, nested) {
                        return true;
                    }
                }

                if wasm_dialect::CallIndirect::matches(ctx, op)
                    || wasm_dialect::ReturnCallIndirect::matches(ctx, op)
                {
                    return true;
                }
            }
        }
        false
    }

    let body = module.body(ctx).unwrap();
    check_region(ctx, body)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn resultless_target_functions_do_not_abort_indirect_collection() {
        let mut ctx = IrContext::new();
        let text = "core.module @m {
                wasm.func @good() -> core.i32 {
                    %callee = wasm.i32_const 0 : core.i32
                    %value = wasm.call_indirect %callee() : () -> core.i32
                    wasm.return %value
                }
                wasm.func @zero() { wasm.return }
            }";
        let module = trunk_ir::parser::parse_test_module(&mut ctx, text);
        let seed = wasm_dialect::func_sig(&mut ctx, [], []).as_type_ref();
        let mut indices = [(seed, 7)].into_iter().collect::<HashMap<_, _>>();
        let added = collect_call_indirect_types(&mut ctx, module, &mut indices, 8, 0).unwrap();
        assert_eq!(added.types.len(), 1);
        assert_eq!(indices[&seed], 7);
    }

    #[test]
    fn resultless_function_is_a_legal_target_contract() {
        let mut ctx = IrContext::new();
        let module = trunk_ir::parser::parse_test_module(
            &mut ctx,
            "core.module @m { wasm.func @f() { wasm.return } }",
        );
        let before = module.op().display(&ctx).to_string();
        let mut indices = HashMap::default();
        let added = collect_call_indirect_types(&mut ctx, module, &mut indices, 0, 0).unwrap();
        assert!(added.types.is_empty());
        assert!(indices.is_empty());
        assert_eq!(module.op().display(&ctx).to_string(), before);
    }
}
