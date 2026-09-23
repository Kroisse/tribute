//! Call indirect type collection for wasm backend emission.
//!
//! This module handles the collection of function types used in call_indirect
//! operations and ref_func declarations.

use std::collections::{HashMap, HashSet};

use trunk_ir::dialect::wasm as wasm_dialect;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{RegionRef, TypeRef};
use trunk_ir::{IrContext, Module, Symbol};

use crate::errors::CompilationResult;

use super::helpers;

/// Register the exact physical signatures carried by indirect calls.
///
/// The type section uses the retained signature as its key, including calls
/// with zero or multiple results. Erased operands cannot reconstruct it.
pub(crate) fn collect_call_indirect_types(
    ctx: &mut IrContext,
    module: Module,
    type_idx_by_type: &mut HashMap<TypeRef, u32>,
    gc_type_count: usize,
    func_type_count: usize,
) -> CompilationResult<Vec<(u32, TypeRef)>> {
    fn collect_from_region(
        ctx: &IrContext,
        region: RegionRef,
        type_idx_by_type: &mut HashMap<TypeRef, u32>,
        next_type_idx: &mut u32,
        new_types: &mut Vec<(u32, TypeRef)>,
    ) -> CompilationResult<()> {
        for &block in &ctx.region(region).blocks {
            for &op in &ctx.block(block).ops {
                for &nested in &ctx.op(op).regions {
                    collect_from_region(ctx, nested, type_idx_by_type, next_type_idx, new_types)?;
                }
                let signature = if wasm_dialect::ReturnCallIndirect::matches(ctx, op) {
                    helpers::exact_return_call_indirect_signature(ctx, op)?
                } else if wasm_dialect::CallIndirect::matches(ctx, op) {
                    helpers::exact_call_indirect_signature(ctx, op)?
                } else {
                    continue;
                };
                if let std::collections::hash_map::Entry::Vacant(entry) =
                    type_idx_by_type.entry(signature)
                {
                    let index = *next_type_idx;
                    *next_type_idx += 1;
                    entry.insert(index);
                    new_types.push((index, signature));
                }
            }
        }
        Ok(())
    }

    let mut next_type_idx = (gc_type_count + func_type_count) as u32;
    let mut new_types = Vec::new();
    let mut staged_indices = type_idx_by_type.clone();
    collect_from_region(
        ctx,
        module.body(ctx).unwrap(),
        &mut staged_indices,
        &mut next_type_idx,
        &mut new_types,
    )?;
    *type_idx_by_type = staged_indices;
    Ok(new_types)
}

/// Collect function names referenced via wasm.ref_func.
///
/// These functions need to be declared in a declarative elem segment.
pub(crate) fn collect_ref_funcs(ctx: &IrContext, module: Module) -> HashSet<Symbol> {
    fn collect_from_region(
        ctx: &IrContext,
        region_ref: RegionRef,
        ref_funcs: &mut HashSet<Symbol>,
    ) {
        for &block_ref in &ctx.region(region_ref).blocks {
            for &op in &ctx.block(block_ref).ops {
                // Recursively process nested regions
                for &nested in &ctx.op(op).regions {
                    collect_from_region(ctx, nested, ref_funcs);
                }

                // Check if this is a ref_func
                if let Ok(ref_func_op) = wasm_dialect::RefFunc::from_op(ctx, op) {
                    ref_funcs.insert(ref_func_op.func_name(ctx));
                }
            }
        }
    }

    let mut ref_funcs = HashSet::new();
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
                for &nested in &ctx.op(op).regions {
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
    use trunk_ir::ops::DialectType;
    #[test]
    fn resultless_target_functions_do_not_abort_indirect_collection() {
        let mut ctx = IrContext::new();
        let text = "core.module @m {
                wasm.func {sym_name = @good, type = wasm.func_sig<() -> core.i32>} {
                    %callee = wasm.i32_const {value = 0} : core.i32
                    %value = wasm.call_indirect %callee {signature = wasm.func_sig<() -> core.i32>} : core.i32
                    wasm.return %value
                }
                wasm.func {sym_name = @zero, type = wasm.func_sig<() -> ()>} { wasm.return }
            }";
        let module = trunk_ir::parser::parse_test_module(&mut ctx, text);
        let seed = wasm_dialect::func_sig(&mut ctx, [], []).as_type_ref();
        let mut indices = HashMap::from([(seed, 7)]);
        let added = collect_call_indirect_types(&mut ctx, module, &mut indices, 8, 0).unwrap();
        assert_eq!(added.len(), 1);
        assert_eq!(indices[&seed], 7);
    }

    #[test]
    fn resultless_function_is_a_legal_target_contract() {
        let mut ctx = IrContext::new();
        let module = trunk_ir::parser::parse_test_module(
            &mut ctx,
            "core.module @m { wasm.func {sym_name = @f, type = wasm.func_sig<() -> ()>} { wasm.return } }",
        );
        let before = trunk_ir::printer::print_module(&ctx, module.op());
        let mut indices = HashMap::new();
        let added = collect_call_indirect_types(&mut ctx, module, &mut indices, 0, 0).unwrap();
        assert!(added.is_empty());
        assert!(indices.is_empty());
        assert_eq!(trunk_ir::printer::print_module(&ctx, module.op()), before);
    }

    #[test]
    fn indirect_call_requires_exact_signature_for_every_result_arity() {
        for source in [
            r#"core.module @m {
  wasm.func {sym_name = @caller, type = wasm.func_sig<(core.i32) -> core.i32>} {
    ^entry(%table_index: core.i32):
      %value = wasm.call_indirect %table_index : core.i32
      wasm.return %value
  }
}"#,
            r#"core.module @m {
  wasm.func {sym_name = @caller, type = wasm.func_sig<(core.i32) -> ()>} {
    ^entry(%table_index: core.i32):
      wasm.call_indirect %table_index
  }
}"#,
            r#"core.module @m {
  wasm.func {sym_name = @caller, type = wasm.func_sig<(core.i32) -> ()>} {
    ^entry(%table_index: core.i32):
      %first, %second = wasm.call_indirect %table_index : core.i32, core.i64
  }
}"#,
        ] {
            let mut ctx = IrContext::new();
            let module = trunk_ir::parser::parse_test_module(&mut ctx, source);
            let error = collect_call_indirect_types(&mut ctx, module, &mut HashMap::new(), 0, 1)
                .expect_err("missing exact signature must fail before type collection");
            assert!(error.to_string().contains("lacks signature"), "{error}");
        }
    }
}
