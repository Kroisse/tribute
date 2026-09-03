//! Lower func dialect operations to wasm dialect (arena IR).
//!
//! This pass converts function-level operations to wasm operations:
//! - `func.func` -> `wasm.func`
//! - `func.call` -> `wasm.call`
//! - `func.call_indirect` -> `wasm.call_indirect`
//! - `func.return` -> `wasm.return`
//! - `func.tail_call` -> `wasm.return_call`
//! - `func.tail_call_indirect` -> `wasm.return_call_indirect`
//! - `func.unreachable` -> `wasm.unreachable`
//! - `func.constant` -> `wasm.i32_const` (function table index)
//!
//! For closures, this pass also:
//! - Collects all functions referenced by `func.constant`
//! - Creates a function table with those functions
//! - Generates `wasm.table` and `wasm.elem` operations

use std::collections::HashMap;

use trunk_ir::Symbol;
use trunk_ir::context::{IrContext, OperationDataBuilder};
use trunk_ir::dialect::func::{self, CallLike, TailCallLike};
use trunk_ir::dialect::wasm as wasm_dialect;
use trunk_ir::op_interface::IndirectCallLikeModel;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{OpRef, RegionRef, TypeRef};
use trunk_ir::rewrite::{
    Module, PatternApplicator, PatternRewriter, RewritePattern, TypeConverter, clone_attrs_except,
    convert_function_type,
};
use trunk_ir::types::{Attribute, TypeDataBuilder};
use trunk_ir::{BlockData, RegionData};

use trunk_ir::smallvec::smallvec;

/// Lower func dialect to wasm dialect using arena IR.
///
/// The `type_converter` parameter allows language-specific backends to provide
/// their own type conversion rules.
pub fn lower(ctx: &mut IrContext, module: Module, type_converter: TypeConverter) {
    // 1. Collect all functions referenced by func.constant operations
    let func_refs = collect_func_constant_refs(ctx, module);

    if func_refs.is_empty() {
        // No func.constant operations - just apply patterns without table generation
        let applicator = PatternApplicator::new(type_converter)
            .add_pattern(FuncFuncPattern)
            .add_pattern(FuncCallPattern)
            .add_pattern(FuncCallIndirectPattern)
            .add_pattern(FuncReturnPattern)
            .add_pattern(FuncTailCallPattern)
            .add_pattern(FuncTailCallIndirectPattern)
            .add_pattern(FuncUnreachablePattern)
            .add_pattern(FuncConstantPattern {
                table_indices: HashMap::new(),
            });
        applicator.apply_partial(ctx, module);
        return;
    }

    // 2. Assign table indices (sorted for deterministic ordering)
    let mut sorted_funcs: Vec<_> = func_refs.into_iter().collect();
    sorted_funcs.sort_by(|a, b| a.with_str(|a_str| b.with_str(|b_str| a_str.cmp(b_str))));

    let table_indices: HashMap<Symbol, u32> = sorted_funcs
        .iter()
        .enumerate()
        .map(|(idx, sym)| (*sym, idx as u32))
        .collect();

    let table_size = sorted_funcs.len() as u32;

    // 3. Apply patterns to transform operations
    let applicator = PatternApplicator::new(type_converter)
        .add_pattern(FuncFuncPattern)
        .add_pattern(FuncCallPattern)
        .add_pattern(FuncCallIndirectPattern)
        .add_pattern(FuncReturnPattern)
        .add_pattern(FuncTailCallPattern)
        .add_pattern(FuncTailCallIndirectPattern)
        .add_pattern(FuncUnreachablePattern)
        .add_pattern(FuncConstantPattern {
            table_indices: table_indices.clone(),
        });
    applicator.apply_partial(ctx, module);

    // 4. Add wasm.table and wasm.elem to the module
    add_function_table(ctx, module, &sorted_funcs, table_size);
}

/// Collect all function symbols referenced by func.constant operations.
fn collect_func_constant_refs(ctx: &IrContext, module: Module) -> Vec<Symbol> {
    let mut funcs = Vec::new();
    if let Some(body) = module.body(ctx) {
        collect_refs_in_region(ctx, body, &mut funcs);
    }

    // Deduplicate while preserving order
    let mut seen = std::collections::HashSet::new();
    funcs.retain(|sym| seen.insert(*sym));

    funcs
}

fn collect_refs_in_region(ctx: &IrContext, region: RegionRef, refs: &mut Vec<Symbol>) {
    for &block in ctx.region(region).blocks.iter() {
        for &op in ctx.block(block).ops.iter() {
            // Check for func.constant
            if let Ok(const_op) = func::Constant::from_op(ctx, op) {
                refs.push(const_op.func_ref(ctx));
            }

            // Recurse into nested regions
            for &nested in ctx.op(op).regions.iter() {
                collect_refs_in_region(ctx, nested, refs);
            }
        }
    }
}

/// Add wasm.table and wasm.elem operations to the module for the function table.
fn add_function_table(ctx: &mut IrContext, module: Module, funcs: &[Symbol], table_size: u32) {
    let Some(first_block) = module.first_block(ctx) else {
        return;
    };

    // Use the location from the module op
    let location = ctx.op(module.op()).location;

    // Create wasm.table for closure functions
    let table_op = wasm_dialect::table(
        ctx,
        location,
        Symbol::new("funcref"),
        table_size,
        Some(table_size),
    );

    // Create wasm.ref_func operations for each function in the element segment
    let funcref_ty = intern_funcref_type(ctx);
    let func_ref_ops: Vec<OpRef> = funcs
        .iter()
        .map(|func_sym| wasm_dialect::ref_func(ctx, location, funcref_ty, *func_sym).op_ref())
        .collect();

    // Create the funcs region for wasm.elem
    let funcs_block = ctx.create_block(BlockData {
        location,
        args: vec![],
        ops: smallvec![],
        parent_region: None,
    });
    for ref_op in &func_ref_ops {
        ctx.push_op(funcs_block, *ref_op);
    }
    let funcs_region = ctx.create_region(RegionData {
        location,
        blocks: smallvec![funcs_block],
        parent_op: None,
    });

    // Create wasm.elem with table 0 and offset 0
    let elem_op = wasm_dialect::elem(ctx, location, Some(0), Some(0), funcs_region);

    // Prepend table and elem operations to the module body.
    // We insert before the first op in the block (if any), or push at the end.
    let existing_ops: Vec<OpRef> = ctx.block(first_block).ops.to_vec();
    if let Some(&first_op) = existing_ops.first() {
        ctx.insert_op_before(first_block, first_op, table_op.op_ref());
        ctx.insert_op_before(first_block, first_op, elem_op.op_ref());
    } else {
        ctx.push_op(first_block, table_op.op_ref());
        ctx.push_op(first_block, elem_op.op_ref());
    }
}

/// Pattern for `func.func` -> `wasm.func`
struct FuncFuncPattern;

impl RewritePattern for FuncFuncPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(func_op) = func::Func::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let sym_name = func_op.sym_name(ctx);
        let func_type = func_op.r#type(ctx);
        let attrs_to_preserve = clone_attrs_except(ctx, op, &["sym_name", "type"]);

        // Generated Func::body() asserts for valid bodyless declarations.
        // A func.func conversion accepts exactly zero or one body region.
        let body = match ctx.op(op).regions.as_slice() {
            [] => None,
            [body] => Some(*body),
            _ => return false,
        };

        // Detach a definition's body region so it can be reused in the new wasm.func.
        if let Some(body) = body {
            ctx.detach_region(body);
        }

        let new_op = match body {
            Some(body) => wasm_dialect::func(ctx, loc, sym_name, func_type, body).op_ref(),
            None => {
                let data = OperationDataBuilder::new(loc, Symbol::new("wasm"), Symbol::new("func"))
                    .attr("sym_name", Attribute::Symbol(sym_name))
                    .attr("type", Attribute::Type(func_type))
                    .build(ctx);
                ctx.create_op(data)
            }
        };
        ctx.op_mut(new_op).attributes.extend(attrs_to_preserve);
        rewriter.replace_op(new_op);
        true
    }
}

/// Pattern for `func.call` -> `wasm.call`
struct FuncCallPattern;

impl RewritePattern for FuncCallPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(call_op) = func::Call::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let callee = call_op.callee(ctx);
        let args: Vec<_> = ctx.op_operands(op).to_vec();
        let result_types: Vec<TypeRef> = ctx.op_result_types(op).to_vec();

        let new_op = wasm_dialect::call(ctx, loc, args, result_types, callee);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// Pattern for `func.call_indirect` -> `wasm.call_indirect`
///
/// Transforms indirect function calls for closures.
/// The callee (i32 table index) is the first operand, followed by arguments.
struct FuncCallIndirectPattern;

impl RewritePattern for FuncCallIndirectPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(call) = func::CallIndirect::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let all_operands = std::iter::once(call.callee(ctx))
            .chain(CallLike::call_args(&call, ctx).iter().copied())
            .collect::<Vec<_>>();
        let result_types = CallLike::call_result_types(&call, ctx).to_vec();

        let signature = if let Some(signature) = call.exact_signature(ctx) {
            let Some(signature) = convert_function_type(ctx, signature, rewriter.type_converter())
            else {
                return false;
            };
            if crate::emit::helpers::exact_call_indirect_signature_with(ctx, op, signature).is_err()
            {
                return false;
            }
            Some(signature)
        } else {
            None
        };

        // The emit phase resolves the type index and table attributes. An
        // optional exact signature stays attached for authoritative indexing.
        let new_op =
            wasm_dialect::call_indirect(ctx, loc, all_operands, result_types, 0, 0, signature);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// Pattern for `func.return` -> `wasm.return`
struct FuncReturnPattern;

impl RewritePattern for FuncReturnPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(_return_op) = func::Return::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let values: Vec<_> = ctx.op_operands(op).to_vec();

        let new_op = wasm_dialect::r#return(ctx, loc, values);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// Pattern for `func.tail_call` -> `wasm.return_call`
struct FuncTailCallPattern;

impl RewritePattern for FuncTailCallPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(tail_call_op) = func::TailCall::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let callee = tail_call_op.callee(ctx);
        let args: Vec<_> = ctx.op_operands(op).to_vec();

        let new_op = wasm_dialect::return_call(ctx, loc, args, callee);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// Pattern for `func.tail_call_indirect` -> `wasm.return_call_indirect`.
///
/// The shared physical-ABI boundary retains the precise callee signature in
/// `signature` after closure lowering has replaced the
/// typed closure with a table index.  This backend must carry that metadata
/// through instead of rebuilding a signature from the runtime operands.
struct FuncTailCallIndirectPattern;

impl RewritePattern for FuncTailCallIndirectPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(tail) = func::TailCallIndirect::from_op(ctx, op) else {
            return false;
        };

        // The exact callable metadata is source-authoritative, but the
        // arguments have already reached their Wasm representations. Convert
        // the attribute with the shared signature converter before validating
        // it; never derive a replacement signature from the operands.
        let Some(signature) = tail.exact_signature(ctx) else {
            return false;
        };
        let Some(signature) = convert_function_type(ctx, signature, rewriter.type_converter())
        else {
            return false;
        };

        let mut converted_attrs = ctx.op(op).attributes.clone();
        func::remove_indirect_call_signature(&mut converted_attrs);
        converted_attrs.remove("table");
        converted_attrs.remove("type_idx");

        // This is intentionally checked before creating any replacement.
        // Missing or malformed metadata remains a residual `func.*` op and is
        // rejected by the Wasm readiness boundary rather than guessed from
        // table-index and argument values.
        if crate::emit::helpers::exact_return_call_indirect_signature_with(ctx, op, signature)
            .is_err()
        {
            return false;
        }

        let operands = std::iter::once(tail.callee(ctx))
            .chain(CallLike::call_args(&tail, ctx).iter().copied())
            .collect::<Vec<_>>();
        if !TailCallLike::is_resultless(&tail, ctx) {
            return false;
        }

        let loc = ctx.op(op).location;
        let new_op = wasm_dialect::return_call_indirect(ctx, loc, operands, 0, 0, Some(signature));
        ctx.op_mut(new_op.op_ref())
            .attributes
            .extend(converted_attrs);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// Pattern for `func.unreachable` -> `wasm.unreachable`
struct FuncUnreachablePattern;

impl RewritePattern for FuncUnreachablePattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(_unreachable_op) = func::Unreachable::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;

        let new_op = wasm_dialect::unreachable(ctx, loc);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// Pattern for `func.constant` -> `wasm.i32_const` (i32 table index)
///
/// Transforms function constant references to i32 table indices.
/// Used for closures where lifted functions are stored via function table.
struct FuncConstantPattern {
    table_indices: HashMap<Symbol, u32>,
}

impl RewritePattern for FuncConstantPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(const_op) = func::Constant::from_op(ctx, op) else {
            return false;
        };

        let func_ref = const_op.func_ref(ctx);

        // All func.constant operations must be registered in the function table.
        // They are collected by collect_func_constant_refs before pattern application.
        let table_idx = self
            .table_indices
            .get(&func_ref)
            .copied()
            .expect("All func.constant must be registered in table");

        let loc = ctx.op(op).location;
        let i32_ty = intern_i32_type(ctx);
        let new_op = wasm_dialect::i32_const(ctx, loc, i32_ty, table_idx as i32);

        rewriter.replace_op(new_op.op_ref());
        true
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Intern a core.i32 type.
fn intern_i32_type(ctx: &mut IrContext) -> TypeRef {
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build())
}

/// Intern a wasm.funcref type.
fn intern_funcref_type(ctx: &mut IrContext) -> TypeRef {
    wasm_dialect::funcref(ctx).as_type_ref()
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;
    use trunk_ir::types::Attribute;

    #[test]
    fn func_to_wasm_preserves_custom_function_attributes() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @main() -> core.nil {
    func.return
  }
}"#,
        );
        let original = module.ops(&ctx)[0];
        ctx.op_mut(original)
            .attributes
            .insert(Symbol::new("custom"), Attribute::Int(7));

        lower(&mut ctx, module, TypeConverter::new());

        let lowered = module.ops(&ctx)[0];
        assert!(wasm_dialect::Func::from_op(&ctx, lowered).is_ok());
        assert_eq!(
            ctx.op(lowered).attributes.get("custom"),
            Some(&Attribute::Int(7))
        );
    }

    #[test]
    fn func_to_wasm_preserves_bodyless_declarations_and_definition_bodies() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @external(%value: core.i32) -> core.i32
  func.func @defined(%value: core.i32) -> core.i32 {
    func.return %value
  }
}"#,
        );
        let original_ops = module.ops(&ctx);
        let declaration_location = ctx.op(original_ops[0]).location;
        let definition_location = ctx.op(original_ops[1]).location;
        ctx.op_mut(original_ops[0])
            .attributes
            .insert(Symbol::new("custom"), Attribute::Int(7));

        lower(&mut ctx, module, TypeConverter::new());

        let lowered_ops = module.ops(&ctx);
        assert_eq!(lowered_ops.len(), 2);
        let declaration = wasm_dialect::Func::from_op(&ctx, lowered_ops[0])
            .expect("bodyless declaration should lower to wasm.func");
        let definition = wasm_dialect::Func::from_op(&ctx, lowered_ops[1])
            .expect("definition should lower to wasm.func");
        assert_eq!(declaration.sym_name(&ctx), Symbol::new("external"));
        assert_eq!(definition.sym_name(&ctx), Symbol::new("defined"));
        assert_eq!(ctx.op(lowered_ops[0]).location, declaration_location);
        assert_eq!(ctx.op(lowered_ops[1]).location, definition_location);
        assert!(ctx.op(lowered_ops[0]).regions.is_empty());
        assert_eq!(ctx.op(lowered_ops[1]).regions.len(), 1);
        assert_eq!(
            ctx.op(lowered_ops[0]).attributes.get("custom"),
            Some(&Attribute::Int(7))
        );

        let output = print_module(&ctx, module.op());
        assert!(
            output.contains(
                "wasm.func {custom = 7, sym_name = @external, type = core.func(core.i32, core.i32)}"
            ),
            "{output}"
        );
        assert!(
            output.contains("wasm.func {sym_name = @defined"),
            "{output}"
        );
        assert!(output.contains("wasm.return %0"), "{output}");
    }

    #[test]
    fn func_to_wasm_leaves_impossible_multi_region_functions_unconverted() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @invalid() -> core.nil {
    func.return
  }
}"#,
        );
        let invalid = module.ops(&ctx)[0];
        let location = ctx.op(invalid).location;
        let extra_region = ctx.create_region(RegionData {
            location,
            blocks: Default::default(),
            parent_op: None,
        });
        ctx.op_mut(invalid).regions.push(extra_region);

        lower(&mut ctx, module, TypeConverter::new());

        let invalid = module.ops(&ctx)[0];
        assert!(func::Func::from_op(&ctx, invalid).is_ok());
        assert_eq!(ctx.op(invalid).regions.len(), 2);
    }

    #[test]
    fn lowers_direct_and_indirect_proper_tail_transfers() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @target(%value: core.i32) -> core.nil {
    func.return
  }
  func.func @direct(%value: core.i32) -> core.nil {
    func.tail_call %value {callee = @target}
  }
  func.func @indirect(%table_index: core.i32, %value: core.i32) -> core.nil {
    func.tail_call_indirect %table_index, %value {signature = core.func(core.nil, core.i32)}
  }
  func.func @ordinary(%table_index: core.i32, %value: core.i32) -> core.i32 {
    %result = func.call_indirect %table_index, %value : core.i32
    func.return %result
  }
}"#,
        );

        lower(&mut ctx, module, TypeConverter::new());

        let output = print_module(&ctx, module.op());
        assert!(
            output.contains("wasm.return_call %0 {callee = @target}"),
            "{output}"
        );
        assert!(
            output.contains("wasm.return_call_indirect %0, %1"),
            "{output}"
        );
        assert!(output.contains(" = wasm.call_indirect "), "{output}");
        assert!(
            output.contains("signature = core.func(core.nil, core.i32)"),
            "{output}"
        );
    }

    #[test]
    fn lowers_exact_unit_call_indirect_without_an_ssa_result() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @caller(%table_index: core.i32) -> core.nil {
    func.call_indirect %table_index {signature = core.func(core.nil)}
    func.return
  }
}"#,
        );

        lower(&mut ctx, module, TypeConverter::new());

        let output = print_module(&ctx, module.op());
        assert!(
            output.contains(
                "wasm.call_indirect %0 {signature = core.func(core.nil), table = 0, type_idx = 0}"
            ),
            "{output}"
        );
        assert!(!output.contains("func.call_indirect"), "{output}");
    }

    #[test]
    fn lowers_indirect_tail_transfer_with_a_function_table_entry() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @target(%value: core.i32) -> core.nil {
    func.return
  }
  func.func @caller(%value: core.i32) -> core.nil {
    %table_index = func.constant {func_ref = @target} : core.i32
    func.tail_call_indirect %table_index, %value {signature = core.func(core.nil, core.i32)}
  }
}"#,
        );

        lower(&mut ctx, module, TypeConverter::new());

        let output = print_module(&ctx, module.op());
        assert!(output.contains("wasm.table"), "{output}");
        assert!(output.contains("wasm.elem"), "{output}");
        assert!(
            output.contains("wasm.ref_func {func_name = @target}"),
            "{output}"
        );
        assert!(output.contains("wasm.i32_const {value = 0}"), "{output}");
        assert!(output.contains("wasm.return_call_indirect"), "{output}");
    }

    #[test]
    fn converts_exact_tail_signature_before_validating_physical_operands() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @physical(%table_index: core.i32, %value: wasm.anyref) -> core.nil {
    func.tail_call_indirect %table_index, %value {signature = core.func(core.nil, tribute_rt.anyref), table = 7, tribute.calling_convention = 2, type_idx = 9}
  }
  func.func @mismatch(%table_index: core.i32, %value: core.i32) -> core.nil {
    func.tail_call_indirect %table_index, %value {signature = core.func(core.nil, tribute_rt.float), tribute.calling_convention = 2}
  }
}"#,
        );

        let anyref_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("wasm"), Symbol::new("anyref")).build());
        let f64_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("f64")).build());
        let mut type_converter = TypeConverter::new();
        type_converter.add_conversion(move |ctx, ty| {
            (ctx.types
                .is_dialect(ty, Symbol::new("tribute_rt"), Symbol::new("anyref")))
            .then_some(anyref_ty)
        });
        type_converter.add_conversion(move |ctx, ty| {
            (ctx.types
                .is_dialect(ty, Symbol::new("tribute_rt"), Symbol::new("float")))
            .then_some(f64_ty)
        });
        lower(&mut ctx, module, type_converter);

        let output = print_module(&ctx, module.op());
        assert!(
            output.contains(
                "wasm.return_call_indirect %0, %1 {signature = core.func(core.nil, wasm.anyref)"
            ),
            "{output}"
        );
        assert!(
            output.contains("wasm.return_call_indirect %0, %1 {signature = core.func(core.nil, wasm.anyref), table = 0, tribute.calling_convention = 2, type_idx = 0}"),
            "the converted transfer must retain its calling convention: {output}"
        );
        assert!(
            !output.contains("signature = core.func(core.nil, tribute_rt.anyref)"),
            "{output}"
        );
        assert!(
            !output.contains("table = 7") && !output.contains("type_idx = 9"),
            "stale source table metadata must not reach Wasm: {output}"
        );
        assert!(
            output.contains("func.tail_call_indirect %0, %1 {signature = core.func(core.nil, tribute_rt.float), tribute.calling_convention = 2}"),
            "the mismatched transfer must remain unchanged: {output}"
        );
    }

    #[test]
    fn converts_exact_indirect_signature_before_validating_physical_operands() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @physical(%table_index: core.i32, %value: wasm.anyref) -> core.i32 {
    %result = func.call_indirect %table_index, %value {signature = core.func(core.i32, tribute_rt.anyref), table = 7, type_idx = 9} : core.i32
    func.return %result
  }
  func.func @mismatch(%table_index: core.i32, %value: core.i32) -> core.i32 {
    %result = func.call_indirect %table_index, %value {signature = core.func(core.i32, tribute_rt.float)} : core.i32
    func.return %result
  }
}"#,
        );

        let anyref_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("wasm"), Symbol::new("anyref")).build());
        let f64_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("f64")).build());
        let mut type_converter = TypeConverter::new();
        type_converter.add_conversion(move |ctx, ty| {
            (ctx.types
                .is_dialect(ty, Symbol::new("tribute_rt"), Symbol::new("anyref")))
            .then_some(anyref_ty)
        });
        type_converter.add_conversion(move |ctx, ty| {
            (ctx.types
                .is_dialect(ty, Symbol::new("tribute_rt"), Symbol::new("float")))
            .then_some(f64_ty)
        });
        lower(&mut ctx, module, type_converter);

        let output = print_module(&ctx, module.op());
        assert!(
            output.contains(
                "wasm.call_indirect %0, %1 {signature = core.func(core.i32, wasm.anyref), table = 0, type_idx = 0}"
            ),
            "{output}"
        );
        assert!(
            !output.contains("table = 7") && !output.contains("type_idx = 9"),
            "stale source table metadata must not reach Wasm: {output}"
        );
        assert!(
            !output.contains("signature = core.func(core.i32, tribute_rt.anyref)"),
            "{output}"
        );
        assert!(
            output.contains(
                "func.call_indirect %0, %1 {signature = core.func(core.i32, tribute_rt.float)} : core.i32"
            ),
            "the mismatched call must remain unchanged: {output}"
        );
    }

    #[test]
    fn leaves_indirect_tail_transfers_without_an_exact_empty_signature_unconverted() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @indirect(%table_index: core.i32, %value: core.i32) -> core.nil {
    func.tail_call_indirect %table_index, %value
  }
  func.func @malformed(%table_index: core.i32, %value: core.i32) -> core.nil {
    func.tail_call_indirect %table_index, %value {signature = core.func(core.i32, core.i32)}
  }
}"#,
        );

        lower(&mut ctx, module, TypeConverter::new());

        let output = print_module(&ctx, module.op());
        assert_eq!(
            output.matches("func.tail_call_indirect").count(),
            2,
            "{output}"
        );
        assert!(!output.contains("wasm.return_call_indirect"), "{output}");
    }
}
