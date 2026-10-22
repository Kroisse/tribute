//! Wasm prompt tag allocator.
//!
//! Evidence lowering itself is shared by the targets
//! ([`crate::evidence_lowering`]). After the boundary exit,
//! [`bind_wasm_prompt_tag_runtime`] gives the prompt tag allocator
//! `__tribute_next_tag` its Wasm implementation.

use trunk_ir::context::{BlockArgData, BlockData, IrContext, RegionData};
use trunk_ir::dialect::{func, wasm as wasm_dialect};
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{OpRef, TypeRef};
use trunk_ir::rewrite::Module;
use trunk_ir::smallvec::smallvec;
use trunk_ir::types::{Attribute, Location};

/// C link name of the helper that allocates a fresh prompt tag.
pub const NEXT_TAG: &str = "__tribute_next_tag";

/// Replace a bodyless `__tribute_next_tag` declaration with its Wasm
/// implementation: a mutable `i32` counter global that the function returns
/// and increments.
///
/// Runs after Wasm dialect lowering has converted the declaration's
/// signature; the implementation adopts it.
pub fn bind_wasm_prompt_tag_runtime(ctx: &mut IrContext, module: Module) {
    let Some(block) = module.first_block(ctx) else {
        return;
    };
    for op in module.ops_snapshot(ctx) {
        let is_function = wasm_dialect::Func::matches(ctx, op) || func::Func::matches(ctx, op);
        if !is_function || ctx.op_has_regions(op) {
            continue;
        }
        let attrs = ctx.op_attrs(op);
        if attrs.get_str("sym_name") != Some(NEXT_TAG) {
            continue;
        }
        let name = attrs
            .get_string_ref("sym_name")
            .expect("a declaration has a name");
        let signature = attrs
            .get_type("type")
            .expect("a function declaration has a signature");
        let location = ctx.op(op).location;
        let global = add_tag_counter(ctx, module, location);
        let implementation = next_tag_function(ctx, location, name, signature, global);
        ctx.insert_op_before(block, op, implementation);
        ctx.remove_op_from_block(block, op);
        ctx.remove_op(op);
    }
}

/// Append a mutable `i32` tag counter global, returning its index.
fn add_tag_counter(ctx: &mut IrContext, module: Module, location: Location) -> u32 {
    let index = module
        .ops(ctx)
        .iter()
        .filter(|&&op| wasm_dialect::Global::matches(ctx, op))
        .count() as u32;
    let global = wasm_dialect::Global::operands()
        .valtype("i32")
        .mutable(true)
        .init(Attribute::Int(0))
        .build(ctx, location);
    let block = module
        .first_block(ctx)
        .expect("a module with declarations has a body block");
    ctx.push_op(block, global.op_ref());
    index
}

/// The `wasm.func` that returns the counter `global` and increments it, with
/// the declaration's (converted) signature.
fn next_tag_function(
    ctx: &mut IrContext,
    location: Location,
    name: trunk_ir::StringRef,
    signature: TypeRef,
    global: u32,
) -> OpRef {
    let (inputs, results) = if let Some(sig) = wasm_dialect::FuncSig::from_type_ref(ctx, signature)
    {
        (sig.inputs(ctx).to_vec(), sig.results(ctx).to_vec())
    } else {
        let sig = func::FuncSig::from_type_ref(ctx, signature)
            .expect("runtime helper declaration must have a function signature");
        (sig.inputs(ctx).to_vec(), sig.results(ctx).to_vec())
    };
    let result_ty = *results
        .first()
        .expect("runtime helper declaration must have one result");
    let block = ctx.create_block(BlockData {
        location,
        args: inputs
            .iter()
            .map(|&ty| BlockArgData {
                ty,
                attrs: Default::default(),
            })
            .collect(),
        ops: smallvec![],
        parent_region: None,
    });
    let current = wasm_dialect::GlobalGet::operands()
        .index(global)
        .results(result_ty)
        .build(ctx, location);
    ctx.push_op(block, current.op_ref());
    let one = wasm_dialect::I32Const::operands()
        .value(1)
        .results(result_ty)
        .build(ctx, location);
    ctx.push_op(block, one.op_ref());
    let next =
        wasm_dialect::I32Add::operands(current.result(ctx), one.result(ctx)).build(ctx, location);
    ctx.push_op(block, next.op_ref());
    let set = wasm_dialect::GlobalSet::operands(next.result(ctx))
        .index(global)
        .build(ctx, location);
    ctx.push_op(block, set.op_ref());
    let ret = wasm_dialect::Return::operands(vec![current.result(ctx)]).build(ctx, location);
    ctx.push_op(block, ret.op_ref());

    let body = ctx.create_region(RegionData {
        location,
        blocks: smallvec![block],
        parent_op: None,
    });
    let func_ty = wasm_dialect::func_sig(ctx, inputs, [result_ty]).as_type_ref();
    wasm_dialect::Func::operands()
        .sym_name(name)
        .r#type(func_ty)
        .regions(body)
        .build(ctx, location)
        .op_ref()
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;

    #[test]
    fn next_tag_counts_in_a_global_appended_after_existing_ones() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  wasm.global {valtype = "i32", mutable = false, init = 7}
  func.func @__tribute_next_tag() -> core.i32 attributes {abi = "C"}
}"#,
        );

        bind_wasm_prompt_tag_runtime(&mut ctx, module);

        assert_eq!(
            print_module(&ctx, module.op()),
            r#"core.module @test {
  wasm.global {init = 7, mutable = false, valtype = "i32"}
  wasm.func {sym_name = "__tribute_next_tag", type = wasm.func_sig<() -> core.i32>} {
      %0 = wasm.global_get {index = 1} : core.i32
      %1 = wasm.i32_const {value = 1} : core.i32
      %2 = wasm.i32_add %0, %1 : core.i32
      wasm.global_set %2 {index = 1}
      wasm.return %0
  }
  wasm.global {init = 0, mutable = true, valtype = "i32"}
}
"#
        );
    }

    #[test]
    fn binding_leaves_modules_without_a_tag_declaration_unchanged() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @main() -> core.nil {
    %nil = core.nil_value : core.nil
    func.return %nil
  }
}"#,
        );
        let before = print_module(&ctx, module.op());

        bind_wasm_prompt_tag_runtime(&mut ctx, module);

        assert_eq!(print_module(&ctx, module.op()), before);
    }
}
