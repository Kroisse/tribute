//! Native entrypoint generation pass.
//!
//! Root bridge composition inside the representation/ABI boundary leaves a
//! parameterless wrapper `main` that nothing in the module references. This
//! pass adapts that wrapper in place into the C ABI `main`: it initializes the
//! runtime first and returns exit code 0.

use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::arith;
use trunk_ir::dialect::core;
use trunk_ir::dialect::func;
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{BlockRef, OpRef};
use trunk_ir::rewrite::Module;
use trunk_ir::types::{Attribute, Location, TypeDataBuilder};

/// Adapt the root wrapper `main` into the native C ABI entrypoint.
///
/// This pass:
/// 1. Ensures `__tribute_init` (and optionally `__asan_init`) declarations exist
/// 2. Calls them at the start of `main`
/// 3. Retypes `main` to `() -> i32` and makes each `func.return` return 0
pub fn generate_native_entrypoint(ctx: &mut IrContext, module: Module, sanitize: bool) {
    let Some(first_block) = module.first_block(ctx) else {
        return;
    };

    let loc = ctx.op(module.op()).location;
    let main_sym = Symbol::new("main");
    let init_sym = Symbol::new("__tribute_init");
    let asan_init_sym = Symbol::new("__asan_init");

    let ops = &ctx.block(first_block).ops;
    let mut main_op = None;
    let mut has_tribute_init = false;
    let mut has_asan_init = false;
    for &op in ops {
        if let Ok(func_op) = func::Func::from_op(ctx, op) {
            let name = func_op.sym_name(ctx);
            if name == main_sym {
                main_op = Some(func_op);
            }
            if name == init_sym {
                has_tribute_init = true;
            }
            if name == asan_init_sym {
                has_asan_init = true;
            }
        }
    }

    let Some(main) = main_op else {
        tracing::warn!("No main function found; skipping entrypoint generation");
        return;
    };

    // Root bridge composition consumes the source convention and exposes a
    // `main` without hidden parameters.
    let signature = func::FuncSig::from_type_ref(ctx, main.r#type(ctx))
        .expect("entrypoint: `main` must have a func.func_sig type");
    assert!(
        signature.inputs(ctx).is_empty(),
        "entrypoint: root `main` must have no hidden parameters after the entry bridge"
    );
    let body = ctx
        .op_region(main.op_ref(), 0)
        .expect("entrypoint: root `main` must be a definition");
    let blocks = ctx.region(body).blocks.clone();
    let entry = *blocks
        .first()
        .expect("entrypoint: root `main` must have an entry block");

    let nil_ty = core::nil(ctx).as_type_ref();
    let i32_ty = ctx.intern_type(TypeDataBuilder::new("core", "i32").build());

    if !has_tribute_init {
        let init_op = super::build_extern_func(ctx, loc, "__tribute_init", &[], nil_ty);
        ctx.insert_op_before(first_block, ctx.block(first_block).ops[0], init_op);
    }
    if sanitize && !has_asan_init {
        let asan_op = super::build_extern_func(ctx, loc, "__asan_init", &[], nil_ty);
        ctx.insert_op_before(first_block, ctx.block(first_block).ops[0], asan_op);
    }

    // Initialize ASan before anything else, then the runtime TLS before any
    // ability use.
    let mut init_calls = Vec::new();
    if sanitize {
        init_calls.push(asan_init_sym);
    }
    init_calls.push(init_sym);
    for callee in init_calls.into_iter().rev() {
        let call = func::Call::operands([])
            .callee(callee)
            .results([nil_ty])
            .build(ctx, loc);
        prepend_op(ctx, entry, call.op_ref());
    }

    // The source-level result is Nil; the process exit code is always 0.
    for block in blocks {
        let returns: Vec<OpRef> = ctx
            .block(block)
            .ops
            .iter()
            .copied()
            .filter(|&op| func::Return::matches(ctx, op))
            .collect();
        for ret in returns {
            let location = ctx.op(ret).location;
            replace_with_exit_code(ctx, block, ret, i32_ty, location);
        }
    }

    let func_ty = func::func_sig(ctx, [], [i32_ty]).as_type_ref();
    ctx.op_mut(main.op_ref())
        .attributes
        .insert(Symbol::new("type"), Attribute::Type(func_ty));
}

fn prepend_op(ctx: &mut IrContext, block: BlockRef, op: OpRef) {
    match ctx.block(block).ops.first().copied() {
        Some(first) => ctx.insert_op_before(block, first, op),
        None => ctx.push_op(block, op),
    }
}

fn replace_with_exit_code(
    ctx: &mut IrContext,
    block: BlockRef,
    ret: OpRef,
    i32_ty: trunk_ir::refs::TypeRef,
    location: Location,
) {
    let zero = arith::Const::operands()
        .value(Attribute::Int(0))
        .results(i32_ty)
        .build(ctx, location);
    ctx.insert_op_before(block, ret, zero.op_ref());
    let exit = func::Return::operands([zero.result(ctx)]).build(ctx, location);
    ctx.insert_op_before(block, ret, exit.op_ref());
    ctx.remove_op_from_block(block, ret);
    ctx.remove_op(ret);
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;

    fn adapt(input: &str, sanitize: bool) -> String {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        generate_native_entrypoint(&mut ctx, module, sanitize);
        print_module(&ctx, module.op())
    }

    const WRAPPER_MAIN: &str = r#"core.module @test {
  func.func @__tribute_main() -> core.nil {
    %nil = core.nil_value : core.nil
    func.return %nil
  }
  func.func @main() -> core.nil {
    %result = func.call {callee = @__tribute_main} : core.nil
    func.return %result
  }
}"#;

    #[test]
    fn wrapper_main_becomes_the_c_entrypoint_in_place() {
        let printed = adapt(WRAPPER_MAIN, false);

        assert_eq!(
            printed,
            r#"core.module @test {
  func.func @__tribute_init() -> core.nil attributes {abi = "C"}
  func.func @__tribute_main() -> core.nil {
      %0 = core.nil_value : core.nil
      func.return %0
  }
  func.func @main() -> core.i32 {
      %0 = func.call {callee = @__tribute_init} : core.nil
      %1 = func.call {callee = @__tribute_main} : core.nil
      %2 = arith.const {value = 0} : core.i32
      func.return %2
  }
}
"#
        );
    }

    #[test]
    fn sanitizer_initialization_runs_first() {
        let printed = adapt(WRAPPER_MAIN, true);

        let asan = printed
            .find("func.call {callee = @__asan_init}")
            .expect("asan init call");
        let init = printed
            .find("func.call {callee = @__tribute_init}")
            .expect("runtime init call");
        let worker = printed
            .find("func.call {callee = @__tribute_main}")
            .expect("worker call");
        assert!(asan < init && init < worker, "{printed}");
        assert!(printed.contains("func.func @__asan_init()"), "{printed}");
    }

    #[test]
    fn module_without_main_is_unchanged() {
        let input = r#"core.module @test {
  func.func @helper() -> core.i32 {
    %one = arith.const {value = 1} : core.i32
    func.return %one
  }
}"#;
        let printed = adapt(input, false);

        assert!(!printed.contains("__tribute_init"), "{printed}");
        assert!(
            printed.contains("func.func @helper() -> core.i32"),
            "{printed}"
        );
    }

    #[test]
    #[should_panic(expected = "must have no hidden parameters")]
    fn main_with_hidden_parameters_is_rejected() {
        adapt(
            r#"core.module @test {
  func.func @main(%evidence: core.i32) -> core.nil {
    func.return
  }
}"#,
            false,
        );
    }
}
