//! Native entrypoint generation pass.
//!
//! Generates a C ABI `main` entrypoint that wraps the user's `main` function.
//! The user's `main` is renamed to `_tribute_main`, and a new `main` is created
//! that calls it and returns exit code 0.

use trunk_ir::Symbol;
use trunk_ir::arena::context::{BlockData, IrContext, RegionData};
use trunk_ir::arena::dialect::arith as arena_arith;
use trunk_ir::arena::dialect::func as arena_func;
use trunk_ir::arena::ops::ArenaDialectOp;
use trunk_ir::arena::refs::{BlockRef, OpRef, RegionRef, TypeRef};
use trunk_ir::arena::rewrite::ArenaModule;
use trunk_ir::arena::types::{
    Attribute as ArenaAttribute, Location as ArenaLocation, TypeDataBuilder,
};
use trunk_ir::smallvec::smallvec;

/// Generate a native C ABI entrypoint wrapper for the user's `main` function.
/// (Arena IR version — mutates `ctx` in-place.)
///
/// This pass:
/// 1. Renames `func.func @main` to `func.func @_tribute_main`
/// 2. Rewrites all `callee = @main` / `func_ref = @main` to `@_tribute_main`
/// 3. Ensures `__tribute_init` (and optionally `__asan_init`) declarations exist
/// 4. Creates a new `func.func @main() -> i32` that calls `_tribute_main()`
///    and returns 0
pub fn generate_native_entrypoint(ctx: &mut IrContext, module: ArenaModule, sanitize: bool) {
    let first_block = match module.first_block(ctx) {
        Some(b) => b,
        None => return,
    };

    let loc = ctx.op(module.op()).location;
    let main_sym = Symbol::new("main");
    let tribute_main_sym = Symbol::new("_tribute_main");

    // Scan module-level ops to find main function and check for existing declarations
    let ops: Vec<OpRef> = ctx.block(first_block).ops.to_vec();
    let mut found_main = false;
    let mut has_tribute_main = false;
    let mut main_return_ty: Option<TypeRef> = None;
    let mut has_tribute_init = false;
    let mut has_asan_init = false;

    let init_sym = Symbol::new("__tribute_init");
    let asan_init_sym = Symbol::new("__asan_init");

    for &op in &ops {
        if let Ok(func_op) = arena_func::Func::from_op(ctx, op) {
            let name = func_op.sym_name(ctx);
            if name == main_sym {
                found_main = true;
                // Extract return type from func type
                let func_ty = func_op.r#type(ctx);
                let type_data = ctx.types.get(func_ty);
                // core.func layout: params[0] = result
                if !type_data.params.is_empty() {
                    main_return_ty = Some(type_data.params[0]);
                }
            }
            if name == tribute_main_sym {
                has_tribute_main = true;
            }
            if name == init_sym {
                has_tribute_init = true;
            }
            if name == asan_init_sym {
                has_asan_init = true;
            }
        }
    }

    if !found_main {
        tracing::warn!("No main function found; skipping entrypoint generation");
        return;
    }

    if has_tribute_main {
        tracing::warn!(
            "`_tribute_main` already exists in module; skipping `main` rename to avoid collision"
        );
        return;
    }

    // Intern needed types
    let i32_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());
    let nil_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("nil")).build());

    let tribute_main_return_ty = main_return_ty.unwrap_or(nil_ty);

    // Step 1: Rename main -> _tribute_main (in-place attribute mutation)
    for &op in &ops {
        if let Ok(func_op) = arena_func::Func::from_op(ctx, op)
            && func_op.sym_name(ctx) == main_sym
        {
            ctx.op_mut(op).attributes.insert(
                Symbol::new("sym_name"),
                ArenaAttribute::Symbol(tribute_main_sym),
            );
        }
    }

    // Step 2: Rewrite all @main references to @_tribute_main in all ops
    for &op in &ops {
        rewrite_symbol_refs(ctx, op, main_sym, tribute_main_sym);
    }

    // Step 3: Ensure __tribute_init is declared
    if !has_tribute_init {
        let init_op = super::build_extern_func(ctx, loc, "__tribute_init", &[], nil_ty);
        ctx.insert_op_before(first_block, ctx.block(first_block).ops[0], init_op);
    }

    // Declare __asan_init when AddressSanitizer is enabled
    if sanitize && !has_asan_init {
        let asan_op = super::build_extern_func(ctx, loc, "__asan_init", &[], nil_ty);
        ctx.insert_op_before(first_block, ctx.block(first_block).ops[0], asan_op);
    }

    // Step 4: Build and append C ABI entrypoint wrapper
    let entrypoint_op =
        build_entrypoint(ctx, loc, tribute_main_return_ty, i32_ty, nil_ty, sanitize);
    ctx.push_op(first_block, entrypoint_op);
}

/// Recursively rewrite symbol references from `old_sym` to `new_sym` in an
/// operation and all its nested regions.
///
/// Rewrites `callee` attributes (on func.call / func.tail_call) and
/// `func_ref` attributes (on func.constant).
fn rewrite_symbol_refs(ctx: &mut IrContext, op: OpRef, old_sym: Symbol, new_sym: Symbol) {
    // Rewrite callee attribute
    let callee_key = Symbol::new("callee");
    let func_ref_key = Symbol::new("func_ref");

    if let Some(ArenaAttribute::Symbol(sym)) = ctx.op(op).attributes.get(&callee_key).cloned()
        && sym == old_sym
    {
        ctx.op_mut(op)
            .attributes
            .insert(callee_key, ArenaAttribute::Symbol(new_sym));
    }

    // Rewrite func_ref attribute
    if let Some(ArenaAttribute::Symbol(sym)) = ctx.op(op).attributes.get(&func_ref_key).cloned()
        && sym == old_sym
    {
        ctx.op_mut(op)
            .attributes
            .insert(func_ref_key, ArenaAttribute::Symbol(new_sym));
    }

    // Recurse into regions
    let regions: Vec<RegionRef> = ctx.op(op).regions.to_vec();
    for region in regions {
        rewrite_symbol_refs_in_region(ctx, region, old_sym, new_sym);
    }
}

fn rewrite_symbol_refs_in_region(
    ctx: &mut IrContext,
    region: RegionRef,
    old_sym: Symbol,
    new_sym: Symbol,
) {
    let blocks: Vec<BlockRef> = ctx.region(region).blocks.to_vec();
    for block in blocks {
        let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();
        for op in ops {
            rewrite_symbol_refs(ctx, op, old_sym, new_sym);
        }
    }
}

/// Build the C ABI entrypoint function:
///
/// ```text
/// func.func @main() -> i32 {
///     call @__asan_init()       // if sanitize
///     call @__tribute_init()
///     call @_tribute_main()
///     %zero = arith.const 0 : i32
///     return %zero
/// }
/// ```
fn build_entrypoint(
    ctx: &mut IrContext,
    loc: ArenaLocation,
    tribute_main_return_ty: TypeRef,
    i32_ty: TypeRef,
    nil_ty: TypeRef,
    sanitize: bool,
) -> OpRef {
    // Build func type: () -> i32
    let func_ty = ctx.types.intern(
        TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func"))
            .param(i32_ty)
            .build(),
    );

    // Create entry block
    let entry_block = ctx.create_block(BlockData {
        location: loc,
        args: vec![],
        ops: smallvec![],
        parent_region: None,
    });

    // Initialize ASan before anything else (must precede libmprompt's SIGSEGV handler)
    if sanitize {
        let asan_call = arena_func::call(ctx, loc, [], nil_ty, Symbol::new("__asan_init"));
        ctx.push_op(entry_block, asan_call.op_ref());
    }

    // Initialize libmprompt runtime before any ability use
    let init_call = arena_func::call(ctx, loc, [], nil_ty, Symbol::new("__tribute_init"));
    ctx.push_op(entry_block, init_call.op_ref());

    // Call _tribute_main() — result is ignored
    let main_call = arena_func::call(
        ctx,
        loc,
        [],
        tribute_main_return_ty,
        Symbol::new("_tribute_main"),
    );
    ctx.push_op(entry_block, main_call.op_ref());

    // Return exit code 0
    let zero = arena_arith::r#const(ctx, loc, i32_ty, ArenaAttribute::IntBits(0));
    ctx.push_op(entry_block, zero.op_ref());

    let ret = arena_func::r#return(ctx, loc, [zero.result(ctx)]);
    ctx.push_op(entry_block, ret.op_ref());

    // Create body region
    let body = ctx.create_region(RegionData {
        location: loc,
        blocks: smallvec![entry_block],
        parent_op: None,
    });

    // Create func.func @main() -> i32
    // NOTE: No "abi" attribute here — `abi` marks extern (imported) functions.
    // The Cranelift backend treats functions named "main" as Export linkage,
    // but functions with `abi` attribute are treated as Import and skipped.
    let main_func = arena_func::func(ctx, loc, Symbol::new("main"), func_ty, body);

    main_func.op_ref()
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::Span;
    use trunk_ir::arena::context::{BlockData as ArenaBlockData, IrContext, RegionData};
    use trunk_ir::arena::dialect::arith as arena_arith;
    use trunk_ir::arena::dialect::func as arena_func;
    use trunk_ir::arena::ops::ArenaDialectOp;
    use trunk_ir::arena::rewrite::ArenaModule;
    use trunk_ir::arena::types::{
        Attribute as ArenaAttribute, Location as ArenaLocation, TypeDataBuilder,
    };

    fn arena_test_ctx() -> (IrContext, ArenaLocation) {
        let mut ctx = IrContext::new();
        let path = ctx.paths.intern("file:///test.trb".to_owned());
        let loc = ArenaLocation::new(path, Span::new(0, 0));
        (ctx, loc)
    }

    fn arena_i32_type(ctx: &mut IrContext) -> trunk_ir::arena::refs::TypeRef {
        ctx.types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build())
    }

    fn arena_nil_type(ctx: &mut IrContext) -> trunk_ir::arena::refs::TypeRef {
        ctx.types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("nil")).build())
    }

    /// Build an arena module with a single main function returning i32.
    fn make_arena_main_module(ctx: &mut IrContext, loc: ArenaLocation) -> ArenaModule {
        let i32_ty = arena_i32_type(ctx);
        let func_ty = ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func"))
                .param(i32_ty)
                .build(),
        );

        // Build main function: const 42, return
        let entry = ctx.create_block(ArenaBlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let c42 = arena_arith::r#const(ctx, loc, i32_ty, ArenaAttribute::IntBits(42));
        ctx.push_op(entry, c42.op_ref());
        let ret = arena_func::r#return(ctx, loc, [c42.result(ctx)]);
        ctx.push_op(entry, ret.op_ref());

        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let main_fn = arena_func::func(ctx, loc, Symbol::new("main"), func_ty, body);

        // Build module
        let module_block = ctx.create_block(ArenaBlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        ctx.push_op(module_block, main_fn.op_ref());

        let module_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![module_block],
            parent_op: None,
        });

        let module_data = trunk_ir::arena::context::OperationDataBuilder::new(
            loc,
            Symbol::new("core"),
            Symbol::new("module"),
        )
        .attr("sym_name", ArenaAttribute::Symbol(Symbol::new("test")))
        .region(module_region)
        .build(ctx);
        let module_op = ctx.create_op(module_data);

        ArenaModule::new(ctx, module_op).expect("valid arena module")
    }

    #[test]
    fn arena_entrypoint_renames_main() {
        let (mut ctx, loc) = arena_test_ctx();
        let module = make_arena_main_module(&mut ctx, loc);

        generate_native_entrypoint(&mut ctx, module, false);

        let ops = module.ops(&ctx);
        let mut names: Vec<String> = Vec::new();
        for &op in &ops {
            if let Ok(f) = arena_func::Func::from_op(&ctx, op) {
                names.push(f.sym_name(&ctx).to_string());
            }
        }

        assert!(
            names.contains(&"__tribute_init".to_string()),
            "Expected __tribute_init, got: {:?}",
            names
        );
        assert!(
            names.contains(&"_tribute_main".to_string()),
            "Expected _tribute_main, got: {:?}",
            names
        );
        assert!(
            names.contains(&"main".to_string()),
            "Expected main wrapper, got: {:?}",
            names
        );
        assert_eq!(
            names.len(),
            3,
            "Expected exactly 3 functions, got: {:?}",
            names
        );
    }

    #[test]
    fn arena_entrypoint_no_main() {
        let (mut ctx, loc) = arena_test_ctx();
        let i32_ty = arena_i32_type(&mut ctx);
        let func_ty = ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func"))
                .param(i32_ty)
                .build(),
        );

        // Build helper function (not main)
        let entry = ctx.create_block(ArenaBlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let c1 = arena_arith::r#const(&mut ctx, loc, i32_ty, ArenaAttribute::IntBits(1));
        ctx.push_op(entry, c1.op_ref());
        let c1_val = c1.result(&ctx);
        let ret = arena_func::r#return(&mut ctx, loc, [c1_val]);
        ctx.push_op(entry, ret.op_ref());

        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let helper_fn = arena_func::func(&mut ctx, loc, Symbol::new("helper"), func_ty, body);

        // Build module
        let module_block = ctx.create_block(ArenaBlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        ctx.push_op(module_block, helper_fn.op_ref());

        let module_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![module_block],
            parent_op: None,
        });

        let module_data = trunk_ir::arena::context::OperationDataBuilder::new(
            loc,
            Symbol::new("core"),
            Symbol::new("module"),
        )
        .attr("sym_name", ArenaAttribute::Symbol(Symbol::new("test")))
        .region(module_region)
        .build(&mut ctx);
        let module_op = ctx.create_op(module_data);
        let module = ArenaModule::new(&ctx, module_op).expect("valid");

        generate_native_entrypoint(&mut ctx, module, false);

        // Should be unchanged — only helper
        let ops = module.ops(&ctx);
        assert_eq!(ops.len(), 1);
        let f = arena_func::Func::from_op(&ctx, ops[0]).unwrap();
        assert_eq!(f.sym_name(&ctx).to_string(), "helper");
    }

    #[test]
    fn arena_entrypoint_wrapper_returns_i32() {
        let (mut ctx, loc) = arena_test_ctx();

        // Build module with main returning nil
        let nil_ty = arena_nil_type(&mut ctx);
        let func_ty = ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func"))
                .param(nil_ty)
                .build(),
        );

        let entry = ctx.create_block(ArenaBlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let ret = arena_func::r#return(&mut ctx, loc, []);
        ctx.push_op(entry, ret.op_ref());

        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let main_fn = arena_func::func(&mut ctx, loc, Symbol::new("main"), func_ty, body);

        let module_block = ctx.create_block(ArenaBlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        ctx.push_op(module_block, main_fn.op_ref());

        let module_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![module_block],
            parent_op: None,
        });

        let module_data = trunk_ir::arena::context::OperationDataBuilder::new(
            loc,
            Symbol::new("core"),
            Symbol::new("module"),
        )
        .attr("sym_name", ArenaAttribute::Symbol(Symbol::new("test")))
        .region(module_region)
        .build(&mut ctx);
        let module_op = ctx.create_op(module_data);
        let module = ArenaModule::new(&ctx, module_op).expect("valid");

        generate_native_entrypoint(&mut ctx, module, false);

        let i32_ty = arena_i32_type(&mut ctx);

        // Find the new main wrapper and check its return type
        for &op in &module.ops(&ctx) {
            if let Ok(f) = arena_func::Func::from_op(&ctx, op)
                && f.sym_name(&ctx) == Symbol::new("main")
            {
                let func_ty_ref = f.r#type(&ctx);
                let type_data = ctx.types.get(func_ty_ref);
                // core.func params[0] = result type
                assert_eq!(
                    type_data.params[0], i32_ty,
                    "main wrapper should return i32"
                );
                return;
            }
        }
        panic!("main wrapper not found after entrypoint generation");
    }

    #[test]
    fn arena_entrypoint_with_sanitize() {
        let (mut ctx, loc) = arena_test_ctx();
        let module = make_arena_main_module(&mut ctx, loc);

        generate_native_entrypoint(&mut ctx, module, true);

        let ops = module.ops(&ctx);
        let mut names: Vec<String> = Vec::new();
        for &op in &ops {
            if let Ok(f) = arena_func::Func::from_op(&ctx, op) {
                names.push(f.sym_name(&ctx).to_string());
            }
        }

        assert!(
            names.contains(&"__asan_init".to_string()),
            "Expected __asan_init with sanitize=true, got: {:?}",
            names
        );
        assert!(
            names.contains(&"__tribute_init".to_string()),
            "Expected __tribute_init, got: {:?}",
            names
        );
        assert!(
            names.contains(&"_tribute_main".to_string()),
            "Expected _tribute_main, got: {:?}",
            names
        );
        assert!(
            names.contains(&"main".to_string()),
            "Expected main wrapper, got: {:?}",
            names
        );
    }
}
