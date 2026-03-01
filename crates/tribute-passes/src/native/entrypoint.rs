//! Native entrypoint generation pass.
//!
//! Generates a C ABI `main` entrypoint that wraps the user's `main` function.
//! The user's `main` is renamed to `_tribute_main`, and a new `main` is created
//! that calls it and returns exit code 0.

use trunk_ir::dialect::{arith, core, func};
use trunk_ir::{
    Attribute, Block, DialectOp, DialectType, IdVec, Location, Operation, Region, Symbol,
};

/// Generate a native C ABI entrypoint wrapper for the user's `main` function.
///
/// This pass:
/// 1. Renames `func.func @main` to `func.func @_tribute_main`
/// 2. Creates a new `func.func @main() -> i32` that calls `_tribute_main()`
///    and returns 0
///
/// This ensures the native binary has a C-compatible `main` entry point
/// while preserving the user's function semantics (which may return `Nil`).
pub fn generate_native_entrypoint<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
    sanitize: bool,
) -> core::Module<'db> {
    let body = module.body(db);
    let blocks = body.blocks(db);

    assert!(
        blocks.len() == 1,
        "Module body must be single-block, got {} blocks",
        blocks.len()
    );

    let Some(entry_block) = blocks.first() else {
        return module;
    };

    // Find the user's main function and its return type
    let mut found_main = false;
    let mut main_return_ty = None;

    for op in entry_block.operations(db).iter() {
        if let Ok(func_op) = func::Func::from_operation(db, *op)
            && func_op.sym_name(db) == Symbol::new("main")
        {
            found_main = true;
            if let Some(func_ty) = core::Func::from_type(db, func_op.r#type(db)) {
                main_return_ty = Some(func_ty.result(db));
            }
        }
    }

    if !found_main {
        tracing::warn!("No main function found; skipping entrypoint generation");
        return module;
    }

    let location = module.location(db);
    let i32_ty = core::I32::new(db).as_type();
    let tribute_main_sym = Symbol::new("_tribute_main");
    let tribute_main_return_ty = main_return_ty.unwrap_or_else(|| core::Nil::new(db).as_type());

    let main_sym = Symbol::new("main");

    // Rebuild module: rename main -> _tribute_main, rewrite call sites, then append new entrypoint
    let mut new_ops: Vec<Operation<'db>> = Vec::new();
    let mut has_tribute_init = false;
    let mut has_asan_init = false;
    let init_sym = Symbol::new("__tribute_init");
    let asan_init_sym = Symbol::new("__asan_init");

    for op in entry_block.operations(db).iter() {
        if let Ok(func_op) = func::Func::from_operation(db, *op) {
            if func_op.sym_name(db) == init_sym {
                has_tribute_init = true;
            }
            if func_op.sym_name(db) == asan_init_sym {
                has_asan_init = true;
            }
            if func_op.sym_name(db) == main_sym {
                let renamed = rebuild_func_with_name(db, &func_op, tribute_main_sym);
                // Also rewrite any call @main inside this function's body to @_tribute_main
                let rewritten = rewrite_symbol_refs(db, &renamed, main_sym, tribute_main_sym);
                new_ops.push(rewritten);
                continue;
            }
        }
        // Rewrite call @main in other functions too
        let rewritten = rewrite_symbol_refs(db, op, main_sym, tribute_main_sym);
        new_ops.push(rewritten);
    }

    // Ensure __tribute_init is declared (may already exist from cont_to_libmprompt FFI pass)
    if !has_tribute_init {
        let nil_ty = core::Nil::new(db).as_type();
        let init_decl = func::Func::build_extern(
            db,
            location,
            "__tribute_init",
            None,
            [],
            nil_ty,
            None,
            Some("C"),
        );
        new_ops.insert(0, init_decl.as_operation());
    }

    // Declare __asan_init when AddressSanitizer is enabled
    if sanitize && !has_asan_init {
        let nil_ty = core::Nil::new(db).as_type();
        let asan_init_decl = func::Func::build_extern(
            db,
            location,
            "__asan_init",
            None,
            [],
            nil_ty,
            None,
            Some("C"),
        );
        new_ops.insert(0, asan_init_decl.as_operation());
    }

    // Append the C ABI entrypoint
    let entrypoint = build_entrypoint(db, location, tribute_main_return_ty, i32_ty, sanitize);
    new_ops.push(entrypoint);

    // Rebuild module
    let new_block = Block::new(
        db,
        entry_block.id(db),
        entry_block.location(db),
        entry_block.args(db).clone(),
        new_ops.into_iter().collect(),
    );
    let new_body = Region::new(db, body.location(db), IdVec::from(vec![new_block]));
    core::Module::create(db, module.location(db), module.name(db), new_body)
}

/// Rebuild a `func.func` operation with a new name, preserving all other attributes and body.
fn rebuild_func_with_name<'db>(
    db: &'db dyn salsa::Database,
    func_op: &func::Func<'db>,
    new_name: Symbol,
) -> Operation<'db> {
    let op = func_op.as_operation();
    let location = op.location(db);
    let sym_name_key = Symbol::new("sym_name");

    let mut builder = Operation::of(db, location, func::DIALECT_NAME(), func::FUNC())
        .attr(sym_name_key, Attribute::Symbol(new_name));

    // Copy all attributes except sym_name
    for (key, value) in op.attributes(db).iter() {
        if *key != sym_name_key {
            builder = builder.attr(*key, value.clone());
        }
    }

    // Copy regions (function body)
    for region in op.regions(db).iter() {
        builder = builder.region(*region);
    }

    builder.build()
}

/// Recursively rewrite symbol references in an operation's attributes and regions.
///
/// Rewrites `func.call`, `func.tail_call` callee attributes and `func.constant` func_ref
/// attributes from `old_sym` to `new_sym`.
fn rewrite_symbol_refs<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    old_sym: Symbol,
    new_sym: Symbol,
) -> Operation<'db> {
    let mut attrs_changed = false;
    let mut new_attrs = op.attributes(db).clone();

    // Rewrite callee attribute on func.call / func.tail_call
    let callee_key = Symbol::new("callee");
    if let Some(Attribute::Symbol(sym)) = new_attrs.get(&callee_key)
        && *sym == old_sym
    {
        new_attrs.insert(callee_key, Attribute::Symbol(new_sym));
        attrs_changed = true;
    }

    // Rewrite func_ref attribute on func.constant
    let func_ref_key = Symbol::new("func_ref");
    if let Some(Attribute::Symbol(sym)) = new_attrs.get(&func_ref_key)
        && *sym == old_sym
    {
        new_attrs.insert(func_ref_key, Attribute::Symbol(new_sym));
        attrs_changed = true;
    }

    // Recurse into regions
    let regions = op.regions(db);
    let new_regions: IdVec<Region<'db>> = regions
        .iter()
        .map(|r| rewrite_symbol_refs_in_region(db, r, old_sym, new_sym))
        .collect();
    let regions_changed = new_regions != *regions;

    if !attrs_changed && !regions_changed {
        return *op;
    }

    Operation::new(
        db,
        op.location(db),
        op.dialect(db),
        op.name(db),
        op.operands(db).clone(),
        op.results(db).clone(),
        new_attrs,
        if regions_changed {
            new_regions
        } else {
            regions.clone()
        },
        op.successors(db).clone(),
    )
}

fn rewrite_symbol_refs_in_region<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
    old_sym: Symbol,
    new_sym: Symbol,
) -> Region<'db> {
    let blocks = region.blocks(db);
    let new_blocks: IdVec<Block<'db>> = blocks
        .iter()
        .map(|block| {
            let new_ops: IdVec<Operation<'db>> = block
                .operations(db)
                .iter()
                .map(|op| rewrite_symbol_refs(db, op, old_sym, new_sym))
                .collect();
            if new_ops == *block.operations(db) {
                return *block;
            }
            Block::new(
                db,
                block.id(db),
                block.location(db),
                block.args(db).clone(),
                new_ops,
            )
        })
        .collect();
    if new_blocks == *blocks {
        return *region;
    }
    Region::new(db, region.location(db), new_blocks)
}

/// Build the C ABI entrypoint function:
///
/// ```text
/// func.func @main() -> i32 {
///     call @_tribute_main()
///     %zero = arith.const 0 : i32
///     return %zero
/// }
/// ```
fn build_entrypoint<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    tribute_main_return_ty: trunk_ir::Type<'db>,
    i32_ty: trunk_ir::Type<'db>,
    sanitize: bool,
) -> Operation<'db> {
    let nil_ty = core::Nil::new(db).as_type();

    func::Func::build(db, location, "main", IdVec::new(), i32_ty, |builder| {
        // Initialize ASan before anything else (must precede libmprompt's SIGSEGV handler)
        if sanitize {
            builder.op(func::call(
                db,
                location,
                vec![],
                nil_ty,
                Symbol::new("__asan_init"),
            ));
        }
        // Initialize libmprompt runtime before any ability use
        builder.op(func::call(
            db,
            location,
            vec![],
            nil_ty,
            Symbol::new("__tribute_init"),
        ));
        // Call _tribute_main() — result is ignored
        builder.op(func::call(
            db,
            location,
            vec![],
            tribute_main_return_ty,
            Symbol::new("_tribute_main"),
        ));
        // Return exit code 0
        let zero = builder.op(arith::Const::i32(db, location, 0));
        builder.op(func::r#return(db, location, vec![zero.result(db)]));
    })
    .as_operation()
}

// ============================================================================
// Arena-based implementation
// ============================================================================

use std::collections::BTreeMap;

use trunk_ir::arena::context::{BlockArgData, BlockData, IrContext, RegionData};
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
pub fn generate_native_entrypoint_arena(ctx: &mut IrContext, module: ArenaModule, sanitize: bool) {
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
        if let Ok(func_op) = arena_func::Func::from_op(ctx, op) {
            if func_op.sym_name(ctx) == main_sym {
                ctx.op_mut(op).attributes.insert(
                    Symbol::new("sym_name"),
                    ArenaAttribute::Symbol(tribute_main_sym),
                );
            }
        }
    }

    // Step 2: Rewrite all @main references to @_tribute_main in all ops
    for &op in &ops {
        rewrite_symbol_refs_arena(ctx, op, main_sym, tribute_main_sym);
    }

    // Step 3: Ensure __tribute_init is declared
    if !has_tribute_init {
        let init_op = build_extern_func_arena(ctx, loc, "__tribute_init", &[], nil_ty);
        ctx.insert_op_before(first_block, ctx.block(first_block).ops[0], init_op);
    }

    // Declare __asan_init when AddressSanitizer is enabled
    if sanitize && !has_asan_init {
        let asan_op = build_extern_func_arena(ctx, loc, "__asan_init", &[], nil_ty);
        ctx.insert_op_before(first_block, ctx.block(first_block).ops[0], asan_op);
    }

    // Step 4: Build and append C ABI entrypoint wrapper
    let entrypoint_op =
        build_entrypoint_arena(ctx, loc, tribute_main_return_ty, i32_ty, nil_ty, sanitize);
    ctx.push_op(first_block, entrypoint_op);
}

/// Recursively rewrite symbol references from `old_sym` to `new_sym` in an
/// operation and all its nested regions.
///
/// Rewrites `callee` attributes (on func.call / func.tail_call) and
/// `func_ref` attributes (on func.constant).
fn rewrite_symbol_refs_arena(ctx: &mut IrContext, op: OpRef, old_sym: Symbol, new_sym: Symbol) {
    // Rewrite callee attribute
    let callee_key = Symbol::new("callee");
    let func_ref_key = Symbol::new("func_ref");

    if let Some(ArenaAttribute::Symbol(sym)) = ctx.op(op).attributes.get(&callee_key).cloned() {
        if sym == old_sym {
            ctx.op_mut(op)
                .attributes
                .insert(callee_key, ArenaAttribute::Symbol(new_sym));
        }
    }

    // Rewrite func_ref attribute
    if let Some(ArenaAttribute::Symbol(sym)) = ctx.op(op).attributes.get(&func_ref_key).cloned() {
        if sym == old_sym {
            ctx.op_mut(op)
                .attributes
                .insert(func_ref_key, ArenaAttribute::Symbol(new_sym));
        }
    }

    // Recurse into regions
    let regions: Vec<RegionRef> = ctx.op(op).regions.to_vec();
    for region in regions {
        rewrite_symbol_refs_in_region_arena(ctx, region, old_sym, new_sym);
    }
}

fn rewrite_symbol_refs_in_region_arena(
    ctx: &mut IrContext,
    region: RegionRef,
    old_sym: Symbol,
    new_sym: Symbol,
) {
    let blocks: Vec<BlockRef> = ctx.region(region).blocks.to_vec();
    for block in blocks {
        let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();
        for op in ops {
            rewrite_symbol_refs_arena(ctx, op, old_sym, new_sym);
        }
    }
}

/// Build an extern `func.func` with an unreachable body (arena version).
fn build_extern_func_arena(
    ctx: &mut IrContext,
    loc: ArenaLocation,
    name: &str,
    params: &[TypeRef],
    result: TypeRef,
) -> OpRef {
    // Build func type: core.func layout is params[0]=result, params[1..]=params
    let func_ty = ctx.types.intern({
        let mut builder = TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func"));
        builder = builder.param(result);
        for &p in params {
            builder = builder.param(p);
        }
        builder.build()
    });

    // Build entry block with params and unreachable
    let args: Vec<BlockArgData> = params
        .iter()
        .map(|&ty| BlockArgData {
            ty,
            attrs: BTreeMap::new(),
        })
        .collect();

    let unreachable_op = arena_func::unreachable(ctx, loc);

    let entry_block = ctx.create_block(BlockData {
        location: loc,
        args,
        ops: smallvec![],
        parent_region: None,
    });
    ctx.push_op(entry_block, unreachable_op.op_ref());

    let body = ctx.create_region(RegionData {
        location: loc,
        blocks: smallvec![entry_block],
        parent_op: None,
    });

    let func_op = arena_func::func(ctx, loc, Symbol::from_dynamic(name), func_ty, body);

    // Add abi attribute
    ctx.op_mut(func_op.op_ref())
        .attributes
        .insert(Symbol::new("abi"), ArenaAttribute::String("C".to_string()));

    func_op.op_ref()
}

/// Build the C ABI entrypoint function (arena version):
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
fn build_entrypoint_arena(
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
    use salsa_test_macros::salsa_test;
    use trunk_ir::{BlockId, PathId, Span, idvec};

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    /// Build a module with a single main function returning i32.
    #[salsa::tracked]
    fn make_main_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        let main_fn = func::Func::build(db, location, "main", IdVec::new(), i32_ty, |builder| {
            let c42 = builder.op(arith::Const::i32(db, location, 42));
            builder.op(func::r#return(db, location, vec![c42.result(db)]));
        });

        let ops: IdVec<Operation<'_>> = idvec![main_fn.as_operation()];
        let block = Block::new(db, BlockId::fresh(), location, idvec![], ops);
        let body = Region::new(db, location, idvec![block]);
        core::Module::create(db, location, "test".into(), body)
    }

    /// Apply entrypoint pass to the main module.
    #[salsa::tracked]
    fn apply_entrypoint_to_main(db: &dyn salsa::Database) -> core::Module<'_> {
        let module = make_main_module(db);
        generate_native_entrypoint(db, module, false)
    }

    #[salsa_test]
    fn test_entrypoint_renames_main(db: &salsa::DatabaseImpl) {
        let result = apply_entrypoint_to_main(db);

        let body = result.body(db);
        let ops = body.blocks(db)[0].operations(db);
        let mut names: Vec<String> = Vec::new();
        for op in ops.iter() {
            if let Ok(f) = func::Func::from_operation(db, *op) {
                names.push(f.sym_name(db).to_string());
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

    /// Build a module with no main function.
    #[salsa::tracked]
    fn make_no_main_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        let helper_fn =
            func::Func::build(db, location, "helper", IdVec::new(), i32_ty, |builder| {
                let c1 = builder.op(arith::Const::i32(db, location, 1));
                builder.op(func::r#return(db, location, vec![c1.result(db)]));
            });

        let ops: IdVec<Operation<'_>> = idvec![helper_fn.as_operation()];
        let block = Block::new(db, BlockId::fresh(), location, idvec![], ops);
        let body = Region::new(db, location, idvec![block]);
        core::Module::create(db, location, "test".into(), body)
    }

    /// Apply entrypoint pass to a module without main.
    #[salsa::tracked]
    fn apply_entrypoint_no_main(db: &dyn salsa::Database) -> core::Module<'_> {
        let module = make_no_main_module(db);
        generate_native_entrypoint(db, module, false)
    }

    #[salsa_test]
    fn test_entrypoint_no_main(db: &salsa::DatabaseImpl) {
        let result = apply_entrypoint_no_main(db);

        let body = result.body(db);
        let ops = body.blocks(db)[0].operations(db);
        assert_eq!(ops.len(), 1);
        if let Ok(f) = func::Func::from_operation(db, ops[0]) {
            assert_eq!(f.sym_name(db).to_string(), "helper");
        }
    }

    /// Build a module with main returning Nil.
    #[salsa::tracked]
    fn make_nil_main_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);
        let nil_ty = core::Nil::new(db).as_type();

        let main_fn = func::Func::build(db, location, "main", IdVec::new(), nil_ty, |builder| {
            builder.op(func::Return::empty(db, location));
        });

        let ops: IdVec<Operation<'_>> = idvec![main_fn.as_operation()];
        let block = Block::new(db, BlockId::fresh(), location, idvec![], ops);
        let body = Region::new(db, location, idvec![block]);
        core::Module::create(db, location, "test".into(), body)
    }

    /// Apply entrypoint pass to nil-returning main.
    #[salsa::tracked]
    fn apply_entrypoint_nil_main(db: &dyn salsa::Database) -> core::Module<'_> {
        let module = make_nil_main_module(db);
        generate_native_entrypoint(db, module, false)
    }

    #[salsa_test]
    fn test_entrypoint_wrapper_returns_i32(db: &salsa::DatabaseImpl) {
        let result = apply_entrypoint_nil_main(db);
        let i32_ty = core::I32::new(db).as_type();

        let body = result.body(db);
        for op in body.blocks(db)[0].operations(db).iter() {
            if let Ok(f) = func::Func::from_operation(db, *op)
                && f.sym_name(db) == Symbol::new("main")
            {
                let func_ty = core::Func::from_type(db, f.r#type(db)).unwrap();
                assert_eq!(func_ty.result(db), i32_ty);
            }
        }
    }

    // ====================================================================
    // Arena tests
    // ====================================================================

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

        generate_native_entrypoint_arena(&mut ctx, module, false);

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

        generate_native_entrypoint_arena(&mut ctx, module, false);

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

        generate_native_entrypoint_arena(&mut ctx, module, false);

        let i32_ty = arena_i32_type(&mut ctx);

        // Find the new main wrapper and check its return type
        for &op in &module.ops(&ctx) {
            if let Ok(f) = arena_func::Func::from_op(&ctx, op) {
                if f.sym_name(&ctx) == Symbol::new("main") {
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
        }
        panic!("main wrapper not found after entrypoint generation");
    }

    #[test]
    fn arena_entrypoint_with_sanitize() {
        let (mut ctx, loc) = arena_test_ctx();
        let module = make_arena_main_module(&mut ctx, loc);

        generate_native_entrypoint_arena(&mut ctx, module, true);

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
