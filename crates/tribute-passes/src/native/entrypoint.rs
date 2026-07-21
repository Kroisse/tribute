//! Native entrypoint generation pass.
//!
//! Generates a C ABI `main` entrypoint that wraps the user's `main` function.
//! The user's `main` is renamed to `_tribute_main`, and a new `main` is created
//! that calls it and returns exit code 0.

use tribute_core::{CallingConvention, get_calling_convention, set_calling_convention};
use tribute_ir::dialect::ability::evidence_abi;
use trunk_ir::Symbol;
use trunk_ir::context::{BlockData, IrContext, RegionData};
use trunk_ir::dialect::arith;
use trunk_ir::dialect::core;
use trunk_ir::dialect::func;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{BlockRef, OpRef, RegionRef, TypeRef};
use trunk_ir::rewrite::Module;
use trunk_ir::smallvec::smallvec;
use trunk_ir::types::{Attribute, Location, TypeDataBuilder};

/// Generate a native C ABI entrypoint wrapper for the user's `main` function.
/// (Arena IR version — mutates `ctx` in-place.)
///
/// This pass:
/// 1. Renames `func.func @main` to `func.func @_tribute_main`
/// 2. Rewrites all `callee = @main` / `func_ref = @main` to `@_tribute_main`
/// 3. Ensures `__tribute_init` (and optionally `__asan_init`) declarations exist
/// 4. Creates a new `func.func @main() -> i32` that calls `_tribute_main()`
///    and returns 0
pub fn generate_native_entrypoint(ctx: &mut IrContext, module: Module, sanitize: bool) {
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
    let mut main_param_types: Vec<TypeRef> = Vec::new();
    let mut main_convention = CallingConvention::Direct;
    let mut has_tribute_init = false;
    let mut has_asan_init = false;

    let init_sym = Symbol::new("__tribute_init");
    let asan_init_sym = Symbol::new("__asan_init");

    for &op in &ops {
        if let Ok(func_op) = func::Func::from_op(ctx, op) {
            let name = func_op.sym_name(ctx);
            if name == main_sym {
                found_main = true;
                // Extract return type from func type
                let func_ty = func_op.r#type(ctx);
                let type_data = ctx.types.get(func_ty);
                // core.func layout: params[0] = result
                if !type_data.params.is_empty() {
                    main_return_ty = Some(type_data.params[0]);
                    main_param_types = type_data.params[1..].to_vec();
                }
                main_convention = get_calling_convention(ctx, op).unwrap_or_default();
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
        panic!(
            "entrypoint: `_tribute_main` already exists in module; \
             cannot rename `main` to `_tribute_main` due to symbol collision. \
             Ensure no user-defined function is named `_tribute_main`."
        );
    }

    // Intern needed types
    let nil_ty = core::nil(ctx).as_type_ref();

    let tribute_main_return_ty = main_return_ty.unwrap_or_else(|| {
        panic!(
            "entrypoint: `func.func @main` has no return type in its signature; \
             expected a valid `core.func` type with at least a return type parameter"
        )
    });

    // Step 1: Rename main -> _tribute_main (in-place attribute mutation)
    for &op in &ops {
        if let Ok(func_op) = func::Func::from_op(ctx, op)
            && func_op.sym_name(ctx) == main_sym
        {
            ctx.op_mut(op)
                .attributes
                .insert(Symbol::new("sym_name"), Attribute::Symbol(tribute_main_sym));
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
    let entrypoint_op = build_entrypoint(
        ctx,
        loc,
        TributeMainAbi {
            return_type: tribute_main_return_ty,
            parameter_types: &main_param_types,
            convention: main_convention,
        },
        sanitize,
    );
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

    if let Some(Attribute::Symbol(sym)) = ctx.op(op).attributes.get(callee_key).cloned()
        && sym == old_sym
    {
        ctx.op_mut(op)
            .attributes
            .insert(callee_key, Attribute::Symbol(new_sym));
    }

    // Rewrite func_ref attribute
    if let Some(Attribute::Symbol(sym)) = ctx.op(op).attributes.get(func_ref_key).cloned()
        && sym == old_sym
    {
        ctx.op_mut(op)
            .attributes
            .insert(func_ref_key, Attribute::Symbol(new_sym));
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
///     %ev = call @__tribute_evidence_empty() // EvidenceDirect only
///     call @_tribute_main(%ev)
///     %zero = arith.const 0 : i32
///     return %zero
/// }
/// ```
struct TributeMainAbi<'a> {
    return_type: TypeRef,
    parameter_types: &'a [TypeRef],
    convention: CallingConvention,
}

fn build_entrypoint(
    ctx: &mut IrContext,
    loc: Location,
    tribute_main: TributeMainAbi<'_>,
    sanitize: bool,
) -> OpRef {
    let i32_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());
    let nil_ty = core::nil(ctx).as_type_ref();

    // Build func type: () -> i32
    let func_ty = core::func(ctx, i32_ty, []).as_type_ref();

    // Create entry block
    let entry_block = ctx.create_block(BlockData {
        location: loc,
        args: vec![],
        ops: smallvec![],
        parent_region: None,
    });

    // Initialize ASan before anything else
    if sanitize {
        let asan_call = func::call(ctx, loc, [], nil_ty, Symbol::new("__asan_init"));
        ctx.push_op(entry_block, asan_call.op_ref());
    }

    // Initialize runtime TLS before any ability use
    let init_call = func::call(ctx, loc, [], nil_ty, Symbol::new("__tribute_init"));
    ctx.push_op(entry_block, init_call.op_ref());

    let main_args = match tribute_main.convention {
        CallingConvention::Direct => {
            assert!(
                tribute_main.parameter_types.is_empty(),
                "entrypoint: Direct `main` must not have hidden parameters"
            );
            Vec::new()
        }
        CallingConvention::EvidenceDirect => {
            assert_eq!(
                tribute_main.parameter_types.len(),
                1,
                "entrypoint: EvidenceDirect `main` must have exactly one evidence parameter"
            );
            let empty_evidence = func::call(
                ctx,
                loc,
                [],
                tribute_main.parameter_types[0],
                Symbol::new(evidence_abi::EMPTY),
            );
            ctx.push_op(entry_block, empty_evidence.op_ref());
            vec![empty_evidence.result(ctx)]
        }
        CallingConvention::Cps => {
            panic!(
                "entrypoint: Cps `main` is invalid; frontend must reject residual control effects"
            )
        }
    };

    // Call _tribute_main — its source-level result is ignored.
    let main_call = func::call(
        ctx,
        loc,
        main_args,
        tribute_main.return_type,
        Symbol::new("_tribute_main"),
    );
    set_calling_convention(ctx, main_call.op_ref(), tribute_main.convention);
    ctx.push_op(entry_block, main_call.op_ref());

    // Return exit code 0
    let zero = arith::r#const(ctx, loc, i32_ty, Attribute::Int(0));
    ctx.push_op(entry_block, zero.op_ref());

    let ret = func::r#return(ctx, loc, [zero.result(ctx)]);
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
    let main_func = func::func(ctx, loc, Symbol::new("main"), func_ty, body);
    set_calling_convention(ctx, main_func.op_ref(), CallingConvention::Direct);

    main_func.op_ref()
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::Span;
    use trunk_ir::context::{BlockArgData, BlockData, IrContext, RegionData};
    use trunk_ir::dialect::arith;
    use trunk_ir::dialect::func;
    use trunk_ir::ops::DialectOp;
    use trunk_ir::rewrite::Module;
    use trunk_ir::types::{Attribute, Location, TypeDataBuilder};

    fn test_ctx() -> (IrContext, Location) {
        let mut ctx = IrContext::new();
        let path = ctx.paths.intern("file:///test.trb".to_owned());
        let loc = Location::new(path, Span::new(0, 0));
        (ctx, loc)
    }

    fn i32_type(ctx: &mut IrContext) -> trunk_ir::refs::TypeRef {
        ctx.types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build())
    }

    fn nil_type(ctx: &mut IrContext) -> trunk_ir::refs::TypeRef {
        core::nil(ctx).as_type_ref()
    }

    /// Build an arena module with a single main function returning i32.
    fn make_main_module(ctx: &mut IrContext, loc: Location) -> Module {
        let i32_ty = i32_type(ctx);
        let func_ty = core::func(ctx, i32_ty, []).as_type_ref();

        // Build main function: const 42, return
        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let c42 = arith::r#const(ctx, loc, i32_ty, Attribute::Int(42));
        ctx.push_op(entry, c42.op_ref());
        let ret = func::r#return(ctx, loc, [c42.result(ctx)]);
        ctx.push_op(entry, ret.op_ref());

        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let main_fn = func::func(ctx, loc, Symbol::new("main"), func_ty, body);

        // Build module
        let module_block = ctx.create_block(BlockData {
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

        let module_data = trunk_ir::context::OperationDataBuilder::new(
            loc,
            Symbol::new("core"),
            Symbol::new("module"),
        )
        .attr("sym_name", Attribute::Symbol(Symbol::new("test")))
        .region(module_region)
        .build(ctx);
        let module_op = ctx.create_op(module_data);

        Module::new(ctx, module_op).expect("valid arena module")
    }

    fn make_evidence_main_module(ctx: &mut IrContext, loc: Location) -> Module {
        let nil_ty = nil_type(ctx);
        let evidence_ty = tribute_ir::dialect::ability::evidence_adt_type_ref(ctx);
        let func_ty = core::func(ctx, nil_ty, [evidence_ty]).as_type_ref();
        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![BlockArgData {
                ty: evidence_ty,
                attrs: Default::default(),
            }],
            ops: smallvec![],
            parent_region: None,
        });
        let ret = func::r#return(ctx, loc, []);
        ctx.push_op(entry, ret.op_ref());
        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let main_fn = func::func(ctx, loc, Symbol::new("main"), func_ty, body);
        set_calling_convention(ctx, main_fn.op_ref(), CallingConvention::EvidenceDirect);

        let module_block = ctx.create_block(BlockData {
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
        let module_data = trunk_ir::context::OperationDataBuilder::new(
            loc,
            Symbol::new("core"),
            Symbol::new("module"),
        )
        .attr("sym_name", Attribute::Symbol(Symbol::new("test")))
        .region(module_region)
        .build(ctx);
        let module_op = ctx.create_op(module_data);
        Module::new(ctx, module_op).expect("valid arena module")
    }

    #[test]
    fn entrypoint_renames_main() {
        let (mut ctx, loc) = test_ctx();
        let module = make_main_module(&mut ctx, loc);

        generate_native_entrypoint(&mut ctx, module, false);

        let ops = module.ops(&ctx);
        let mut names: Vec<String> = Vec::new();
        for &op in &ops {
            if let Ok(f) = func::Func::from_op(&ctx, op) {
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
    fn entrypoint_no_main() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);
        let func_ty = core::func(&mut ctx, i32_ty, []).as_type_ref();

        // Build helper function (not main)
        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let c1 = arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(1));
        ctx.push_op(entry, c1.op_ref());
        let c1_val = c1.result(&ctx);
        let ret = func::r#return(&mut ctx, loc, [c1_val]);
        ctx.push_op(entry, ret.op_ref());

        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let helper_fn = func::func(&mut ctx, loc, Symbol::new("helper"), func_ty, body);

        // Build module
        let module_block = ctx.create_block(BlockData {
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

        let module_data = trunk_ir::context::OperationDataBuilder::new(
            loc,
            Symbol::new("core"),
            Symbol::new("module"),
        )
        .attr("sym_name", Attribute::Symbol(Symbol::new("test")))
        .region(module_region)
        .build(&mut ctx);
        let module_op = ctx.create_op(module_data);
        let module = Module::new(&ctx, module_op).expect("valid");

        generate_native_entrypoint(&mut ctx, module, false);

        // Should be unchanged — only helper
        let ops = module.ops(&ctx);
        assert_eq!(ops.len(), 1);
        let f = func::Func::from_op(&ctx, ops[0]).unwrap();
        assert_eq!(f.sym_name(&ctx).to_string(), "helper");
    }

    #[test]
    fn entrypoint_wrapper_returns_i32() {
        let (mut ctx, loc) = test_ctx();

        // Build module with main returning nil
        let nil_ty = nil_type(&mut ctx);
        let func_ty = core::func(&mut ctx, nil_ty, []).as_type_ref();

        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let ret = func::r#return(&mut ctx, loc, []);
        ctx.push_op(entry, ret.op_ref());

        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let main_fn = func::func(&mut ctx, loc, Symbol::new("main"), func_ty, body);

        let module_block = ctx.create_block(BlockData {
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

        let module_data = trunk_ir::context::OperationDataBuilder::new(
            loc,
            Symbol::new("core"),
            Symbol::new("module"),
        )
        .attr("sym_name", Attribute::Symbol(Symbol::new("test")))
        .region(module_region)
        .build(&mut ctx);
        let module_op = ctx.create_op(module_data);
        let module = Module::new(&ctx, module_op).expect("valid");

        generate_native_entrypoint(&mut ctx, module, false);

        let i32_ty = i32_type(&mut ctx);

        // Find the new main wrapper and check its return type
        for &op in &module.ops(&ctx) {
            if let Ok(f) = func::Func::from_op(&ctx, op)
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
    fn evidence_direct_main_receives_initial_evidence() {
        let (mut ctx, loc) = test_ctx();
        let module = make_evidence_main_module(&mut ctx, loc);

        generate_native_entrypoint(&mut ctx, module, false);

        let wrapper = module
            .ops(&ctx)
            .into_iter()
            .find_map(|op| {
                let f = func::Func::from_op(&ctx, op).ok()?;
                (f.sym_name(&ctx) == Symbol::new("main")).then_some(f)
            })
            .expect("generated main wrapper");
        let entry = ctx.region(wrapper.body(&ctx)).blocks[0];
        let calls: Vec<_> = ctx
            .block(entry)
            .ops
            .iter()
            .copied()
            .filter(|op| {
                let data = ctx.op(*op);
                data.dialect == Symbol::new("func") && data.name == Symbol::new("call")
            })
            .collect();
        let empty_call = calls
            .iter()
            .copied()
            .find(|op| {
                ctx.op(*op).attributes.get("callee")
                    == Some(&Attribute::Symbol(Symbol::new(evidence_abi::EMPTY)))
            })
            .expect("entrypoint should create empty evidence");
        let user_main_call = calls
            .iter()
            .copied()
            .find(|op| {
                ctx.op(*op).attributes.get("callee")
                    == Some(&Attribute::Symbol(Symbol::new("_tribute_main")))
            })
            .expect("entrypoint should call renamed user main");

        assert!(ctx.op_operands(empty_call).is_empty());
        assert_eq!(ctx.op_operands(user_main_call).len(), 1);
        assert_eq!(
            get_calling_convention(&ctx, user_main_call),
            Some(CallingConvention::EvidenceDirect)
        );
    }

    #[test]
    fn entrypoint_with_sanitize() {
        let (mut ctx, loc) = test_ctx();
        let module = make_main_module(&mut ctx, loc);

        generate_native_entrypoint(&mut ctx, module, true);

        let ops = module.ops(&ctx);
        let mut names: Vec<String> = Vec::new();
        for &op in &ops {
            if let Ok(f) = func::Func::from_op(&ctx, op) {
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
