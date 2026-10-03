//! Global Dead Code Elimination (DCE) pass for arena IR.
//!
//! Removes function definitions that are not reachable from reachability roots.
//! Functions are keyed by root-qualified name. Reachability roots include:
//! - The root module's `main` or `_start`
//! - Functions referenced outside any function definition, such as by
//!   `wasm.export_func`
//! - Function definitions with an `abi` attribute (externally callable)
//!
//! A bodyless `abi` declaration is an import, not a root: it stays only while
//! something reachable references it.
//! - Custom entry points from configuration, by qualified name
//!
//! Follows the [`CallGraph`] edges of calls and address references, then
//! removes unreachable functions via BFS.

use std::collections::{HashSet, VecDeque};
use std::ops::ControlFlow;

use crate::analysis::AnalysisCache;
use crate::context::IrContext;
use crate::dialect::{core, func};
use crate::ops::DialectOp;
use crate::refs::OpRef;
use crate::rewrite::Module;
use crate::symbol::Symbol;
use crate::symbol_table::SymbolTable;
use crate::transforms::call_graph::CallGraph;
use crate::walk::{WalkAction, walk_region};

/// Configuration for global dead code elimination.
#[derive(Debug, Clone)]
pub struct GlobalDceConfig {
    /// Additional entry point qualified function names (besides main/_start).
    pub extra_entry_points: Vec<String>,
    /// Whether to recursively process nested modules. Default: true.
    pub recursive: bool,
}

impl Default for GlobalDceConfig {
    fn default() -> Self {
        Self {
            extra_entry_points: Vec::new(),
            recursive: true,
        }
    }
}

/// Result of running global DCE.
pub struct GlobalDceResult {
    /// Number of functions removed.
    pub removed_count: usize,
    /// Names of removed functions (for debugging).
    pub removed_functions: Vec<Symbol>,
}

/// Eliminate unreachable functions from a module using default configuration.
pub fn eliminate_dead_functions(
    ctx: &mut IrContext,
    module: Module,
    analyses: &mut AnalysisCache,
) -> GlobalDceResult {
    eliminate_dead_functions_with_config(ctx, module, GlobalDceConfig::default(), analyses)
}

/// Eliminate unreachable functions with custom configuration, using the
/// [`SymbolTable`] and [`CallGraph`] cached in `analyses`.
pub fn eliminate_dead_functions_with_config(
    ctx: &mut IrContext,
    module: Module,
    config: GlobalDceConfig,
    analyses: &mut AnalysisCache,
) -> GlobalDceResult {
    run(ctx, module, &config, analyses)
}

/// Eliminate the functions of `module` not reachable from its roots.
///
/// Functions are keyed by root-qualified name. Every definition of a
/// duplicated name shares its reachability. With `recursive: false`, functions
/// in nested modules are neither removed nor analyzed; they are kept as roots.
fn run(
    ctx: &mut IrContext,
    module: Module,
    config: &GlobalDceConfig,
    analyses: &mut AnalysisCache,
) -> GlobalDceResult {
    let symbols = analyses.require::<SymbolTable>(ctx, module.op());
    let functions = || {
        symbols
            .all_definitions()
            .filter(|&(_, op)| func::Func::matches(ctx, op))
    };
    let is_candidate = |op| config.recursive || !in_nested_module(ctx, module, op);
    let candidates: Vec<(Symbol, OpRef)> =
        functions().filter(|&(_, op)| is_candidate(op)).collect();

    let mut roots: HashSet<Symbol> = functions()
        .filter(|&(name, op)| !is_candidate(op) || is_root(ctx, name, op, config))
        .map(|(name, _)| name)
        .collect();
    let _ = walk_region::<()>(ctx, module.body(ctx).expect("module body"), &mut |op| {
        if func::Func::matches(ctx, op) {
            return ControlFlow::Continue(WalkAction::Skip);
        }
        ctx.op(op).attributes.visit_symbol_refs(&mut |reference| {
            roots.insert(reference);
        });
        ControlFlow::Continue(WalkAction::Advance)
    });

    let graph = analyses.require::<CallGraph>(ctx, module.op());
    let reachable = compute_reachable(&graph, roots);

    // A function containing a reachable function definition is kept with it.
    let mut kept = HashSet::new();
    for (_, op) in functions().filter(|(name, _)| reachable.contains(name)) {
        let mut current = Some(op);
        while let Some(op) = current
            && op != module.op()
        {
            kept.insert(op);
            current = parent_op(ctx, op);
        }
    }
    let dead: Vec<(Symbol, OpRef)> = candidates
        .into_iter()
        .filter(|(_, op)| !kept.contains(op))
        .collect();
    let dead_ops: HashSet<OpRef> = dead.iter().map(|&(_, op)| op).collect();

    let mut removed = Vec::new();
    for (name, op) in dead {
        removed.push(name);
        // A function inside a dead function goes with its container.
        if has_ancestor_in(ctx, op, &dead_ops) {
            continue;
        }
        // Erase the unreachable function: `erase_op` clears the operand
        // use-chains of the func and its body subtree, so dead funcs don't
        // leave stale uses behind (#710). Func scopes are independent (SSA),
        // so clearing the body's operand uses cannot affect other reachable
        // functions.
        crate::rewrite::erase_op(ctx, op);
    }

    GlobalDceResult {
        removed_count: removed.len(),
        removed_functions: removed,
    }
}

/// Whether `name` is a reachability root: the root `main` or `_start`, a
/// function definition with an `abi` attribute (externally callable), or a
/// configured extra entry point.
fn is_root(ctx: &IrContext, name: Symbol, op: OpRef, config: &GlobalDceConfig) -> bool {
    name == Symbol::new("main")
        || name == Symbol::new("_start")
        || (ctx.op(op).attributes.contains_key("abi") && ctx.op_has_regions(op))
        || name.with_str(|name| config.extra_entry_points.iter().any(|extra| extra == name))
}

/// Whether `op` lies inside a `core.module` nested in `module`.
fn in_nested_module(ctx: &IrContext, module: Module, op: OpRef) -> bool {
    let mut current = op;
    while let Some(parent) = parent_op(ctx, current) {
        if parent == module.op() {
            return false;
        }
        if core::Module::matches(ctx, parent) {
            return true;
        }
        current = parent;
    }
    false
}

/// Whether an operation enclosing `op` is in `ops`.
fn has_ancestor_in(ctx: &IrContext, op: OpRef, ops: &HashSet<OpRef>) -> bool {
    let mut current = parent_op(ctx, op);
    while let Some(op) = current {
        if ops.contains(&op) {
            return true;
        }
        current = parent_op(ctx, op);
    }
    false
}

fn parent_op(ctx: &IrContext, op: OpRef) -> Option<OpRef> {
    let block = ctx.op(op).parent_block?;
    ctx.region(ctx.block(block).parent_region?).parent_op
}

/// Functions reachable from `roots` via BFS over call and reference edges.
fn compute_reachable(graph: &CallGraph, roots: HashSet<Symbol>) -> HashSet<Symbol> {
    let mut reachable = HashSet::new();
    let mut worklist: VecDeque<Symbol> = roots.into_iter().collect();

    while let Some(func) = worklist.pop_front() {
        if !reachable.insert(func) {
            continue;
        }
        if let Some(callees) = graph.edges.get(&func) {
            worklist.extend(callees.iter().filter(|callee| !reachable.contains(*callee)));
        }
    }

    reachable
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialect::func;
    use crate::location::Span;
    use crate::symbol::Symbol;
    use crate::*;
    use smallvec::smallvec;

    fn test_ctx() -> (IrContext, Location) {
        let mut ctx = IrContext::new();
        let path = ctx.intern_path("test.trb");
        let loc = Location::new(path, Span::new(0, 0));
        (ctx, loc)
    }

    fn fn_type(ctx: &mut IrContext) -> TypeRef {
        let nil_ty = crate::dialect::core::nil(ctx).as_type_ref();
        crate::dialect::func::func_sig(ctx, [], [nil_ty]).as_type_ref()
    }

    fn i32_type(ctx: &mut IrContext) -> TypeRef {
        ctx.intern_type(TypeDataBuilder::new("core", "i32").build())
    }

    fn build_simple_func(ctx: &mut IrContext, loc: Location, name: &str) -> OpRef {
        let fn_ty = fn_type(ctx);
        let sym_name = Symbol::from_dynamic(name);
        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let ret = func::Return::operands(std::iter::empty()).build(ctx, loc);
        ctx.push_op(entry, ret.op_ref());
        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        func::Func::operands()
            .sym_name(sym_name)
            .r#type(fn_ty)
            .regions(body)
            .build(ctx, loc)
            .op_ref()
    }

    fn build_func_with_call(ctx: &mut IrContext, loc: Location, name: &str, callee: &str) -> OpRef {
        let fn_ty = fn_type(ctx);
        let i32_ty = i32_type(ctx);
        let sym_name = Symbol::from_dynamic(name);
        let sym_callee = Symbol::from_dynamic(callee);
        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let call = func::Call::operands(std::iter::empty())
            .callee(sym_callee)
            .results([i32_ty])
            .build(ctx, loc);
        let call_result = call.result(ctx);
        ctx.push_op(entry, call.op_ref());
        let ret = func::Return::operands([call_result]).build(ctx, loc);
        ctx.push_op(entry, ret.op_ref());
        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        func::Func::operands()
            .sym_name(sym_name)
            .r#type(fn_ty)
            .regions(body)
            .build(ctx, loc)
            .op_ref()
    }

    fn build_module(ctx: &mut IrContext, loc: Location, ops: Vec<OpRef>) -> Module {
        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        for op in ops {
            ctx.push_op(block, op);
        }
        let region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block],
            parent_op: None,
        });
        let module_data =
            OperationDataBuilder::new(loc, Symbol::new("core"), Symbol::new("module"))
                .attr(
                    "sym_name",
                    Attribute::String(ctx.intern_symbol_text(Symbol::new("test"))),
                )
                .region(region)
                .build(ctx);
        let module_op = ctx.create_op(module_data);
        Module::new(ctx, module_op).unwrap()
    }

    fn count_funcs(ctx: &IrContext, module: Module) -> usize {
        module
            .ops(ctx)
            .iter()
            .filter(|&&op| {
                ctx.op(op).dialect == Symbol::new("func") && ctx.op(op).name == Symbol::new("func")
            })
            .count()
    }

    #[test]
    fn removes_unreachable_function() {
        let (mut ctx, loc) = test_ctx();
        let main = build_simple_func(&mut ctx, loc, "main");
        let unused = build_simple_func(&mut ctx, loc, "unused");
        let module = build_module(&mut ctx, loc, vec![main, unused]);

        let result = eliminate_dead_functions(&mut ctx, module, &mut Default::default());

        assert_eq!(result.removed_count, 1);
        assert_eq!(count_funcs(&ctx, module), 1);
    }

    #[test]
    fn keeps_called_function() {
        let (mut ctx, loc) = test_ctx();
        let helper = build_simple_func(&mut ctx, loc, "helper");
        let main = build_func_with_call(&mut ctx, loc, "main", "helper");
        let module = build_module(&mut ctx, loc, vec![helper, main]);

        let result = eliminate_dead_functions(&mut ctx, module, &mut Default::default());

        assert_eq!(result.removed_count, 0);
        assert_eq!(count_funcs(&ctx, module), 2);
    }

    #[test]
    fn keeps_transitive_calls() {
        let (mut ctx, loc) = test_ctx();
        let leaf = build_simple_func(&mut ctx, loc, "leaf");
        let middle = build_func_with_call(&mut ctx, loc, "middle", "leaf");
        let main = build_func_with_call(&mut ctx, loc, "main", "middle");
        let unreachable = build_simple_func(&mut ctx, loc, "unreachable");
        let module = build_module(&mut ctx, loc, vec![leaf, middle, main, unreachable]);

        let result = eliminate_dead_functions(&mut ctx, module, &mut Default::default());

        assert_eq!(result.removed_count, 1);
        assert_eq!(count_funcs(&ctx, module), 3);
    }

    #[test]
    fn keeps_func_constant_reference() {
        let (mut ctx, loc) = test_ctx();
        let fn_ty = fn_type(&mut ctx);

        let callback = build_simple_func(&mut ctx, loc, "callback");

        // Build main that references callback via func.constant
        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let const_op = func::Constant::operands()
            .func_ref(Symbol::new("callback"))
            .results(fn_ty)
            .build(&mut ctx, loc);
        ctx.push_op(entry, const_op.op_ref());
        let ret = func::Return::operands(std::iter::empty()).build(&mut ctx, loc);
        ctx.push_op(entry, ret.op_ref());
        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let main = func::Func::operands()
            .sym_name(Symbol::new("main"))
            .r#type(fn_ty)
            .regions(body)
            .build(&mut ctx, loc)
            .op_ref();

        let module = build_module(&mut ctx, loc, vec![callback, main]);

        let result = eliminate_dead_functions(&mut ctx, module, &mut Default::default());

        assert_eq!(result.removed_count, 0);
    }

    #[test]
    fn handles_start_entry_point() {
        let (mut ctx, loc) = test_ctx();
        let start = build_simple_func(&mut ctx, loc, "_start");
        let module = build_module(&mut ctx, loc, vec![start]);

        let result = eliminate_dead_functions(&mut ctx, module, &mut Default::default());

        assert_eq!(result.removed_count, 0);
    }

    #[test]
    fn extra_entry_points_config() {
        let (mut ctx, loc) = test_ctx();
        let custom = build_simple_func(&mut ctx, loc, "custom_init");
        let module = build_module(&mut ctx, loc, vec![custom]);

        let config = GlobalDceConfig {
            extra_entry_points: vec!["custom_init".to_string()],
            recursive: true,
        };
        let result =
            eliminate_dead_functions_with_config(&mut ctx, module, config, &mut Default::default());

        assert_eq!(result.removed_count, 0);
    }

    #[test]
    fn keeps_wasm_exported_function() {
        let (mut ctx, loc) = test_ctx();
        let exported = build_simple_func(&mut ctx, loc, "exported_func");
        let unused = build_simple_func(&mut ctx, loc, "unused_func");

        // Create wasm.export_func op
        let export_data =
            OperationDataBuilder::new(loc, Symbol::new("wasm"), Symbol::new("export_func"))
                .attr("name", ctx.string_attr("my_export"))
                .attr("func", Attribute::SymbolRef(Symbol::new("exported_func")))
                .build(&mut ctx);
        let export_op = ctx.create_op(export_data);

        let module = build_module(&mut ctx, loc, vec![exported, export_op, unused]);

        let result = eliminate_dead_functions(&mut ctx, module, &mut Default::default());

        assert_eq!(result.removed_count, 1); // Only unused_func removed
    }

    #[test]
    fn keeps_functions_referenced_by_any_operation() {
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(
            &mut ctx,
            r#"core.module @root {
  test.table {entries = [@exported]}
  func.func @exported() {
    test.make_closure {func_ref = @captured}
    func.return
  }
  func.func @captured() {
    func.return
  }
  func.func @unused() {
    test.make_closure {func_ref = @only_from_unused}
    func.return
  }
  func.func @only_from_unused() {
    func.return
  }
}"#,
        );

        eliminate_dead_functions(&mut ctx, module, &mut Default::default());

        assert_eq!(
            surviving_functions(&ctx, module),
            HashSet::from([Symbol::new("exported"), Symbol::new("captured")])
        );
    }

    #[test]
    fn preserves_extern_declarations() {
        let (mut ctx, loc) = test_ctx();
        let fn_ty = fn_type(&mut ctx);
        let main = build_simple_func(&mut ctx, loc, "main");

        // Build an unreachable extern func with abi attribute
        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let extern_data = OperationDataBuilder::new(loc, Symbol::new("func"), Symbol::new("func"))
            .attr(
                "sym_name",
                Attribute::String(ctx.intern_symbol_text(Symbol::new("extern_fn"))),
            )
            .attr("type", Attribute::Type(fn_ty))
            .attr("abi", ctx.string_attr("C"))
            .region(body)
            .build(&mut ctx);
        let extern_op = ctx.create_op(extern_data);

        let module = build_module(&mut ctx, loc, vec![main, extern_op]);

        let result = eliminate_dead_functions(&mut ctx, module, &mut Default::default());

        assert_eq!(result.removed_count, 0);
        assert_eq!(count_funcs(&ctx, module), 2);
    }

    #[test]
    fn unreferenced_abi_declarations_are_removed() {
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @used() -> core.i32 attributes {abi = "C"}
  func.func @unused() -> core.i32 attributes {abi = "C"}
  func.func @exported() -> core.i32 attributes {abi = "C"} {
    %value = func.call {callee = @used} : core.i32
    func.return %value
  }
}"#,
        );

        let result = eliminate_dead_functions(&mut ctx, module, &mut Default::default());

        assert_eq!(result.removed_functions, [Symbol::new("unused")]);
        assert_eq!(count_funcs(&ctx, module), 2);
    }

    #[test]
    fn abi_function_callees_are_reachable() {
        let (mut ctx, loc) = test_ctx();
        let fn_ty = fn_type(&mut ctx);

        let main = build_simple_func(&mut ctx, loc, "main");

        // helper is only called by extern_fn
        let helper = build_simple_func(&mut ctx, loc, "helper");

        // extern_fn (abi) calls helper
        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let i32_ty = i32_type(&mut ctx);
        let call = func::Call::operands(std::iter::empty())
            .callee(Symbol::new("helper"))
            .results([i32_ty])
            .build(&mut ctx, loc);
        ctx.push_op(entry, call.op_ref());
        let ret = func::Return::operands(std::iter::empty()).build(&mut ctx, loc);
        ctx.push_op(entry, ret.op_ref());
        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let extern_data = OperationDataBuilder::new(loc, Symbol::new("func"), Symbol::new("func"))
            .attr(
                "sym_name",
                Attribute::String(ctx.intern_symbol_text(Symbol::new("extern_fn"))),
            )
            .attr("type", Attribute::Type(fn_ty))
            .attr("abi", ctx.string_attr("C"))
            .region(body)
            .build(&mut ctx);
        let extern_op = ctx.create_op(extern_data);

        let module = build_module(&mut ctx, loc, vec![main, helper, extern_op]);

        let result = eliminate_dead_functions(&mut ctx, module, &mut Default::default());

        // extern_fn is preserved (abi) and helper is reachable from extern_fn
        assert_eq!(result.removed_count, 0);
        assert_eq!(count_funcs(&ctx, module), 3);
    }

    #[test]
    fn nested_module_recursive() {
        let (mut ctx, loc) = test_ctx();

        let top_main = build_simple_func(&mut ctx, loc, "main");

        // Build nested module with its own main and an unused func. Only the
        // root module's `main` is an entry point.
        let nested_main = build_simple_func(&mut ctx, loc, "main");
        let nested_unused = build_simple_func(&mut ctx, loc, "unused_in_nested");

        let nested_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        ctx.push_op(nested_block, nested_main);
        ctx.push_op(nested_block, nested_unused);
        let nested_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![nested_block],
            parent_op: None,
        });
        let nested_module_data =
            OperationDataBuilder::new(loc, Symbol::new("core"), Symbol::new("module"))
                .attr(
                    "sym_name",
                    Attribute::String(ctx.intern_symbol_text(Symbol::new("nested"))),
                )
                .region(nested_region)
                .build(&mut ctx);
        let nested_module_op = ctx.create_op(nested_module_data);

        let module = build_module(&mut ctx, loc, vec![top_main, nested_module_op]);

        let config = GlobalDceConfig {
            extra_entry_points: vec![],
            recursive: true,
        };
        let result =
            eliminate_dead_functions_with_config(&mut ctx, module, config, &mut Default::default());

        assert_eq!(result.removed_count, 2);
        assert_eq!(
            HashSet::<Symbol>::from_iter(result.removed_functions),
            HashSet::from([
                Symbol::from_dynamic("nested::main"),
                Symbol::from_dynamic("nested::unused_in_nested"),
            ])
        );
    }

    #[test]
    fn non_recursive_keeps_nested() {
        let (mut ctx, loc) = test_ctx();

        let top_main = build_simple_func(&mut ctx, loc, "main");

        // Same nested module setup
        let nested_unused = build_simple_func(&mut ctx, loc, "unused_in_nested");
        let nested_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        ctx.push_op(nested_block, nested_unused);
        let nested_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![nested_block],
            parent_op: None,
        });
        let nested_module_data =
            OperationDataBuilder::new(loc, Symbol::new("core"), Symbol::new("module"))
                .attr(
                    "sym_name",
                    Attribute::String(ctx.intern_symbol_text(Symbol::new("nested"))),
                )
                .region(nested_region)
                .build(&mut ctx);
        let nested_module_op = ctx.create_op(nested_module_data);

        let module = build_module(&mut ctx, loc, vec![top_main, nested_module_op]);

        let config = GlobalDceConfig {
            extra_entry_points: vec![],
            recursive: false,
        };
        let result =
            eliminate_dead_functions_with_config(&mut ctx, module, config, &mut Default::default());

        // With recursive=false, nested module is not analyzed
        assert_eq!(result.removed_count, 0);
    }

    fn surviving_functions(ctx: &IrContext, module: Module) -> HashSet<Symbol> {
        SymbolTable::collect(ctx, module)
            .all_definitions()
            .filter(|&(_, op)| func::Func::matches(ctx, op))
            .map(|(name, _)| name)
            .collect()
    }

    #[test]
    fn reachability_follows_root_qualified_references() {
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(
            &mut ctx,
            r#"core.module @root {
  func.func @main() {
    func.call {callee = @"a::same"}
    func.return
  }
  core.module @a {
    func.func @same() {
      func.return
    }
  }
  core.module @b {
    func.func @same() {
      func.return
    }
  }
}"#,
        );

        let result = eliminate_dead_functions(&mut ctx, module, &mut Default::default());

        assert_eq!(result.removed_functions, [Symbol::from_dynamic("b::same")]);
        assert_eq!(
            surviving_functions(&ctx, module),
            HashSet::from([Symbol::new("main"), Symbol::from_dynamic("a::same")])
        );
    }

    #[test]
    fn non_recursive_keeps_callees_of_nested_functions() {
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(
            &mut ctx,
            r#"core.module @root {
  func.func @main() {
    func.return
  }
  func.func @helper() {
    func.return
  }
  func.func @unused() {
    func.return
  }
  core.module @nested {
    func.func @user() {
      func.call {callee = @helper}
      func.return
    }
  }
}"#,
        );

        let config = GlobalDceConfig {
            extra_entry_points: vec![],
            recursive: false,
        };
        let result =
            eliminate_dead_functions_with_config(&mut ctx, module, config, &mut Default::default());

        assert_eq!(result.removed_functions, [Symbol::new("unused")]);
    }

    #[test]
    fn keeps_functions_containing_reachable_definitions() {
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(
            &mut ctx,
            r#"core.module @root {
  func.func @main() {
    func.call {callee = @inner}
    func.return
  }
  func.func @host() {
    func.func @inner() {
      func.return
    }
    func.return
  }
  func.func @dead_host() {
    func.func @dead_inner() {
      func.return
    }
    func.return
  }
}"#,
        );

        let result = eliminate_dead_functions(&mut ctx, module, &mut Default::default());

        assert_eq!(
            HashSet::<Symbol>::from_iter(result.removed_functions),
            HashSet::from([Symbol::new("dead_host"), Symbol::new("dead_inner")])
        );
        assert_eq!(
            surviving_functions(&ctx, module),
            HashSet::from([
                Symbol::new("main"),
                Symbol::new("host"),
                Symbol::new("inner")
            ])
        );
    }
}
