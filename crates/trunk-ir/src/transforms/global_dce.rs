//! Global Dead Code Elimination (DCE) pass for arena IR.
//!
//! Removes function definitions that are not reachable from reachability roots.
//! Reachability roots include:
//! - The root module's `main` or `_start`
//! - Functions referenced outside any function definition, such as by
//!   `wasm.export_func`
//! - Function definitions with an `abi` attribute (externally callable)
//!
//! A bodyless `abi` declaration is an import, not a root: it stays only while
//! something reachable references it.
//! - Custom entry points from configuration, by root-qualified path
//!
//! Follows the [`CallGraph`] edges of calls and address references, then
//! removes unreachable functions via BFS.

use rustc_hash::FxHashSet as HashSet;
use std::collections::VecDeque;

use crate::analysis::AnalysisCache;
use crate::context::IrContext;
use crate::dialect::{core, func};
use crate::ops::DialectOp;
use crate::pass::{Pass, pass_fn};
use crate::refs::OpRef;
use crate::rewrite::Module;
use crate::symbol::SymbolPath;
use crate::symbol_table::{SymbolTable, nearest_table, qualified_name};
use crate::transforms::call_graph::{CallGraph, FunctionDefinition, call_graph_over};

/// Configuration for global dead code elimination.
#[derive(Debug, Clone)]
pub struct GlobalDceConfig {
    /// Additional entry points (besides main/_start), by root-qualified
    /// path: `SymbolPath::new(["m", "init"])` names `init` in the nested
    /// module `m`.
    pub extra_entry_points: Vec<SymbolPath>,
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
    pub removed_functions: Vec<SymbolPath>,
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
    let symbols = analyses.require::<SymbolTable>(ctx, module.op());
    let graph = analyses.require::<CallGraph>(ctx, module.op());
    run(ctx, module, &config, &symbols, &graph, func::Func::matches)
}

/// Eliminate the unreachable function definitions `is_function` identifies.
///
/// This is global DCE for a dialect whose function definition is not
/// `func.func`, such as a source-level callable. It follows the same roots
/// and reference edges, and builds its own symbol table and call graph.
pub fn eliminate_dead_definitions(
    ctx: &mut IrContext,
    module: Module,
    is_function: FunctionDefinition,
) -> GlobalDceResult {
    eliminate_dead_definitions_with_config(ctx, module, GlobalDceConfig::default(), is_function)
}

/// [`eliminate_dead_definitions`] with custom configuration.
pub fn eliminate_dead_definitions_with_config(
    ctx: &mut IrContext,
    module: Module,
    config: GlobalDceConfig,
    is_function: FunctionDefinition,
) -> GlobalDceResult {
    let symbols = SymbolTable::collect(ctx, module);
    let graph = call_graph_over(ctx, module.op(), &symbols, is_function);
    run(ctx, module, &config, &symbols, &graph, is_function)
}

/// The function definitions `is_function` identifies that are reachable
/// from the roots [`eliminate_dead_definitions_with_config`] keeps: the
/// definitions it would not remove.
pub fn reachable_definitions_with_config(
    ctx: &IrContext,
    module: Module,
    config: &GlobalDceConfig,
    is_function: FunctionDefinition,
) -> HashSet<OpRef> {
    let symbols = SymbolTable::collect(ctx, module);
    let graph = call_graph_over(ctx, module.op(), &symbols, is_function);
    reachable(ctx, module, config, &symbols, &graph)
}

/// The functions of `graph` reachable from the roots of `module`.
fn reachable(
    ctx: &IrContext,
    module: Module,
    config: &GlobalDceConfig,
    symbols: &SymbolTable,
    graph: &CallGraph,
) -> HashSet<OpRef> {
    let is_candidate = |op| config.recursive || nearest_table(ctx, op) == Some(module.op());
    let extra: HashSet<OpRef> = config
        .extra_entry_points
        .iter()
        .filter_map(|path| symbols.resolve_in(ctx, module.op(), path))
        .collect();
    let mut roots: HashSet<OpRef> = graph
        .functions
        .iter()
        .copied()
        .filter(|&op| !is_candidate(op) || is_root(ctx, module, op) || extra.contains(&op))
        .collect();
    roots.extend(graph.module_references.iter().copied());
    compute_reachable(graph, roots)
}

/// Eliminate the functions of `module` not reachable from its roots.
///
/// Every definition of a duplicated name is a separate function, which no
/// reference reaches. With `recursive: false`, functions in nested modules
/// are neither removed nor analyzed; they are kept as roots.
fn run(
    ctx: &mut IrContext,
    module: Module,
    config: &GlobalDceConfig,
    symbols: &SymbolTable,
    graph: &CallGraph,
    is_function: FunctionDefinition,
) -> GlobalDceResult {
    let is_candidate = |op| config.recursive || nearest_table(ctx, op) == Some(module.op());
    let reachable = reachable(ctx, module, config, symbols, graph);

    let dead: Vec<OpRef> = graph
        .functions
        .iter()
        .copied()
        .filter(|&op| is_function(ctx, op) && is_candidate(op) && !reachable.contains(&op))
        .collect();
    let mut removed = Vec::with_capacity(dead.len());
    for op in dead {
        removed.extend(qualified_name(ctx, op));
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

/// Whether the function `op` is a reachability root: `main` or `_start` in
/// the root module, or a function definition with an `abi` attribute
/// (externally callable).
fn is_root(ctx: &IrContext, module: Module, op: OpRef) -> bool {
    let entry = nearest_table(ctx, op) == Some(module.op())
        && matches!(
            ctx.op_attrs(op).get_str("sym_name"),
            Some("main" | "_start")
        );
    entry || (ctx.op_attrs(op).contains_key("abi") && ctx.op_has_regions(op))
}

/// Functions reachable from `roots` via BFS over call and reference edges.
fn compute_reachable(graph: &CallGraph, roots: HashSet<OpRef>) -> HashSet<OpRef> {
    let mut reachable = HashSet::default();
    let mut worklist: VecDeque<OpRef> = roots.into_iter().collect();

    while let Some(func) = worklist.pop_front() {
        if !reachable.insert(func) {
            continue;
        }
        if let Some(callees) = graph.edges.get(&func) {
            worklist.extend(
                callees
                    .iter()
                    .filter(|callee| !reachable.contains(*callee))
                    .copied(),
            );
        }
    }

    reachable
}

#[cfg(test)]
mod prop_tests;

/// A module pass that runs [`eliminate_dead_functions_with_config`] with
/// `config`.
pub fn global_dce_pass(config: GlobalDceConfig) -> impl Pass<Target = core::Module> {
    pass_fn("global-dce", move |ctx, module: core::Module, analyses| {
        eliminate_dead_functions_with_config(ctx, module.into(), config.clone(), analyses);
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbol::SymbolPath;
    use crate::*;

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
  func.func @named_by_module() {
    func.return
  }
}"#,
        );
        ctx.op_attrs_mut(module.op()).insert(
            "entry",
            Attribute::SymbolRef(SymbolPath::new(["named_by_module"])),
        );

        eliminate_dead_functions(&mut ctx, module, &mut Default::default());

        assert_eq!(
            surviving_functions(&ctx, module),
            ["exported", "captured", "named_by_module"]
                .map(|name| SymbolPath::new([name]))
                .into_iter()
                .collect::<HashSet<_>>()
        );
    }

    fn surviving_functions(ctx: &IrContext, module: Module) -> HashSet<SymbolPath> {
        SymbolTable::collect(ctx, module)
            .symbols()
            .iter()
            .filter(|&&op| func::Func::matches(ctx, op))
            .filter_map(|&op| qualified_name(ctx, op))
            .collect()
    }

    #[test]
    fn reachability_follows_root_qualified_references() {
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(
            &mut ctx,
            r#"core.module @root {
  func.func @main() {
    func.call {callee = @a::@same}
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

        assert_eq!(result.removed_functions, [SymbolPath::new(["b", "same"])]);
        assert_eq!(
            surviving_functions(&ctx, module),
            [SymbolPath::new(["main"]), SymbolPath::new(["a", "same"])]
                .into_iter()
                .collect::<HashSet<_>>()
        );
    }

    #[test]
    fn extra_entry_point_names_a_nested_function_by_qualified_name() {
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(
            &mut ctx,
            r#"core.module @root {
  core.module @m {
    func.func @init() {
      func.return
    }
  }
}"#,
        );
        let config = GlobalDceConfig {
            extra_entry_points: vec![SymbolPath::new(["m", "init"])],
            recursive: true,
        };

        let result =
            eliminate_dead_functions_with_config(&mut ctx, module, config, &mut Default::default());

        assert_eq!(result.removed_functions, [] as [SymbolPath; 0]);
    }

    #[test]
    fn definitions_inside_function_bodies_go_with_their_host() {
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(
            &mut ctx,
            r#"core.module @root {
  func.func @main() {
    func.call {callee = @host}
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

        assert_eq!(result.removed_functions, [SymbolPath::new(["dead_host"])]);
        assert_eq!(
            surviving_functions(&ctx, module),
            [SymbolPath::new(["main"]), SymbolPath::new(["host"])]
                .into_iter()
                .collect::<HashSet<_>>()
        );
    }
}
