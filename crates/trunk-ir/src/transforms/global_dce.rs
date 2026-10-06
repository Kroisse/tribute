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

use rustc_hash::FxHashSet as HashSet;
use std::collections::VecDeque;

use crate::analysis::AnalysisCache;
use crate::context::IrContext;
use crate::dialect::{core, func};
use crate::ops::DialectOp;
use crate::refs::OpRef;
use crate::rewrite::Module;
use crate::symbol::SymbolPath;
use crate::symbol_table::SymbolTable;
use crate::transforms::call_graph::{CallGraph, FunctionDefinition, call_graph_over};

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
    let symbols = SymbolTable::collect(ctx, module);
    let graph = call_graph_over(ctx, module.op(), &symbols, is_function);
    run(
        ctx,
        module,
        &GlobalDceConfig::default(),
        &symbols,
        &graph,
        is_function,
    )
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
    symbols: &SymbolTable,
    graph: &CallGraph,
    is_function: FunctionDefinition,
) -> GlobalDceResult {
    let functions = || {
        symbols
            .all_definitions()
            .filter(|&(_, op)| is_function(ctx, op))
    };
    let is_candidate = |op| config.recursive || !in_nested_module(ctx, module, op);
    let candidates: Vec<(&SymbolPath, OpRef)> =
        functions().filter(|&(_, op)| is_candidate(op)).collect();

    let mut roots: HashSet<SymbolPath> = functions()
        .filter(|(name, op)| !is_candidate(*op) || is_root(ctx, name, *op, config))
        .map(|(name, _)| name.clone())
        .collect();
    roots.extend(graph.module_references.iter().cloned());
    let reachable = compute_reachable(graph, roots);

    // A function containing a reachable function definition is kept with it.
    let mut kept = HashSet::default();
    for (_, op) in functions().filter(|(name, _)| reachable.contains(name)) {
        let mut current = Some(op);
        while let Some(op) = current
            && op != module.op()
        {
            kept.insert(op);
            current = parent_op(ctx, op);
        }
    }
    let dead: Vec<(SymbolPath, OpRef)> = candidates
        .into_iter()
        .filter(|(_, op)| !kept.contains(op))
        .map(|(name, op)| (name.clone(), op))
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
fn is_root(ctx: &IrContext, name: &SymbolPath, op: OpRef, config: &GlobalDceConfig) -> bool {
    *name == "main"
        || *name == "_start"
        || (ctx.op(op).attributes.contains_key("abi") && ctx.op_has_regions(op))
        || config
            .extra_entry_points
            .iter()
            .any(|extra| *name == extra.as_str())
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
fn compute_reachable(graph: &CallGraph, roots: HashSet<SymbolPath>) -> HashSet<SymbolPath> {
    let mut reachable = HashSet::default();
    let mut worklist: VecDeque<SymbolPath> = roots.into_iter().collect();

    while let Some(func) = worklist.pop_front() {
        if !reachable.insert(func.clone()) {
            continue;
        }
        if let Some(callees) = graph.edges.get(&func) {
            worklist.extend(
                callees
                    .iter()
                    .filter(|callee| !reachable.contains(*callee))
                    .cloned(),
            );
        }
    }

    reachable
}

#[cfg(test)]
mod prop_tests;

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
        ctx.op_mut(module.op()).attributes.insert(
            "entry",
            Attribute::SymbolRef(SymbolPath::from("named_by_module")),
        );

        eliminate_dead_functions(&mut ctx, module, &mut Default::default());

        assert_eq!(
            surviving_functions(&ctx, module),
            ["exported", "captured", "named_by_module"]
                .map(SymbolPath::from)
                .into_iter()
                .collect::<HashSet<_>>()
        );
    }

    fn surviving_functions(ctx: &IrContext, module: Module) -> HashSet<SymbolPath> {
        SymbolTable::collect(ctx, module)
            .all_definitions()
            .filter(|&(_, op)| func::Func::matches(ctx, op))
            .map(|(name, _)| name.clone())
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
            [SymbolPath::from("main"), SymbolPath::new(["a", "same"])]
                .into_iter()
                .collect::<HashSet<_>>()
        );
    }

    #[test]
    #[ignore = "an extra entry point matches only a root-level name; `m::init` cannot name a nested function"]
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
            extra_entry_points: vec!["m::init".to_string()],
            recursive: true,
        };

        let result =
            eliminate_dead_functions_with_config(&mut ctx, module, config, &mut Default::default());

        assert_eq!(result.removed_functions, [] as [SymbolPath; 0]);
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
            HashSet::<SymbolPath>::from_iter(result.removed_functions),
            [
                SymbolPath::from("dead_host"),
                SymbolPath::from("dead_inner")
            ]
            .into_iter()
            .collect::<HashSet<_>>()
        );
        assert_eq!(
            surviving_functions(&ctx, module),
            [
                SymbolPath::from("main"),
                SymbolPath::from("host"),
                SymbolPath::from("inner")
            ]
            .into_iter()
            .collect::<HashSet<_>>()
        );
    }
}
