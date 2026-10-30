//! Global Dead Code Elimination (DCE) pass for arena IR.
//!
//! Removes the [`SymbolDefinition`]s that no live operation uses. The live
//! operations are:
//! - every operation that is not a symbol definition of a module's table
//! - every definition that is not discardable, such as a function with a
//!   body and an `abi` attribute (externally callable)
//! - the configured entry points, by root-qualified path (`main` and
//!   `_start` by default), and declarations if the configuration keeps them
//!
//! A definition's uses are the symbol references in its own attributes and
//! in its regions. A use that resolves to a definition makes it live, and
//! liveness follows the uses of live definitions. A bodyless `abi`
//! declaration is an import: it stays only while something live uses it.

use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;
use std::ops::ControlFlow;

use crate::analysis::AnalysisCache;
use crate::context::IrContext;
use crate::dialect::core;
use crate::op_interface::DynSymbolDefinition;
#[cfg(doc)]
use crate::op_interface::SymbolDefinition;
use crate::pass::{Pass, PassRunResult};
use crate::refs::OpRef;
use crate::rewrite::Module;
use crate::symbol::SymbolPath;
use crate::symbol_table::{SymbolTable, nearest_table, qualified_name};
use crate::walk::{WalkAction, walk_op, walk_region};

/// Configuration for global dead code elimination.
#[derive(Debug, Clone)]
pub struct GlobalDceConfig {
    /// The program's entry points, by root-qualified path. Default: `main`
    /// and `_start` in the root module.
    pub entry_points: Vec<SymbolPath>,
    /// Additional entry points, by root-qualified path:
    /// `SymbolPath::new(["m", "init"])` names `init` in the nested module
    /// `m`.
    pub extra_entry_points: Vec<SymbolPath>,
    /// Whether to keep every [`SymbolDefinition::is_declaration`]
    /// definition. Default: false.
    pub keep_declarations: bool,
    /// Whether to recursively process nested modules. Default: true.
    pub recursive: bool,
}

impl Default for GlobalDceConfig {
    fn default() -> Self {
        Self {
            entry_points: vec![SymbolPath::new(["main"]), SymbolPath::new(["_start"])],
            extra_entry_points: Vec::new(),
            keep_declarations: false,
            recursive: true,
        }
    }
}

/// Result of running global DCE.
pub struct GlobalDceResult {
    /// Number of definitions removed.
    pub removed_count: usize,
    /// Names of removed definitions (for debugging).
    pub removed_functions: Vec<SymbolPath>,
}

/// Eliminate unused definitions from a module using default configuration.
#[cfg(test)]
fn eliminate_dead_functions(
    ctx: &mut IrContext,
    module: Module,
    analyses: &mut AnalysisCache,
) -> GlobalDceResult {
    eliminate_dead_functions_with_config(ctx, module, GlobalDceConfig::default(), analyses)
}

/// Eliminate unused definitions with custom configuration, using the
/// [`SymbolTable`] cached in `analyses`.
fn eliminate_dead_functions_with_config(
    ctx: &mut IrContext,
    module: Module,
    config: GlobalDceConfig,
    analyses: &mut AnalysisCache,
) -> GlobalDceResult {
    let symbols = analyses.require::<SymbolTable>(ctx, module.op());
    run(ctx, module, &config, &symbols)
}

/// Eliminate the [`SymbolDefinition`]s of `module` that no live operation
/// uses, building its own symbol table.
pub fn eliminate_dead_definitions(ctx: &mut IrContext, module: Module) -> GlobalDceResult {
    eliminate_dead_definitions_with_config(ctx, module, GlobalDceConfig::default())
}

/// [`eliminate_dead_definitions`] with custom configuration.
pub fn eliminate_dead_definitions_with_config(
    ctx: &mut IrContext,
    module: Module,
    config: GlobalDceConfig,
) -> GlobalDceResult {
    let symbols = SymbolTable::collect(ctx, module);
    run(ctx, module, &config, &symbols)
}

/// The live [`SymbolDefinition`]s of `module` under `config`: the
/// definitions [`eliminate_dead_definitions_with_config`] would not remove,
/// except those of nested modules a non-recursive configuration does not
/// analyze.
pub fn reachable_definitions_with_config(
    ctx: &IrContext,
    module: Module,
    config: &GlobalDceConfig,
) -> HashSet<OpRef> {
    let symbols = SymbolTable::collect(ctx, module);
    Liveness::compute(ctx, module, config, &symbols).live
}

/// Which definitions of a module are live.
struct Liveness {
    /// The symbol definitions of the module's tables that DCE may remove.
    candidates: Vec<OpRef>,
    /// The candidates that are live.
    live: HashSet<OpRef>,
}

impl Liveness {
    fn compute(
        ctx: &IrContext,
        module: Module,
        config: &GlobalDceConfig,
        symbols: &SymbolTable,
    ) -> Self {
        let is_candidate = |op| config.recursive || nearest_table(ctx, op) == Some(module.op());
        let definitions: Vec<OpRef> = symbols
            .symbols()
            .iter()
            .copied()
            .filter(|&op| DynSymbolDefinition::is_implemented(ctx, op))
            .collect();
        let definition_set: HashSet<OpRef> = definitions.iter().copied().collect();

        let mut worklist: Vec<OpRef> = Vec::new();
        let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
            if definition_set.contains(&op) {
                return ControlFlow::Continue(WalkAction::Skip);
            }
            uses_of_operation(ctx, op, symbols, &mut worklist);
            ControlFlow::Continue(WalkAction::Advance)
        });
        for path in config.entry_points.iter().chain(&config.extra_entry_points) {
            worklist.extend(symbols.resolve_in(ctx, module.op(), path));
        }
        let mut uses: HashMap<OpRef, Vec<OpRef>> = HashMap::default();
        for &op in &definitions {
            let definition = DynSymbolDefinition::get(ctx, op)
                .expect("every definition implements SymbolDefinition");
            if !is_candidate(op)
                || !definition.is_discardable(ctx)
                || (config.keep_declarations && definition.is_declaration(ctx))
            {
                worklist.push(op);
            }
            let mut targets = Vec::new();
            uses_of_operation(ctx, op, symbols, &mut targets);
            for region in ctx.op_regions(op) {
                let _ = walk_region::<()>(ctx, region, &mut |nested| {
                    uses_of_operation(ctx, nested, symbols, &mut targets);
                    ControlFlow::Continue(WalkAction::Advance)
                });
            }
            uses.insert(op, targets);
        }

        let mut live = HashSet::default();
        while let Some(op) = worklist.pop() {
            if !definition_set.contains(&op) || !live.insert(op) {
                continue;
            }
            if let Some(targets) = uses.get(&op) {
                worklist.extend(targets.iter().filter(|t| !live.contains(*t)).copied());
            }
        }
        let candidates = definitions
            .into_iter()
            .filter(|&op| is_candidate(op))
            .collect();
        Liveness { candidates, live }
    }
}

/// Push the definitions the symbol references in `op`'s attributes resolve
/// to onto `targets`.
fn uses_of_operation(ctx: &IrContext, op: OpRef, symbols: &SymbolTable, targets: &mut Vec<OpRef>) {
    ctx.op(op).attributes.visit_symbol_refs(&mut |reference| {
        targets.extend(symbols.resolve(ctx, op, reference));
    });
}

/// Eliminate the definitions of `module` that no live operation uses.
///
/// Every definition of a duplicated name is a separate definition, which no
/// reference reaches. With `recursive: false`, definitions in nested modules
/// are neither removed nor analyzed; they are kept, with their uses.
fn run(
    ctx: &mut IrContext,
    module: Module,
    config: &GlobalDceConfig,
    symbols: &SymbolTable,
) -> GlobalDceResult {
    let Liveness { candidates, live } = Liveness::compute(ctx, module, config, symbols);
    let dead: Vec<OpRef> = candidates
        .into_iter()
        .filter(|op| !live.contains(op))
        .collect();
    let mut removed = Vec::with_capacity(dead.len());
    for op in dead {
        removed.extend(qualified_name(ctx, op));
        // `erase_op` clears the operand use-chains of the definition and its
        // body subtree, so dead definitions don't leave stale uses behind
        // (#710). A definition's body is isolated from other definitions, so
        // clearing its operand uses cannot affect a live one.
        crate::rewrite::erase_op(ctx, op);
    }

    GlobalDceResult {
        removed_count: removed.len(),
        removed_functions: removed,
    }
}

#[cfg(test)]
mod prop_tests;

/// Remove the symbol definitions no live operation uses, using the
/// [`SymbolTable`] cached in the run's analyses.
pub struct GlobalDce {
    pub config: GlobalDceConfig,
}

impl Pass for GlobalDce {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "global-dce"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: core::Module,
        analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        eliminate_dead_functions_with_config(ctx, target.into(), self.config.clone(), analyses);
        Ok(())
    }
}

#[cfg(test)]
#[crate::dialect]
mod test_symbol {
    /// A region-less definition whose `targets` name other definitions.
    #[implements(crate::op_interface::SymbolDefinition)]
    fn table(sym_name: Attr<String>, targets: Attr<[crate::attr_kind::SymbolRef]>) {}
}

#[cfg(test)]
impl crate::op_interface::SymbolDefinition for Table {
    fn is_declaration(&self, _ctx: &IrContext) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialect::func;
    use crate::ops::DialectOp;
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
            ..Default::default()
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

    fn surviving_definitions(ctx: &IrContext, module: Module) -> HashSet<SymbolPath> {
        SymbolTable::collect(ctx, module)
            .symbols()
            .iter()
            .filter(|&&op| DynSymbolDefinition::is_implemented(ctx, op))
            .filter_map(|&op| qualified_name(ctx, op))
            .collect()
    }

    fn paths<const N: usize>(names: [&str; N]) -> HashSet<SymbolPath> {
        names
            .map(|name| SymbolPath::new([name]))
            .into_iter()
            .collect()
    }

    const TABLE_MODULE: &str = r#"core.module @root {
  func.func @main() {
    test.use {table = @REFERENCE}
    func.return
  }
  func.func @dead() {
    test.use {table = @from_dead}
    func.return
  }
  test_symbol.table {sym_name = "live", targets = [@a, @b]}
  test_symbol.table {sym_name = "from_dead", targets = [@c]}
  test_symbol.table {sym_name = "unreferenced", targets = [@d]}
  func.func @a() {
    func.return
  }
  func.func @b() {
    func.call {callee = @shared}
    func.return
  }
  func.func @c() {
    func.return
  }
  func.func @d() {
    func.return
  }
  func.func @shared() {
    func.return
  }
}"#;

    #[test]
    fn a_reachable_definition_keeps_the_targets_of_its_attributes() {
        let mut ctx = IrContext::new();
        let source = TABLE_MODULE.replace("REFERENCE", "live");
        let module = crate::parser::parse_test_module(&mut ctx, &source);

        eliminate_dead_functions(&mut ctx, module, &mut Default::default());

        assert_eq!(
            surviving_definitions(&ctx, module),
            paths(["main", "live", "a", "b", "shared"])
        );
    }

    #[test]
    fn an_unreferenced_definition_goes_with_the_targets_only_it_uses() {
        let mut ctx = IrContext::new();
        let source = TABLE_MODULE.replace("REFERENCE", "nothing");
        let module = crate::parser::parse_test_module(&mut ctx, &source);

        eliminate_dead_functions(&mut ctx, module, &mut Default::default());

        assert_eq!(surviving_definitions(&ctx, module), paths(["main"]));
    }

    #[test]
    fn declarations_stay_when_configured() {
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(
            &mut ctx,
            r#"core.module @root {
  func.func @import() attributes {abi = "C"}
  func.func @unused() {
    func.return
  }
}"#,
        );
        let config = GlobalDceConfig {
            keep_declarations: true,
            ..Default::default()
        };

        let result =
            eliminate_dead_functions_with_config(&mut ctx, module, config, &mut Default::default());

        assert_eq!(result.removed_functions, [SymbolPath::new(["unused"])]);
    }
}
