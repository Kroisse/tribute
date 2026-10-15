//! Symbols over a module tree, resolved by the MLIR rule.
//!
//! A `core.module` is a symbol table. A definition registers its `sym_name` in
//! the module that directly encloses it, and a named nested module is itself a
//! symbol of its parent. A reference's first component is looked up in the
//! module nearest to the operation that holds it, never in an enclosing one;
//! each further component descends into the module the previous one names.
//! Every operation that carries a symbol `sym_name`, including a named
//! `core.module`, is a symbol of the module that encloses it, and a name
//! defined more than once in one module is an IR error. Consumers check the
//! kind of the symbol they resolve.
//!
//! Definitions are keyed by their root-qualified name: the names of the nested
//! `core.module`s that enclose the definition, excluding the root module
//! itself, followed by its `sym_name`. Each name is one path component as it
//! is written. [`absolute_reference`] turns a reference into that key.

use rustc_hash::FxHashMap as HashMap;
use smallvec::SmallVec;

use crate::analysis::{Analysis, AnalysisContext, AnalysisError, InfallibleAnalysis};
use crate::context::IrContext;
use crate::dialect::core;
use crate::ops::DialectOp;
use crate::refs::{OpRef, RegionRef};
use crate::rewrite::Module;
use crate::symbol::{Symbol, SymbolPath};

/// The `sym_name` attribute of symbol definitions and modules.
const SYM_NAME: &str = "sym_name";

/// Definitions in a module tree, keyed by root-qualified name.
#[derive(Debug, Default, Clone)]
pub struct SymbolTable {
    /// Every definition of each name, in traversal order. A name with more
    /// than one definition is duplicated.
    definitions: HashMap<SymbolPath, SmallVec<[OpRef; 1]>>,
}

impl SymbolTable {
    /// Collect every definition in `module` and its nested modules. For a
    /// nested `module`, names still start from the root module.
    ///
    /// Within an [`AnalysisCache`](crate::analysis::AnalysisCache), query the
    /// table as an analysis instead.
    pub fn collect(ctx: &IrContext, module: Module) -> Self {
        let mut table = Self::default();
        if let Some(body) = module.body(ctx) {
            table.collect_region(ctx, body, &module_path(ctx, module.op()));
        }
        table
    }

    fn collect_region(&mut self, ctx: &IrContext, region: RegionRef, path: &SymbolPath) {
        for &block in &ctx.region(region).blocks {
            for &op in &ctx.block(block).ops {
                if core::Module::matches(ctx, op) {
                    let mut nested = path.clone();
                    if let Some(name) = ctx.op(op).attributes.get_str(ctx, SYM_NAME) {
                        nested.push(name);
                        // A named module is a symbol of its parent, so it
                        // shares the parent's names with its definitions.
                        self.definitions.entry(nested.clone()).or_default().push(op);
                    }
                    for region in ctx.op_regions(op) {
                        self.collect_region(ctx, region, &nested);
                    }
                } else {
                    if let Some(name) = ctx.op(op).attributes.get_str(ctx, SYM_NAME) {
                        self.definitions
                            .entry(path.child(name))
                            .or_default()
                            .push(op);
                    }
                    // Only modules contribute path components, as in
                    // `qualified_name`.
                    for region in ctx.op_regions(op) {
                        self.collect_region(ctx, region, path);
                    }
                }
            }
        }
    }

    /// The unique definition the reference `reference`, held by `op`, names.
    ///
    /// Returns `None` for an unknown name or one defined more than once.
    pub fn resolve_from(&self, ctx: &IrContext, op: OpRef, reference: &[Symbol]) -> Option<OpRef> {
        self.resolve(&absolute_reference(ctx, op, reference))
    }

    /// The unique definition named by a root-qualified reference.
    ///
    /// Returns `None` for an unknown name or one defined more than once.
    pub fn resolve(&self, reference: &[Symbol]) -> Option<OpRef> {
        match self.definitions_of(reference) {
            &[op] => Some(op),
            _ => None,
        }
    }

    /// Every definition of a qualified name, in traversal order; empty for an
    /// unknown name.
    ///
    /// For diagnostics that continue after [`Self::duplicates`] has already
    /// been reported; lowering must use [`Self::resolve`].
    pub fn definitions_of(&self, reference: &[Symbol]) -> &[OpRef] {
        self.definitions.get(reference).map_or(&[], |ops| ops)
    }

    /// Qualified names defined more than once, sorted by name, with every
    /// definition in traversal order.
    pub fn duplicates(&self) -> Vec<(&SymbolPath, &[OpRef])> {
        let mut duplicates: Vec<_> = self.iter().filter(|(_, ops)| ops.len() > 1).collect();
        duplicates.sort_unstable_by_key(|(a, _)| *a);
        duplicates
    }

    /// Every collected name with its definitions, in unspecified order.
    pub fn iter(&self) -> impl Iterator<Item = (&SymbolPath, &[OpRef])> + '_ {
        self.definitions
            .iter()
            .map(|(name, ops)| (name, ops.as_slice()))
    }

    /// Every collected definition, including each duplicate of a name.
    pub fn all_definitions(&self) -> impl Iterator<Item = (&SymbolPath, OpRef)> + '_ {
        self.iter()
            .flat_map(|(name, ops)| ops.iter().map(move |&op| (name, op)))
    }
}

/// `SymbolTable` as an [`Analysis`]: expects `target` to be a `core.module` op
/// and delegates to [`SymbolTable::collect`].
impl Analysis for SymbolTable {
    fn compute(ctx: &mut AnalysisContext<'_>, target: OpRef) -> Result<Self, AnalysisError> {
        let module = Module::new(ctx.ir(), target)
            .expect("SymbolTable analysis target must be a `core.module` op");
        Ok(Self::collect(ctx.ir(), module))
    }
}

impl InfallibleAnalysis for SymbolTable {}

/// The root-qualified name of the definition `op`, which must carry a
/// `sym_name`.
pub fn qualified_name(ctx: &IrContext, op: OpRef) -> Option<SymbolPath> {
    let name = ctx.op(op).attributes.get_str(ctx, SYM_NAME)?;
    let module =
        parent_op(ctx, op).map_or_else(SymbolPath::root, |parent| module_path(ctx, parent));
    Some(module.child(name))
}

/// The root-qualified name of the definition the reference `reference`,
/// held by `op`, names: the reference resolved in the module nearest to `op`.
pub fn absolute_reference(ctx: &IrContext, op: OpRef, reference: &[Symbol]) -> SymbolPath {
    let mut path = module_path(ctx, op);
    for component in reference {
        path.push(component.clone());
    }
    path
}

/// The root-qualified path of the modules enclosing `op`, including `op`
/// itself if it is a module: the module whose symbols a reference held by
/// `op` names.
pub fn module_path(ctx: &IrContext, op: OpRef) -> SymbolPath {
    let mut path = Vec::new();
    let mut current = Some(op);
    while let Some(op) = current {
        let parent = parent_op(ctx, op);
        // The root module has no parent and contributes nothing.
        if core::Module::matches(ctx, op) && parent.is_some() {
            path.extend(
                ctx.op(op)
                    .attributes
                    .get_str(ctx, SYM_NAME)
                    .map(Symbol::new),
            );
        }
        current = parent;
    }
    SymbolPath::new(path.into_iter().rev())
}

fn parent_op(ctx: &IrContext, op: OpRef) -> Option<OpRef> {
    let block = ctx.op(op).parent_block?;
    ctx.region(ctx.block(block).parent_region?).parent_op
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_test_module;

    const NESTED: &str = r#"core.module @root {
  func.func @top() {
    func.return
  }
  core.module @outer {
    func.func @same() {
      func.return
    }
    core.module @inner {
      func.func @same() {
        func.return
      }
    }
  }
}"#;

    /// A `sym_name` containing `::` is one component, distinct from the
    /// nested modules it spells out.
    #[test]
    fn qualified_sym_name_is_not_a_module_path() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @root {
  func.func @"outer::flat"() {
    func.return
  }
  core.module @outer {
    func.func @nested() {
      func.return
    }
  }
}"#,
        );
        let table = SymbolTable::collect(&ctx, module);
        let flat = SymbolPath::new(["outer::flat"]);
        let nested = SymbolPath::new(["outer", "nested"]);
        for path in [&flat, &nested] {
            let op = table
                .resolve(path)
                .unwrap_or_else(|| panic!("{path} must resolve"));
            assert_eq!(qualified_name(&ctx, op).as_ref(), Some(path));
        }
        assert_eq!(table.resolve(&SymbolPath::new(["outer", "flat"])), None);
        assert_eq!(table.resolve(&SymbolPath::new(["outer::nested"])), None);
    }

    #[test]
    fn definitions_are_keyed_by_root_qualified_path() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, NESTED);
        let table = SymbolTable::collect(&ctx, module);

        for path in [
            SymbolPath::new(["top"]),
            SymbolPath::new(["outer", "same"]),
            SymbolPath::new(["outer", "inner", "same"]),
        ] {
            let op = table
                .resolve(&path)
                .unwrap_or_else(|| panic!("{path} must resolve"));
            assert_eq!(qualified_name(&ctx, op), Some(path));
        }
        // Root-qualified keys name nothing for a bare nested name.
        assert_eq!(table.resolve(&SymbolPath::new(["same"])), None);
        assert!(table.duplicates().is_empty());
        assert_eq!(table.definitions_of(&SymbolPath::new(["same"])), &[]);
    }

    /// A reference is looked up in the module nearest to the operation that
    /// holds it, and descends into named modules, but never climbs out.
    #[test]
    fn references_resolve_in_the_nearest_module() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, NESTED);
        let table = SymbolTable::collect(&ctx, module);
        let definition = |path: &[&str]| {
            table
                .resolve(&SymbolPath::new(path.iter().copied()))
                .expect("definition")
        };
        let top = definition(&["top"]);
        let outer_same = definition(&["outer", "same"]);
        let inner_same = definition(&["outer", "inner", "same"]);
        let resolve = |from: OpRef, path: &[&str]| {
            table.resolve_from(&ctx, from, &SymbolPath::new(path.iter().copied()))
        };

        // From the root: its own definitions and, through named modules,
        // nested ones.
        assert_eq!(resolve(top, &["top"]), Some(top));
        assert_eq!(resolve(top, &["outer", "same"]), Some(outer_same));
        assert_eq!(resolve(top, &["outer", "inner", "same"]), Some(inner_same));
        // From `outer`: its own and its nested definitions; the root's `top`
        // is not visible.
        assert_eq!(resolve(outer_same, &["same"]), Some(outer_same));
        assert_eq!(resolve(outer_same, &["inner", "same"]), Some(inner_same));
        assert_eq!(resolve(outer_same, &["top"]), None);
        assert_eq!(resolve(outer_same, &["outer", "same"]), None);
        // From `inner`: only `inner`'s own definitions.
        assert_eq!(resolve(inner_same, &["same"]), Some(inner_same));
        assert_eq!(resolve(inner_same, &["inner", "same"]), None);
        // A component before the last must name a module, not a function.
        assert_eq!(resolve(top, &["top", "same"]), None);
    }

    #[test]
    fn nested_module_targets_keep_root_qualified_names() {
        let mut ctx = IrContext::new();
        let root = parse_test_module(&mut ctx, NESTED);
        let outer = root
            .ops(&ctx)
            .iter()
            .copied()
            .find_map(|op| Module::new(&ctx, op))
            .expect("outer module");
        let table = SymbolTable::collect(&ctx, outer);

        for (name, op) in table.all_definitions() {
            assert_eq!(qualified_name(&ctx, op), Some(name).cloned());
        }
        assert!(table.resolve(&SymbolPath::new(["outer", "same"])).is_some());
        assert!(table.resolve(&SymbolPath::new(["same"])).is_none());
    }

    #[test]
    fn duplicated_qualified_names_do_not_resolve() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @root {
  core.module @outer {
    func.func @twice() {
      func.return
    }
    func.func @twice() {
      func.return
    }
  }
}"#,
        );
        let table = SymbolTable::collect(&ctx, module);
        let twice = SymbolPath::new(["outer", "twice"]);
        assert_eq!(table.resolve(&twice), None);
        let duplicates = table.duplicates();
        assert_eq!(duplicates.len(), 1);
        assert_eq!(duplicates[0].0, &twice);
        assert_eq!(duplicates[0].1.len(), 2);
        assert_eq!(table.definitions_of(&twice), duplicates[0].1);
        // `outer` itself and both definitions of `twice`.
        assert_eq!(table.all_definitions().count(), 3);
    }

    /// A named module is a symbol of its parent: a definition of the same
    /// name in the parent is a duplicate, not a separate namespace.
    #[test]
    fn a_named_module_shares_its_parents_names() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @root {
  func.func @item() {
    func.return
  }
  core.module @item {
    func.func @inner() {
      func.return
    }
  }
}"#,
        );
        let table = SymbolTable::collect(&ctx, module);
        let item = SymbolPath::new(["item"]);
        assert_eq!(table.resolve(&item), None);
        assert_eq!(table.definitions_of(&item).len(), 2);
        assert_eq!(table.duplicates().len(), 1);
        // Names inside the module still resolve through it.
        assert!(table.resolve(&SymbolPath::new(["item", "inner"])).is_some());
    }

    #[test]
    fn definitions_inside_non_module_regions_resolve_like_qualified_name() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @root {
  core.module @outer {
    func.func @host() {
      func.func @hidden() {
        func.return
      }
      func.return
    }
  }
}"#,
        );
        let table = SymbolTable::collect(&ctx, module);
        let hidden = table
            .resolve(&SymbolPath::new(["outer", "hidden"]))
            .expect("a definition nested in a function body is still collected");
        assert_eq!(
            qualified_name(&ctx, hidden),
            Some(SymbolPath::new(["outer", "hidden"]))
        );
    }

    #[test]
    fn analysis_matches_direct_collection() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, NESTED);
        let mut analyses = crate::analysis::AnalysisCache::new();
        let table = analyses.require::<SymbolTable>(&ctx, module.op());
        let direct = SymbolTable::collect(&ctx, module);

        let mut names: Vec<_> = table
            .iter()
            .map(|(name, ops)| (name, ops.to_vec()))
            .collect();
        let mut expected: Vec<_> = direct
            .iter()
            .map(|(name, ops)| (name, ops.to_vec()))
            .collect();
        names.sort_unstable_by_key(|(a, _)| *a);
        expected.sort_unstable_by_key(|(a, _)| *a);
        assert_eq!(names, expected);
    }
}
