//! Symbol tables over a module tree, resolved by the MLIR rule.
//!
//! A `core.module` is a symbol table. Its symbols are the operations directly
//! in its body that carry a `sym_name`, a named nested module included; an
//! operation nested in another operation's region is not a symbol of any
//! table. A name defined more than once in one table is an IR error.
//!
//! A reference's first component is looked up in the table nearest to the
//! operation that holds it (the operation itself if it is a module), never in
//! an enclosing one; each further component is looked up in the module the
//! previous one names. Resolution yields the symbol operation, which is its
//! identity; [`qualified_name`] spells a symbol's path from the root module
//! for diagnostics and stable output. Consumers check the kind of the symbol
//! they resolve.

use rustc_hash::FxHashMap as HashMap;
use smallvec::SmallVec;

use crate::analysis::{Analysis, AnalysisContext, AnalysisError, InfallibleAnalysis};
use crate::context::IrContext;
use crate::dialect::core;
use crate::ops::DialectOp;
use crate::refs::OpRef;
use crate::rewrite::Module;
use crate::symbol::{Symbol, SymbolPath};

/// The `sym_name` attribute of symbol definitions and modules.
const SYM_NAME: &str = "sym_name";

/// The attribute that gives a declaration its linker name when that differs
/// from its `sym_name`.
pub const LINK_NAME: &str = "link_name";

/// The symbol tables of a module tree.
#[derive(Debug, Default, Clone)]
pub struct SymbolTable {
    /// The symbols of each table, by name. A name with more than one symbol is
    /// duplicated.
    tables: HashMap<OpRef, HashMap<Symbol, SmallVec<[OpRef; 1]>>>,
    /// Every symbol of every table, in traversal order.
    symbols: Vec<OpRef>,
}

impl SymbolTable {
    /// Collect the tables of `module` and of the modules nested in it.
    ///
    /// Within an [`AnalysisCache`](crate::analysis::AnalysisCache), query the
    /// table as an analysis instead.
    pub fn collect(ctx: &IrContext, module: Module) -> Self {
        let mut table = Self::default();
        table.collect_table(ctx, module.op());
        table
    }

    fn collect_table(&mut self, ctx: &IrContext, table: OpRef) {
        let mut names: HashMap<Symbol, SmallVec<[OpRef; 1]>> = HashMap::default();
        let mut nested = Vec::new();
        for region in ctx.op_regions(table) {
            for &block in &ctx.region(region).blocks {
                for &op in &ctx.block(block).ops {
                    if let Some(name) = ctx.op_attrs(op).get_str(SYM_NAME) {
                        names.entry(Symbol::new(name)).or_default().push(op);
                        self.symbols.push(op);
                    }
                    if core::Module::matches(ctx, op) {
                        nested.push(op);
                    }
                }
            }
        }
        self.tables.insert(table, names);
        for module in nested {
            self.collect_table(ctx, module);
        }
    }

    /// The symbols named `name` in the table `table`, in traversal order;
    /// empty for an unknown name or a table outside the collected tree.
    pub fn lookup(&self, table: OpRef, name: &Symbol) -> &[OpRef] {
        self.tables
            .get(&table)
            .and_then(|names| names.get(name))
            .map_or(&[], |ops| ops)
    }

    /// Every symbol the reference `reference` names when it is looked up in
    /// `table`; empty when a component before the last does not name exactly
    /// one module.
    ///
    /// For diagnostics that continue after [`Self::duplicates`] has already
    /// been reported; lowering must use [`Self::resolve`] or
    /// [`Self::resolve_in`].
    pub fn definitions_in(&self, ctx: &IrContext, table: OpRef, reference: &[Symbol]) -> &[OpRef] {
        let Some((last, modules)) = reference.split_last() else {
            return &[];
        };
        let mut table = table;
        for name in modules {
            match self.lookup(table, name) {
                &[module] if core::Module::matches(ctx, module) => table = module,
                _ => return &[],
            }
        }
        self.lookup(table, last)
    }

    /// The unique symbol the reference `reference` names when it is looked up
    /// in `table`.
    pub fn resolve_in(&self, ctx: &IrContext, table: OpRef, reference: &[Symbol]) -> Option<OpRef> {
        match self.definitions_in(ctx, table, reference) {
            &[op] => Some(op),
            _ => None,
        }
    }

    /// The unique symbol the reference `reference`, held by `op`, names.
    ///
    /// Returns `None` for an unknown name, one defined more than once, or an
    /// operation outside every table.
    pub fn resolve(&self, ctx: &IrContext, op: OpRef, reference: &[Symbol]) -> Option<OpRef> {
        self.resolve_in(ctx, nearest_table(ctx, op)?, reference)
    }

    /// Every symbol the reference `reference`, held by `op`, names; see
    /// [`Self::definitions_in`].
    pub fn definitions(&self, ctx: &IrContext, op: OpRef, reference: &[Symbol]) -> &[OpRef] {
        nearest_table(ctx, op).map_or(&[], |table| self.definitions_in(ctx, table, reference))
    }

    /// The symbols of each name defined more than once in one table, in
    /// traversal order of their first symbol.
    pub fn duplicates(&self) -> Vec<&[OpRef]> {
        let order: HashMap<OpRef, usize> = self
            .symbols
            .iter()
            .enumerate()
            .map(|(index, &op)| (op, index))
            .collect();
        let mut duplicates: Vec<&[OpRef]> = self
            .tables
            .values()
            .flat_map(|names| names.values())
            .filter(|ops| ops.len() > 1)
            .map(|ops| ops.as_slice())
            .collect();
        duplicates.sort_unstable_by_key(|ops| order[&ops[0]]);
        duplicates
    }

    /// Every symbol of every table, named modules included, in traversal
    /// order.
    pub fn symbols(&self) -> &[OpRef] {
        &self.symbols
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

/// The table a reference held by `op` is looked up in: `op` itself if it is a
/// module, and the nearest module enclosing it otherwise.
pub fn nearest_table(ctx: &IrContext, op: OpRef) -> Option<OpRef> {
    let mut current = Some(op);
    while let Some(op) = current {
        if core::Module::matches(ctx, op) {
            return Some(op);
        }
        current = parent_op(ctx, op);
    }
    None
}

/// The path of the symbol `op` from the root module: the names of the
/// nested `core.module`s that enclose it, excluding the root module, then its
/// own `sym_name`. For diagnostics and stable output; a symbol's identity is
/// its operation.
pub fn qualified_name(ctx: &IrContext, op: OpRef) -> Option<SymbolPath> {
    let name = ctx.op_attrs(op).get_str(SYM_NAME)?;
    let module =
        parent_op(ctx, op).map_or_else(SymbolPath::root, |parent| module_path(ctx, parent));
    Some(module.child(name))
}

/// The name the definition `op` is linked under: its `link_name`, or its
/// `sym_name` when it has none. Only symbol emission reads it; references and
/// symbol tables use the `sym_name`.
pub fn link_name(ctx: &IrContext, op: OpRef) -> Option<Symbol> {
    let attributes = &ctx.op(op).attributes;
    attributes
        .view(ctx)
        .get_str(LINK_NAME)
        .or_else(|| attributes.view(ctx).get_str(SYM_NAME))
        .map(Symbol::new)
}

/// The root-qualified path of the modules enclosing `op`, including `op`
/// itself if it is a module.
fn module_path(ctx: &IrContext, op: OpRef) -> SymbolPath {
    let mut path = Vec::new();
    let mut current = Some(op);
    while let Some(op) = current {
        let parent = parent_op(ctx, op);
        // The root module has no parent and contributes nothing.
        if core::Module::matches(ctx, op) && parent.is_some() {
            path.extend(ctx.op_attrs(op).get_str(SYM_NAME).map(Symbol::new));
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

    fn path(components: &[&str]) -> SymbolPath {
        SymbolPath::new(components.iter().copied())
    }

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
        let root = module.op();
        for components in [&["outer::flat"][..], &["outer", "nested"]] {
            let reference = path(components);
            let op = table
                .resolve_in(&ctx, root, &reference)
                .unwrap_or_else(|| panic!("{reference} must resolve"));
            assert_eq!(qualified_name(&ctx, op), Some(reference));
        }
        assert_eq!(
            table.resolve_in(&ctx, root, &path(&["outer", "flat"])),
            None
        );
        assert_eq!(
            table.resolve_in(&ctx, root, &path(&["outer::nested"])),
            None
        );
    }

    /// A reference is looked up in the module nearest to the operation that
    /// holds it, and descends into named modules, but never climbs out.
    #[test]
    fn references_resolve_in_the_nearest_module() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, NESTED);
        let table = SymbolTable::collect(&ctx, module);
        let definition = |components: &[&str]| {
            table
                .resolve_in(&ctx, module.op(), &path(components))
                .expect("definition")
        };
        let top = definition(&["top"]);
        let outer = definition(&["outer"]);
        let outer_same = definition(&["outer", "same"]);
        let inner_same = definition(&["outer", "inner", "same"]);
        assert_eq!(
            qualified_name(&ctx, inner_same),
            Some(path(&["outer", "inner", "same"]))
        );
        let resolve =
            |from: OpRef, components: &[&str]| table.resolve(&ctx, from, &path(components));

        // From the root: its own symbols and, through named modules, nested
        // ones.
        assert_eq!(resolve(top, &["top"]), Some(top));
        assert_eq!(resolve(top, &["outer"]), Some(outer));
        assert_eq!(resolve(top, &["outer", "same"]), Some(outer_same));
        assert_eq!(resolve(top, &["outer", "inner", "same"]), Some(inner_same));
        // From `outer`: its own and its nested symbols; the root's `top` is not
        // visible.
        assert_eq!(resolve(outer_same, &["same"]), Some(outer_same));
        assert_eq!(resolve(outer_same, &["inner", "same"]), Some(inner_same));
        assert_eq!(resolve(outer_same, &["top"]), None);
        assert_eq!(resolve(outer_same, &["outer", "same"]), None);
        // A module's own reference attributes are looked up in its own table.
        assert_eq!(resolve(outer, &["same"]), Some(outer_same));
        // From `inner`: only `inner`'s own symbols.
        assert_eq!(resolve(inner_same, &["same"]), Some(inner_same));
        assert_eq!(resolve(inner_same, &["inner", "same"]), None);
        // A component before the last must name a module, not a function.
        assert_eq!(resolve(top, &["top", "same"]), None);
    }

    #[test]
    fn duplicated_names_do_not_resolve() {
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
        let twice = path(&["outer", "twice"]);
        assert_eq!(table.resolve_in(&ctx, module.op(), &twice), None);
        let duplicates = table.duplicates();
        assert_eq!(duplicates.len(), 1);
        assert_eq!(duplicates[0].len(), 2);
        assert_eq!(
            table.definitions_in(&ctx, module.op(), &twice),
            duplicates[0]
        );
        // `outer` itself and both definitions of `twice`.
        assert_eq!(table.symbols().len(), 3);
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
        let item = path(&["item"]);
        assert_eq!(table.resolve_in(&ctx, module.op(), &item), None);
        assert_eq!(table.definitions_in(&ctx, module.op(), &item).len(), 2);
        assert_eq!(table.duplicates().len(), 1);
        // The module is not unique, so nothing resolves through it.
        assert_eq!(
            table.resolve_in(&ctx, module.op(), &path(&["item", "inner"])),
            None
        );
    }

    /// Only the operations directly in a module's body are its symbols.
    #[test]
    fn definitions_inside_non_module_regions_are_not_symbols() {
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
        assert!(
            table
                .resolve_in(&ctx, module.op(), &path(&["outer", "host"]))
                .is_some()
        );
        assert_eq!(
            table.resolve_in(&ctx, module.op(), &path(&["outer", "hidden"])),
            None
        );
        assert_eq!(table.symbols().len(), 2);
    }

    #[test]
    fn analysis_matches_direct_collection() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, NESTED);
        let mut analyses = crate::analysis::AnalysisCache::new();
        let table = analyses.require::<SymbolTable>(&ctx, module.op());
        let direct = SymbolTable::collect(&ctx, module);
        assert_eq!(table.symbols(), direct.symbols());
    }
}
