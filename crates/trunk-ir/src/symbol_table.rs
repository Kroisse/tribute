//! Root-qualified symbols over a module tree.
//!
//! A symbol reference always names its target by its path from the root
//! module: the names of the nested `core.module`s that enclose the
//! definition, excluding the root module itself, joined with `::` and followed
//! by the definition's `sym_name`. References are never resolved relative to
//! the referencing operation's module. A qualified name defined more than once
//! is an IR error.

use itertools::Itertools;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::context::IrContext;
use crate::dialect::core;
use crate::ops::DialectOp;
use crate::refs::{OpRef, RegionRef};
use crate::rewrite::Module;
use crate::symbol::Symbol;

/// The `sym_name` attribute of symbol definitions and modules.
const SYM_NAME: &str = "sym_name";

/// Definitions in a module tree, keyed by root-qualified name.
#[derive(Debug, Default, Clone)]
pub struct SymbolTable {
    /// Every definition of each name, in traversal order. A name with more
    /// than one definition is duplicated.
    definitions: FxHashMap<Symbol, SmallVec<[OpRef; 1]>>,
}

impl SymbolTable {
    /// Collect every operation accepted by `is_definition` in `module` and its
    /// nested modules.
    pub fn collect(
        ctx: &IrContext,
        module: Module,
        is_definition: impl Fn(&IrContext, OpRef) -> bool,
    ) -> Self {
        let mut table = Self::default();
        if let Some(body) = module.body(ctx) {
            table.collect_region(ctx, body, &[], &is_definition);
        }
        table
    }

    fn collect_region(
        &mut self,
        ctx: &IrContext,
        region: RegionRef,
        path: &[Symbol],
        is_definition: &impl Fn(&IrContext, OpRef) -> bool,
    ) {
        for &block in &ctx.region(region).blocks {
            for &op in &ctx.block(block).ops {
                if core::Module::matches(ctx, op) {
                    let mut nested = path.to_vec();
                    nested.extend(ctx.op(op).attributes.get_symbol(SYM_NAME));
                    for &region in &ctx.op(op).regions {
                        self.collect_region(ctx, region, &nested, is_definition);
                    }
                } else {
                    if is_definition(ctx, op)
                        && let Some(name) = ctx.op(op).attributes.get_symbol(SYM_NAME)
                    {
                        self.definitions
                            .entry(qualify(path, name))
                            .or_default()
                            .push(op);
                    }
                    // Only modules contribute path components, as in
                    // `qualified_name`.
                    for &region in &ctx.op(op).regions {
                        self.collect_region(ctx, region, path, is_definition);
                    }
                }
            }
        }
    }

    /// The unique definition named by a root-qualified reference.
    ///
    /// Returns `None` for an unknown name or one defined more than once.
    pub fn resolve(&self, reference: Symbol) -> Option<OpRef> {
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
    pub fn definitions_of(&self, reference: Symbol) -> &[OpRef] {
        self.definitions.get(&reference).map_or(&[], |ops| ops)
    }

    /// Qualified names defined more than once, sorted by name, with every
    /// definition in traversal order.
    pub fn duplicates(&self) -> Vec<(Symbol, &[OpRef])> {
        let mut duplicates: Vec<_> = self.iter().filter(|(_, ops)| ops.len() > 1).collect();
        duplicates.sort_unstable_by_key(|&(name, _)| name);
        duplicates
    }

    /// Every collected name with its definitions, in unspecified order.
    pub fn iter(&self) -> impl Iterator<Item = (Symbol, &[OpRef])> + '_ {
        self.definitions
            .iter()
            .map(|(&name, ops)| (name, ops.as_slice()))
    }

    /// Every collected definition, including each duplicate of a name.
    pub fn all_definitions(&self) -> impl Iterator<Item = (Symbol, OpRef)> + '_ {
        self.iter()
            .flat_map(|(name, ops)| ops.iter().map(move |&op| (name, op)))
    }
}

/// The root-qualified name of the definition `op`, which must carry a
/// `sym_name`.
pub fn qualified_name(ctx: &IrContext, op: OpRef) -> Option<Symbol> {
    let name = ctx.op(op).attributes.get_symbol(SYM_NAME)?;
    let mut path = Vec::new();
    let mut current = op;
    while let Some(parent) = parent_op(ctx, current) {
        // The root module has no parent and contributes nothing.
        if core::Module::matches(ctx, parent) && parent_op(ctx, parent).is_some() {
            path.extend(ctx.op(parent).attributes.get_symbol(SYM_NAME));
        }
        current = parent;
    }
    path.reverse();
    Some(qualify(&path, name))
}

fn parent_op(ctx: &IrContext, op: OpRef) -> Option<OpRef> {
    let block = ctx.op(op).parent_block?;
    ctx.region(ctx.block(block).parent_region?).parent_op
}

fn qualify(path: &[Symbol], name: Symbol) -> Symbol {
    if path.is_empty() {
        return name;
    }
    Symbol::from_dynamic(&path.iter().chain(std::iter::once(&name)).join("::"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialect::func;
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

    fn is_func(ctx: &IrContext, op: OpRef) -> bool {
        func::Func::matches(ctx, op)
    }

    #[test]
    fn definitions_are_keyed_by_root_qualified_path() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, NESTED);
        let table = SymbolTable::collect(&ctx, module, is_func);

        for name in ["top", "outer::same", "outer::inner::same"] {
            let op = table
                .resolve(Symbol::from_dynamic(name))
                .unwrap_or_else(|| panic!("{name} must resolve"));
            assert_eq!(qualified_name(&ctx, op), Some(Symbol::from_dynamic(name)));
        }
        // A reference is never resolved relative to a nested module.
        assert_eq!(table.resolve(Symbol::new("same")), None);
        assert!(table.duplicates().is_empty());
        assert_eq!(table.definitions_of(Symbol::new("same")), &[]);
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
        let table = SymbolTable::collect(&ctx, module, is_func);
        let twice = Symbol::from_dynamic("outer::twice");
        assert_eq!(table.resolve(twice), None);
        let duplicates = table.duplicates();
        assert_eq!(duplicates.len(), 1);
        assert_eq!(duplicates[0].0, twice);
        assert_eq!(duplicates[0].1.len(), 2);
        assert_eq!(table.definitions_of(twice), duplicates[0].1);
        assert_eq!(table.all_definitions().count(), 2);
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
        let table = SymbolTable::collect(&ctx, module, is_func);
        let hidden = table
            .resolve(Symbol::from_dynamic("outer::hidden"))
            .expect("a definition nested in a function body is still collected");
        assert_eq!(
            qualified_name(&ctx, hidden),
            Some(Symbol::from_dynamic("outer::hidden"))
        );
    }
}
