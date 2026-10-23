//! What a package's names see: its own environment and the environments
//! of the packages it depends on.

use trunk_ir::{Symbol, SymbolPath};

use std::sync::LazyLock;

use super::env::{Binding, PackageEnv, Restriction};
use super::{LIBRARY_PACKAGE, PRELUDE_MODULE};
use crate::ast::FuncDefId;

/// The path of the prelude module from the root of its package.
fn prelude_module() -> &'static SymbolPath {
    static PRELUDE: LazyLock<SymbolPath> = LazyLock::new(|| SymbolPath::new(PRELUDE_MODULE));
    &PRELUDE
}

/// The packages a package depends on, each by its name.
#[derive(Clone, Debug, Default)]
pub struct Dependencies<'db> {
    packages: Vec<(Symbol, &'db PackageEnv<'db>)>,
}

impl<'db> Dependencies<'db> {
    /// No dependencies.
    pub fn none() -> Self {
        Self::default()
    }

    /// The dependency on package `std`, whose environment is `std`.
    pub fn std(std: &'db PackageEnv<'db>) -> Self {
        Self {
            packages: vec![(Symbol::new(LIBRARY_PACKAGE), std)],
        }
    }

    /// The environment of the dependency named `name`.
    pub fn get(&self, name: &Symbol) -> Option<&'db PackageEnv<'db>> {
        self.packages
            .iter()
            .find_map(|(known, env)| (known == name).then_some(*env))
    }

    /// The package whose prelude module every module sees: package `std`.
    fn prelude(&self) -> Option<(&Symbol, &'db PackageEnv<'db>)> {
        self.packages
            .iter()
            .find(|(name, _)| name == LIBRARY_PACKAGE)
            .map(|(name, env)| (name, *env))
    }

    fn iter(&self) -> impl Iterator<Item = &'db PackageEnv<'db>> + '_ {
        self.packages.iter().map(|(_, env)| *env)
    }
}

/// A package's view of names: its own environment, then its dependencies'.
///
/// Its paths are read from the package's root. A path whose first segment
/// names a dependency, and names nothing the package itself declares or
/// imports, continues from that dependency's root.
#[derive(Clone, Copy)]
pub struct Scope<'s, 'db> {
    own: &'s PackageEnv<'db>,
    dependencies: &'s Dependencies<'db>,
}

impl<'s, 'db> Scope<'s, 'db> {
    pub fn new(own: &'s PackageEnv<'db>, dependencies: &'s Dependencies<'db>) -> Self {
        Self { own, dependencies }
    }

    /// The package's own environment.
    pub fn own(&self) -> &'s PackageEnv<'db> {
        self.own
    }

    /// The dependency a path whose first segment is `first` may continue
    /// in. The package's own entries at the path come first, as they shadow
    /// the dependency's.
    fn dependency<'p>(&self, path: &'p [Symbol]) -> Option<(&'s PackageEnv<'db>, &'p [Symbol])> {
        let (first, rest) = path.split_first()?;
        Some((self.dependencies.get(first)?, rest))
    }

    /// Look up a name the package root declares or imports. The prelude's
    /// names are not the root's; [`Self::lookup_library`] finds them.
    pub fn lookup(&self, name: &Symbol) -> Option<&'s Binding<'db>> {
        self.own.lookup(name)
    }

    /// Look up `name` in `namespace`.
    pub fn lookup_qualified(
        &self,
        namespace: &[Symbol],
        name: &Symbol,
    ) -> Option<&'s Binding<'db>> {
        self.own.lookup_qualified(namespace, name).or_else(|| {
            let (env, namespace) = self.dependency(namespace)?;
            match namespace {
                [] => env.root_item(name),
                namespace => env.lookup_qualified(namespace, name),
            }
        })
    }

    /// Whether `namespace` exists.
    pub fn has_namespace(&self, namespace: &[Symbol]) -> bool {
        self.own.has_namespace(namespace)
            || self.dependency(namespace).is_some_and(|(env, namespace)| {
                namespace.is_empty() || env.has_namespace(namespace)
            })
    }

    /// The bindings of `namespace`, the package's own first.
    pub fn collect_namespace(&self, namespace: &[Symbol]) -> Vec<(Symbol, Binding<'db>)> {
        self.iter_namespace(namespace)
            .map(|(name, binding)| (name, binding.clone()))
            .collect()
    }

    /// The names and bindings of `namespace`, the package's own first.
    pub fn iter_namespace(
        &self,
        namespace: &[Symbol],
    ) -> impl Iterator<Item = (Symbol, &'s Binding<'db>)> + use<'s, 'db> {
        let mut entries: Vec<(Symbol, &'s Binding<'db>)> =
            self.own.iter_namespace(namespace).collect();
        if let Some((env, namespace)) = self.dependency(namespace) {
            let theirs: Vec<(Symbol, &'s Binding<'db>)> = match namespace {
                [] => env.root_items().collect(),
                namespace => env.iter_namespace(namespace).collect(),
            };
            for (name, binding) in theirs {
                if !entries.iter().any(|(known, _)| *known == name) {
                    entries.push((name, binding));
                }
            }
        }
        entries.into_iter()
    }

    /// Where `binding` is visible, if it is not public, in whichever
    /// package declares it.
    pub fn restriction(&self, binding: &Binding<'db>) -> Option<&'s Restriction> {
        self.own.restriction(binding).or_else(|| {
            self.dependencies
                .iter()
                .find_map(|env| env.restriction(binding))
        })
    }

    /// Where the module at `module` is visible, if it is not public.
    pub fn module_restriction(&self, module: &[Symbol]) -> Option<&'s Restriction> {
        self.own.module_restriction(module).or_else(|| {
            let (env, module) = self.dependency(module)?;
            env.module_restriction(module)
        })
    }

    /// Where the re-exported entry `name` of `namespace` is visible, if it
    /// is not public.
    pub fn entry_restriction(
        &self,
        namespace: &[Symbol],
        name: &Symbol,
    ) -> Option<&'s Restriction> {
        self.own.entry_restriction(namespace, name).or_else(|| {
            let (env, namespace) = self.dependency(namespace)?;
            env.entry_restriction(namespace, name)
        })
    }

    /// The package that holds the prelude module, if it is a dependency,
    /// and its environment. Package `std` sees its own prelude module.
    fn prelude(&self) -> Option<(Option<&'s Symbol>, &'s PackageEnv<'db>)> {
        if self.own.package().is_library() {
            return Some((None, self.own));
        }
        self.dependencies
            .prelude()
            .map(|(name, env)| (Some(name), env))
    }

    /// A name the prelude supplies to every module: a `pub` item of the
    /// prelude module.
    pub fn lookup_library(&self, name: &Symbol) -> Option<&'s Binding<'db>> {
        let (package, env) = self.prelude()?;
        let prelude = prelude_module();
        if package.is_some() && env.entry_restriction(prelude, name).is_some() {
            return None;
        }
        env.lookup_qualified(prelude, name)
    }

    /// The names the prelude supplies to every module.
    pub fn iter_library_names(&self) -> Vec<Symbol> {
        let Some((_, env)) = self.prelude() else {
            return Vec::new();
        };
        let mut names: Vec<Symbol> = env
            .iter_namespace(prelude_module())
            .map(|(name, _)| name)
            .filter(|name| self.lookup_library(name).is_some())
            .collect();
        names.sort();
        names
    }

    /// The path, from the root of the package that holds it, of what the
    /// prelude name `name` names: a re-export's target, or the prelude
    /// module's own item.
    fn library_target(
        &self,
        name: &Symbol,
    ) -> Option<(Option<&'s Symbol>, &'s PackageEnv<'db>, SymbolPath)> {
        self.lookup_library(name)?;
        let (package, env) = self.prelude()?;
        let prelude = prelude_module();
        let target = env
            .export_target(prelude, name)
            .cloned()
            .unwrap_or_else(|| prelude.child(name.clone()));
        Some((package, env, target))
    }

    /// The package path of what the prelude name `name` names.
    pub fn library_path(&self, name: &Symbol) -> Option<Vec<Symbol>> {
        let (package, _, target) = self.library_target(name)?;
        Some(package.into_iter().chain(target.iter()).cloned().collect())
    }

    /// The namespace a path starting with the prelude name `alias`
    /// continues in: that of the item it names.
    pub fn library_namespace(&self, alias: &Symbol) -> Option<SymbolPath> {
        let (package, env, target) = self.library_target(alias)?;
        env.has_namespace(&target)
            .then(|| SymbolPath::new(package.into_iter().chain(target.iter()).cloned()))
    }

    /// Whether a path may start with `first` to name a dependency.
    pub fn is_library_root(&self, first: &Symbol) -> bool {
        self.dependencies.get(first).is_some()
    }

    /// Whether the package root sees `name` only because the prelude
    /// supplies it, not through a declaration or a `use`.
    pub fn is_library_only(&self, name: &Symbol) -> bool {
        self.own.lookup(name).is_none()
            && self.own.get_use_path(name).is_none()
            && self.lookup_library(name).is_some()
    }

    /// The names an unqualified reference at the package root may use.
    pub fn iter_all_names(&self) -> Vec<Symbol> {
        let mut names: Vec<Symbol> = self.own.iter_all_names().collect();
        names.extend(self.iter_library_names());
        names
    }

    pub fn declares(&self, name: Symbol) -> bool {
        self.own.declares(name)
    }

    pub fn has_definition(&self, name: &Symbol) -> bool {
        self.own.has_definition(name)
    }

    pub fn get_use_path(&self, name: &Symbol) -> Option<&'s SymbolPath> {
        self.own.get_use_path(name)
    }

    pub fn imported_functions(&self, name: &Symbol) -> Option<&'s [FuncDefId<'db>]> {
        self.own.imported_functions(name)
    }
}
