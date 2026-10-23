//! What a package's names see: its own environment and the environments
//! of the packages it depends on.

use trunk_ir::{Symbol, SymbolPath};

use super::LIBRARY_PACKAGE;
use super::env::{Binding, PackageEnv, Restriction};
use crate::ast::FuncDefId;

/// The packages a package depends on, each by its name.
#[derive(Clone, Debug, Default)]
pub struct Dependencies<'a, 'db> {
    packages: Vec<(Symbol, &'a PackageEnv<'db>)>,
}

impl<'a, 'db> Dependencies<'a, 'db> {
    /// No dependencies.
    pub fn none() -> Self {
        Self::default()
    }

    /// The dependency on package `std`, whose environment is `std`.
    pub fn std(std: &'a PackageEnv<'db>) -> Self {
        Self {
            packages: vec![(Symbol::new(LIBRARY_PACKAGE), std)],
        }
    }

    /// The environment of the dependency named `name`.
    pub fn get(&self, name: &Symbol) -> Option<&'a PackageEnv<'db>> {
        self.packages
            .iter()
            .find_map(|(known, env)| (known == name).then_some(*env))
    }

    /// The package whose root items every module sees under their short
    /// names: the prelude's.
    fn prelude(&self) -> Option<(&Symbol, &'a PackageEnv<'db>)> {
        self.packages
            .iter()
            .find(|(name, _)| name == LIBRARY_PACKAGE)
            .map(|(name, env)| (name, *env))
    }

    fn iter(&self) -> impl Iterator<Item = &'a PackageEnv<'db>> + '_ {
        self.packages.iter().map(|(_, env)| *env)
    }
}

/// A package's view of names: its own environment, then its dependencies'.
///
/// Its paths are read from the package's root. A path whose first segment
/// names a dependency, and names nothing the package itself declares or
/// imports, continues from that dependency's root.
#[derive(Clone, Copy)]
pub struct Scope<'s, 'a, 'db> {
    own: &'s PackageEnv<'db>,
    dependencies: &'s Dependencies<'a, 'db>,
}

impl<'s, 'a: 's, 'db> Scope<'s, 'a, 'db> {
    pub fn new(own: &'s PackageEnv<'db>, dependencies: &'s Dependencies<'a, 'db>) -> Self {
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

    /// Look up an unqualified name: the package's own, then a prelude name.
    pub fn lookup(&self, name: &Symbol) -> Option<&'s Binding<'db>> {
        self.own.lookup(name).or_else(|| self.lookup_library(name))
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

    /// The prelude's package and environment, for a package that depends
    /// on it; the package's own compiler-owned names otherwise.
    fn prelude(&self) -> Option<(Option<&Symbol>, &'s PackageEnv<'db>)> {
        if self.own.package().is_library() {
            return Some((None, self.own));
        }
        self.dependencies
            .prelude()
            .map(|(name, env)| (Some(name), env))
    }

    /// A name the prelude supplies to every module.
    pub fn lookup_library(&self, name: &Symbol) -> Option<&'s Binding<'db>> {
        match self.prelude()? {
            (None, own) => own.builtin(name),
            (Some(_), env) => env.prelude_item(name),
        }
    }

    /// The names the prelude supplies to every module.
    pub fn iter_library_names(&self) -> Vec<Symbol> {
        match self.prelude() {
            Some((None, own)) => own.builtin_names().collect(),
            Some((Some(_), env)) => env.prelude_items().map(|(name, _)| name).collect(),
            None => Vec::new(),
        }
    }

    /// The identity path of the prelude name `name`.
    pub fn library_path(&self, name: &Symbol) -> Option<Vec<Symbol>> {
        self.lookup_library(name)?;
        let (package, env) = self.prelude()?;
        Some(match env.builtin_path(name) {
            Some(path) => path.to_vec(),
            None => package.into_iter().chain([name]).cloned().collect(),
        })
    }

    /// The namespace a path starting with the prelude name `alias`
    /// continues in.
    pub fn library_namespace(&self, alias: &Symbol) -> Option<SymbolPath> {
        let (package, env) = self.prelude()?;
        let under = |path: &[Symbol]| {
            SymbolPath::new(package.into_iter().cloned().chain(path.iter().cloned()))
        };
        if let Some(namespace) = env.builtin_namespace(alias) {
            return Some(under(namespace));
        }
        let root = std::slice::from_ref(alias);
        (package.is_some() && env.has_namespace(root)).then(|| under(root))
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

    pub fn get_use_path(&self, name: &Symbol) -> Option<&'s Vec<Symbol>> {
        self.own.get_use_path(name)
    }

    pub fn imported_functions(&self, name: &Symbol) -> Option<&'s [FuncDefId<'db>]> {
        self.own.imported_functions(name)
    }
}
