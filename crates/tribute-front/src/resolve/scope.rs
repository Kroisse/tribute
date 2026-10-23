//! What a package's names see: its own environment and the environments
//! of the packages it depends on.

use trunk_ir::{Symbol, SymbolPath};

use super::LIBRARY_PACKAGE;
use super::env::{Binding, PackageEnv, Restriction};
use super::package::{IdentityPath, Location, PackageName};
use crate::ast::FuncDefId;

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

    /// The package whose root items every module sees under their short
    /// names: the prelude's.
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
/// Its places are [`Location`]s: a path from the root of the package
/// itself or of one of its dependencies. Which package a written path
/// starts in is the resolver's decision, not the scope's.
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

    /// The location `path` names from the package's own root.
    pub fn at_root(&self, path: SymbolPath) -> Location {
        self.own.package().at(path)
    }

    /// The environment of `package`: the package's own or a dependency's.
    fn env(&self, package: &PackageName) -> Option<&'s PackageEnv<'db>> {
        if package == self.own.package() {
            return Some(self.own);
        }
        self.dependencies
            .iter()
            .find(|env| env.package() == package)
    }

    /// The root of the dependency named `name`.
    pub fn dependency_root(&self, name: &Symbol) -> Option<Location> {
        let env = self.dependencies.get(name)?;
        Some(env.package().at(SymbolPath::root()))
    }

    /// The location of the identity path `identity`, in the package or a
    /// dependency.
    pub fn locate(&self, identity: &[Symbol]) -> Option<Location> {
        std::iter::once(self.own)
            .chain(self.dependencies.iter())
            .find_map(|env| {
                let path = env.package().module_path(identity)?;
                Some(env.package().at(SymbolPath::new(path.iter().cloned())))
            })
    }

    /// Look up a name the package root declares or imports. The prelude's
    /// names are not the root's; [`Self::lookup_library`] finds them.
    pub fn lookup(&self, name: &Symbol) -> Option<&'s Binding<'db>> {
        self.own.lookup(name)
    }

    /// Look up `name` in `namespace`. Another package's root shows the
    /// items it gives other packages.
    pub fn lookup_qualified(
        &self,
        namespace: &Location,
        name: &Symbol,
    ) -> Option<&'s Binding<'db>> {
        let env = self.env(&namespace.package)?;
        if namespace.path.is_root() && env.package() != self.own.package() {
            return env.root_item(name);
        }
        env.lookup_qualified(&namespace.path, name)
    }

    /// The binding at `location`: an item the package root declares or
    /// imports, or an entry of a namespace.
    pub fn lookup_path(&self, location: &Location) -> Option<&'s Binding<'db>> {
        let (last, namespace) = location.split_last()?;
        if namespace.path.is_root() && namespace.package == *self.own.package() {
            return self.lookup(last);
        }
        self.lookup_qualified(&namespace, last)
    }

    /// Whether `namespace` exists. Every package root does.
    pub fn has_namespace(&self, namespace: &Location) -> bool {
        self.env(&namespace.package).is_some_and(|env| {
            (namespace.path.is_root() && env.package() != self.own.package())
                || env.has_namespace(&namespace.path)
        })
    }

    /// The bindings of `namespace`.
    pub fn collect_namespace(&self, namespace: &Location) -> Vec<(Symbol, Binding<'db>)> {
        self.iter_namespace(namespace)
            .map(|(name, binding)| (name, binding.clone()))
            .collect()
    }

    /// The names and bindings of `namespace`.
    pub fn iter_namespace(
        &self,
        namespace: &Location,
    ) -> impl Iterator<Item = (Symbol, &'s Binding<'db>)> + use<'s, 'db> {
        let entries: Vec<(Symbol, &'s Binding<'db>)> = match self.env(&namespace.package) {
            Some(env) if namespace.path.is_root() && env.package() != self.own.package() => {
                env.root_items().collect()
            }
            Some(env) => env.iter_namespace(&namespace.path).collect(),
            None => Vec::new(),
        };
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
    pub fn module_restriction(&self, module: &Location) -> Option<&'s Restriction> {
        self.env(&module.package)?.module_restriction(&module.path)
    }

    /// Where the re-exported entry `name` of `namespace` is visible, if it
    /// is not public.
    pub fn entry_restriction(
        &self,
        namespace: &Location,
        name: &Symbol,
    ) -> Option<&'s Restriction> {
        self.env(&namespace.package)?
            .entry_restriction(&namespace.path, name)
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
    pub fn library_path(&self, name: &Symbol) -> Option<IdentityPath> {
        self.lookup_library(name)?;
        let (package, env) = self.prelude()?;
        Some(IdentityPath::new(match env.builtin_path(name) {
            Some(path) => SymbolPath::new(path.iter().cloned()),
            None => SymbolPath::new(package.into_iter().chain([name]).cloned()),
        }))
    }

    /// The namespace a path starting with the prelude name `alias`
    /// continues in, in the package that supplies the prelude.
    pub fn library_namespace(&self, alias: &Symbol) -> Option<Location> {
        let (package, env) = self.prelude()?;
        if let Some(namespace) = env.builtin_namespace(alias) {
            return Some(env.package().at(namespace.clone()));
        }
        let root = std::slice::from_ref(alias);
        (package.is_some() && env.has_namespace(root))
            .then(|| env.package().at(SymbolPath::new(root.iter().cloned())))
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

    pub fn get_use_path(&self, name: &Symbol) -> Option<&'s Location> {
        self.own.get_use_path(name)
    }

    pub fn imported_functions(&self, name: &Symbol) -> Option<&'s [FuncDefId<'db>]> {
        self.own.imported_functions(name)
    }
}
