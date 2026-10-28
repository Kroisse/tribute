//! What a package's names see: its own environment and the environments
//! of the packages it depends on.

use trunk_ir::{Symbol, SymbolPath};

use std::sync::LazyLock;

use super::env::{Binding, PackageEnv, Restriction};
use super::package::{IdentityPath, Location, PackageName};
use super::{PRELUDE_MODULE, STD_PACKAGE};
use crate::ast::FuncDefId;

/// The path of the prelude module from the root of its package.
fn prelude_module() -> &'static SymbolPath {
    static PRELUDE: LazyLock<SymbolPath> = LazyLock::new(|| SymbolPath::new(PRELUDE_MODULE));
    &PRELUDE
}

/// The packages a package depends on, each by its name, and the packages
/// those depend on in turn. Only the named ones are found by name; the
/// others are read where a re-export names a location in them.
#[derive(Clone, Debug, Default)]
pub struct Dependencies<'db> {
    packages: Vec<(Symbol, &'db PackageEnv<'db>)>,
    indirect: Vec<&'db PackageEnv<'db>>,
}

impl<'db> Dependencies<'db> {
    /// No dependencies.
    pub fn none() -> Self {
        Self::default()
    }

    /// The dependencies named in a package's dependency table, each with
    /// its environment.
    pub fn new(packages: impl IntoIterator<Item = (Symbol, &'db PackageEnv<'db>)>) -> Self {
        Self {
            packages: packages.into_iter().collect(),
            indirect: Vec::new(),
        }
    }

    /// These dependencies with the environments of the packages they
    /// depend on, transitively, that the table does not name.
    pub fn with_indirect(
        mut self,
        indirect: impl IntoIterator<Item = &'db PackageEnv<'db>>,
    ) -> Self {
        self.indirect.extend(indirect);
        self
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
            .find(|(name, _)| name == STD_PACKAGE)
            .map(|(name, env)| (name, *env))
    }

    /// The names of the dependencies.
    pub fn names(&self) -> impl Iterator<Item = &Symbol> + '_ {
        self.packages.iter().map(|(name, _)| name)
    }

    /// The environments of the named and the indirect dependencies.
    fn iter(&self) -> impl Iterator<Item = &'db PackageEnv<'db>> + '_ {
        self.packages
            .iter()
            .map(|(_, env)| *env)
            .chain(self.indirect.iter().copied())
    }
}

/// A package's environment with the dependencies it was built against,
/// which resolution reads it with. Only [`build_env`](super::build_env)
/// pairs them, so they cannot disagree. [`ScopeRef`] is its borrowed view.
#[derive(Debug)]
pub struct Scope<'db> {
    env: PackageEnv<'db>,
    dependencies: Dependencies<'db>,
}

impl<'db> Scope<'db> {
    pub(super) fn new(env: PackageEnv<'db>, dependencies: Dependencies<'db>) -> Self {
        Self { env, dependencies }
    }

    /// The package's own environment, without its dependencies.
    pub fn into_env(self) -> PackageEnv<'db> {
        self.env
    }

    /// The borrowed view of this scope.
    pub(super) fn as_ref(&self) -> ScopeRef<'_, 'db> {
        ScopeRef::new(&self.env, &self.dependencies)
    }
}

/// A package's view of names: its own environment, then its dependencies'.
/// It borrows a [`Scope`], or an environment still being built with the
/// dependencies it is built against.
///
/// Its places are [`Location`]s: a path from the root of the package
/// itself or of one of its dependencies. Which package a written path
/// starts in is the resolver's decision, not the scope's.
#[derive(Clone, Copy)]
pub struct ScopeRef<'s, 'db> {
    own: &'s PackageEnv<'db>,
    dependencies: &'s Dependencies<'db>,
}

impl<'s, 'db> ScopeRef<'s, 'db> {
    pub fn new(own: &'s PackageEnv<'db>, dependencies: &'s Dependencies<'db>) -> Self {
        Self { own, dependencies }
    }

    /// The names of the packages the package depends on.
    pub fn dependency_names(&self) -> impl Iterator<Item = &'s Symbol> + use<'s, 'db> {
        self.dependencies.names()
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

    /// The bindings of the package and its dependencies that are not
    /// public, with where each is visible.
    pub fn restrictions(&self) -> impl Iterator<Item = (&'s Binding<'db>, &'s Restriction)> {
        self.own
            .restrictions()
            .chain(self.dependencies.iter().flat_map(|env| env.restrictions()))
    }

    /// The location the re-export of `name` in `namespace` names.
    pub fn export_target(&self, namespace: &Location, name: &Symbol) -> Option<&'s Location> {
        self.env(&namespace.package)?
            .export_target(&namespace.path, name)
    }

    /// Where the module at `module` is visible, if it is not public.
    pub fn module_restriction(&self, module: &Location) -> Option<&'s Restriction> {
        self.env(&module.package)?.module_restriction(&module.path)
    }

    /// The restrictions of the re-exported entry `name` of `namespace`,
    /// none if it is public. It is visible only where every one of them is.
    pub fn entry_restrictions(&self, namespace: &Location, name: &Symbol) -> &'s [Restriction] {
        self.env(&namespace.package)
            .map_or(&[], |env| env.entry_restrictions(&namespace.path, name))
    }

    /// The package that holds the prelude module, if it is a dependency,
    /// and its environment. Package `std` sees its own prelude module.
    fn prelude(&self) -> Option<(Option<&'s Symbol>, &'s PackageEnv<'db>)> {
        if self.own.package().is_std() {
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
        if package.is_some() && !env.entry_restrictions(prelude, name).is_empty() {
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
    fn library_target(&self, name: &Symbol) -> Option<Location> {
        self.lookup_library(name)?;
        let (_, env) = self.prelude()?;
        let prelude = prelude_module();
        Some(
            env.export_target(prelude, name)
                .cloned()
                .unwrap_or_else(|| env.package().at(prelude.child(name.clone()))),
        )
    }

    /// The identity path of what the prelude name `name` names.
    pub fn library_path(&self, name: &Symbol) -> Option<IdentityPath> {
        Some(self.library_target(name)?.identity())
    }

    /// The namespace a path starting with the prelude name `alias`
    /// continues in: that of the item it names, in the package that holds
    /// it.
    pub fn library_namespace(&self, alias: &Symbol) -> Option<Location> {
        let target = self.library_target(alias)?;
        self.has_namespace(&target).then_some(target)
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
