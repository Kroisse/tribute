//! The name-resolution environment of a package.
//!
//! This module provides structures for tracking definitions and looking up names
//! during the name resolution phase.

use rustc_hash::FxHashMap as HashMap;

use trunk_ir::{Symbol, SymbolPath};

use super::package::PackageName;
use crate::ast::{AbilityId, CtorId, FuncDefId, OpDeclKind, TypeDefId, Visibility};

/// Information about a resolved name binding.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub enum Binding<'db> {
    /// A function defined in this module or imported.
    Function { id: FuncDefId<'db> },

    /// A type constructor (struct or enum variant).
    Constructor {
        /// The constructor ID.
        id: CtorId<'db>,
        /// For enum variants, the variant tag. For structs, None.
        tag: Option<Symbol>,
        /// Number of constructor arguments.
        arity: usize,
    },

    /// A type definition (struct or enum).
    TypeDef {
        /// The type definition ID.
        id: TypeDefId<'db>,
    },

    /// A module or namespace binding.
    Module {
        /// The namespace's identity path.
        path: Vec<Symbol>,
    },

    /// A `use` whose target is not resolved yet: the path it imports, as
    /// written or expanded from its path keywords.
    Import {
        /// The imported path.
        path: Vec<Symbol>,
    },

    /// An ability operation.
    ///
    /// The `kind` field (`OpDeclKind`) determines the lowering path:
    /// - `OpDeclKind::Fn` (tail-resumptive): lowered to `ability.call` with
    ///   direct evidence-based dispatch (no CPS, no continuation).
    /// - `OpDeclKind::Op` (general): lowered to `ability.perform` with CPS
    ///   continuations and runtime evidence lookup.
    AbilityOp {
        /// The ability identifier (e.g., AbilityId for "State").
        ability: AbilityId<'db>,
        /// The operation name (e.g., "get").
        op: Symbol,
        /// Whether this is a `fn` (tail-resumptive) or `op` (general) operation.
        kind: OpDeclKind,
    },

    /// An ability definition (not an operation).
    ///
    /// Used when resolving ability names in handler patterns.
    Ability {
        /// The ability identifier.
        id: AbilityId<'db>,
    },
}

/// Where a binding or module that is not public is visible.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub struct Restriction {
    /// The package that declares it. Another package never sees it.
    pub package: PackageName,
    /// The module of that package whose descendants, itself included, see
    /// it.
    pub within: SymbolPath,
    /// The declaration whose visibility restricts it: its own, or that of a
    /// module around it.
    pub declaration: SymbolPath,
    /// That declaration's visibility.
    pub visibility: Visibility,
}

/// Names the declaration and its visibility, as in "`m::f` is private".
impl std::fmt::Display for Restriction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "`{}` is ",
            itertools::Itertools::format(self.declaration.iter(), "::")
        )?;
        match &self.visibility {
            Visibility::Private => f.write_str("private"),
            visibility => write!(f, "declared `{visibility}`"),
        }
    }
}

impl Restriction {
    /// Whether `module`, a module of `package`, sees it.
    pub fn is_visible_from(&self, package: &PackageName, module: &[Symbol]) -> bool {
        self.package == *package && module.starts_with(&self.within)
    }

    /// The narrower of `self` and `outer`, the restriction of a module
    /// around it. Both are visible within modules around the same
    /// declaration, so the longer path is the narrower one.
    pub fn within_module(self, outer: Option<&Restriction>) -> Restriction {
        match outer {
            Some(outer) if outer.within.len() > self.within.len() => outer.clone(),
            _ => self,
        }
    }
}

/// A `use` that re-exports a name from an inline module, to be added to
/// that module's namespace once its target resolves.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub struct Export {
    /// The package-root path of the module that declares the `use`.
    pub module: SymbolPath,
    /// The name it exports.
    pub name: Symbol,
    /// The path it imports, as written.
    pub path: Vec<Symbol>,
    /// Where the exported name is visible, if not everywhere.
    pub restriction: Option<Restriction>,
}

/// The name-resolution environment of one package's module tree.
///
/// Holds the root module's names and every module's namespace, read from
/// the package root.
#[derive(Clone, Debug, Default, PartialEq, Eq, salsa::SalsaValue)]
pub struct PackageEnv<'db> {
    /// The package whose module tree this environment holds.
    package: PackageName,
    /// Names defined in this module (simple name → binding).
    definitions: HashMap<Symbol, Binding<'db>>,
    /// Names imported via `use` statements (simple name → binding).
    imports: HashMap<Symbol, Binding<'db>>,
    /// Qualified paths (namespace → name → binding).
    namespaces: HashMap<SymbolPath, HashMap<Symbol, Binding<'db>>>,
    /// Original paths for resolved `use` imports (import name → original path).
    /// Used to rewrite ability annotations from imported names to qualified paths.
    use_paths: HashMap<Symbol, Vec<Symbol>>,
    /// The paths of the names that several `use`s import, in declaration
    /// order.
    repeated_uses: HashMap<Symbol, Vec<Vec<Symbol>>>,
    /// The functions a name imports when several `use`s give it different
    /// ones. A call selects one by the type of its first argument.
    imported_functions: HashMap<Symbol, Vec<FuncDefId<'db>>>,
    /// The compiler-owned names the package supplies to every module, each
    /// with its identity path, e.g. `List`.
    builtins: HashMap<Symbol, (Binding<'db>, Vec<Symbol>)>,
    /// The namespaces a path starting with a compiler-owned name continues
    /// in, e.g. `List` → `collections::List`.
    builtin_namespaces: HashMap<Symbol, SymbolPath>,
    /// The bindings that are not public, each with the modules that see it.
    restrictions: HashMap<Binding<'db>, Restriction>,
    /// The modules that are not public, by package-root path.
    module_restrictions: HashMap<SymbolPath, Restriction>,
    /// The re-exports that inline modules declare.
    exports: Vec<Export>,
    /// The `use`s of each inline module: the imported name and the path as
    /// written.
    module_uses: HashMap<(SymbolPath, Symbol), Vec<Symbol>>,
    /// The namespace entries re-exports add that are not public, by
    /// namespace and name. Such an entry is visible where its `use` makes
    /// it, not where its binding is.
    entry_restrictions: HashMap<(SymbolPath, Symbol), Restriction>,
}

impl<'db> PackageEnv<'db> {
    /// Create a new empty environment of the package being compiled.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new empty environment of `package`'s module tree.
    pub fn for_package(package: PackageName) -> Self {
        Self {
            package,
            ..Self::default()
        }
    }

    /// The package whose module tree this environment holds.
    pub fn package(&self) -> &PackageName {
        &self.package
    }

    /// Add a function definition.
    pub fn add_function(&mut self, name: Symbol, id: FuncDefId<'db>) {
        self.definitions.insert(name, Binding::Function { id });
    }

    /// Add a type constructor.
    pub fn add_constructor(
        &mut self,
        name: Symbol,
        id: CtorId<'db>,
        tag: Option<Symbol>,
        arity: usize,
    ) {
        self.definitions
            .insert(name, Binding::Constructor { id, tag, arity });
    }

    /// Add a type definition.
    pub fn add_type(&mut self, name: Symbol, id: TypeDefId<'db>) {
        self.definitions.insert(name, Binding::TypeDef { id });
    }

    /// Add an ability definition.
    pub fn add_ability(&mut self, name: Symbol, id: AbilityId<'db>) {
        self.definitions.insert(name, Binding::Ability { id });
    }

    /// Add a qualified name to a namespace.
    pub fn add_to_namespace(&mut self, namespace: SymbolPath, name: Symbol, binding: Binding<'db>) {
        self.namespaces
            .entry(namespace)
            .or_default()
            .insert(name, binding);
    }

    /// Add an import.
    pub fn add_import(&mut self, name: Symbol, binding: Binding<'db>) {
        self.imports.insert(name, binding);
    }

    /// Add the placeholder of a `use` that imports `path` as `name`.
    pub fn add_use(&mut self, name: Symbol, path: Vec<Symbol>) {
        if let Some(Binding::Import { path: earlier }) = self.imports.get(&name)
            && *earlier != path
        {
            let paths = self
                .repeated_uses
                .entry(name.clone())
                .or_insert_with(|| vec![earlier.clone()]);
            if !paths.contains(&path) {
                paths.push(path.clone());
            }
        }
        self.add_import(name, Binding::Import { path });
    }

    /// Take the paths of the names that several `use`s import, to put back
    /// with [`Self::set_repeated_uses`].
    pub fn take_repeated_uses(&mut self) -> HashMap<Symbol, Vec<Vec<Symbol>>> {
        std::mem::take(&mut self.repeated_uses)
    }

    pub fn set_repeated_uses(&mut self, repeated: HashMap<Symbol, Vec<Vec<Symbol>>>) {
        self.repeated_uses = repeated;
    }

    /// Record the functions that several `use`s import as `name`.
    pub fn set_imported_functions(&mut self, name: Symbol, functions: Vec<FuncDefId<'db>>) {
        self.imported_functions.insert(name, functions);
    }

    /// The functions `name` imports when it imports several and the module
    /// declares none of that name.
    pub fn imported_functions(&self, name: &Symbol) -> Option<&[FuncDefId<'db>]> {
        if self.definitions.contains_key(name) {
            return None;
        }
        self.imported_functions.get(name).map(Vec::as_slice)
    }

    /// Supply the compiler-owned `binding` to every module as `name`, with
    /// the identity path `path`; a path starting with `name` continues in
    /// `namespace`.
    pub fn add_builtin(
        &mut self,
        name: Symbol,
        binding: Binding<'db>,
        path: Vec<Symbol>,
        namespace: SymbolPath,
    ) {
        self.builtin_namespaces.insert(name.clone(), namespace);
        self.builtins.insert(name, (binding, path));
    }

    /// The compiler-owned binding the package supplies as `name`.
    pub fn builtin(&self, name: &Symbol) -> Option<&Binding<'db>> {
        self.builtins.get(name).map(|(binding, _)| binding)
    }

    /// The identity path of the compiler-owned name `name`.
    pub fn builtin_path(&self, name: &Symbol) -> Option<&[Symbol]> {
        self.builtins.get(name).map(|(_, path)| path.as_slice())
    }

    /// The namespace a path starting with the compiler-owned name `name`
    /// continues in.
    pub fn builtin_namespace(&self, name: &Symbol) -> Option<&SymbolPath> {
        self.builtin_namespaces.get(name)
    }

    /// The compiler-owned names the package supplies.
    pub fn builtin_names(&self) -> impl Iterator<Item = Symbol> + '_ {
        self.builtins.keys().cloned()
    }

    /// The item `name` of the package root, as another package sees it:
    /// a definition, what the root re-exports, or a module.
    pub fn root_item(&self, name: &Symbol) -> Option<&Binding<'db>> {
        self.definitions
            .get(name)
            .or_else(|| self.lookup_qualified(&[], name))
            .or_else(|| match self.imports.get(name) {
                Some(binding @ Binding::Module { path })
                    if *path == self.package.identity_path(std::slice::from_ref(name))[..] =>
                {
                    Some(binding)
                }
                _ => None,
            })
    }

    /// The items of the package root, as another package sees them.
    pub fn root_items(&self) -> impl Iterator<Item = (Symbol, &Binding<'db>)> {
        let mut names: Vec<Symbol> = self
            .definitions
            .keys()
            .chain(
                self.namespaces
                    .get(&SymbolPath::root())
                    .into_iter()
                    .flat_map(|n| n.keys()),
            )
            .chain(self.imports.keys())
            .cloned()
            .collect();
        names.sort();
        names.dedup();
        names
            .into_iter()
            .filter_map(|name| Some((name.clone(), self.root_item(&name)?)))
            .collect::<Vec<_>>()
            .into_iter()
    }

    /// The name `name` the package's prelude gives another package: a root
    /// item, except a re-export the package restricts, or a compiler-owned
    /// name.
    pub fn prelude_item(&self, name: &Symbol) -> Option<&Binding<'db>> {
        if self.entry_restriction(&[], name).is_some() {
            return self.builtin(name);
        }
        self.root_item(name).or_else(|| self.builtin(name))
    }

    /// The names the package's prelude gives another package.
    pub fn prelude_items(&self) -> impl Iterator<Item = (Symbol, &Binding<'db>)> {
        let mut names: Vec<Symbol> = self
            .root_items()
            .map(|(name, _)| name)
            .chain(self.builtin_names())
            .collect();
        names.sort();
        names.dedup();
        names
            .into_iter()
            .filter_map(|name| Some((name.clone(), self.prelude_item(&name)?)))
            .collect::<Vec<_>>()
            .into_iter()
    }

    /// Restrict where `binding` is visible.
    pub fn restrict(&mut self, binding: Binding<'db>, restriction: Restriction) {
        self.restrictions.insert(binding, restriction);
    }

    /// Restrict where the module at the package-root path `module` is
    /// visible.
    pub fn restrict_module(&mut self, module: SymbolPath, restriction: Restriction) {
        self.module_restrictions.insert(module, restriction);
    }

    /// Where `binding` is visible, if it is not public.
    pub fn restriction(&self, binding: &Binding<'db>) -> Option<&Restriction> {
        self.restrictions.get(binding)
    }

    /// Where the module at the package-root path `module` is visible, if it
    /// is not public.
    pub fn module_restriction(&self, module: &[Symbol]) -> Option<&Restriction> {
        self.module_restrictions.get(module)
    }

    /// Record a re-export, to resolve with
    /// [`crate::resolve::resolve_exports`].
    pub fn add_export(&mut self, export: Export) {
        self.exports.push(export);
    }

    /// Record that the inline module `module` imports `path` as `name`.
    pub fn add_module_use(&mut self, module: SymbolPath, name: Symbol, path: Vec<Symbol>) {
        self.module_uses.entry((module, name)).or_insert(path);
    }

    /// The path, as written, the inline module `module` imports as `name`.
    pub fn module_use(&self, module: &SymbolPath, name: &Symbol) -> Option<&[Symbol]> {
        self.module_uses
            .get(&(module.clone(), name.clone()))
            .map(Vec::as_slice)
    }

    /// How many `use`s the inline modules declare.
    pub fn module_use_count(&self) -> usize {
        self.module_uses.len()
    }

    /// Take the re-exports recorded so far.
    pub fn take_exports(&mut self) -> Vec<Export> {
        std::mem::take(&mut self.exports)
    }

    /// Restrict where the entry `name` of `namespace` is visible.
    pub fn restrict_entry(
        &mut self,
        namespace: SymbolPath,
        name: Symbol,
        restriction: Restriction,
    ) {
        self.entry_restrictions
            .insert((namespace, name), restriction);
    }

    /// Where the entry `name` of `namespace` is visible, if a re-export
    /// added it and it is not public.
    pub fn entry_restriction(&self, namespace: &[Symbol], name: &Symbol) -> Option<&Restriction> {
        self.entry_restrictions
            .get(&(SymbolPath::new(namespace.iter().cloned()), name.clone()))
    }

    /// Take the restrictions of `other`, an environment of the same
    /// package.
    pub fn absorb_restrictions(&mut self, other: &PackageEnv<'db>) {
        self.restrictions.extend(
            other
                .restrictions
                .iter()
                .map(|(binding, restriction)| (binding.clone(), restriction.clone())),
        );
        self.module_restrictions.extend(
            other
                .module_restrictions
                .iter()
                .map(|(path, restriction)| (path.clone(), restriction.clone())),
        );
        self.exports.extend(other.exports.iter().cloned());
        self.module_uses.extend(
            other
                .module_uses
                .iter()
                .map(|(key, path)| (key.clone(), path.clone())),
        );
    }

    /// Add a qualified name to a namespace only if it doesn't already exist.
    pub fn add_to_namespace_if_absent(
        &mut self,
        namespace: SymbolPath,
        name: Symbol,
        binding: Binding<'db>,
    ) {
        self.namespaces
            .entry(namespace)
            .or_default()
            .entry(name)
            .or_insert(binding);
    }

    /// Look up an unqualified name.
    pub fn lookup(&self, name: &Symbol) -> Option<&Binding<'db>> {
        // First check local definitions
        if let Some(b) = self.definitions.get(name) {
            return Some(b);
        }
        // Then check imports
        self.imports.get(name)
    }

    /// Check whether a name is defined locally, excluding imports.
    pub fn has_definition(&self, name: &Symbol) -> bool {
        self.definitions.contains_key(name)
    }

    /// Look up a qualified path (e.g., "List::map").
    pub fn lookup_qualified(&self, namespace: &[Symbol], name: &Symbol) -> Option<&Binding<'db>> {
        self.namespaces.get(namespace)?.get(name)
    }

    /// Iterate over all bindings in a namespace.
    pub fn iter_namespace(
        &self,
        namespace: &[Symbol],
    ) -> impl Iterator<Item = (Symbol, &Binding<'db>)> {
        self.namespaces
            .get(namespace)
            .into_iter()
            .flat_map(|bindings| bindings.iter().map(|(k, v)| (k.clone(), v)))
    }

    /// Collect all bindings in a namespace (owned, avoids borrow issues).
    pub fn collect_namespace(&self, namespace: &[Symbol]) -> Vec<(Symbol, Binding<'db>)> {
        self.namespaces
            .get(namespace)
            .map(|bindings| {
                bindings
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Check whether a namespace exists.
    pub fn has_namespace(&self, namespace: &[Symbol]) -> bool {
        self.namespaces.contains_key(namespace)
    }

    /// Check whether the package declares `name` at its root: an item, a
    /// namespace, or a module, which may define no namespace entries.
    pub fn declares(&self, name: Symbol) -> bool {
        self.has_definition(&name)
            || self.has_namespace(std::slice::from_ref(&name))
            || match self.imports.get(&name) {
                Some(Binding::Module { path }) => {
                    *path == self.package.identity_path(std::slice::from_ref(&name))[..]
                }
                Some(Binding::Import { path }) => path[..] == [name],
                _ => false,
            }
    }

    /// Iterate over all imports.
    pub fn iter_imports(&self) -> impl Iterator<Item = (Symbol, &Binding<'db>)> {
        self.imports.iter().map(|(k, v)| (k.clone(), v))
    }

    /// Replace an import binding, recording the original path.
    pub fn replace_import(&mut self, name: Symbol, binding: Binding<'db>, path: Vec<Symbol>) {
        self.use_paths.insert(name.clone(), path);
        self.imports.insert(name, binding);
    }

    /// Get the original `use` path for an import name.
    pub fn get_use_path(&self, name: &Symbol) -> Option<&Vec<Symbol>> {
        self.use_paths.get(name)
    }

    /// Iterate over all definitions in this environment.
    pub fn iter_definitions(&self) -> impl Iterator<Item = (Symbol, &Binding<'db>)> {
        self.definitions.iter().map(|(k, v)| (k.clone(), v))
    }

    /// Iterate over all namespaces in this environment.
    pub fn iter_namespaces(
        &self,
    ) -> impl Iterator<Item = (SymbolPath, impl Iterator<Item = (Symbol, &Binding<'db>)>)> {
        self.namespaces
            .iter()
            .map(|(ns, bindings)| (ns.clone(), bindings.iter().map(|(k, v)| (k.clone(), v))))
    }

    /// Iterate over all names visible via unqualified lookup (definitions + imports).
    pub fn iter_all_names(&self) -> impl Iterator<Item = Symbol> + '_ {
        self.definitions
            .keys()
            .cloned()
            .chain(self.imports.keys().cloned())
    }
}
