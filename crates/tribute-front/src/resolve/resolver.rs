//! Name resolution logic.
//!
//! This module transforms `Expr<UnresolvedName>` into `Expr<ResolvedRef<'db>>`
//! by looking up names in the package environment and local scopes.

use rustc_hash::FxHashMap as HashMap;

use itertools::Itertools;
use salsa::Accumulator as _;
use tribute_core::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use trunk_ir::{Symbol, SymbolPath};

use crate::ast::{
    Arm, Candidates, CtorId, Decl, DeclEntry, DeclTable, Expr, ExprKind, FIELD_LENS_FUNCTIONS,
    FieldDecl, FieldInit, FieldPattern, FuncDecl, HandlerArm, HandlerKind, LocalId, LocalIdGen,
    Module, ModuleItem, ModulePath, NodeId, Param, ParamDecl, Pattern, PatternKind, ResolvedRef,
    SpanMap, Stmt, TypeAnnotation, TypeAnnotationKind, TypeKind, TypeParamDecl, UnresolvedName,
    UseDecl, Visibility,
};

use super::env::{Binding, PackageEnv, Restriction};
use super::package::{IdentityPath, Location, PackageName};
use super::path::{PathKeywordError, absolute_path, dependency_path};
use super::scope::{Dependencies, Scope};

/// Find the best matches from `candidates` by a caller-provided score function.
///
/// Returns up to `max` items sorted by descending score, keeping only scores >= `threshold`.
/// Uses a min-heap to maintain only the top entries without collecting all candidates.
fn best_matches_by<T>(
    candidates: impl IntoIterator<Item = T>,
    score: impl Fn(&T) -> f64,
    threshold: f64,
    max: usize,
) -> Vec<T> {
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    struct Entry<T> {
        item: T,
        score: f64,
    }

    impl<T> PartialEq for Entry<T> {
        fn eq(&self, other: &Self) -> bool {
            self.score == other.score
        }
    }

    impl<T> Eq for Entry<T> {}

    impl<T> Ord for Entry<T> {
        fn cmp(&self, other: &Self) -> Ordering {
            // Reverse: lower score = greater → pops first from max-heap
            other
                .score
                .partial_cmp(&self.score)
                .unwrap_or(Ordering::Equal)
        }
    }

    impl<T> PartialOrd for Entry<T> {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    let mut heap: BinaryHeap<Entry<T>> = BinaryHeap::with_capacity(max + 1);

    for item in candidates {
        let s = score(&item);
        if s < threshold {
            continue;
        }
        if heap.len() == max && heap.peek().is_some_and(|top| s <= top.score) {
            continue;
        }
        heap.push(Entry { item, score: s });
        if heap.len() > max {
            heap.pop();
        }
    }

    heap.into_sorted_vec().into_iter().map(|e| e.item).collect()
}

/// What the path of a qualified method call names.
enum MethodCandidates<'db> {
    /// Functions by rank, one of which the call's types select.
    Functions(Candidates<ResolvedRef<'db>>),
    /// The one callee the path names, which is not a function.
    Callee(ResolvedRef<'db>),
    /// Nothing; the path was reported.
    Unresolved,
}

/// Resolver for transforming unresolved names to resolved references.
pub struct Resolver<'db> {
    db: &'db dyn salsa::Database,
    /// The environment of the package being resolved.
    env: PackageEnv<'db>,
    /// The packages it depends on.
    dependencies: Dependencies<'db>,
    /// Span map for emitting diagnostics with source locations.
    span_map: &'db SpanMap,
    /// Lexical nested-module namespace used for unqualified references in an
    /// inline module body.
    module_path: Vec<Symbol>,
    /// For each inline module entered, its imports: the imported name and the
    /// location it names. An import is visible only in the body of
    /// the module that declares it.
    module_imports: Vec<HashMap<Symbol, Location>>,
    /// The functions a name imports in each enclosing inline module when
    /// several of the module's `use`s give it different ones.
    module_imported_functions: Vec<HashMap<Symbol, Vec<crate::ast::FuncDefId<'db>>>>,
    /// Source functions that redefine a struct field's function. They are
    /// reported and dropped, so the field's function is the definition.
    redefinitions: Vec<NodeId>,
    /// The package whose module tree `module_path` is a path of.
    package: PackageName,
}

/// Resolver for the body of one module-level function, in the scope its
/// module gives. Its local variables, and so their `LocalId`s, belong to that
/// function alone.
struct FunctionResolver<'r, 'db> {
    /// The resolver of the module the function is in.
    module: &'r Resolver<'db>,
    /// Stack of local scopes (function parameters, let bindings, etc.).
    local_scopes: Vec<HashMap<Symbol, LocalId>>,
    /// Generator for the function's LocalIds.
    local_id_gen: LocalIdGen,
    /// Stack of LocalIds for `resume` in `op` handler arms.
    resume_local_id_stack: Vec<LocalId>,
    /// Ability operations injected from ability annotations (effect-directed resolution).
    /// Maps unqualified operation name → Binding.
    effect_ops: HashMap<Symbol, Binding<'db>>,
    /// The functions each method name has named so far.
    method_functions: HashMap<Symbol, Candidates<crate::ast::FuncDefId<'db>>>,
}

impl<'db> Resolver<'db> {
    /// Create a new resolver with the given environment, which was built
    /// against `dependencies`.
    pub fn new(
        db: &'db dyn salsa::Database,
        env: PackageEnv<'db>,
        dependencies: &Dependencies<'db>,
        span_map: &'db SpanMap,
    ) -> Self {
        Self {
            db,
            env,
            dependencies: dependencies.clone(),
            span_map,
            module_path: Vec::new(),
            module_imports: Vec::new(),
            module_imported_functions: Vec::new(),
            redefinitions: Vec::new(),
            package: PackageName::compiled(),
        }
    }

    /// Resolve the module tree of `package`.
    pub fn with_package(mut self, package: PackageName) -> Self {
        self.package = package;
        self
    }

    /// The names the package and its dependencies give.
    fn view(&self) -> Scope<'_, 'db> {
        Scope::new(&self.env, &self.dependencies)
    }

    /// The reference `name` makes to `binding`, or an unresolved one if the
    /// current module does not see the binding.
    fn visible_ref(&self, binding: &Binding<'db>, name: &UnresolvedName) -> ResolvedRef<'db> {
        if let Some(restriction) = self.hidden(binding) {
            self.report_hidden(name.id, name, restriction);
            return ResolvedRef::local(LocalId::UNRESOLVED, name.name());
        }
        self.binding_to_ref(binding, name.name())
    }

    /// The reference `name` makes to `binding`, the entry of `namespace` it
    /// names, or an unresolved one if the current module does not see that
    /// entry.
    fn visible_entry_ref(
        &self,
        namespace: &Location,
        binding: &Binding<'db>,
        name: &UnresolvedName,
    ) -> ResolvedRef<'db> {
        if let Some(restriction) = self.hidden_entry(namespace, &name.name(), binding) {
            self.report_hidden(name.id, name, restriction);
            return ResolvedRef::local(LocalId::UNRESOLVED, name.name());
        }
        self.binding_to_ref(binding, name.name())
    }

    /// The restriction of the entry `name` of `namespace`, which holds
    /// `binding`: a re-export's own, or else the binding's.
    fn entry_restrictions(
        &self,
        namespace: &Location,
        name: &Symbol,
        binding: &Binding<'db>,
    ) -> Vec<&Restriction> {
        let entry = self.view().entry_restrictions(namespace, name);
        if entry.is_empty() {
            self.view().restriction(binding).into_iter().collect()
        } else {
            entry.iter().collect()
        }
    }

    /// The restriction that hides the entry `name` of `namespace` from the
    /// current module.
    fn hidden_entry(
        &self,
        namespace: &Location,
        name: &Symbol,
        binding: &Binding<'db>,
    ) -> Option<&Restriction> {
        self.entry_restrictions(namespace, name, binding)
            .into_iter()
            .find(|restriction| !restriction.is_visible_from(&self.package, &self.module_path))
    }

    /// Whether the current module sees the re-export, if any, that the entry
    /// `name` of `namespace` is. A function it does not see is no method
    /// candidate; one it sees is checked by its binding.
    fn visible_through_entry(
        &self,
        namespace: &Location,
        name: &Symbol,
        binding: &Binding<'db>,
    ) -> bool {
        self.view().entry_restrictions(namespace, name).is_empty()
            || self.hidden_entry(namespace, name, binding).is_none()
    }

    /// The restriction that hides `binding` from the current module.
    fn hidden(&self, binding: &Binding<'db>) -> Option<&Restriction> {
        self.view()
            .restriction(binding)
            .filter(|restriction| !restriction.is_visible_from(&self.package, &self.module_path))
    }

    /// The restrictions of what `location` names, from the modules on its
    /// path and the binding at its end. It is visible only where every one
    /// of them is.
    fn path_restrictions(&self, location: &Location) -> Vec<&Restriction> {
        // A path that names an item does not pass through the companion
        // module of the item's name, whose visibility is its own.
        let item = self
            .lookup_path(location)
            .filter(|binding| !matches!(binding, Binding::Module { .. } | Binding::Import { .. }));
        let path = &location.path;
        let modules = if item.is_some() {
            path.len().saturating_sub(1)
        } else {
            path.len()
        };
        let own = item
            .and_then(|binding| {
                let (name, namespace) = location.split_last()?;
                Some(self.entry_restrictions(&namespace, name, binding))
            })
            .unwrap_or_default();
        (1..=modules)
            .filter_map(|end| {
                let module = location
                    .package
                    .at(SymbolPath::new(path[..end].iter().cloned()));
                self.view().module_restriction(&module)
            })
            .chain(own)
            .collect()
    }

    /// How narrow `restriction` is to the current package: another
    /// package's hides it however wide it is.
    fn narrowness(&self, restriction: &Restriction) -> usize {
        if restriction.package == self.package {
            restriction.within.len()
        } else {
            usize::MAX
        }
    }

    /// The narrowest restriction that hides what `location` names from the
    /// current module.
    fn hidden_path(&self, location: &Location) -> Option<&Restriction> {
        self.path_restrictions(location)
            .into_iter()
            .filter(|restriction| !restriction.is_visible_from(&self.package, &self.module_path))
            .max_by_key(|restriction| self.narrowness(restriction))
    }

    /// Report a reference, written as `written`, to what `restriction` hides
    /// from the current module.
    fn report_hidden(
        &self,
        node: NodeId,
        written: impl std::fmt::Display,
        restriction: &Restriction,
    ) {
        Diagnostic::new(
            format!("`{written}` is not visible here: {restriction}"),
            self.span_map.get_or_default(node),
            DiagnosticSeverity::Error,
            CompilationPhase::NameResolution,
        )
        .accumulate(self.db);
    }

    /// The binding a simple name has in the current scope, apart from local
    /// variables.
    fn lookup_simple(&self, sym: &Symbol) -> Option<&Binding<'db>> {
        self.lookup_direct(sym)
            .or_else(|| self.view().lookup_library(sym))
    }

    /// The binding a simple name has in the current scope through its
    /// declarations and `use`s, apart from local variables and the prelude.
    fn lookup_direct(&self, sym: &Symbol) -> Option<&Binding<'db>> {
        if !self.module_path.is_empty()
            && let Some(binding) = self.view().lookup_qualified(&self.here(), sym)
        {
            return Some(binding);
        }

        // Then the enclosing inline module's imports.
        if let Some(target) = self.module_import(sym)
            && let Some(binding) = self.lookup_path(target)
        {
            return Some(binding);
        }

        // An inline module sees its companion, not the items of the modules
        // around it; the package root sees its own items too.
        if self.module_path.is_empty() {
            self.env
                .lookup(sym)
                .filter(|_| !self.view().is_library_only(sym))
        } else {
            self.companion(sym)
        }
    }

    /// The namespace and the binding a qualified path names from the
    /// current scope.
    ///
    /// A path keyword names a package-root path, read from nowhere else. A
    /// first segment the enclosing module imports names only the imported
    /// path; it does not fall back to the package root.
    fn qualified_entry(
        &self,
        name: &UnresolvedName,
    ) -> Result<Option<(Location, &Binding<'db>)>, PathKeywordError> {
        let sym = name.name();
        let Some(namespace) = name.namespace() else {
            return Ok(None);
        };
        if let Some(namespace) = self.fixed_location(&namespace)? {
            let namespace = self.through_exports(namespace);
            let binding = self.view().lookup_path(&namespace.child(sym));
            return Ok(binding.map(|binding| (namespace, binding)));
        }
        Ok(self.namespace_in_scope(namespace).and_then(|namespace| {
            let namespace = self.through_exports(namespace);
            let binding = self.view().lookup_qualified(&namespace, &sym)?;
            Some((namespace, binding))
        }))
    }

    /// `namespace`, or, when it is not a namespace itself, where it continues
    /// through a re-export on its path: `m::library::Option`, after `pub use
    /// std as library` in `m`, is `Option` in package `std`. A re-export the
    /// current module does not see stops the path.
    fn through_exports(&self, namespace: Location) -> Location {
        if self.view().has_namespace(&namespace) {
            return namespace;
        }
        // Each segment continues from the one before it: to what a re-export
        // of that name names, whose own entries alone the exporting module
        // holds a copy of, or else into a namespace.
        let mut current = namespace.package.at(SymbolPath::root());
        for (index, segment) in namespace.path.iter().enumerate() {
            let next = current.child(segment.clone());
            current = if let Some(target) = self.export_chain(&current, segment) {
                target
            } else if self.view().has_namespace(&next) {
                next
            } else {
                return next.join(&namespace.path[index + 1..]);
            };
        }
        current
    }

    /// What the re-export of `name` in `namespace` names, followed through
    /// the re-exports it names in turn, if the current module sees each.
    fn export_chain(&self, namespace: &Location, name: &Symbol) -> Option<Location> {
        use rustc_hash::FxHashSet as HashSet;

        let visible = |outer: &Location, name: &Symbol| {
            let target = self.view().export_target(outer, name)?;
            let hidden = self
                .view()
                .entry_restrictions(outer, name)
                .iter()
                .any(|restriction| !restriction.is_visible_from(&self.package, &self.module_path));
            (!hidden).then(|| target.clone())
        };
        let mut target = visible(namespace, name)?;
        // A target that is itself a re-export continues to what that names.
        // Each step replaces the location with a recorded target, so the
        // targets seen so far end a cycle.
        let mut seen = HashSet::default();
        while seen.insert(target.clone()) {
            let Some(next) = target
                .split_last()
                .and_then(|(name, outer)| visible(&outer, name))
            else {
                break;
            };
            target = next;
        }
        Some(target)
    }

    /// The functions several `use`s of the enclosing module give `name`,
    /// whatever the local variables are.
    fn module_imported_functions(&self, name: &Symbol) -> Option<&[crate::ast::FuncDefId<'db>]> {
        if self.module_path.is_empty() {
            return self.view().imported_functions(name);
        }
        if self.defined_in_module(name) {
            return None;
        }
        self.module_imported_functions
            .last()?
            .get(name)
            .map(Vec::as_slice)
    }

    fn report_unresolved_method(&self, node: NodeId, method: &Symbol) {
        Diagnostic::new(
            format!(
                "unresolved method `{method}`: the scope names no function `{method}`, \
                 and no module or type in it has one"
            ),
            self.span_map.get_or_default(node),
            DiagnosticSeverity::Error,
            CompilationPhase::NameResolution,
        )
        .accumulate(self.db);
    }

    /// Whether the first segment of a qualified `path` is a prelude name the
    /// scope does not name through its declarations and `use`s.
    fn reaches_through_prelude(&self, path: &UnresolvedName) -> bool {
        let first = path.path.first().expect("a method path has a segment");
        self.view().library_namespace(first).is_some() && !self.module_names(first)
    }

    /// Whether the current module names `first` itself, through its
    /// declarations, modules, and `use`s. Such a name hides a prelude name
    /// and a dependency's name from the start of a path.
    fn module_names(&self, first: &Symbol) -> bool {
        if self.module_import(first).is_some() {
            return true;
        }
        if self.module_path.is_empty() {
            self.view().declares(first.clone())
                || self.view().get_use_path(first).is_some()
                || self.view().lookup(first).is_some()
        } else {
            self.defined_in_module(first)
                || self.view().has_namespace(&self.here().child(first.clone()))
        }
    }

    /// The one location `path` names wherever it is written, when it
    /// starts with a path keyword or a leading `::`; `None` for a path that
    /// starts with a name.
    fn fixed_location(&self, path: &[Symbol]) -> Result<Option<Location>, PathKeywordError> {
        if let Some((name, rest)) = dependency_path(path) {
            let root = self
                .view()
                .dependency_root(name)
                .ok_or_else(|| PathKeywordError::UnknownPackage(name.clone()))?;
            return Ok(Some(root.join(rest)));
        }
        Ok(absolute_path(&self.module_path, path)?
            .map(|path| self.package.at(SymbolPath::new(path))))
    }

    /// The location of the current module.
    fn here(&self) -> Location {
        self.package
            .at(SymbolPath::new(self.module_path.iter().cloned()))
    }

    /// Whether a qualified `path` starts with a path keyword or a dependency
    /// package the scope does not shadow, and so names one path only.
    fn names_one_path(&self, path: &UnresolvedName) -> bool {
        let first = path.path.first().expect("a method path has a segment");
        matches!(self.fixed_location(&path.path), Ok(Some(_)))
            || (self.view().dependency_root(first).is_some() && !self.module_names(first))
    }

    /// The functions `m::path` names for each name `m` in the scope.
    fn scope_namespace_functions(&self, path: &UnresolvedName) -> Vec<crate::ast::FuncDefId<'db>> {
        let mut functions = Vec::new();
        for module in self.names_in_scope() {
            let nested = UnresolvedName::new(
                SymbolPath::new(std::iter::once(&module).chain(path.path.iter()).cloned()),
                path.id,
            );
            if let Ok(Some((namespace, binding @ Binding::Function { id }))) =
                self.qualified_entry(&nested)
                && self.visible_through_entry(&namespace, &nested.name(), binding)
                && !functions.contains(id)
            {
                functions.push(*id);
            }
        }
        functions
    }

    /// The candidates of an implicit constructor `.Name`, which type
    /// checking selects from by the type its context gives.
    fn implicit_constructor(&self, name: &UnresolvedName) -> ResolvedRef<'db> {
        let ctor = name.name();
        let mut in_scope: Vec<_> = self
            .names_in_scope()
            .into_iter()
            .filter_map(|ty| {
                let binding = self.lookup_direct(&ty)?;
                let path =
                    UnresolvedName::new(SymbolPath::new([ty.clone(), ctor.clone()]), name.id);
                let variant = match self.qualified_entry(&path) {
                    Ok(Some((namespace, binding))) => self
                        .visible_through_entry(&namespace, &ctor, binding)
                        .then_some(binding),
                    _ => None,
                };
                self.type_constructor(&ty, binding, &ctor, variant)
            })
            .collect();
        let mut prelude: Vec<_> = self
            .view()
            .iter_library_names()
            .into_iter()
            .filter_map(|ty| {
                let binding = self.view().lookup_library(&ty)?;
                let variant = self.view().library_namespace(&ty).and_then(|namespace| {
                    self.view()
                        .lookup_qualified(&namespace, &ctor)
                        .filter(|binding| self.visible_through_entry(&namespace, &ctor, binding))
                });
                self.type_constructor(&ty, binding, &ctor, variant)
            })
            .collect();
        for rank in [&mut in_scope, &mut prelude] {
            rank.sort_by_key(|ctor: &CtorId<'db>| ctor.path(self.db));
            rank.dedup();
        }
        let mut hidden: Vec<_> = self
            .view()
            .restrictions()
            .filter_map(|(binding, restriction)| match binding {
                Binding::Constructor { id, .. }
                    if id.name(self.db) == ctor
                        && !restriction.is_visible_from(&self.package, &self.module_path) =>
                {
                    Some((*id, restriction.to_string()))
                }
                _ => None,
            })
            .collect();
        hidden.sort_by_key(|(id, _)| id.path(self.db));
        ResolvedRef::ImplicitConstructor {
            name: ctor,
            candidates: Candidates::new(Vec::new(), in_scope, prelude),
            hidden,
        }
    }

    /// The constructor `ctor` of the type a scope's name `ty` binds to
    /// `binding`: the struct itself when it has that name, or the variant
    /// `variant` that the namespace `ty::ctor` holds when it is the enum's.
    fn type_constructor(
        &self,
        ty: &Symbol,
        binding: &Binding<'db>,
        ctor: &Symbol,
        variant: Option<&Binding<'db>>,
    ) -> Option<CtorId<'db>> {
        match (binding, variant) {
            (Binding::Constructor { id, tag: None, .. }, _) if ty == ctor => {
                self.hidden(binding).is_none().then_some(*id)
            }
            (
                Binding::TypeDef { id },
                Some(Binding::Constructor {
                    id: variant,
                    tag: Some(_),
                    ..
                }),
            ) if variant.path(self.db).parent().as_ref() == Some(id.path(self.db)) => {
                Some(*variant)
            }
            _ => None,
        }
    }

    /// The functions a method path names through the prelude, whatever the
    /// scope's declarations and `use`s hide: the prelude's reading of the
    /// path itself, and `p::path` for each prelude name `p`.
    fn library_method_functions(&self, path: &[Symbol]) -> Vec<crate::ast::FuncDefId<'db>> {
        let mut functions = Vec::new();
        // A name the prelude supplies has no namespace.
        let mut push = |namespace: Option<&Location>, name: &Symbol| {
            if let Some(binding @ Binding::Function { id }) = match namespace {
                None => self.view().lookup_library(name),
                Some(namespace) => self.view().lookup_qualified(namespace, name),
            } && namespace
                .is_none_or(|namespace| self.visible_through_entry(namespace, name, binding))
                && !functions.contains(id)
            {
                functions.push(*id);
            }
        };
        let (name, namespace) = path.split_last().expect("a method path has a segment");
        match self.library_namespace_path(namespace) {
            Some(namespace) => push(Some(&namespace), name),
            None if namespace.is_empty() => push(None, name),
            None => {}
        }
        for prelude_name in self.view().iter_library_names() {
            if let Some(library) = self.view().library_namespace(&prelude_name) {
                push(Some(&library.join(namespace)), name);
            }
        }
        functions
    }

    /// The names an unqualified reference may use at this point through
    /// declarations and `use`s, apart from local variables and the prelude.
    fn names_in_scope(&self) -> Vec<Symbol> {
        if self.module_path.is_empty() {
            return self
                .env
                .iter_all_names()
                .filter(|name| !self.view().is_library_only(name))
                .collect();
        }
        let imports = self.module_imports.last();
        self.view()
            .iter_namespace(&self.here())
            .map(|(name, _)| name)
            .chain(
                imports
                    .into_iter()
                    .flat_map(|imports| imports.keys().cloned()),
            )
            .collect()
    }

    /// Whether the enclosing inline module itself defines `name`.
    fn defined_in_module(&self, name: &Symbol) -> bool {
        !self.module_path.is_empty() && self.view().lookup_qualified(&self.here(), name).is_some()
    }

    /// The location an import of the enclosing inline module gives `name`.
    fn module_import(&self, name: &Symbol) -> Option<&Location> {
        self.module_imports.last()?.get(name)
    }

    /// `namespace` with its first segment replaced by the location an
    /// import of the enclosing inline module gives it, e.g. `Abort` after
    /// `use outer::abilities::Abort` becomes `outer::abilities::Abort`.
    fn imported_namespace(&self, namespace: &SymbolPath) -> Option<Location> {
        let imports = self.module_imports.last()?;
        let (first, rest) = namespace.split_first()?;
        Some(imports.get(first)?.join(rest))
    }

    /// The location a qualified path's namespace names.
    ///
    /// The path starts from one of the current module's imports or from
    /// the module itself, else from a namespace the prelude supplies or a
    /// dependency. A first segment the module names keeps the path in the
    /// module even where the path names nothing there. At the package root
    /// a path that starts nowhere else stays in the package.
    fn namespace_in_scope(&self, namespace: SymbolPath) -> Option<Location> {
        if let Some(imported) = self.imported_namespace(&namespace) {
            return Some(imported);
        }
        let (first, rest) = namespace.split_first()?;
        if self.module_path.is_empty() {
            let own = self.package.at(namespace.clone());
            if self.view().has_namespace(&own) {
                return Some(own);
            }
            // A `use` gives the imported name's own namespace; a namespace
            // nested in it continues from the imported location.
            if !rest.is_empty()
                && let Some(location) = self.view().get_use_path(first)
            {
                return Some(location.join(rest));
            }
            if self.view().declares(first.clone()) {
                return Some(own);
            }
        } else {
            // A first segment the module names keeps the path in it.
            let nested = self.here().join(namespace.iter());
            if self.view().has_namespace(&nested) || self.module_names(first) {
                return Some(nested);
            }
        }
        if let Some(library) = self.view().library_namespace(first) {
            return Some(library.join(rest));
        }
        if let Some(root) = self.view().dependency_root(first) {
            return Some(root.join(rest));
        }
        self.module_path
            .is_empty()
            .then(|| self.package.at(namespace))
    }

    /// The location a path starting with a library namespace alias
    /// continues to, e.g. `Option::Some` → `Some` in package `std`'s
    /// `Option`.
    fn library_namespace_path(&self, path: &[Symbol]) -> Option<Location> {
        let (first, rest) = path.split_first()?;
        Some(self.view().library_namespace(first)?.join(rest))
    }

    /// The binding at `location`.
    fn lookup_path(&self, location: &Location) -> Option<&Binding<'db>> {
        self.view().lookup_path(location)
    }

    /// Report a path whose keywords do not denote a module.
    fn report_path_keyword(&self, node: NodeId, error: PathKeywordError) {
        Diagnostic::new(
            error.to_string(),
            self.span_map.get_or_default(node),
            DiagnosticSeverity::Error,
            CompilationPhase::NameResolution,
        )
        .accumulate(self.db);
    }

    /// Rewrite the names in `ann` to the identity paths of what they name.
    ///
    /// A path starting with a path keyword names the path it expands to.
    /// Inside an inline module, a type or ability name the module does not
    /// declare comes from its imports or the prelude, and a path starts from
    /// the module, an import, a prelude namespace, or a dependency.
    fn resolve_annotation_paths(&self, ann: &mut TypeAnnotation) {
        self.resolve_annotation_names(ann);
        self.check_identity_path(ann);
    }

    /// Check the identity path at the top of `ann`, if it has one: report
    /// it if the current module does not see what it names, and rewrite it
    /// to the declaration's identity.
    fn check_identity_path(&self, ann: &mut TypeAnnotation) {
        if let TypeAnnotationKind::Path(path) = &mut ann.kind
            && let Some(location) = self.view().locate(path)
        {
            if let Some(restriction) = self.hidden_path(&location) {
                self.report_hidden(ann.id, path.iter().format("::"), restriction);
                ann.kind = TypeAnnotationKind::Error;
            } else {
                // Later phases read an annotation by its identity.
                let declared = self.declared_path(&location);
                if declared[..] != path[..] {
                    *path = declared.to_vec();
                }
            }
        }
    }

    /// The identity path of what `location` names: the declaration's, which
    /// differs from the location's through a re-export.
    fn declared_path(&self, location: &Location) -> IdentityPath {
        let declared = match self.lookup_path(location) {
            Some(Binding::TypeDef { id }) => id.path(self.db),
            Some(Binding::Constructor { id, tag: None, .. }) => id.path(self.db),
            Some(Binding::Ability { id }) => id.path(self.db),
            _ => return location.identity(),
        };
        IdentityPath::new(declared.clone())
    }

    /// Rewrite the name at the top of `ann` to the identity path of what it
    /// names, and the names below it as [`Self::resolve_annotation_paths`]
    /// does.
    fn resolve_annotation_names(&self, ann: &mut TypeAnnotation) {
        match &mut ann.kind {
            TypeAnnotationKind::Named(name) if !self.module_path.is_empty() => {
                let local = name.with_str(|spelling| {
                    spelling.starts_with(|c: char| c.is_ascii_lowercase())
                        || TypeKind::from_primitive_name(spelling).is_some()
                });
                if local || self.defined_in_module(name) {
                    return;
                }
                if let Some(location) = self.companion_type(name) {
                    ann.kind = TypeAnnotationKind::Path(location.identity().to_vec());
                } else if let Some(target) = self.module_import(name) {
                    ann.kind = TypeAnnotationKind::Path(target.identity().to_vec());
                } else if let Some(path) = self.view().library_path(name) {
                    ann.kind = TypeAnnotationKind::Path(path.to_vec());
                } else {
                    self.report_unresolved_annotation(ann.id, name);
                    ann.kind = TypeAnnotationKind::Error;
                }
            }
            // At the package root a name the package does not declare or
            // import comes from the prelude.
            TypeAnnotationKind::Named(name) => {
                let local = name.with_str(|spelling| {
                    spelling.starts_with(|c: char| c.is_ascii_lowercase())
                        || TypeKind::from_primitive_name(spelling).is_some()
                });
                if !local
                    && !self.view().declares(name.clone())
                    && self.view().get_use_path(name).is_none()
                    && let Some(path) = self.view().library_path(name)
                {
                    ann.kind = TypeAnnotationKind::Path(path.to_vec());
                }
            }
            TypeAnnotationKind::Path(segments) => {
                // A path that starts with a keyword or `::` names one
                // location; any other path starts where the scope reads it.
                let location = match self.fixed_location(segments) {
                    Ok(Some(location)) => Some(location),
                    Ok(None) => self.namespace_in_scope(SymbolPath::new(segments.clone())),
                    Err(error) => {
                        self.report_path_keyword(ann.id, error);
                        ann.kind = TypeAnnotationKind::Error;
                        return;
                    }
                };
                // Its namespace continues through a re-export on it, as a
                // value path's does.
                let location = location.map(|location| match location.split_last() {
                    Some((name, namespace)) => self.through_exports(namespace).child(name.clone()),
                    None => location,
                });
                // The path names only what is at its location.
                match location
                    .as_ref()
                    .map(|location| (location, self.lookup_path(location)))
                {
                    Some((_, Some(Binding::Module { .. }))) => self.report_module_annotation(ann),
                    Some((location, Some(_))) => *segments = location.identity().to_vec(),
                    Some((location, None)) if location.path.is_empty() => {
                        self.report_module_annotation(ann)
                    }
                    _ => {
                        let path = Symbol::new(&segments.iter().format("::").to_string());
                        self.report_unresolved_annotation(ann.id, &path);
                        ann.kind = TypeAnnotationKind::Error;
                    }
                }
            }
            TypeAnnotationKind::App { ctor, args } => {
                self.resolve_annotation_paths(ctor);
                args.iter_mut()
                    .for_each(|arg| self.resolve_annotation_paths(arg));
            }
            TypeAnnotationKind::Func {
                params,
                result,
                abilities,
            } => {
                params
                    .iter_mut()
                    .for_each(|ann| self.resolve_annotation_paths(ann));
                self.resolve_effect_annotations(abilities);
                self.resolve_annotation_paths(result);
            }
            TypeAnnotationKind::Tuple(elements) => elements
                .iter_mut()
                .for_each(|element| self.resolve_annotation_paths(element)),
            TypeAnnotationKind::Infer | TypeAnnotationKind::Error => {}
        }
    }

    /// The location of the type or ability a module's own name names: the
    /// declaration beside the module that it accompanies, as `mod Option`
    /// accompanies `enum Option`.
    fn companion_type(&self, name: &Symbol) -> Option<Location> {
        let binding = self.companion(name)?;
        // A struct's name binds its constructor.
        matches!(
            binding,
            Binding::TypeDef { .. }
                | Binding::Ability { .. }
                | Binding::Constructor { tag: None, .. }
        )
        .then(|| {
            let parent = &self.module_path[..self.module_path.len() - 1];
            self.package
                .at(SymbolPath::new(parent.iter().chain([name]).cloned()))
        })
    }

    /// The declaration beside the current module that shares its name, which
    /// the module sees under that name.
    fn companion(&self, name: &Symbol) -> Option<&Binding<'db>> {
        let (own, parent) = self.module_path.split_last()?;
        if *own != name {
            return None;
        }
        let location = self
            .package
            .at(SymbolPath::new(parent.iter().chain([name]).cloned()));
        self.lookup_path(&location)
    }

    /// Report an annotation path that names a module rather than a type.
    fn report_module_annotation(&self, ann: &mut TypeAnnotation) {
        if let TypeAnnotationKind::Path(segments) = &ann.kind {
            Diagnostic::new(
                format!(
                    "`{}` names a module, not a type",
                    segments.iter().format("::")
                ),
                self.span_map.get_or_default(ann.id),
                DiagnosticSeverity::Error,
                CompilationPhase::NameResolution,
            )
            .accumulate(self.db);
        }
        ann.kind = TypeAnnotationKind::Error;
    }

    /// Report a type or ability name an annotation cannot see.
    fn report_unresolved_annotation(&self, node: NodeId, name: &Symbol) {
        Diagnostic::new(
            format!("unresolved name `{name}`"),
            self.span_map.get_or_default(node),
            DiagnosticSeverity::Error,
            CompilationPhase::NameResolution,
        )
        .accumulate(self.db);
    }

    fn resolve_param_paths(&self, params: &mut [ParamDecl]) {
        for param in params {
            if let Some(ty) = &mut param.ty {
                self.resolve_annotation_paths(ty);
            }
        }
    }

    fn resolve_field_paths(&self, fields: &mut [FieldDecl]) {
        for field in fields {
            self.resolve_annotation_paths(&mut field.ty);
        }
    }

    fn resolve_bound_paths(&self, type_params: &mut [TypeParamDecl]) {
        for param in type_params {
            param
                .bounds
                .iter_mut()
                .for_each(|bound| self.resolve_annotation_paths(bound));
        }
    }

    /// Convert a binding to a resolved reference.
    fn binding_to_ref(&self, binding: &Binding<'db>, name: Symbol) -> ResolvedRef<'db> {
        match binding {
            Binding::Function { id } => ResolvedRef::function(*id),
            Binding::Constructor { id, tag, .. } => {
                ResolvedRef::constructor(*id, tag.clone().unwrap_or(name))
            }
            Binding::TypeDef { id } => ResolvedRef::type_def(*id),
            Binding::Module { path } => {
                let path_ref = ModulePath::new(self.db, path.to_vec());
                ResolvedRef::Module { path: path_ref }
            }
            // A `use` of one segment names the package root's own module of
            // that name. Any longer `use` still unresolved names nothing, and
            // its declaration is reported.
            Binding::Import { path } if path.len() == 1 => {
                let identity = self.package.identity_path(path).to_vec();
                ResolvedRef::Module {
                    path: ModulePath::new(self.db, identity),
                }
            }
            Binding::Import { .. } => ResolvedRef::local(LocalId::UNRESOLVED, name),
            Binding::AbilityOp { ability, op, kind } => {
                ResolvedRef::ability_op(*ability, op.clone(), *kind)
            }
            Binding::Ability { id } => ResolvedRef::ability(*id),
        }
    }

    /// Resolve a module into the table of its declarations, each in the
    /// module its scope follows.
    pub fn resolve_module(
        &mut self,
        module: &Module<UnresolvedName>,
    ) -> DeclTable<ResolvedRef<'db>> {
        self.report_repeated_definitions(&module.decls);
        self.report_field_lens_redefinitions(&module.decls);
        let mut entries = Vec::new();
        self.resolve_items(&module.decls, &mut entries);
        DeclTable {
            id: module.id,
            name: module.name.clone(),
            entries,
        }
    }

    /// Resolve a declaration.
    fn resolve_decl(&mut self, decl: &Decl<UnresolvedName>) -> Decl<ResolvedRef<'db>> {
        match decl {
            Decl::Function(f) => Decl::Function(FunctionResolver::new(self).resolve_func_decl(f)),
            // These declarations contain no expressions to resolve.
            Decl::ExternFunction(e) => {
                let mut e = e.clone();
                self.resolve_param_paths(&mut e.params);
                self.resolve_annotation_paths(&mut e.return_ty);
                Decl::ExternFunction(e)
            }
            Decl::Struct(s) => {
                let mut s = s.clone();
                self.resolve_bound_paths(&mut s.type_params);
                self.resolve_field_paths(&mut s.fields);
                Decl::Struct(s)
            }
            Decl::Enum(e) => {
                let mut e = e.clone();
                self.resolve_bound_paths(&mut e.type_params);
                for variant in &mut e.variants {
                    self.resolve_field_paths(&mut variant.fields);
                }
                Decl::Enum(e)
            }
            Decl::Ability(a) => {
                let mut a = a.clone();
                self.resolve_bound_paths(&mut a.type_params);
                for op in &mut a.operations {
                    self.resolve_param_paths(&mut op.params);
                    self.resolve_annotation_paths(&mut op.return_ty);
                }
                Decl::Ability(a)
            }
            Decl::Use(u) => Decl::Use(self.resolve_use_decl(u)),
        }
    }

    /// Resolve the declarations of an inline module in its scope.
    fn resolve_module_decl(
        &mut self,
        module: &crate::ast::ModuleDecl<UnresolvedName>,
        entries: &mut Vec<DeclEntry<ResolvedRef<'db>>>,
    ) {
        self.report_repeated_definitions(module.body.as_deref().unwrap_or_default());
        self.report_field_lens_redefinitions(module.body.as_deref().unwrap_or_default());
        // For inline modules, recursively resolve nested declarations
        self.module_path.push(module.name.clone());
        // The module's imports are in scope throughout its body, including
        // before the `use` that declares them, and an import's path may start
        // from another import of the module.
        let uses: Vec<&UseDecl> = module
            .body
            .iter()
            .flatten()
            .filter_map(|item| match item {
                ModuleItem::Decl(Decl::Use(import)) => Some(import),
                _ => None,
            })
            .collect();
        self.module_imports.push(HashMap::default());
        loop {
            let resolved: Vec<(Symbol, Location)> = uses
                .iter()
                .filter_map(|import| {
                    let name = import
                        .alias
                        .clone()
                        .or_else(|| import.path.last().cloned())?;
                    let imports = self.module_imports.last()?;
                    if imports.contains_key(&name) {
                        return None;
                    }
                    Some((name, self.use_target(&import.path)?))
                })
                .collect();
            if resolved.is_empty() {
                break;
            }
            self.module_imports
                .last_mut()
                .expect("pushed above")
                .extend(resolved);
        }
        // A name that also imports something other than a function imports
        // no choice of functions.
        let mut functions: HashMap<Symbol, Option<Vec<crate::ast::FuncDefId<'db>>>> =
            HashMap::default();
        for import in &uses {
            let Some(name) = import.alias.clone().or_else(|| import.path.last().cloned()) else {
                continue;
            };
            let function =
                self.use_target(&import.path)
                    .and_then(|target| match self.lookup_path(&target) {
                        Some(Binding::Function { id }) => Some(*id),
                        _ => None,
                    });
            let imported = functions.entry(name).or_insert_with(|| Some(Vec::new()));
            match (imported.as_mut(), function) {
                (Some(imported), Some(function)) if !imported.contains(&function) => {
                    imported.push(function);
                }
                (Some(_), Some(_)) => {}
                _ => *imported = None,
            }
        }
        let functions = functions
            .into_iter()
            .filter_map(|(name, imported)| Some((name, imported.filter(|f| f.len() > 1)?)))
            .collect();
        self.module_imported_functions.push(functions);
        if let Some(items) = &module.body {
            self.resolve_items(items, entries);
        }
        self.module_imported_functions.pop();
        self.module_imports.pop();
        self.module_path.pop();
    }

    /// Report each declaration that repeats the name of an earlier one in its
    /// module, and record it as a redefinition. The earlier declaration is
    /// the definition.
    fn report_repeated_definitions(&mut self, items: &[ModuleItem<UnresolvedName>]) {
        for repeat in super::repeated_definitions(items) {
            Diagnostic::builder(
                format!("duplicate definition of `{}`", repeat.name),
                self.span_map.get_or_default(repeat.id),
                DiagnosticSeverity::Error,
                CompilationPhase::NameResolution,
            )
            .label(
                self.span_map.get_or_default(repeat.earlier),
                "first defined here",
            )
            .build()
            .accumulate(self.db);
            self.redefinitions.push(repeat.id);
        }
    }

    /// Report a companion module that declares a function the struct beside
    /// it already has, `T::f::set` or `T::f::modify` for a named field `f`,
    /// and record the function as a redefinition.
    fn report_field_lens_redefinitions(&mut self, items: &[ModuleItem<UnresolvedName>]) {
        let companions = items.iter().filter_map(|item| match item {
            ModuleItem::Module(module) => Some(module),
            ModuleItem::Decl(_) => None,
        });
        for companion in companions {
            let fields: Vec<&Symbol> = items
                .iter()
                .filter_map(|item| match item {
                    ModuleItem::Decl(Decl::Struct(declaration))
                        if declaration.name == companion.name
                            && !self.redefinitions.contains(&declaration.id) =>
                    {
                        Some(&declaration.fields)
                    }
                    _ => None,
                })
                .flatten()
                .filter_map(|field| field.name.as_ref())
                .collect();
            let lenses = companion
                .body
                .iter()
                .flatten()
                .filter_map(|item| match item {
                    ModuleItem::Module(field) if fields.contains(&&field.name) => Some(field),
                    _ => None,
                });
            for field in lenses {
                for item in field.body.iter().flatten() {
                    let (id, name) = match item {
                        ModuleItem::Decl(Decl::Function(function)) => (function.id, &function.name),
                        ModuleItem::Decl(Decl::ExternFunction(function)) => {
                            (function.id, &function.name)
                        }
                        _ => continue,
                    };
                    if !FIELD_LENS_FUNCTIONS.iter().any(|lens| name == lens) {
                        continue;
                    }
                    Diagnostic::new(
                        format!(
                            "duplicate definition of `{}::{}::{name}`: struct `{}` defines it for its field `{}`",
                            companion.name, field.name, companion.name, field.name
                        ),
                        self.span_map.get_or_default(id),
                        DiagnosticSeverity::Error,
                        CompilationPhase::NameResolution,
                    )
                    .accumulate(self.db);
                    self.redefinitions.push(id);
                }
            }
        }
    }

    /// Resolve the items of a module body into `entries`, without the
    /// declarations reported as redefinitions.
    fn resolve_items(
        &mut self,
        items: &[ModuleItem<UnresolvedName>],
        entries: &mut Vec<DeclEntry<ResolvedRef<'db>>>,
    ) {
        for item in items {
            if let ModuleItem::Decl(decl) = item
                && self.is_redefinition(decl)
            {
                continue;
            }
            self.report_visibility_path(item);
            match item {
                ModuleItem::Decl(decl) => {
                    let decl = self.resolve_decl(decl);
                    entries.push(DeclEntry {
                        module: self.package.identity_path(&self.module_path),
                        decl,
                    });
                }
                ModuleItem::Module(module) => self.resolve_module_decl(module, entries),
            }
        }
    }

    /// Whether `decl` was reported as a redefinition.
    fn is_redefinition(&self, decl: &Decl<UnresolvedName>) -> bool {
        let id = match decl {
            Decl::Function(f) => f.id,
            Decl::ExternFunction(f) => f.id,
            Decl::Struct(s) => s.id,
            Decl::Enum(e) => e.id,
            Decl::Ability(a) => a.id,
            Decl::Use(_) => return false,
        };
        self.redefinitions.contains(&id)
    }

    /// Report a `pub(path)` marker on `item` whose path names neither the
    /// current module nor a module around it.
    fn report_visibility_path(&self, item: &ModuleItem<UnresolvedName>) {
        let (id, visibility) = match item {
            ModuleItem::Decl(Decl::Function(f)) => (f.id, &f.visibility),
            ModuleItem::Decl(Decl::ExternFunction(f)) => (f.id, &f.visibility),
            ModuleItem::Decl(Decl::Struct(s)) => (s.id, &s.visibility),
            ModuleItem::Decl(Decl::Enum(e)) => (e.id, &e.visibility),
            ModuleItem::Decl(Decl::Ability(a)) => (a.id, &a.visibility),
            ModuleItem::Decl(Decl::Use(u)) => (u.id, &u.visibility),
            ModuleItem::Module(m) => (m.id, &m.visibility),
        };
        let Visibility::Restricted(scope) = visibility else {
            return;
        };
        let message = match absolute_path(&self.module_path, scope) {
            Err(error) => format!("`{visibility}`: {error}"),
            Ok(_) if self.restriction_module(scope).is_some() => return,
            Ok(_) => format!(
                "`{visibility}` names no module around this declaration: \
                 a restricted visibility names the declaring module or a module around it"
            ),
        };
        Diagnostic::new(
            message,
            self.span_map.get_or_default(id),
            DiagnosticSeverity::Error,
            CompilationPhase::NameResolution,
        )
        .accumulate(self.db);
    }

    /// The module the path of a `pub(scope)` marker in the current module
    /// names, if it is that module or a module around it.
    fn restriction_module(&self, scope: &[Symbol]) -> Option<SymbolPath> {
        super::restriction_module(&self.module_path, scope)
    }

    /// Extract the ability name (as a Symbol) from a type annotation.
    ///
    /// - `Named(sym)` → sym (e.g., `Abort`)
    /// - `Path(segs)` → qualified symbol (e.g., `abilities::Throw`)
    /// - `App { ctor, .. }` → recurse into ctor (e.g., `Throw` from `Throw(Nat)`)
    fn extract_ability_name(ann: &TypeAnnotation) -> Option<SymbolPath> {
        match &ann.kind {
            TypeAnnotationKind::Named(sym) => {
                // Row tail variables (lowercase, e.g., `e`) are not concrete abilities
                sym.with_str(|s| s.starts_with(|c: char| c.is_ascii_uppercase()))
                    .then(|| SymbolPath::new([sym.clone()]))
            }
            TypeAnnotationKind::Path(segs) => {
                (!segs.is_empty()).then(|| SymbolPath::new(segs.iter().cloned()))
            }
            TypeAnnotationKind::App { ctor, .. } => Self::extract_ability_name(ctor),
            _ => None,
        }
    }

    /// Resolve the names in effect annotations, and at the package root
    /// rewrite imported ability names to qualified paths.
    ///
    /// When an ability is imported via `use` (e.g., `use abilities::Abort`), the
    /// import stores the location it imports. At the package root this
    /// rewrites unqualified ability names in ability annotations to its
    /// identity so that `annotation_to_effect` creates the correct AbilityId.
    /// Inside an inline module, `resolve_annotation_paths` resolves imports.
    fn resolve_effect_annotations(&self, effects: &mut [TypeAnnotation]) {
        for ann in effects {
            self.resolve_annotation_paths(ann);
            if self.module_path.is_empty() {
                self.resolve_ability_in_annotation(ann);
            }
        }
    }

    fn resolve_ability_in_annotation(&self, ann: &mut TypeAnnotation) {
        match &mut ann.kind {
            TypeAnnotationKind::Named(sym)
                if sym.with_str(|s| s.starts_with(|c: char| c.is_ascii_uppercase()))
                    && !self.view().has_definition(sym) =>
            {
                if let Some(location) = self.view().get_use_path(sym)
                    && let path = location.identity()
                    && path.len() >= 2
                {
                    ann.kind = TypeAnnotationKind::Path(path.to_vec());
                    self.check_identity_path(ann);
                }
            }
            TypeAnnotationKind::App { ctor, .. } => self.resolve_ability_in_annotation(ctor),
            _ => {}
        }
    }

    /// Resolve a use declaration, recording the identity path of what it
    /// names.
    fn resolve_use_decl(&self, u: &UseDecl) -> UseDecl {
        UseDecl {
            target: self.resolve_use_target(u),
            ..u.clone()
        }
    }

    /// The identity path of what `u` names. A path that names nothing is
    /// reported, since it would otherwise leave the import as a module
    /// placeholder.
    fn resolve_use_target(&self, u: &UseDecl) -> Option<Vec<Symbol>> {
        let message = if let Err(error) = self.fixed_location(&u.path) {
            error.to_string()
        } else if let Some(first) = self.ambiguous_use_start(&u.path) {
            format!(
                "`use {}` is ambiguous: `{first}` names both a name of this module and \
                 package `{first}`; write `self::{first}` or `pkg::{first}` for the module's \
                 name, or `::{first}` for the package",
                u.path.iter().format("::")
            )
        } else if let Some(target) = self.use_target(&u.path) {
            self.check_use_visibility(u, &target);
            // Later phases read the target by its identity.
            return Some(self.declared_path(&target).to_vec());
        } else {
            format!("unresolved import `{}`", u.path.iter().format("::"))
        };
        Diagnostic::new(
            message,
            self.span_map.get_or_default(u.id),
            DiagnosticSeverity::Error,
            CompilationPhase::NameResolution,
        )
        .accumulate(self.db);
        None
    }

    /// The first segment of a `use` path that names both a name of the
    /// current module and a dependency, which makes the path ambiguous.
    fn ambiguous_use_start<'p>(&self, path: &'p [Symbol]) -> Option<&'p Symbol> {
        let (first, rest) = path.split_first()?;
        // A `use` of one segment imports that name itself.
        (!rest.is_empty()
            && matches!(self.fixed_location(path), Ok(None))
            && self.view().dependency_root(first).is_some()
            && self.module_names(first))
        .then_some(first)
    }

    /// Report a `use` of `target` that the current module does not see, or
    /// that re-exports it more widely than it is visible.
    fn check_use_visibility(&self, u: &UseDecl, target: &Location) {
        let written = u.path.iter().format("::");
        if let Some(restriction) = self.hidden_path(target) {
            self.report_hidden(u.id, written, restriction);
            return;
        }
        let restrictions = self.path_restrictions(target);
        if restrictions.is_empty() {
            return;
        }
        let exported = match &u.visibility {
            Visibility::Private => return,
            // An invalid path is reported with the declaration.
            Visibility::Restricted(scope) => match self.restriction_module(scope) {
                Some(module) => Some(module),
                None => return,
            },
            Visibility::Public => None,
        };
        // A `use` may not show what it names wider than any of its
        // restrictions.
        let wider = restrictions
            .into_iter()
            .find(|restriction| match &exported {
                Some(exported) => {
                    restriction.package != self.package || exported.len() < restriction.within.len()
                }
                None => true,
            });
        if let Some(restriction) = wider {
            Diagnostic::new(
                format!(
                    "`{written}` cannot be re-exported as `{}`: {restriction}",
                    u.visibility
                ),
                self.span_map.get_or_default(u.id),
                DiagnosticSeverity::Error,
                CompilationPhase::NameResolution,
            )
            .accumulate(self.db);
        }
    }

    /// The location of the definition or module `path` names.
    ///
    /// A path starting with a path keyword names exactly the path it expands
    /// to. Any other path starts from the current module, or else from a
    /// namespace the prelude supplies or a dependency, unless the module
    /// names its first segment itself.
    fn use_target(&self, path: &[Symbol]) -> Option<Location> {
        let names = |location: &Location| {
            // A package root names that package.
            let Some((last, namespace)) = location.split_last() else {
                return self.view().has_namespace(location);
            };
            // A single-segment path must name a definition: `env.lookup`
            // would also find the module placeholder this import inserted.
            let found = if namespace.path.is_root() && namespace.package == self.package {
                self.view().has_definition(last)
            } else {
                self.view().lookup_qualified(&namespace, last).is_some()
            };
            found || self.view().has_namespace(location)
        };
        match self.fixed_location(path) {
            Ok(Some(location)) => {
                return names(&location).then_some(location);
            }
            Err(_) => return None,
            Ok(None) => {}
        }
        if self.module_path.is_empty() {
            let location = super::root_location(self.view(), path)?;
            return names(&location).then_some(location);
        }
        // An inline module's path starts from one of its imports, the module
        // itself, or a namespace the prelude supplies.
        let (first, rest) = path.split_first()?;
        if let Some(target) = self.module_import(first) {
            let imported = target.join(rest);
            return names(&imported).then_some(imported);
        }
        let nested = self.here().join(path);
        if names(&nested) {
            return Some(nested);
        }
        if self.module_names(first) {
            return None;
        }
        if let Some(library) = self.library_namespace_path(path)
            && names(&library)
        {
            return Some(library);
        }
        let dependency = self.view().dependency_root(first)?.join(rest);
        names(&dependency).then_some(dependency)
    }
}

impl<'r, 'db> FunctionResolver<'r, 'db> {
    /// A resolver for one function in the module `module` is resolving.
    fn new(module: &'r Resolver<'db>) -> Self {
        Self {
            module,
            local_scopes: vec![HashMap::default()],
            local_id_gen: LocalIdGen::new(),
            resume_local_id_stack: Vec::new(),
            effect_ops: HashMap::default(),
            method_functions: HashMap::default(),
        }
    }

    /// Enter a new local scope.
    fn push_scope(&mut self) {
        self.local_scopes.push(HashMap::default());
    }

    /// Exit the current local scope.
    fn pop_scope(&mut self) {
        self.local_scopes.pop();
    }

    /// Bind a local variable in the current scope.
    fn bind_local(&mut self, name: Symbol) -> LocalId {
        let id = self.local_id_gen.fresh();
        if let Some(scope) = self.local_scopes.last_mut() {
            scope.insert(name, id);
        }
        id
    }

    /// Look up a local variable in all scopes.
    fn lookup_local(&self, name: &Symbol) -> Option<LocalId> {
        for scope in self.local_scopes.iter().rev() {
            if let Some(&id) = scope.get(name) {
                return Some(id);
            }
        }
        None
    }

    /// Resolve an unresolved name to a ResolvedRef.
    fn resolve_name(&self, name: &UnresolvedName) -> ResolvedRef<'db> {
        let sym = name.name();

        // For simple names: check locals, builtins, then the package environment
        if name.is_simple() {
            // First check local variables
            if let Some(local_id) = self.lookup_local(&sym) {
                return ResolvedRef::local(local_id, sym);
            }

            if let Some(binding) = self.module.lookup_simple(&sym) {
                return self.module.visible_ref(binding, name);
            }

            // Check effect-injected ability operations (effect-directed resolution)
            if let Some(binding) = self.effect_ops.get(&sym) {
                return self.module.visible_ref(binding, name);
            }
        } else {
            // Qualified path: e.g., State::get, Option::Some, abilities::Throw::throw
            match self.module.qualified_entry(name) {
                Ok(Some((namespace, binding))) => {
                    return self.module.visible_entry_ref(&namespace, binding, name);
                }
                Ok(None) => {}
                Err(error) => {
                    self.module.report_path_keyword(name.id, error);
                    return ResolvedRef::local(LocalId::UNRESOLVED, sym);
                }
            }
        }

        // Not found - emit diagnostic and return unresolved sentinel
        self.report_unresolved_name(name);
        ResolvedRef::local(LocalId::UNRESOLVED, sym)
    }

    /// The functions an unqualified call or reference `name` may be, by
    /// rank, when it may be more than one: the functions the scope names
    /// directly, then the prelude function of that name. A local variable of
    /// the name hides them all.
    fn unqualified_functions(&self, name: &UnresolvedName) -> Option<Candidates<ResolvedRef<'db>>> {
        let sym = name.name();
        if !name.is_simple() || self.lookup_local(&sym).is_some() {
            return None;
        }
        let direct = match self.module.module_imported_functions(&sym) {
            Some(imported) => imported.to_vec(),
            None => match self.module.lookup_direct(&sym) {
                Some(Binding::Function { id }) => vec![*id],
                Some(_) => return None,
                None => Vec::new(),
            },
        };
        let library = match self.module.view().lookup_library(&sym) {
            Some(binding @ Binding::Function { id }) if self.module.hidden(binding).is_none() => {
                vec![*id]
            }
            _ => Vec::new(),
        };
        let mut functions = Candidates::new(direct, Vec::new(), library);
        for rank in functions.ranks_mut() {
            rank.sort_by_key(|function| function.path(self.module.db));
        }
        (functions.len() > 1).then(|| functions.map(|&id| ResolvedRef::function(id)))
    }

    /// The functions the path of a qualified method call `x.a::b(y)` may
    /// name, ranked by how its first segment comes into scope: `a::b` as the
    /// call's scope resolves it, unless `a` is a prelude name, then
    /// `m::a::b` for each name `m` in that scope, then `a::b` and `p::a::b`
    /// through the prelude names `p`. A path that starts with a keyword or a
    /// dependency package names its one function only. A declaration's name
    /// counts as the module that shares it.
    fn method_candidates(&self, path: &UnresolvedName) -> MethodCandidates<'db> {
        let mut direct: Vec<crate::ast::FuncDefId<'db>> = Vec::new();
        match self.module.qualified_entry(path) {
            // A path through a prelude name is the prelude's reading of it.
            Ok(Some((_, Binding::Function { .. })))
                if self.module.reaches_through_prelude(path) => {}
            Ok(Some((namespace, binding @ Binding::Function { id })))
                if self
                    .module
                    .visible_through_entry(&namespace, &path.name(), binding) =>
            {
                direct.push(*id)
            }
            // A constructor or an ability operation takes the receiver as
            // its first argument without a choice by type.
            Ok(Some((namespace, binding))) => {
                return MethodCandidates::Callee(
                    self.module.visible_entry_ref(&namespace, binding, path),
                );
            }
            Ok(None) => {}
            Err(error) => {
                self.module.report_path_keyword(path.id, error);
                return MethodCandidates::Unresolved;
            }
        }
        let functions = if self.module.names_one_path(path) {
            Candidates::new(direct, Vec::new(), Vec::new())
        } else {
            Candidates::new(
                direct,
                self.module.scope_namespace_functions(path),
                self.module.library_method_functions(&path.path),
            )
        };
        let Some(functions) = self.visible_functions(functions, path.id, path) else {
            return MethodCandidates::Unresolved;
        };
        MethodCandidates::Functions(functions)
    }

    /// The functions of `functions` the current module sees, each rank
    /// sorted by path. When it sees none, report the method written as
    /// `written`: as hidden if it would see one otherwise, else as
    /// unresolved.
    fn visible_functions(
        &self,
        mut functions: Candidates<crate::ast::FuncDefId<'db>>,
        node: NodeId,
        written: &UnresolvedName,
    ) -> Option<Candidates<ResolvedRef<'db>>> {
        for rank in functions.ranks_mut() {
            rank.sort_by_key(|function| function.path(self.module.db));
        }
        let visible = functions.filter_map(|&id| {
            self.module
                .hidden(&Binding::Function { id })
                .is_none()
                .then(|| ResolvedRef::function(id))
        });
        if !visible.is_empty() {
            return Some(visible);
        }
        match functions.iter().next() {
            Some(&id) => {
                let binding = Binding::Function { id };
                let restriction = self
                    .module
                    .hidden(&binding)
                    .expect("every function is hidden");
                self.module.report_hidden(node, written, restriction);
            }
            None if written.is_simple() => {
                self.module.report_unresolved_method(node, &written.name())
            }
            None => self.report_unresolved_name(written),
        }
        None
    }

    /// The functions an unqualified method call `x.f(y)` may name, by rank:
    /// `f` as the call's scope names it directly, then `m::f` for each name
    /// `m` in that scope, then the prelude's `f` and `p::f` for each prelude
    /// name `p`. A declaration's name counts as the module that shares it.
    fn method_functions(
        &mut self,
        method: &Symbol,
        node: NodeId,
    ) -> Candidates<crate::ast::FuncDefId<'db>> {
        if let Some(functions) = self.method_functions.get(method) {
            return functions.clone();
        }
        // A method is a declared function; a local variable of its name does
        // not hide it.
        let direct = match self.module.module_imported_functions(method) {
            Some(imported) => imported.to_vec(),
            None => match self.module.lookup_direct(method) {
                Some(Binding::Function { id }) => vec![*id],
                _ => Vec::new(),
            },
        };
        let path = UnresolvedName::new(SymbolPath::new([method.clone()]), node);
        let functions = Candidates::new(
            direct,
            self.module.scope_namespace_functions(&path),
            self.module.library_method_functions(&path.path),
        );
        self.method_functions
            .insert(method.clone(), functions.clone());
        functions
    }

    /// The constructor a reference names: a path as [`resolve_name`]
    /// resolves it, or the candidates of an implicit constructor `.Name`.
    ///
    /// [`resolve_name`]: Self::resolve_name
    fn resolve_constructor(&self, name: &UnresolvedName) -> ResolvedRef<'db> {
        if name.implicit {
            self.module.implicit_constructor(name)
        } else {
            self.resolve_name(name)
        }
    }

    /// Report an unresolved name diagnostic, with "did you mean?" suggestions.
    fn report_unresolved_name(&self, name: &UnresolvedName) {
        let span = self.module.span_map.get_or_default(name.id);
        let similar = self.find_similar_names(&name.name());
        // A variant is declared only in its enum's namespace, so a bare
        // variant name is written as an implicit constructor.
        let implicit = name.is_simple()
            && matches!(
                self.module.implicit_constructor(&UnresolvedName::implicit(name.name(), name.id)),
                ResolvedRef::ImplicitConstructor { candidates, .. } if !candidates.is_empty()
            );

        let message = if implicit {
            format!("unresolved name `{name}`; did you mean `.{name}`?")
        } else if similar.is_empty() {
            format!("unresolved name `{}`", name)
        } else {
            let suggestions = similar
                .iter()
                .format_with(", ", |s, f| s.with_str(|name| f(&format_args!("`{name}`"))));
            format!("unresolved name `{}`; did you mean {}?", name, suggestions)
        };

        Diagnostic::new(
            message,
            span,
            DiagnosticSeverity::Error,
            CompilationPhase::NameResolution,
        )
        .accumulate(self.module.db);
    }

    /// Find names in scope that are similar to the given name.
    fn find_similar_names(&self, name: &Symbol) -> Vec<Symbol> {
        use rustc_hash::FxHashSet as HashSet;

        let candidates: HashSet<Symbol> = self
            .local_scopes
            .iter()
            .flat_map(|scope| scope.keys().cloned())
            .chain(self.module.view().iter_all_names())
            .collect();

        let target = name.to_string();
        best_matches_by(
            candidates,
            |sym: &Symbol| sym.with_str(|s| strsim::jaro_winkler(&target, s)),
            0.8,
            3,
        )
    }

    /// Resolve a function declaration.
    fn resolve_func_decl(&mut self, func: &FuncDecl<UnresolvedName>) -> FuncDecl<ResolvedRef<'db>> {
        // Enter a new scope for function body
        self.push_scope();

        // Bind parameters and assign local IDs
        let mut params: Vec<ParamDecl> = func
            .params
            .iter()
            .map(|p| {
                let mut p = p.clone();
                p.local_id = Some(self.bind_local(p.name.clone()));
                p
            })
            .collect();
        self.module.resolve_param_paths(&mut params);
        let mut type_params = func.type_params.clone();
        self.module.resolve_bound_paths(&mut type_params);
        let mut return_ty = func.return_ty.clone();
        if let Some(ty) = &mut return_ty {
            self.module.resolve_annotation_paths(ty);
        }

        // Resolve imported ability names in ability annotations to qualified paths.
        // e.g., after `use abilities::Abort`, rewrite `{Abort}` → `{abilities::Abort}`
        let mut effects = func.effects.clone();
        if let Some(effs) = &mut effects {
            self.module.resolve_effect_annotations(effs);
        }

        // Inject ability operations from ability annotations into scope.
        // This enables effect-directed name resolution: when a function declares
        // an effect like `->{abilities::Abort}`, its operations (e.g., `abort()`)
        // become directly callable without qualification. The resolved row names
        // an ability imported by the enclosing module by its full path.
        if let Some(effects) = &effects {
            self.inject_ability_operations(effects);
        }

        // Resolve body
        let body = self.resolve_expr(&func.body);

        self.effect_ops.clear();
        self.pop_scope();

        FuncDecl {
            id: func.id,
            visibility: func.visibility.clone(),
            name: func.name.clone(),
            type_params,
            params,
            return_ty,
            effects,
            body,
        }
    }

    /// Inject ability operations from ability annotations into the current scope.
    ///
    /// For each ability in the ability row, look up its operations in the module
    /// environment and make them available as unqualified names. This enables
    /// calling `abort()` instead of `abilities::Abort::abort()` when the function
    /// declares `->{abilities::Abort}`.
    ///
    /// Parameters and local variables take precedence (already bound before this).
    fn inject_ability_operations(&mut self, effects: &[TypeAnnotation]) {
        self.effect_ops.clear();
        for ann in effects {
            let Some(ability_name) = Resolver::extract_ability_name(ann) else {
                continue;
            };

            // The annotation names the ability by its identity.
            let Some(namespace) = self.module.view().locate(&ability_name) else {
                continue;
            };
            let operations: Vec<(Symbol, Binding<'db>)> = self
                .module
                .view()
                .iter_namespace(&namespace)
                .filter(|(_, binding)| matches!(binding, Binding::AbilityOp { .. }))
                .map(|(name, binding)| (name, binding.clone()))
                .collect();
            for (op_name, binding) in operations {
                // Don't override existing entries (first ability wins;
                // TODO: detect ambiguity when multiple abilities export same op name)
                self.effect_ops.entry(op_name).or_insert(binding);
            }
        }
    }

    /// Resolve an expression.
    fn resolve_expr(&mut self, expr: &Expr<UnresolvedName>) -> Expr<ResolvedRef<'db>> {
        let kind = match &*expr.kind {
            ExprKind::Var(name) => {
                // The type the name is used at selects among the functions
                // several `use`s give it.
                match self.unqualified_functions(name) {
                    Some(candidates) => ExprKind::Functions {
                        name: name.path.clone(),
                        candidates,
                    },
                    None => ExprKind::Var(self.resolve_name(name)),
                }
            }

            ExprKind::NatLit(n) => ExprKind::NatLit(*n),
            ExprKind::IntLit(n) => ExprKind::IntLit(*n),
            ExprKind::FloatLit(f) => ExprKind::FloatLit(*f),
            ExprKind::StringLit(s) => ExprKind::StringLit(s.clone()),
            ExprKind::BytesLit(b) => ExprKind::BytesLit(b.clone()),
            ExprKind::BoolLit(b) => ExprKind::BoolLit(*b),
            ExprKind::Nil => ExprKind::Nil,
            ExprKind::RuneLit(c) => ExprKind::RuneLit(*c),

            ExprKind::Call {
                callee,
                args,
                syntax,
            } => {
                let args: Vec<_> = args.iter().map(|a| self.resolve_expr(a)).collect();
                let callee = match &*callee.kind {
                    ExprKind::Var(name) => match self.unqualified_functions(name) {
                        // The call's types select the function, as they do
                        // for a method call.
                        Some(candidates) => Expr::new(
                            callee.id,
                            ExprKind::Functions {
                                name: name.path.clone(),
                                candidates,
                            },
                        ),
                        None => self.resolve_expr(callee),
                    },
                    _ => self.resolve_expr(callee),
                };
                ExprKind::Call {
                    callee,
                    args,
                    syntax: *syntax,
                }
            }

            ExprKind::Cons { ctor, args } => {
                let resolved_ctor = self.resolve_constructor(ctor);
                let args = args.iter().map(|a| self.resolve_expr(a)).collect();
                ExprKind::Cons {
                    ctor: resolved_ctor,
                    args,
                }
            }

            ExprKind::Record {
                type_name,
                fields,
                spread,
            } => {
                let resolved_type = self.resolve_constructor(type_name);
                let fields = fields
                    .iter()
                    .map(|f| FieldInit {
                        id: f.id,
                        name: f.name.clone(),
                        value: self.resolve_expr(&f.value),
                    })
                    .collect();
                let spread = spread.as_ref().map(|e| self.resolve_expr(e));
                ExprKind::Record {
                    type_name: resolved_type,
                    fields,
                    spread,
                }
            }

            ExprKind::Functions { name, .. } => {
                let candidates = if let Some(simple) = name.as_simple() {
                    let functions = self.method_functions(simple, expr.id);
                    let written = UnresolvedName::new(name.clone(), expr.id);
                    match self.visible_functions(functions, expr.id, &written) {
                        Some(functions) => MethodCandidates::Functions(functions),
                        None => MethodCandidates::Unresolved,
                    }
                } else {
                    self.method_candidates(&UnresolvedName::new(name.clone(), expr.id))
                };
                match candidates {
                    // One function is the callee, as a name that resolves to
                    // it would be.
                    MethodCandidates::Functions(candidates) if candidates.single().is_some() => {
                        ExprKind::Var(candidates.single().expect("one candidate").clone())
                    }
                    MethodCandidates::Functions(candidates) => ExprKind::Functions {
                        name: name.clone(),
                        candidates,
                    },
                    MethodCandidates::Callee(callee) => ExprKind::Var(callee),
                    MethodCandidates::Unresolved => {
                        let name = name.leaf().expect("a method path has a segment");
                        ExprKind::Var(ResolvedRef::local(LocalId::UNRESOLVED, name.clone()))
                    }
                }
            }

            ExprKind::Block { stmts, value } => {
                self.push_scope();
                let stmts = stmts.iter().map(|s| self.resolve_stmt(s)).collect();
                let value = self.resolve_expr(value);
                self.pop_scope();
                ExprKind::Block { stmts, value }
            }

            ExprKind::Case { scrutinee, arms } => {
                let scrutinee = self.resolve_expr(scrutinee);
                let arms = arms.iter().map(|a| self.resolve_arm(a)).collect();
                ExprKind::Case { scrutinee, arms }
            }

            ExprKind::Lambda { params, body } => {
                self.push_scope();
                let params: Vec<Param> = params
                    .iter()
                    .map(|p| {
                        let mut p = p.clone();
                        if let Some(ty) = &mut p.ty {
                            self.module.resolve_annotation_paths(ty);
                        }
                        p.local_id = Some(self.bind_local(p.name.clone()));
                        p
                    })
                    .collect();
                let body = self.resolve_expr(body);
                self.pop_scope();
                ExprKind::Lambda { params, body }
            }

            ExprKind::Handle { body, handlers } => {
                let body = self.resolve_expr(body);
                let handlers = handlers
                    .iter()
                    .map(|h| self.resolve_handler_arm(h))
                    .collect();
                ExprKind::Handle { body, handlers }
            }

            ExprKind::Resume { arg, .. } => {
                let arg = self.resolve_expr(arg);
                let local_id = self.resume_local_id_stack.last().copied();
                ExprKind::Resume { arg, local_id }
            }

            ExprKind::Become { call } => ExprKind::Become {
                call: self.resolve_expr(call),
            },

            ExprKind::Tuple(exprs) => {
                let exprs = exprs.iter().map(|e| self.resolve_expr(e)).collect();
                ExprKind::Tuple(exprs)
            }

            ExprKind::List(exprs) => {
                let exprs = exprs.iter().map(|e| self.resolve_expr(e)).collect();
                ExprKind::List(exprs)
            }

            ExprKind::BinOp { op, lhs, rhs } => {
                let lhs = self.resolve_expr(lhs);
                let rhs = self.resolve_expr(rhs);
                ExprKind::BinOp { op: *op, lhs, rhs }
            }

            ExprKind::Error => ExprKind::Error,
        };

        Expr::new(expr.id, kind)
    }

    /// Resolve a statement.
    fn resolve_stmt(&mut self, stmt: &Stmt<UnresolvedName>) -> Stmt<ResolvedRef<'db>> {
        match stmt {
            Stmt::Let {
                id,
                pattern,
                ty,
                value,
            } => {
                let value = self.resolve_expr(value);
                let pattern = self.resolve_pattern_with_bindings(pattern);
                let mut ty = ty.clone();
                if let Some(ty) = &mut ty {
                    self.module.resolve_annotation_paths(ty);
                }
                Stmt::Let {
                    id: *id,
                    pattern,
                    ty,
                    value,
                }
            }
            Stmt::Expr { id, expr } => {
                let expr = self.resolve_expr(expr);
                Stmt::Expr { id: *id, expr }
            }
        }
    }

    /// Resolve a case arm.
    fn resolve_arm(&mut self, arm: &Arm<UnresolvedName>) -> Arm<ResolvedRef<'db>> {
        self.push_scope();
        let pattern = self.resolve_pattern_with_bindings(&arm.pattern);
        let guard = arm.guard.as_ref().map(|e| self.resolve_expr(e));
        let body = self.resolve_expr(&arm.body);
        self.pop_scope();

        Arm {
            id: arm.id,
            pattern,
            guard,
            body,
        }
    }

    /// Resolve a handler arm.
    fn resolve_handler_arm(
        &mut self,
        handler: &HandlerArm<UnresolvedName>,
    ) -> HandlerArm<ResolvedRef<'db>> {
        self.push_scope();

        let kind = match &handler.kind {
            HandlerKind::Do { binding } => {
                let binding = self.resolve_pattern_with_bindings(binding);
                HandlerKind::Do { binding }
            }
            HandlerKind::Fn {
                ability,
                op,
                params,
            } => {
                let resolved_ability = self.resolve_handler_ability(ability);
                let resolved_params = params
                    .iter()
                    .map(|p| self.resolve_pattern_with_bindings(p))
                    .collect();
                HandlerKind::Fn {
                    ability: resolved_ability,
                    op: op.clone(),
                    params: resolved_params,
                }
            }
            HandlerKind::Op {
                ability,
                op,
                params,
                ..
            } => {
                let resolved_ability = self.resolve_handler_ability(ability);
                let resolved_params = params
                    .iter()
                    .map(|p| self.resolve_pattern_with_bindings(p))
                    .collect();
                // Allocate a synthetic LocalId for `resume` so that lambda
                // capture analysis can track the continuation value.
                let resume_id = self.local_id_gen.fresh();
                self.resume_local_id_stack.push(resume_id);
                HandlerKind::Op {
                    ability: resolved_ability,
                    op: op.clone(),
                    params: resolved_params,
                    resume_local_id: Some(resume_id),
                }
            }
        };

        let is_op = matches!(kind, HandlerKind::Op { .. });
        let body = self.resolve_expr(&handler.body);
        if is_op {
            self.resume_local_id_stack.pop();
        }
        self.pop_scope();

        HandlerArm {
            id: handler.id,
            kind,
            body,
        }
    }

    /// Resolve ability reference in a handler arm.
    /// Skips "_" placeholder (unqualified ops) without emitting diagnostics.
    fn resolve_handler_ability(&mut self, ability: &UnresolvedName) -> ResolvedRef<'db> {
        if ability.path == ["_"] {
            ResolvedRef::local(LocalId::UNRESOLVED, ability.name())
        } else {
            self.resolve_name(ability)
        }
    }

    /// Resolve a pattern, binding any names it introduces.
    fn resolve_pattern_with_bindings(
        &mut self,
        pattern: &Pattern<UnresolvedName>,
    ) -> Pattern<ResolvedRef<'db>> {
        let kind = match &*pattern.kind {
            PatternKind::Wildcard => PatternKind::Wildcard,

            PatternKind::Bind { name, .. } => {
                let local_id = self.bind_local(name.clone());
                PatternKind::Bind {
                    name: name.clone(),
                    local_id: Some(local_id),
                }
            }

            PatternKind::Literal(lit) => PatternKind::Literal(lit.clone()),

            PatternKind::Variant { ctor, fields } => {
                let resolved_ctor = self.resolve_constructor(ctor);
                let fields = fields
                    .iter()
                    .map(|p| self.resolve_pattern_with_bindings(p))
                    .collect();
                PatternKind::Variant {
                    ctor: resolved_ctor,
                    fields,
                }
            }

            PatternKind::Record {
                type_name,
                fields,
                rest,
            } => {
                let resolved_type = self.resolve_constructor(type_name);
                let fields = fields
                    .iter()
                    .map(|f| {
                        let pattern = if let Some(pattern) = &f.pattern {
                            self.resolve_pattern_with_bindings(pattern)
                        } else {
                            let local_id = self.bind_local(f.name.clone());
                            Pattern::new(
                                f.id,
                                PatternKind::Bind {
                                    name: f.name.clone(),
                                    local_id: Some(local_id),
                                },
                            )
                        };
                        FieldPattern {
                            id: f.id,
                            name_id: f.name_id,
                            name: f.name.clone(),
                            pattern: Some(pattern),
                        }
                    })
                    .collect();
                PatternKind::Record {
                    type_name: resolved_type,
                    fields,
                    rest: *rest,
                }
            }

            PatternKind::Tuple(patterns) => {
                let patterns = patterns
                    .iter()
                    .map(|p| self.resolve_pattern_with_bindings(p))
                    .collect();
                PatternKind::Tuple(patterns)
            }

            PatternKind::List(patterns) => {
                let patterns = patterns
                    .iter()
                    .map(|p| self.resolve_pattern_with_bindings(p))
                    .collect();
                PatternKind::List(patterns)
            }

            PatternKind::ListRest { head, rest, .. } => {
                let head = head
                    .iter()
                    .map(|p| self.resolve_pattern_with_bindings(p))
                    .collect();
                // Bind the rest variable if it's not "_"
                let rest_local_id = if let Some(rest_name) = rest.clone()
                    && rest_name != "_"
                {
                    Some(self.bind_local(rest_name))
                } else {
                    None
                };
                PatternKind::ListRest {
                    head,
                    rest: rest.clone(),
                    rest_local_id,
                }
            }

            PatternKind::As { pattern, name, .. } => {
                let pattern = self.resolve_pattern_with_bindings(pattern);
                let local_id = Some(self.bind_local(name.clone()));
                PatternKind::As {
                    pattern,
                    name: name.clone(),
                    local_id,
                }
            }

            PatternKind::Error => PatternKind::Error,
        };

        Pattern::new(pattern.id, kind)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::NodeId;
    use salsa_test_macros::salsa_test;

    #[salsa_test]
    fn test_resolve_local_variable(db: &salsa::DatabaseImpl) {
        let env = PackageEnv::new();
        let module = Resolver::new(db, env, &Dependencies::none(), SpanMap::empty());
        let mut resolver = FunctionResolver::new(&module);

        let param_name = Symbol::new("x");

        // Bind the parameter
        resolver.push_scope();
        resolver.bind_local(param_name.clone());

        // Create unresolved reference to the parameter
        let body_var =
            UnresolvedName::new(SymbolPath::new([param_name.clone()]), NodeId::from_raw(2));

        // Resolve the variable reference
        let resolved = resolver.resolve_name(&body_var);

        // Should resolve to a local variable
        match resolved {
            ResolvedRef::Local { name, .. } => {
                assert_eq!(name, param_name);
            }
            _ => panic!("Expected local variable, got {:?}", resolved),
        }
    }

    #[salsa_test]
    fn test_scope_isolation(db: &salsa::DatabaseImpl) {
        let env = PackageEnv::new();
        let module = Resolver::new(db, env, &Dependencies::none(), SpanMap::empty());
        let mut resolver = FunctionResolver::new(&module);

        let x = Symbol::new("x");
        let y = Symbol::new("y");

        // Bind x in outer scope, y in inner scope
        resolver.push_scope();
        let x_id = resolver.bind_local(x.clone());

        resolver.push_scope();
        let y_id = resolver.bind_local(y);
        resolver.pop_scope();

        // After popping inner scope, x should still be visible
        let x_ref = UnresolvedName::new(SymbolPath::new([x]), NodeId::from_raw(1));
        let resolved = resolver.resolve_name(&x_ref);

        match resolved {
            ResolvedRef::Local { id, .. } => assert_eq!(id, x_id),
            _ => panic!("Expected local x to still be visible after inner scope pop"),
        }

        // Note: Testing that y is NOT visible would trigger accumulate(),
        // which requires a tracked function context. Such tests belong in
        // integration tests via resolved_module query.
        let _ = y_id; // suppress unused warning
    }

    #[salsa_test]
    fn test_nested_scopes(db: &salsa::DatabaseImpl) {
        let env = PackageEnv::new();
        let module = Resolver::new(db, env, &Dependencies::none(), SpanMap::empty());
        let mut resolver = FunctionResolver::new(&module);

        let x = Symbol::new("x");
        let y = Symbol::new("y");

        // Outer scope: bind x
        resolver.push_scope();
        let x_id = resolver.bind_local(x.clone());

        // Inner scope: bind y
        resolver.push_scope();
        let y_id = resolver.bind_local(y.clone());

        // Both x and y should be visible from inner scope
        let x_ref = UnresolvedName::new(SymbolPath::new([x]), NodeId::from_raw(1));
        let y_ref = UnresolvedName::new(SymbolPath::new([y]), NodeId::from_raw(2));

        let resolved_x = resolver.resolve_name(&x_ref);
        let resolved_y = resolver.resolve_name(&y_ref);

        match resolved_x {
            ResolvedRef::Local { id, .. } => assert_eq!(id, x_id),
            _ => panic!("Expected local x"),
        }
        match resolved_y {
            ResolvedRef::Local { id, .. } => assert_eq!(id, y_id),
            _ => panic!("Expected local y"),
        }

        // Exit inner scope
        resolver.pop_scope();

        // x should still be visible
        let resolved_x = resolver.resolve_name(&x_ref);
        match resolved_x {
            ResolvedRef::Local { id, .. } => assert_eq!(id, x_id),
            _ => panic!("Expected local x after pop"),
        }

        // Note: Testing that y is NOT visible after pop would trigger accumulate(),
        // which requires a tracked function context. Such tests belong in
        // integration tests via resolved_module query.
    }

    #[salsa_test]
    fn test_list_rest_pattern_local_id(db: &salsa::DatabaseImpl) {
        use crate::ast::{Pattern, PatternKind};

        let env = PackageEnv::new();
        let module = Resolver::new(db, env, &Dependencies::none(), SpanMap::empty());
        let mut resolver = FunctionResolver::new(&module);
        resolver.push_scope();

        // Create a ListRest pattern: [head, ..rest]
        let head_pattern = Pattern::new(
            NodeId::from_raw(1),
            PatternKind::Bind {
                name: Symbol::new("head"),
                local_id: None,
            },
        );
        let rest_name = Symbol::new("rest");
        let list_rest_pattern = Pattern::new(
            NodeId::from_raw(2),
            PatternKind::ListRest {
                head: vec![head_pattern],
                rest: Some(rest_name),
                rest_local_id: None,
            },
        );

        // Resolve the pattern
        let resolved = resolver.resolve_pattern_with_bindings(&list_rest_pattern);

        // Check that rest has a LocalId
        let PatternKind::ListRest {
            rest_local_id,
            rest,
            ..
        } = resolved.kind.as_ref()
        else {
            panic!("Expected ListRest pattern");
        };

        assert!(rest.is_some());
        assert!(rest_local_id.is_some(), "rest should have a LocalId");
    }

    #[salsa_test]
    fn test_as_pattern_local_id(db: &salsa::DatabaseImpl) {
        use crate::ast::{Pattern, PatternKind};

        let env = PackageEnv::new();
        let module = Resolver::new(db, env, &Dependencies::none(), SpanMap::empty());
        let mut resolver = FunctionResolver::new(&module);
        resolver.push_scope();

        // Create an As pattern: _ as all
        let inner_pattern = Pattern::new(NodeId::from_raw(1), PatternKind::Wildcard);
        let as_name = Symbol::new("all");
        let as_pattern = Pattern::new(
            NodeId::from_raw(2),
            PatternKind::As {
                pattern: inner_pattern,
                name: as_name.clone(),
                local_id: None,
            },
        );

        // Resolve the pattern
        let resolved = resolver.resolve_pattern_with_bindings(&as_pattern);

        // Check that the as-binding has a LocalId
        let PatternKind::As { local_id, name, .. } = resolved.kind.as_ref() else {
            panic!("Expected As pattern");
        };

        assert_eq!(*name, as_name);
        assert!(local_id.is_some(), "as-binding should have a LocalId");
    }

    mod best_matches {
        use super::super::best_matches_by;

        #[test]
        fn finds_close_typo() {
            let candidates = ["compute", "compare", "display"];
            let result = best_matches_by(
                candidates.iter(),
                |s| strsim::jaro_winkler("compue", s),
                0.8,
                3,
            );
            assert!(result.contains(&&"compute"));
            assert!(!result.contains(&&"display"));
        }

        #[test]
        fn no_match_for_unrelated_name() {
            let candidates = ["foo", "bar", "baz"];
            let result = best_matches_by(
                candidates.iter(),
                |s| strsim::jaro_winkler("xyzzy", s),
                0.8,
                3,
            );
            assert!(result.is_empty());
        }

        #[test]
        fn respects_max_limit() {
            let candidates = ["print", "printf", "println", "printa", "printi"];
            let result = best_matches_by(
                candidates.iter(),
                |s| strsim::jaro_winkler("printt", s),
                0.5,
                2,
            );
            assert!(result.len() <= 2);
        }

        #[test]
        fn sorted_by_descending_score() {
            // "prnt" vs candidates: "print" is closer than "point"
            let candidates = ["point", "print", "paint"];
            let result = best_matches_by(
                candidates.iter(),
                |s| strsim::jaro_winkler("prnt", s),
                0.5,
                3,
            );
            assert!(result.len() >= 2);
            // Verify descending score order
            for pair in result.windows(2) {
                let s0 = strsim::jaro_winkler("prnt", pair[0]);
                let s1 = strsim::jaro_winkler("prnt", pair[1]);
                assert!(
                    s0 >= s1,
                    "{} ({}) should score >= {} ({})",
                    pair[0],
                    s0,
                    pair[1],
                    s1
                );
            }
        }

        #[test]
        fn threshold_filters_low_scores() {
            let candidates = ["abc", "xyz", "abcd"];
            let result = best_matches_by(
                candidates.iter(),
                |s| strsim::jaro_winkler("abcde", s),
                0.95,
                3,
            );
            // With threshold 0.95, only very close matches survive
            assert!(result.len() <= 1);
        }
    }
}
