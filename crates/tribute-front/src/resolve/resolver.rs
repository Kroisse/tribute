//! Name resolution logic.
//!
//! This module transforms `Expr<UnresolvedName>` into `Expr<ResolvedRef<'db>>`
//! by looking up names in the module environment and local scopes.

use rustc_hash::FxHashMap as HashMap;

use itertools::Itertools;
use salsa::Accumulator as _;
use tribute_core::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use tribute_ir::ModulePathExt as _;
use trunk_ir::Symbol;

use crate::ast::{
    Arm, Decl, Expr, ExprKind, FIELD_LENS_FUNCTIONS, FieldDecl, FieldInit, FieldPattern, FuncDecl,
    HandlerArm, HandlerKind, LocalId, LocalIdGen, MethodPath, Module, ModulePath, NodeId, Param,
    ParamDecl, Pattern, PatternKind, ResolvedRef, SpanMap, Stmt, TypeAnnotation,
    TypeAnnotationKind, TypeKind, TypeParamDecl, UnresolvedName, UseDecl,
};

use super::env::{Binding, ModuleEnv};
use super::path::{PathKeywordError, absolute_path};

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
    /// Functions, one of which the receiver's type selects.
    Functions(Vec<ResolvedRef<'db>>),
    /// The one callee the path names, which is not a function.
    Callee(ResolvedRef<'db>),
    /// Nothing; the path was reported.
    Unresolved,
}

/// Resolver for transforming unresolved names to resolved references.
pub struct Resolver<'db> {
    db: &'db dyn salsa::Database,
    /// Module-level environment with function and type definitions.
    env: ModuleEnv<'db>,
    /// Stack of local scopes (function parameters, let bindings, etc.).
    local_scopes: Vec<HashMap<Symbol, LocalId>>,
    /// Generator for unique LocalIds.
    local_id_gen: LocalIdGen,
    /// Span map for emitting diagnostics with source locations.
    span_map: SpanMap,
    /// Stack of LocalIds for `resume` in `op` handler arms.
    resume_local_id_stack: Vec<LocalId>,
    /// Ability operations injected from effect annotations (effect-directed resolution).
    /// Maps unqualified operation name → Binding. Cleared on each function scope.
    effect_ops: HashMap<Symbol, Binding<'db>>,
    /// Lexical nested-module namespace used for unqualified references in an
    /// inline module body.
    module_path: Vec<Symbol>,
    /// For each inline module entered, its imports: the imported name and the
    /// package-root path it names. An import is visible only in the body of
    /// the module that declares it.
    module_imports: Vec<HashMap<Symbol, Vec<Symbol>>>,
    /// The functions a name imports in each enclosing inline module when
    /// several of the module's `use`s give it different ones.
    module_imported_functions: Vec<HashMap<Symbol, Vec<crate::ast::FuncDefId<'db>>>>,
    /// Source functions that redefine a struct field's function. They are
    /// reported and dropped, so the field's function is the definition.
    redefinitions: Vec<NodeId>,
    /// How many leading segments of `module_path` are the package root:
    /// zero for a user package, one for the prelude inside its `std` module.
    package_depth: usize,
}

impl<'db> Resolver<'db> {
    /// Create a new resolver with the given environment.
    pub fn new(db: &'db dyn salsa::Database, env: ModuleEnv<'db>, span_map: SpanMap) -> Self {
        Self {
            db,
            env,
            local_scopes: vec![HashMap::default()],
            local_id_gen: LocalIdGen::new(),
            resume_local_id_stack: Vec::new(),
            span_map,
            effect_ops: HashMap::default(),
            module_path: Vec::new(),
            module_imports: Vec::new(),
            module_imported_functions: Vec::new(),
            redefinitions: Vec::new(),
            package_depth: 0,
        }
    }

    /// Resolve a package whose root is the module at the first
    /// `package_depth` segments of every module path.
    pub fn with_package_depth(mut self, package_depth: usize) -> Self {
        self.package_depth = package_depth;
        self
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

        // For simple names: check locals, builtins, then module environment
        if name.is_simple() {
            // First check local variables
            if let Some(local_id) = self.lookup_local(&sym) {
                return ResolvedRef::local(local_id, sym);
            }

            if !self.module_path.is_empty() {
                let namespace = Symbol::new(
                    &self
                        .module_path
                        .iter()
                        .map(Symbol::to_string)
                        .collect::<Vec<_>>()
                        .join("::"),
                );
                if let Some(binding) = self.env.lookup_qualified(&namespace, &sym) {
                    return self.binding_to_ref(binding, sym);
                }
            }

            // Then the enclosing inline module's imports.
            if let Some(target) = self.module_import(&sym)
                && let Some(binding) = self.lookup_path(target)
            {
                return self.binding_to_ref(binding, sym);
            }

            // An inline module sees its companion and the prelude, not the
            // items of the modules around it; the package root sees its own
            // items too.
            let binding = if self.module_path.is_empty() {
                self.env.lookup(&sym)
            } else {
                self.companion(&sym)
                    .or_else(|| self.env.lookup_library(&sym))
            };
            if let Some(binding) = binding {
                return self.binding_to_ref(binding, sym);
            }

            // Check effect-injected ability operations (effect-directed resolution)
            if let Some(binding) = self.effect_ops.get(&sym) {
                return self.binding_to_ref(binding, sym);
            }
        } else {
            // Qualified path: e.g., State::get, Option::Some, abilities::Throw::throw
            match self.lookup_qualified_name(name) {
                Ok(Some(binding)) => return self.binding_to_ref(binding, sym),
                Ok(None) => {}
                Err(error) => {
                    self.report_path_keyword(name.id, error);
                    return ResolvedRef::local(LocalId::UNRESOLVED, sym);
                }
            }
        }

        // Not found - emit diagnostic and return unresolved sentinel
        self.report_unresolved_name(name);
        ResolvedRef::local(LocalId::UNRESOLVED, sym)
    }

    /// The binding a qualified path names from the current scope.
    ///
    /// A path keyword names a package-root path, read from nowhere else. A
    /// first segment the enclosing module imports names only the imported
    /// path; it does not fall back to the package root.
    fn lookup_qualified_name(
        &self,
        name: &UnresolvedName,
    ) -> Result<Option<&Binding<'db>>, PathKeywordError> {
        let sym = name.name();
        let Some(namespace) = name.namespace() else {
            return Ok(None);
        };
        let segments: Vec<Symbol> = namespace.to_string().split("::").map(Symbol::new).collect();
        if let Some(path) = absolute_path(self.package_depth, &self.module_path, &segments)? {
            return Ok(if path.is_empty() {
                self.env.lookup(&sym)
            } else {
                let namespace = Symbol::new(&path.iter().format("::").to_string());
                self.env.lookup_qualified(&namespace, &sym)
            });
        }
        Ok(self
            .namespace_in_scope(namespace)
            .and_then(|namespace| self.env.lookup_qualified(&namespace, &sym)))
    }

    /// The functions an unqualified `name` imports when several `use`s give
    /// it different ones.
    fn imported_functions(&self, name: &UnresolvedName) -> Option<&[crate::ast::FuncDefId<'db>]> {
        let sym = name.name();
        if !name.is_simple() || self.lookup_local(&sym).is_some() {
            return None;
        }
        if self.module_path.is_empty() {
            return self.env.imported_functions(&sym);
        }
        if self.defined_in_module(&sym) {
            return None;
        }
        self.module_imported_functions
            .last()?
            .get(&sym)
            .map(Vec::as_slice)
    }

    fn report_unselected_import(&self, name: &UnresolvedName, usage: &str) {
        Diagnostic::new(
            format!(
                "`{}` imports several functions and {usage}, so no argument type selects one; \
                 name the function by its path",
                name.name()
            ),
            self.span_map.get_or_default(name.id),
            DiagnosticSeverity::Error,
            CompilationPhase::NameResolution,
        )
        .accumulate(self.db);
    }

    /// The functions the path of a qualified method call `x.a::b(y)` may
    /// name: `a::b` as the call's scope resolves it, and `m::a::b` for each
    /// module `m` in that scope. A declaration's name counts as the module
    /// that shares it.
    fn method_candidates(&self, path: &UnresolvedName) -> MethodCandidates<'db> {
        let mut functions: Vec<crate::ast::FuncDefId<'db>> = Vec::new();
        match self.lookup_qualified_name(path) {
            Ok(Some(Binding::Function { id })) => functions.push(*id),
            // A constructor or an ability operation takes the receiver as
            // its first argument without a choice by type.
            Ok(Some(binding)) => {
                return MethodCandidates::Callee(self.binding_to_ref(binding, path.name()));
            }
            Ok(None) => {}
            Err(error) => {
                self.report_path_keyword(path.id, error);
                return MethodCandidates::Unresolved;
            }
        }
        for module in self.names_in_scope() {
            let nested = UnresolvedName::new(
                Symbol::new(&format!("{module}::{}", path.qualified)),
                path.id,
            );
            if let Ok(Some(Binding::Function { id })) = self.lookup_qualified_name(&nested)
                && !functions.contains(id)
            {
                functions.push(*id);
            }
        }
        if functions.is_empty() {
            self.report_unresolved_name(path);
            return MethodCandidates::Unresolved;
        }
        functions.sort_by(|left, right| {
            left.qualified(self.db)
                .with_str(|left| right.qualified(self.db).with_str(|right| left.cmp(right)))
        });
        MethodCandidates::Functions(functions.into_iter().map(ResolvedRef::function).collect())
    }

    /// The names an unqualified reference may use at this point, apart from
    /// local variables.
    fn names_in_scope(&self) -> Vec<Symbol> {
        if self.module_path.is_empty() {
            return self.env.iter_all_names().collect();
        }
        let namespace = Symbol::new(&self.module_path.iter().format("::").to_string());
        let imports = self.module_imports.last();
        self.env
            .iter_namespace(&namespace)
            .map(|(name, _)| name)
            .chain(
                imports
                    .into_iter()
                    .flat_map(|imports| imports.keys().cloned()),
            )
            .chain(self.env.iter_library_names())
            .collect()
    }

    /// Whether the enclosing inline module itself defines `name`.
    fn defined_in_module(&self, name: &Symbol) -> bool {
        !self.module_path.is_empty() && {
            let namespace = Symbol::new(&self.module_path.iter().format("::").to_string());
            self.env.lookup_qualified(&namespace, name).is_some()
        }
    }

    /// The package-root path an import of the enclosing inline module gives
    /// `name`.
    fn module_import(&self, name: &Symbol) -> Option<&[Symbol]> {
        self.module_imports.last()?.get(name).map(Vec::as_slice)
    }

    /// `namespace` with its first segment replaced by the path an import of
    /// the enclosing inline module gives it, e.g. `Abort` after
    /// `use outer::abilities::Abort` becomes `outer::abilities::Abort`.
    fn imported_namespace(&self, namespace: &Symbol) -> Option<Symbol> {
        let imports = self.module_imports.last()?;
        let (first, rest) = namespace.with_str(|path| {
            let (first, rest) = path.split_once("::").unwrap_or((path, ""));
            (first.to_owned(), rest.to_owned())
        });
        let target = imports.get(&Symbol::new(&first))?;
        let mut path = target.iter().format("::").to_string();
        if !rest.is_empty() {
            path.push_str("::");
            path.push_str(&rest);
        }
        Some(Symbol::new(&path))
    }

    /// The package-root namespace a qualified path's namespace names.
    ///
    /// Inside an inline module the path starts from one of its imports, from
    /// the module, or from a namespace the prelude supplies. At the package
    /// root it is already a package-root namespace.
    fn namespace_in_scope(&self, namespace: Symbol) -> Option<Symbol> {
        if let Some(imported) = self.imported_namespace(&namespace) {
            return Some(imported);
        }
        let spelling = namespace.to_string();
        let (first, rest) = spelling.split_once("::").unwrap_or((&spelling, ""));
        let first = Symbol::new(first);
        if self.module_path.is_empty() {
            if self.env.has_namespace(&namespace) {
                return Some(namespace);
            }
            // A `use` gives the imported name's own namespace; a namespace
            // nested in it continues from the imported path.
            if !rest.is_empty()
                && let Some(path) = self.env.get_use_path(&first)
            {
                return Some(Symbol::new(&format!(
                    "{}::{rest}",
                    path.iter().format("::")
                )));
            }
            if self.env.declares(first.clone()) {
                return Some(namespace);
            }
        } else {
            let nested = Symbol::new(&format!(
                "{}::{namespace}",
                self.module_path.iter().format("::")
            ));
            if self.env.has_namespace(&nested) || self.defined_in_module(&first) {
                return Some(nested);
            }
        }
        if let Some(library) = self.env.library_namespace(&first) {
            let mut path = library.to_string();
            if !rest.is_empty() {
                path.push_str("::");
                path.push_str(rest);
            }
            return Some(Symbol::new(&path));
        }
        (self.module_path.is_empty() || self.env.is_library_root(&first)).then_some(namespace)
    }

    /// The package path a path starting with a library namespace alias
    /// continues to, e.g. `Option::Some` → `std::Option::Some`.
    fn library_namespace_path(&self, path: &[Symbol]) -> Option<Vec<Symbol>> {
        let (first, rest) = path.split_first()?;
        let namespace = self.env.library_namespace(first)?.to_string();
        Some(
            namespace
                .split("::")
                .map(Symbol::new)
                .chain(rest.iter().cloned())
                .collect(),
        )
    }

    /// The binding a package-root path names.
    fn lookup_path(&self, path: &[Symbol]) -> Option<&Binding<'db>> {
        let (last, namespace) = path.split_last()?;
        if namespace.is_empty() {
            return self.env.lookup(last);
        }
        let namespace = Symbol::new(&namespace.iter().format("::").to_string());
        self.env.lookup_qualified(&namespace, last)
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

    /// Rewrite the names in `ann` to the package-root paths they name.
    ///
    /// A path starting with a path keyword names the path it expands to.
    /// Inside an inline module, a type or ability name the module does not
    /// declare comes from its imports or the prelude, and a path starts from
    /// the module, an import, or a prelude namespace.
    fn resolve_annotation_paths(&self, ann: &mut TypeAnnotation) {
        match &mut ann.kind {
            TypeAnnotationKind::Named(name) if !self.module_path.is_empty() => {
                let local = name.with_str(|spelling| {
                    spelling.starts_with(|c: char| c.is_ascii_lowercase())
                        || TypeKind::from_primitive_name(spelling).is_some()
                });
                if local || self.defined_in_module(name) {
                    return;
                }
                if let Some(path) = self.companion_type(name) {
                    ann.kind = TypeAnnotationKind::Path(path);
                } else if let Some(target) = self.module_import(name) {
                    ann.kind = TypeAnnotationKind::Path(target.to_vec());
                } else if let Some(path) = self.env.library_path(name) {
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
                    && !self.env.declares(name.clone())
                    && self.env.get_use_path(name).is_none()
                    && let Some(path) = self.env.library_path(name)
                {
                    ann.kind = TypeAnnotationKind::Path(path.to_vec());
                }
            }
            TypeAnnotationKind::Path(segments) => {
                match absolute_path(self.package_depth, &self.module_path, segments) {
                    Ok(Some(path)) if !path.is_empty() => *segments = path,
                    Ok(Some(_)) => {
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
                        ann.kind = TypeAnnotationKind::Error;
                    }
                    Ok(None) if self.module_path.is_empty() => {
                        let first = segments[0].clone();
                        if !self.env.declares(first)
                            && let Some(path) = self.library_namespace_path(segments)
                        {
                            *segments = path;
                        }
                    }
                    Ok(None) => {
                        let (first, rest) = segments.split_first().expect("a path has a segment");
                        if let Some(target) = self.module_import(first) {
                            *segments = target.iter().chain(rest).cloned().collect();
                        } else if self.defined_in_module(first) {
                            *segments =
                                self.module_path.iter().chain(&*segments).cloned().collect();
                        } else if let Some(path) = self.library_namespace_path(segments) {
                            *segments = path;
                        } else if !self.env.is_library_root(first) {
                            let path = Symbol::new(&segments.iter().format("::").to_string());
                            self.report_unresolved_annotation(ann.id, &path);
                            ann.kind = TypeAnnotationKind::Error;
                        }
                    }
                    Err(error) => {
                        self.report_path_keyword(ann.id, error);
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
                if self.module_path.is_empty() {
                    self.resolve_effect_annotations(abilities);
                }
                params
                    .iter_mut()
                    .chain(abilities)
                    .for_each(|ann| self.resolve_annotation_paths(ann));
                self.resolve_annotation_paths(result);
            }
            TypeAnnotationKind::Tuple(elements) => elements
                .iter_mut()
                .for_each(|element| self.resolve_annotation_paths(element)),
            TypeAnnotationKind::Infer | TypeAnnotationKind::Error => {}
        }
    }

    /// The package-root path of the type or ability a module's own name
    /// names: the declaration beside the module that it accompanies, as
    /// `mod Option` accompanies `enum Option`.
    fn companion_type(&self, name: &Symbol) -> Option<Vec<Symbol>> {
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
            parent.iter().chain([name]).cloned().collect()
        })
    }

    /// The declaration beside the current module that shares its name, which
    /// the module sees under that name.
    fn companion(&self, name: &Symbol) -> Option<&Binding<'db>> {
        let (own, parent) = self.module_path.split_last()?;
        if *own != name {
            return None;
        }
        if parent.is_empty() {
            return self.env.lookup(name);
        }
        let namespace = Symbol::new(&parent.iter().format("::").to_string());
        self.env.lookup_qualified(&namespace, name)
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

    /// Report an unresolved name diagnostic, with "did you mean?" suggestions.
    fn report_unresolved_name(&self, name: &UnresolvedName) {
        let span = self.span_map.get_or_default(name.id);
        let similar = self.find_similar_names(&name.name());

        let message = if similar.is_empty() {
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
        .accumulate(self.db);
    }

    /// Find names in scope that are similar to the given name.
    fn find_similar_names(&self, name: &Symbol) -> Vec<Symbol> {
        use rustc_hash::FxHashSet as HashSet;

        let candidates: HashSet<Symbol> = self
            .local_scopes
            .iter()
            .flat_map(|scope| scope.keys().cloned())
            .chain(self.env.iter_all_names())
            .collect();

        let target = name.to_string();
        best_matches_by(
            candidates,
            |sym: &Symbol| sym.with_str(|s| strsim::jaro_winkler(&target, s)),
            0.8,
            3,
        )
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
                let path_ref = ModulePath::new(self.db, path.clone());
                ResolvedRef::Module { path: path_ref }
            }
            Binding::AbilityOp { ability, op, kind } => {
                ResolvedRef::ability_op(*ability, op.clone(), *kind)
            }
            Binding::Ability { id } => ResolvedRef::ability(*id),
        }
    }

    /// Resolve a module, transforming all declarations.
    pub fn resolve_module(&mut self, module: &Module<UnresolvedName>) -> Module<ResolvedRef<'db>> {
        self.report_field_lens_redefinitions(&module.decls);
        let decls = module
            .decls
            .iter()
            .map(|decl| self.resolve_decl(decl))
            .collect();

        Module {
            id: module.id,
            name: module.name.clone(),
            decls,
        }
    }

    /// Resolve a declaration.
    fn resolve_decl(&mut self, decl: &Decl<UnresolvedName>) -> Decl<ResolvedRef<'db>> {
        match decl {
            Decl::Function(f) => Decl::Function(self.resolve_func_decl(f)),
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
            Decl::Module(m) => Decl::Module(self.resolve_module_decl(m)),
        }
    }

    /// Resolve a module declaration.
    fn resolve_module_decl(
        &mut self,
        module: &crate::ast::ModuleDecl<UnresolvedName>,
    ) -> crate::ast::ModuleDecl<ResolvedRef<'db>> {
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
            .filter_map(|decl| match decl {
                Decl::Use(import) => Some(import),
                _ => None,
            })
            .collect();
        self.module_imports.push(HashMap::default());
        loop {
            let resolved: Vec<(Symbol, Vec<Symbol>)> = uses
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
        let body = module.body.as_ref().map(|decls| self.resolve_decls(decls));
        self.module_imported_functions.pop();
        self.module_imports.pop();
        self.module_path.pop();

        crate::ast::ModuleDecl {
            id: module.id,
            name: module.name.clone(),
            is_pub: module.is_pub,
            body,
        }
    }

    /// Report a companion module that declares a function the struct beside
    /// it already has, `T::f::set` or `T::f::modify` for a named field `f`,
    /// and record the function as a redefinition.
    fn report_field_lens_redefinitions(&mut self, decls: &[Decl<UnresolvedName>]) {
        let companions = decls.iter().filter_map(|decl| match decl {
            Decl::Module(module) => Some(module),
            _ => None,
        });
        for companion in companions {
            let fields: Vec<&Symbol> = decls
                .iter()
                .filter_map(|decl| match decl {
                    Decl::Struct(declaration) if declaration.name == companion.name => {
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
                .filter_map(|decl| match decl {
                    Decl::Module(field) if fields.contains(&&field.name) => Some(field),
                    _ => None,
                });
            for field in lenses {
                for decl in field.body.iter().flatten() {
                    let (id, name) = match decl {
                        Decl::Function(function) => (function.id, &function.name),
                        Decl::ExternFunction(function) => (function.id, &function.name),
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

    /// Resolve the declarations of a module body, without the functions
    /// reported as redefinitions.
    fn resolve_decls(&mut self, decls: &[Decl<UnresolvedName>]) -> Vec<Decl<ResolvedRef<'db>>> {
        let mut resolved = Vec::with_capacity(decls.len());
        for decl in decls {
            let redefinition = match decl {
                Decl::Function(function) => self.redefinitions.contains(&function.id),
                Decl::ExternFunction(function) => self.redefinitions.contains(&function.id),
                _ => false,
            };
            if !redefinition {
                resolved.push(self.resolve_decl(decl));
            }
        }
        resolved
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
        self.resolve_param_paths(&mut params);
        let mut type_params = func.type_params.clone();
        self.resolve_bound_paths(&mut type_params);
        let mut return_ty = func.return_ty.clone();
        if let Some(ty) = &mut return_ty {
            self.resolve_annotation_paths(ty);
        }

        // Resolve imported ability names in effect annotations to qualified paths.
        // e.g., after `use abilities::Abort`, rewrite `{Abort}` → `{abilities::Abort}`
        let mut effects = func.effects.clone();
        if let Some(effs) = &mut effects {
            // Inside an inline module the general pass resolves imports too.
            if self.module_path.is_empty() {
                self.resolve_effect_annotations(effs);
            }
            effs.iter_mut()
                .for_each(|ann| self.resolve_annotation_paths(ann));
        }

        // Inject ability operations from effect annotations into scope.
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
            is_pub: func.is_pub,
            name: func.name.clone(),
            type_params,
            params,
            return_ty,
            effects,
            body,
        }
    }

    /// Inject ability operations from effect annotations into the current scope.
    ///
    /// For each ability in the effect row, look up its operations in the module
    /// environment and make them available as unqualified names. This enables
    /// calling `abort()` instead of `abilities::Abort::abort()` when the function
    /// declares `->{abilities::Abort}`.
    ///
    /// Parameters and local variables take precedence (already bound before this).
    fn inject_ability_operations(&mut self, effects: &[TypeAnnotation]) {
        self.effect_ops.clear();
        for ann in effects {
            let Some(ability_name) = Self::extract_ability_name(ann) else {
                continue;
            };

            for (op_name, binding) in self.env.iter_namespace(&ability_name) {
                if matches!(binding, Binding::AbilityOp { .. }) {
                    // Don't override existing entries (first ability wins;
                    // TODO: detect ambiguity when multiple abilities export same op name)
                    self.effect_ops.entry(op_name).or_insert(binding.clone());
                }
            }
        }
    }

    /// Extract the ability name (as a Symbol) from a type annotation.
    ///
    /// - `Named(sym)` → sym (e.g., `Abort`)
    /// - `Path(segs)` → qualified symbol (e.g., `abilities::Throw`)
    /// - `App { ctor, .. }` → recurse into ctor (e.g., `Throw` from `Throw(Nat)`)
    fn extract_ability_name(ann: &TypeAnnotation) -> Option<Symbol> {
        match &ann.kind {
            TypeAnnotationKind::Named(sym) => {
                // Row tail variables (lowercase, e.g., `e`) are not concrete abilities
                sym.with_str(|s| {
                    if s.starts_with(|c: char| c.is_ascii_uppercase()) {
                        Some(sym.clone())
                    } else {
                        None
                    }
                })
            }
            TypeAnnotationKind::Path(segs) => {
                if segs.is_empty() {
                    None
                } else {
                    Some(Symbol::new(
                        &segs
                            .iter()
                            .map(|s| s.to_string())
                            .collect::<Vec<_>>()
                            .join("::"),
                    ))
                }
            }
            TypeAnnotationKind::App { ctor, .. } => Self::extract_ability_name(ctor),
            _ => None,
        }
    }

    /// Resolve the package root's imported ability names in effect
    /// annotations to qualified paths.
    ///
    /// When an ability is imported via `use` (e.g., `use abilities::Abort`), the
    /// import stores the original module path. This method rewrites unqualified
    /// ability names in effect annotations to their qualified form so that
    /// `annotation_to_effect` creates the correct AbilityId. Inside an inline
    /// module, `resolve_annotation_paths` resolves imports instead.
    fn resolve_effect_annotations(&self, effects: &mut [TypeAnnotation]) {
        for ann in effects {
            self.resolve_ability_in_annotation(ann);
        }
    }

    fn resolve_ability_in_annotation(&self, ann: &mut TypeAnnotation) {
        match &mut ann.kind {
            TypeAnnotationKind::Named(sym)
                if sym.with_str(|s| s.starts_with(|c: char| c.is_ascii_uppercase()))
                    && !self.env.has_definition(sym) =>
            {
                if let Some(path) = self.env.get_use_path(sym)
                    && path.len() >= 2
                {
                    ann.kind = TypeAnnotationKind::Path(path.clone());
                }
            }
            TypeAnnotationKind::App { ctor, .. } => self.resolve_ability_in_annotation(ctor),
            _ => {}
        }
    }

    /// Resolve a use declaration, recording the package-root path of what it
    /// names.
    fn resolve_use_decl(&self, u: &UseDecl) -> UseDecl {
        UseDecl {
            target: self.resolve_use_target(u),
            ..u.clone()
        }
    }

    /// The package-root path of what `u` names. A path that names nothing is
    /// reported, since it would otherwise leave the import as a module
    /// placeholder.
    fn resolve_use_target(&self, u: &UseDecl) -> Option<Vec<Symbol>> {
        let message =
            if let Err(error) = absolute_path(self.package_depth, &self.module_path, &u.path) {
                error.to_string()
            } else if let Some(target) = self.use_target(&u.path) {
                return Some(target);
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

    /// The package-root path of the definition or module `path` names.
    ///
    /// A path starting with a path keyword names exactly the path it expands
    /// to. Any other path starts from the current module, or inside an inline
    /// module from a namespace the prelude supplies.
    fn use_target(&self, path: &[Symbol]) -> Option<Vec<Symbol>> {
        let names = |path: &[Symbol]| {
            let Some((last, namespace)) = path.split_last() else {
                return false;
            };
            let full = Symbol::new(&path.iter().format("::").to_string());
            // A single-segment path must name a definition: `env.lookup`
            // would also find the module placeholder this import inserted.
            let found = if namespace.is_empty() {
                self.env.has_definition(last)
            } else {
                let namespace = Symbol::new(&namespace.iter().format("::").to_string());
                self.env.lookup_qualified(&namespace, last).is_some()
            };
            found || self.env.has_namespace(&full)
        };
        match absolute_path(self.package_depth, &self.module_path, path) {
            Ok(Some(path)) => return names(&path).then_some(path),
            Err(_) => return None,
            Ok(None) => {}
        }
        if self.module_path.is_empty() {
            if names(path) {
                return Some(path.to_vec());
            }
            if self.env.declares(path.first()?.clone()) {
                return None;
            }
            let library = self.library_namespace_path(path)?;
            return names(&library).then_some(library);
        }
        // An inline module's path starts from one of its imports, the module
        // itself, or a namespace the prelude supplies.
        if let Some((first, rest)) = path.split_first()
            && let Some(target) = self.module_import(first)
        {
            let imported: Vec<Symbol> = target.iter().chain(rest).cloned().collect();
            return names(&imported).then_some(imported);
        }
        let nested: Vec<Symbol> = self.module_path.iter().chain(path).cloned().collect();
        if names(&nested) {
            return Some(nested);
        }
        if let Some(library) = self.library_namespace_path(path)
            && names(&library)
        {
            return Some(library);
        }
        let library = path
            .first()
            .is_some_and(|first| self.env.is_library_root(first));
        (library && names(path)).then(|| path.to_vec())
    }

    /// Resolve an expression.
    pub fn resolve_expr(&mut self, expr: &Expr<UnresolvedName>) -> Expr<ResolvedRef<'db>> {
        let kind = match &*expr.kind {
            ExprKind::Var(name) => {
                if self.imported_functions(name).is_some() {
                    self.report_unselected_import(name, "is not called");
                    ExprKind::Var(ResolvedRef::local(LocalId::UNRESOLVED, name.name()))
                } else {
                    ExprKind::Var(self.resolve_name(name))
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

            ExprKind::Call { callee, args } => {
                let candidates = match &*callee.kind {
                    ExprKind::Var(name) => self.imported_functions(name).map(|functions| {
                        let candidates = functions
                            .iter()
                            .map(|id| ResolvedRef::Function { id: *id })
                            .collect();
                        (name, candidates)
                    }),
                    _ => None,
                };
                let mut args: Vec<_> = args.iter().map(|a| self.resolve_expr(a)).collect();
                match candidates {
                    // The first argument selects the function by its type,
                    // as the receiver of a method call does.
                    Some((name, candidates)) if !args.is_empty() => ExprKind::MethodCall {
                        receiver: args.remove(0),
                        method: name.qualified.clone(),
                        path: Some(MethodPath {
                            id: callee.id,
                            candidates,
                        }),
                        args,
                    },
                    Some((name, _)) => {
                        self.report_unselected_import(name, "is called without an argument");
                        let unresolved = ResolvedRef::local(LocalId::UNRESOLVED, name.name());
                        ExprKind::Call {
                            callee: Expr::new(callee.id, ExprKind::Var(unresolved)),
                            args,
                        }
                    }
                    None => {
                        let callee = self.resolve_expr(callee);
                        ExprKind::Call { callee, args }
                    }
                }
            }

            ExprKind::Cons { ctor, args } => {
                let resolved_ctor = self.resolve_name(ctor);
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
                let resolved_type = self.resolve_name(type_name);
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

            ExprKind::MethodCall {
                receiver,
                method,
                path,
                args,
            } => {
                let receiver = self.resolve_expr(receiver);
                let mut args: Vec<_> = args.iter().map(|a| self.resolve_expr(a)).collect();
                let candidates = path.as_ref().map(|path| {
                    let name = UnresolvedName::new(method.clone(), path.id);
                    (path.id, self.method_candidates(&name))
                });
                match candidates {
                    None => ExprKind::MethodCall {
                        receiver,
                        method: method.clone(),
                        path: None,
                        args,
                    },
                    Some((id, MethodCandidates::Functions(candidates))) => ExprKind::MethodCall {
                        receiver,
                        method: method.clone(),
                        path: Some(MethodPath { id, candidates }),
                        args,
                    },
                    Some((id, candidates)) => {
                        let callee = match candidates {
                            MethodCandidates::Callee(callee) => callee,
                            _ => ResolvedRef::local(LocalId::UNRESOLVED, method.last_segment()),
                        };
                        args.insert(0, receiver);
                        ExprKind::Call {
                            callee: Expr::new(id, ExprKind::Var(callee)),
                            args,
                        }
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
                            self.resolve_annotation_paths(ty);
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
                    self.resolve_annotation_paths(ty);
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
        if ability.qualified == "_" {
            ResolvedRef::local(LocalId::UNRESOLVED, ability.qualified.clone())
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
                let resolved_ctor = self.resolve_name(ctor);
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
                let resolved_type = self.resolve_name(type_name);
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
        let env = ModuleEnv::new();
        let mut resolver = Resolver::new(db, env, SpanMap::default());

        let param_name = Symbol::new("x");

        // Bind the parameter
        resolver.push_scope();
        resolver.bind_local(param_name.clone());

        // Create unresolved reference to the parameter
        let body_var = UnresolvedName::new(param_name.clone(), NodeId::from_raw(2));

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
        let env = ModuleEnv::new();
        let mut resolver = Resolver::new(db, env, SpanMap::default());

        let x = Symbol::new("x");
        let y = Symbol::new("y");

        // Bind x in outer scope, y in inner scope
        resolver.push_scope();
        let x_id = resolver.bind_local(x.clone());

        resolver.push_scope();
        let y_id = resolver.bind_local(y);
        resolver.pop_scope();

        // After popping inner scope, x should still be visible
        let x_ref = UnresolvedName::new(x, NodeId::from_raw(1));
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
        let env = ModuleEnv::new();
        let mut resolver = Resolver::new(db, env, SpanMap::default());

        let x = Symbol::new("x");
        let y = Symbol::new("y");

        // Outer scope: bind x
        resolver.push_scope();
        let x_id = resolver.bind_local(x.clone());

        // Inner scope: bind y
        resolver.push_scope();
        let y_id = resolver.bind_local(y.clone());

        // Both x and y should be visible from inner scope
        let x_ref = UnresolvedName::new(x, NodeId::from_raw(1));
        let y_ref = UnresolvedName::new(y, NodeId::from_raw(2));

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

        let env = ModuleEnv::new();
        let mut resolver = Resolver::new(db, env, SpanMap::default());
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

        let env = ModuleEnv::new();
        let mut resolver = Resolver::new(db, env, SpanMap::default());
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
