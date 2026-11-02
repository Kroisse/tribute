//! Name resolution for the AST.
//!
//! This module transforms `Module<UnresolvedName>` into `Module<ResolvedRef<'db>>`.
//!
//! ## Pipeline
//!
//! 1. Build a `PackageEnv` from declarations
//! 2. Walk the AST, resolving each name reference
//!
//! ## Resolution Strategy
//!
//! Names are resolved in this order:
//! 1. Local variables (function parameters, let bindings, pattern bindings)
//! 2. Builtin operations
//! 3. Module-level definitions (functions, types, constructors)

mod env;
mod package;
mod path;
mod resolver;
mod scope;

pub use env::{Binding, Export, PackageEnv, Restriction};
pub use package::{IdentityPath, Location, NameDisplay, PackageName};
pub use resolver::Resolver;
pub use scope::{Dependencies, Scope, ScopeRef};

use rustc_hash::FxHashMap as HashMap;
use trunk_ir::Symbol;
use trunk_ir::SymbolPath;

use crate::ast::{
    AbilityId, CtorId, Decl, DeclTable, FIELD_LENS_FUNCTIONS, FuncDefId, Module, ModuleDecl,
    ModuleItem, NodeId, ResolvedRef, SpanMap, TypeDefId, UnresolvedName, Visibility,
};

/// Resolve names in a module with the environment [`build_env`] built for
/// it.
pub fn resolve_with_env<'db>(
    db: &'db dyn salsa::Database,
    module: &Module<UnresolvedName>,
    scope: &Scope<'db>,
    span_map: SpanMap<'db>,
) -> DeclTable<ResolvedRef<'db>> {
    let mut resolver = Resolver::new(db, scope.as_ref(), span_map);
    resolver.resolve_module(module)
}

/// Resolve `use` imports that reference module-qualified paths.
///
/// During `build_env`, `use` declarations are stored as `Binding::Import` placeholders
/// until the whole package is collected. This function resolves them against the
/// package and its dependencies.
///
/// For example, `use abilities::Abort` upgrades from a placeholder to the actual
/// `Ability` binding, and registers the ability's operations under the imported name's
/// namespace (so `Abort::abort` works).
fn resolve_use_imports<'db>(env: &mut PackageEnv<'db>, dependencies: &Dependencies<'db>) {
    // Collect Module bindings that need resolution
    let module_imports: Vec<(Symbol, Option<Vec<Symbol>>)> = env
        .iter_imports()
        .filter_map(|(name, binding)| match binding {
            Binding::Import { path } => (path.len() >= 2).then(|| (name, Some(path.clone()))),
            _ => env.get_use_path(&name).map(|_| (name, None)),
        })
        .collect();

    for (import_name, path) in module_imports {
        let scope = ScopeRef::new(env, dependencies);
        let target = match path {
            Some(path) => use_target(scope, &path),
            None => env.get_use_path(&import_name).and_then(|location| {
                Some((location.clone(), scope.lookup_path(location)?.clone()))
            }),
        };
        let Some((path, binding)) = target else {
            continue;
        };
        let entries = scope.collect_namespace(&path);

        // Transfer the imported name's sub-namespace entries.
        // e.g., `use abilities::Abort` makes `Abort::abort` available.
        let namespace = SymbolPath::new([import_name.clone()]);
        for (op_name, op_binding) in entries {
            env.add_to_namespace_if_absent(&namespace, &op_name, op_binding);
        }

        // Replace the placeholder with the actual binding
        env.replace_import(import_name, binding, path);
    }

    // A name that several `use`s give different functions imports them all.
    let repeated = env.take_repeated_uses();
    for (name, paths) in &repeated {
        let mut functions = Vec::new();
        for path in paths {
            match use_target(ScopeRef::new(env, dependencies), path) {
                Some((_, Binding::Function { id })) if !functions.contains(&id) => {
                    functions.push(id);
                }
                Some((_, Binding::Function { .. })) => {}
                _ => {
                    functions.clear();
                    break;
                }
            }
        }
        if functions.len() > 1 {
            env.set_imported_functions(name.clone(), functions);
        }
    }
    env.set_repeated_uses(repeated);
}

/// The location a `use` of `path` at the package root names and what is
/// declared there.
fn use_target<'db>(scope: ScopeRef<'_, 'db>, path: &[Symbol]) -> Option<(Location, Binding<'db>)> {
    let location = root_location(scope, path)?;
    let binding = match location.split_last() {
        Some((target_name, namespace)) => scope.lookup_qualified(&namespace, target_name)?.clone(),
        // A package root, such as `::std`, is a module.
        None => scope.has_namespace(&location).then(|| Binding::Module {
            path: location.identity(),
        })?,
    };
    Some((location, binding))
}

/// The location a path written at the package root names, without path
/// keywords. Its first segment is a name the package root declares or
/// imports, then a namespace the prelude supplies, then a dependency; a
/// dependency whose name the root has is out of reach.
pub(crate) fn root_location(scope: ScopeRef<'_, '_>, path: &[Symbol]) -> Option<Location> {
    if let Some((name, rest)) = path::dependency_path(path) {
        return Some(scope.dependency_root(name)?.join(rest));
    }
    let (first, rest) = path.split_first()?;
    if !scope.declares(first.clone()) && scope.lookup(first).is_none() {
        if let Some(library) = scope.library_namespace(first) {
            return Some(library.join(rest));
        }
        if let Some(root) = scope.dependency_root(first) {
            return Some(root.join(rest));
        }
    }
    Some(scope.at_root(SymbolPath::new(path.iter().cloned())))
}

/// Build the environment of the root package of an executable build from
/// its AST, with the packages it depends on.
pub fn build_env<'db>(
    db: &'db dyn salsa::Database,
    module: &Module<UnresolvedName>,
    dependencies: Dependencies<'db>,
) -> Scope<'db> {
    let env = build_package_env(db, module, &PackageName::executable_root(), &dependencies);
    Scope::new(env, dependencies)
}

/// Build the environment of the module tree of `package`, whose root
/// module is `module`, with the packages its dependency table names.
pub fn build_package_env<'db>(
    db: &'db dyn salsa::Database,
    module: &Module<UnresolvedName>,
    package: &PackageName,
    dependencies: &Dependencies<'db>,
) -> PackageEnv<'db> {
    let decls = &module.decls;
    let mut env = PackageEnv::for_package(package.clone());

    // Note: We don't use module.name as the initial prefix because
    // module.name is derived from the file name (e.g., "add" from "add.trb"),
    // but top-level functions within the file should use simple names.
    // The file-derived name is for external references, not internal ones.
    // Only nested `mod foo { ... }` declarations extend the module path.
    let mut prefix = SymbolPath::root();
    let scope = DeclScope {
        package: package.clone(),
        outer: None,
    };

    let repeats = repeated_definitions(decls);
    for item in decls {
        if !is_repeat(item, &repeats) {
            collect_item(db, &mut env, item, &mut prefix, &scope);
        }
    }

    if package.is_std() {
        inject_builtin_bindings(db, &mut env);
        inject_runtime_bindings(db, &mut env);
    }
    // A re-export may name what an import gives, and an import what a
    // re-export adds.
    resolve_use_imports(&mut env, dependencies);
    resolve_exports(&mut env, dependencies);
    resolve_use_imports(&mut env, dependencies);

    env
}

/// Add each re-export whose target resolves to the namespace of the module
/// that declares it, until no more resolve.
///
/// A re-export's path starts with a path keyword, from an import of the
/// declaring module, from the module itself, or from a library namespace.
/// The resolver reports a `use` whose target names nothing.
fn resolve_exports<'db>(env: &mut PackageEnv<'db>, dependencies: &Dependencies<'db>) {
    let mut pending = env.take_exports();
    loop {
        let before = pending.len();
        pending.retain(|export| !add_export(env, dependencies, export));
        if pending.len() == before {
            break;
        }
    }
    for export in pending {
        env.add_export(export);
    }
}

/// The locations a `use` of `path` in the module `module` may name, in
/// order. A first segment the module imports continues only from what that
/// import names; `hops` counts the imports followed. A first segment the
/// module itself has keeps the path in the module's package.
fn use_candidates(
    scope: ScopeRef<'_, '_>,
    module: &SymbolPath,
    path: &[Symbol],
    hops: usize,
) -> Vec<Location> {
    if let Some((name, rest)) = path::dependency_path(path) {
        return scope
            .dependency_root(name)
            .map(|root| root.join(rest))
            .into_iter()
            .collect();
    }
    match path::absolute_path(module, path) {
        Ok(Some(target)) => return vec![scope.at_root(SymbolPath::new(target))],
        Err(_) => return Vec::new(),
        Ok(None) => {}
    }
    let Some((first, rest)) = path.split_first() else {
        return Vec::new();
    };
    if let Some(imported) = scope.own().module_use(module, first) {
        // A cycle of imports names nothing.
        if hops > scope.own().module_use_count() {
            return Vec::new();
        }
        return use_candidates(scope, module, imported, hops + 1)
            .into_iter()
            .map(|target| target.join(rest))
            .collect();
    }
    let here = scope.at_root(module.clone());
    let mut candidates = vec![here.join(path)];
    let own = scope.lookup_qualified(&here, first).is_some()
        || scope.has_namespace(&here.child(first.clone()));
    if !own {
        if let Some(library) = scope.library_namespace(first) {
            candidates.push(library.join(rest));
        } else if let Some(root) = scope.dependency_root(first) {
            candidates.push(root.join(rest));
        }
    }
    candidates
}

/// Add `export` to its module's namespace if its target resolves.
fn add_export<'db>(
    env: &mut PackageEnv<'db>,
    dependencies: &Dependencies<'db>,
    export: &Export,
) -> bool {
    let scope = ScopeRef::new(env, dependencies);
    let candidates = use_candidates(scope, &export.module, &export.path, 0);
    let Some((target, binding)) = candidates.into_iter().find_map(|target| {
        let binding = scope
            .lookup_path(&target)
            .cloned()
            .filter(|b| !matches!(b, Binding::Import { .. }))
            .or_else(|| {
                scope.has_namespace(&target).then(|| Binding::Module {
                    path: target.identity(),
                })
            })?;
        Some((target, binding))
    }) else {
        return false;
    };
    // An entry under the exported name is visible only where both it and
    // the re-export are.
    let entries: Vec<_> = scope
        .collect_namespace(&target)
        .into_iter()
        .map(|(name, entry)| {
            let member = scope.entry_restrictions(&target, &name);
            let member: Vec<_> = if member.is_empty() {
                scope.restriction(&entry).into_iter().collect()
            } else {
                member.iter().collect()
            };
            let restrictions: Vec<Restriction> =
                export.restriction.iter().chain(member).cloned().collect();
            (name, entry, restrictions)
        })
        .collect();
    // The exported name's own namespace, e.g. `facade::Abort::abort`.
    let own = export.module.child(export.name.clone());
    for (name, entry, restrictions) in entries {
        if env.add_to_namespace_if_absent(&own, &name, entry) {
            for restriction in restrictions {
                env.restrict_entry(own.clone(), name.clone(), restriction);
            }
        }
    }
    env.add_to_namespace_if_absent(&export.module, &export.name, binding);
    env.set_export_target(export.module.clone(), export.name.clone(), target);
    if let Some(restriction) = &export.restriction {
        env.restrict_entry(
            export.module.clone(),
            export.name.clone(),
            restriction.clone(),
        );
    }
    true
}

/// What a declaration's visibility is relative to while its environment is
/// collected.
struct DeclScope {
    /// The package the declaration belongs to.
    package: PackageName,
    /// The restriction of the modules around the declaration, if any of them
    /// is not public.
    outer: Option<Restriction>,
}

impl DeclScope {
    /// The restriction of the declaration at `path` with `visibility`, in
    /// the module `module`, or `None` if it is public.
    fn restriction(
        &self,
        module: &SymbolPath,
        path: SymbolPath,
        visibility: &Visibility,
    ) -> Option<Restriction> {
        let within = match visibility {
            Visibility::Public => return self.outer.clone(),
            Visibility::Private => module.clone(),
            // The resolver reports a path that names no module around the
            // declaration, which is then private.
            Visibility::Restricted(scope) => {
                restriction_module(module, scope).unwrap_or_else(|| module.clone())
            }
        };
        let own = Restriction {
            package: self.package.clone(),
            within,
            declaration: path,
            visibility: visibility.clone(),
        };
        Some(own.within_module(self.outer.as_ref()))
    }
}

/// The module `scope`, the path of a `pub(scope)` marker, names from the
/// module `module` that declares it, if that is `module` or a module
/// around it.
pub(crate) fn restriction_module(module: &[Symbol], scope: &[Symbol]) -> Option<SymbolPath> {
    let named = path::absolute_path(module, scope).ok()??;
    module.starts_with(&named).then(|| SymbolPath::new(named))
}

/// The name of package `std`, which the prelude belongs to and every
/// package depends on. Its declarations' identities and symbols start
/// with it.
pub const STD_PACKAGE: &str = "std";

/// The module of [`STD_PACKAGE`] whose `pub` items every module sees
/// under their short names.
pub const PRELUDE_MODULE: [&str; 2] = ["prelude", "v0"];

/// Resolve the library package whose root module is `module`, in the
/// environment `build_package_env` built for it against `dependencies`.
pub fn resolve_library_with_env<'db>(
    db: &'db dyn salsa::Database,
    module: &Module<UnresolvedName>,
    env: &PackageEnv<'db>,
    dependencies: &Dependencies<'db>,
    span_map: SpanMap<'db>,
) -> DeclTable<ResolvedRef<'db>> {
    Resolver::new(db, ScopeRef::new(env, dependencies), span_map)
        .with_package(env.package().clone())
        .resolve_module(module)
}

/// Expose compiler-owned definitions through the ordinary resolver namespace
/// of package `std`, which owns them; other packages see them through it.
fn inject_builtin_bindings<'db>(db: &'db dyn salsa::Database, env: &mut PackageEnv<'db>) {
    let list = Symbol::new("List");
    let list_type = Binding::TypeDef {
        id: TypeDefId::builtin_list(db),
    };
    // The source List module contributes members, not a new nominal type.
    // Keep its namespace while exposing the compiler-owned type at this path.
    let collections = SymbolPath::new(["collections"]);
    if matches!(
        env.lookup_qualified(&collections, &list),
        None | Some(Binding::Module { .. })
    ) {
        env.add_to_namespace(collections, list, list_type);
    }
    env.add_to_namespace(
        SymbolPath::new(["io"]),
        Symbol::new("Io"),
        Binding::Ability {
            id: AbilityId::builtin_io(db),
        },
    );
}

/// Export the compiler-owned runtime types into `std::runtime`, as
/// `pub(pkg)` items of the library package (`new-plans/modules.md`,
/// "Compiler 소유 runtime 모듈"). A package that merges the library sees
/// them as another package's restricted items, so it cannot name them.
fn inject_runtime_bindings<'db>(db: &'db dyn salsa::Database, env: &mut PackageEnv<'db>) {
    let runtime = SymbolPath::new(["runtime"]);
    let pkg = vec![Symbol::new("pkg")];
    for id in [
        TypeDefId::builtin_slots(db),
        TypeDefId::builtin_dispatch(db),
    ] {
        let name = id.name(db);
        let restriction = Restriction {
            package: PackageName::std(),
            within: SymbolPath::root(),
            declaration: id.path(db).clone(),
            visibility: Visibility::Restricted(pkg.clone()),
        };
        let binding = Binding::TypeDef { id };
        env.restrict(binding.clone(), restriction);
        // The intrinsics' module `Slots` keeps its namespace; the name is the type.
        if matches!(
            env.lookup_qualified(&runtime, &name),
            None | Some(Binding::Module { .. })
        ) {
            env.add_to_namespace(runtime.clone(), name, binding);
        }
    }
}

/// A declaration that repeats the name of an earlier one in its module.
pub(crate) struct RepeatedDefinition {
    pub name: Symbol,
    pub id: NodeId,
    pub earlier: NodeId,
}

/// The declarations of a module body that repeat the name of an earlier
/// declaration. Functions, structs, enums, and abilities share the
/// environment's definition table, so a name is declared once among them.
/// A companion module and a `use` do not declare in that table.
pub(crate) fn repeated_definitions(
    items: &[ModuleItem<UnresolvedName>],
) -> Vec<RepeatedDefinition> {
    let mut seen: HashMap<Symbol, NodeId> = HashMap::default();
    let mut repeats = Vec::new();
    for item in items {
        let ModuleItem::Decl(decl) = item else {
            continue;
        };
        let (name, id) = match decl {
            Decl::Function(f) => (&f.name, f.id),
            Decl::ExternFunction(f) => (&f.name, f.id),
            Decl::Struct(s) => (&s.name, s.id),
            Decl::Enum(e) => (&e.name, e.id),
            Decl::Ability(a) => (&a.name, a.id),
            Decl::Use(_) => continue,
        };
        match seen.get(name) {
            Some(&earlier) => repeats.push(RepeatedDefinition {
                name: name.clone(),
                id,
                earlier,
            }),
            None => {
                seen.insert(name.clone(), id);
            }
        }
    }
    repeats
}

/// Whether `item` is one of the `repeats`, which the environment leaves out.
pub(crate) fn is_repeat(item: &ModuleItem<UnresolvedName>, repeats: &[RepeatedDefinition]) -> bool {
    let ModuleItem::Decl(decl) = item else {
        return false;
    };
    let id = match decl {
        Decl::Function(f) => f.id,
        Decl::ExternFunction(f) => f.id,
        Decl::Struct(s) => s.id,
        Decl::Enum(e) => e.id,
        Decl::Ability(a) => a.id,
        Decl::Use(_) => return false,
    };
    repeats.iter().any(|repeat| repeat.id == id)
}

/// Collect the definitions of a module item into the environment.
fn collect_item<'db>(
    db: &'db dyn salsa::Database,
    env: &mut PackageEnv<'db>,
    item: &ModuleItem<UnresolvedName>,
    prefix: &mut SymbolPath,
    scope: &DeclScope,
) {
    match item {
        ModuleItem::Decl(decl) => collect_definition(db, env, decl, prefix, scope),
        ModuleItem::Module(module) => collect_module(db, env, module, prefix, scope),
    }
}

/// Collect a definition from a declaration into the environment.
fn collect_definition<'db>(
    db: &'db dyn salsa::Database,
    env: &mut PackageEnv<'db>,
    decl: &Decl<UnresolvedName>,
    prefix: &mut SymbolPath,
    scope: &DeclScope,
) {
    // The identity path of the module, which the declarations' ids extend.
    let mut ids = scope.package.identity_path(prefix);
    // Every binding of a declaration, its members' included, has the
    // declaration's visibility.
    let restriction = match decl {
        Decl::Function(func) => Some((&func.name, &func.visibility)),
        Decl::ExternFunction(func) => Some((&func.name, &func.visibility)),
        Decl::Struct(s) => Some((&s.name, &s.visibility)),
        Decl::Enum(e) => Some((&e.name, &e.visibility)),
        Decl::Ability(a) => Some((&a.name, &a.visibility)),
        Decl::Use(_) => None,
    }
    .and_then(|(name, visibility)| scope.restriction(prefix, ids.child(name.clone()), visibility));
    let restrict = |env: &mut PackageEnv<'db>, binding: Binding<'db>| {
        if let Some(restriction) = &restriction {
            env.restrict(binding, restriction.clone());
        }
    };
    match decl {
        Decl::Function(func) => {
            let qualified = ids.child(func.name.clone());
            let id = FuncDefId::new(db, qualified);
            env.add_function(func.name.clone(), id);
            restrict(env, Binding::Function { id });
        }

        Decl::ExternFunction(func) => {
            let qualified = ids.child(func.name.clone());
            let id = FuncDefId::new(db, qualified);
            env.add_function(func.name.clone(), id);
            restrict(env, Binding::Function { id });
        }

        Decl::Struct(s) => {
            // Struct is both a type and a constructor
            let qualified = ids.child(s.name.clone());
            let type_def_id = TypeDefId::declared(db, qualified.clone(), s.disambiguator);
            let ctor_id = CtorId::new(db, qualified);
            env.add_type(s.name.clone(), type_def_id);
            env.add_constructor(s.name.clone(), ctor_id, None, s.fields.len());
            restrict(env, Binding::TypeDef { id: type_def_id });
            restrict(
                env,
                Binding::Constructor {
                    id: ctor_id,
                    tag: None,
                    arity: s.fields.len(),
                },
            );

            // Register field accessors in struct's namespace
            // e.g., struct Point { x: Int, y: Int } → Point::x, Point::y functions
            ids.push(s.name.clone());
            for field in &s.fields {
                if let Some(field_name) = field.name.clone() {
                    let field_qualified = ids.child(field_name.clone());
                    let func_id = FuncDefId::new(db, field_qualified);
                    let binding = Binding::Function { id: func_id };
                    restrict(env, binding.clone());
                    // Add to namespace (e.g., Point::x)
                    env.add_to_namespace(
                        SymbolPath::new([s.name.clone()]),
                        field_name.clone(),
                        binding,
                    );

                    // The field's setter and modifier, e.g. Point::x::set
                    let namespace = SymbolPath::new([s.name.clone(), field_name.clone()]);
                    ids.push(field_name.clone());
                    for name in FIELD_LENS_FUNCTIONS {
                        let name = Symbol::new(name);
                        let id = FuncDefId::new(db, ids.child(name.clone()));
                        restrict(env, Binding::Function { id });
                        env.add_to_namespace(namespace.clone(), name, Binding::Function { id });
                    }
                    ids.pop();
                }
            }
            ids.pop();
        }

        Decl::Enum(e) => {
            // Enum is a type, and each variant is a constructor
            let qualified = ids.child(e.name.clone());
            let type_def_id = TypeDefId::declared(db, qualified.clone(), e.disambiguator);
            env.add_type(e.name.clone(), type_def_id);
            restrict(env, Binding::TypeDef { id: type_def_id });

            // Each variant is a constructor in the enum's namespace
            for variant in &e.variants {
                let variant_qualified = qualified.child(variant.name.clone());
                let variant_id = CtorId::new(db, variant_qualified);
                let binding = Binding::Constructor {
                    id: variant_id,
                    tag: Some(variant.name.clone()),
                    arity: variant.fields.len(),
                };
                restrict(env, binding.clone());
                // A variant is declared only in its enum's namespace
                // (e.g., Option::Some).
                env.add_to_namespace(
                    SymbolPath::new([e.name.clone()]),
                    variant.name.clone(),
                    binding,
                );
            }
        }

        Decl::Ability(a) => {
            // Create AbilityId for this ability
            let qualified = ids.child(a.name.clone());
            let ability_id = AbilityId::source(db, qualified);

            // Register the ability itself (for handler pattern resolution)
            env.add_ability(a.name.clone(), ability_id);
            restrict(env, Binding::Ability { id: ability_id });

            // Ability operations are added to the ability's namespace
            for op in &a.operations {
                let binding = Binding::AbilityOp {
                    ability: ability_id,
                    op: op.name.clone(),
                    kind: op.kind,
                };
                restrict(env, binding.clone());
                env.add_to_namespace(SymbolPath::new([a.name.clone()]), op.name.clone(), binding);
            }
        }

        Decl::Use(u) => {
            // Import the last segment of the path
            if let Some(name) = u.path.last().cloned() {
                let import_name = u.alias.clone().unwrap_or(name);
                // A path keyword names a package-root path; the resolver
                // reports a keyword that names none.
                let path = path::absolute_path(prefix, &u.path)
                    .ok()
                    .flatten()
                    .unwrap_or_else(|| u.path.clone());
                // An inline module's re-export joins its namespace, and a
                // library package root's joins what the package exports; an
                // executable root package's imports are already seen
                // package-wide.
                let exports = !prefix.is_root() || scope.package.is_library();
                if exports {
                    env.add_module_use(prefix.clone(), import_name.clone(), u.path.clone());
                }
                if u.visibility != Visibility::Private && exports {
                    let restriction =
                        scope.restriction(prefix, ids.child(import_name.clone()), &u.visibility);
                    env.add_export(Export {
                        module: prefix.clone(),
                        name: import_name.clone(),
                        path: u.path.clone(),
                        restriction,
                    });
                }
                env.add_use(import_name, path);
            }
        }
    }
}

/// Collect the definitions of an inline module into the environment, under
/// the module's namespace.
fn collect_module<'db>(
    db: &'db dyn salsa::Database,
    env: &mut PackageEnv<'db>,
    m: &ModuleDecl<UnresolvedName>,
    prefix: &mut SymbolPath,
    scope: &DeclScope,
) {
    // For inline modules, collect bindings into a temporary environment
    // then register them under the module's namespace
    if let Some(body) = &m.body {
        let path = prefix.child(m.name.clone());
        let identity = scope.package.identity_path(&path);
        let restriction = scope.restriction(prefix, identity, &m.visibility);
        if let Some(restriction) = &restriction {
            env.restrict_module(path.clone(), restriction.clone());
        }
        let inner_scope = DeclScope {
            package: scope.package.clone(),
            outer: restriction,
        };

        // Build nested module path by appending current module name
        prefix.push(m.name.clone());

        // Collect inner declarations into a temporary environment
        let mut inner_env = PackageEnv::for_package(scope.package.clone());
        let repeats = repeated_definitions(body);
        for inner in body {
            if !is_repeat(inner, &repeats) {
                collect_item(db, &mut inner_env, inner, prefix, &inner_scope);
            }
        }

        prefix.pop();
        env.absorb_restrictions(&inner_env);

        // Register each inner definition under the module's namespace
        // e.g., `mod Foo { fn bar() {} }` makes `Foo::bar` available
        for (name, binding) in inner_env.iter_definitions() {
            env.add_to_namespace(SymbolPath::new([m.name.clone()]), name, binding.clone());
        }

        // Also transfer any nested namespaces
        // e.g., `mod Foo { enum Bar { Baz } }` makes `Foo::Bar::Baz` available
        for (inner_ns, inner_bindings) in inner_env.iter_namespaces() {
            // Create qualified namespace path: Foo::Bar
            let qualified_ns =
                SymbolPath::new(std::iter::once(&m.name).chain(inner_ns.iter()).cloned());
            for (name, binding) in inner_bindings {
                env.add_to_namespace(qualified_ns.clone(), name, binding.clone());
            }
            // Also add the inner namespace itself under the module
            // so Foo::Bar resolves to the Bar namespace
            // A type definition and its member namespace share a name
            // (`Point` and `Point::x`, `Error` and `Error::Variant`).
            // Preserve the type binding already transferred above.
            if let Some(inner) = inner_ns.as_simple() {
                env.add_to_namespace_if_absent(
                    &SymbolPath::new([m.name.clone()]),
                    inner,
                    Binding::Module {
                        path: scope.package.at(path.child(inner.clone())).identity(),
                    },
                );
            }
        }

        // Register the module itself as a namespace binding
        env.add_import(
            m.name.clone(),
            Binding::Module {
                path: scope.package.at(path.clone()).identity(),
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{
        EnumDecl, Expr, ExprKind, ExternFuncDecl, FieldDecl, FuncDecl, Module, ModuleDecl, NodeId,
        ParamDecl, StructDecl, TypeAnnotation, TypeAnnotationKind, VariantDecl,
    };
    use salsa_test_macros::salsa_test;

    fn fresh_node_id() -> NodeId {
        NodeId::from_raw(0)
    }

    /// Input wrapper for Module to use in tracked functions.
    #[salsa::input]
    struct TestModuleInput {
        module: Module<UnresolvedName>,
    }

    // =========================================================================
    // Tracked test functions - one per test case
    // =========================================================================

    #[salsa::tracked]
    fn verify_module_function_namespace(db: &dyn salsa::Database, input: TestModuleInput) {
        let env = build_env(db, input.module(db), Dependencies::none()).into_env();

        // Math::add should be accessible
        let math_add = env.lookup_qualified(&SymbolPath::new(["Math"]), &Symbol::new("add"));
        assert!(math_add.is_some(), "Math::add should be in namespace");
        assert!(
            matches!(math_add, Some(Binding::Function { .. })),
            "Math::add should be a function"
        );

        // "add" should NOT be directly accessible (not leaked to parent scope)
        let add_direct = env.lookup(&Symbol::new("add"));
        assert!(
            add_direct.is_none(),
            "add should not be leaked to parent scope"
        );

        // "Math" itself should be accessible as a module binding
        let math_module = env.lookup(&Symbol::new("Math"));
        assert!(math_module.is_some(), "Math should be accessible");
        assert!(
            matches!(math_module, Some(Binding::Module { .. })),
            "Math should be a module binding"
        );
    }

    #[salsa::tracked]
    fn verify_module_struct_namespace(db: &dyn salsa::Database, input: TestModuleInput) {
        let env = build_env(db, input.module(db), Dependencies::none()).into_env();

        // Types::Point should be accessible as a constructor
        let types_point = env.lookup_qualified(&SymbolPath::new(["Types"]), &Symbol::new("Point"));
        assert!(types_point.is_some(), "Types::Point should be in namespace");

        // "Point" should NOT be directly accessible
        let point_direct = env.lookup(&Symbol::new("Point"));
        assert!(
            point_direct.is_none(),
            "Point should not be leaked to parent scope"
        );
    }

    #[salsa::tracked]
    fn verify_module_enum_namespace(db: &dyn salsa::Database, input: TestModuleInput) {
        let env = build_env(db, input.module(db), Dependencies::none()).into_env();

        // Types::Color should be accessible
        let types_color = env.lookup_qualified(&SymbolPath::new(["Types"]), &Symbol::new("Color"));
        assert!(types_color.is_some(), "Types::Color should be in namespace");

        // Types::Color::Red is the variant; the module has no `Red`.
        let types_red =
            env.lookup_qualified(&SymbolPath::new(["Types", "Color"]), &Symbol::new("Red"));
        assert!(
            types_red.is_some(),
            "Types::Color::Red should be in namespace"
        );
        assert!(
            env.lookup_qualified(&SymbolPath::new(["Types"]), &Symbol::new("Red"))
                .is_none(),
            "Red should not be in the module's namespace"
        );

        // "Color" and "Red" should NOT be directly accessible
        assert!(
            env.lookup(&Symbol::new("Color")).is_none(),
            "Color should not be leaked"
        );
        assert!(
            env.lookup(&Symbol::new("Red")).is_none(),
            "Red should not be leaked"
        );
    }

    #[salsa::tracked]
    fn verify_module_multiple_definitions(db: &dyn salsa::Database, input: TestModuleInput) {
        let env = build_env(db, input.module(db), Dependencies::none()).into_env();

        // All should be accessible via Utils::
        assert!(
            env.lookup_qualified(&SymbolPath::new(["Utils"]), &Symbol::new("helper"))
                .is_some()
        );
        assert!(
            env.lookup_qualified(&SymbolPath::new(["Utils"]), &Symbol::new("another"))
                .is_some()
        );
        assert!(
            env.lookup_qualified(&SymbolPath::new(["Utils"]), &Symbol::new("Data"))
                .is_some()
        );

        // None should be directly accessible
        assert!(env.lookup(&Symbol::new("helper")).is_none());
        assert!(env.lookup(&Symbol::new("another")).is_none());
        assert!(env.lookup(&Symbol::new("Data")).is_none());
    }

    #[salsa::tracked]
    fn verify_multiple_modules_namespaces(db: &dyn salsa::Database, input: TestModuleInput) {
        let env = build_env(db, input.module(db), Dependencies::none()).into_env();

        // Both A::foo and B::foo should exist
        let a_foo = env.lookup_qualified(&SymbolPath::new(["A"]), &Symbol::new("foo"));
        let b_foo = env.lookup_qualified(&SymbolPath::new(["B"]), &Symbol::new("foo"));

        assert!(a_foo.is_some(), "A::foo should exist");
        assert!(b_foo.is_some(), "B::foo should exist");

        // They should be different bindings (different FuncDefIds)
        if let (Some(Binding::Function { id: id_a }), Some(Binding::Function { id: id_b })) =
            (a_foo, b_foo)
        {
            assert_ne!(id_a, id_b, "A::foo and B::foo should have different IDs");
        }
    }

    #[salsa::tracked]
    fn verify_nested_module_enum_namespace(db: &dyn salsa::Database, input: TestModuleInput) {
        let env = build_env(db, input.module(db), Dependencies::none()).into_env();

        // Outer::Inner should exist
        assert!(
            env.lookup_qualified(&SymbolPath::new(["Outer"]), &Symbol::new("Inner"))
                .is_some(),
            "Outer::Inner should exist"
        );

        // A variant is only in its enum's namespace
        assert!(
            env.lookup_qualified(&SymbolPath::new(["Outer"]), &Symbol::new("Variant"))
                .is_none(),
            "Outer::Variant should not exist"
        );

        // Outer::Inner::Variant should exist
        let qualified_ns = SymbolPath::new(["Outer", "Inner"]);
        assert!(
            env.lookup_qualified(&qualified_ns, &Symbol::new("Variant"))
                .is_some(),
            "Outer::Inner::Variant should exist"
        );
    }

    #[salsa::tracked]
    fn verify_module_no_override_parent(db: &dyn salsa::Database, input: TestModuleInput) {
        let env = build_env(db, input.module(db), Dependencies::none()).into_env();

        // foo should still be directly accessible
        assert!(
            env.lookup(&Symbol::new("foo")).is_some(),
            "foo should be accessible"
        );

        // M::bar should be accessible
        assert!(
            env.lookup_qualified(&SymbolPath::new(["M"]), &Symbol::new("bar"))
                .is_some(),
            "M::bar should be accessible"
        );

        // bar should not be directly accessible
        assert!(
            env.lookup(&Symbol::new("bar")).is_none(),
            "bar should not be directly accessible"
        );
    }

    /// Create a simple function declaration for testing.
    fn simple_func(name: &str) -> FuncDecl<UnresolvedName> {
        FuncDecl {
            id: fresh_node_id(),
            visibility: Visibility::Private,
            name: Symbol::new(name),
            type_params: vec![],
            params: vec![],
            return_ty: Some(TypeAnnotation {
                id: fresh_node_id(),
                kind: TypeAnnotationKind::Named(Symbol::new("Int")),
            }),
            effects: None,
            body: Expr::new(fresh_node_id(), ExprKind::NatLit(42)),
        }
    }

    /// Create a simple struct declaration for testing.
    fn simple_struct(name: &str) -> StructDecl {
        StructDecl {
            id: fresh_node_id(),
            visibility: Visibility::Private,
            name: Symbol::new(name),
            disambiguator: 0,
            type_params: vec![],
            fields: vec![FieldDecl {
                id: fresh_node_id(),
                name_id: fresh_node_id(),
                is_pub: false,
                name: Some(Symbol::new("value")),
                ty: TypeAnnotation {
                    id: fresh_node_id(),
                    kind: TypeAnnotationKind::Named(Symbol::new("Int")),
                },
            }],
        }
    }

    /// Create a simple enum declaration for testing.
    fn simple_enum(name: &str, variants: &[&str]) -> EnumDecl {
        EnumDecl {
            id: fresh_node_id(),
            visibility: Visibility::Private,
            name: Symbol::new(name),
            disambiguator: 0,
            type_params: vec![],
            variants: variants
                .iter()
                .map(|v| VariantDecl {
                    id: fresh_node_id(),
                    name: Symbol::new(v),
                    fields: vec![],
                })
                .collect(),
        }
    }

    /// Create an inline module declaration for testing.
    fn inline_module(
        name: &str,
        items: Vec<ModuleItem<UnresolvedName>>,
    ) -> ModuleDecl<UnresolvedName> {
        ModuleDecl {
            id: fresh_node_id(),
            name: Symbol::new(name),
            visibility: Visibility::Private,
            body: Some(items),
        }
    }

    // =========================================================================
    // Module namespace tests
    // =========================================================================

    #[salsa_test]
    fn test_module_function_registered_in_namespace(db: &dyn salsa::Database) {
        // mod Math { fn add() -> Int { 42 } }
        let module = Module::new(
            fresh_node_id(),
            Some(Symbol::new("test")),
            vec![ModuleItem::Module(inline_module(
                "Math",
                vec![ModuleItem::Decl(Decl::Function(simple_func("add")))],
            ))],
        );

        let input = TestModuleInput::new(db, module);
        verify_module_function_namespace(db, input);
    }

    #[salsa_test]
    fn test_module_struct_registered_in_namespace(db: &dyn salsa::Database) {
        // mod Types { struct Point { value: Int } }
        let module = Module::new(
            fresh_node_id(),
            Some(Symbol::new("test")),
            vec![ModuleItem::Module(inline_module(
                "Types",
                vec![ModuleItem::Decl(Decl::Struct(simple_struct("Point")))],
            ))],
        );

        let input = TestModuleInput::new(db, module);
        verify_module_struct_namespace(db, input);
    }

    #[salsa_test]
    fn test_module_enum_registered_in_namespace(db: &dyn salsa::Database) {
        // mod Types { enum Color { Red, Green, Blue } }
        let module = Module::new(
            fresh_node_id(),
            Some(Symbol::new("test")),
            vec![ModuleItem::Module(inline_module(
                "Types",
                vec![ModuleItem::Decl(Decl::Enum(simple_enum(
                    "Color",
                    &["Red", "Green", "Blue"],
                )))],
            ))],
        );

        let input = TestModuleInput::new(db, module);
        verify_module_enum_namespace(db, input);
    }

    #[salsa_test]
    fn test_module_multiple_definitions(db: &dyn salsa::Database) {
        // mod Utils {
        //     fn helper() -> Int { 1 }
        //     fn another() -> Int { 2 }
        //     struct Data { value: Int }
        // }
        let module = Module::new(
            fresh_node_id(),
            Some(Symbol::new("test")),
            vec![ModuleItem::Module(inline_module(
                "Utils",
                vec![
                    ModuleItem::Decl(Decl::Function(simple_func("helper"))),
                    ModuleItem::Decl(Decl::Function(simple_func("another"))),
                    ModuleItem::Decl(Decl::Struct(simple_struct("Data"))),
                ],
            ))],
        );

        let input = TestModuleInput::new(db, module);
        verify_module_multiple_definitions(db, input);
    }

    #[salsa_test]
    fn test_multiple_modules_separate_namespaces(db: &dyn salsa::Database) {
        // mod A { fn foo() -> Int { 1 } }
        // mod B { fn foo() -> Int { 2 } }
        let module = Module::new(
            fresh_node_id(),
            Some(Symbol::new("test")),
            vec![
                ModuleItem::Module(inline_module(
                    "A",
                    vec![ModuleItem::Decl(Decl::Function(simple_func("foo")))],
                )),
                ModuleItem::Module(inline_module(
                    "B",
                    vec![ModuleItem::Decl(Decl::Function(simple_func("foo")))],
                )),
            ],
        );

        let input = TestModuleInput::new(db, module);
        verify_multiple_modules_namespaces(db, input);
    }

    #[salsa_test]
    fn test_nested_module_enum_namespace(db: &dyn salsa::Database) {
        // mod Outer { enum Inner { Variant } }
        // Should create:
        // - Outer::Inner (the enum type)
        // - Outer::Inner::Variant (the variant via enum namespace)
        let module = Module::new(
            fresh_node_id(),
            Some(Symbol::new("test")),
            vec![ModuleItem::Module(inline_module(
                "Outer",
                vec![ModuleItem::Decl(Decl::Enum(simple_enum(
                    "Inner",
                    &["Variant"],
                )))],
            ))],
        );

        let input = TestModuleInput::new(db, module);
        verify_nested_module_enum_namespace(db, input);
    }

    #[salsa_test]
    fn test_nested_module_preserves_type_bindings(db: &dyn salsa::Database) {
        // A type's member namespace must not replace the type/constructor
        // binding when nested module namespaces are flattened.
        let module = Module::new(
            fresh_node_id(),
            Some(Symbol::new("test")),
            vec![ModuleItem::Module(inline_module(
                "std",
                vec![ModuleItem::Module(inline_module(
                    "io",
                    vec![ModuleItem::Decl(Decl::Struct(simple_struct("SystemError")))],
                ))],
            ))],
        );

        let env = build_env(db, &module, Dependencies::none()).into_env();
        assert!(matches!(
            env.lookup_qualified(&SymbolPath::new(["std", "io"]), &Symbol::new("SystemError")),
            Some(Binding::Constructor { .. })
        ));
    }

    #[salsa_test]
    fn test_module_does_not_override_parent_definitions(db: &dyn salsa::Database) {
        // fn foo() -> Int { 1 }
        // mod M { fn bar() -> Int { 2 } }
        let module = Module::new(
            fresh_node_id(),
            Some(Symbol::new("test")),
            vec![
                ModuleItem::Decl(Decl::Function(simple_func("foo"))),
                ModuleItem::Module(inline_module(
                    "M",
                    vec![ModuleItem::Decl(Decl::Function(simple_func("bar")))],
                )),
            ],
        );

        let input = TestModuleInput::new(db, module);
        verify_module_no_override_parent(db, input);
    }

    // =========================================================================
    // Extern function tests
    // =========================================================================

    #[salsa::tracked]
    fn verify_extern_function_registered(db: &dyn salsa::Database, input: TestModuleInput) {
        let env = build_env(db, input.module(db), Dependencies::none()).into_env();

        let binding = env.lookup(&Symbol::new("__bytes_len"));
        assert!(binding.is_some(), "__bytes_len should be registered");
        assert!(
            matches!(binding, Some(Binding::Function { .. })),
            "__bytes_len should be a function binding"
        );
    }

    #[salsa_test]
    fn test_extern_function_registered_as_binding(db: &dyn salsa::Database) {
        let module = Module::new(
            fresh_node_id(),
            Some(Symbol::new("test")),
            vec![Decl::ExternFunction(ExternFuncDecl {
                id: fresh_node_id(),
                visibility: Visibility::Private,
                name: Symbol::new("__bytes_len"),
                abi: Symbol::new("intrinsic"),
                params: vec![ParamDecl {
                    id: fresh_node_id(),
                    name: Symbol::new("bytes"),
                    ty: Some(TypeAnnotation {
                        id: fresh_node_id(),
                        kind: TypeAnnotationKind::Named(Symbol::new("Bytes")),
                    }),
                    local_id: None,
                }],
                return_ty: TypeAnnotation {
                    id: fresh_node_id(),
                    kind: TypeAnnotationKind::Named(Symbol::new("Int")),
                },
            })],
        );

        let input = TestModuleInput::new(db, module);
        verify_extern_function_registered(db, input);
    }

    #[salsa::tracked]
    fn verify_struct_field_accessors_in_namespace(
        db: &dyn salsa::Database,
        input: TestModuleInput,
    ) {
        let env = build_env(db, input.module(db), Dependencies::none()).into_env();

        // Point::x should be registered as a function
        let point_x = env.lookup_qualified(&SymbolPath::new(["Point"]), &Symbol::new("x"));
        assert!(
            point_x.is_some(),
            "Point::x accessor should be in namespace"
        );
        assert!(
            matches!(point_x, Some(Binding::Function { .. })),
            "Point::x should be a function binding"
        );

        // Point::y should be registered as a function
        let point_y = env.lookup_qualified(&SymbolPath::new(["Point"]), &Symbol::new("y"));
        assert!(
            point_y.is_some(),
            "Point::y accessor should be in namespace"
        );
        assert!(
            matches!(point_y, Some(Binding::Function { .. })),
            "Point::y should be a function binding"
        );
    }

    #[salsa_test]
    fn test_struct_field_accessors_in_namespace(db: &dyn salsa::Database) {
        let module = Module::new(
            fresh_node_id(),
            Some(Symbol::new("test")),
            vec![Decl::Struct(StructDecl {
                id: fresh_node_id(),
                visibility: Visibility::Private,
                name: Symbol::new("Point"),
                disambiguator: 0,
                type_params: vec![],
                fields: vec![
                    FieldDecl {
                        id: fresh_node_id(),
                        name_id: fresh_node_id(),
                        is_pub: false,
                        name: Some(Symbol::new("x")),
                        ty: TypeAnnotation {
                            id: fresh_node_id(),
                            kind: TypeAnnotationKind::Named(Symbol::new("Int")),
                        },
                    },
                    FieldDecl {
                        id: fresh_node_id(),
                        name_id: fresh_node_id(),
                        is_pub: false,
                        name: Some(Symbol::new("y")),
                        ty: TypeAnnotation {
                            id: fresh_node_id(),
                            kind: TypeAnnotationKind::Named(Symbol::new("Int")),
                        },
                    },
                ],
            })],
        );

        let input = TestModuleInput::new(db, module);
        verify_struct_field_accessors_in_namespace(db, input);
    }
}
