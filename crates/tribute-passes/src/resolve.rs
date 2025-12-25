//! Name resolution pass for Tribute.
//!
//! This pass resolves `src.*` operations to their concrete targets:
//! - `src.var` → function reference, constructor, or deferred (UFCS)
//! - `src.path` → qualified function/constructor reference
//! - `src.call` → resolved function call
//! - `src.type` → concrete type
//!
//! ## Resolution Strategy
//!
//! The resolution happens in multiple phases:
//!
//! 1. **Basic resolution**: Resolve qualified paths and unambiguous names
//! 2. **Type inference**: Infer types for resolved code
//! 3. **Type-directed resolution**: Resolve UFCS using inferred types
//! 4. **Complete type inference**: Finish type inference after UFCS resolution
//!
//! This module handles phase 1. UFCS resolution happens during type checking.

use std::collections::HashMap;

use crate::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use salsa::Accumulator;
use trunk_ir::dialect::adt;
use trunk_ir::dialect::arith;
use trunk_ir::dialect::case;
use trunk_ir::dialect::core::{self, Module};
use trunk_ir::dialect::func;
use trunk_ir::dialect::pat;
use trunk_ir::dialect::src;
use trunk_ir::dialect::ty;
use trunk_ir::rewrite::RewriteContext;
use trunk_ir::{
    Attribute, Attrs, Block, DialectOp, DialectType, IdVec, Operation, QualifiedName, Region,
    Symbol, Type, Value, ValueDef,
};

// =============================================================================
// Attribute Keys (hot path helpers)
// =============================================================================

// Symbol helpers for frequently-used attribute keys.
// These are used in hot paths (resolve_operation, collect_definitions, etc.)
// to keep call sites concise.
trunk_ir::symbols! {
    ATTR_NAME => "name",
    ATTR_TYPE => "type",
    ATTR_PATH => "path",
    ATTR_SYM_NAME => "sym_name",
    ATTR_VARIANTS => "variants",
    ATTR_ALIAS => "alias",
    #[allow(dead_code)] // Used in test helpers
    ATTR_CALLEE => "callee",
    ATTR_REST_NAME => "rest_name",
    ATTR_RESOLVED_LOCAL => "resolved_local",
    ATTR_VALUE => "value",
    ATTR_RESOLVED_CONST => "resolved_const",
}

// =============================================================================
// Module Environment
// =============================================================================

/// Information about a resolved name.
#[derive(Clone, Debug)]
pub enum Binding<'db> {
    /// A function defined in this module or imported.
    Function {
        /// Fully qualified path (e.g., "List::map")
        path: QualifiedName,
        /// Function type
        ty: Type<'db>,
    },
    /// A module/namespace binding (possibly with an associated type).
    Module {
        /// Fully qualified namespace path (e.g., "collections::List")
        namespace: QualifiedName,
        /// Optional type definition for the same name.
        type_def: Option<Type<'db>>,
    },
    /// A type constructor (struct or enum variant).
    Constructor {
        /// The type being constructed
        ty: Type<'db>,
        /// For enums, the variant tag
        tag: Option<Symbol>,
        /// Constructor parameter types
        params: IdVec<Type<'db>>,
    },
    /// A type alias or definition.
    TypeDef {
        /// The defined type
        ty: Type<'db>,
    },
    /// A constant definition.
    Const {
        /// Constant name
        name: Symbol,
        /// Constant value (to be inlined)
        value: Attribute<'db>,
        /// Constant type
        ty: Type<'db>,
    },
}

/// Module environment for name resolution.
///
/// Tracks all names visible in the current module.
#[derive(Debug, Default)]
pub struct ModuleEnv<'db> {
    /// Names defined in this module.
    definitions: HashMap<Symbol, Binding<'db>>,
    /// Names imported via `use` statements.
    imports: HashMap<Symbol, Binding<'db>>,
    /// Qualified paths (namespace → name → binding).
    namespaces: HashMap<Symbol, HashMap<Symbol, Binding<'db>>>,
}

impl<'db> ModuleEnv<'db> {
    /// Create a new empty module environment.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a function definition.
    pub fn add_function(&mut self, name: Symbol, path: QualifiedName, ty: Type<'db>) {
        self.definitions
            .insert(name, Binding::Function { path, ty });
    }

    /// Add a type constructor.
    pub fn add_constructor(
        &mut self,
        name: Symbol,
        ty: Type<'db>,
        tag: Option<Symbol>,
        params: IdVec<Type<'db>>,
    ) {
        self.definitions
            .insert(name, Binding::Constructor { ty, tag, params });
    }

    /// Add a type definition.
    pub fn add_type(&mut self, name: Symbol, ty: Type<'db>) {
        self.definitions.insert(name, Binding::TypeDef { ty });
    }

    /// Add a constant definition.
    pub fn add_const(&mut self, name: Symbol, value: Attribute<'db>, ty: Type<'db>) {
        self.definitions
            .insert(name, Binding::Const { name, value, ty });
    }

    /// Add a qualified name to a namespace.
    pub fn add_to_namespace(&mut self, namespace: Symbol, name: Symbol, binding: Binding<'db>) {
        self.namespaces
            .entry(namespace)
            .or_default()
            .insert(name, binding);
    }

    /// Look up an unqualified name.
    pub fn lookup(&self, name: Symbol) -> Option<&Binding<'db>> {
        // First check local definitions
        if let Some(b) = self.definitions.get(&name) {
            return Some(b);
        }
        // Then check imports
        self.imports.get(&name)
    }

    /// Look up a qualified path (e.g., "List::map").
    pub fn lookup_qualified(&self, namespace: Symbol, name: Symbol) -> Option<&Binding<'db>> {
        self.namespaces.get(&namespace)?.get(&name)
    }

    /// Check whether a namespace exists (e.g., "collections::List").
    pub fn has_namespace(&self, namespace: Symbol) -> bool {
        self.namespaces.contains_key(&namespace)
    }
}

// =============================================================================
// Environment Builder
// =============================================================================

/// Build a module environment from a TrunkIR module.
pub fn build_env<'db>(db: &'db dyn salsa::Database, module: &Module<'db>) -> ModuleEnv<'db> {
    let mut env = ModuleEnv::new();

    // Scan the module for definitions
    let body = module.body(db);
    for block in body.blocks(db).iter() {
        collect_definitions_from_block(db, &mut env, block);
    }

    env
}

/// Collect definitions from a block.
fn collect_definitions_from_block<'db>(
    db: &'db dyn salsa::Database,
    env: &mut ModuleEnv<'db>,
    block: &Block<'db>,
) {
    for op in block.operations(db).iter() {
        collect_definition(db, env, op);
    }
}

/// Collect a definition from an operation.
fn collect_definition<'db>(
    db: &'db dyn salsa::Database,
    env: &mut ModuleEnv<'db>,
    op: &Operation<'db>,
) {
    let dialect = op.dialect(db);
    let op_name = op.name(db);

    match (dialect, op_name) {
        (d, n) if d == func::DIALECT_NAME() && n == func::FUNC() => {
            // Function definition
            let attrs = op.attributes(db);

            if let (Some(Attribute::Symbol(sym)), Some(Attribute::Type(ty))) =
                (attrs.get(&ATTR_SYM_NAME()), attrs.get(&ATTR_TYPE()))
            {
                let path = QualifiedName::simple(*sym);
                env.add_function(*sym, path, *ty);
            }
        }
        (d, n) if d == ty::DIALECT_NAME() && n == ty::STRUCT() => {
            // Struct definition → creates constructor
            let attrs = op.attributes(db);

            if let Some(Attribute::Symbol(sym)) = attrs.get(&ATTR_SYM_NAME()) {
                let ty = op
                    .results(db)
                    .first()
                    .copied()
                    .unwrap_or_else(|| ty::var(db, std::collections::BTreeMap::new()));
                // Add type definition
                env.add_type(*sym, ty);
                // Struct constructor has same name as type
                // TODO: Get field types for constructor params
                env.add_constructor(*sym, ty, None, IdVec::new());
            }
        }
        (d, n) if d == ty::DIALECT_NAME() && n == ty::ENUM() => {
            // Enum definition → creates constructors for each variant
            let attrs = op.attributes(db);
            if let Some(Attribute::Symbol(sym)) = attrs.get(&ATTR_SYM_NAME()) {
                let ty = op
                    .results(db)
                    .first()
                    .copied()
                    .unwrap_or_else(|| ty::var(db, std::collections::BTreeMap::new()));
                // Add type definition
                env.add_type(*sym, ty);
                // Extract variants from the operation's regions or attributes
                // Each variant becomes a constructor in the type's namespace
                collect_enum_constructors(db, env, op, *sym, ty);
            }
        }
        (d, n) if d == src::DIALECT_NAME() && n == src::CONST() => {
            // Const definition
            let attrs = op.attributes(db);

            if let (Some(Attribute::Symbol(sym)), Some(value)) =
                (attrs.get(&ATTR_NAME()), attrs.get(&ATTR_VALUE()))
            {
                // Get the type from the operation result
                let ty = op
                    .results(db)
                    .first()
                    .copied()
                    .unwrap_or_else(|| ty::var(db, std::collections::BTreeMap::new()));
                env.add_const(*sym, value.clone(), ty);
            }
        }
        (d, n) if d == core::DIALECT_NAME() && n == core::MODULE() => {
            // Nested module → collect definitions into a namespace
            let attrs = op.attributes(db);

            if let Some(Attribute::Symbol(sym)) = attrs.get(&ATTR_SYM_NAME()) {
                // Recursively collect definitions from the module's body
                let mut mod_env = ModuleEnv::new();
                for region in op.regions(db).iter() {
                    for block in region.blocks(db).iter() {
                        collect_definitions_from_block(db, &mut mod_env, block);
                    }
                }

                // Add module's exported definitions to parent's namespace
                // TODO: Handle visibility (only pub items should be accessible)
                for (name, binding) in mod_env.definitions.iter() {
                    env.add_to_namespace(*sym, *name, binding.clone());
                }

                // Also add nested namespaces (e.g., mod::submod::item)
                // TODO: Handle nested namespace paths better
                for (nested_ns, nested_bindings) in mod_env.namespaces.iter() {
                    // For now, skip nested namespaces - need better path handling
                    // This would require changing from Symbol to path representation
                    let _ = (nested_ns, nested_bindings);
                }
            }
        }
        _ => {}
    }
}

/// Collect enum constructors from an enum definition.
fn collect_enum_constructors<'db>(
    db: &'db dyn salsa::Database,
    env: &mut ModuleEnv<'db>,
    op: &Operation<'db>,
    type_name: Symbol,
    ty: Type<'db>,
) {
    let attrs = op.attributes(db);

    if let Some(Attribute::List(variants)) = attrs.get(&ATTR_VARIANTS()) {
        for variant in variants.iter() {
            let Attribute::List(parts) = variant else {
                continue;
            };
            let Some(Attribute::Symbol(variant_sym)) = parts.first() else {
                continue;
            };

            env.add_to_namespace(
                type_name,
                *variant_sym,
                Binding::Constructor {
                    ty,
                    tag: Some(*variant_sym),
                    params: IdVec::new(), // TODO: Get variant params
                },
            );
            if env.lookup(*variant_sym).is_none() {
                env.add_constructor(*variant_sym, ty, Some(*variant_sym), IdVec::new());
            }
        }
    }
}

// =============================================================================
// Resolver
// =============================================================================

/// Local binding for function parameters and let bindings.
#[derive(Clone, Debug)]
pub enum LocalBinding<'db> {
    /// A function parameter or block argument.
    Parameter {
        /// The value (block argument)
        value: Value<'db>,
        /// The parameter type
        ty: Type<'db>,
    },
    /// A let-bound variable.
    LetBinding {
        /// The value from the let binding
        value: Value<'db>,
        /// The binding type
        ty: Type<'db>,
    },
    /// A pattern binding (from case arms).
    /// The value is not available during name resolution;
    /// it will be produced by pattern matching at runtime.
    PatternBinding {
        /// The binding type (usually inferred)
        ty: Type<'db>,
    },
}

/// Name resolver context.
///
/// Transforms `src.*` operations into resolved operations (`func.*`, `adt.*`).
pub struct Resolver<'db> {
    db: &'db dyn salsa::Database,
    env: ModuleEnv<'db>,
    /// Rewrite context for value mapping.
    ctx: RewriteContext<'db>,
    /// Import scope stack (for use declarations).
    import_scopes: Vec<HashMap<Symbol, Binding<'db>>>,
    /// Local scope stack (for function parameters, let bindings).
    /// Each entry is a scope level mapping names to local bindings.
    local_scopes: Vec<HashMap<Symbol, LocalBinding<'db>>>,
    /// If true, emit diagnostics for unresolved references instead of passing through.
    report_unresolved: bool,
}

impl<'db> Resolver<'db> {
    /// Create a new resolver with the given environment.
    pub fn new(db: &'db dyn salsa::Database, env: ModuleEnv<'db>) -> Self {
        Self {
            db,
            env,
            ctx: RewriteContext::new(),
            import_scopes: vec![HashMap::new()],
            local_scopes: Vec::new(),
            report_unresolved: false,
        }
    }

    /// Create a resolver that reports unresolved references as errors.
    ///
    /// Use this for the final resolution pass after TDNR.
    pub fn with_unresolved_reporting(db: &'db dyn salsa::Database, env: ModuleEnv<'db>) -> Self {
        Self {
            db,
            env,
            ctx: RewriteContext::new(),
            import_scopes: vec![HashMap::new()],
            local_scopes: Vec::new(),
            report_unresolved: true,
        }
    }

    /// Get the environment.
    pub fn env(&self) -> &ModuleEnv<'db> {
        &self.env
    }

    /// Push a new local scope.
    fn push_scope(&mut self) {
        self.local_scopes.push(HashMap::new());
    }

    /// Pop the current local scope.
    fn pop_scope(&mut self) {
        self.local_scopes.pop();
    }

    /// Push a new import scope.
    fn push_import_scope(&mut self) {
        self.import_scopes.push(HashMap::new());
    }

    /// Pop the current import scope.
    fn pop_import_scope(&mut self) {
        self.import_scopes.pop();
    }

    /// Add an import binding to the current scope.
    fn add_import(&mut self, name: Symbol, binding: Binding<'db>) {
        if let Some(scope) = self.import_scopes.last_mut() {
            scope.insert(name, binding);
        }
    }

    /// Look up a name in import scopes (innermost first).
    fn lookup_import(&self, name: Symbol) -> Option<&Binding<'db>> {
        for scope in self.import_scopes.iter().rev() {
            if let Some(binding) = scope.get(&name) {
                return Some(binding);
            }
        }
        None
    }

    /// Look up a name, checking imports before module definitions.
    fn lookup_binding(&self, name: Symbol) -> Option<&Binding<'db>> {
        self.lookup_import(name).or_else(|| self.env.lookup(name))
    }

    /// Add a local binding to the current scope.
    fn add_local(&mut self, name: Symbol, binding: LocalBinding<'db>) {
        if let Some(scope) = self.local_scopes.last_mut() {
            scope.insert(name, binding);
        }
    }

    /// Mark a src.var operation as a resolved local binding.
    fn mark_resolved_local(&self, op: Operation<'db>) -> Operation<'db> {
        op.modify(self.db)
            .attr(ATTR_RESOLVED_LOCAL(), Attribute::Bool(true))
            .build()
    }

    fn is_marked_resolved_local(&self, op: &Operation<'db>) -> bool {
        matches!(
            op.attributes(self.db).get(&ATTR_RESOLVED_LOCAL()),
            Some(Attribute::Bool(true)) | Some(Attribute::IntBits(1))
        )
    }

    /// Look up a name in local scopes (innermost first).
    fn lookup_local(&self, name: Symbol) -> Option<&LocalBinding<'db>> {
        for scope in self.local_scopes.iter().rev() {
            if let Some(binding) = scope.get(&name) {
                return Some(binding);
            }
        }
        None
    }

    /// Resolve a type.
    ///
    /// Converts `src.type` to concrete types:
    /// - `Int` → `core.i64`
    /// - `Bool` → `core.i1`
    /// - `String` → `core.string`
    /// - User-defined types are looked up in the environment
    fn resolve_type(&self, ty: Type<'db>) -> Type<'db> {
        // Check if this is an unresolved type (src.type)
        if ty.dialect(self.db) == "src" && ty.name(self.db) == "type" {
            // Get the type name from the name attribute (stored as Symbol)
            if let Some(Attribute::Symbol(name_sym)) = ty.get_attr(self.db, src::Type::name_sym()) {
                return self.resolve_type_name(*name_sym);
            }
        }

        // For non-src types, recursively resolve type parameters
        let params = ty.params(self.db);
        if params.is_empty() {
            return ty;
        }

        let new_params: IdVec<Type<'db>> = params.iter().map(|&t| self.resolve_type(t)).collect();
        if new_params.as_slice() == params {
            return ty;
        }

        // Create a new type with resolved parameters, preserving attrs
        Type::new(
            self.db,
            ty.dialect(self.db),
            ty.name(self.db),
            new_params,
            ty.attrs(self.db).clone(),
        )
    }

    /// Resolve a type name to a concrete type.
    fn resolve_type_name(&self, name: Symbol) -> Type<'db> {
        let name_str = name.to_string();
        match &*name_str {
            // Primitive types
            "Int" => *core::I64::new(self.db),
            "Bool" => *core::I1::new(self.db),
            "Float" => *core::F64::new(self.db),
            "Nat" => *core::I64::new(self.db), // Nat is implemented as i64 for now
            "String" => *core::String::new(self.db),
            "Bytes" => *core::Bytes::new(self.db),
            "Nil" => *core::Nil::new(self.db),
            _ => {
                // Look up user-defined types in the environment
                if let Some(binding) = self.lookup_binding(name) {
                    match binding {
                        Binding::TypeDef { ty } => return *ty,
                        Binding::Module {
                            type_def: Some(ty), ..
                        } => return *ty,
                        _ => {}
                    }
                }

                // Leave unresolved - will be caught by type checker
                Type::new(
                    self.db,
                    src::DIALECT_NAME(),
                    src::TYPE(),
                    IdVec::new(),
                    Attrs::new(),
                )
            }
        }
    }

    fn binding_from_path(&self, path: &QualifiedName) -> Option<Binding<'db>> {
        if path.is_simple() {
            return self.env.lookup(path.name()).cloned();
        }

        // For now, only support single-level namespaces (Type::Constructor)
        // TODO: Support multi-level namespaces
        let namespace = *path.as_parent().last()?;
        let name = path.name();
        self.env.lookup_qualified(namespace, name).cloned()
    }

    fn apply_use(&mut self, op: &Operation<'db>) {
        let attrs = op.attributes(self.db);

        let Some(Attribute::QualifiedName(path)) = attrs.get(&ATTR_PATH()) else {
            return;
        };

        let local_name = if let Some(Attribute::Symbol(alias)) = attrs.get(&ATTR_ALIAS()) {
            *alias
        } else {
            path.name()
        };

        // Check if this path represents a namespace
        let namespace_sym = path.name();
        let binding = self.binding_from_path(path);

        if self.env.has_namespace(namespace_sym) {
            let type_def = match binding {
                Some(Binding::TypeDef { ty }) => Some(ty),
                _ => None,
            };
            self.add_import(
                local_name,
                Binding::Module {
                    namespace: path.clone(),
                    type_def,
                },
            );
            return;
        }

        if let Some(binding) = binding {
            self.add_import(local_name, binding);
        }
    }

    /// Resolve names in a module.
    ///
    /// Returns the module with `src.*` operations transformed to resolved forms.
    pub fn resolve_module(&mut self, module: &Module<'db>) -> Module<'db> {
        let body = module.body(self.db);
        self.push_import_scope();
        let new_body = self.resolve_region(&body);
        self.pop_import_scope();

        Module::create(
            self.db,
            module.location(self.db),
            module.name(self.db),
            new_body,
        )
    }

    /// Resolve names in a region.
    fn resolve_region(&mut self, region: &Region<'db>) -> Region<'db> {
        let new_blocks: IdVec<Block<'db>> = region
            .blocks(self.db)
            .iter()
            .map(|block| self.resolve_block(block))
            .collect();

        Region::new(self.db, region.location(self.db), new_blocks)
    }

    /// Resolve names in a block.
    fn resolve_block(&mut self, block: &Block<'db>) -> Block<'db> {
        // Resolve block argument types
        let new_args: IdVec<Type<'db>> = block
            .args(self.db)
            .iter()
            .map(|&ty| self.resolve_type(ty))
            .collect();

        let new_ops: IdVec<Operation<'db>> = block
            .operations(self.db)
            .iter()
            .flat_map(|op| self.resolve_operation(op))
            .collect();

        Block::new(
            self.db,
            block.id(self.db),
            block.location(self.db),
            new_args,
            new_ops,
        )
    }

    /// Resolve a single operation.
    ///
    /// Returns the resolved operation(s). May return empty vec if erased,
    /// or multiple ops if expanded.
    fn resolve_operation(&mut self, op: &Operation<'db>) -> Vec<Operation<'db>> {
        // First, remap operands from previous transformations
        let remapped_op = self.ctx.remap_operands(self.db, op);

        // If operands were remapped, map old results to new results
        if remapped_op != *op {
            self.ctx.map_results(self.db, op, &remapped_op);
        }

        let dialect = remapped_op.dialect(self.db);
        let op_name = remapped_op.name(self.db);

        match (dialect, op_name) {
            (d, n) if d == func::DIALECT_NAME() && n == func::FUNC() => {
                // Handle function with local scope for parameters
                vec![self.resolve_func(&remapped_op)]
            }
            (d, n) if d == src::DIALECT_NAME() && n == src::USE() => {
                self.apply_use(&remapped_op);
                Vec::new()
            }
            (d, n) if d == src::DIALECT_NAME() && n == src::VAR() => {
                if let Some(resolved) = self.try_resolve_var(&remapped_op) {
                    resolved
                } else {
                    // Check if already resolved (has a concrete type, not src.type or type.var)
                    // This happens when re-processing an already-resolved module:
                    // - src.type: Not yet resolved
                    // - type.var: Unresolved, with type variable from inference
                    // - Concrete types (core.*, adt.*): Already resolved local var
                    let is_already_resolved =
                        remapped_op.results(self.db).first().is_some_and(|ty| {
                            // Type is resolved if it's not a source type placeholder and not a type variable
                            let dialect = ty.dialect(self.db);
                            dialect != src::DIALECT_NAME() && dialect != ty::DIALECT_NAME()
                        });

                    let is_resolved_local = self.is_marked_resolved_local(&remapped_op);
                    if self.report_unresolved && !is_already_resolved && !is_resolved_local {
                        self.emit_unresolved_var_diagnostic(&remapped_op);
                    }
                    vec![self.resolve_op_regions(&remapped_op)]
                }
            }
            (d, n) if d == src::DIALECT_NAME() && n == src::PATH() => {
                if let Some(resolved) = self.try_resolve_path(&remapped_op) {
                    vec![resolved]
                } else {
                    // Unresolved
                    if self.report_unresolved {
                        self.emit_unresolved_path_diagnostic(&remapped_op);
                    }
                    vec![self.resolve_op_regions(&remapped_op)]
                }
            }
            (d, n) if d == src::DIALECT_NAME() && n == src::CALL() => {
                if let Some(resolved) = self.try_resolve_call(&remapped_op) {
                    vec![resolved]
                } else {
                    // Unresolved
                    if self.report_unresolved {
                        self.emit_unresolved_call_diagnostic(&remapped_op);
                    }
                    vec![self.resolve_op_regions(&remapped_op)]
                }
            }
            (d, n) if d == src::DIALECT_NAME() && n == src::CONS() => {
                if let Some(resolved) = self.try_resolve_cons(&remapped_op) {
                    vec![resolved]
                } else {
                    if self.report_unresolved {
                        self.emit_unresolved_cons_diagnostic(&remapped_op);
                    }
                    vec![self.resolve_op_regions(&remapped_op)]
                }
            }
            (d, n) if d == core::DIALECT_NAME() && n == core::MODULE() => {
                self.push_import_scope();
                let resolved = self.resolve_op_regions(&remapped_op);
                self.pop_import_scope();
                vec![resolved]
            }
            (d, n) if d == case::DIALECT_NAME() && n == case::ARM() => {
                // Handle case arm with pattern bindings
                vec![self.resolve_case_arm(&remapped_op)]
            }
            _ => {
                // Not a src.* operation - recursively process regions
                vec![self.resolve_op_regions(&remapped_op)]
            }
        }
    }

    /// Resolve a func.func operation with local scope for parameters.
    fn resolve_func(&mut self, op: &Operation<'db>) -> Operation<'db> {
        let regions = op.regions(self.db);
        if regions.is_empty() {
            return *op;
        }

        // Push a new scope for function parameters
        self.push_scope();

        // Process the body region
        let body_region = &regions[0];
        let new_body = self.resolve_func_region(body_region);

        // Pop the function scope
        self.pop_scope();

        // Create new regions vector with resolved body
        let mut new_regions: IdVec<Region<'db>> = IdVec::new();
        new_regions.push(new_body);
        for region in regions.iter().skip(1) {
            new_regions.push(self.resolve_region(region));
        }

        // Resolve the function type attribute
        let attrs = op.attributes(self.db);
        let mut builder = op.modify(self.db).regions(new_regions);

        // Update the type attribute if present
        if let Some(Attribute::Type(func_ty)) = attrs.get(&ATTR_TYPE()) {
            let resolved_ty = self.resolve_func_type(*func_ty);
            builder = builder.attr(ATTR_TYPE(), Attribute::Type(resolved_ty));
        }

        builder.build()
    }

    /// Resolve a function type (params and result).
    fn resolve_func_type(&self, ty: Type<'db>) -> Type<'db> {
        // Check if this is a func type (core.func)
        if let Some(func_ty) = core::Func::from_type(self.db, ty) {
            let params = func_ty.params(self.db);
            let result = func_ty.result(self.db);

            // Resolve all parameter types
            let resolved_params: IdVec<_> = params.iter().map(|p| self.resolve_type(*p)).collect();

            // Resolve result type
            let resolved_result = self.resolve_type(result);

            // Create resolved function type
            *core::Func::new(self.db, resolved_params, resolved_result)
        } else {
            // Not a func type, just resolve it normally
            self.resolve_type(ty)
        }
    }

    /// Resolve a case.arm operation with pattern bindings.
    ///
    /// case.arm has two regions: pattern and body.
    /// Pattern bindings are extracted and added to local scope before resolving body.
    fn resolve_case_arm(&mut self, op: &Operation<'db>) -> Operation<'db> {
        let regions = op.regions(self.db);
        if regions.len() < 2 {
            return *op;
        }

        let pattern_region = &regions[0];
        let body_region = &regions[1];

        // Push a new scope for pattern bindings
        self.push_scope();

        // Extract bindings from pattern region and add to scope
        self.collect_pattern_bindings(pattern_region);

        // Resolve body region with pattern bindings in scope
        let new_body = self.resolve_region(body_region);

        // Pop the pattern scope
        self.pop_scope();

        // Create new regions vector (pattern unchanged, body resolved)
        let mut new_regions: IdVec<Region<'db>> = IdVec::new();
        new_regions.push(*pattern_region);
        new_regions.push(new_body);

        op.modify(self.db).regions(new_regions).build()
    }

    /// Collect pattern bindings from a pattern region.
    ///
    /// Recursively walks the pattern region to find pat.bind, pat.as_pat,
    /// and pat.list_rest operations, adding their bindings to local scope.
    fn collect_pattern_bindings(&mut self, region: &Region<'db>) {
        for block in region.blocks(self.db).iter() {
            for op in block.operations(self.db).iter() {
                self.collect_pattern_binding_from_op(op);
            }
        }
    }

    /// Collect bindings from a single pattern operation.
    fn collect_pattern_binding_from_op(&mut self, op: &Operation<'db>) {
        let dialect = op.dialect(self.db);
        let op_name = op.name(self.db);

        match (dialect, op_name) {
            (d, n) if d == pat::DIALECT_NAME() && n == pat::BIND() => {
                // pat.bind has a "name" attribute
                let attrs = op.attributes(self.db);
                if let Some(Attribute::Symbol(sym)) = attrs.get(&ATTR_NAME()) {
                    // Pattern binding - value comes from pattern matching at runtime
                    let infer_ty =
                        trunk_ir::dialect::ty::var(self.db, std::collections::BTreeMap::new());
                    self.add_local(*sym, LocalBinding::PatternBinding { ty: infer_ty });
                }
            }
            (d, n) if d == pat::DIALECT_NAME() && n == pat::AS_PAT() => {
                // pat.as_pat has a "name" attribute and an inner pattern region
                let attrs = op.attributes(self.db);
                if let Some(Attribute::Symbol(sym)) = attrs.get(&ATTR_NAME()) {
                    let infer_ty =
                        trunk_ir::dialect::ty::var(self.db, std::collections::BTreeMap::new());
                    self.add_local(*sym, LocalBinding::PatternBinding { ty: infer_ty });
                }
                // Also collect from inner pattern region
                for region in op.regions(self.db).iter() {
                    self.collect_pattern_bindings(region);
                }
            }
            (d, n) if d == pat::DIALECT_NAME() && n == pat::LIST_REST() => {
                // pat.list_rest has a "rest_name" attribute
                let attrs = op.attributes(self.db);
                if let Some(Attribute::Symbol(sym)) = attrs.get(&ATTR_REST_NAME()) {
                    let infer_ty =
                        trunk_ir::dialect::ty::var(self.db, std::collections::BTreeMap::new());
                    self.add_local(*sym, LocalBinding::PatternBinding { ty: infer_ty });
                }
                // Also collect from head pattern region
                for region in op.regions(self.db).iter() {
                    self.collect_pattern_bindings(region);
                }
            }
            _ => {
                // Other pattern ops may have nested regions with bindings
                for region in op.regions(self.db).iter() {
                    self.collect_pattern_bindings(region);
                }
            }
        }
    }

    /// Resolve a function body region, handling parameter bindings.
    fn resolve_func_region(&mut self, region: &Region<'db>) -> Region<'db> {
        let blocks = region.blocks(self.db);
        if blocks.is_empty() {
            return *region;
        }

        // Process entry block specially (it has the parameter declarations)
        let entry_block = &blocks[0];
        let new_entry = self.resolve_func_entry_block(entry_block);

        // Process remaining blocks normally
        let mut new_blocks: IdVec<Block<'db>> = IdVec::new();
        new_blocks.push(new_entry);
        for block in blocks.iter().skip(1) {
            new_blocks.push(self.resolve_block(block));
        }

        Region::new(self.db, region.location(self.db), new_blocks)
    }

    /// Resolve a function's entry block, binding parameters.
    ///
    /// The entry block starts with src.var operations that declare parameter names.
    /// These are mapped to block arguments and erased from output.
    fn resolve_func_entry_block(&mut self, block: &Block<'db>) -> Block<'db> {
        let block_args = block.args(self.db);
        let operations = block.operations(self.db);

        // Scan for initial src.var operations that declare parameters
        // Only scan up to the number of block arguments
        // Parameter declarations have the function's overall span (from lower_function)
        let func_span = block.location(self.db).span;
        let mut param_declarations = Vec::new();
        for op in operations.iter() {
            // Stop early if we've found all expected parameters
            if param_declarations.len() >= block_args.len() {
                break;
            }

            if op.dialect(self.db) == "src" && op.name(self.db) == "var" {
                // Only consider as parameter declaration if span matches function span
                // Body references have their own specific span, not the function span
                let op_span = op.location(self.db).span;
                if op_span != func_span {
                    break; // Different span means it's a body reference, not a param decl
                }

                // This is a parameter declaration
                let attrs = op.attributes(self.db);
                if let Some(Attribute::Symbol(sym)) = attrs.get(&ATTR_NAME()) {
                    param_declarations.push((*sym, *op));
                } else {
                    break; // Not a proper src.var, stop scanning
                }
            } else {
                break; // No more parameter declarations
            }
        }

        // Map parameter names to block argument values
        for (i, (name, op)) in param_declarations.iter().enumerate() {
            if i < block_args.len() {
                // Create block argument value
                let block_arg = Value::new(self.db, ValueDef::BlockArg(block.id(self.db)), i);
                let param_ty = block_args[i];

                // Add to local scope
                self.add_local(
                    *name,
                    LocalBinding::Parameter {
                        value: block_arg,
                        ty: param_ty,
                    },
                );

                // Map the src.var result to the block argument
                let old_result = op.result(self.db, 0);
                self.ctx.map_value(old_result, block_arg);
            }
        }

        // Resolve block argument types
        let new_args: IdVec<Type<'db>> =
            block_args.iter().map(|&ty| self.resolve_type(ty)).collect();

        // Now resolve the block, skipping parameter declaration src.var ops
        let num_param_decls = param_declarations.len();

        let new_ops: IdVec<Operation<'db>> = operations
            .iter()
            .enumerate()
            .filter(|(i, _)| *i >= num_param_decls) // Skip parameter declarations
            .flat_map(|(_, op)| self.resolve_operation(op))
            .collect();

        Block::new(
            self.db,
            block.id(self.db),
            block.location(self.db),
            new_args,
            new_ops,
        )
    }

    /// Recursively resolve regions and types within an operation.
    fn resolve_op_regions(&mut self, op: &Operation<'db>) -> Operation<'db> {
        // Resolve result types
        let results = op.results(self.db);
        let new_results: IdVec<Type<'db>> =
            results.iter().map(|&ty| self.resolve_type(ty)).collect();
        let results_changed = new_results.as_slice() != results.as_slice();

        // Resolve nested regions
        let regions = op.regions(self.db);
        let new_regions: IdVec<Region<'db>> = regions
            .iter()
            .map(|region| self.resolve_region(region))
            .collect();
        let regions_changed = !regions.is_empty();

        if !results_changed && !regions_changed {
            return *op;
        }

        let new_op = op
            .modify(self.db)
            .results(new_results)
            .regions(new_regions)
            .build();

        // Map old results to new results so subsequent operations can find them
        self.ctx.map_results(self.db, op, &new_op);

        new_op
    }

    /// Try to resolve a `src.var` operation.
    ///
    /// Returns:
    /// - Some(vec![]) if resolved to a local binding (erased, value already mapped)
    /// - Some(vec![op]) if resolved to a function/constructor (replaced)
    /// - None if unresolved
    fn try_resolve_var(&mut self, op: &Operation<'db>) -> Option<Vec<Operation<'db>>> {
        let attrs = op.attributes(self.db);
        let Attribute::Symbol(sym) = attrs.get(&ATTR_NAME())? else {
            return None;
        };
        let name = *sym;
        let location = op.location(self.db);

        // Special case: Nil is the unit value (built-in)
        if name == Symbol::new("Nil") {
            let nil_ty = core::Nil::new(self.db).as_type();
            // Create a unit value constant - arith.const with nil type produces no runtime value
            let const_op = arith::r#const(self.db, location, nil_ty, Attribute::Unit);
            let new_operation = const_op.as_operation();

            // Map old result to new result
            let old_result = op.result(self.db, 0);
            let new_result = new_operation.result(self.db, 0);
            self.ctx.map_value(old_result, new_result);

            return Some(vec![new_operation]);
        }

        // First, check local scopes (function parameters, let bindings, pattern bindings)
        if let Some(local) = self.lookup_local(name) {
            match local {
                LocalBinding::Parameter { value, ty } | LocalBinding::LetBinding { value, ty } => {
                    // Local binding found - keep src.var with resolved type for hover span
                    let resolved_ty = self.resolve_type(*ty);

                    // Create new src.var with resolved type (keeps span for hover)
                    let new_op = src::var(self.db, location, resolved_ty, *sym);
                    let new_operation = self.mark_resolved_local(new_op.as_operation());

                    // Map old result to the actual bound value (not the new src.var's result)
                    // This ensures use sites get the correct value
                    let old_result = op.result(self.db, 0);
                    self.ctx.map_value(old_result, *value);

                    // Return the new src.var to keep it in IR for hover
                    return Some(vec![new_operation]);
                }
                LocalBinding::PatternBinding { ty } => {
                    // Pattern binding - use case.bind to extract value from pattern matching
                    // This produces proper SSA form: case.bind result is the pattern-bound value
                    let resolved_ty = self.resolve_type(*ty);
                    let bind_op = case::bind(self.db, location, resolved_ty, *sym);

                    // Remap old result to case.bind result (proper SSA value)
                    let old_result = op.result(self.db, 0);
                    let new_result = bind_op.as_operation().result(self.db, 0);
                    self.ctx.map_value(old_result, new_result);

                    return Some(vec![bind_op.as_operation()]);
                }
            }
        }

        // Then check module environment
        match self.lookup_binding(name)? {
            Binding::Function { path, ty } => {
                // Create func.constant operation
                // func::constant(db, location, result_type, func_ref)
                let new_op = func::constant(self.db, location, *ty, path.clone());
                let new_operation = new_op.as_operation();

                // Map old result to new result
                let old_result = op.result(self.db, 0);
                let new_result = new_operation.result(self.db, 0);
                self.ctx.map_value(old_result, new_result);

                Some(vec![new_operation])
            }
            Binding::Constructor { ty, tag, .. } => {
                // Create adt.struct_new or adt.variant_new
                // No args here since src.var is just a reference (not a call)
                let new_operation = if let Some(tag) = tag {
                    // Enum variant constructor (with tag)
                    // variant_new(db, location, fields, result_type, ty, tag)
                    adt::variant_new(self.db, location, vec![], *ty, *ty, *tag).as_operation()
                } else {
                    // Struct constructor (no tag)
                    // struct_new(db, location, fields, result_type, ty)
                    adt::struct_new(self.db, location, vec![], *ty, *ty).as_operation()
                };

                // Map old result to new result
                let old_result = op.result(self.db, 0);
                let new_result = new_operation.result(self.db, 0);
                self.ctx.map_value(old_result, new_result);

                Some(vec![new_operation])
            }
            Binding::Const { value, ty, .. } => {
                // Mark as resolved const reference (inlining happens in a separate pass)
                let resolved_ty = self.resolve_type(*ty);

                // Keep src.var but mark it as a resolved const reference
                // Store the const value in an attribute for the inlining pass
                let new_op = op
                    .modify(self.db)
                    .results(std::iter::once(resolved_ty).collect())
                    .attr(ATTR_RESOLVED_CONST(), Attribute::Bool(true))
                    .attr(ATTR_VALUE(), value.clone())
                    .build();

                // Map old result to new result
                let old_result = op.result(self.db, 0);
                let new_result = new_op.result(self.db, 0);
                self.ctx.map_value(old_result, new_result);

                Some(vec![new_op])
            }
            Binding::TypeDef { .. } | Binding::Module { .. } => {
                // Type used in value position - error (leave unresolved for diagnostics)
                None
            }
        }
    }

    /// Try to resolve a `src.path` operation.
    fn try_resolve_path(&mut self, op: &Operation<'db>) -> Option<Operation<'db>> {
        let attrs = op.attributes(self.db);
        let Attribute::QualifiedName(path) = attrs.get(&ATTR_PATH())? else {
            return None;
        };

        if path.len() < 2 {
            return None;
        }

        let location = op.location(self.db);

        // For now, only support single-level qualified names (Type::Constructor)
        // TODO: Support multi-level paths
        let namespace = *path.as_parent().last()?;
        let name = path.name();

        match self.env.lookup_qualified(namespace, name)? {
            Binding::Function { path, ty } => {
                let new_op = func::constant(self.db, location, *ty, path.clone());
                let new_operation = new_op.as_operation();

                let old_result = op.result(self.db, 0);
                let new_result = new_operation.result(self.db, 0);
                self.ctx.map_value(old_result, new_result);

                Some(new_operation)
            }
            Binding::Constructor { ty, tag, .. } => {
                let new_operation = if let Some(tag) = tag {
                    // variant_new(db, location, fields, result_type, ty, tag)
                    adt::variant_new(self.db, location, vec![], *ty, *ty, *tag).as_operation()
                } else {
                    // struct_new(db, location, fields, result_type, ty)
                    adt::struct_new(self.db, location, vec![], *ty, *ty).as_operation()
                };

                let old_result = op.result(self.db, 0);
                let new_result = new_operation.result(self.db, 0);
                self.ctx.map_value(old_result, new_result);

                Some(new_operation)
            }
            Binding::Const { .. } => {
                // Qualified const reference (e.g., Module::CONST)
                // Not commonly used, but handle similarly to unqualified const
                // For now, leave unresolved (could be implemented if needed)
                None
            }
            Binding::TypeDef { .. } | Binding::Module { .. } => None,
        }
    }

    /// Try to resolve a `src.call` operation.
    fn try_resolve_call(&mut self, op: &Operation<'db>) -> Option<Operation<'db>> {
        let attrs = op.attributes(self.db);
        let Attribute::QualifiedName(path) = attrs.get(&ATTR_NAME())? else {
            return None;
        };

        let location = op.location(self.db);
        let result_ty = op.results(self.db).first().copied()?;
        let args: Vec<Value<'db>> = op.operands(self.db).iter().copied().collect();

        // Try to resolve the callee
        let binding = if path.is_simple() {
            let name = path.name();
            if let Some(local) = self.lookup_local(name) {
                let callee = match local {
                    LocalBinding::Parameter { value, .. }
                    | LocalBinding::LetBinding { value, .. } => *value,
                    LocalBinding::PatternBinding { .. } => return None,
                };
                let new_op = func::call_indirect(self.db, location, callee, args, result_ty);
                let new_operation = new_op.as_operation();

                let old_result = op.result(self.db, 0);
                let new_result = new_operation.result(self.db, 0);
                self.ctx.map_value(old_result, new_result);

                return Some(new_operation);
            }
            self.lookup_binding(name)
        } else {
            // For now, only support single-level qualified calls (Type::method)
            // TODO: Support multi-level paths
            if path.len() != 2 {
                return None;
            }
            let namespace = *path.as_parent().last().unwrap();
            let name = path.name();
            self.env.lookup_qualified(namespace, name)
        }?;

        match binding {
            Binding::Function { path, ty: func_ty } => {
                // Direct function call
                // Use the callee function's return type instead of the src.call's type variable
                // Also resolve the return type (it may be src.type that needs resolution)
                let call_result_ty = core::Func::from_type(self.db, *func_ty)
                    .map(|f| self.resolve_type(f.result(self.db)))
                    .unwrap_or(result_ty);
                // func::call(db, location, args, result_type, callee)
                let new_op = func::call(self.db, location, args, call_result_ty, path.clone());
                let new_operation = new_op.as_operation();

                let old_result = op.result(self.db, 0);
                let new_result = new_operation.result(self.db, 0);
                self.ctx.map_value(old_result, new_result);

                Some(new_operation)
            }
            Binding::Constructor { .. }
            | Binding::TypeDef { .. }
            | Binding::Module { .. }
            | Binding::Const { .. } => {
                // Const cannot be called as a function
                None
            }
        }
    }

    /// Try to resolve a `src.cons` operation.
    fn try_resolve_cons(&mut self, op: &Operation<'db>) -> Option<Operation<'db>> {
        let attrs = op.attributes(self.db);
        let Attribute::QualifiedName(path) = attrs.get(&ATTR_NAME())? else {
            return None;
        };

        let location = op.location(self.db);
        let args: Vec<Value<'db>> = op.operands(self.db).iter().copied().collect();

        let binding = if path.is_simple() {
            let name = path.name();
            self.lookup_binding(name)
        } else {
            // For now, only support single-level qualified constructors (Type::Variant)
            if path.len() != 2 {
                return None;
            }
            let namespace = *path.as_parent().last().unwrap();
            let name = path.name();
            self.env.lookup_qualified(namespace, name)
        }?;

        match binding {
            Binding::Constructor { ty, tag, .. } => {
                let new_operation = if let Some(tag) = tag {
                    adt::variant_new(self.db, location, args, *ty, *ty, *tag).as_operation()
                } else {
                    adt::struct_new(self.db, location, args, *ty, *ty).as_operation()
                };

                let old_result = op.result(self.db, 0);
                let new_result = new_operation.result(self.db, 0);
                self.ctx.map_value(old_result, new_result);

                Some(new_operation)
            }
            _ => None,
        }
    }

    // === Diagnostic Helpers ===

    /// Emit diagnostic for unresolved `src.var`.
    fn emit_unresolved_var_diagnostic(&self, op: &Operation<'db>) {
        let name = op
            .attributes(self.db)
            .get(&ATTR_NAME())
            .and_then(|a| {
                if let Attribute::Symbol(s) = a {
                    Some(s.to_string())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| "unknown".to_string());

        Diagnostic {
            message: format!("unresolved name: `{}`", name),
            span: op.location(self.db).span,
            severity: DiagnosticSeverity::Error,
            phase: CompilationPhase::NameResolution,
        }
        .accumulate(self.db);
    }

    /// Emit diagnostic for unresolved `src.path`.
    fn emit_unresolved_path_diagnostic(&self, op: &Operation<'db>) {
        let path = op
            .attributes(self.db)
            .get(&ATTR_PATH())
            .and_then(|a| {
                if let Attribute::QualifiedName(qual_name) = a {
                    Some(qual_name.to_string())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| "unknown".to_string());

        Diagnostic {
            message: format!("unresolved path: `{}`", path),
            span: op.location(self.db).span,
            severity: DiagnosticSeverity::Error,
            phase: CompilationPhase::NameResolution,
        }
        .accumulate(self.db);
    }

    /// Emit diagnostic for unresolved `src.call`.
    fn emit_unresolved_call_diagnostic(&self, op: &Operation<'db>) {
        let name = op
            .attributes(self.db)
            .get(&ATTR_NAME())
            .and_then(|a| {
                if let Attribute::QualifiedName(qual_name) = a {
                    Some(qual_name.to_string())
                } else if let Attribute::Symbol(s) = a {
                    Some(s.to_string())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| "unknown".to_string());

        Diagnostic {
            message: format!("unresolved function call: `{}`", name),
            span: op.location(self.db).span,
            severity: DiagnosticSeverity::Error,
            phase: CompilationPhase::NameResolution,
        }
        .accumulate(self.db);
    }

    /// Emit diagnostic for unresolved `src.cons`.
    fn emit_unresolved_cons_diagnostic(&self, op: &Operation<'db>) {
        let name = op
            .attributes(self.db)
            .get(&ATTR_NAME())
            .and_then(|a| {
                if let Attribute::QualifiedName(qual_name) = a {
                    Some(qual_name.to_string())
                } else if let Attribute::Symbol(s) = a {
                    Some(s.to_string())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| "unknown".to_string());

        Diagnostic {
            message: format!("unresolved constructor: `{}`", name),
            span: op.location(self.db).span,
            severity: DiagnosticSeverity::Error,
            phase: CompilationPhase::NameResolution,
        }
        .accumulate(self.db);
    }
}

// =============================================================================
// Pipeline Integration
// =============================================================================

/// Resolve names in a module (non-tracked version for internal use).
///
/// The tracked version is in pipeline.rs (stage_resolve).
pub fn resolve_module<'db>(db: &'db dyn salsa::Database, module: &Module<'db>) -> Module<'db> {
    // Build environment from module
    let env = build_env(db, module);

    // Create resolver and transform module
    let mut resolver = Resolver::new(db, env);
    resolver.resolve_module(module)
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::dialect::{arith, core, func, src};
    use trunk_ir::{Location, PathId, QualifiedName, Span, idvec};

    fn test_location<'db>(db: &'db dyn salsa::Database) -> Location<'db> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    fn simple_func<'db>(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        name: &str,
    ) -> func::Func<'db> {
        func::Func::build(db, location, name, idvec![], *core::I64::new(db), |entry| {
            let value = entry.op(arith::Const::i64(db, location, 42));
            entry.op(func::Return::value(db, location, value.result(db)));
        })
    }

    fn module_with_use_call<'db>(db: &'db dyn salsa::Database, alias: Option<&str>) -> Module<'db> {
        let location = test_location(db);
        let helpers = core::Module::build(db, location, Symbol::new("helpers"), |inner| {
            inner.op(simple_func(db, location, "double"));
        });
        let path = QualifiedName::from_strs(["helpers", "double"]).unwrap();
        let alias_sym = Symbol::from_dynamic(alias.unwrap_or(""));

        let name = alias.unwrap_or("double");
        let call_path = QualifiedName::simple(Symbol::from_dynamic(name));
        let arg = arith::Const::i64(db, location, 1);
        let call_result_ty = src::unresolved_type(db, Symbol::new("Int"), idvec![]);
        let call = src::call(
            db,
            location,
            vec![arg.result(db)],
            call_result_ty,
            call_path,
        );

        let main_func = func::Func::build(
            db,
            location,
            "main",
            idvec![],
            *core::I64::new(db),
            |entry| {
                entry.op(arg);
                let op = entry.op(call);
                entry.op(func::Return::value(db, location, op.result(db)));
            },
        );

        core::Module::build(db, location, Symbol::new("main"), |top| {
            top.op(helpers);
            top.op(src::r#use(db, location, path, alias_sym, false));
            top.op(main_func);
        })
    }

    #[salsa::tracked]
    fn module_with_hello(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        core::Module::build(db, location, Symbol::new("main"), |top| {
            top.op(simple_func(db, location, "hello"));
        })
    }

    #[salsa::tracked]
    fn module_with_nested_math(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let math_module = core::Module::build(db, location, Symbol::new("math"), |inner| {
            inner.op(simple_func(db, location, "add"));
            inner.op(simple_func(db, location, "sub"));
        });
        core::Module::build(db, location, Symbol::new("main"), |top| {
            top.op(math_module);
        })
    }

    #[salsa::tracked]
    fn module_with_outer_inner(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let inner_mod = core::Module::build(db, location, Symbol::new("inner"), |inner| {
            inner.op(simple_func(db, location, "deep"));
        });
        let outer = core::Module::build(db, location, Symbol::new("outer"), |outer| {
            outer.op(inner_mod);
        });
        core::Module::build(db, location, Symbol::new("main"), |top| {
            top.op(outer);
        })
    }

    #[salsa::tracked]
    fn resolve_use_call_module(db: &dyn salsa::Database) -> Module<'_> {
        let module = module_with_use_call(db, None);
        let env = build_env(db, &module);
        let mut resolver = Resolver::new(db, env);
        resolver.resolve_module(&module)
    }

    #[salsa::tracked]
    fn resolve_use_alias_module(db: &dyn salsa::Database) -> Module<'_> {
        let module = module_with_use_call(db, Some("dbl"));
        let env = build_env(db, &module);
        let mut resolver = Resolver::new(db, env);
        resolver.resolve_module(&module)
    }

    fn collect_ops<'db>(
        db: &'db dyn salsa::Database,
        region: &Region<'db>,
        out: &mut Vec<Operation<'db>>,
    ) {
        for block in region.blocks(db).iter() {
            for op in block.operations(db).iter() {
                out.push(*op);
                for sub_region in op.regions(db).iter() {
                    collect_ops(db, sub_region, out);
                }
            }
        }
    }

    fn has_src_call_named<'db>(
        db: &'db dyn salsa::Database,
        ops: &[Operation<'db>],
        name: &str,
    ) -> bool {
        let name_sym = Symbol::from_dynamic(name);
        ops.iter().any(|op| {
            op.dialect(db) == "src"
                && op.name(db) == "call"
                && matches!(
                    op.attributes(db).get(&ATTR_NAME()),
                    Some(Attribute::QualifiedName(qual_name)) if qual_name.name() == name_sym
                )
        })
    }

    fn has_func_call_named<'db>(
        db: &'db dyn salsa::Database,
        ops: &[Operation<'db>],
        name: &str,
    ) -> bool {
        let name_sym = Symbol::from_dynamic(name);
        ops.iter().any(|op| {
            op.dialect(db) == "func"
                && op.name(db) == "call"
                && matches!(
                    op.attributes(db).get(&ATTR_CALLEE()),
                    Some(Attribute::QualifiedName(qual_name)) if qual_name.name() == name_sym
                )
        })
    }

    #[test]
    fn test_module_env_lookup() {
        // Basic smoke test for ModuleEnv
        let env: ModuleEnv<'_> = ModuleEnv::new();
        assert!(env.lookup(Symbol::new("foo")).is_none());
    }

    #[salsa_test]
    fn test_build_env_from_module(db: &salsa::DatabaseImpl) {
        let module = module_with_hello(db);
        let env = build_env(db, &module);

        // Should find the 'hello' function
        assert!(env.lookup(Symbol::new("hello")).is_some());
        match env.lookup(Symbol::new("hello")) {
            Some(Binding::Function { path, .. }) => {
                assert!(path.is_simple());
            }
            _ => panic!("Expected function binding"),
        }
    }

    #[salsa_test]
    fn test_nested_module_resolution(db: &salsa::DatabaseImpl) {
        let module = module_with_nested_math(db);
        let env = build_env(db, &module);

        // Should find math::add and math::sub
        assert!(
            env.lookup_qualified(Symbol::new("math"), Symbol::new("add"))
                .is_some(),
            "should find math::add"
        );
        assert!(
            env.lookup_qualified(Symbol::new("math"), Symbol::new("sub"))
                .is_some(),
            "should find math::sub"
        );

        // Should not find 'add' at top level
        assert!(
            env.lookup(Symbol::new("add")).is_none(),
            "add should not be at top level"
        );
    }

    #[salsa_test]
    #[ignore = "TODO: Implement multi-level namespace support"]
    fn test_deeply_nested_module_resolution(db: &salsa::DatabaseImpl) {
        let module = module_with_outer_inner(db);
        let env = build_env(db, &module);

        // Should find outer::inner::deep
        assert!(
            env.lookup_qualified(Symbol::new("outer::inner"), Symbol::new("deep"))
                .is_some(),
            "should find outer::inner::deep"
        );

        // Should find outer::inner as a module
        // (inner is in outer's namespace)
        // Note: We don't track modules as bindings yet, so this tests the qualified path
    }

    #[salsa_test]
    #[ignore = "TODO: Fix use import resolution"]
    fn test_use_import_resolves_call(db: &salsa::DatabaseImpl) {
        let module = resolve_use_call_module(db);

        let mut ops = Vec::new();
        collect_ops(db, &module.body(db), &mut ops);

        assert!(
            !has_src_call_named(db, &ops, "double"),
            "use import should resolve src.call to func.call"
        );
        assert!(
            has_func_call_named(db, &ops, "double"),
            "expected func.call to helpers::double"
        );
    }

    #[salsa_test]
    fn test_use_alias_resolves_call(db: &salsa::DatabaseImpl) {
        let module = resolve_use_alias_module(db);

        let mut ops = Vec::new();
        collect_ops(db, &module.body(db), &mut ops);

        assert!(
            !has_src_call_named(db, &ops, "dbl"),
            "use alias should resolve src.call to func.call"
        );
        assert!(
            has_func_call_named(db, &ops, "double"),
            "expected func.call to helpers::double"
        );
    }

    #[salsa::tracked]
    fn module_with_const_definition(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let max_size_sym = Symbol::new("MAX_SIZE");
        let int_ty = *core::I64::new(db);

        core::Module::build(db, location, Symbol::new("main"), |top| {
            let const_op =
                src::r#const(db, location, int_ty, max_size_sym, Attribute::IntBits(1024));
            top.op(const_op);
        })
    }

    #[salsa::tracked]
    pub fn module_with_const_reference(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let max_size_sym = Symbol::new("MAX_SIZE");
        let int_ty = *core::I64::new(db);

        core::Module::build(db, location, Symbol::new("main"), |top| {
            let const_op =
                src::r#const(db, location, int_ty, max_size_sym, Attribute::IntBits(1024));
            top.op(const_op);

            let func_op = func::Func::build(db, location, "test", idvec![], int_ty, |entry| {
                let const_ref = entry.op(src::var(db, location, int_ty, max_size_sym));
                entry.op(func::Return::value(db, location, const_ref.result(db)));
            });
            top.op(func_op);
        })
    }

    #[salsa::tracked]
    pub fn resolve_const_reference_module(db: &dyn salsa::Database) -> Module<'_> {
        let module = module_with_const_reference(db);
        resolve_module(db, &module)
    }

    #[salsa_test]
    fn test_const_definition_collected(db: &salsa::DatabaseImpl) {
        let module = module_with_const_definition(db);
        let env = build_env(db, &module);
        let max_size_sym = Symbol::new("MAX_SIZE");
        let binding = env.lookup(max_size_sym);

        assert!(binding.is_some(), "const MAX_SIZE should be in environment");
        match binding {
            Some(Binding::Const { name, value, ty }) => {
                assert_eq!(*name, max_size_sym, "const name should match");
                assert_eq!(
                    ty.dialect(db),
                    Symbol::new("core"),
                    "const type should be concrete"
                );
                assert_eq!(
                    *value,
                    Attribute::IntBits(1024),
                    "const value should be 1024"
                );
            }
            _ => panic!("Expected Const binding, got {:?}", binding),
        }
    }

    #[salsa_test]
    fn test_const_reference_resolved(db: &salsa::DatabaseImpl) {
        let resolved = resolve_const_reference_module(db);

        let mut ops = Vec::new();
        collect_ops(db, &resolved.body(db), &mut ops);

        let max_size_sym = Symbol::new("MAX_SIZE");
        let const_refs: Vec<_> = ops
            .iter()
            .filter(|op| {
                op.dialect(db) == "src"
                    && op.name(db) == "var"
                    && matches!(
                        op.attributes(db).get(&ATTR_NAME()),
                        Some(Attribute::Symbol(sym)) if *sym == max_size_sym
                    )
            })
            .collect();

        assert!(!const_refs.is_empty(), "should find const reference");

        for const_ref in const_refs {
            let attrs = const_ref.attributes(db);
            assert_eq!(
                attrs.get(&ATTR_RESOLVED_CONST()),
                Some(&Attribute::Bool(true)),
                "const reference should be marked with resolved_const"
            );
            assert_eq!(
                attrs.get(&ATTR_VALUE()),
                Some(&Attribute::IntBits(1024)),
                "const reference should have value attribute for inlining pass"
            );
        }
    }
}
