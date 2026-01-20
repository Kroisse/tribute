//! Name resolution pass for Tribute.
//!
//! This pass resolves `tribute.*` operations to their concrete targets:
//! - `tribute.var` → function reference, constructor, or deferred (UFCS)
//! - `tribute.path` → qualified function/constructor reference
//! - `tribute.call` → resolved function call
//! - `tribute.type` → concrete type
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
use tribute_ir::ModulePathExt;
use tribute_ir::dialect::{ability, tribute, tribute_pat, tribute_rt};
use trunk_ir::dialect::adt;
use trunk_ir::dialect::arith;
use trunk_ir::dialect::core::{self, AbilityRefType, Module};
use trunk_ir::dialect::func;
use trunk_ir::rewrite::RewriteContext;
use trunk_ir::{
    Attribute, Attrs, Block, BlockArg, DialectOp, DialectType, IdVec, Operation, Region, Symbol,
    Type, Value, ValueDef,
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
        path: Symbol,
        /// Function type
        ty: Type<'db>,
    },
    /// A module/namespace binding (possibly with an associated type).
    Module {
        /// Fully qualified namespace path (e.g., "collections::List")
        namespace: Symbol,
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
        /// Field names (for named structs/variants, None for positional)
        field_names: Option<Vec<Symbol>>,
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
    /// An ability operation (e.g., State::get, Console::print).
    AbilityOp {
        /// The ability this operation belongs to as a `core.ability_ref` type.
        /// This supports parameterized abilities like `State(Int)`.
        ability: Type<'db>,
        /// The operation name (e.g., "get")
        op_name: Symbol,
        /// Parameter types
        params: IdVec<Type<'db>>,
        /// Return type
        return_ty: Type<'db>,
    },
}

/// Module environment for name resolution.
///
/// Tracks all names visible in the current module.
#[derive(Debug, Default)]
pub struct ModuleEnv<'db> {
    /// Names defined in this module (qualified path → binding).
    definitions: HashMap<Symbol, Binding<'db>>,
    /// Names imported via `use` statements (simple name → binding).
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
    pub fn add_function(&mut self, path: Symbol, ty: Type<'db>) {
        self.definitions
            .insert(path, Binding::Function { path, ty });
    }

    /// Add a type constructor.
    pub fn add_constructor(
        &mut self,
        name: Symbol,
        ty: Type<'db>,
        tag: Option<Symbol>,
        params: IdVec<Type<'db>>,
        field_names: Option<Vec<Symbol>>,
    ) {
        self.definitions.insert(
            name,
            Binding::Constructor {
                ty,
                tag,
                params,
                field_names,
            },
        );
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

    /// Add an ability operation to the environment.
    ///
    /// Ability operations are added to the ability's namespace (e.g., `State::get`).
    ///
    /// - `ability_name`: The ability name symbol (e.g., `State`)
    /// - `ability_ty`: The ability type as `core.ability_ref` (e.g., `AbilityRefType::simple(db, "State")`)
    /// - `op_name`: The operation name (e.g., `get`)
    pub fn add_ability_op(
        &mut self,
        ability_name: Symbol,
        ability_ty: Type<'db>,
        op_name: Symbol,
        params: IdVec<Type<'db>>,
        return_ty: Type<'db>,
    ) {
        let binding = Binding::AbilityOp {
            ability: ability_ty,
            op_name,
            params,
            return_ty,
        };
        // Add to the ability's namespace for qualified lookup (State::get)
        self.add_to_namespace(ability_name, op_name, binding);
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
        // First check local definitions (simple name)
        if let Some(b) = self.definitions.get(&name) {
            return Some(b);
        }
        // Then check imports
        self.imports.get(&name)
    }

    /// Look up a qualified path (e.g., "List::map").
    pub fn lookup_qualified(&self, namespace: Symbol, name: Symbol) -> Option<&Binding<'db>> {
        // Fall back to namespace lookup (for enum variants, etc.)
        self.namespaces.get(&namespace)?.get(&name)
    }

    /// Look up by full qualified name path.
    pub fn lookup_path(&self, path: Symbol) -> Option<&Binding<'db>> {
        self.definitions.get(&path)
    }

    /// Check whether a namespace exists (e.g., "collections::List").
    pub fn has_namespace(&self, namespace: Symbol) -> bool {
        self.namespaces.contains_key(&namespace)
    }

    /// Iterate over all definitions (for type-directed resolution).
    pub fn definitions_iter(&self) -> impl Iterator<Item = (&Symbol, &Binding<'db>)> {
        self.definitions.iter()
    }

    /// Get namespace contents for debugging.
    pub fn get_namespace(&self, namespace: Symbol) -> Option<&HashMap<Symbol, Binding<'db>>> {
        self.namespaces.get(&namespace)
    }

    /// Iterate over all namespaces.
    pub fn namespaces_iter(
        &self,
    ) -> impl Iterator<Item = (&Symbol, &HashMap<Symbol, Binding<'db>>)> {
        self.namespaces.iter()
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
                env.add_function(*sym, *ty);
            }
        }
        (d, n) if d == tribute::DIALECT_NAME() && n == tribute::STRUCT_DEF() => {
            // Struct definition → creates constructor
            if let Ok(struct_def) = tribute::StructDef::from_operation(db, *op) {
                let sym = struct_def.sym_name(db);
                // Use the result type directly (adt.typeref from tirgen)
                let ty = op
                    .results(db)
                    .first()
                    .copied()
                    .unwrap_or_else(|| adt::typeref(db, sym));
                // Add type definition
                env.add_type(sym, ty);

                // Collect field info from fields region
                let (field_names, field_types) = collect_struct_fields(db, struct_def);
                let params: IdVec<Type<'db>> = field_types.into_iter().collect();

                // Struct constructor has same name as type
                env.add_constructor(sym, ty, None, params, Some(field_names));
            }
        }
        (d, n) if d == tribute::DIALECT_NAME() && n == tribute::ENUM_DEF() => {
            // Enum definition → creates constructors for each variant
            let attrs = op.attributes(db);
            if let Some(Attribute::Symbol(sym)) = attrs.get(&ATTR_SYM_NAME()) {
                // Use the result type directly (adt.typeref from tirgen)
                let ty = op
                    .results(db)
                    .first()
                    .copied()
                    .unwrap_or_else(|| adt::typeref(db, *sym));
                // Add type definition
                env.add_type(*sym, ty);
                // Extract variants from the operation's regions or attributes
                // Each variant becomes a constructor in the type's namespace
                collect_enum_constructors(db, env, op, *sym, ty);
            }
        }
        (d, n) if d == tribute::DIALECT_NAME() && n == tribute::ABILITY_DEF() => {
            // Ability definition → creates operations in the ability's namespace
            if let Ok(ability_decl) = tribute::AbilityDef::from_operation(db, *op) {
                let ability_name = ability_decl.sym_name(db);
                // Also register as a Module binding for the namespace
                env.definitions.insert(
                    ability_name,
                    Binding::Module {
                        namespace: ability_name,
                        type_def: None,
                    },
                );

                // Extract operations from the ability
                collect_ability_operations(db, env, ability_decl, ability_name);
            }
        }
        (d, n) if d == tribute::DIALECT_NAME() && n == tribute::CONST() => {
            // Const definition
            let attrs = op.attributes(db);

            if let (Some(Attribute::Symbol(sym)), Some(value)) =
                (attrs.get(&ATTR_NAME()), attrs.get(&ATTR_VALUE()))
            {
                // Get the type from the operation result
                let ty = op.results(db).first().copied().unwrap_or_else(|| {
                    tribute::new_type_var(db, std::collections::BTreeMap::new())
                });
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

                // Add module's exported definitions to both:
                // 1. Parent's namespace (for lookup_qualified)
                // 2. Parent's definitions with full path (for lookup_path)
                // TODO: Handle visibility (only pub items should be accessible)
                for (def_sym, binding) in mod_env.definitions.iter() {
                    // Add to namespace for backward compatibility
                    env.add_to_namespace(*sym, def_sym.last_segment(), binding.clone());
                    // Add to definitions with the full qualified path (module::name)
                    let full_path = sym.join_path(*def_sym);
                    env.definitions.insert(full_path, binding.clone());
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

/// Resolve a primitive type name to its concrete type.
///
/// This function resolves `tribute.type(name=X)` for well-known primitive types
/// without requiring access to the full module environment. User-defined types
/// are left unresolved and will be handled during the full resolve pass.
///
/// This is used during environment building (e.g., collecting enum constructors)
/// where the environment is not yet complete.
fn resolve_primitive_type<'db>(db: &'db dyn salsa::Database, ty: Type<'db>) -> Type<'db> {
    // Check if this is an unresolved type (tribute.type)
    if ty.dialect(db) != tribute::DIALECT_NAME() || ty.name(db) != tribute::TYPE() {
        // Not an unresolved type - recursively resolve type parameters
        let params = ty.params(db);
        if params.is_empty() {
            return ty;
        }

        let new_params: IdVec<Type<'db>> = params
            .iter()
            .map(|&t| resolve_primitive_type(db, t))
            .collect();
        if new_params.as_slice() == params {
            return ty;
        }

        return Type::new(
            db,
            ty.dialect(db),
            ty.name(db),
            new_params,
            ty.attrs(db).clone(),
        );
    }

    // Get the type name from the name attribute
    let Some(Attribute::Symbol(name_sym)) = ty.get_attr(db, tribute::Type::name_sym()) else {
        return ty;
    };

    // Resolve well-known primitive types
    let name_str = name_sym.to_string();
    match &*name_str {
        "Int" => tribute_rt::int_type(db),
        "Bool" => tribute_rt::bool_type(db),
        "Float" => tribute_rt::float_type(db),
        "Nat" => tribute_rt::nat_type(db),
        "String" => *core::String::new(db),
        "Bytes" => *core::Bytes::new(db),
        "Nil" => *core::Nil::new(db),
        // User-defined types are left unresolved - will be handled during full resolve pass
        _ => ty,
    }
}

/// Collect struct field names and types from a struct definition.
fn collect_struct_fields<'db>(
    db: &'db dyn salsa::Database,
    struct_def: tribute::StructDef<'db>,
) -> (Vec<Symbol>, Vec<Type<'db>>) {
    let fields_region = struct_def.fields(db);
    let mut field_names = Vec::new();
    let mut field_types = Vec::new();

    for block in fields_region.blocks(db).iter() {
        for field_op in block.operations(db).iter().copied() {
            if let Ok(field_def) = tribute::FieldDef::from_operation(db, field_op) {
                field_names.push(field_def.sym_name(db));
                // Resolve primitive types in field definitions
                field_types.push(resolve_primitive_type(db, field_def.r#type(db)));
            }
        }
    }

    (field_names, field_types)
}

/// Collect enum constructors from an enum definition.
fn collect_enum_constructors<'db>(
    db: &'db dyn salsa::Database,
    env: &mut ModuleEnv<'db>,
    op: &Operation<'db>,
    type_name: Symbol,
    ty: Type<'db>,
) {
    // Try to parse as tribute.enum_def with region-based variants
    let Ok(enum_def) = tribute::EnumDef::from_operation(db, *op) else {
        return;
    };

    let variants_region = enum_def.variants(db);

    for block in variants_region.blocks(db).iter() {
        for variant_op in block.operations(db).iter().copied() {
            // Check if this is a tribute.variant_def
            let Ok(variant_def) = tribute::VariantDef::from_operation(db, variant_op) else {
                continue;
            };

            let variant_sym = variant_def.sym_name(db);

            // Collect variant field types from the fields region
            // Resolve primitive types (Int, Bool, etc.) during collection
            let fields_region = variant_def.fields(db);
            let mut field_types = Vec::new();
            for field_block in fields_region.blocks(db).iter() {
                for field_op in field_block.operations(db).iter().copied() {
                    if let Ok(field_def) = tribute::FieldDef::from_operation(db, field_op) {
                        field_types.push(resolve_primitive_type(db, field_def.r#type(db)));
                    }
                }
            }

            let params: IdVec<Type<'db>> = field_types.into_iter().collect();

            // TODO: Extract field names from variant definition for named variants
            env.add_to_namespace(
                type_name,
                variant_sym,
                Binding::Constructor {
                    ty,
                    tag: Some(variant_sym),
                    params: params.clone(),
                    field_names: None, // Named field support to be added
                },
            );
            if env.lookup(variant_sym).is_none() {
                env.add_constructor(variant_sym, ty, Some(variant_sym), params, None);
            }
        }
    }
}

trunk_ir::symbols! {
    ATTR_OPERATIONS => "operations",
}

/// Collect ability operations from an ability definition.
///
/// Walks the operations region and extracts ability.op operations.
fn collect_ability_operations<'db>(
    db: &'db dyn salsa::Database,
    env: &mut ModuleEnv<'db>,
    ability_decl: tribute::AbilityDef<'db>,
    ability_name: Symbol,
) {
    let operations_region = ability_decl.operations(db);

    // Create the ability type as a core.ability_ref.
    // At this point we don't know the type parameters (they come from usage sites),
    // so we create a simple reference without parameters.
    let ability_ty = AbilityRefType::simple(db, ability_name).as_type();

    for block in operations_region.blocks(db).iter() {
        for op in block.operations(db).iter().copied() {
            // Check if this is a tribute.op_def
            let Ok(ability_op) = tribute::OpDef::from_operation(db, op) else {
                continue;
            };

            let op_name = ability_op.sym_name(db);
            let op_type = ability_op.r#type(db);

            // Extract params and return type from the function type
            let (params, return_ty) = if let Some(func_ty) = core::Func::from_type(db, op_type) {
                (func_ty.params(db), func_ty.result(db))
            } else {
                (IdVec::new(), core::Nil::new(db).as_type())
            };

            env.add_ability_op(ability_name, ability_ty, op_name, params, return_ty);
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
    /// Counter for generating fresh type variable IDs.
    /// We use a high starting value to avoid collision with tirgen-generated IDs.
    next_type_var_id: u64,
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
            // Start at a high value to avoid collision with tirgen IDs
            next_type_var_id: 100_000,
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
            next_type_var_id: 100_000,
        }
    }

    /// Generate a fresh type variable with a unique ID.
    fn fresh_type_var(&mut self) -> Type<'db> {
        let id = self.next_type_var_id;
        self.next_type_var_id += 1;
        tribute::type_var_with_id(self.db, id)
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

    /// Mark a tribute.var operation as a resolved local binding.
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
    /// Converts `tribute.type` to concrete types:
    /// - `Int` → `core.i64`
    /// - `Bool` → `core.i1`
    /// - `String` → `core.string`
    /// - User-defined types are looked up in the environment
    fn resolve_type(&self, ty: Type<'db>) -> Type<'db> {
        // Check if this is an unresolved type (tribute.type)
        if ty.dialect(self.db) == tribute::DIALECT_NAME() && ty.name(self.db) == tribute::TYPE() {
            // Get the type name from the name attribute (stored as Symbol)
            if let Some(Attribute::Symbol(name_sym)) =
                ty.get_attr(self.db, tribute::Type::name_sym())
            {
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
            // Primitive types (from tribute_rt dialect)
            "Int" => tribute_rt::int_type(self.db),
            "Bool" => tribute_rt::bool_type(self.db),
            "Float" => tribute_rt::float_type(self.db),
            "Nat" => tribute_rt::nat_type(self.db),
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
                    tribute::DIALECT_NAME(),
                    tribute::TYPE(),
                    IdVec::new(),
                    Attrs::new(),
                )
            }
        }
    }

    fn binding_from_path(&self, path: Symbol) -> Option<Binding<'db>> {
        if path.is_simple() {
            return self.env.lookup(path).cloned();
        }

        // For now, only support single-level namespaces (Type::Constructor)
        // TODO: Support multi-level namespaces
        let namespace = path.parent_path()?;
        let name = path.last_segment();
        self.env.lookup_qualified(namespace, name).cloned()
    }

    fn apply_use(&mut self, op: &Operation<'db>) {
        let attrs = op.attributes(self.db);

        let Some(Attribute::Symbol(path)) = attrs.get(&ATTR_PATH()) else {
            return;
        };

        let local_name = if let Some(Attribute::Symbol(alias)) = attrs.get(&ATTR_ALIAS()) {
            *alias
        } else {
            path.last_segment()
        };

        // Check if this path represents a namespace
        let namespace_sym = path.last_segment();
        let binding = self.binding_from_path(*path);

        if self.env.has_namespace(namespace_sym) {
            let type_def = match binding {
                Some(Binding::TypeDef { ty }) => Some(ty),
                _ => None,
            };
            self.add_import(
                local_name,
                Binding::Module {
                    namespace: *path,
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

        let result = Module::create(
            self.db,
            module.location(self.db),
            module.name(self.db),
            new_body,
        );

        // Sanity check: verify output module has no stale references
        #[cfg(debug_assertions)]
        verify_operand_references(self.db, result, "Resolver::resolve_module output");

        result
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
        let new_args = self.resolve_block_args(block.args(self.db).iter());

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

    /// Resolve block argument types while preserving attributes.
    fn resolve_block_args<'a>(
        &mut self,
        args: impl Iterator<Item = &'a BlockArg<'db>>,
    ) -> IdVec<BlockArg<'db>>
    where
        'db: 'a,
    {
        args.map(|arg| {
            let resolved_ty = self.resolve_type(arg.ty(self.db));
            BlockArg::new(self.db, resolved_ty, arg.attrs(self.db).clone())
        })
        .collect()
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
            (d, n) if d == tribute::DIALECT_NAME() && n == tribute::USE() => {
                self.apply_use(&remapped_op);
                Vec::new()
            }
            (d, n) if d == tribute::DIALECT_NAME() && n == tribute::ABILITY_DEF() => {
                // ability_def is kept for typeck to collect ability operation types
                // It will be filtered out in wasm lowering phase
                vec![self.resolve_op_regions(&remapped_op)]
            }
            (d, n) if d == tribute::DIALECT_NAME() && n == tribute::VAR() => {
                if let Some(resolved) = self.try_resolve_var(&remapped_op) {
                    resolved
                } else {
                    // Check if this is a resolved local variable.
                    // The `resolved_local` attribute is set when the var was bound
                    // to a local (function parameter, let binding, etc.).
                    // Note: We can't use result type to determine resolution status
                    // because type inference may assign concrete types to unresolved vars.
                    let is_resolved_local = self.is_marked_resolved_local(&remapped_op);
                    if self.report_unresolved && !is_resolved_local {
                        self.emit_unresolved_var_diagnostic(&remapped_op);
                    }
                    vec![self.resolve_op_regions(&remapped_op)]
                }
            }
            (d, n) if d == tribute::DIALECT_NAME() && n == tribute::PATH() => {
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
            (d, n) if d == tribute::DIALECT_NAME() && n == tribute::CALL() => {
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
            (d, n) if d == tribute::DIALECT_NAME() && n == tribute::CONS() => {
                let resolved = self.try_resolve_cons(&remapped_op);
                if !resolved.is_empty() {
                    resolved
                } else {
                    if self.report_unresolved {
                        self.emit_unresolved_cons_diagnostic(&remapped_op);
                    }
                    vec![self.resolve_op_regions(&remapped_op)]
                }
            }
            (d, n) if d == tribute::DIALECT_NAME() && n == tribute::RECORD() => {
                let resolved = self.try_resolve_record(&remapped_op);
                if !resolved.is_empty() {
                    resolved
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
            (d, n) if d == tribute::DIALECT_NAME() && n == tribute::LET() => {
                // Handle let binding with pattern region
                self.resolve_let(&remapped_op)
            }
            (d, n) if d == tribute::DIALECT_NAME() && n == tribute::ARM() => {
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

    /// Resolve a function type (params and result), preserving effect annotation.
    fn resolve_func_type(&self, ty: Type<'db>) -> Type<'db> {
        // Check if this is a func type (core.func)
        if let Some(func_ty) = core::Func::from_type(self.db, ty) {
            let params = func_ty.params(self.db);
            let result = func_ty.result(self.db);
            let effect = func_ty.effect(self.db);

            // Resolve all parameter types
            let resolved_params: IdVec<_> = params.iter().map(|p| self.resolve_type(*p)).collect();

            // Resolve result type
            let resolved_result = self.resolve_type(result);

            // Resolve effect type if present
            let resolved_effect = effect.map(|e| self.resolve_type(e));

            // Create resolved function type with effect preserved
            *core::Func::with_effect(self.db, resolved_params, resolved_result, resolved_effect)
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

    /// Resolve a tribute.let operation with pattern bindings.
    ///
    /// tribute.let has one operand (the bound value) and one pattern region.
    /// Pattern bindings are extracted and added to local scope.
    /// The tribute.let operation is erased from output (bindings are tracked in scope).
    ///
    /// ## Effect Propagation (Issue #200)
    ///
    /// Effects from the let binding's init expression naturally propagate because:
    /// 1. The init expression produces a value with associated operations
    /// 2. Those operations are type-checked, and their effects are merged
    /// 3. Let bindings map names directly to values, so effects flow through
    ///
    /// No special handling is needed for effect propagation in let bindings.
    fn resolve_let(&mut self, op: &Operation<'db>) -> Vec<Operation<'db>> {
        let operands = op.operands(self.db);
        let regions = op.regions(self.db);

        if operands.is_empty() || regions.is_empty() {
            return Vec::new();
        }

        // Look up the bound value (may have been remapped by previous transformations)
        let bound_value = self.ctx.lookup(operands[0]);
        let pattern_region = &regions[0];

        // Collect let bindings from pattern region
        // For simple patterns like tribute_pat.bind("x"), bind x directly to the value
        // For complex patterns, we need extraction operations
        self.collect_let_bindings(pattern_region, bound_value);

        // tribute.let is erased from output - the bindings are tracked in scope
        Vec::new()
    }

    /// Collect let bindings from a pattern region for tribute.let.
    ///
    /// Unlike case arm pattern bindings (which use PatternBinding and case.bind),
    /// let bindings map names directly to values or extraction results.
    fn collect_let_bindings(&mut self, region: &Region<'db>, value: Value<'db>) {
        for block in region.blocks(self.db).iter() {
            for op in block.operations(self.db).iter() {
                self.collect_let_binding_from_op(op, value);
            }
        }
    }

    /// Collect a single let binding from a pattern operation.
    fn collect_let_binding_from_op(&mut self, op: &Operation<'db>, value: Value<'db>) {
        if let Ok(bind_op) = tribute_pat::Bind::from_operation(self.db, *op) {
            // tribute_pat.bind("x") - bind x to the value
            let name = bind_op.name(self.db);
            let ty = self.fresh_type_var();
            self.add_local(name, LocalBinding::LetBinding { value, ty });
        } else if let Ok(as_pat_op) = tribute_pat::AsPat::from_operation(self.db, *op) {
            // tribute_pat.as_pat: bind the name to value, then recurse on inner pattern
            let name = as_pat_op.name(self.db);
            let ty = self.fresh_type_var();
            self.add_local(name, LocalBinding::LetBinding { value, ty });
            // Recurse on inner pattern with the same value
            self.collect_let_bindings(&as_pat_op.inner(self.db), value);
        } else if tribute_pat::Wildcard::from_operation(self.db, *op).is_ok() {
            // Wildcard pattern - no binding needed
        } else if let Ok(tuple_op) = tribute_pat::Tuple::from_operation(self.db, *op) {
            // TODO: Handle tuple patterns by generating extraction operations
            // For now, fall through to nested regions which may have bindings
            // Each element in the tuple needs extraction - for now just recurse
            // with the same value (not fully correct for extraction)
            self.collect_let_bindings(&tuple_op.elements(self.db), value);
        } else {
            // Other pattern ops may have nested regions with bindings
            for region in op.regions(self.db).iter() {
                self.collect_let_bindings(region, value);
            }
        }
    }

    /// Collect pattern bindings from a pattern region.
    ///
    /// Recursively walks the pattern region to find tribute_pat.bind, tribute_pat.as_pat,
    /// and tribute_pat.list_rest operations, adding their bindings to local scope.
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
            (d, n) if d == tribute_pat::DIALECT_NAME() && n == tribute_pat::BIND() => {
                // pat.bind has a "name" attribute
                let attrs = op.attributes(self.db);
                if let Some(Attribute::Symbol(sym)) = attrs.get(&ATTR_NAME()) {
                    // Pattern binding - value comes from pattern matching at runtime
                    let infer_ty = self.fresh_type_var();
                    self.add_local(*sym, LocalBinding::PatternBinding { ty: infer_ty });
                }
            }
            (d, n) if d == tribute_pat::DIALECT_NAME() && n == tribute_pat::AS_PAT() => {
                // pat.as_pat has a "name" attribute and an inner pattern region
                let attrs = op.attributes(self.db);
                if let Some(Attribute::Symbol(sym)) = attrs.get(&ATTR_NAME()) {
                    let infer_ty = self.fresh_type_var();
                    self.add_local(*sym, LocalBinding::PatternBinding { ty: infer_ty });
                }
                // Also collect from inner pattern region
                for region in op.regions(self.db).iter() {
                    self.collect_pattern_bindings(region);
                }
            }
            (d, n) if d == tribute_pat::DIALECT_NAME() && n == tribute_pat::LIST_REST() => {
                // pat.list_rest has a "rest_name" attribute
                let attrs = op.attributes(self.db);
                if let Some(Attribute::Symbol(sym)) = attrs.get(&ATTR_REST_NAME()) {
                    let infer_ty = self.fresh_type_var();
                    self.add_local(*sym, LocalBinding::PatternBinding { ty: infer_ty });
                }
                // Also collect from head pattern region
                for region in op.regions(self.db).iter() {
                    self.collect_pattern_bindings(region);
                }
            }
            (d, n) if d == tribute_pat::DIALECT_NAME() && n == tribute_pat::HANDLER_SUSPEND() => {
                // tribute_pat.handler_suspend has args and continuation regions
                // The continuation region contains a tribute_pat.bind for the continuation variable
                // We need to use the continuation_type attribute as the binding's type
                let continuation_type = op
                    .attributes(self.db)
                    .get(&tribute_pat::handler_suspend_attrs::CONTINUATION_TYPE())
                    .and_then(|attr| match attr {
                        Attribute::Type(ty) => Some(*ty),
                        _ => None,
                    });

                // Process args region normally (they get fresh type vars)
                if let Ok(handler_suspend) =
                    tribute_pat::HandlerSuspend::from_operation(self.db, *op)
                {
                    self.collect_pattern_bindings(&handler_suspend.args(self.db));

                    // Process continuation region with special handling for bind
                    let cont_region = handler_suspend.continuation(self.db);
                    self.collect_handler_continuation_bindings(&cont_region, continuation_type);
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

    /// Collect bindings from a handler's continuation region.
    ///
    /// The continuation region contains a single tribute_pat.bind for the continuation variable.
    /// Instead of creating a fresh type variable, we use the continuation_type attribute
    /// from the handler_suspend operation to ensure proper type propagation.
    fn collect_handler_continuation_bindings(
        &mut self,
        region: &Region<'db>,
        continuation_type: Option<Type<'db>>,
    ) {
        for block in region.blocks(self.db).iter() {
            for op in block.operations(self.db).iter() {
                let dialect = op.dialect(self.db);
                let op_name = op.name(self.db);

                if dialect == tribute_pat::DIALECT_NAME() && op_name == tribute_pat::BIND() {
                    // pat.bind has a "name" attribute
                    let attrs = op.attributes(self.db);
                    if let Some(Attribute::Symbol(sym)) = attrs.get(&ATTR_NAME()) {
                        // Use the continuation type if available, otherwise fall back to fresh type var
                        let ty = continuation_type.unwrap_or_else(|| self.fresh_type_var());
                        self.add_local(*sym, LocalBinding::PatternBinding { ty });
                    }
                } else {
                    // Recurse into nested regions
                    for nested_region in op.regions(self.db).iter() {
                        self.collect_handler_continuation_bindings(
                            nested_region,
                            continuation_type,
                        );
                    }
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
    /// The entry block starts with tribute.var operations that declare parameter names.
    /// These are mapped to block arguments and erased from output.
    fn resolve_func_entry_block(&mut self, block: &Block<'db>) -> Block<'db> {
        let block_args = block.args(self.db);
        let operations = block.operations(self.db);

        // Scan for initial tribute.var operations that declare parameters
        // Only scan up to the number of block arguments
        // Parameter declarations have the function's overall span (from lower_function)
        let func_span = block.location(self.db).span;
        let mut param_declarations = Vec::new();
        for op in operations.iter() {
            // Stop early if we've found all expected parameters
            if param_declarations.len() >= block_args.len() {
                break;
            }

            if op.dialect(self.db) == tribute::DIALECT_NAME() && op.name(self.db) == tribute::VAR()
            {
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
                    break; // Not a proper tribute.var, stop scanning
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
                let param_ty = block_args[i].ty(self.db);

                // Add to local scope
                self.add_local(
                    *name,
                    LocalBinding::Parameter {
                        value: block_arg,
                        ty: param_ty,
                    },
                );

                // Map the tribute.var result to the block argument
                let old_result = op.result(self.db, 0);
                self.ctx.map_value(old_result, block_arg);
            }
        }

        let new_args = self.resolve_block_args(block_args.iter());

        // Now resolve the block, skipping parameter declaration tribute.var ops
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

        // Resolve nested regions (operand remapping happens in resolve_operation)
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

    /// Try to resolve a `tribute.var` operation.
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
        if name == "Nil" {
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
                    // Local binding found - keep tribute.var with resolved type for hover span
                    let resolved_ty = self.resolve_type(*ty);

                    // Create new tribute.var with resolved type (keeps span for hover)
                    let new_op = tribute::var(self.db, location, resolved_ty, *sym);
                    let new_operation = self.mark_resolved_local(new_op.as_operation());

                    // Map old result to the actual bound value (not the new tribute.var's result)
                    // This ensures use sites get the correct value
                    let old_result = op.result(self.db, 0);
                    self.ctx.map_value(old_result, *value);

                    // Return the new tribute.var to keep it in IR for hover
                    return Some(vec![new_operation]);
                }
                LocalBinding::PatternBinding { ty } => {
                    // Pattern binding - keep tribute.var with resolved type
                    // tribute_to_scf will remap the result to the bound value
                    let resolved_ty = self.resolve_type(*ty);
                    let new_op = tribute::var(self.db, location, resolved_ty, *sym);
                    let new_operation = self.mark_resolved_local(new_op.as_operation());

                    // Map old result to new tribute.var result (will be remapped in tribute_to_scf)
                    let old_result = op.result(self.db, 0);
                    let new_result = new_operation.result(self.db, 0);
                    self.ctx.map_value(old_result, new_result);

                    return Some(vec![new_operation]);
                }
            }
        }

        // Then check module environment
        match self.lookup_binding(name)? {
            Binding::Function { path, ty } => {
                // Create func.constant operation
                // func::constant(db, location, result_type, func_ref)
                let new_op = func::constant(self.db, location, *ty, *path);
                let new_operation = new_op.as_operation();

                // Map old result to new result
                let old_result = op.result(self.db, 0);
                let new_result = new_operation.result(self.db, 0);
                self.ctx.map_value(old_result, new_result);

                Some(vec![new_operation])
            }
            Binding::Constructor { ty, tag, .. } => {
                // Create adt.struct_new or adt.variant_new
                // No args here since tribute.var is just a reference (not a call)
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

                // Keep tribute.var but mark it as a resolved const reference
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
            Binding::AbilityOp {
                ability,
                op_name,
                return_ty,
                ..
            } => {
                // Ability operation reference - create ability.perform
                // NOTE: This case handles `State::get` as a bare reference (rare).
                // Usually ability calls come through tribute.call + tribute.path which is
                // handled in try_resolve_call.
                let resolved_return_ty = self.resolve_type(*return_ty);
                let new_op = ability::perform(
                    self.db,
                    location,
                    std::iter::empty::<Value<'db>>(),
                    resolved_return_ty,
                    *ability,
                    *op_name,
                );

                let old_result = op.result(self.db, 0);
                let new_result = new_op.as_operation().result(self.db, 0);
                self.ctx.map_value(old_result, new_result);

                Some(vec![new_op.as_operation()])
            }
            Binding::TypeDef { .. } | Binding::Module { .. } => {
                // Type used in value position - error (leave unresolved for diagnostics)
                None
            }
        }
    }

    /// Try to resolve a `tribute.path` operation.
    fn try_resolve_path(&mut self, op: &Operation<'db>) -> Option<Operation<'db>> {
        let attrs = op.attributes(self.db);
        let Attribute::Symbol(path) = attrs.get(&ATTR_PATH())? else {
            return None;
        };

        if path.is_simple() {
            return None;
        }

        let location = op.location(self.db);

        // First try direct lookup in definitions, then fall back to namespace lookup
        let binding = self.env.lookup_path(*path).or_else(|| {
            // Fall back to namespace lookup for enum variants, etc.
            let namespace = path.parent_path()?;
            self.env.lookup_qualified(namespace, path.last_segment())
        })?;

        match binding {
            Binding::Function { path, ty } => {
                let new_op = func::constant(self.db, location, *ty, *path);
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
            Binding::AbilityOp {
                ability,
                op_name,
                return_ty,
                ..
            } => {
                // Ability operation reference via path (e.g., State::get as a value)
                // This creates an ability.perform with no arguments
                let resolved_return_ty = self.resolve_type(*return_ty);
                let new_op = ability::perform(
                    self.db,
                    location,
                    std::iter::empty::<Value<'db>>(),
                    resolved_return_ty,
                    *ability,
                    *op_name,
                );

                let old_result = op.result(self.db, 0);
                let new_result = new_op.as_operation().result(self.db, 0);
                self.ctx.map_value(old_result, new_result);

                Some(new_op.as_operation())
            }
            Binding::TypeDef { .. } | Binding::Module { .. } => None,
        }
    }

    /// Try to resolve a `tribute.call` operation.
    fn try_resolve_call(&mut self, op: &Operation<'db>) -> Option<Operation<'db>> {
        let attrs = op.attributes(self.db);
        let Attribute::Symbol(path) = attrs.get(&ATTR_NAME())? else {
            return None;
        };

        let location = op.location(self.db);
        let result_ty = op.results(self.db).first().copied()?;
        let args: Vec<Value<'db>> = op.operands(self.db).iter().copied().collect();

        // Try to resolve the callee
        // Track whether this is a qualified path (affects which path to use in the call)
        let is_qualified = !path.is_simple();
        let binding = if path.is_simple() {
            let name = *path;
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
            // First try direct lookup in definitions, then fall back to namespace lookup
            self.env.lookup_path(*path).or_else(|| {
                // Fall back to namespace lookup for enum variants, etc.
                let namespace = path.parent_path()?;
                self.env.lookup_qualified(namespace, path.last_segment())
            })
        }?;

        match binding {
            Binding::Function {
                path: binding_path,
                ty: func_ty,
            } => {
                // Direct function call
                // Use the callee function's return type instead of the tribute.call's type variable
                // Also resolve the return type (it may be tribute.type that needs resolution)
                let call_result_ty = core::Func::from_type(self.db, *func_ty)
                    .map(|f| self.resolve_type(f.result(self.db)))
                    .unwrap_or(result_ty);
                // For qualified calls, use the original path (e.g., math::double)
                // The binding's stored path may just be the simple name (double)
                let callee_path = if is_qualified { *path } else { *binding_path };
                // func::call(db, location, args, result_type, callee)
                let new_op = func::call(self.db, location, args, call_result_ty, callee_path);
                let new_operation = new_op.as_operation();

                let old_result = op.result(self.db, 0);
                let new_result = new_operation.result(self.db, 0);
                self.ctx.map_value(old_result, new_result);

                Some(new_operation)
            }
            Binding::AbilityOp {
                ability,
                op_name,
                return_ty,
                ..
            } => {
                // Ability operation call (e.g., State::get(), Console::print(msg))
                // Create ability.perform with the call arguments
                let resolved_return_ty = self.resolve_type(*return_ty);
                let new_op = ability::perform(
                    self.db,
                    location,
                    args,
                    resolved_return_ty,
                    *ability,
                    *op_name,
                );

                let old_result = op.result(self.db, 0);
                let new_result = new_op.as_operation().result(self.db, 0);
                self.ctx.map_value(old_result, new_result);

                Some(new_op.as_operation())
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

    /// Try to resolve a `tribute.cons` operation (positional constructor).
    ///
    /// Returns a single operation wrapped in a vector for consistency.
    fn try_resolve_cons(&mut self, op: &Operation<'db>) -> Vec<Operation<'db>> {
        let attrs = op.attributes(self.db);
        let Some(Attribute::Symbol(path)) = attrs.get(&ATTR_NAME()) else {
            return vec![];
        };

        let location = op.location(self.db);
        let operands = op.operands(self.db);

        // Look up binding
        let binding = if path.is_simple() {
            let name = *path;
            self.lookup_binding(name).cloned()
        } else {
            // For now, only support single-level qualified constructors (Type::Variant)
            let Some(namespace) = path.parent_path() else {
                return vec![];
            };
            let name = path.last_segment();
            self.env.lookup_qualified(namespace, name).cloned()
        };

        let Some(binding) = binding else {
            return vec![];
        };

        // All operands are field values (positional)
        let args: Vec<Value<'db>> = operands.iter().copied().collect();

        match binding {
            Binding::Constructor { ty, tag, .. } => {
                let new_operation = if let Some(tag) = tag {
                    adt::variant_new(self.db, location, args, ty, ty, tag).as_operation()
                } else {
                    adt::struct_new(self.db, location, args, ty, ty).as_operation()
                };

                let old_result = op.result(self.db, 0);
                let new_result = new_operation.result(self.db, 0);
                self.ctx.map_value(old_result, new_result);

                vec![new_operation]
            }
            _ => vec![],
        }
    }

    /// Try to resolve a `tribute.record` operation (named fields + spread).
    ///
    /// Returns a vector of operations: field_get operations followed by struct_new.
    fn try_resolve_record(&mut self, op: &Operation<'db>) -> Vec<Operation<'db>> {
        let attrs = op.attributes(self.db);
        let Some(Attribute::Symbol(path)) = attrs.get(&ATTR_NAME()) else {
            return vec![];
        };

        let location = op.location(self.db);
        let operands = op.operands(self.db);
        let regions = op.regions(self.db);

        // Look up binding
        let binding = if path.is_simple() {
            let name = *path;
            self.lookup_binding(name).cloned()
        } else {
            let Some(namespace) = path.parent_path() else {
                return vec![];
            };
            let name = path.last_segment();
            self.env.lookup_qualified(namespace, name).cloned()
        };

        let Some(Binding::Constructor {
            ty,
            tag,
            field_names,
            ..
        }) = binding
        else {
            return vec![];
        };

        // Record syntax only supported for structs (no tag), not enum variants
        if tag.is_some() {
            tracing::warn!("Record syntax not supported for enum variants");
            return vec![];
        }

        // Get struct field names from binding
        let Some(struct_fields) = field_names.as_ref() else {
            return vec![];
        };

        // Parse field_arg ops from the fields region to get override field names and values
        let mut override_fields: Vec<(Symbol, Value<'db>)> = Vec::new();
        if let Some(fields_region) = regions.first() {
            for block in fields_region.blocks(self.db).iter() {
                for field_op in block.operations(self.db).iter().copied() {
                    if let Ok(field_arg) = tribute::FieldArg::from_operation(self.db, field_op) {
                        let field_name = field_arg.name(self.db);
                        // Get the value operand - need to remap it
                        let field_ops = field_op.operands(self.db);
                        if let Some(&field_value) = field_ops.first() {
                            let remapped_value = self.ctx.lookup(field_value);
                            override_fields.push((field_name, remapped_value));
                        }
                    }
                }
            }
        }

        // Get base value if spread is present
        let base_value = operands.first().map(|&v| self.ctx.lookup(v));
        let has_spread = base_value.is_some();

        if struct_fields.is_empty() {
            // No fields, just create empty struct
            let new_operation = adt::struct_new(self.db, location, vec![], ty, ty).as_operation();
            let old_result = op.result(self.db, 0);
            let new_result = new_operation.result(self.db, 0);
            self.ctx.map_value(old_result, new_result);
            return vec![new_operation];
        }

        // Build field values
        let mut all_ops: Vec<Operation<'db>> = Vec::new();
        let mut all_values: Vec<Value<'db>> = Vec::new();

        for (field_idx, struct_field_name) in struct_fields.iter().enumerate() {
            // Check if this field is overridden
            let override_value = override_fields
                .iter()
                .find(|(name, _)| *name == *struct_field_name)
                .map(|(_, val)| *val);

            if let Some(val) = override_value {
                // Use the override value
                all_values.push(val);
            } else if has_spread {
                // Get field from base using adt.struct_get
                let base = base_value.unwrap();
                let field_ty = ty; // TODO: Get actual field type from struct definition
                let struct_get = adt::struct_get(
                    self.db,
                    location,
                    base,
                    field_ty,
                    ty,
                    Attribute::IntBits(field_idx as u64),
                );
                let field_val = struct_get.result(self.db);

                all_ops.push(struct_get.as_operation());
                all_values.push(field_val);
            } else {
                // No spread and no override - this is an error (missing field)
                tracing::warn!("Missing field {} in record construction", struct_field_name);
                return vec![];
            }
        }

        // Create struct_new with all field values
        let new_operation = adt::struct_new(self.db, location, all_values, ty, ty).as_operation();
        let old_result = op.result(self.db, 0);
        let new_result = new_operation.result(self.db, 0);
        self.ctx.map_value(old_result, new_result);

        all_ops.push(new_operation);
        all_ops
    }

    // === Diagnostic Helpers ===

    /// Emit diagnostic for unresolved `tribute.var`.
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

    /// Emit diagnostic for unresolved `tribute.path`.
    fn emit_unresolved_path_diagnostic(&self, op: &Operation<'db>) {
        let path = op
            .attributes(self.db)
            .get(&ATTR_PATH())
            .and_then(|a| {
                if let Attribute::Symbol(sym) = a {
                    Some(sym.to_string())
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

    /// Emit diagnostic for unresolved `tribute.call`.
    fn emit_unresolved_call_diagnostic(&self, op: &Operation<'db>) {
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
            message: format!("unresolved function call: `{}`", name),
            span: op.location(self.db).span,
            severity: DiagnosticSeverity::Error,
            phase: CompilationPhase::NameResolution,
        }
        .accumulate(self.db);
    }

    /// Emit diagnostic for unresolved `tribute.cons`.
    fn emit_unresolved_cons_diagnostic(&self, op: &Operation<'db>) {
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
    let result = resolver.resolve_module(module);

    // Sanity check: verify output module has no stale references
    #[cfg(debug_assertions)]
    verify_operand_references(db, result, "resolve output");

    result
}

#[cfg(debug_assertions)]
fn verify_operand_references<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
    context: &str,
) {
    use std::collections::HashSet;

    // Collect all operations in the module
    let mut all_ops: HashSet<trunk_ir::Operation<'db>> = HashSet::new();
    collect_ops_in_region(db, module.body(db), &mut all_ops);

    // Verify all operand references point to operations in the set
    verify_refs_in_region(db, module.body(db), &all_ops, context);
}

#[cfg(debug_assertions)]
fn collect_ops_in_region<'db>(
    db: &'db dyn salsa::Database,
    region: Region<'db>,
    ops: &mut std::collections::HashSet<trunk_ir::Operation<'db>>,
) {
    for block in region.blocks(db).iter() {
        for op in block.operations(db).iter().copied() {
            ops.insert(op);
            for nested in op.regions(db).iter().copied() {
                collect_ops_in_region(db, nested, ops);
            }
        }
    }
}

#[cfg(debug_assertions)]
fn verify_refs_in_region<'db>(
    db: &'db dyn salsa::Database,
    region: Region<'db>,
    all_ops: &std::collections::HashSet<trunk_ir::Operation<'db>>,
    context: &str,
) {
    for block in region.blocks(db).iter() {
        for op in block.operations(db).iter().copied() {
            for operand in op.operands(db).iter() {
                if let ValueDef::OpResult(ref_op) = operand.def(db)
                    && !all_ops.contains(&ref_op)
                {
                    tracing::warn!(
                        "STALE REFERENCE DETECTED in {}!\n  \
                         Operation {}.{} references {}.{} which is NOT in the module",
                        context,
                        op.dialect(db),
                        op.name(db),
                        ref_op.dialect(db),
                        ref_op.name(db)
                    );
                }
            }
            for nested in op.regions(db).iter().copied() {
                verify_refs_in_region(db, nested, all_ops, context);
            }
        }
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use tribute_ir::dialect::tribute;
    use trunk_ir::dialect::{arith, core, func};
    use trunk_ir::{Location, PathId, Span, idvec};

    fn test_location<'db>(db: &'db dyn salsa::Database) -> Location<'db> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    fn simple_func<'db>(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        name: &str,
    ) -> func::Func<'db> {
        let name_sym = Symbol::from_dynamic(name);
        func::Func::build(
            db,
            location,
            name_sym,
            idvec![],
            *core::I64::new(db),
            |entry| {
                let value = entry.op(arith::Const::i64(db, location, 42));
                entry.op(func::Return::value(db, location, value.result(db)));
            },
        )
    }

    fn module_with_use_call<'db>(db: &'db dyn salsa::Database, alias: Option<&str>) -> Module<'db> {
        let location = test_location(db);
        let helpers = core::Module::build(db, location, Symbol::new("helpers"), |inner| {
            inner.op(simple_func(db, location, "double"));
        });
        let path = Symbol::new("helpers::double");
        let alias_sym = Symbol::from_dynamic(alias.unwrap_or(""));

        let name = alias.unwrap_or("double");
        let call_path = Symbol::from_dynamic(name);
        let arg = arith::Const::i64(db, location, 1);
        let call_result_ty = tribute::unresolved_type(db, Symbol::new("Int"), idvec![]);
        let call = tribute::call(
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
            top.op(tribute::r#use(db, location, path, alias_sym, false));
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
            op.dialect(db) == "tribute"
                && op.name(db) == "call"
                && matches!(
                    op.attributes(db).get(&ATTR_NAME()),
                    Some(Attribute::Symbol(sym)) if sym.last_segment() == name_sym
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
                    Some(Attribute::Symbol(sym)) if sym.last_segment() == name_sym
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
            "use import should resolve tribute.call to func.call"
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
            "use alias should resolve tribute.call to func.call"
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
                tribute::r#const(db, location, int_ty, max_size_sym, Attribute::IntBits(1024));
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
                tribute::r#const(db, location, int_ty, max_size_sym, Attribute::IntBits(1024));
            top.op(const_op);

            let func_op = func::Func::build(db, location, "test", idvec![], int_ty, |entry| {
                let const_ref = entry.op(tribute::var(db, location, int_ty, max_size_sym));
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
                op.dialect(db) == "tribute"
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

    /// Create a module with a let binding: fn main() { let x = 42; x }
    #[salsa::tracked]
    fn module_with_let_binding(db: &dyn salsa::Database) -> Module<'_> {
        use tribute_ir::dialect::tribute_pat;

        let location = test_location(db);
        let infer_ty = tribute::new_type_var(db, std::collections::BTreeMap::new());

        // Pre-create the pattern region outside the closure
        let pattern_region = tribute_pat::helpers::bind_region(db, location, Symbol::new("x"));

        // Create the main function with let binding
        let main_func = func::Func::build(
            db,
            location,
            "main",
            idvec![],
            *core::I64::new(db),
            |entry| {
                // %0 = arith.const 42
                let const_val = entry.op(arith::Const::i64(db, location, 42));

                // tribute.let(%0) { tribute_pat.bind("x") }
                entry.op(tribute::r#let(
                    db,
                    location,
                    const_val.result(db),
                    pattern_region,
                ));

                // %1 = tribute.var("x")
                let var_ref = entry.op(tribute::var(db, location, infer_ty, Symbol::new("x")));

                // return %1
                entry.op(func::Return::value(db, location, var_ref.result(db)));
            },
        );

        core::Module::build(db, location, Symbol::new("main"), |top| {
            top.op(main_func);
        })
    }

    #[salsa::tracked]
    fn resolve_let_binding_module(db: &dyn salsa::Database) -> Module<'_> {
        let module = module_with_let_binding(db);
        resolve_module(db, &module)
    }

    #[salsa_test]
    fn test_let_binding_resolved(db: &salsa::DatabaseImpl) {
        let resolved = resolve_let_binding_module(db);

        let mut ops = Vec::new();
        collect_ops(db, &resolved.body(db), &mut ops);

        // After resolution:
        // 1. tribute.let should be erased
        // 2. tribute.var("x") should be marked as resolved_local
        // 3. The value mapping should connect the var to the const result

        // Check that tribute.let is erased
        let let_ops: Vec<_> = ops
            .iter()
            .filter(|op| op.dialect(db) == "tribute" && op.name(db) == "let")
            .collect();
        assert!(
            let_ops.is_empty(),
            "tribute.let should be erased after resolution"
        );

        // Check that tribute.var("x") is resolved
        let x_sym = Symbol::new("x");
        let var_refs: Vec<_> = ops
            .iter()
            .filter(|op| {
                op.dialect(db) == "tribute"
                    && op.name(db) == "var"
                    && matches!(
                        op.attributes(db).get(&ATTR_NAME()),
                        Some(Attribute::Symbol(sym)) if *sym == x_sym
                    )
            })
            .collect();

        assert!(!var_refs.is_empty(), "should find tribute.var(x) reference");

        for var_ref in var_refs {
            let attrs = var_ref.attributes(db);
            assert_eq!(
                attrs.get(&ATTR_RESOLVED_LOCAL()),
                Some(&Attribute::Bool(true)),
                "let binding reference should be marked as resolved_local"
            );
        }
    }

    /// Create a module with an ability declaration and a call to it.
    #[salsa::tracked]
    fn module_with_ability_call(db: &dyn salsa::Database) -> Module<'_> {
        use trunk_ir::BlockBuilder;

        let location = test_location(db);
        let infer_ty = tribute::new_type_var(db, std::collections::BTreeMap::new());

        // Create ability declaration: ability Console { fn print(msg: String) -> Nil }
        // Use actual Type attributes for param and return types
        let string_ty = *core::String::new(db);
        let nil_ty = *core::Nil::new(db);

        // Build operations region with tribute.op_def
        let mut ops_block = BlockBuilder::new(db, location);
        let print_type = core::Func::new(db, idvec![string_ty], nil_ty).as_type();
        ops_block.op(tribute::op_def(
            db,
            location,
            Symbol::new("print"),
            print_type,
        ));
        let operations_region = Region::new(db, location, idvec![ops_block.build()]);

        let ability_decl = tribute::ability_def(
            db,
            location,
            infer_ty,
            Symbol::new("Console"),
            operations_region,
        );

        // Create main function that calls Console::print()
        let print_path = Symbol::new("Console::print");
        let main_func = func::Func::build(db, location, "main", idvec![], infer_ty, |entry| {
            // tribute.call(Console::print)()
            let call_result = entry.op(tribute::call(db, location, vec![], infer_ty, print_path));
            entry.op(func::Return::value(db, location, call_result.result(db)));
        });

        core::Module::build(db, location, Symbol::new("main"), |top| {
            top.op(ability_decl);
            top.op(main_func);
        })
    }

    #[salsa::tracked]
    fn resolve_ability_call_module(db: &dyn salsa::Database) -> Module<'_> {
        let module = module_with_ability_call(db);
        resolve_module(db, &module)
    }

    #[salsa_test]
    fn test_ability_call_resolved_to_perform(db: &salsa::DatabaseImpl) {
        let resolved = resolve_ability_call_module(db);

        let mut ops = Vec::new();
        collect_ops(db, &resolved.body(db), &mut ops);

        // After resolution, tribute.call to Console::print should become ability.perform
        let perform_ops: Vec<_> = ops
            .iter()
            .filter(|op| op.dialect(db) == "ability" && op.name(db) == "perform")
            .collect();

        assert!(
            !perform_ops.is_empty(),
            "ability call should be resolved to ability.perform"
        );

        // Check that the perform op has the correct ability_ref and op attributes
        let perform_op = perform_ops[0];
        let attrs = perform_op.attributes(db);

        // Check ability_ref attribute - now stored as Type (core.ability_ref)
        if let Some(Attribute::Type(ability_ty)) = attrs.get(&Symbol::new("ability_ref")) {
            // Use AbilityRefType to extract the name
            if let Some(ability_ref) = AbilityRefType::from_type(db, *ability_ty) {
                assert!(ability_ref.name(db) == Some(Symbol::new("Console")));
            } else {
                panic!("ability_ref should be a valid AbilityRefType");
            }
        } else {
            panic!("ability.perform should have ability_ref attribute");
        }

        // Check op attribute
        if let Some(Attribute::Symbol(op_name)) = attrs.get(&Symbol::new("op")) {
            assert!(*op_name == "print");
        } else {
            panic!("ability.perform should have op attribute");
        }
    }

    #[salsa_test]
    fn test_ability_operation_types_extracted(db: &salsa::DatabaseImpl) {
        // Build the module and extract the environment
        let module = module_with_ability_call(db);
        let env = build_env(db, &module);

        // Look up the Console::print operation
        let print_binding = env.lookup_qualified(Symbol::new("Console"), Symbol::new("print"));

        assert!(
            print_binding.is_some(),
            "Console::print should be found in environment"
        );

        let binding = print_binding.unwrap();
        if let Binding::AbilityOp {
            params, return_ty, ..
        } = binding
        {
            // Verify parameter types are correctly extracted
            assert_eq!(params.len(), 1, "print should have 1 parameter");
            let param_ty = params[0];
            assert!(
                param_ty.is_dialect(db, core::DIALECT_NAME(), core::STRING()),
                "print parameter should be String type, got {:?}",
                param_ty
            );

            // Verify return type is correctly extracted
            assert!(
                return_ty.is_dialect(db, core::DIALECT_NAME(), core::NIL()),
                "print return type should be Nil, got {:?}",
                return_ty
            );
        } else {
            panic!(
                "Console::print should be an AbilityOp binding, got {:?}",
                binding
            );
        }
    }
}
