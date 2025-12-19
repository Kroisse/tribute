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

use salsa::Accumulator;
use tribute_core::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use tribute_trunk_ir::dialect::adt;
use tribute_trunk_ir::dialect::core::{self, Module};
use tribute_trunk_ir::dialect::func;
use tribute_trunk_ir::dialect::src;
use tribute_trunk_ir::{
    Attribute, Attrs, Block, DialectOp, IdVec, Operation, Region, Symbol, Type, Value, ValueDef,
};

// =============================================================================
// Module Environment
// =============================================================================

/// Information about a resolved name.
#[derive(Clone, Debug)]
pub enum Binding<'db> {
    /// A function defined in this module or imported.
    Function {
        /// Fully qualified path (e.g., ["List", "map"])
        path: IdVec<Symbol<'db>>,
        /// Function type
        ty: Type<'db>,
    },
    /// A type constructor (struct or enum variant).
    Constructor {
        /// The type being constructed
        ty: Type<'db>,
        /// For enums, the variant tag
        tag: Option<Symbol<'db>>,
        /// Constructor parameter types
        params: IdVec<Type<'db>>,
    },
    /// A type alias or definition.
    TypeDef {
        /// The defined type
        ty: Type<'db>,
    },
}

/// Module environment for name resolution.
///
/// Tracks all names visible in the current module.
#[derive(Debug, Default)]
pub struct ModuleEnv<'db> {
    /// Names defined in this module.
    definitions: HashMap<String, Binding<'db>>,
    /// Names imported via `use` statements.
    imports: HashMap<String, Binding<'db>>,
    /// Qualified paths (namespace → name → binding).
    namespaces: HashMap<String, HashMap<String, Binding<'db>>>,
}

impl<'db> ModuleEnv<'db> {
    /// Create a new empty module environment.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a function definition.
    pub fn add_function(&mut self, name: &str, path: IdVec<Symbol<'db>>, ty: Type<'db>) {
        self.definitions
            .insert(name.to_string(), Binding::Function { path, ty });
    }

    /// Add a type constructor.
    pub fn add_constructor(
        &mut self,
        name: &str,
        ty: Type<'db>,
        tag: Option<Symbol<'db>>,
        params: IdVec<Type<'db>>,
    ) {
        self.definitions
            .insert(name.to_string(), Binding::Constructor { ty, tag, params });
    }

    /// Add a type definition.
    pub fn add_type(&mut self, name: &str, ty: Type<'db>) {
        self.definitions
            .insert(name.to_string(), Binding::TypeDef { ty });
    }

    /// Add a qualified name to a namespace.
    pub fn add_to_namespace(&mut self, namespace: &str, name: &str, binding: Binding<'db>) {
        self.namespaces
            .entry(namespace.to_string())
            .or_default()
            .insert(name.to_string(), binding);
    }

    /// Look up an unqualified name.
    pub fn lookup(&self, name: &str) -> Option<&Binding<'db>> {
        // First check local definitions
        if let Some(b) = self.definitions.get(name) {
            return Some(b);
        }
        // Then check imports
        self.imports.get(name)
    }

    /// Look up a qualified path (e.g., "List::map").
    pub fn lookup_qualified(&self, namespace: &str, name: &str) -> Option<&Binding<'db>> {
        self.namespaces.get(namespace)?.get(name)
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
    let dialect = op.dialect(db).text(db);
    let op_name = op.name(db).text(db);

    match (dialect, op_name) {
        ("func", "func") => {
            // Function definition
            let attrs = op.attributes(db);
            let sym_key = Symbol::new(db, "sym_name");
            let type_key = Symbol::new(db, "type");

            if let (Some(Attribute::Symbol(sym)), Some(Attribute::Type(ty))) =
                (attrs.get(&sym_key), attrs.get(&type_key))
            {
                let fn_name = sym.text(db);
                let path: IdVec<Symbol<'db>> = vec![*sym].into_iter().collect();
                env.add_function(fn_name, path, *ty);
            }
        }
        ("type", "struct") => {
            // Struct definition → creates constructor
            let attrs = op.attributes(db);
            let name_key = Symbol::new(db, "name");
            let type_key = Symbol::new(db, "type");

            if let (Some(Attribute::Symbol(sym)), Some(Attribute::Type(ty))) =
                (attrs.get(&name_key), attrs.get(&type_key))
            {
                let type_name = sym.text(db);
                // Add type definition
                env.add_type(type_name, *ty);
                // Struct constructor has same name as type
                // TODO: Get field types for constructor params
                env.add_constructor(type_name, *ty, None, IdVec::new());
            }
        }
        ("type", "enum") => {
            // Enum definition → creates constructors for each variant
            let attrs = op.attributes(db);
            let name_key = Symbol::new(db, "name");
            let type_key = Symbol::new(db, "type");

            if let (Some(Attribute::Symbol(sym)), Some(Attribute::Type(ty))) =
                (attrs.get(&name_key), attrs.get(&type_key))
            {
                let type_name = sym.text(db);
                // Add type definition
                env.add_type(type_name, *ty);

                // Extract variants from the operation's regions or attributes
                // Each variant becomes a constructor in the type's namespace
                collect_enum_constructors(db, env, op, type_name, *ty);
            }
        }
        ("core", "module") => {
            // Nested module → collect definitions into a namespace
            let attrs = op.attributes(db);
            let sym_key = Symbol::new(db, "sym_name");

            if let Some(Attribute::Symbol(sym)) = attrs.get(&sym_key) {
                let mod_name = sym.text(db);

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
                    env.add_to_namespace(mod_name, name, binding.clone());
                }

                // Also add nested namespaces (e.g., mod::submod::item)
                for (nested_ns, nested_bindings) in mod_env.namespaces.iter() {
                    let qualified_ns = format!("{}::{}", mod_name, nested_ns);
                    for (name, binding) in nested_bindings.iter() {
                        env.add_to_namespace(&qualified_ns, name, binding.clone());
                    }
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
    type_name: &str,
    ty: Type<'db>,
) {
    // Check for variants attribute
    let attrs = op.attributes(db);
    let variants_key = Symbol::new(db, "variants");

    if let Some(Attribute::SymbolRef(variants)) = attrs.get(&variants_key) {
        for variant_sym in variants.iter() {
            let variant_name = variant_sym.text(db);
            // Add to type's namespace
            env.add_to_namespace(
                type_name,
                variant_name,
                Binding::Constructor {
                    ty,
                    tag: Some(*variant_sym),
                    params: IdVec::new(), // TODO: Get variant params
                },
            );
            // Also add unqualified if it doesn't conflict
            if env.lookup(variant_name).is_none() {
                env.add_constructor(variant_name, ty, Some(*variant_sym), IdVec::new());
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
}

/// Name resolver context.
///
/// Transforms `src.*` operations into resolved operations (`func.*`, `adt.*`).
pub struct Resolver<'db> {
    db: &'db dyn salsa::Database,
    env: ModuleEnv<'db>,
    /// Maps old values to their replacements during transformation.
    value_map: HashMap<Value<'db>, Value<'db>>,
    /// Local scope stack (for function parameters, let bindings).
    /// Each entry is a scope level mapping names to local bindings.
    local_scopes: Vec<HashMap<String, LocalBinding<'db>>>,
    /// If true, emit diagnostics for unresolved references instead of passing through.
    report_unresolved: bool,
}

impl<'db> Resolver<'db> {
    /// Create a new resolver with the given environment.
    pub fn new(db: &'db dyn salsa::Database, env: ModuleEnv<'db>) -> Self {
        Self {
            db,
            env,
            value_map: HashMap::new(),
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
            value_map: HashMap::new(),
            local_scopes: Vec::new(),
            report_unresolved: true,
        }
    }

    /// Get the environment.
    pub fn env(&self) -> &ModuleEnv<'db> {
        &self.env
    }

    /// Look up a mapped value, or return the original if not mapped.
    fn lookup_value(&self, old: Value<'db>) -> Value<'db> {
        self.value_map.get(&old).copied().unwrap_or(old)
    }

    /// Map a value from old to new.
    fn map_value(&mut self, old: Value<'db>, new: Value<'db>) {
        self.value_map.insert(old, new);
    }

    /// Push a new local scope.
    fn push_scope(&mut self) {
        self.local_scopes.push(HashMap::new());
    }

    /// Pop the current local scope.
    fn pop_scope(&mut self) {
        self.local_scopes.pop();
    }

    /// Add a local binding to the current scope.
    fn add_local(&mut self, name: String, binding: LocalBinding<'db>) {
        if let Some(scope) = self.local_scopes.last_mut() {
            scope.insert(name, binding);
        }
    }

    /// Look up a name in local scopes (innermost first).
    fn lookup_local(&self, name: &str) -> Option<&LocalBinding<'db>> {
        for scope in self.local_scopes.iter().rev() {
            if let Some(binding) = scope.get(name) {
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
        if ty.dialect(self.db).text(self.db) == "src" && ty.name(self.db).text(self.db) == "type" {
            // Get the type name from the name attribute (stored as Symbol)
            if let Some(Attribute::Symbol(name_sym)) = ty.get_attr(self.db, "name") {
                return self.resolve_type_name(name_sym.text(self.db));
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
    fn resolve_type_name(&self, name: &str) -> Type<'db> {
        match name {
            // Primitive types
            "Int" => *core::I64::new(self.db),
            "Bool" => *core::I1::new(self.db),
            "Float" => *core::F64::new(self.db),
            "Nat" => *core::I64::new(self.db), // Nat is implemented as i64 for now
            // TODO: String type - for now, leave unresolved
            // "String" => ...,

            // Look up user-defined types in the environment
            _ => {
                if let Some(Binding::TypeDef { ty }) = self.env.lookup(name) {
                    *ty
                } else {
                    // Leave unresolved - will be caught by type checker
                    Type::new(
                        self.db,
                        Symbol::new(self.db, "src"),
                        Symbol::new(self.db, "type"),
                        IdVec::new(),
                        Attrs::new(),
                    )
                }
            }
        }
    }

    /// Resolve names in a module.
    ///
    /// Returns the module with `src.*` operations transformed to resolved forms.
    pub fn resolve_module(&mut self, module: &Module<'db>) -> Module<'db> {
        let body = module.body(self.db);
        let new_body = self.resolve_region(&body);

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

        Block::new(self.db, block.location(self.db), new_args, new_ops)
    }

    /// Resolve a single operation.
    ///
    /// Returns the resolved operation(s). May return empty vec if erased,
    /// or multiple ops if expanded.
    fn resolve_operation(&mut self, op: &Operation<'db>) -> Vec<Operation<'db>> {
        // First, remap operands from previous transformations
        let remapped_op = self.remap_operands(op);

        let dialect = remapped_op.dialect(self.db).text(self.db);
        let op_name = remapped_op.name(self.db).text(self.db);

        match (dialect, op_name) {
            ("func", "func") => {
                // Handle function with local scope for parameters
                vec![self.resolve_func(&remapped_op)]
            }
            ("src", "var") => {
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
                            let dialect = ty.dialect(self.db).text(self.db);
                            // Type is resolved if it's not a source type placeholder and not a type variable
                            dialect != "src" && dialect != "type"
                        });

                    if self.report_unresolved && !is_already_resolved {
                        self.emit_unresolved_var_diagnostic(&remapped_op);
                    }
                    vec![self.resolve_op_regions(&remapped_op)]
                }
            }
            ("src", "path") => {
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
            ("src", "call") => {
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

        Operation::new(
            self.db,
            op.location(self.db),
            op.dialect(self.db),
            op.name(self.db),
            op.operands(self.db).clone(),
            op.results(self.db).clone(),
            op.attributes(self.db).clone(),
            new_regions,
            op.successors(self.db).clone(),
        )
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

            if op.dialect(self.db).text(self.db) == "src" && op.name(self.db).text(self.db) == "var"
            {
                // Only consider as parameter declaration if span matches function span
                // Body references have their own specific span, not the function span
                let op_span = op.location(self.db).span;
                if op_span != func_span {
                    break; // Different span means it's a body reference, not a param decl
                }

                // This is a parameter declaration
                let attrs = op.attributes(self.db);
                let name_key = Symbol::new(self.db, "name");
                if let Some(Attribute::Symbol(sym)) = attrs.get(&name_key) {
                    param_declarations.push((sym.text(self.db).to_string(), *op));
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
                let block_arg = Value::new(self.db, ValueDef::BlockArg(*block), i);
                let param_ty = block_args[i];

                // Add to local scope
                self.add_local(
                    name.clone(),
                    LocalBinding::Parameter {
                        value: block_arg,
                        ty: param_ty,
                    },
                );

                // Map the src.var result to the block argument
                let old_result = op.result(self.db, 0);
                self.map_value(old_result, block_arg);
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

        Block::new(self.db, block.location(self.db), new_args, new_ops)
    }

    /// Remap operands using the current value map.
    fn remap_operands(&self, op: &Operation<'db>) -> Operation<'db> {
        let operands = op.operands(self.db);
        let mut new_operands: IdVec<Value<'db>> = IdVec::new();
        let mut changed = false;

        for &operand in operands.iter() {
            let mapped = self.lookup_value(operand);
            new_operands.push(mapped);
            if mapped != operand {
                changed = true;
            }
        }

        if !changed {
            return *op;
        }

        Operation::new(
            self.db,
            op.location(self.db),
            op.dialect(self.db),
            op.name(self.db),
            new_operands,
            op.results(self.db).clone(),
            op.attributes(self.db).clone(),
            op.regions(self.db).clone(),
            op.successors(self.db).clone(),
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

        Operation::new(
            self.db,
            op.location(self.db),
            op.dialect(self.db),
            op.name(self.db),
            op.operands(self.db).clone(),
            new_results,
            op.attributes(self.db).clone(),
            new_regions,
            op.successors(self.db).clone(),
        )
    }

    /// Try to resolve a `src.var` operation.
    ///
    /// Returns:
    /// - Some(vec![]) if resolved to a local binding (erased, value already mapped)
    /// - Some(vec![op]) if resolved to a function/constructor (replaced)
    /// - None if unresolved
    fn try_resolve_var(&mut self, op: &Operation<'db>) -> Option<Vec<Operation<'db>>> {
        let attrs = op.attributes(self.db);
        let name_key = Symbol::new(self.db, "name");
        let Attribute::Symbol(sym) = attrs.get(&name_key)? else {
            return None;
        };
        let name = sym.text(self.db);
        let location = op.location(self.db);

        // First, check local scopes (function parameters, let bindings)
        if let Some(local) = self.lookup_local(name) {
            // Local binding found - keep src.var with resolved type for hover span
            let (value, ty) = match local {
                LocalBinding::Parameter { value, ty } => (*value, *ty),
                LocalBinding::LetBinding { value, ty } => (*value, *ty),
            };

            // Resolve the type (it may still be src.type at this point)
            let resolved_ty = self.resolve_type(ty);

            // Create new src.var with resolved type (keeps span for hover)
            let new_op = src::var(self.db, location, resolved_ty, *sym);
            let new_operation = new_op.as_operation();

            // Map old result to the actual bound value (not the new src.var's result)
            // This ensures use sites get the correct value
            let old_result = op.result(self.db, 0);
            self.map_value(old_result, value);

            // Return the new src.var to keep it in IR for hover
            return Some(vec![new_operation]);
        }

        // Then check module environment
        match self.env.lookup(name)? {
            Binding::Function { path, ty } => {
                // Create func.constant operation
                // func::constant(db, location, result_type, func_ref)
                let new_op = func::constant(self.db, location, *ty, path.clone());
                let new_operation = new_op.as_operation();

                // Map old result to new result
                let old_result = op.result(self.db, 0);
                let new_result = new_operation.result(self.db, 0);
                self.map_value(old_result, new_result);

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
                self.map_value(old_result, new_result);

                Some(vec![new_operation])
            }
            Binding::TypeDef { .. } => {
                // Type used in value position - error (leave unresolved for diagnostics)
                None
            }
        }
    }

    /// Try to resolve a `src.path` operation.
    fn try_resolve_path(&mut self, op: &Operation<'db>) -> Option<Operation<'db>> {
        let attrs = op.attributes(self.db);
        let path_key = Symbol::new(self.db, "path");
        let Attribute::SymbolRef(segments) = attrs.get(&path_key)? else {
            return None;
        };

        if segments.len() < 2 {
            return None;
        }

        let location = op.location(self.db);

        // For now, only support 2-segment paths (Namespace::name)
        let namespace = segments[0].text(self.db);
        let name = segments[1].text(self.db);

        match self.env.lookup_qualified(namespace, name)? {
            Binding::Function { path, ty } => {
                let new_op = func::constant(self.db, location, *ty, path.clone());
                let new_operation = new_op.as_operation();

                let old_result = op.result(self.db, 0);
                let new_result = new_operation.result(self.db, 0);
                self.map_value(old_result, new_result);

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
                self.map_value(old_result, new_result);

                Some(new_operation)
            }
            Binding::TypeDef { .. } => None,
        }
    }

    /// Try to resolve a `src.call` operation.
    fn try_resolve_call(&mut self, op: &Operation<'db>) -> Option<Operation<'db>> {
        let attrs = op.attributes(self.db);
        let name_key = Symbol::new(self.db, "name");
        let Attribute::SymbolRef(path_segments) = attrs.get(&name_key)? else {
            return None;
        };

        if path_segments.is_empty() {
            return None;
        }

        let location = op.location(self.db);
        let result_ty = op.results(self.db).first().copied()?;
        let args: Vec<Value<'db>> = op.operands(self.db).iter().copied().collect();

        // Try to resolve the callee
        let callee_name = if path_segments.len() == 1 {
            // Unqualified name
            path_segments[0].text(self.db)
        } else {
            // Qualified path - for now just use the last segment
            // TODO: Proper qualified lookup
            path_segments.last()?.text(self.db)
        };

        match self.env.lookup(callee_name)? {
            Binding::Function { path, .. } => {
                // Direct function call
                // func::call(db, location, args, result_type, callee)
                let new_op = func::call(self.db, location, args, result_ty, path.clone());
                let new_operation = new_op.as_operation();

                let old_result = op.result(self.db, 0);
                let new_result = new_operation.result(self.db, 0);
                self.map_value(old_result, new_result);

                Some(new_operation)
            }
            Binding::Constructor { ty, tag, .. } => {
                // Constructor application
                let new_operation = if let Some(tag) = tag {
                    // variant_new(db, location, fields, result_type, ty, tag)
                    adt::variant_new(self.db, location, args, *ty, *ty, *tag).as_operation()
                } else {
                    // struct_new(db, location, fields, result_type, ty)
                    adt::struct_new(self.db, location, args, *ty, *ty).as_operation()
                };

                let old_result = op.result(self.db, 0);
                let new_result = new_operation.result(self.db, 0);
                self.map_value(old_result, new_result);

                Some(new_operation)
            }
            Binding::TypeDef { .. } => None,
        }
    }

    // === Diagnostic Helpers ===

    /// Emit diagnostic for unresolved `src.var`.
    fn emit_unresolved_var_diagnostic(&self, op: &Operation<'db>) {
        let name_key = Symbol::new(self.db, "name");
        let name = op
            .attributes(self.db)
            .get(&name_key)
            .and_then(|a| {
                if let Attribute::Symbol(s) = a {
                    Some(s.text(self.db))
                } else {
                    None
                }
            })
            .unwrap_or("unknown");

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
        let path_key = Symbol::new(self.db, "path");
        let path = op
            .attributes(self.db)
            .get(&path_key)
            .and_then(|a| {
                if let Attribute::SymbolRef(segments) = a {
                    Some(
                        segments
                            .iter()
                            .map(|s| s.text(self.db))
                            .collect::<Vec<_>>()
                            .join("::"),
                    )
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
        let name_key = Symbol::new(self.db, "name");
        let name = op
            .attributes(self.db)
            .get(&name_key)
            .and_then(|a| {
                if let Attribute::SymbolRef(segments) = a {
                    Some(
                        segments
                            .iter()
                            .map(|s| s.text(self.db))
                            .collect::<Vec<_>>()
                            .join("::"),
                    )
                } else if let Attribute::Symbol(s) = a {
                    Some(s.text(self.db).to_string())
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
mod tests {
    use super::*;
    use salsa::Database;
    use tribute_core::TributeDatabaseImpl;

    #[test]
    fn test_module_env_lookup() {
        // Basic smoke test for ModuleEnv
        let env: ModuleEnv<'_> = ModuleEnv::new();
        assert!(env.lookup("foo").is_none());
    }

    #[test]
    fn test_build_env_from_module() {
        TributeDatabaseImpl::default().attach(|db| {
            use crate::cst_to_tir::{lower_cst, parse_cst};
            use tribute_core::SourceFile;

            let source = SourceFile::new(
                db,
                std::path::PathBuf::from("test.tr"),
                "fn hello() -> Int { 42 }".to_string(),
            );

            let cst = parse_cst(db, source).expect("parse should succeed");
            let module = lower_cst(db, source, cst);

            let env = build_env(db, &module);

            // Should find the 'hello' function
            assert!(env.lookup("hello").is_some());
            match env.lookup("hello") {
                Some(Binding::Function { path, .. }) => {
                    assert_eq!(path.len(), 1);
                }
                _ => panic!("Expected function binding"),
            }
        });
    }

    #[test]
    fn test_nested_module_resolution() {
        TributeDatabaseImpl::default().attach(|db| {
            use crate::cst_to_tir::{lower_cst, parse_cst};
            use tribute_core::SourceFile;

            let source = SourceFile::new(
                db,
                std::path::PathBuf::from("test.tr"),
                r#"
                    pub mod math {
                        pub fn add(x: Int, y: Int) -> Int { x + y }
                        pub fn sub(x: Int, y: Int) -> Int { x - y }
                    }
                "#
                .to_string(),
            );

            let cst = parse_cst(db, source).expect("parse should succeed");
            let module = lower_cst(db, source, cst);

            let env = build_env(db, &module);

            // Should find math::add and math::sub
            assert!(
                env.lookup_qualified("math", "add").is_some(),
                "should find math::add"
            );
            assert!(
                env.lookup_qualified("math", "sub").is_some(),
                "should find math::sub"
            );

            // Should not find 'add' at top level
            assert!(env.lookup("add").is_none(), "add should not be at top level");
        });
    }

    #[test]
    fn test_deeply_nested_module_resolution() {
        TributeDatabaseImpl::default().attach(|db| {
            use crate::cst_to_tir::{lower_cst, parse_cst};
            use tribute_core::SourceFile;

            let source = SourceFile::new(
                db,
                std::path::PathBuf::from("test.tr"),
                r#"
                    pub mod outer {
                        pub mod inner {
                            pub fn deep() -> Int { 42 }
                        }
                    }
                "#
                .to_string(),
            );

            let cst = parse_cst(db, source).expect("parse should succeed");
            let module = lower_cst(db, source, cst);

            let env = build_env(db, &module);

            // Should find outer::inner::deep
            assert!(
                env.lookup_qualified("outer::inner", "deep").is_some(),
                "should find outer::inner::deep"
            );

            // Should find outer::inner as a module
            // (inner is in outer's namespace)
            // Note: We don't track modules as bindings yet, so this tests the qualified path
        });
    }
}
