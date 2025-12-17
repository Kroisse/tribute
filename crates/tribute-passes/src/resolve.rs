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

use tribute_trunk_ir::dialect::adt;
use tribute_trunk_ir::dialect::core::Module;
use tribute_trunk_ir::dialect::func;
use tribute_trunk_ir::{
    Attribute, Block, DialectOp, IdVec, Operation, Region, Symbol, Type, Value,
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
            let type_key = Symbol::new(db, "r#type"); // Note: r#type because 'type' is reserved

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
            let type_key = Symbol::new(db, "r#type"); // Note: r#type because 'type' is reserved

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
            let type_key = Symbol::new(db, "r#type"); // Note: r#type because 'type' is reserved

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

/// Name resolver context.
///
/// Transforms `src.*` operations into resolved operations (`func.*`, `adt.*`).
pub struct Resolver<'db> {
    db: &'db dyn salsa::Database,
    env: ModuleEnv<'db>,
    /// Maps old values to their replacements during transformation.
    value_map: HashMap<Value<'db>, Value<'db>>,
}

impl<'db> Resolver<'db> {
    /// Create a new resolver with the given environment.
    pub fn new(db: &'db dyn salsa::Database, env: ModuleEnv<'db>) -> Self {
        Self {
            db,
            env,
            value_map: HashMap::new(),
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
        let new_ops: IdVec<Operation<'db>> = block
            .operations(self.db)
            .iter()
            .flat_map(|op| self.resolve_operation(op))
            .collect();

        Block::new(
            self.db,
            block.location(self.db),
            block.args(self.db).clone(),
            new_ops,
        )
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
            ("src", "var") => {
                if let Some(resolved) = self.try_resolve_var(&remapped_op) {
                    vec![resolved]
                } else {
                    // Unresolved - keep for later (UFCS resolution)
                    vec![self.resolve_op_regions(&remapped_op)]
                }
            }
            ("src", "path") => {
                if let Some(resolved) = self.try_resolve_path(&remapped_op) {
                    vec![resolved]
                } else {
                    // Unresolved - keep for later
                    vec![self.resolve_op_regions(&remapped_op)]
                }
            }
            ("src", "call") => {
                if let Some(resolved) = self.try_resolve_call(&remapped_op) {
                    vec![resolved]
                } else {
                    // Unresolved - keep for later
                    vec![self.resolve_op_regions(&remapped_op)]
                }
            }
            _ => {
                // Not a src.* operation - recursively process regions
                vec![self.resolve_op_regions(&remapped_op)]
            }
        }
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

    /// Recursively resolve regions within an operation.
    fn resolve_op_regions(&mut self, op: &Operation<'db>) -> Operation<'db> {
        let regions = op.regions(self.db);
        if regions.is_empty() {
            return *op;
        }

        let new_regions: IdVec<Region<'db>> = regions
            .iter()
            .map(|region| self.resolve_region(region))
            .collect();

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

    /// Try to resolve a `src.var` operation.
    ///
    /// Returns the resolved operation if successful, None if unresolved.
    fn try_resolve_var(&mut self, op: &Operation<'db>) -> Option<Operation<'db>> {
        let attrs = op.attributes(self.db);
        let name_key = Symbol::new(self.db, "name");
        let Attribute::Symbol(sym) = attrs.get(&name_key)? else {
            return None;
        };
        let name = sym.text(self.db);
        let location = op.location(self.db);

        // Look up in environment
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

                Some(new_operation)
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

                Some(new_operation)
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
}
