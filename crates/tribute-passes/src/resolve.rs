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

use tribute_trunk_ir::dialect::core::Module;
use tribute_trunk_ir::{Attribute, Block, IdVec, Operation, Region, Symbol, Type};

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
pub struct Resolver<'db> {
    db: &'db dyn salsa::Database,
    env: ModuleEnv<'db>,
}

impl<'db> Resolver<'db> {
    /// Create a new resolver with the given environment.
    pub fn new(db: &'db dyn salsa::Database, env: ModuleEnv<'db>) -> Self {
        Self { db, env }
    }

    /// Get the environment.
    pub fn env(&self) -> &ModuleEnv<'db> {
        &self.env
    }

    /// Resolve names in a module.
    ///
    /// Returns the module with `src.*` operations transformed.
    /// Note: Full transformation requires IR rewriting which is complex.
    /// For now, this performs validation and returns the module unchanged.
    /// UFCS resolution happens during type checking.
    pub fn resolve_module(&mut self, module: &Module<'db>) -> Module<'db> {
        // Walk the module and validate/resolve what we can
        let body = module.body(self.db);
        for block in body.blocks(self.db).iter() {
            self.resolve_block(block);
        }

        // TODO: Actually rewrite the IR with resolved operations
        // For now, return unchanged - type checker will handle src.* ops
        *module
    }

    /// Resolve names in a block.
    fn resolve_block(&self, block: &Block<'db>) {
        for op in block.operations(self.db).iter() {
            self.resolve_operation(op);
        }
    }

    /// Resolve names in an operation.
    fn resolve_operation(&self, op: &Operation<'db>) {
        let dialect = op.dialect(self.db).text(self.db);
        let op_name = op.name(self.db).text(self.db);

        match (dialect, op_name) {
            ("src", "var") => {
                // Try to resolve the variable
                if let Some(_resolved) = self.resolve_var(op) {
                    // TODO: Replace op with resolved reference
                }
            }
            ("src", "path") => {
                // Try to resolve the path
                if let Some(_resolved) = self.resolve_path(op) {
                    // TODO: Replace op with resolved reference
                }
            }
            ("src", "call") => {
                // Try to resolve the call target
                // The callee might be a qualified path or unqualified name
            }
            _ => {
                // Recurse into regions
                for region in op.regions(self.db).iter() {
                    self.resolve_region(region);
                }
            }
        }
    }

    /// Resolve names in a region.
    fn resolve_region(&self, region: &Region<'db>) {
        for block in region.blocks(self.db).iter() {
            self.resolve_block(block);
        }
    }

    /// Resolve a `src.var` operation.
    fn resolve_var(&self, op: &Operation<'db>) -> Option<ResolvedRef<'db>> {
        let attrs = op.attributes(self.db);
        let name_key = Symbol::new(self.db, "name");
        let Attribute::Symbol(sym) = attrs.get(&name_key)? else {
            return None;
        };
        let name = sym.text(self.db);

        // Look up in environment
        match self.env.lookup(name)? {
            Binding::Function { path, ty } => Some(ResolvedRef::Function {
                path: path.clone(),
                ty: *ty,
            }),
            Binding::Constructor { ty, tag, params } => Some(ResolvedRef::Constructor {
                ty: *ty,
                tag: *tag,
                params: params.clone(),
            }),
            Binding::TypeDef { .. } => {
                // Type used in value position - error
                None
            }
        }
    }

    /// Resolve a `src.path` operation.
    fn resolve_path(&self, op: &Operation<'db>) -> Option<ResolvedRef<'db>> {
        let attrs = op.attributes(self.db);
        let path_key = Symbol::new(self.db, "path");
        let Attribute::SymbolRef(segments) = attrs.get(&path_key)? else {
            return None;
        };

        if segments.len() < 2 {
            return None;
        }

        // For now, only support 2-segment paths (Namespace::name)
        let namespace = segments[0].text(self.db);
        let name = segments[1].text(self.db);

        match self.env.lookup_qualified(namespace, name)? {
            Binding::Function { path, ty } => Some(ResolvedRef::Function {
                path: path.clone(),
                ty: *ty,
            }),
            Binding::Constructor { ty, tag, params } => Some(ResolvedRef::Constructor {
                ty: *ty,
                tag: *tag,
                params: params.clone(),
            }),
            Binding::TypeDef { .. } => None,
        }
    }
}

/// Result of resolving a name reference.
#[derive(Clone, Debug)]
pub enum ResolvedRef<'db> {
    /// Resolved to a function.
    Function {
        path: IdVec<Symbol<'db>>,
        ty: Type<'db>,
    },
    /// Resolved to a constructor.
    Constructor {
        ty: Type<'db>,
        tag: Option<Symbol<'db>>,
        params: IdVec<Type<'db>>,
    },
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
