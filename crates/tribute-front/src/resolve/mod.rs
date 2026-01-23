//! Name resolution for the AST.
//!
//! This module transforms `Module<UnresolvedName>` into `Module<ResolvedRef<'db>>`.
//!
//! ## Pipeline
//!
//! 1. Build a `ModuleEnv` from declarations
//! 2. Walk the AST, resolving each name reference
//!
//! ## Resolution Strategy
//!
//! Names are resolved in this order:
//! 1. Local variables (function parameters, let bindings, pattern bindings)
//! 2. Builtin operations
//! 3. Module-level definitions (functions, types, constructors)

mod env;
mod resolver;

pub use env::{Binding, ModuleEnv};
pub use resolver::Resolver;

use trunk_ir::Symbol;

use crate::ast::{CtorId, Decl, FuncDefId, Module, ResolvedRef, UnresolvedName};

/// Resolve names in a module.
///
/// This is the main entry point for name resolution.
pub fn resolve_module<'db>(
    db: &'db dyn salsa::Database,
    module: Module<UnresolvedName>,
) -> Module<ResolvedRef<'db>> {
    // Build the module environment from declarations
    let env = build_env(db, &module);

    // Create resolver and process the module
    let mut resolver = Resolver::new(db, env);
    resolver.resolve_module(module)
}

/// Build a module environment from AST declarations.
fn build_env<'db>(db: &'db dyn salsa::Database, module: &Module<UnresolvedName>) -> ModuleEnv<'db> {
    let mut env = ModuleEnv::new();

    for decl in &module.decls {
        collect_definition(db, &mut env, decl);
    }

    env
}

/// Collect a definition from a declaration into the environment.
fn collect_definition<'db>(
    db: &'db dyn salsa::Database,
    env: &mut ModuleEnv<'db>,
    decl: &Decl<UnresolvedName>,
) {
    match decl {
        Decl::Function(func) => {
            let id = FuncDefId::new(db, func.name);
            env.add_function(func.name, id);
        }

        Decl::Struct(s) => {
            // Struct is both a type and a constructor
            let ctor_id = CtorId::new(db, s.name);
            env.add_type(s.name, Some(ctor_id));
            env.add_constructor(s.name, ctor_id, None, s.fields.len());
        }

        Decl::Enum(e) => {
            // Enum is a type, and each variant is a constructor
            let ctor_id = CtorId::new(db, e.name);
            env.add_type(e.name, Some(ctor_id));

            // Add each variant as a constructor in the enum's namespace
            for variant in &e.variants {
                let variant_id = CtorId::new(db, variant.name);
                let binding = Binding::Constructor {
                    id: variant_id,
                    tag: Some(variant.name),
                    arity: variant.fields.len(),
                };
                // Add to namespace (e.g., Option::Some)
                env.add_to_namespace(e.name, variant.name, binding.clone());
                // Also add directly for unqualified access (e.g., Some)
                env.add_constructor(
                    variant.name,
                    variant_id,
                    Some(variant.name),
                    variant.fields.len(),
                );
            }
        }

        Decl::Ability(a) => {
            // Ability operations are added to the ability's namespace
            for op in &a.operations {
                let func_id = FuncDefId::new(
                    db,
                    Symbol::from_dynamic(&format!("{}::{}", a.name, op.name)),
                );
                let binding = Binding::Function { id: func_id };
                env.add_to_namespace(a.name, op.name, binding);
            }
        }

        Decl::Use(u) => {
            // Import the last segment of the path
            if let Some(&name) = u.path.last() {
                let binding = Binding::Module {
                    path: u.path.clone(),
                };
                let import_name = u.alias.unwrap_or(name);
                env.add_import(import_name, binding);
            }
        }

        Decl::Const(c) => {
            // Constants are treated like functions for name resolution purposes.
            // They can be referenced by name and resolved to a FuncDefId.
            let id = FuncDefId::new(db, c.name);
            env.add_function(c.name, id);
        }
    }
}
