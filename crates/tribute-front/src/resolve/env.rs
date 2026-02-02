//! Module environment for name resolution.
//!
//! This module provides structures for tracking definitions and looking up names
//! during the name resolution phase.

use std::collections::HashMap;

use trunk_ir::Symbol;

use crate::ast::{AbilityId, CtorId, FuncDefId, TypeDefId};

/// Information about a resolved name binding.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
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
        /// The namespace path.
        path: Vec<Symbol>,
    },

    /// An ability operation.
    ///
    /// Unlike regular functions, ability operations are lowered to `cont.shift`
    /// with runtime evidence lookup.
    AbilityOp {
        /// The ability identifier (e.g., AbilityId for "State").
        ability: AbilityId<'db>,
        /// The operation name (e.g., "get").
        op: Symbol,
    },
}

/// Module environment for name resolution.
///
/// Tracks all names visible in the current module.
#[derive(Clone, Debug, Default, PartialEq, Eq, salsa::Update)]
pub struct ModuleEnv<'db> {
    /// Names defined in this module (simple name → binding).
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

    /// Add a qualified name to a namespace.
    pub fn add_to_namespace(&mut self, namespace: Symbol, name: Symbol, binding: Binding<'db>) {
        self.namespaces
            .entry(namespace)
            .or_default()
            .insert(name, binding);
    }

    /// Add an import.
    pub fn add_import(&mut self, name: Symbol, binding: Binding<'db>) {
        self.imports.insert(name, binding);
    }

    /// Add an import only if no binding exists for that name.
    ///
    /// This is used to expose prelude definitions at the top level
    /// without overriding user-defined names.
    pub fn add_import_if_absent(&mut self, name: Symbol, binding: Binding<'db>) {
        if !self.definitions.contains_key(&name) && !self.imports.contains_key(&name) {
            self.imports.insert(name, binding);
        }
    }

    /// Add a qualified name to a namespace only if it doesn't already exist.
    pub fn add_to_namespace_if_absent(
        &mut self,
        namespace: Symbol,
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

    /// Check whether a namespace exists.
    pub fn has_namespace(&self, namespace: Symbol) -> bool {
        self.namespaces.contains_key(&namespace)
    }

    /// Iterate over all definitions in this environment.
    pub fn iter_definitions(&self) -> impl Iterator<Item = (Symbol, &Binding<'db>)> {
        self.definitions.iter().map(|(k, v)| (*k, v))
    }

    /// Iterate over all namespaces in this environment.
    pub fn iter_namespaces(
        &self,
    ) -> impl Iterator<Item = (Symbol, impl Iterator<Item = (Symbol, &Binding<'db>)>)> {
        self.namespaces
            .iter()
            .map(|(ns, bindings)| (*ns, bindings.iter().map(|(k, v)| (*k, v))))
    }

    /// Merge another environment into this one.
    ///
    /// Self takes precedence: user definitions shadow prelude.
    /// This is used to inject prelude bindings into user code's environment.
    pub fn merge(&mut self, other: &ModuleEnv<'db>) {
        // Add other's definitions as imports (so they don't override user definitions)
        for (name, binding) in other.iter_definitions() {
            self.add_import_if_absent(name, binding.clone());
        }

        // Add other's namespaces
        for (ns, bindings) in other.iter_namespaces() {
            for (name, binding) in bindings {
                self.add_to_namespace_if_absent(ns, name, binding.clone());
            }
        }
    }
}
