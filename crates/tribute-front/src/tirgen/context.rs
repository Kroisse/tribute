//! Lowering context for CST to TrunkIR conversion.

use std::collections::HashMap;

use ropey::Rope;
use tree_sitter::Node;
use trunk_ir::{
    DialectType, IdVec, QualifiedName, Symbol, SymbolVec, Type, Value,
    dialect::{core, ty},
};
use trunk_ir::{Location, PathId};

use super::helpers::{node_text, span_from_node};

/// Context for lowering, tracking local variable bindings and type variable generation.
pub struct CstLoweringCtx<'db> {
    pub db: &'db dyn salsa::Database,
    pub path: PathId<'db>,
    pub source: Rope,
    /// Map from variable names to their SSA values.
    bindings: HashMap<Symbol, Value<'db>>,
    /// Map from type variable names to their Type representations.
    type_var_bindings: HashMap<Symbol, Type<'db>>,
    /// Counter for generating unique type variable IDs.
    next_type_var_id: u64,
    /// Counter for generating unique effect row variable IDs.
    next_row_var_id: u64,
    /// Current module path for qualified name generation.
    module_path: SymbolVec,
}

impl<'db> CstLoweringCtx<'db> {
    pub fn new(db: &'db dyn salsa::Database, path: PathId<'db>, source: Rope) -> Self {
        Self {
            db,
            path,
            source,
            bindings: HashMap::new(),
            type_var_bindings: HashMap::new(),
            next_type_var_id: 0,
            next_row_var_id: 0,
            module_path: SymbolVec::new(),
        }
    }

    /// Create a qualified name by prepending the current module path.
    pub fn qualified_name(&self, name: Symbol) -> QualifiedName {
        QualifiedName::new(self.module_path.clone(), name)
    }

    /// Enter a module scope.
    pub fn enter_module(&mut self, name: Symbol) {
        self.module_path.push(name);
    }

    /// Exit a module scope.
    pub fn exit_module(&mut self) {
        self.module_path.pop();
    }

    /// Generate a fresh type variable with a unique ID.
    pub fn fresh_type_var(&mut self) -> Type<'db> {
        let id = self.next_type_var_id;
        self.next_type_var_id += 1;
        ty::var_with_id(self.db, id)
    }

    /// Generate a fresh effect row type with a unique tail variable.
    pub fn fresh_effect_row_type(&mut self) -> Type<'db> {
        let id = self.next_row_var_id;
        self.next_row_var_id += 1;
        core::EffectRowType::var(self.db, id).as_type()
    }

    /// Get or create a named type variable.
    /// Same name always returns the same type variable within a scope.
    pub fn named_type_var(&mut self, name: Symbol) -> Type<'db> {
        if let Some(&ty) = self.type_var_bindings.get(&name) {
            ty
        } else {
            let ty = self.fresh_type_var();
            self.type_var_bindings.insert(name, ty);
            ty
        }
    }

    /// Resolve a type node to an IR Type.
    pub fn resolve_type_node(&mut self, node: Node) -> Type<'db> {
        use trunk_ir::dialect::src;

        let mut cursor = node.walk();
        match node.kind() {
            "type_identifier" => {
                // Concrete named type
                let name = node_text(&node, &self.source);
                src::unresolved_type(self.db, name.into(), IdVec::new())
            }
            "type_variable" => {
                // Type variable (lowercase)
                let name = node_text(&node, &self.source);
                self.named_type_var(name.into())
            }
            "generic_type" => {
                // Generic type: List(a), Option(b)
                let mut name = None;
                let mut args = Vec::new();

                for child in node.named_children(&mut cursor) {
                    match child.kind() {
                        "type_identifier" if name.is_none() => {
                            name = Some(Symbol::from(node_text(&child, &self.source)));
                        }
                        "type_variable" | "type_identifier" | "generic_type" => {
                            args.push(self.resolve_type_node(child));
                        }
                        _ => {}
                    }
                }

                let name = name.unwrap_or(Symbol::new("Unknown"));
                let params: IdVec<Type<'db>> = args.into_iter().collect();
                src::unresolved_type(self.db, name, params)
            }
            _ => {
                // Fallback to fresh type var
                self.fresh_type_var()
            }
        }
    }

    /// Bind a name to a value.
    pub fn bind(&mut self, name: Symbol, value: Value<'db>) {
        self.bindings.insert(name, value);
    }

    /// Look up a binding by name.
    pub fn lookup(&self, name: Symbol) -> Option<Value<'db>> {
        self.bindings.get(&name).copied()
    }

    /// Execute a closure in a new scope. Bindings created inside are discarded after.
    pub fn scoped<R>(&mut self, f: impl FnOnce(&mut Self) -> R) -> R {
        let saved_bindings = self.bindings.clone();
        let saved_type_vars = self.type_var_bindings.clone();
        let result = f(self);
        self.bindings = saved_bindings;
        self.type_var_bindings = saved_type_vars;
        result
    }

    /// Create a Location from a node.
    pub fn location(&self, node: &Node) -> Location<'db> {
        Location::new(self.path, span_from_node(node))
    }
}
