//! Lowering context for CST to TrunkIR conversion.

use std::collections::HashMap;

use tree_sitter::Node;
use tribute_trunk_ir::{
    DialectType, IdVec, Type, Value,
    dialect::{core, ty},
};
use tribute_trunk_ir::{Location, PathId};

use super::helpers::{node_text, span_from_node};

/// Context for lowering, tracking local variable bindings and type variable generation.
pub struct CstLoweringCtx<'db, 'src> {
    pub db: &'db dyn salsa::Database,
    pub path: PathId<'db>,
    pub source: &'src str,
    /// Map from variable names to their SSA values.
    bindings: HashMap<String, Value<'db>>,
    /// Map from type variable names to their Type representations.
    type_var_bindings: HashMap<String, Type<'db>>,
    /// Counter for generating unique type variable IDs.
    next_type_var_id: u64,
    /// Counter for generating unique effect row variable IDs.
    next_row_var_id: u64,
}

impl<'db, 'src> CstLoweringCtx<'db, 'src> {
    pub fn new(db: &'db dyn salsa::Database, path: PathId<'db>, source: &'src str) -> Self {
        Self {
            db,
            path,
            source,
            bindings: HashMap::new(),
            type_var_bindings: HashMap::new(),
            next_type_var_id: 0,
            next_row_var_id: 0,
        }
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
    pub fn named_type_var(&mut self, name: &str) -> Type<'db> {
        if let Some(&ty) = self.type_var_bindings.get(name) {
            ty
        } else {
            let ty = self.fresh_type_var();
            self.type_var_bindings.insert(name.to_string(), ty);
            ty
        }
    }

    /// Resolve a type node to an IR Type.
    pub fn resolve_type_node(&mut self, node: Node) -> Type<'db> {
        use tribute_trunk_ir::dialect::src;

        let mut cursor = node.walk();
        match node.kind() {
            "type_identifier" => {
                // Concrete named type
                let name = node_text(&node, self.source);
                src::unresolved_type(self.db, name, IdVec::new())
            }
            "type_variable" => {
                // Type variable (lowercase)
                let name = node_text(&node, self.source);
                self.named_type_var(name)
            }
            "generic_type" => {
                // Generic type: List(a), Option(b)
                let mut name = None;
                let mut args = Vec::new();

                for child in node.named_children(&mut cursor) {
                    match child.kind() {
                        "type_identifier" if name.is_none() => {
                            name = Some(node_text(&child, self.source));
                        }
                        "type_variable" | "type_identifier" | "generic_type" => {
                            args.push(self.resolve_type_node(child));
                        }
                        _ => {}
                    }
                }

                let name = name.unwrap_or("Unknown");
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
    pub fn bind(&mut self, name: String, value: Value<'db>) {
        self.bindings.insert(name, value);
    }

    /// Look up a binding by name.
    pub fn lookup(&self, name: &str) -> Option<Value<'db>> {
        self.bindings.get(name).copied()
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
