//! Lowering context for CST to TrunkIR conversion.

use std::collections::HashMap;

use ropey::Rope;
use tree_sitter::Node;
use tribute_ir::dialect::tribute;
use trunk_ir::{DialectType, IdVec, QualifiedName, Symbol, SymbolVec, Type, Value, dialect::core};
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
            // Start at 1 because EffectRowType::new treats 0 as "no tail"
            next_row_var_id: 1,
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
        tribute::type_var_with_id(self.db, id)
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
        let mut cursor = node.walk();
        match node.kind() {
            "type_identifier" => {
                // Concrete named type
                let name = node_text(&node, &self.source);
                tribute::unresolved_type(self.db, name.into(), IdVec::new())
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
                tribute::unresolved_type(self.db, name, params)
            }
            "function_type" => {
                // Function type: fn(Int, Int) -> Int, fn(a) ->{State(s)} b
                let params = self.resolve_function_type_params(node);
                let return_type = node
                    .child_by_field_name("return_type")
                    .map(|n| self.resolve_type_node(n))
                    .unwrap_or_else(|| self.fresh_type_var());
                let effect = self.resolve_ability_row(node.child_by_field_name("abilities"));

                core::Func::with_effect(self.db, params, return_type, effect).as_type()
            }
            _ => {
                // Fallback to fresh type var
                self.fresh_type_var()
            }
        }
    }

    /// Resolve parameter types from a function_type node.
    fn resolve_function_type_params(&mut self, node: Node) -> IdVec<Type<'db>> {
        let Some(params_node) = node.child_by_field_name("params") else {
            return IdVec::new();
        };

        // params is a type_list: type, type, ...
        let mut cursor = params_node.walk();
        params_node
            .named_children(&mut cursor)
            .map(|child| self.resolve_type_node(child))
            .collect()
    }

    /// Resolve an ability_row node to an effect type.
    ///
    /// - `None` -> fresh effect row variable (implicit polymorphism)
    /// - `Some` with empty `{}` -> empty effect row (pure)
    /// - `Some` with abilities -> concrete effect row
    fn resolve_ability_row(&mut self, node: Option<Node>) -> Option<Type<'db>> {
        let Some(ability_row) = node else {
            // No ability annotation -> implicit effect polymorphism
            return Some(self.fresh_effect_row_type());
        };

        // Find ability_list inside ability_row
        let ability_list = ability_row.child_by_field_name("abilities").or_else(|| {
            // Try to find ability_list as a direct child
            let mut cursor = ability_row.walk();
            ability_row
                .named_children(&mut cursor)
                .find(|c| c.kind() == "ability_list")
        });

        let Some(list_node) = ability_list else {
            // Empty ability row: {} -> pure function
            return Some(core::EffectRowType::empty(self.db).as_type());
        };

        // Parse ability_list: ability_item | ability_tail, ...
        let mut abilities: IdVec<Type<'db>> = IdVec::new();
        let mut tail_var: Option<u64> = None;

        let mut cursor = list_node.walk();
        for child in list_node.named_children(&mut cursor) {
            match child.kind() {
                "ability_item" => {
                    // ability_item: type_identifier with optional type_arguments
                    let ability_type = self.resolve_ability_item(child);
                    abilities.push(ability_type);
                }
                "ability_tail" => {
                    // ability_tail: type_variable (row variable)
                    // Structure: ability_tail > type_variable > identifier
                    // Get the text of the entire ability_tail node (which is just the identifier)
                    let name = node_text(&child, &self.source);
                    let name_sym = Symbol::from(name);

                    // Reuse the type var binding mechanism for row vars
                    // (they share namespace for simplicity)
                    if let Some(&ty) = self.type_var_bindings.get(&name_sym) {
                        // Extract the tail var ID from an existing effect row type
                        if let Some(row) = core::EffectRowType::from_type(self.db, ty) {
                            tail_var = row.tail_var(self.db);
                        } else {
                            // Existing binding is a regular type var, not a row var.
                            // Create a fresh row var for this context.
                            let id = self.next_row_var_id;
                            self.next_row_var_id += 1;
                            tail_var = Some(id);
                        }
                    } else {
                        // Create a fresh row var and store it
                        let id = self.next_row_var_id;
                        self.next_row_var_id += 1;
                        tail_var = Some(id);
                        let row_type = core::EffectRowType::var(self.db, id).as_type();
                        self.type_var_bindings.insert(name_sym, row_type);
                    }
                }
                _ => {}
            }
        }

        // Build the effect row type
        let effect_type = match tail_var {
            Some(id) => core::EffectRowType::with_tail(self.db, abilities, id).as_type(),
            None => {
                if abilities.is_empty() {
                    core::EffectRowType::empty(self.db).as_type()
                } else {
                    core::EffectRowType::concrete(self.db, abilities).as_type()
                }
            }
        };

        Some(effect_type)
    }

    /// Resolve an ability_item node to an AbilityRefType.
    fn resolve_ability_item(&mut self, node: Node) -> Type<'db> {
        let mut cursor = node.walk();
        let mut name: Option<Symbol> = None;
        let mut type_args: IdVec<Type<'db>> = IdVec::new();

        for child in node.named_children(&mut cursor) {
            match child.kind() {
                "type_identifier" if name.is_none() => {
                    name = Some(Symbol::from(node_text(&child, &self.source)));
                }
                "type_arguments" => {
                    // Parse type arguments: (Int), (String, Bool)
                    let mut args_cursor = child.walk();
                    for arg in child.named_children(&mut args_cursor) {
                        type_args.push(self.resolve_type_node(arg));
                    }
                }
                _ => {}
            }
        }

        let ability_name = name.unwrap_or_else(|| Symbol::new("Unknown"));
        if type_args.is_empty() {
            core::AbilityRefType::simple(self.db, ability_name).as_type()
        } else {
            core::AbilityRefType::with_params(self.db, ability_name, type_args).as_type()
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
