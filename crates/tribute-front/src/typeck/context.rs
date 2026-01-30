//! Type checking context.
//!
//! This module provides `ModuleTypeEnv`, which holds module-level type information
//! (function/constructor/type definitions). This is populated during collect_declarations
//! and is read-only afterward.
//!
//! For function-level type inference, see `FunctionInferenceContext` in `func_context.rs`.

use std::collections::HashMap;

use trunk_ir::Symbol;

use crate::ast::{CtorId, EffectRow, FuncDefId, Type, TypeKind, TypeParam, TypeScheme};

// =========================================================================
// ModuleTypeEnv: Module-level type information (read-only after collection)
// =========================================================================

/// Struct field information: (type_params, fields).
/// - type_params: The struct's type parameters for field type generalization
/// - fields: Vec of (field_name, field_type) pairs
pub type StructFieldInfo<'db> = (Vec<TypeParam>, Vec<(Symbol, Type<'db>)>);

/// Module-level type environment.
///
/// This struct holds type information that is shared across all functions in a module:
/// - Function type schemes (polymorphic signatures)
/// - Constructor type schemes (for enum variants)
/// - Type definitions (struct/enum type schemes)
/// - Struct field definitions for UFCS/accessor resolution
///
/// After `collect_declarations` populates this, it becomes read-only during
/// function body type checking.
pub struct ModuleTypeEnv<'db> {
    db: &'db dyn salsa::Database,

    /// Function signatures (polymorphic).
    function_types: HashMap<FuncDefId<'db>, TypeScheme<'db>>,

    /// Constructor types.
    constructor_types: HashMap<CtorId<'db>, TypeScheme<'db>>,

    /// Type definitions (struct/enum names to their types).
    type_defs: HashMap<Symbol, TypeScheme<'db>>,

    /// Struct field definitions: struct_name → (type_params, [(field_name, field_type)])
    struct_fields: HashMap<Symbol, StructFieldInfo<'db>>,

    /// Enum variant information: enum_name → [variant_names]
    /// Used for exhaustiveness checking in case expressions.
    enum_variants: HashMap<Symbol, Vec<Symbol>>,
}

impl<'db> ModuleTypeEnv<'db> {
    /// Create a new empty module type environment.
    pub fn new(db: &'db dyn salsa::Database) -> Self {
        Self {
            db,
            function_types: HashMap::new(),
            constructor_types: HashMap::new(),
            type_defs: HashMap::new(),
            struct_fields: HashMap::new(),
            enum_variants: HashMap::new(),
        }
    }

    /// Get the database.
    pub fn db(&self) -> &'db dyn salsa::Database {
        self.db
    }

    // =========================================================================
    // Registration (used during collect_declarations)
    // =========================================================================

    /// Register a function's type scheme.
    pub fn register_function(&mut self, id: FuncDefId<'db>, scheme: TypeScheme<'db>) {
        self.function_types.insert(id, scheme);
    }

    /// Register a constructor's type scheme.
    pub fn register_constructor(&mut self, id: CtorId<'db>, scheme: TypeScheme<'db>) {
        self.constructor_types.insert(id, scheme);
    }

    /// Register a type definition.
    pub fn register_type_def(&mut self, name: Symbol, scheme: TypeScheme<'db>) {
        self.type_defs.insert(name, scheme);
    }

    /// Register struct field information.
    pub fn register_struct_fields(
        &mut self,
        struct_name: Symbol,
        type_params: Vec<TypeParam>,
        fields: Vec<(Symbol, Type<'db>)>,
    ) {
        self.struct_fields
            .insert(struct_name, (type_params, fields));
    }

    /// Register enum variant information.
    pub fn register_enum_variants(&mut self, enum_name: Symbol, variants: Vec<Symbol>) {
        self.enum_variants.insert(enum_name, variants);
    }

    // =========================================================================
    // Lookup (used during type checking)
    // =========================================================================

    /// Look up a function's type scheme.
    pub fn lookup_function(&self, id: FuncDefId<'db>) -> Option<TypeScheme<'db>> {
        self.function_types.get(&id).copied()
    }

    /// Look up a constructor's type scheme.
    pub fn lookup_constructor(&self, id: CtorId<'db>) -> Option<TypeScheme<'db>> {
        self.constructor_types.get(&id).copied()
    }

    /// Look up a type definition.
    pub fn lookup_type_def(&self, name: Symbol) -> Option<TypeScheme<'db>> {
        self.type_defs.get(&name).copied()
    }

    /// Look up struct field type by struct name and field name.
    /// Returns (type_params, field_type) if found.
    pub fn lookup_struct_field(
        &self,
        struct_name: Symbol,
        field_name: Symbol,
    ) -> Option<(&[TypeParam], Type<'db>)> {
        let (type_params, fields) = self.struct_fields.get(&struct_name)?;
        for (name, ty) in fields {
            if *name == field_name {
                return Some((type_params.as_slice(), *ty));
            }
        }
        None
    }

    /// Return the number of registered constructors.
    pub fn constructor_count(&self) -> usize {
        self.constructor_types.len()
    }

    /// Look up enum variants by enum name.
    pub fn lookup_enum_variants(&self, enum_name: Symbol) -> Option<&[Symbol]> {
        self.enum_variants.get(&enum_name).map(|v| v.as_slice())
    }

    /// Debug: print all registered constructors.
    pub fn debug_print_constructors(&self, db: &'db dyn salsa::Database) {
        eprintln!(
            "DEBUG: Registered constructors ({}):",
            self.constructor_types.len()
        );
        for id in self.constructor_types.keys() {
            eprintln!("  - {:?} (type_name: {:?})", id, id.type_name(db));
        }
    }

    // =========================================================================
    // Prelude injection
    // =========================================================================

    /// Inject prelude's resolved type information into this environment.
    ///
    /// This is called before type checking user code to make prelude's
    /// types available. The injected types contain only BoundVars (no UniVars).
    pub fn inject_prelude(&mut self, exports: &super::PreludeExports<'db>) {
        for (id, scheme) in exports.function_types(self.db) {
            self.function_types.insert(*id, *scheme);
        }
        for (id, scheme) in exports.constructor_types(self.db) {
            self.constructor_types.insert(*id, *scheme);
        }
        for (name, scheme) in exports.type_defs(self.db) {
            self.type_defs.insert(*name, *scheme);
        }
        for (name, info) in exports.struct_fields(self.db) {
            self.struct_fields.insert(*name, info.clone());
        }
        for (name, variants) in exports.enum_variants(self.db) {
            self.enum_variants.insert(*name, variants.clone());
        }
    }

    // =========================================================================
    // Export methods
    // =========================================================================

    /// Export function type schemes as a Vec keyed by Symbol (function name).
    pub fn export_function_types(&self) -> Vec<(Symbol, TypeScheme<'db>)> {
        self.function_types
            .iter()
            .map(|(id, scheme)| (id.name(self.db), *scheme))
            .collect()
    }

    /// Export function types with FuncDefId (for PreludeExports).
    pub fn export_function_types_with_ids(&self) -> Vec<(FuncDefId<'db>, TypeScheme<'db>)> {
        self.function_types.iter().map(|(k, v)| (*k, *v)).collect()
    }

    /// Export constructor types for PreludeExports.
    pub fn export_constructor_types(&self) -> Vec<(CtorId<'db>, TypeScheme<'db>)> {
        self.constructor_types
            .iter()
            .map(|(k, v)| (*k, *v))
            .collect()
    }

    /// Export type definitions for PreludeExports.
    pub fn export_type_defs(&self) -> Vec<(Symbol, TypeScheme<'db>)> {
        self.type_defs.iter().map(|(k, v)| (*k, *v)).collect()
    }

    /// Export struct field definitions for PreludeExports.
    pub fn export_struct_fields(&self) -> Vec<(Symbol, StructFieldInfo<'db>)> {
        self.struct_fields
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect()
    }

    /// Export enum variant information for PreludeExports.
    pub fn export_enum_variants(&self) -> Vec<(Symbol, Vec<Symbol>)> {
        self.enum_variants
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect()
    }

    // =========================================================================
    // Primitive types (convenience methods)
    // =========================================================================

    /// Create the Int type.
    pub fn int_type(&self) -> Type<'db> {
        Type::new(self.db, TypeKind::Int)
    }

    /// Create the Nat type.
    pub fn nat_type(&self) -> Type<'db> {
        Type::new(self.db, TypeKind::Nat)
    }

    /// Create the Float type.
    pub fn float_type(&self) -> Type<'db> {
        Type::new(self.db, TypeKind::Float)
    }

    /// Create the Bool type.
    pub fn bool_type(&self) -> Type<'db> {
        Type::new(self.db, TypeKind::Bool)
    }

    /// Create the String type.
    pub fn string_type(&self) -> Type<'db> {
        Type::new(self.db, TypeKind::String)
    }

    /// Create the Bytes type.
    pub fn bytes_type(&self) -> Type<'db> {
        Type::new(self.db, TypeKind::Bytes)
    }

    /// Create the Rune type (Unicode code point).
    pub fn rune_type(&self) -> Type<'db> {
        Type::new(self.db, TypeKind::Rune)
    }

    /// Create the Nil (unit) type.
    pub fn nil_type(&self) -> Type<'db> {
        Type::new(self.db, TypeKind::Nil)
    }

    /// Create an error type.
    pub fn error_type(&self) -> Type<'db> {
        Type::new(self.db, TypeKind::Error)
    }

    /// Create a tuple type.
    pub fn tuple_type(&self, elements: Vec<Type<'db>>) -> Type<'db> {
        Type::new(self.db, TypeKind::Tuple(elements))
    }

    /// Create a function type.
    pub fn func_type(
        &self,
        params: Vec<Type<'db>>,
        result: Type<'db>,
        effect: EffectRow<'db>,
    ) -> Type<'db> {
        Type::new(
            self.db,
            TypeKind::Func {
                params,
                result,
                effect,
            },
        )
    }

    /// Create a named type.
    pub fn named_type(&self, name: Symbol, args: Vec<Type<'db>>) -> Type<'db> {
        Type::new(self.db, TypeKind::Named { name, args })
    }
}
