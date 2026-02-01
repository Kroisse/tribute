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

/// Information about an ability operation.
#[derive(Clone, Debug)]
pub struct AbilityOpInfo<'db> {
    /// Operation name.
    pub name: Symbol,
    /// Parameter types.
    pub param_types: Vec<Type<'db>>,
    /// Return type.
    pub return_type: Type<'db>,
}

/// Information about an ability declaration.
#[derive(Clone, Debug)]
pub struct AbilityInfo<'db> {
    /// Ability name.
    pub name: Symbol,
    /// Type parameters for the ability.
    pub type_params: Vec<TypeParam>,
    /// Operations defined by this ability.
    pub operations: HashMap<Symbol, AbilityOpInfo<'db>>,
}

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

    /// Ability definitions: ability_name → AbilityInfo
    /// Used for handler arm type checking.
    ability_defs: HashMap<Symbol, AbilityInfo<'db>>,
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
            ability_defs: HashMap::new(),
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

    /// Register an ability definition.
    pub fn register_ability(&mut self, name: Symbol, info: AbilityInfo<'db>) {
        self.ability_defs.insert(name, info);
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

    /// Look up an ability definition by name.
    pub fn lookup_ability(&self, name: Symbol) -> Option<&AbilityInfo<'db>> {
        self.ability_defs.get(&name)
    }

    /// Look up an ability operation by ability name and operation name.
    pub fn lookup_ability_op(&self, ability: Symbol, op: Symbol) -> Option<&AbilityOpInfo<'db>> {
        self.ability_defs
            .get(&ability)
            .and_then(|info| info.operations.get(&op))
    }

    /// Debug: print all registered constructors.
    pub fn debug_print_constructors(&self, db: &'db dyn salsa::Database) {
        eprintln!(
            "DEBUG: Registered constructors ({}):",
            self.constructor_types.len()
        );
        for id in self.constructor_types.keys() {
            eprintln!("  - {:?} (ctor_name: {:?})", id, id.ctor_name(db));
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
    ///
    /// Results are sorted alphabetically by name for deterministic output.
    pub fn export_function_types(&self) -> Vec<(Symbol, TypeScheme<'db>)> {
        let mut result: Vec<_> = self
            .function_types
            .iter()
            .map(|(id, scheme)| (id.name(self.db), *scheme))
            .collect();
        result.sort_by(|(a, _), (b, _)| a.with_str(|a| b.with_str(|b| a.cmp(b))));
        result
    }

    /// Export function types with FuncDefId (for PreludeExports).
    ///
    /// Results are sorted alphabetically by function name for deterministic output.
    pub fn export_function_types_with_ids(&self) -> Vec<(FuncDefId<'db>, TypeScheme<'db>)> {
        let mut result: Vec<_> = self.function_types.iter().map(|(k, v)| (*k, *v)).collect();
        result.sort_by(|(a, _), (b, _)| {
            a.name(self.db)
                .with_str(|a| b.name(self.db).with_str(|b| a.cmp(b)))
        });
        result
    }

    /// Export constructor types for PreludeExports.
    ///
    /// Results are sorted alphabetically by constructor name for deterministic output.
    pub fn export_constructor_types(&self) -> Vec<(CtorId<'db>, TypeScheme<'db>)> {
        let mut result: Vec<_> = self
            .constructor_types
            .iter()
            .map(|(k, v)| (*k, *v))
            .collect();
        result.sort_by(|(a, _), (b, _)| {
            a.ctor_name(self.db)
                .with_str(|a| b.ctor_name(self.db).with_str(|b| a.cmp(b)))
        });
        result
    }

    /// Export type definitions for PreludeExports.
    ///
    /// Results are sorted alphabetically by name for deterministic output.
    pub fn export_type_defs(&self) -> Vec<(Symbol, TypeScheme<'db>)> {
        let mut result: Vec<_> = self.type_defs.iter().map(|(k, v)| (*k, *v)).collect();
        result.sort_by(|(a, _), (b, _)| a.with_str(|a| b.with_str(|b| a.cmp(b))));
        result
    }

    /// Export struct field definitions for PreludeExports.
    ///
    /// Results are sorted alphabetically by struct name for deterministic output.
    pub fn export_struct_fields(&self) -> Vec<(Symbol, StructFieldInfo<'db>)> {
        let mut result: Vec<_> = self
            .struct_fields
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect();
        result.sort_by(|(a, _), (b, _)| a.with_str(|a| b.with_str(|b| a.cmp(b))));
        result
    }

    /// Export enum variant information for PreludeExports.
    ///
    /// Results are sorted alphabetically by enum name for deterministic output.
    pub fn export_enum_variants(&self) -> Vec<(Symbol, Vec<Symbol>)> {
        let mut result: Vec<_> = self
            .enum_variants
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect();
        result.sort_by(|(a, _), (b, _)| a.with_str(|a| b.with_str(|b| a.cmp(b))));
        result
    }

    /// Export ability definitions.
    ///
    /// Results are sorted alphabetically by ability name for deterministic output.
    pub fn export_ability_defs(&self) -> Vec<(Symbol, AbilityInfo<'db>)> {
        let mut result: Vec<_> = self
            .ability_defs
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect();
        result.sort_by(|(a, _), (b, _)| a.with_str(|a| b.with_str(|b| a.cmp(b))));
        result
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

#[cfg(test)]
mod tests {
    use salsa_test_macros::salsa_test;
    use trunk_ir::{Symbol, SymbolVec};

    use crate::ast::{CtorId, FuncDefId, Type, TypeKind, TypeScheme};

    use super::ModuleTypeEnv;

    // =========================================================================
    // Export ordering tests - verify deterministic output
    // =========================================================================

    #[salsa_test]
    fn test_export_function_types_sorted(db: &dyn salsa::Database) {
        let mut env = ModuleTypeEnv::new(db);

        // Insert in non-alphabetical order
        let func_z = FuncDefId::new(db, SymbolVec::new(), Symbol::new("zebra"));
        let func_a = FuncDefId::new(db, SymbolVec::new(), Symbol::new("alpha"));
        let func_m = FuncDefId::new(db, SymbolVec::new(), Symbol::new("middle"));

        let int_ty = Type::new(db, TypeKind::Int);
        let scheme = TypeScheme::mono(db, int_ty);

        env.register_function(func_z, scheme);
        env.register_function(func_a, scheme);
        env.register_function(func_m, scheme);

        let exported = env.export_function_types();

        // Should be sorted alphabetically by name
        let names: Vec<_> = exported.iter().map(|(name, _)| *name).collect();
        assert_eq!(
            names,
            vec![
                Symbol::new("alpha"),
                Symbol::new("middle"),
                Symbol::new("zebra")
            ]
        );
    }

    #[salsa_test]
    fn test_export_function_types_with_ids_sorted(db: &dyn salsa::Database) {
        let mut env = ModuleTypeEnv::new(db);

        let func_z = FuncDefId::new(db, SymbolVec::new(), Symbol::new("zebra"));
        let func_a = FuncDefId::new(db, SymbolVec::new(), Symbol::new("alpha"));
        let func_m = FuncDefId::new(db, SymbolVec::new(), Symbol::new("middle"));

        let int_ty = Type::new(db, TypeKind::Int);
        let scheme = TypeScheme::mono(db, int_ty);

        env.register_function(func_z, scheme);
        env.register_function(func_a, scheme);
        env.register_function(func_m, scheme);

        let exported = env.export_function_types_with_ids();

        // Should be sorted alphabetically by function name
        let names: Vec<_> = exported.iter().map(|(id, _)| id.name(db)).collect();
        assert_eq!(
            names,
            vec![
                Symbol::new("alpha"),
                Symbol::new("middle"),
                Symbol::new("zebra")
            ]
        );
    }

    #[salsa_test]
    fn test_export_constructor_types_sorted(db: &dyn salsa::Database) {
        let mut env = ModuleTypeEnv::new(db);

        let ctor_z = CtorId::new(db, SymbolVec::new(), Symbol::new("Zebra"));
        let ctor_a = CtorId::new(db, SymbolVec::new(), Symbol::new("Alpha"));
        let ctor_m = CtorId::new(db, SymbolVec::new(), Symbol::new("Middle"));

        let int_ty = Type::new(db, TypeKind::Int);
        let scheme = TypeScheme::mono(db, int_ty);

        env.register_constructor(ctor_z, scheme);
        env.register_constructor(ctor_a, scheme);
        env.register_constructor(ctor_m, scheme);

        let exported = env.export_constructor_types();

        // Should be sorted alphabetically by constructor name
        let names: Vec<_> = exported.iter().map(|(id, _)| id.ctor_name(db)).collect();
        assert_eq!(
            names,
            vec![
                Symbol::new("Alpha"),
                Symbol::new("Middle"),
                Symbol::new("Zebra")
            ]
        );
    }

    #[salsa_test]
    fn test_export_type_defs_sorted(db: &dyn salsa::Database) {
        let mut env = ModuleTypeEnv::new(db);

        let int_ty = Type::new(db, TypeKind::Int);
        let scheme = TypeScheme::mono(db, int_ty);

        env.register_type_def(Symbol::new("Zebra"), scheme);
        env.register_type_def(Symbol::new("Alpha"), scheme);
        env.register_type_def(Symbol::new("Middle"), scheme);

        let exported = env.export_type_defs();

        // Should be sorted alphabetically by name
        let names: Vec<_> = exported.iter().map(|(name, _)| *name).collect();
        assert_eq!(
            names,
            vec![
                Symbol::new("Alpha"),
                Symbol::new("Middle"),
                Symbol::new("Zebra")
            ]
        );
    }

    #[salsa_test]
    fn test_export_struct_fields_sorted(db: &dyn salsa::Database) {
        let mut env = ModuleTypeEnv::new(db);

        let int_ty = Type::new(db, TypeKind::Int);
        let fields = vec![(Symbol::new("x"), int_ty)];

        env.register_struct_fields(Symbol::new("Zebra"), vec![], fields.clone());
        env.register_struct_fields(Symbol::new("Alpha"), vec![], fields.clone());
        env.register_struct_fields(Symbol::new("Middle"), vec![], fields);

        let exported = env.export_struct_fields();

        // Should be sorted alphabetically by struct name
        let names: Vec<_> = exported.iter().map(|(name, _)| *name).collect();
        assert_eq!(
            names,
            vec![
                Symbol::new("Alpha"),
                Symbol::new("Middle"),
                Symbol::new("Zebra")
            ]
        );
    }

    #[salsa_test]
    fn test_export_enum_variants_sorted(db: &dyn salsa::Database) {
        let mut env = ModuleTypeEnv::new(db);

        let variants = vec![Symbol::new("A"), Symbol::new("B")];

        env.register_enum_variants(Symbol::new("Zebra"), variants.clone());
        env.register_enum_variants(Symbol::new("Alpha"), variants.clone());
        env.register_enum_variants(Symbol::new("Middle"), variants);

        let exported = env.export_enum_variants();

        // Should be sorted alphabetically by enum name
        let names: Vec<_> = exported.iter().map(|(name, _)| *name).collect();
        assert_eq!(
            names,
            vec![
                Symbol::new("Alpha"),
                Symbol::new("Middle"),
                Symbol::new("Zebra")
            ]
        );
    }

    #[salsa_test]
    fn test_export_ordering_is_deterministic(db: &dyn salsa::Database) {
        // Run export multiple times and verify consistent ordering
        let mut env = ModuleTypeEnv::new(db);

        let int_ty = Type::new(db, TypeKind::Int);
        let scheme = TypeScheme::mono(db, int_ty);

        // Add items in random order
        for name in ["d", "b", "e", "a", "c"] {
            let func_id = FuncDefId::new(db, SymbolVec::new(), Symbol::new(name));
            env.register_function(func_id, scheme);
        }

        // Export multiple times and verify same ordering
        let first = env.export_function_types();
        let second = env.export_function_types();
        let third = env.export_function_types();

        let first_names: Vec<_> = first.iter().map(|(n, _)| *n).collect();
        let second_names: Vec<_> = second.iter().map(|(n, _)| *n).collect();
        let third_names: Vec<_> = third.iter().map(|(n, _)| *n).collect();

        assert_eq!(first_names, second_names);
        assert_eq!(second_names, third_names);
        assert_eq!(
            first_names,
            vec![
                Symbol::new("a"),
                Symbol::new("b"),
                Symbol::new("c"),
                Symbol::new("d"),
                Symbol::new("e")
            ]
        );
    }
}
