//! Type checking context.
//!
//! This module provides `ModuleTypeEnv`, which holds module-level type information
//! (function/constructor/type definitions). This is populated during collect_declarations
//! and is read-only afterward.
//!
//! For function-level type inference, see `FunctionInferenceContext` in `func_context.rs`.

use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;
use trunk_ir::SymbolPath;

use crate::resolve::PackageName;

use trunk_ir::Symbol;

use crate::ast::{
    AbilityId, AbilityOrigin, AbilityRow, CtorId, FuncDefId, OpDeclKind, Type, TypeDefId, TypeKind,
    TypeParam, TypeScheme,
};

// =========================================================================
// ModuleTypeEnv: Module-level type information (read-only after collection)
// =========================================================================

/// Struct field information: (type_params, fields).
/// - type_params: The struct's type parameters for field type generalization
/// - fields: Vec of (field_name, field_type) pairs
pub type StructFieldInfo<'db> = (Vec<TypeParam>, Vec<(Symbol, Type<'db>)>);

/// Information about an ability operation.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub struct AbilityOpInfo<'db> {
    /// Operation name.
    pub name: Symbol,
    /// Whether this is a `fn` (tail-resumptive) or `op` (general) operation.
    pub kind: OpDeclKind,
    /// Parameter types.
    pub param_types: Vec<Type<'db>>,
    /// Return type.
    pub return_type: Type<'db>,
}

/// Information about an ability declaration.
#[derive(Clone, Debug)]
pub struct AbilityInfo<'db> {
    /// Ability identifier (module path + name).
    pub id: AbilityId<'db>,
    /// Type parameters for the ability.
    pub type_params: Vec<TypeParam>,
    /// Operations defined by this ability.
    pub operations: HashMap<Symbol, AbilityOpInfo<'db>>,
}

/// A function a call may name, with its type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub struct MethodEntry<'db> {
    pub func_id: FuncDefId<'db>,
    pub func_ty: Type<'db>,
}

/// Whether a parameter declared as `declared` takes an argument of type
/// `actual` when a call selects among several functions.
///
/// Nominal types match by declaration and primitives by name, as for a
/// receiver; function types and tuples match by their number of items. A
/// type variable of the declaration takes any type, and a type not inferred
/// yet excludes no declaration.
pub fn parameter_type_matches<'db>(
    db: &'db dyn salsa::Database,
    declared: Type<'db>,
    actual: Type<'db>,
) -> bool {
    match (declared.kind(db), actual.kind(db)) {
        (TypeKind::BoundVar { .. }, _)
        | (_, TypeKind::UniVar { .. } | TypeKind::Never | TypeKind::Error) => true,
        (
            TypeKind::Func {
                params: declared, ..
            },
            TypeKind::Func { params: actual, .. },
        ) => declared.len() == actual.len(),
        (TypeKind::Tuple(declared), TypeKind::Tuple(actual)) => declared.len() == actual.len(),
        (TypeKind::Named { id: left, .. }, TypeKind::Named { id: right, .. }) => left == right,
        (left, right) => {
            left.primitive_name().is_some() && left.primitive_name() == right.primitive_name()
        }
    }
}

/// A function a struct has for one of its named fields.
#[derive(Clone, Debug)]
pub struct FieldFunction<'db> {
    pub owner: TypeDefId<'db>,
    pub field: Symbol,
    pub kind: super::FieldFunctionKind,
    pub scheme: TypeScheme<'db>,
}

/// Module-level type environment.
///
/// This struct holds type information that is shared across all functions in a module:
/// - Function type schemes (polymorphic signatures)
/// - Constructor type schemes (for enum variants)
/// - Type definitions (struct/enum type schemes)
/// - Struct field definitions for UFCS/accessor resolution
/// - Method index for UFCS method resolution
///
/// After `collect_declarations` populates this, it becomes read-only during
/// function body type checking.
pub struct ModuleTypeEnv<'db> {
    db: &'db dyn salsa::Database,

    /// Function signatures (polymorphic).
    function_types: HashMap<FuncDefId<'db>, TypeScheme<'db>>,

    /// Functions declared `extern`. They use a foreign calling convention, so
    /// `become` cannot transfer a frame to them.
    extern_functions: HashSet<FuncDefId<'db>>,

    /// Constructor types.
    constructor_types: HashMap<CtorId<'db>, TypeScheme<'db>>,

    /// Type definitions (struct/enum names to their types).
    type_defs: HashMap<SymbolPath, TypeScheme<'db>>,

    /// Struct field definitions keyed by nominal declaration identity.
    struct_fields: HashMap<TypeDefId<'db>, StructFieldInfo<'db>>,

    /// Enum variant information: enum_name → [variant_names]
    /// Used for exhaustiveness checking in case expressions.
    enum_variants: HashMap<SymbolPath, Vec<Symbol>>,

    /// Field names of constructors whose fields are all named, in
    /// declaration order: structs, named-field variants, and constructors
    /// without fields. Positional-only variants are absent.
    constructor_field_names: HashMap<CtorId<'db>, Vec<Symbol>>,

    /// Ability definitions: AbilityId → AbilityInfo
    /// Used for handler arm type checking.
    ability_defs: HashMap<AbilityId<'db>, AbilityInfo<'db>>,

    /// Well-known identities this package declares; only package `std` has
    /// them.
    well_known_types: super::WellKnownTypes<'db>,

    /// What the package `std` this module depends on declares. Lookups read
    /// it after this module's own declarations, which never copy it.
    dependency: Option<Box<DependencyTypes<'db>>>,
}

/// The declarations a dependency package exports, indexed for lookup.
struct DependencyTypes<'db> {
    function_types: HashMap<FuncDefId<'db>, TypeScheme<'db>>,
    extern_functions: HashSet<FuncDefId<'db>>,
    constructor_types: HashMap<CtorId<'db>, TypeScheme<'db>>,
    type_defs: HashMap<SymbolPath, TypeScheme<'db>>,
    struct_fields: HashMap<TypeDefId<'db>, StructFieldInfo<'db>>,
    enum_variants: HashMap<SymbolPath, Vec<Symbol>>,
    constructor_field_names: HashMap<CtorId<'db>, Vec<Symbol>>,
    ability_defs: HashMap<AbilityId<'db>, AbilityInfo<'db>>,
    well_known_types: super::WellKnownTypes<'db>,
}

impl<'db> DependencyTypes<'db> {
    fn new() -> Self {
        Self {
            function_types: HashMap::default(),
            extern_functions: HashSet::default(),
            constructor_types: HashMap::default(),
            type_defs: HashMap::default(),
            struct_fields: HashMap::default(),
            enum_variants: HashMap::default(),
            constructor_field_names: HashMap::default(),
            ability_defs: HashMap::default(),
            well_known_types: super::WellKnownTypes::default(),
        }
    }

    /// Add the declarations one more dependency package exports. Their
    /// identities are package-qualified, so the packages' entries never
    /// collide.
    fn add(&mut self, db: &'db dyn salsa::Database, exports: &super::PackageExports<'db>) {
        self.function_types
            .extend(exports.function_types(db).iter().copied());
        self.extern_functions
            .extend(exports.extern_functions(db).iter().copied());
        self.constructor_types
            .extend(exports.constructor_types(db).iter().copied());
        self.type_defs.extend(exports.type_defs(db).iter().cloned());
        self.struct_fields
            .extend(exports.struct_fields(db).iter().cloned());
        self.enum_variants
            .extend(exports.enum_variants(db).iter().cloned());
        self.constructor_field_names
            .extend(exports.constructor_field_names(db).iter().cloned());
        self.ability_defs
            .extend(exports.ability_definitions(db).iter().map(
                |(ability, type_params, operations)| {
                    let info = AbilityInfo {
                        id: *ability,
                        type_params: type_params.clone(),
                        operations: operations
                            .iter()
                            .map(|op| (op.name.clone(), op.clone()))
                            .collect(),
                    };
                    (*ability, info)
                },
            ));
        // Only package `std` publishes well-known identities.
        let known = exports.well_known_types(db);
        let own = &mut self.well_known_types;
        own.string = own.string.or(known.string);
        own.string_equality = own.string_equality.or(known.string_equality);
        own.bytes_equality = own.bytes_equality.or(known.bytes_equality);
    }
}

impl<'db> ModuleTypeEnv<'db> {
    /// Create a new empty module type environment.
    pub fn new(db: &'db dyn salsa::Database) -> Self {
        let io = AbilityId::builtin_io(db);
        let mut ability_defs = HashMap::default();
        ability_defs.insert(
            io,
            AbilityInfo {
                id: io,
                type_params: vec![],
                operations: HashMap::default(),
            },
        );

        let list_name = Symbol::new("List");
        let list_arg = Type::new(db, TypeKind::BoundVar { index: 0 });
        let list_ty = Type::new(
            db,
            TypeKind::Named {
                id: TypeDefId::builtin_list(db),
                name: list_name.clone(),
                args: vec![list_arg],
            },
        );
        let list_scheme = TypeScheme::new(
            db,
            vec![TypeParam::named(Symbol::new("a"))],
            Vec::new(),
            list_ty,
        );
        let slots_id = TypeDefId::builtin_slots(db);
        let slots_ty = Type::new(
            db,
            TypeKind::Named {
                id: slots_id,
                name: Symbol::new("Slots"),
                args: vec![list_arg],
            },
        );
        let slots_scheme = TypeScheme::new(
            db,
            vec![TypeParam::named(Symbol::new("a"))],
            Vec::new(),
            slots_ty,
        );
        let mut type_defs = HashMap::default();
        type_defs.insert(SymbolPath::new([list_name]), list_scheme);
        type_defs.insert(
            PackageName::std().identity_path_of(&["collections", "List"]),
            list_scheme,
        );
        type_defs.insert(slots_id.path(db).clone(), slots_scheme);
        let dispatch_id = TypeDefId::builtin_dispatch(db);
        let dispatch_ty = Type::new(
            db,
            TypeKind::Named {
                id: dispatch_id,
                name: Symbol::new("Dispatch"),
                args: Vec::new(),
            },
        );
        type_defs.insert(
            dispatch_id.path(db).clone(),
            TypeScheme::mono(db, dispatch_ty),
        );

        Self {
            db,
            function_types: HashMap::default(),
            extern_functions: HashSet::default(),
            constructor_types: HashMap::default(),
            type_defs,
            struct_fields: HashMap::default(),
            enum_variants: HashMap::default(),
            constructor_field_names: HashMap::default(),
            ability_defs,
            well_known_types: super::WellKnownTypes::empty(),
            dependency: None,
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

    /// Mark a registered function as declared `extern`.
    pub fn register_extern_function(&mut self, id: FuncDefId<'db>) {
        self.extern_functions.insert(id);
    }

    /// Whether a function was declared `extern`.
    pub fn is_extern_function(&self, id: FuncDefId<'db>) -> bool {
        self.extern_functions.contains(&id)
            || self
                .dependency
                .as_ref()
                .is_some_and(|dependency| dependency.extern_functions.contains(&id))
    }

    /// Register a constructor's type scheme.
    pub fn register_constructor(&mut self, id: CtorId<'db>, scheme: TypeScheme<'db>) {
        self.constructor_types.insert(id, scheme);
    }

    /// Register a type definition.
    pub fn register_type_def(&mut self, path: SymbolPath, scheme: TypeScheme<'db>) {
        self.type_defs.insert(path, scheme);
    }

    /// Register struct field information.
    pub fn register_struct_fields(
        &mut self,
        struct_id: TypeDefId<'db>,
        type_params: Vec<TypeParam>,
        fields: Vec<(Symbol, Type<'db>)>,
    ) {
        self.struct_fields.insert(struct_id, (type_params, fields));
    }

    /// Register enum variant information.
    pub fn register_enum_variants(&mut self, enum_path: SymbolPath, variants: Vec<Symbol>) {
        self.enum_variants.insert(enum_path, variants);
    }

    /// Register the field names of a constructor whose fields are all named.
    pub fn register_constructor_field_names(&mut self, id: CtorId<'db>, names: Vec<Symbol>) {
        self.constructor_field_names.insert(id, names);
    }

    /// Register an ability definition.
    pub fn register_ability(&mut self, id: AbilityId<'db>, info: AbilityInfo<'db>) {
        self.ability_defs.insert(id, info);
    }

    // =========================================================================
    // Lookup (used during type checking)
    // =========================================================================

    /// Look up a function's type scheme.
    pub fn lookup_function(&self, id: FuncDefId<'db>) -> Option<TypeScheme<'db>> {
        self.function_types
            .get(&id)
            .or_else(|| self.dependency.as_ref()?.function_types.get(&id))
            .copied()
    }

    /// Look up a constructor's type scheme.
    pub fn lookup_constructor(&self, id: CtorId<'db>) -> Option<TypeScheme<'db>> {
        self.constructor_types
            .get(&id)
            .or_else(|| self.dependency.as_ref()?.constructor_types.get(&id))
            .copied()
    }

    /// Look up a type definition.
    pub fn lookup_type_def(&self, path: &[Symbol]) -> Option<TypeScheme<'db>> {
        self.type_defs
            .get(path)
            .or_else(|| self.dependency.as_ref()?.type_defs.get(path))
            .copied()
    }

    /// Look up a type definition from a lexical module scope.
    pub fn lookup_type_def_in_scope(
        &self,
        path: &[Symbol],
        prefix: &SymbolPath,
    ) -> Option<TypeScheme<'db>> {
        // Name resolution leaves a bare name only for a declaration of the
        // current module; the package root holds the rest.
        if let [name] = path
            && !prefix.is_root()
            && let Some(scheme) = self.lookup_type_def(&prefix.child(name.clone()))
        {
            return Some(scheme);
        }

        self.lookup_type_def(path)
    }

    /// Look up fields in declaration order using the exact struct identity.
    pub(crate) fn lookup_struct_fields(
        &self,
        struct_id: TypeDefId<'db>,
    ) -> Option<&[(Symbol, Type<'db>)]> {
        self.struct_field_info(struct_id)
            .map(|(_, fields)| fields.as_slice())
    }

    /// Look up struct field type by declaration identity and field name.
    /// Returns (type_params, field_type) if found.
    pub fn lookup_struct_field(
        &self,
        struct_id: TypeDefId<'db>,
        field_name: &Symbol,
    ) -> Option<(&[TypeParam], Type<'db>)> {
        let (type_params, fields) = self.struct_field_info(struct_id)?;
        for (name, ty) in fields {
            if *name == *field_name {
                return Some((type_params.as_slice(), *ty));
            }
        }
        None
    }

    /// The function `id` names if a struct has it for one of its named
    /// fields: the getter `T::f`, the setter `T::f::set`, or the modifier
    /// `T::f::modify`. Its scheme is derived from the field's type and takes
    /// the struct's own type parameters.
    pub fn field_function(&self, id: FuncDefId<'db>) -> Option<FieldFunction<'db>> {
        use super::FieldFunctionKind;
        let path = id.path(self.db);
        let leaf = path.leaf()?.clone();
        let parent = path.parent().filter(|parent| !parent.is_root())?;
        let (kind, owner_name, field) = [FieldFunctionKind::Set, FieldFunctionKind::Modify]
            .into_iter()
            .find(|kind| kind.name().is_some_and(|name| leaf == name))
            .and_then(|kind| {
                let owner = parent.parent().filter(|owner| !owner.is_root())?;
                Some((kind, owner, parent.leaf()?.clone()))
            })
            .filter(|(_, owner, field)| self.struct_field(owner, field).is_some())
            .unwrap_or((FieldFunctionKind::Get, parent, leaf));
        let (owner, parameters, field_ty) = self.struct_field(&owner_name, &field)?;

        let receiver = self.named_type_with_id(
            owner,
            owner.qualified(self.db).clone(),
            (0..parameters.len() as u32)
                .map(|index| Type::new(self.db, TypeKind::BoundVar { index }))
                .collect(),
        );
        let pure = AbilityRow::pure(self.db);
        let body = match kind {
            FieldFunctionKind::Get => self.func_type(vec![receiver], field_ty, pure),
            FieldFunctionKind::Set => self.func_type(vec![receiver, field_ty], receiver, pure),
            FieldFunctionKind::Modify => {
                // A row variable the field's own type does not use.
                let id = crate::ast::collect_row_vars(self.db, field_ty)
                    .iter()
                    .map(|var| var.id + 1)
                    .max()
                    .unwrap_or(0);
                let row = AbilityRow::open(self.db, crate::ast::RowVar { id });
                let callback = self.func_type(vec![field_ty], field_ty, row);
                self.func_type(vec![receiver, callback], receiver, row)
            }
        };
        let scheme = TypeScheme::new(
            self.db,
            parameters,
            crate::ast::collect_row_vars(self.db, body),
            body,
        );
        Some(FieldFunction {
            owner,
            field,
            kind,
            scheme,
        })
    }

    /// The named field `field` of the struct at `owner`: the struct's
    /// identity and type parameters, and the field's type.
    fn struct_field(
        &self,
        owner: &[Symbol],
        field: &Symbol,
    ) -> Option<(TypeDefId<'db>, Vec<TypeParam>, Type<'db>)> {
        let TypeKind::Named { id, .. } = self.lookup_type_def(owner)?.body(self.db).kind(self.db)
        else {
            return None;
        };
        let (parameters, ty) = self.lookup_struct_field(*id, field)?;
        Some((*id, parameters.to_vec(), ty))
    }

    /// The scheme of the function `id` and where it comes from: a
    /// declaration, or a struct's named field.
    pub fn function_scheme(
        &self,
        id: FuncDefId<'db>,
    ) -> Option<(TypeScheme<'db>, super::FunctionInstanceOrigin<'db>)> {
        if let Some(scheme) = self.lookup_function(id) {
            return Some((scheme, super::FunctionInstanceOrigin::Declaration));
        }
        let function = self.field_function(id)?;
        Some((
            function.scheme,
            super::FunctionInstanceOrigin::FieldAccessor {
                owner: function.owner,
                field: function.field,
                kind: function.kind,
            },
        ))
    }

    /// Return the number of registered constructors.
    pub fn constructor_count(&self) -> usize {
        self.constructor_types.len()
    }

    /// Look up enum variants by enum name.
    pub fn lookup_enum_variants(&self, enum_path: &[Symbol]) -> Option<&[Symbol]> {
        self.enum_variants
            .get(enum_path)
            .or_else(|| self.dependency.as_ref()?.enum_variants.get(enum_path))
            .map(|v| v.as_slice())
    }

    /// Field names of a constructor in declaration order, or `None` if its
    /// fields are positional.
    pub fn lookup_constructor_field_names(&self, id: CtorId<'db>) -> Option<&[Symbol]> {
        self.constructor_field_names
            .get(&id)
            .or_else(|| self.dependency.as_ref()?.constructor_field_names.get(&id))
            .map(Vec::as_slice)
    }

    /// Look up an ability definition by ID.
    pub fn lookup_ability(&self, id: AbilityId<'db>) -> Option<&AbilityInfo<'db>> {
        self.ability_defs
            .get(&id)
            .or_else(|| self.dependency.as_ref()?.ability_defs.get(&id))
    }

    /// Look up an ability operation by ability ID and operation name.
    pub fn lookup_ability_op(
        &self,
        ability: AbilityId<'db>,
        op: &Symbol,
    ) -> Option<&AbilityOpInfo<'db>> {
        self.lookup_ability(ability)
            .and_then(|info| info.operations.get(op))
    }

    /// Debug: print all registered constructors.
    pub fn debug_print_constructors(&self, db: &'db dyn salsa::Database) {
        eprintln!(
            "DEBUG: Registered constructors ({}):",
            self.constructor_types.len()
        );
        for id in self.constructor_types.keys() {
            eprintln!("  - {:?} (ctor_name: {:?})", id, id.name(db));
        }
    }

    // =========================================================================
    // Dependency
    // =========================================================================

    /// Read the declarations one dependency package exports after this
    /// module's own. Called once per package the module depends on,
    /// transitively. The module's output holds only its own declarations.
    pub fn depend_on(&mut self, exports: &super::PackageExports<'db>) {
        self.dependency
            .get_or_insert_with(|| Box::new(DependencyTypes::new()))
            .add(self.db, exports);
    }

    /// The struct field declarations of `struct_id`, from this module or its
    /// dependency.
    fn struct_field_info(&self, struct_id: TypeDefId<'db>) -> Option<&StructFieldInfo<'db>> {
        self.struct_fields
            .get(&struct_id)
            .or_else(|| self.dependency.as_ref()?.struct_fields.get(&struct_id))
    }

    /// The well-known identities in scope: this package's own, or those of
    /// the package `std` it depends on.
    pub fn well_known_types(&self) -> super::WellKnownTypes<'db> {
        match &self.dependency {
            Some(dependency) => dependency.well_known_types,
            None => self.well_known_types,
        }
    }

    /// The well-known identities this package declares.
    pub fn own_well_known_types(&self) -> super::WellKnownTypes<'db> {
        self.well_known_types
    }

    /// Record the semantic identities selected from the prelude.
    pub(crate) fn set_prelude_well_known_types(&mut self, types: super::WellKnownTypes<'db>) {
        self.well_known_types = types;
    }

    // =========================================================================
    // Export methods
    // =========================================================================

    /// Export function type schemes as a Vec keyed by Symbol (fully qualified function name).
    ///
    /// Results are sorted alphabetically by name for deterministic output.
    pub fn export_function_types(&self) -> Vec<(SymbolPath, TypeScheme<'db>)> {
        let mut result: Vec<_> = self
            .function_types
            .iter()
            .map(|(id, scheme)| (id.path(self.db).clone(), *scheme))
            .collect();
        result.sort_by(|(left, _), (right, _)| left.cmp(right));
        result
    }

    /// Export function types with FuncDefId (for PackageExports).
    ///
    /// Results are sorted by fully qualified function name for deterministic output.
    pub fn export_function_types_with_ids(&self) -> Vec<(FuncDefId<'db>, TypeScheme<'db>)> {
        let mut result: Vec<_> = self.function_types.iter().map(|(k, v)| (*k, *v)).collect();
        result.sort_by_key(|(id, ..)| id.path(self.db));
        result
    }

    /// Export constructor types for PackageExports.
    ///
    /// Results are sorted by fully qualified constructor name for deterministic output.
    pub fn export_constructor_types(&self) -> Vec<(CtorId<'db>, TypeScheme<'db>)> {
        let mut result: Vec<_> = self
            .constructor_types
            .iter()
            .map(|(k, v)| (*k, *v))
            .collect();
        result.sort_by_key(|(id, ..)| id.path(self.db));
        result
    }

    /// Export type definitions for PackageExports.
    ///
    /// Results are sorted alphabetically by name for deterministic output.
    pub fn export_type_defs(&self) -> Vec<(SymbolPath, TypeScheme<'db>)> {
        let mut result: Vec<_> = self
            .type_defs
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        result.sort_by(|(left, _), (right, _)| left.cmp(right));
        result
    }

    /// Export struct field definitions for PackageExports.
    ///
    /// Results are sorted alphabetically by struct name for deterministic output.
    pub fn export_struct_fields(&self) -> Vec<(TypeDefId<'db>, StructFieldInfo<'db>)> {
        let mut result: Vec<_> = self
            .struct_fields
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect();
        result.sort_by_key(|(id, ..)| id.path(self.db));
        result
    }

    /// Export enum variant information for PackageExports.
    ///
    /// Results are sorted alphabetically by enum name for deterministic output.
    pub fn export_enum_variants(&self) -> Vec<(SymbolPath, Vec<Symbol>)> {
        let mut result: Vec<_> = self
            .enum_variants
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        result.sort_by(|(left, _), (right, _)| left.cmp(right));
        result
    }

    /// Export constructor field names for PackageExports.
    ///
    /// Results are sorted by constructor name for deterministic output.
    pub fn export_constructor_field_names(&self) -> Vec<(CtorId<'db>, Vec<Symbol>)> {
        let mut result: Vec<_> = self
            .constructor_field_names
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect();
        result.sort_by_key(|(id, ..)| id.path(self.db));
        result
    }

    /// Export ability definitions.
    ///
    /// Results are sorted by fully qualified ability name, then origin, for
    /// deterministic output even when source and builtin identities share a
    /// display path.
    pub fn export_ability_defs(&self) -> Vec<(AbilityId<'db>, AbilityInfo<'db>)> {
        let mut result: Vec<_> = self
            .ability_defs
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect();
        result.sort_by_key(|(id, ..)| (id.path(self.db), ability_origin_rank(id.origin(self.db))));
        result
    }

    /// Export prelude ability schemas without a HashMap so Salsa can track them.
    pub fn export_ability_defs_for_prelude(
        &self,
    ) -> Vec<(AbilityId<'db>, Vec<TypeParam>, Vec<AbilityOpInfo<'db>>)> {
        let mut result: Vec<_> = self
            .ability_defs
            .iter()
            .map(|(id, info)| {
                let mut operations: Vec<_> = info.operations.values().cloned().collect();
                operations.sort_by(|a, b| a.name.with_str(|a| b.name.with_str(|b| a.cmp(b))));
                (*id, info.type_params.clone(), operations)
            })
            .collect();
        result.sort_by_key(|(id, ..)| (id.path(self.db), ability_origin_rank(id.origin(self.db))));
        result
    }

    /// Export the `extern` functions sorted by fully qualified name.
    pub fn export_extern_functions(&self) -> Vec<FuncDefId<'db>> {
        let mut result: Vec<_> = self.extern_functions.iter().copied().collect();
        result.sort_by_key(|id| id.path(self.db));
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

    /// Create the String type (prelude-defined enum).
    pub fn string_type(&self) -> Type<'db> {
        self.well_known_types()
            .string
            .map(|string| string.ty)
            .unwrap_or_else(|| Type::new(self.db, TypeKind::string(self.db)))
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

    /// Create the Never (bottom) type.
    pub fn never_type(&self) -> Type<'db> {
        Type::new(self.db, TypeKind::Never)
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
        effect: AbilityRow<'db>,
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
        self.named_type_in_scope(SymbolPath::new([name]), args, &SymbolPath::root())
    }

    /// The named type a type annotation's path names. Name resolution spells
    /// the path from the package root, so it is not read through the
    /// enclosing modules.
    pub fn path_type(&self, path: &[Symbol]) -> Type<'db> {
        if path.is_empty() {
            return self.error_type();
        }
        let path = SymbolPath::new(path.iter().cloned());
        self.named_type_in_scope(path, vec![], &SymbolPath::root())
    }

    /// Create a named type using lexical module lookup.
    pub fn named_type_in_scope(
        &self,
        path: SymbolPath,
        args: Vec<Type<'db>>,
        prefix: &SymbolPath,
    ) -> Type<'db> {
        let name = path.to_symbol();
        let id = self
            .lookup_type_def_in_scope(&path, prefix)
            .and_then(|scheme| match scheme.body(self.db).kind(self.db) {
                TypeKind::Named { id, .. } => Some(*id),
                _ => None,
            })
            .unwrap_or_else(|| TypeDefId::synthetic(self.db, path));
        self.named_type_with_id(id, name, args)
    }

    pub fn named_type_with_id(
        &self,
        id: TypeDefId<'db>,
        name: Symbol,
        args: Vec<Type<'db>>,
    ) -> Type<'db> {
        Type::new(self.db, TypeKind::Named { id, name, args })
    }

    pub fn canonical_list_type(&self, element: Type<'db>) -> Type<'db> {
        let name = Symbol::new("List");
        self.named_type_with_id(TypeDefId::builtin_list(self.db), name, vec![element])
    }
}

fn ability_origin_rank(origin: AbilityOrigin) -> u8 {
    match origin {
        AbilityOrigin::Source => 0,
        AbilityOrigin::Builtin(_) => 1,
    }
}

#[cfg(test)]
mod tests {
    use rustc_hash::FxHashMap as HashMap;

    use salsa_test_macros::salsa_test;
    use trunk_ir::Symbol;

    use super::{AbilityInfo, ModuleTypeEnv};
    use crate::ast::{
        AbilityId, AbilityOrigin, BuiltinAbility, CtorId, FuncDefId, Type, TypeDefId, TypeKind,
        TypeOrigin, TypeScheme,
    };

    // =========================================================================
    // Export ordering tests - verify deterministic output
    // =========================================================================

    #[salsa_test]
    fn string_type_fallback_ignores_user_string(db: &dyn salsa::Database) {
        let mut env = ModuleTypeEnv::new(db);
        let name = Symbol::new("String");
        let user_id = TypeDefId::source(db, trunk_ir::SymbolPath::new([name.clone()]));
        let user_ty = env.named_type_with_id(user_id, name.clone(), vec![]);
        env.register_type_def(
            trunk_ir::SymbolPath::new([name]),
            TypeScheme::mono(db, user_ty),
        );

        let string_ty = env.string_type();
        let TypeKind::Named { id, .. } = string_ty.kind(db) else {
            panic!("String fallback should be nominal");
        };
        assert_ne!(*id, user_id);
        assert_eq!(id.origin(db), TypeOrigin::Synthetic);
    }

    #[salsa_test]
    fn test_export_function_types_sorted(db: &dyn salsa::Database) {
        let mut env = ModuleTypeEnv::new(db);

        // Insert in non-alphabetical order
        let func_z = FuncDefId::new(db, trunk_ir::SymbolPath::new(["zebra"]));
        let func_a = FuncDefId::new(db, trunk_ir::SymbolPath::new(["alpha"]));
        let func_m = FuncDefId::new(db, trunk_ir::SymbolPath::new(["middle"]));

        let int_ty = Type::new(db, TypeKind::Int);
        let scheme = TypeScheme::mono(db, int_ty);

        env.register_function(func_z, scheme);
        env.register_function(func_a, scheme);
        env.register_function(func_m, scheme);

        let exported = env.export_function_types();

        // Should be sorted alphabetically by name
        itertools::assert_equal(
            exported.iter().map(|(path, _)| path),
            [["alpha"], ["middle"], ["zebra"]],
        );
    }

    #[salsa_test]
    fn test_export_function_types_with_ids_sorted(db: &dyn salsa::Database) {
        let mut env = ModuleTypeEnv::new(db);

        let func_z = FuncDefId::new(db, trunk_ir::SymbolPath::new(["zebra"]));
        let func_a = FuncDefId::new(db, trunk_ir::SymbolPath::new(["alpha"]));
        let func_m = FuncDefId::new(db, trunk_ir::SymbolPath::new(["middle"]));

        let int_ty = Type::new(db, TypeKind::Int);
        let scheme = TypeScheme::mono(db, int_ty);

        env.register_function(func_z, scheme);
        env.register_function(func_a, scheme);
        env.register_function(func_m, scheme);

        let exported = env.export_function_types_with_ids();

        // Should be sorted alphabetically by function name
        itertools::assert_equal(
            exported.iter().map(|(id, _)| id.name(db)),
            ["alpha", "middle", "zebra"],
        );
    }

    #[salsa_test]
    fn test_export_constructor_types_sorted(db: &dyn salsa::Database) {
        let mut env = ModuleTypeEnv::new(db);

        let ctor_z = CtorId::new(db, trunk_ir::SymbolPath::new(["Zebra"]));
        let ctor_a = CtorId::new(db, trunk_ir::SymbolPath::new(["Alpha"]));
        let ctor_m = CtorId::new(db, trunk_ir::SymbolPath::new(["Middle"]));

        let int_ty = Type::new(db, TypeKind::Int);
        let scheme = TypeScheme::mono(db, int_ty);

        env.register_constructor(ctor_z, scheme);
        env.register_constructor(ctor_a, scheme);
        env.register_constructor(ctor_m, scheme);

        let exported = env.export_constructor_types();

        // Should be sorted alphabetically by constructor name
        itertools::assert_equal(
            exported.iter().map(|(id, _)| id.name(db)),
            ["Alpha", "Middle", "Zebra"],
        );
    }

    #[salsa_test]
    fn id_exports_sort_same_short_names_by_qualified_identity(db: &dyn salsa::Database) {
        let mut env = ModuleTypeEnv::new(db);
        let scheme = TypeScheme::mono(db, Type::new(db, TypeKind::Int));

        let function_z = FuncDefId::new(db, trunk_ir::SymbolPath::new(["z", "same"]));
        let function_a = FuncDefId::new(db, trunk_ir::SymbolPath::new(["a", "same"]));
        env.register_function(function_z, scheme);
        env.register_function(function_a, scheme);
        itertools::assert_equal(
            env.export_function_types_with_ids()
                .into_iter()
                .map(|(id, _)| id.path(db))
                .filter(|path| path.leaf().is_some_and(|name| name == "same")),
            [["a", "same"], ["z", "same"]],
        );

        let constructor_z = CtorId::new(db, trunk_ir::SymbolPath::new(["z", "same"]));
        let constructor_a = CtorId::new(db, trunk_ir::SymbolPath::new(["a", "same"]));
        env.register_constructor(constructor_z, scheme);
        env.register_constructor(constructor_a, scheme);
        itertools::assert_equal(
            env.export_constructor_types()
                .into_iter()
                .map(|(id, _)| id.path(db))
                .filter(|path| path.leaf().is_some_and(|name| name == "same")),
            [["a", "same"], ["z", "same"]],
        );

        let ability_a = AbilityId::source(db, trunk_ir::SymbolPath::new(["a", "Audit"]));
        let ability_source = AbilityId::source(db, trunk_ir::SymbolPath::new(["z", "Audit"]));
        let ability_builtin = AbilityId::new(
            db,
            AbilityOrigin::Builtin(BuiltinAbility::Io),
            trunk_ir::SymbolPath::new(["z", "Audit"]),
        );
        for ability in [ability_source, ability_builtin, ability_a] {
            env.register_ability(
                ability,
                AbilityInfo {
                    id: ability,
                    type_params: vec![],
                    operations: HashMap::default(),
                },
            );
        }
        let expected = vec![
            (Symbol::new("a::Audit"), AbilityOrigin::Source),
            (Symbol::new("z::Audit"), AbilityOrigin::Source),
            (
                Symbol::new("z::Audit"),
                AbilityOrigin::Builtin(BuiltinAbility::Io),
            ),
        ];
        let ability_keys = |ids: Vec<AbilityId<'_>>| {
            ids.into_iter()
                .filter(|id| id.name(db) == "Audit")
                .map(|id| (id.qualified(db).clone(), id.origin(db)))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            ability_keys(
                env.export_ability_defs()
                    .into_iter()
                    .map(|(id, _)| id)
                    .collect(),
            ),
            expected
        );
        assert_eq!(
            ability_keys(
                env.export_ability_defs_for_prelude()
                    .into_iter()
                    .map(|(id, ..)| id)
                    .collect(),
            ),
            expected
        );
    }

    #[salsa_test]
    fn test_export_type_defs_sorted(db: &dyn salsa::Database) {
        let mut env = ModuleTypeEnv::new(db);

        let int_ty = Type::new(db, TypeKind::Int);
        let scheme = TypeScheme::mono(db, int_ty);

        env.register_type_def(trunk_ir::SymbolPath::new(["Zebra"]), scheme);
        env.register_type_def(trunk_ir::SymbolPath::new(["Alpha"]), scheme);
        env.register_type_def(trunk_ir::SymbolPath::new(["Middle"]), scheme);

        let exported = env.export_type_defs();

        // Should be sorted alphabetically by name
        let expected: [&[&str]; 7] = [
            &["Alpha"],
            &["List"],
            &["Middle"],
            &["Zebra"],
            &["std", "collections", "List"],
            &["std", "runtime", "Dispatch"],
            &["std", "runtime", "Slots"],
        ];
        itertools::assert_equal(exported.iter().map(|(path, _)| path), expected);
    }

    #[salsa_test]
    fn test_export_struct_fields_sorted(db: &dyn salsa::Database) {
        let mut env = ModuleTypeEnv::new(db);

        let int_ty = Type::new(db, TypeKind::Int);
        let fields = vec![(Symbol::new("x"), int_ty)];

        let zebra = TypeDefId::synthetic(db, trunk_ir::SymbolPath::new(["Zebra"]));
        let alpha = TypeDefId::synthetic(db, trunk_ir::SymbolPath::new(["Alpha"]));
        let middle = TypeDefId::synthetic(db, trunk_ir::SymbolPath::new(["Middle"]));
        env.register_struct_fields(zebra, vec![], fields.clone());
        env.register_struct_fields(alpha, vec![], fields.clone());
        env.register_struct_fields(middle, vec![], fields);

        let exported = env.export_struct_fields();

        // Should be sorted alphabetically by struct name
        itertools::assert_equal(
            exported.iter().map(|(id, _)| id.path(db)),
            [["Alpha"], ["Middle"], ["Zebra"]],
        );
    }

    #[salsa_test]
    fn test_export_enum_variants_sorted(db: &dyn salsa::Database) {
        let mut env = ModuleTypeEnv::new(db);

        let variants = vec![Symbol::new("A"), Symbol::new("B")];

        env.register_enum_variants(trunk_ir::SymbolPath::new(["Zebra"]), variants.clone());
        env.register_enum_variants(trunk_ir::SymbolPath::new(["Alpha"]), variants.clone());
        env.register_enum_variants(trunk_ir::SymbolPath::new(["Middle"]), variants);

        let exported = env.export_enum_variants();

        // Should be sorted alphabetically by enum name
        itertools::assert_equal(
            exported.iter().map(|(path, _)| path),
            [["Alpha"], ["Middle"], ["Zebra"]],
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
            let func_id = FuncDefId::new(db, trunk_ir::SymbolPath::new([name]));
            env.register_function(func_id, scheme);
        }

        // Export multiple times and verify same ordering
        let first = env.export_function_types();
        let second = env.export_function_types();
        let third = env.export_function_types();

        assert_eq!(first, second);
        assert_eq!(second, third);
        itertools::assert_equal(
            first.iter().map(|(path, _)| path),
            [["a"], ["b"], ["c"], ["d"], ["e"]],
        );
    }

    // =========================================================================
    // UFCS utility function tests
    // =========================================================================

    use crate::ast::AbilityRow;

    /// Test helper: create a Named type from a string.
    fn named<'db>(db: &'db dyn salsa::Database, name: &str) -> Type<'db> {
        applied(db, name, vec![])
    }

    /// Test helper: the synthetic nominal type `name` applied to `args`.
    fn applied<'db>(db: &'db dyn salsa::Database, name: &str, args: Vec<Type<'db>>) -> Type<'db> {
        Type::new(
            db,
            TypeKind::Named {
                id: crate::ast::TypeDefId::synthetic(db, trunk_ir::SymbolPath::new([name])),
                name: Symbol::new(name),
                args,
            },
        )
    }

    /// Test helper: create a pure function type `fn(params) -> result`.
    fn func<'db>(
        db: &'db dyn salsa::Database,
        params: &[Type<'db>],
        result: Type<'db>,
    ) -> Type<'db> {
        Type::new(
            db,
            TypeKind::Func {
                params: params.to_vec(),
                result,
                effect: AbilityRow::pure(db),
            },
        )
    }

    #[test]
    fn parameter_type_matches_by_declaration_and_shape() {
        let db = salsa::DatabaseImpl::new();
        let int = Type::new(&db, TypeKind::Int);
        let bool = Type::new(&db, TypeKind::Bool);
        let foo = named(&db, "Foo");
        let bar = named(&db, "Bar");
        let app = |name, arg| applied(&db, name, vec![arg]);
        let id = crate::ast::UniVarId::new(&db, crate::ast::UniVarSource::Anonymous(0), 0);
        let unknown = Type::new(&db, TypeKind::UniVar { id });
        let variable = Type::new(&db, TypeKind::BoundVar { index: 0 });

        // A nominal type matches its declaration, whatever its arguments.
        assert!(super::parameter_type_matches(&db, foo, foo));
        assert!(!super::parameter_type_matches(&db, foo, bar));
        assert!(super::parameter_type_matches(
            &db,
            app("Foo", int),
            app("Foo", bool)
        ));
        assert!(super::parameter_type_matches(&db, app("Foo", int), foo));
        assert!(super::parameter_type_matches(&db, foo, app("Foo", int)));
        assert!(!super::parameter_type_matches(
            &db,
            app("Foo", int),
            app("Bar", int)
        ));
        assert!(super::parameter_type_matches(&db, int, int));
        assert!(!super::parameter_type_matches(&db, int, bool));
        // Functions and tuples match by their number of items.
        let unary = func(&db, &[int], int);
        assert!(super::parameter_type_matches(
            &db,
            unary,
            func(&db, &[bool], bool)
        ));
        assert!(!super::parameter_type_matches(
            &db,
            unary,
            func(&db, &[int, int], int)
        ));
        assert!(!super::parameter_type_matches(&db, unary, int));
        assert!(super::parameter_type_matches(
            &db,
            Type::new(&db, TypeKind::Tuple(vec![int, int])),
            Type::new(&db, TypeKind::Tuple(vec![bool, foo]))
        ));
        assert!(!super::parameter_type_matches(
            &db,
            Type::new(&db, TypeKind::Tuple(vec![int, int])),
            Type::new(&db, TypeKind::Tuple(vec![int]))
        ));
        // A type variable takes any type, and an unknown type excludes none.
        assert!(super::parameter_type_matches(&db, variable, foo));
        assert!(super::parameter_type_matches(&db, foo, unknown));
        assert!(!super::parameter_type_matches(&db, foo, variable));
    }

    // =========================================================================
    // UFCS integration tests — parse Tribute source and verify resolution
    // =========================================================================

    use crate::ast::ExprKind;

    fn parse_and_typecheck<'db>(
        db: &'db dyn salsa::Database,
        src: &str,
    ) -> &'db crate::ast::DeclTable<crate::ast::TypedRef<'db>> {
        let source = crate::SourceCst::from_source_str(db, "test.trb", src);
        crate::query::type_check_output(db, source)
            .expect("should typecheck successfully")
            .module(db)
    }

    /// Check that no call is left without a selected function.
    fn has_method_call(expr: &crate::ast::Expr<crate::ast::TypedRef<'_>>) -> bool {
        match &*expr.kind {
            ExprKind::Functions { .. } => true,
            ExprKind::Call { callee, args, .. } => {
                has_method_call(callee) || args.iter().any(has_method_call)
            }
            ExprKind::Block { stmts, value } => {
                stmts.iter().any(|s| match s {
                    crate::ast::Stmt::Let { value, .. } => has_method_call(value),
                    crate::ast::Stmt::Expr { expr, .. } => has_method_call(expr),
                }) || has_method_call(value)
            }
            ExprKind::BinOp { lhs, rhs, .. } => has_method_call(lhs) || has_method_call(rhs),
            ExprKind::Case { scrutinee, arms } => {
                has_method_call(scrutinee) || arms.iter().any(|a| has_method_call(&a.body))
            }
            ExprKind::Lambda { body, .. } => has_method_call(body),
            _ => false,
        }
    }

    #[salsa_test]
    fn test_ufcs_single_method_resolves(db: &salsa::DatabaseImpl) {
        let module = parse_and_typecheck(
            db,
            r#"
            struct Foo { value: Nat }
            pub mod Foo {
                pub fn get_value(f: Foo) -> Nat { f.value }
            }
            fn main() -> Nil {
                let f = Foo { value: 42 }
                let _ = f.get_value()
            }
        "#,
        );
        for decl in module.decls() {
            if let crate::ast::Decl::Function(func) = decl {
                assert!(
                    !has_method_call(&func.body),
                    "a call should have its function selected in function '{}'",
                    func.name
                );
            }
        }
    }

    #[salsa_test]
    fn test_ufcs_disambiguation_by_receiver_type(db: &salsa::DatabaseImpl) {
        let module = parse_and_typecheck(
            db,
            r#"
            struct A { x: Nat }
            struct B { x: Nat }
            pub mod A {
                pub fn get(a: A) -> Nat { a.x }
            }
            pub mod B {
                pub fn get(b: B) -> Nat { b.x }
            }
            fn main() -> Nil {
                let a = A { x: 1 }
                let b = B { x: 2 }
                let _ = a.get()
                let _ = b.get()
            }
        "#,
        );
        for decl in module.decls() {
            if let crate::ast::Decl::Function(func) = decl
                && func.name == "main"
            {
                assert!(
                    !has_method_call(&func.body),
                    "Both A::get and B::get should resolve by receiver type"
                );
            }
        }
    }
}
