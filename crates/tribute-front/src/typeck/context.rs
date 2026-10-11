//! Type checking context.
//!
//! This module provides `ModuleTypeEnv`, which holds module-level type information
//! (function/constructor/type definitions). This is populated during collect_declarations
//! and is read-only afterward.
//!
//! For function-level type inference, see `FunctionInferenceContext` in `func_context.rs`.

use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;

use trunk_ir::Symbol;

use crate::ast::{
    AbilityId, AbilityOrigin, CtorId, EffectRow, FuncDefId, OpDeclKind, Type, TypeDefId, TypeKind,
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

/// A candidate method entry for UFCS resolution.
///
/// Stored in the method index, keyed by method name. Receiver type
/// disambiguation happens at lookup time via `receiver_type_matches`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub struct MethodEntry<'db> {
    pub func_id: FuncDefId<'db>,
    pub func_ty: Type<'db>,
}

impl<'db> MethodEntry<'db> {
    /// Extract the receiver type from the function type's first parameter.
    pub fn receiver_ty(&self, db: &'db dyn salsa::Database) -> Option<Type<'db>> {
        match self.func_ty.kind(db) {
            TypeKind::Func { params, .. } => params.first().copied(),
            _ => None,
        }
    }
}

/// Extract the type constructor name from a `Type` value.
///
/// Handles `Named`, `App` (recursing into the constructor), and primitive
/// type kinds. Used for receiver type matching in UFCS resolution.
pub fn extract_type_name_from_type<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
) -> Option<Symbol> {
    let kind = ty.kind(db);
    if let Some(name) = kind.primitive_name() {
        return Some(Symbol::new(name));
    }
    match kind {
        TypeKind::Named { name, .. } => Some(name.clone()),
        TypeKind::App { ctor, .. } => extract_type_name_from_type(db, *ctor),
        _ => None,
    }
}

/// Check if a method entry's receiver type matches an actual receiver type.
///
/// Compares nominal types by declaration identity, allowing their generic
/// arguments to differ (e.g., `Option(a)` matches `Option(Int)`). Primitive
/// receivers continue to match by their compiler-defined names.
pub fn receiver_type_matches<'db>(
    db: &'db dyn salsa::Database,
    entry: &MethodEntry<'db>,
    actual: Type<'db>,
) -> bool {
    let Some(declared) = entry.receiver_ty(db) else {
        return false;
    };
    fn same_constructor<'db>(
        db: &'db dyn salsa::Database,
        declared: Type<'db>,
        actual: Type<'db>,
    ) -> bool {
        match (declared.kind(db), actual.kind(db)) {
            (TypeKind::App { ctor: left, .. }, _) => same_constructor(db, *left, actual),
            (_, TypeKind::App { ctor: right, .. }) => same_constructor(db, declared, *right),
            (TypeKind::Named { id: left, .. }, TypeKind::Named { id: right, .. }) => left == right,
            (left, right) => {
                left.primitive_name().is_some() && left.primitive_name() == right.primitive_name()
            }
        }
    }

    same_constructor(db, declared, actual)
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
        (TypeKind::App { ctor, .. }, _) => parameter_type_matches(db, *ctor, actual),
        (_, TypeKind::App { ctor, .. }) => parameter_type_matches(db, declared, *ctor),
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
    type_defs: HashMap<Symbol, TypeScheme<'db>>,

    /// Struct field definitions keyed by nominal declaration identity.
    struct_fields: HashMap<TypeDefId<'db>, StructFieldInfo<'db>>,

    /// Enum variant information: enum_name → [variant_names]
    /// Used for exhaustiveness checking in case expressions.
    enum_variants: HashMap<Symbol, Vec<Symbol>>,

    /// Field names of constructors whose fields are all named, in
    /// declaration order: structs, named-field variants, and constructors
    /// without fields. Positional-only variants are absent.
    constructor_field_names: HashMap<CtorId<'db>, Vec<Symbol>>,

    /// Ability definitions: AbilityId → AbilityInfo
    /// Used for handler arm type checking.
    ability_defs: HashMap<AbilityId<'db>, AbilityInfo<'db>>,

    /// Method index for UFCS resolution: method_name → candidates.
    /// Populated from function declarations (first param = receiver)
    /// and struct field accessors.
    method_index: HashMap<Symbol, Vec<MethodEntry<'db>>>,

    well_known_types: super::WellKnownTypes<'db>,
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
        let mut type_defs = HashMap::default();
        type_defs.insert(list_name, list_scheme);
        type_defs.insert(Symbol::new("std::collections::List"), list_scheme);

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
            method_index: HashMap::default(),
            well_known_types: super::WellKnownTypes::empty(),
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
    }

    /// Register a method for UFCS resolution.
    pub fn register_method(&mut self, method_name: Symbol, entry: MethodEntry<'db>) {
        self.method_index
            .entry(method_name)
            .or_default()
            .push(entry);
    }

    /// Look up UFCS method candidates by method name and receiver type.
    ///
    /// Returns the matching `MethodEntry` if exactly one candidate matches,
    /// `None` if no candidates or ambiguous (multiple matches).
    pub fn lookup_method(
        &self,
        method_name: &Symbol,
        receiver_ty: Type<'db>,
    ) -> Option<&MethodEntry<'db>> {
        let candidates = self.method_index.get(method_name)?;
        let mut iter = candidates
            .iter()
            .filter(|entry| receiver_type_matches(self.db, entry, receiver_ty));
        let matched = iter.next()?;
        if iter.next().is_some() {
            return None; // ambiguous
        }
        Some(matched)
    }

    /// The functions a method call may name by `method_name`.
    pub fn methods_named(&self, method_name: &Symbol) -> &[MethodEntry<'db>] {
        self.method_index
            .get(method_name)
            .map_or(&[], Vec::as_slice)
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
        struct_id: TypeDefId<'db>,
        type_params: Vec<TypeParam>,
        fields: Vec<(Symbol, Type<'db>)>,
    ) {
        self.struct_fields.insert(struct_id, (type_params, fields));
    }

    /// Register enum variant information.
    pub fn register_enum_variants(&mut self, enum_name: Symbol, variants: Vec<Symbol>) {
        self.enum_variants.insert(enum_name, variants);
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
        self.function_types.get(&id).copied()
    }

    /// Look up a constructor's type scheme.
    pub fn lookup_constructor(&self, id: CtorId<'db>) -> Option<TypeScheme<'db>> {
        self.constructor_types.get(&id).copied()
    }

    /// Look up a type definition.
    pub fn lookup_type_def(&self, name: &Symbol) -> Option<TypeScheme<'db>> {
        self.type_defs.get(name).copied()
    }

    /// Look up a type definition from a lexical module scope.
    pub fn lookup_type_def_in_scope(&self, name: &Symbol, prefix: &str) -> Option<TypeScheme<'db>> {
        let spelling = name.to_string();
        if spelling.contains("::") {
            return self.lookup_type_def(name);
        }

        // Name resolution leaves a bare name only for a declaration of the
        // current module; the package root holds the rest.
        let scope = prefix.trim_end_matches("::");
        if !scope.is_empty() {
            let candidate = Symbol::new(&format!("{scope}::{spelling}"));
            if let Some(scheme) = self.lookup_type_def(&candidate) {
                return Some(scheme);
            }
        }

        self.lookup_type_def(name)
    }

    /// Look up fields in declaration order using the exact struct identity.
    pub(crate) fn lookup_struct_fields(
        &self,
        struct_id: TypeDefId<'db>,
    ) -> Option<&[(Symbol, Type<'db>)]> {
        self.struct_fields
            .get(&struct_id)
            .map(|(_, fields)| fields.as_slice())
    }

    /// Look up struct field type by declaration identity and field name.
    /// Returns (type_params, field_type) if found.
    pub fn lookup_struct_field(
        &self,
        struct_id: TypeDefId<'db>,
        field_name: &Symbol,
    ) -> Option<(&[TypeParam], Type<'db>)> {
        let (type_params, fields) = self.struct_fields.get(&struct_id)?;
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
        use tribute_ir::ModulePathExt as _;

        let qualified = id.qualified(self.db);
        let leaf = qualified.last_segment();
        let parent = qualified.parent_path()?;
        let (kind, owner_name, field) = [FieldFunctionKind::Set, FieldFunctionKind::Modify]
            .into_iter()
            .find(|kind| kind.name().is_some_and(|name| leaf == name))
            .and_then(|kind| Some((kind, parent.parent_path()?, parent.last_segment())))
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
        let pure = EffectRow::pure(self.db);
        let body = match kind {
            FieldFunctionKind::Get => self.func_type(vec![receiver], field_ty, pure),
            FieldFunctionKind::Set => self.func_type(vec![receiver, field_ty], receiver, pure),
            FieldFunctionKind::Modify => {
                // A row variable the field's own type does not use.
                let id = crate::ast::collect_effect_vars(self.db, field_ty)
                    .iter()
                    .map(|var| var.id + 1)
                    .max()
                    .unwrap_or(0);
                let row = EffectRow::open(self.db, crate::ast::EffectVar { id });
                let callback = self.func_type(vec![field_ty], field_ty, row);
                self.func_type(vec![receiver, callback], receiver, row)
            }
        };
        let scheme = TypeScheme::new(
            self.db,
            parameters,
            crate::ast::collect_effect_vars(self.db, body),
            body,
        );
        Some(FieldFunction {
            owner,
            field,
            kind,
            scheme,
        })
    }

    /// The named field `field` of the struct named `owner`: the struct's
    /// identity and type parameters, and the field's type.
    fn struct_field(
        &self,
        owner: &Symbol,
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
    pub fn lookup_enum_variants(&self, enum_name: &Symbol) -> Option<&[Symbol]> {
        self.enum_variants.get(enum_name).map(|v| v.as_slice())
    }

    /// Field names of a constructor in declaration order, or `None` if its
    /// fields are positional.
    pub fn lookup_constructor_field_names(&self, id: CtorId<'db>) -> Option<&[Symbol]> {
        self.constructor_field_names.get(&id).map(Vec::as_slice)
    }

    /// Look up an ability definition by ID.
    pub fn lookup_ability(&self, id: AbilityId<'db>) -> Option<&AbilityInfo<'db>> {
        self.ability_defs.get(&id)
    }

    /// Look up an ability operation by ability ID and operation name.
    pub fn lookup_ability_op(
        &self,
        ability: AbilityId<'db>,
        op: &Symbol,
    ) -> Option<&AbilityOpInfo<'db>> {
        self.ability_defs
            .get(&ability)
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
    // Prelude injection
    // =========================================================================

    /// Inject prelude's resolved type information into this environment.
    ///
    /// This is called before type checking user code to make prelude's
    /// types available. The injected types contain only BoundVars (no UniVars).
    pub fn inject_prelude(&mut self, exports: &super::PreludeExports<'db>) {
        self.well_known_types = *exports.well_known_types(self.db);
        for (id, scheme) in exports.function_types(self.db) {
            self.function_types.insert(*id, *scheme);
        }
        self.extern_functions
            .extend(exports.extern_functions(self.db).iter().copied());
        for (id, scheme) in exports.constructor_types(self.db) {
            self.constructor_types.insert(*id, *scheme);
        }
        for (name, scheme) in exports.type_defs(self.db) {
            self.type_defs.insert(name.clone(), *scheme);
        }
        for (id, info) in exports.struct_fields(self.db) {
            self.struct_fields.insert(*id, info.clone());
        }
        for (name, variants) in exports.enum_variants(self.db) {
            self.enum_variants.insert(name.clone(), variants.clone());
        }
        for (id, names) in exports.constructor_field_names(self.db) {
            self.constructor_field_names.insert(*id, names.clone());
        }
        for (name, entries) in exports.method_index(self.db) {
            self.method_index
                .entry(name.clone())
                .or_default()
                .extend(entries.iter().copied());
        }
        for (ability, type_params, operations) in exports.ability_definitions(self.db) {
            self.ability_defs.insert(
                *ability,
                AbilityInfo {
                    id: *ability,
                    type_params: type_params.clone(),
                    operations: operations
                        .iter()
                        .map(|op| (op.name.clone(), op.clone()))
                        .collect(),
                },
            );
        }
    }

    pub fn well_known_types(&self) -> super::WellKnownTypes<'db> {
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
    pub fn export_function_types(&self) -> Vec<(Symbol, TypeScheme<'db>)> {
        let mut result: Vec<_> = self
            .function_types
            .iter()
            .map(|(id, scheme)| (id.qualified(self.db).clone(), *scheme))
            .collect();
        result.sort_by(|(a, _), (b, _)| a.with_str(|a| b.with_str(|b| a.cmp(b))));
        result
    }

    /// Export function types with FuncDefId (for PreludeExports).
    ///
    /// Results are sorted by fully qualified function name for deterministic output.
    pub fn export_function_types_with_ids(&self) -> Vec<(FuncDefId<'db>, TypeScheme<'db>)> {
        let mut result: Vec<_> = self.function_types.iter().map(|(k, v)| (*k, *v)).collect();
        result.sort_by(|(a, _), (b, _)| {
            a.qualified(self.db)
                .with_str(|a| b.qualified(self.db).with_str(|b| a.cmp(b)))
        });
        result
    }

    /// Export constructor types for PreludeExports.
    ///
    /// Results are sorted by fully qualified constructor name for deterministic output.
    pub fn export_constructor_types(&self) -> Vec<(CtorId<'db>, TypeScheme<'db>)> {
        let mut result: Vec<_> = self
            .constructor_types
            .iter()
            .map(|(k, v)| (*k, *v))
            .collect();
        result.sort_by(|(a, _), (b, _)| {
            a.qualified(self.db)
                .with_str(|a| b.qualified(self.db).with_str(|b| a.cmp(b)))
        });
        result
    }

    /// Export type definitions for PreludeExports.
    ///
    /// Results are sorted alphabetically by name for deterministic output.
    pub fn export_type_defs(&self) -> Vec<(Symbol, TypeScheme<'db>)> {
        let mut result: Vec<_> = self
            .type_defs
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        result.sort_by(|(a, _), (b, _)| a.with_str(|a| b.with_str(|b| a.cmp(b))));
        result
    }

    /// Export struct field definitions for PreludeExports.
    ///
    /// Results are sorted alphabetically by struct name for deterministic output.
    pub fn export_struct_fields(&self) -> Vec<(TypeDefId<'db>, StructFieldInfo<'db>)> {
        let mut result: Vec<_> = self
            .struct_fields
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect();
        result.sort_by(|(a, _), (b, _)| {
            a.qualified(self.db)
                .with_str(|a| b.qualified(self.db).with_str(|b| a.cmp(b)))
        });
        result
    }

    /// Export enum variant information for PreludeExports.
    ///
    /// Results are sorted alphabetically by enum name for deterministic output.
    pub fn export_enum_variants(&self) -> Vec<(Symbol, Vec<Symbol>)> {
        let mut result: Vec<_> = self
            .enum_variants
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        result.sort_by(|(a, _), (b, _)| a.with_str(|a| b.with_str(|b| a.cmp(b))));
        result
    }

    /// Export constructor field names for PreludeExports.
    ///
    /// Results are sorted by constructor name for deterministic output.
    pub fn export_constructor_field_names(&self) -> Vec<(CtorId<'db>, Vec<Symbol>)> {
        let mut result: Vec<_> = self
            .constructor_field_names
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect();
        result.sort_by(|(a, _), (b, _)| {
            a.qualified(self.db)
                .with_str(|a| b.qualified(self.db).with_str(|b| a.cmp(b)))
        });
        result
    }

    /// Export method index for PreludeExports.
    ///
    /// Results are sorted alphabetically by method name for deterministic output.
    pub fn export_method_index(&self) -> Vec<(Symbol, Vec<MethodEntry<'db>>)> {
        let mut result: Vec<_> = self
            .method_index
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        result.sort_by(|(a, _), (b, _)| a.with_str(|a| b.with_str(|b| a.cmp(b))));
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
        result.sort_by(|(a, _), (b, _)| {
            a.qualified(self.db)
                .with_str(|a_str| b.qualified(self.db).with_str(|b_str| a_str.cmp(b_str)))
                .then_with(|| {
                    ability_origin_rank(a.origin(self.db))
                        .cmp(&ability_origin_rank(b.origin(self.db)))
                })
        });
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
        result.sort_by(|(a, ..), (b, ..)| {
            a.qualified(self.db)
                .with_str(|a| b.qualified(self.db).with_str(|b| a.cmp(b)))
                .then_with(|| {
                    ability_origin_rank(a.origin(self.db))
                        .cmp(&ability_origin_rank(b.origin(self.db)))
                })
        });
        result
    }

    /// Export the `extern` functions sorted by fully qualified name.
    pub fn export_extern_functions(&self) -> Vec<FuncDefId<'db>> {
        let mut result: Vec<_> = self.extern_functions.iter().copied().collect();
        result.sort_by(|a, b| {
            a.qualified(self.db)
                .with_str(|a| b.qualified(self.db).with_str(|b| a.cmp(b)))
        });
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
        self.well_known_types
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
        self.named_type_in_scope(name, args, "")
    }

    /// The named type a type annotation's path names. Name resolution spells
    /// the path from the package root, so it is not read through the
    /// enclosing modules.
    pub fn path_type(&self, path: &[Symbol]) -> Type<'db> {
        crate::qualified_path_symbol(path).map_or_else(
            || self.error_type(),
            |name| self.named_type_in_scope(name, vec![], ""),
        )
    }

    /// Create a named type using lexical module lookup.
    pub fn named_type_in_scope(
        &self,
        name: Symbol,
        args: Vec<Type<'db>>,
        prefix: &str,
    ) -> Type<'db> {
        let id = self
            .lookup_type_def_in_scope(&name, prefix)
            .and_then(|scheme| match scheme.body(self.db).kind(self.db) {
                TypeKind::Named { id, .. } => Some(*id),
                _ => None,
            })
            .unwrap_or_else(|| TypeDefId::synthetic(self.db, name.clone()));
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
        AbilityId, AbilityOrigin, BuiltinAbility, CtorId, FuncDefId, NodeId, Type, TypeDefId,
        TypeKind, TypeOrigin, TypeScheme,
    };

    // =========================================================================
    // Export ordering tests - verify deterministic output
    // =========================================================================

    #[salsa_test]
    fn string_type_fallback_ignores_user_string(db: &dyn salsa::Database) {
        let mut env = ModuleTypeEnv::new(db);
        let name = Symbol::new("String");
        let user_id = TypeDefId::source(db, name.clone(), NodeId::from_raw(1));
        let user_ty = env.named_type_with_id(user_id, name.clone(), vec![]);
        env.register_type_def(name, TypeScheme::mono(db, user_ty));

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
        let func_z = FuncDefId::new(db, Symbol::new("zebra"));
        let func_a = FuncDefId::new(db, Symbol::new("alpha"));
        let func_m = FuncDefId::new(db, Symbol::new("middle"));

        let int_ty = Type::new(db, TypeKind::Int);
        let scheme = TypeScheme::mono(db, int_ty);

        env.register_function(func_z, scheme);
        env.register_function(func_a, scheme);
        env.register_function(func_m, scheme);

        let exported = env.export_function_types();

        // Should be sorted alphabetically by name
        let names: Vec<_> = exported.iter().map(|(name, _)| name.clone()).collect();
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

        let func_z = FuncDefId::new(db, Symbol::new("zebra"));
        let func_a = FuncDefId::new(db, Symbol::new("alpha"));
        let func_m = FuncDefId::new(db, Symbol::new("middle"));

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

        let ctor_z = CtorId::new(db, Symbol::new("Zebra"));
        let ctor_a = CtorId::new(db, Symbol::new("Alpha"));
        let ctor_m = CtorId::new(db, Symbol::new("Middle"));

        let int_ty = Type::new(db, TypeKind::Int);
        let scheme = TypeScheme::mono(db, int_ty);

        env.register_constructor(ctor_z, scheme);
        env.register_constructor(ctor_a, scheme);
        env.register_constructor(ctor_m, scheme);

        let exported = env.export_constructor_types();

        // Should be sorted alphabetically by constructor name
        let names: Vec<_> = exported.iter().map(|(id, _)| id.name(db)).collect();
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
    fn id_exports_sort_same_short_names_by_qualified_identity(db: &dyn salsa::Database) {
        let mut env = ModuleTypeEnv::new(db);
        let scheme = TypeScheme::mono(db, Type::new(db, TypeKind::Int));

        let function_z = FuncDefId::new(db, Symbol::new("z::same"));
        let function_a = FuncDefId::new(db, Symbol::new("a::same"));
        env.register_function(function_z, scheme);
        env.register_function(function_a, scheme);
        let functions: Vec<_> = env
            .export_function_types_with_ids()
            .into_iter()
            .filter(|(id, _)| id.name(db) == "same")
            .map(|(id, _)| id.qualified(db))
            .collect();
        assert_eq!(
            functions,
            vec![Symbol::new("a::same"), Symbol::new("z::same")]
        );

        let constructor_z = CtorId::new(db, Symbol::new("z::same"));
        let constructor_a = CtorId::new(db, Symbol::new("a::same"));
        env.register_constructor(constructor_z, scheme);
        env.register_constructor(constructor_a, scheme);
        let constructors: Vec<_> = env
            .export_constructor_types()
            .into_iter()
            .filter(|(id, _)| id.name(db) == "same")
            .map(|(id, _)| id.qualified(db))
            .collect();
        assert_eq!(
            constructors,
            vec![Symbol::new("a::same"), Symbol::new("z::same")]
        );

        let ability_a = AbilityId::source(db, Symbol::new("a::Audit"));
        let ability_source = AbilityId::source(db, Symbol::new("z::Audit"));
        let ability_builtin = AbilityId::new(
            db,
            AbilityOrigin::Builtin(BuiltinAbility::Io),
            Symbol::new("z::Audit"),
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

        env.register_type_def(Symbol::new("Zebra"), scheme);
        env.register_type_def(Symbol::new("Alpha"), scheme);
        env.register_type_def(Symbol::new("Middle"), scheme);

        let exported = env.export_type_defs();

        // Should be sorted alphabetically by name
        let names: Vec<_> = exported.iter().map(|(name, _)| name.clone()).collect();
        assert_eq!(
            names,
            vec![
                Symbol::new("Alpha"),
                Symbol::new("List"),
                Symbol::new("Middle"),
                Symbol::new("Zebra"),
                Symbol::new("std::collections::List"),
            ]
        );
    }

    #[salsa_test]
    fn test_export_struct_fields_sorted(db: &dyn salsa::Database) {
        let mut env = ModuleTypeEnv::new(db);

        let int_ty = Type::new(db, TypeKind::Int);
        let fields = vec![(Symbol::new("x"), int_ty)];

        let zebra = TypeDefId::synthetic(db, Symbol::new("Zebra"));
        let alpha = TypeDefId::synthetic(db, Symbol::new("Alpha"));
        let middle = TypeDefId::synthetic(db, Symbol::new("Middle"));
        env.register_struct_fields(zebra, vec![], fields.clone());
        env.register_struct_fields(alpha, vec![], fields.clone());
        env.register_struct_fields(middle, vec![], fields);

        let exported = env.export_struct_fields();

        // Should be sorted alphabetically by struct name
        let names: Vec<_> = exported.iter().map(|(id, _)| id.qualified(db)).collect();
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
        let names: Vec<_> = exported.iter().map(|(name, _)| name.clone()).collect();
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
            let func_id = FuncDefId::new(db, Symbol::new(name));
            env.register_function(func_id, scheme);
        }

        // Export multiple times and verify same ordering
        let first = env.export_function_types();
        let second = env.export_function_types();
        let third = env.export_function_types();

        let first_names: Vec<_> = first.iter().map(|(n, _)| n.clone()).collect();
        let second_names: Vec<_> = second.iter().map(|(n, _)| n.clone()).collect();
        let third_names: Vec<_> = third.iter().map(|(n, _)| n.clone()).collect();

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

    #[salsa_test]
    fn test_export_method_index_sorted(db: &dyn salsa::Database) {
        use super::MethodEntry;
        use crate::ast::EffectRow;

        let mut env = ModuleTypeEnv::new(db);
        let int_ty = Type::new(db, TypeKind::Int);
        let effect = EffectRow::pure(db);

        // Insert in non-alphabetical order
        for name in ["zebra", "alpha", "middle"] {
            let receiver = Type::new(
                db,
                TypeKind::Named {
                    id: crate::ast::TypeDefId::synthetic(db, Symbol::new(name)),
                    name: Symbol::new(name),
                    args: vec![],
                },
            );
            let func_ty = Type::new(
                db,
                TypeKind::Func {
                    params: vec![receiver],
                    result: int_ty,
                    effect,
                },
            );
            env.register_method(
                Symbol::new(name),
                MethodEntry {
                    func_id: FuncDefId::new(db, Symbol::new(name)),
                    func_ty,
                },
            );
        }

        let exported = env.export_method_index();
        let names: Vec<_> = exported.iter().map(|(name, _)| name.clone()).collect();
        assert_eq!(
            names,
            vec![
                Symbol::new("alpha"),
                Symbol::new("middle"),
                Symbol::new("zebra"),
            ]
        );
        // Each entry should have exactly one MethodEntry
        for (_, entries) in &exported {
            assert_eq!(entries.len(), 1);
        }
    }

    // =========================================================================
    // UFCS utility function tests
    // =========================================================================

    use super::{MethodEntry, extract_type_name_from_type, receiver_type_matches};
    use crate::ast::EffectRow;

    /// Test helper: create a Named type from a string.
    fn named<'db>(db: &'db dyn salsa::Database, name: &str) -> Type<'db> {
        Type::new(
            db,
            TypeKind::Named {
                id: crate::ast::TypeDefId::synthetic(db, Symbol::new(name)),
                name: Symbol::new(name),
                args: vec![],
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
                effect: EffectRow::pure(db),
            },
        )
    }

    /// Test helper: create a MethodEntry.
    fn method_entry<'db>(
        db: &'db dyn salsa::Database,
        name: &str,
        func_ty: Type<'db>,
    ) -> MethodEntry<'db> {
        MethodEntry {
            func_id: FuncDefId::new(db, Symbol::new(name)),
            func_ty,
        }
    }

    #[test]
    fn test_extract_type_name_named() {
        let db = salsa::DatabaseImpl::new();
        assert_eq!(
            extract_type_name_from_type(&db, named(&db, "Foo")),
            Some(Symbol::new("Foo"))
        );
    }

    #[test]
    fn test_extract_type_name_app() {
        let db = salsa::DatabaseImpl::new();
        let list = named(&db, "List");
        let int = Type::new(&db, TypeKind::Int);
        let app = Type::new(
            &db,
            TypeKind::App {
                ctor: list,
                args: vec![int],
            },
        );
        assert_eq!(
            extract_type_name_from_type(&db, app),
            Some(Symbol::new("List"))
        );
    }

    #[test]
    fn test_extract_type_name_nested_app() {
        let db = salsa::DatabaseImpl::new();
        let map = named(&db, "Map");
        let int = Type::new(&db, TypeKind::Int);
        let inner = Type::new(
            &db,
            TypeKind::App {
                ctor: map,
                args: vec![int],
            },
        );
        let string = Type::new(&db, TypeKind::string(&db));
        let outer = Type::new(
            &db,
            TypeKind::App {
                ctor: inner,
                args: vec![string],
            },
        );
        assert_eq!(
            extract_type_name_from_type(&db, outer),
            Some(Symbol::new("Map"))
        );
    }

    #[test]
    fn test_extract_type_name_primitive() {
        let db = salsa::DatabaseImpl::new();
        let cases = [
            (TypeKind::Int, "Int"),
            (TypeKind::Nat, "Nat"),
            (TypeKind::Float, "Float"),
            (TypeKind::Bool, "Bool"),
            (TypeKind::string(&db), "String"),
            (TypeKind::Bytes, "Bytes"),
            (TypeKind::Rune, "Rune"),
            (TypeKind::Nil, "Nil"),
        ];
        for (kind, expected) in cases {
            assert_eq!(
                extract_type_name_from_type(&db, Type::new(&db, kind)),
                Some(Symbol::new(expected)),
                "type name of primitive {expected}",
            );
        }
    }

    #[test]
    fn test_extract_type_name_tuple_returns_none() {
        let db = salsa::DatabaseImpl::new();
        let tuple = Type::new(
            &db,
            TypeKind::Tuple(vec![
                Type::new(&db, TypeKind::Int),
                Type::new(&db, TypeKind::Bool),
            ]),
        );
        assert_eq!(extract_type_name_from_type(&db, tuple), None);
    }

    #[test]
    fn test_extract_type_name_univar_returns_none() {
        let db = salsa::DatabaseImpl::new();
        let id = crate::ast::UniVarId::new(&db, crate::ast::UniVarSource::Anonymous(0), 0);
        assert_eq!(
            extract_type_name_from_type(&db, Type::new(&db, TypeKind::UniVar { id })),
            None
        );
    }

    #[test]
    fn parameter_type_matches_by_declaration_and_shape() {
        let db = salsa::DatabaseImpl::new();
        let int = Type::new(&db, TypeKind::Int);
        let bool = Type::new(&db, TypeKind::Bool);
        let foo = named(&db, "Foo");
        let bar = named(&db, "Bar");
        let app = |ctor, arg| {
            Type::new(
                &db,
                TypeKind::App {
                    ctor,
                    args: vec![arg],
                },
            )
        };
        let id = crate::ast::UniVarId::new(&db, crate::ast::UniVarSource::Anonymous(0), 0);
        let unknown = Type::new(&db, TypeKind::UniVar { id });
        let variable = Type::new(&db, TypeKind::BoundVar { index: 0 });

        // A nominal type matches its declaration, whatever its arguments.
        assert!(super::parameter_type_matches(&db, foo, foo));
        assert!(!super::parameter_type_matches(&db, foo, bar));
        assert!(super::parameter_type_matches(
            &db,
            app(foo, int),
            app(foo, bool)
        ));
        assert!(super::parameter_type_matches(&db, app(foo, int), foo));
        assert!(super::parameter_type_matches(&db, foo, app(foo, int)));
        assert!(!super::parameter_type_matches(
            &db,
            app(foo, int),
            app(bar, int)
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

    #[test]
    fn test_receiver_type_matches_same_type() {
        let db = salsa::DatabaseImpl::new();
        let foo = named(&db, "Foo");
        let entry = method_entry(&db, "bar", func(&db, &[foo], Type::new(&db, TypeKind::Int)));
        assert!(receiver_type_matches(&db, &entry, foo));
    }

    #[test]
    fn test_receiver_type_matches_different_type() {
        let db = salsa::DatabaseImpl::new();
        let foo = named(&db, "Foo");
        let bar = named(&db, "Bar");
        let entry = method_entry(&db, "m", func(&db, &[foo], Type::new(&db, TypeKind::Int)));
        assert!(!receiver_type_matches(&db, &entry, bar));
    }

    #[test]
    fn test_receiver_type_rejects_same_spelling_from_different_declarations() {
        let db = salsa::DatabaseImpl::new();
        let name = Symbol::new("Thing");
        let first = Type::new(
            &db,
            TypeKind::Named {
                id: crate::ast::TypeDefId::source(
                    &db,
                    name.clone(),
                    crate::ast::NodeId::from_raw(1),
                ),
                name: name.clone(),
                args: vec![],
            },
        );
        let second = Type::new(
            &db,
            TypeKind::Named {
                id: crate::ast::TypeDefId::source(
                    &db,
                    name.clone(),
                    crate::ast::NodeId::from_raw(2),
                ),
                name,
                args: vec![],
            },
        );
        let entry = method_entry(
            &db,
            "method",
            func(&db, &[first], Type::new(&db, TypeKind::Int)),
        );
        assert!(!receiver_type_matches(&db, &entry, second));
    }

    #[salsa_test]
    fn test_lookup_method_single_match(db: &dyn salsa::Database) {
        let mut env = ModuleTypeEnv::new(db);
        let foo = named(db, "Foo");
        let ft = func(db, &[foo], Type::new(db, TypeKind::Int));
        env.register_method(Symbol::new("bar"), method_entry(db, "bar", ft));
        assert!(env.lookup_method(&Symbol::new("bar"), foo).is_some());
    }

    #[salsa_test]
    fn test_lookup_method_no_match(db: &dyn salsa::Database) {
        let env = ModuleTypeEnv::new(db);
        assert!(
            env.lookup_method(&Symbol::new("x"), named(db, "Foo"))
                .is_none()
        );
    }

    #[salsa_test]
    fn test_lookup_method_ambiguous(db: &dyn salsa::Database) {
        let mut env = ModuleTypeEnv::new(db);
        let foo = named(db, "Foo");
        let int = Type::new(db, TypeKind::Int);
        let float = Type::new(db, TypeKind::Float);
        env.register_method(
            Symbol::new("m"),
            method_entry(db, "m1", func(db, &[foo], int)),
        );
        env.register_method(
            Symbol::new("m"),
            method_entry(db, "m2", func(db, &[foo], float)),
        );
        // Two candidates with same receiver → ambiguous → None
        assert!(env.lookup_method(&Symbol::new("m"), foo).is_none());
    }

    // =========================================================================
    // UFCS integration tests — parse Tribute source and verify resolution
    // =========================================================================

    use crate::ast::ExprKind;

    fn parse_and_typecheck<'db>(
        db: &'db dyn salsa::Database,
        src: &str,
    ) -> &'db crate::ast::Module<crate::ast::TypedRef<'db>> {
        let source = crate::SourceCst::from_source_str(db, "test.trb", src);
        crate::query::type_check_output(db, source)
            .expect("should typecheck successfully")
            .module(db)
    }

    /// Check that no MethodCall nodes remain in the typed AST (all resolved to Call).
    fn has_method_call(expr: &crate::ast::Expr<crate::ast::TypedRef<'_>>) -> bool {
        match &*expr.kind {
            ExprKind::MethodCall { .. } => true,
            ExprKind::Call { callee, args } => {
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
        for decl in &module.decls {
            if let crate::ast::Decl::Function(func) = decl {
                assert!(
                    !has_method_call(&func.body),
                    "MethodCall should be resolved in function '{}'",
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
        for decl in &module.decls {
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
