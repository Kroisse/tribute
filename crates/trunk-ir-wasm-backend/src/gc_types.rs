//! GC Type Registry for WebAssembly GC types.
//!
//! This module provides a centralized registry for managing WasmGC type definitions.
//! It separates type collection from the emit phase, allowing lowering passes to
//! work with concrete type indices rather than high-level Type attributes.
//!
//! ## Type Index Layout
//!
//! ```text
//! Index 0: BoxedF64 - Float wrapper for polymorphic contexts
//! Index 1: BytesArray - array i8 backing storage for Bytes
//! Index 2: BytesStruct - struct { data: ref BytesArray, offset: i32, len: i32 }
//! Index 3: Step - struct { tag: i32, value: anyref, prompt: i32, op_idx: i32 } (trampoline)
//! Index 4: ClosureStruct - struct { i32, anyref } (table index + env)
//! Index 5+: User-defined types (structs, arrays, variants, closures, etc.)
//! ```
//!
//! ## Usage
//!
//! The registry is populated by a collection pass before lowering:
//!
//! ```text
//! let registry = collect_gc_types(db, module);
//! let lowered = adt_to_wasm::lower(db, module, &registry);
//! ```

use std::collections::HashMap;

use tracing::debug;
use trunk_ir::dialect::{adt, cont, core, wasm};
use trunk_ir::{Attribute, DialectType, Symbol, Type};
use wasm_encoder::{AbstractHeapType, FieldType, HeapType, RefType, StorageType, ValType};

/// Type index for BoxedF64 (Float wrapper for polymorphic contexts).
/// This is always index 0 in the GC type section.
pub const BOXED_F64_IDX: u32 = 0;

/// Type index for BytesArray (array i8) - backing storage for Bytes.
/// This is always index 1 in the GC type section.
pub const BYTES_ARRAY_IDX: u32 = 1;

/// Type index for BytesStruct (struct { data: ref BytesArray, offset: i32, len: i32 }).
/// This is always index 2 in the GC type section.
pub const BYTES_STRUCT_IDX: u32 = 2;

/// Type index for Step (struct { tag: i32, value: anyref, prompt: i32, op_idx: i32 }).
/// This is always index 3 in the GC type section.
/// Used for trampoline-based effect system in WasmGC backend (without stack switching).
pub const STEP_IDX: u32 = 3;

/// Type index for ClosureStruct (struct { i32, anyref }).
/// This is always index 4 in the GC type section.
/// All closures share this uniform representation: (table_idx: i32, env: anyref).
pub const CLOSURE_STRUCT_IDX: u32 = 4;

/// First type index available for user-defined types.
pub const FIRST_USER_TYPE_IDX: u32 = 5;

/// Check if a type is a closure struct (adt.struct with name "_closure").
/// Closure structs contain (funcref, anyref) and are used for call_indirect.
pub fn is_closure_struct_type<'db>(db: &'db dyn salsa::Database, ty: Type<'db>) -> bool {
    // Check if it's an adt.struct type
    if ty.dialect(db) != adt::DIALECT_NAME() {
        return false;
    }
    if ty.name(db) != Symbol::new("struct") {
        return false;
    }
    // Check if the struct name is "_closure"
    ty.attrs(db).get(&Symbol::new("name")).is_some_and(|attr| {
        if let Attribute::Symbol(name) = attr {
            name.with_str(|s| s == "_closure")
        } else {
            false
        }
    })
}

/// Definition of a GC type (struct or array).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GcTypeDef {
    /// Struct type with ordered field types.
    Struct(Vec<FieldType>),
    /// Array type with element type.
    Array(FieldType),
}

impl GcTypeDef {
    /// Create a struct type definition with the given field types.
    pub fn struct_type(fields: Vec<FieldType>) -> Self {
        GcTypeDef::Struct(fields)
    }

    /// Create an array type definition with the given element type.
    pub fn array_type(element: FieldType) -> Self {
        GcTypeDef::Array(element)
    }

    /// Returns the number of fields if this is a struct type.
    pub fn field_count(&self) -> Option<usize> {
        match self {
            GcTypeDef::Struct(fields) => Some(fields.len()),
            GcTypeDef::Array(_) => None,
        }
    }
}

/// Registry for GC types, mapping high-level types to WasmGC type indices.
///
/// This registry is populated during a dedicated type collection pass and used
/// by subsequent lowering passes to emit operations with concrete type indices.
#[derive(Debug, Clone)]
pub struct GcTypeRegistry<'db> {
    /// All GC type definitions in index order (starting from FIRST_USER_TYPE_IDX).
    types: Vec<GcTypeDef>,

    /// High-level Type → type_idx mapping for ADT types.
    type_indices: HashMap<Type<'db>, u32>,

    /// Placeholder type mapping: (placeholder_type, field_count) → type_idx.
    /// Used for structref placeholders where multiple struct types share the same
    /// placeholder type but differ in field count.
    placeholder_indices: HashMap<(Type<'db>, usize), u32>,

    /// Next available type index.
    next_type_idx: u32,
}

impl<'db> GcTypeRegistry<'db> {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            types: Vec::new(),
            type_indices: HashMap::new(),
            placeholder_indices: HashMap::new(),
            next_type_idx: FIRST_USER_TYPE_IDX,
        }
    }

    /// Returns the builtin type definitions (BoxedF64, BytesArray, BytesStruct, Step, ClosureStruct).
    ///
    /// These must be prepended to the user-defined types when emitting.
    pub fn builtin_types() -> Vec<GcTypeDef> {
        vec![
            // Index 0: BoxedF64 - struct { value: f64 }
            GcTypeDef::Struct(vec![FieldType {
                element_type: StorageType::Val(ValType::F64),
                mutable: false,
            }]),
            // Index 1: BytesArray - array i8
            GcTypeDef::Array(FieldType {
                element_type: StorageType::I8,
                mutable: true,
            }),
            // Index 2: BytesStruct - struct { data: ref BytesArray, offset: i32, len: i32 }
            GcTypeDef::Struct(vec![
                FieldType {
                    element_type: StorageType::Val(ValType::Ref(wasm_encoder::RefType {
                        nullable: false,
                        heap_type: wasm_encoder::HeapType::Concrete(BYTES_ARRAY_IDX),
                    })),
                    mutable: false,
                },
                FieldType {
                    element_type: StorageType::Val(ValType::I32),
                    mutable: false,
                },
                FieldType {
                    element_type: StorageType::Val(ValType::I32),
                    mutable: false,
                },
            ]),
            // Index 3: Step - struct { tag: i32, value: anyref, prompt: i32, op_idx: i32 }
            // Used for trampoline-based effect system in WasmGC backend.
            // Tag: 0 = Done (completed with result), 1 = Shift (suspended with continuation)
            // Value: result value (if Done) or continuation struct (if Shift)
            // Prompt: target handler's prompt tag (only used when Shift)
            // OpIdx: ability operation index (only used when Shift)
            GcTypeDef::Struct(vec![
                FieldType {
                    element_type: StorageType::Val(ValType::I32),
                    mutable: false,
                },
                FieldType {
                    element_type: StorageType::Val(ValType::Ref(RefType::ANYREF)),
                    mutable: false,
                },
                FieldType {
                    element_type: StorageType::Val(ValType::I32),
                    mutable: false,
                },
                FieldType {
                    element_type: StorageType::Val(ValType::I32),
                    mutable: false,
                },
            ]),
            // Index 4: ClosureStruct - struct { func_idx: i32, env: anyref }
            // Uniform representation for all closures: function table index + environment.
            // All closures share this type regardless of their specific function/env types.
            GcTypeDef::Struct(vec![
                FieldType {
                    element_type: StorageType::Val(ValType::I32),
                    mutable: false,
                },
                FieldType {
                    element_type: StorageType::Val(ValType::Ref(RefType::ANYREF)),
                    mutable: false,
                },
            ]),
        ]
    }

    /// Register a type and get its type index.
    ///
    /// If the type is already registered, returns the existing index.
    /// Otherwise, allocates a new index and stores the type definition.
    pub fn register_type(&mut self, ty: Type<'db>, def: GcTypeDef) -> u32 {
        if let Some(&idx) = self.type_indices.get(&ty) {
            return idx;
        }

        let idx = self.next_type_idx;
        self.next_type_idx += 1;
        self.types.push(def);
        self.type_indices.insert(ty, idx);
        idx
    }

    /// Register a placeholder type with field count.
    ///
    /// Placeholder types (like wasm.structref) can represent multiple struct types
    /// that differ only in field count. This method uses (type, field_count) as key.
    pub fn register_placeholder(
        &mut self,
        ty: Type<'db>,
        field_count: usize,
        def: GcTypeDef,
    ) -> u32 {
        let key = (ty, field_count);
        if let Some(&idx) = self.placeholder_indices.get(&key) {
            return idx;
        }

        let idx = self.next_type_idx;
        self.next_type_idx += 1;
        self.types.push(def);
        self.placeholder_indices.insert(key, idx);
        idx
    }

    /// Look up the type index for a high-level type.
    pub fn get_type_idx(&self, ty: Type<'db>) -> Option<u32> {
        self.type_indices.get(&ty).copied()
    }

    /// Look up the type index for a placeholder type with field count.
    pub fn get_placeholder_idx(&self, ty: Type<'db>, field_count: usize) -> Option<u32> {
        self.placeholder_indices.get(&(ty, field_count)).copied()
    }

    /// Returns all user-defined type definitions in index order.
    pub fn user_types(&self) -> &[GcTypeDef] {
        &self.types
    }

    /// Returns all type definitions including builtins.
    ///
    /// The returned vector has builtins at indices 0-4 and user types starting at index 5.
    pub fn all_types(&self) -> Vec<GcTypeDef> {
        let mut all = Self::builtin_types();
        all.extend(self.types.iter().cloned());
        all
    }

    /// Returns the type → index mapping for use in emit phase.
    pub fn type_idx_map(&self) -> &HashMap<Type<'db>, u32> {
        &self.type_indices
    }

    /// Returns the placeholder → index mapping for use in emit phase.
    pub fn placeholder_idx_map(&self) -> &HashMap<(Type<'db>, usize), u32> {
        &self.placeholder_indices
    }

    /// Returns the next available type index.
    pub fn next_idx(&self) -> u32 {
        self.next_type_idx
    }

    /// Returns the total number of types (builtin + user).
    pub fn total_count(&self) -> u32 {
        FIRST_USER_TYPE_IDX + self.types.len() as u32
    }

    /// Create a registry view from existing type index maps.
    ///
    /// This is useful for compatibility with code that maintains its own
    /// type index HashMaps. The returned registry can be used with
    /// `type_to_field_type` and other functions that take a `&GcTypeRegistry`.
    ///
    /// Note: The returned registry does not contain type definitions (types vec is empty).
    /// It is only suitable for type index lookups, not for building new types.
    pub fn from_type_maps(
        type_indices: HashMap<Type<'db>, u32>,
        placeholder_indices: HashMap<(Type<'db>, usize), u32>,
    ) -> Self {
        // Calculate next_type_idx from the maximum index in the maps
        let max_type_idx = type_indices
            .values()
            .copied()
            .max()
            .unwrap_or(FIRST_USER_TYPE_IDX);
        let max_placeholder_idx = placeholder_indices
            .values()
            .copied()
            .max()
            .unwrap_or(FIRST_USER_TYPE_IDX);
        let next_type_idx = max_type_idx.max(max_placeholder_idx).saturating_add(1);

        Self {
            types: Vec::new(), // No type definitions - this is a view-only registry
            type_indices,
            placeholder_indices,
            next_type_idx,
        }
    }
}

impl Default for GcTypeRegistry<'_> {
    fn default() -> Self {
        Self::new()
    }
}

trunk_ir::symbols! {
    /// Attribute name for type information on wasm operations.
    ATTR_TYPE => "type",
    /// Attribute name for explicit type index.
    ATTR_TYPE_IDX => "type_idx",
    /// Attribute name for field index on struct operations.
    ATTR_FIELD_IDX => "field_idx",
    /// Attribute name for field count (used with placeholder types).
    ATTR_FIELD_COUNT => "field_count",
}

// ============================================================================
// Type Conversion
// ============================================================================

/// Convert a Trunk IR Type to a WasmGC FieldType.
///
/// This function maps high-level Tribute types to their WasmGC representations:
/// - Primitives (i32, i64, f32, f64) map directly to ValType
/// - Int/Nat map to i64 (arbitrary precision TBD)
/// - Wasm types (funcref, anyref, structref) map to their RefType equivalents
/// - Variant instance types map to anyref for polymorphism
/// - Registered struct types map to concrete refs
/// - Unknown types default to anyref
pub fn type_to_field_type<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    registry: &GcTypeRegistry<'db>,
) -> FieldType {
    let element_type = type_to_storage_type(db, ty, registry);
    FieldType {
        element_type,
        mutable: false,
    }
}

/// Convert a Trunk IR Type to a WasmGC StorageType.
pub fn type_to_storage_type<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    registry: &GcTypeRegistry<'db>,
) -> StorageType {
    debug!("type_to_storage_type: {}.{}", ty.dialect(db), ty.name(db));

    // Core primitive types
    if core::I32::from_type(db, ty).is_some() {
        return StorageType::Val(ValType::I32);
    }
    if core::I64::from_type(db, ty).is_some() {
        // Int/Nat (arbitrary precision) is lowered to i64 for Phase 1
        return StorageType::Val(ValType::I64);
    }
    if core::F32::from_type(db, ty).is_some() {
        return StorageType::Val(ValType::F32);
    }
    if core::F64::from_type(db, ty).is_some() {
        return StorageType::Val(ValType::F64);
    }

    // Note: tribute_rt types (int, nat, bool, float, any) should be
    // converted to core/wasm types by normalize_primitive_types pass before emit.

    // Core function type (core.func) - stored as funcref
    if core::Func::from_type(db, ty).is_some() {
        debug!("type_to_storage_type: core.func -> FUNCREF");
        return StorageType::Val(ValType::Ref(RefType::FUNCREF));
    }

    // Wasm dialect types
    if ty.dialect(db) == wasm::DIALECT_NAME() {
        let name = ty.name(db);
        if name == Symbol::new("funcref") {
            return StorageType::Val(ValType::Ref(RefType::FUNCREF));
        }
        if name == Symbol::new("anyref") {
            return StorageType::Val(ValType::Ref(RefType::ANYREF));
        }
        if name == Symbol::new("structref") {
            return StorageType::Val(ValType::Ref(RefType {
                nullable: true,
                heap_type: HeapType::Abstract {
                    shared: false,
                    ty: AbstractHeapType::Struct,
                },
            }));
        }
        if name == Symbol::new("i31ref") {
            return StorageType::Val(ValType::Ref(RefType {
                nullable: true,
                heap_type: HeapType::Abstract {
                    shared: false,
                    ty: AbstractHeapType::I31,
                },
            }));
        }
        // Unknown wasm type -> anyref
        return StorageType::Val(ValType::Ref(RefType::ANYREF));
    }

    // Closure types (adt.struct with name="_closure") map to the builtin CLOSURE_STRUCT_IDX
    // Note: closure::Closure types are converted to adt.struct(name="_closure") by TypeConverter
    if is_closure_struct_type(db, ty) {
        debug!("type_to_storage_type: closure struct -> Concrete(CLOSURE_STRUCT_IDX)");
        return StorageType::Val(ValType::Ref(RefType {
            nullable: true,
            heap_type: HeapType::Concrete(CLOSURE_STRUCT_IDX),
        }));
    }

    // Continuation types are represented as GC structs at runtime
    if cont::Continuation::from_type(db, ty).is_some() {
        debug!("type_to_storage_type: cont.continuation -> STRUCTREF");
        return StorageType::Val(ValType::Ref(RefType {
            nullable: true,
            heap_type: HeapType::Abstract {
                shared: false,
                ty: AbstractHeapType::Struct,
            },
        }));
    }

    // Variant instance types use anyref for polymorphism
    if adt::is_variant_instance_type(db, ty) {
        debug!(
            "type_to_storage_type: variant type {}.{} -> ANYREF",
            ty.dialect(db),
            ty.name(db)
        );
        return StorageType::Val(ValType::Ref(RefType::ANYREF));
    }

    // Check if type is registered in the registry
    if let Some(type_idx) = registry.get_type_idx(ty) {
        debug!(
            "type_to_storage_type: {}.{} -> Concrete({})",
            ty.dialect(db),
            ty.name(db),
            type_idx
        );
        return StorageType::Val(ValType::Ref(RefType {
            nullable: true,
            heap_type: HeapType::Concrete(type_idx),
        }));
    }

    // Default to anyref for unknown types
    debug!(
        "type_to_storage_type: {}.{} -> ANYREF (unknown)",
        ty.dialect(db),
        ty.name(db)
    );
    StorageType::Val(ValType::Ref(RefType::ANYREF))
}

// ============================================================================
// ADT Type Collection
// ============================================================================

/// Collect ADT struct type and register it in the registry.
///
/// Returns the assigned type index for the struct.
pub fn collect_adt_struct_type<'db>(
    db: &'db dyn salsa::Database,
    registry: &mut GcTypeRegistry<'db>,
    ty: Type<'db>,
) -> Option<u32> {
    // Only process adt.struct types
    if !adt::is_struct_type(db, ty) {
        return None;
    }

    // Check if already registered
    if let Some(idx) = registry.get_type_idx(ty) {
        return Some(idx);
    }

    // Get struct fields from type
    let fields = adt::get_struct_fields(db, ty)?;
    let field_count = fields.len();

    debug!(
        "collect_adt_struct_type: {} with {} fields",
        ty.name(db),
        field_count
    );

    // Create field types - use anyref as placeholder initially
    // The actual field types will be determined during emit phase
    // when operand types are available
    let field_types: Vec<FieldType> = fields
        .iter()
        .map(|(_name, field_ty)| type_to_field_type(db, *field_ty, registry))
        .collect();

    let def = GcTypeDef::Struct(field_types);
    Some(registry.register_type(ty, def))
}

/// Collect ADT enum type and its variant types, registering them in the registry.
///
/// For each enum, this registers:
/// - The base enum type (optional, for type references)
/// - Each variant as a separate struct type
///
/// Returns the type indices for all registered variants.
#[allow(dead_code)] // Will be used in future phases
pub fn collect_adt_enum_type<'db>(
    db: &'db dyn salsa::Database,
    registry: &mut GcTypeRegistry<'db>,
    ty: Type<'db>,
) -> Option<Vec<(Symbol, u32)>> {
    // Only process adt.enum types
    if !adt::is_enum_type(db, ty) {
        return None;
    }

    let variants = adt::get_enum_variants(db, ty)?;
    let mut result = Vec::new();

    for (variant_name, field_types) in variants {
        // Create variant-specific type name: Enum$Variant
        let base_name = adt::get_type_name(db, ty).unwrap_or_else(|| ty.name(db));
        let variant_type_name = Symbol::from_dynamic(&format!("{}${}", base_name, variant_name));

        // Create variant type with attributes
        let mut attrs = trunk_ir::Attrs::new();
        attrs.insert(adt::ATTR_IS_VARIANT(), trunk_ir::Attribute::Bool(true));
        attrs.insert(adt::ATTR_BASE_ENUM(), trunk_ir::Attribute::Type(ty));
        attrs.insert(
            adt::ATTR_VARIANT_TAG(),
            trunk_ir::Attribute::Symbol(variant_name),
        );

        let params: trunk_ir::IdVec<Type<'db>> = ty.params(db).iter().copied().collect();
        let variant_ty = Type::new(db, ty.dialect(db), variant_type_name, params, attrs);

        // Check if already registered
        if let Some(idx) = registry.get_type_idx(variant_ty) {
            result.push((variant_name, idx));
            continue;
        }

        debug!(
            "collect_adt_enum_type: variant {} with {} fields",
            variant_type_name,
            field_types.len()
        );

        // Create struct type for variant (no tag field in WasmGC subtyping approach)
        let wasm_field_types: Vec<FieldType> = field_types
            .iter()
            .map(|field_ty| type_to_field_type(db, *field_ty, registry))
            .collect();

        let def = GcTypeDef::Struct(wasm_field_types);
        let idx = registry.register_type(variant_ty, def);
        result.push((variant_name, idx));
    }

    Some(result)
}

// ============================================================================
// Closure Type Collection
// ============================================================================

/// Closure struct field count.
/// Closure structs always have 2 fields: (table_idx: i32, env: anyref)
pub const CLOSURE_FIELD_COUNT: usize = 2;

/// Register the closure struct type in the registry.
///
/// Closures are represented as WasmGC structs with two fields:
/// - Field 0: function table index (i32) - index into function table
/// - Field 1: environment (anyref) - captured variables as struct
///
/// This function uses the structref placeholder with CLOSURE_FIELD_COUNT
/// to ensure all closure operations use the same type index.
pub fn register_closure_type<'db>(
    db: &'db dyn salsa::Database,
    registry: &mut GcTypeRegistry<'db>,
) -> u32 {
    let structref_ty = wasm::Structref::new(db).as_type();

    // Check if already registered
    if let Some(idx) = registry.get_placeholder_idx(structref_ty, CLOSURE_FIELD_COUNT) {
        return idx;
    }

    // Create closure struct type: (func_idx: i32, env: anyref)
    let def = GcTypeDef::Struct(vec![
        FieldType {
            element_type: StorageType::Val(ValType::I32),
            mutable: false,
        },
        FieldType {
            element_type: StorageType::Val(ValType::Ref(RefType::ANYREF)),
            mutable: false,
        },
    ]);

    registry.register_placeholder(structref_ty, CLOSURE_FIELD_COUNT, def)
}

/// Check if a type represents a closure based on its attributes.
#[allow(dead_code)] // Will be used in future phases
pub fn is_closure_type<'db>(db: &'db dyn salsa::Database, ty: Type<'db>) -> bool {
    // Check for closure dialect type
    if ty.dialect(db) == Symbol::new("closure") {
        return true;
    }

    // Check for structref with is_closure attribute
    if wasm::Structref::from_type(db, ty).is_some()
        && matches!(
            ty.get_attr(db, Symbol::new("is_closure")),
            Some(trunk_ir::Attribute::Bool(true))
        )
    {
        return true;
    }

    false
}

// ============================================================================
// Step Type Collection (Trampoline-based Effect System)
// ============================================================================

/// Step struct field count.
/// Step structs have 4 fields: (tag: i32, value: anyref, prompt: i32, op_idx: i32)
pub const STEP_FIELD_COUNT: usize = 4;

/// Tag value for Done (successful completion with result value).
pub const STEP_TAG_DONE: i32 = 0;

/// Tag value for Shift (suspended with continuation, needs handler dispatch).
pub const STEP_TAG_SHIFT: i32 = 1;

/// Create the Step marker type.
///
/// This creates a unique ADT struct type to use as a key in the registry.
/// Step is used for trampoline-based effect system.
///
/// Layout: (tag: i32, value: anyref, prompt: i32, op_idx: i32)
pub fn step_marker_type<'db>(db: &'db dyn salsa::Database) -> Type<'db> {
    // Note: adt is imported from trunk_ir::dialect at the module level

    let i32_ty = core::I32::new(db).as_type();
    let anyref_ty = wasm::Anyref::new(db).as_type();

    adt::struct_type(
        db,
        "_Step",
        vec![
            (Symbol::new("tag"), i32_ty),
            (Symbol::new("value"), anyref_ty),
            (Symbol::new("prompt"), i32_ty),
            (Symbol::new("op_idx"), i32_ty),
        ],
    )
}

/// Register the Step struct type in the registry.
///
/// Step is used for trampoline-based effect system in the WasmGC backend.
/// All effectful functions return this type to indicate whether they completed
/// normally or shifted (suspended) due to an ability operation.
///
/// Structure:
/// - Field 0: tag (i32) - 0 = Done, 1 = Shift
/// - Field 1: value (anyref) - result value (if Done) or continuation (if Shift)
/// - Field 2: prompt (i32) - target handler's prompt tag (only used when Shift)
/// - Field 3: op_idx (i32) - ability operation index (only used when Shift)
///
/// Unlike closure types, Step uses a dedicated marker type to ensure it
/// gets a unique type index (not shared with other struct types).
pub fn register_step_type<'db>(
    db: &'db dyn salsa::Database,
    registry: &mut GcTypeRegistry<'db>,
) -> u32 {
    let marker_ty = step_marker_type(db);

    // Step is a builtin type at STEP_IDX (defined in builtin_types()).
    // Just record the marker → builtin index mapping.
    registry.type_indices.insert(marker_ty, STEP_IDX);
    STEP_IDX
}

/// Get the Step type index.
///
/// Step is always a builtin type at STEP_IDX, so this always returns Some(STEP_IDX).
pub fn get_step_type_idx<'db>(
    _db: &'db dyn salsa::Database,
    _registry: &GcTypeRegistry<'db>,
) -> Option<u32> {
    Some(STEP_IDX)
}

// ============================================================================
// Continuation State Type Collection
// ============================================================================

/// Register a continuation state struct type for a specific field count.
///
/// State structs capture live locals at shift points. Each shift point may
/// have a different number of captured locals, resulting in different field counts.
///
/// The actual field types are determined from operand types at emit time.
/// This function registers a placeholder struct with the given field count,
/// using anyref for all fields as a safe default.
pub fn register_cont_state_type<'db>(
    db: &'db dyn salsa::Database,
    registry: &mut GcTypeRegistry<'db>,
    field_count: usize,
) -> u32 {
    let structref_ty = wasm::Structref::new(db).as_type();

    // Check if already registered
    if let Some(idx) = registry.get_placeholder_idx(structref_ty, field_count) {
        return idx;
    }

    // Create state struct with anyref fields as placeholder
    // The actual field types will be refined during emit when operand types are known
    let fields: Vec<FieldType> = (0..field_count)
        .map(|_| FieldType {
            element_type: StorageType::Val(ValType::Ref(RefType::ANYREF)),
            mutable: false,
        })
        .collect();

    let def = GcTypeDef::Struct(fields);
    registry.register_placeholder(structref_ty, field_count, def)
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::DialectType;
    use trunk_ir::dialect::core;

    #[test]
    fn test_builtin_types() {
        let builtins = GcTypeRegistry::builtin_types();
        assert_eq!(builtins.len(), 5);

        // BoxedF64
        assert!(matches!(&builtins[0], GcTypeDef::Struct(fields) if fields.len() == 1));
        // BytesArray
        assert!(matches!(&builtins[1], GcTypeDef::Array(_)));
        // BytesStruct
        assert!(matches!(&builtins[2], GcTypeDef::Struct(fields) if fields.len() == 3));
        // Step (4 fields: tag, value, prompt, op_idx)
        assert!(matches!(&builtins[3], GcTypeDef::Struct(fields) if fields.len() == 4));
        // ClosureStruct
        assert!(matches!(&builtins[4], GcTypeDef::Struct(fields) if fields.len() == 2));
    }

    #[test]
    fn test_register_type() {
        let db = salsa::DatabaseImpl::default();
        let mut registry = GcTypeRegistry::new();

        let i32_ty = core::I32::new(&db).as_type();
        let def = GcTypeDef::Struct(vec![FieldType {
            element_type: StorageType::Val(ValType::I32),
            mutable: false,
        }]);

        let idx1 = registry.register_type(i32_ty, def.clone());
        assert_eq!(idx1, FIRST_USER_TYPE_IDX);

        // Same type should return same index
        let idx2 = registry.register_type(i32_ty, def);
        assert_eq!(idx2, FIRST_USER_TYPE_IDX);

        assert_eq!(registry.user_types().len(), 1);
    }

    #[test]
    fn test_register_placeholder() {
        let db = salsa::DatabaseImpl::default();
        let mut registry = GcTypeRegistry::new();

        let i32_ty = core::I32::new(&db).as_type();

        // Register placeholder with 2 fields
        let def2 = GcTypeDef::Struct(vec![
            FieldType {
                element_type: StorageType::Val(ValType::I32),
                mutable: false,
            },
            FieldType {
                element_type: StorageType::Val(ValType::I32),
                mutable: false,
            },
        ]);
        let idx1 = registry.register_placeholder(i32_ty, 2, def2);
        assert_eq!(idx1, FIRST_USER_TYPE_IDX);

        // Register placeholder with 3 fields (different key)
        let def3 = GcTypeDef::Struct(vec![
            FieldType {
                element_type: StorageType::Val(ValType::I32),
                mutable: false,
            },
            FieldType {
                element_type: StorageType::Val(ValType::I32),
                mutable: false,
            },
            FieldType {
                element_type: StorageType::Val(ValType::I32),
                mutable: false,
            },
        ]);
        let idx2 = registry.register_placeholder(i32_ty, 3, def3);
        assert_eq!(idx2, FIRST_USER_TYPE_IDX + 1);

        // Same placeholder should return same index
        let idx3 = registry.register_placeholder(i32_ty, 2, GcTypeDef::Struct(vec![]));
        assert_eq!(idx3, FIRST_USER_TYPE_IDX);

        assert_eq!(registry.user_types().len(), 2);
    }

    #[test]
    fn test_all_types() {
        let db = salsa::DatabaseImpl::default();
        let mut registry = GcTypeRegistry::new();

        let i32_ty = core::I32::new(&db).as_type();
        let def = GcTypeDef::Struct(vec![]);
        registry.register_type(i32_ty, def);

        let all = registry.all_types();
        assert_eq!(all.len(), 6); // 5 builtins + 1 user type
    }

    #[test]
    fn test_type_to_storage_type_primitives() {
        let db = salsa::DatabaseImpl::default();
        let registry = GcTypeRegistry::new();

        // Test i32
        let i32_ty = core::I32::new(&db).as_type();
        assert!(matches!(
            type_to_storage_type(&db, i32_ty, &registry),
            StorageType::Val(ValType::I32)
        ));

        // Test i64
        let i64_ty = core::I64::new(&db).as_type();
        assert!(matches!(
            type_to_storage_type(&db, i64_ty, &registry),
            StorageType::Val(ValType::I64)
        ));

        // Test f32
        let f32_ty = core::F32::new(&db).as_type();
        assert!(matches!(
            type_to_storage_type(&db, f32_ty, &registry),
            StorageType::Val(ValType::F32)
        ));

        // Test f64
        let f64_ty = core::F64::new(&db).as_type();
        assert!(matches!(
            type_to_storage_type(&db, f64_ty, &registry),
            StorageType::Val(ValType::F64)
        ));
    }

    #[test]
    fn test_type_to_storage_type_wasm_refs() {
        let db = salsa::DatabaseImpl::default();
        let registry = GcTypeRegistry::new();

        // Test funcref
        let funcref_ty = wasm::Funcref::new(&db).as_type();
        assert!(matches!(
            type_to_storage_type(&db, funcref_ty, &registry),
            StorageType::Val(ValType::Ref(RefType::FUNCREF))
        ));

        // Test anyref
        let anyref_ty = wasm::Anyref::new(&db).as_type();
        assert!(matches!(
            type_to_storage_type(&db, anyref_ty, &registry),
            StorageType::Val(ValType::Ref(RefType::ANYREF))
        ));

        // Test structref
        let structref_ty = wasm::Structref::new(&db).as_type();
        let storage = type_to_storage_type(&db, structref_ty, &registry);
        assert!(matches!(
            storage,
            StorageType::Val(ValType::Ref(RefType {
                nullable: true,
                heap_type: HeapType::Abstract {
                    ty: AbstractHeapType::Struct,
                    ..
                }
            }))
        ));
    }

    #[test]
    fn test_register_closure_type() {
        let db = salsa::DatabaseImpl::default();
        let mut registry = GcTypeRegistry::new();

        let idx1 = register_closure_type(&db, &mut registry);
        assert_eq!(idx1, FIRST_USER_TYPE_IDX);

        // Calling again should return the same index
        let idx2 = register_closure_type(&db, &mut registry);
        assert_eq!(idx2, FIRST_USER_TYPE_IDX);

        // Verify closure type structure
        let types = registry.user_types();
        assert_eq!(types.len(), 1);
        match &types[0] {
            GcTypeDef::Struct(fields) => {
                assert_eq!(fields.len(), CLOSURE_FIELD_COUNT);
                // Field 0: i32 (function table index)
                assert!(matches!(
                    fields[0].element_type,
                    StorageType::Val(ValType::I32)
                ));
                // Field 1: anyref
                assert!(matches!(
                    fields[1].element_type,
                    StorageType::Val(ValType::Ref(RefType::ANYREF))
                ));
            }
            _ => panic!("Expected struct type for closure"),
        }
    }

    #[test]
    fn test_closure_and_other_placeholders() {
        let db = salsa::DatabaseImpl::default();
        let mut registry = GcTypeRegistry::new();

        // Register closure type (2 fields)
        let closure_idx = register_closure_type(&db, &mut registry);
        assert_eq!(closure_idx, FIRST_USER_TYPE_IDX);

        // Register another placeholder with 3 fields (different from closure)
        let structref_ty = wasm::Structref::new(&db).as_type();
        let other_idx = registry.register_placeholder(
            structref_ty,
            3,
            GcTypeDef::Struct(vec![
                FieldType {
                    element_type: StorageType::Val(ValType::I32),
                    mutable: false,
                },
                FieldType {
                    element_type: StorageType::Val(ValType::I32),
                    mutable: false,
                },
                FieldType {
                    element_type: StorageType::Val(ValType::I32),
                    mutable: false,
                },
            ]),
        );
        assert_eq!(other_idx, FIRST_USER_TYPE_IDX + 1);

        // Closure type should still be retrievable
        let closure_idx2 = register_closure_type(&db, &mut registry);
        assert_eq!(closure_idx2, closure_idx);

        assert_eq!(registry.user_types().len(), 2);
    }

    #[test]
    fn test_register_cont_state_type() {
        let db = salsa::DatabaseImpl::default();
        let mut registry = GcTypeRegistry::new();

        // Register state with 0 fields (empty state)
        let idx0 = register_cont_state_type(&db, &mut registry, 0);
        assert_eq!(idx0, FIRST_USER_TYPE_IDX);

        // Register state with 3 fields
        let idx3 = register_cont_state_type(&db, &mut registry, 3);
        assert_eq!(idx3, FIRST_USER_TYPE_IDX + 1);

        // Same field count should return same index
        let idx3_again = register_cont_state_type(&db, &mut registry, 3);
        assert_eq!(idx3_again, idx3);

        // Different field count gets different index
        let idx5 = register_cont_state_type(&db, &mut registry, 5);
        assert_eq!(idx5, FIRST_USER_TYPE_IDX + 2);

        assert_eq!(registry.user_types().len(), 3);
    }

    #[test]
    fn test_register_step_type() {
        let db = salsa::DatabaseImpl::default();
        let mut registry = GcTypeRegistry::new();

        // Step is a builtin type at STEP_IDX
        let idx1 = register_step_type(&db, &mut registry);
        assert_eq!(idx1, STEP_IDX);

        // Calling again should return the same index
        let idx2 = register_step_type(&db, &mut registry);
        assert_eq!(idx2, STEP_IDX);

        // Step is a builtin, so it shouldn't appear in user_types()
        assert_eq!(registry.user_types().len(), 0);

        // Verify the marker type is mapped to STEP_IDX
        let marker_ty = step_marker_type(&db);
        assert_eq!(registry.get_type_idx(marker_ty), Some(STEP_IDX));
    }

    #[test]
    fn test_step_and_closure_different_types() {
        let db = salsa::DatabaseImpl::default();
        let mut registry = GcTypeRegistry::new();

        // Step and closure have different structures
        // Step: (i32, anyref, i32, i32) - builtin at STEP_IDX
        // Closure: (i32, anyref) - user type at FIRST_USER_TYPE_IDX
        // They have different type indices.

        // Register Step first (builtin at STEP_IDX)
        let step_idx = register_step_type(&db, &mut registry);
        assert_eq!(step_idx, STEP_IDX);

        // Register closure - first user type
        let closure_idx = register_closure_type(&db, &mut registry);
        assert_eq!(closure_idx, FIRST_USER_TYPE_IDX);

        // Verify they are different
        assert_ne!(step_idx, closure_idx);

        // Only closure should be in user_types (Step is builtin)
        let types = registry.user_types();
        assert_eq!(types.len(), 1);

        // Closure: (i32, anyref)
        match &types[0] {
            GcTypeDef::Struct(fields) => {
                assert!(matches!(
                    fields[0].element_type,
                    StorageType::Val(ValType::I32)
                ));
            }
            _ => panic!("Expected struct type for closure"),
        }
    }

    #[test]
    fn test_get_step_type_idx() {
        let db = salsa::DatabaseImpl::default();
        let registry = GcTypeRegistry::new();

        // Step is a builtin, so get_step_type_idx always returns Some(STEP_IDX)
        assert_eq!(get_step_type_idx(&db, &registry), Some(STEP_IDX));
    }
}
