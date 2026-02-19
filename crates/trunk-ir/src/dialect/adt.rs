//! Algebraic Data Type dialect operations and types.
//!
//! Target-independent operations for structs, variants (enums), arrays, and references.
//!
//! ## Types
//!
//! This dialect provides self-descriptive composite types:
//!
//! - `adt.struct` - struct type with name and fields
//! - `adt.enum` - enum type with name and variants
//! - `adt.typeref` - type reference for circular/recursive types

use std::fmt::Write;

use crate::type_interface::Printable;
use crate::{Attribute, Attrs, IdVec, Symbol, Type, dialect};

// === Type name constants ===
crate::symbols! {
    STRUCT_TYPE => "struct",
    ENUM_TYPE => "enum",
    TYPEREF => "typeref",
    ATTR_NAME => "name",
    ATTR_FIELDS => "fields",
    ATTR_VARIANTS => "variants",
    // Variant instance type attributes (used by adt_to_wasm lowering)
    ATTR_IS_VARIANT => "is_variant",
    ATTR_BASE_ENUM => "base_enum",
    ATTR_VARIANT_TAG => "variant_tag",
    ATTR_VARIANT_FIELDS => "variant_fields",
}

dialect! {
    mod adt {
        // === Struct (Product Type) ===

        /// `adt.struct_new` operation: creates a struct instance.
        #[attr(r#type: Type)]
        fn struct_new(#[rest] fields) -> result;

        /// `adt.struct_get` operation: reads a field from a struct by index.
        /// Type attribute specifies the struct type (for WASM GC lowering).
        #[attr(r#type: Type, field: u64)]
        fn struct_get(r#ref) -> result;

        /// `adt.struct_set` operation: writes a field in a struct by index.
        /// Type attribute specifies the struct type (for layout computation).
        #[attr(r#type: Type, field: u64)]
        fn struct_set(r#ref, value);

        // === Variant (Sum Type) ===

        /// `adt.variant_new` operation: creates a variant instance.
        #[attr(r#type: Type, tag: Symbol)]
        fn variant_new(#[rest] fields) -> result;

        /// `adt.variant_is` operation: tests if a variant is of a specific tag.
        /// Returns true (i1) if the variant matches the given tag.
        #[attr(r#type: Type, tag: Symbol)]
        fn variant_is(r#ref) -> result;

        /// `adt.variant_cast` operation: casts a variant to a specific variant type.
        /// Used after pattern matching to access variant-specific fields.
        #[attr(r#type: Type, tag: Symbol)]
        fn variant_cast(r#ref) -> result;

        /// `adt.variant_get` operation: reads a field from a variant by index.
        /// Type and tag attributes identify the enum type and variant for layout computation.
        #[attr(r#type: Type, tag: Symbol, field: u64)]
        fn variant_get(r#ref) -> result;

        // === Array ===

        /// `adt.array_new` operation: creates an array.
        /// First operand is size, optional second is init value.
        #[attr(r#type: Type)]
        fn array_new(#[rest] operands) -> result;

        /// `adt.array_get` operation: reads an element from an array.
        fn array_get(r#ref, index) -> result;

        /// `adt.array_set` operation: writes an element to an array.
        fn array_set(r#ref, index, value);

        /// `adt.array_len` operation: returns the length of an array.
        fn array_len(r#ref) -> result;

        // === Reference ===

        /// `adt.ref_null` operation: creates a null reference.
        #[attr(r#type: Type)]
        fn ref_null() -> result;

        /// `adt.ref_is_null` operation: checks if a reference is null.
        fn ref_is_null(r#ref) -> result;

        /// `adt.ref_cast` operation: casts a reference type (runtime checked).
        #[attr(r#type: Type)]
        fn ref_cast(r#ref) -> result;

        // === Literals ===

        /// `adt.string_const` operation: string constant.
        #[attr(value: String)]
        fn string_const() -> result;

        /// `adt.bytes_const` operation: byte array constant.
        #[attr(value: any)]
        fn bytes_const() -> result;
    }
}

// === Pure operation registrations ===
// Struct operations: new and get are pure (set modifies)
// Variant operations: new, is, cast, and get are pure
// Array operations: new, get, and len are pure (set modifies)
// Reference operations: all are pure
// Literal operations: all are pure

crate::register_pure_op!(StructNew<'_>);
crate::register_pure_op!(StructGet<'_>);

crate::register_pure_op!(VariantNew<'_>);
crate::register_pure_op!(VariantIs<'_>);
crate::register_pure_op!(VariantCast<'_>);
crate::register_pure_op!(VariantGet<'_>);

crate::register_pure_op!(ArrayNew<'_>);
crate::register_pure_op!(ArrayGet<'_>);
crate::register_pure_op!(ArrayLen<'_>);

crate::register_pure_op!(RefNull<'_>);
crate::register_pure_op!(RefIsNull<'_>);
crate::register_pure_op!(RefCast<'_>);

crate::register_pure_op!(StringConst<'_>);
crate::register_pure_op!(BytesConst<'_>);

// =============================================================================
// ADT Type Constructors
// =============================================================================

/// Create an `adt.struct` type with name and fields.
///
/// Fields are stored as a list of [name, type] pairs in the `fields` attribute.
///
/// # Arguments
/// * `name` - Qualified name of the struct (e.g., "Point", "std::Vec")
/// * `fields` - Field definitions as (name, type) pairs
pub fn struct_type<'db>(
    db: &'db dyn salsa::Database,
    name: impl Into<Symbol>,
    fields: Vec<(Symbol, Type<'db>)>,
) -> Type<'db> {
    let fields_attr: Vec<Attribute<'db>> = fields
        .into_iter()
        .map(|(field_name, field_type)| {
            Attribute::List(vec![
                Attribute::Symbol(field_name),
                Attribute::Type(field_type),
            ])
        })
        .collect();

    let mut attrs = Attrs::new();
    attrs.insert(ATTR_NAME(), Attribute::Symbol(name.into()));
    attrs.insert(ATTR_FIELDS(), Attribute::List(fields_attr));

    Type::new(db, DIALECT_NAME(), STRUCT_TYPE(), IdVec::new(), attrs)
}

/// Create an `adt.enum` type with name and variants.
///
/// Variants are stored as a list of [name, [field_types...]] in the `variants` attribute.
///
/// # Arguments
/// * `name` - Qualified name of the enum (e.g., "Option", "Result")
/// * `variants` - Variant definitions as (name, field_types) pairs
pub fn enum_type<'db>(
    db: &'db dyn salsa::Database,
    name: impl Into<Symbol>,
    variants: Vec<(Symbol, Vec<Type<'db>>)>,
) -> Type<'db> {
    let variants_attr: Vec<Attribute<'db>> = variants
        .into_iter()
        .map(|(variant_name, field_types)| {
            let field_attrs: Vec<Attribute<'db>> =
                field_types.into_iter().map(Attribute::Type).collect();
            Attribute::List(vec![
                Attribute::Symbol(variant_name),
                Attribute::List(field_attrs),
            ])
        })
        .collect();

    let mut attrs = Attrs::new();
    attrs.insert(ATTR_NAME(), Attribute::Symbol(name.into()));
    attrs.insert(ATTR_VARIANTS(), Attribute::List(variants_attr));

    Type::new(db, DIALECT_NAME(), ENUM_TYPE(), IdVec::new(), attrs)
}

/// Create an `adt.typeref` type - a reference to a named type.
///
/// Used for recursive/circular type references to avoid infinite structures.
///
/// # Arguments
/// * `name` - Qualified name of the referenced type
pub fn typeref<'db>(db: &'db dyn salsa::Database, name: impl Into<Symbol>) -> Type<'db> {
    let mut attrs = Attrs::new();
    attrs.insert(ATTR_NAME(), Attribute::Symbol(name.into()));

    Type::new(db, DIALECT_NAME(), TYPEREF(), IdVec::new(), attrs)
}

// === Type inspection helpers ===

/// Check if a type is an `adt.struct` type.
pub fn is_struct_type(db: &dyn salsa::Database, ty: Type<'_>) -> bool {
    ty.is_dialect(db, DIALECT_NAME(), STRUCT_TYPE())
}

/// Check if a type is an `adt.enum` type.
pub fn is_enum_type(db: &dyn salsa::Database, ty: Type<'_>) -> bool {
    ty.is_dialect(db, DIALECT_NAME(), ENUM_TYPE())
}

/// Check if a type is an `adt.typeref` type.
pub fn is_typeref(db: &dyn salsa::Database, ty: Type<'_>) -> bool {
    ty.is_dialect(db, DIALECT_NAME(), TYPEREF())
}

/// Check if a type is a variant instance type (created by adt_to_wasm lowering).
///
/// Variant instance types are generated when lowering `adt.variant_new` operations
/// to WasmGC struct types. They carry `is_variant = true` attribute.
pub fn is_variant_instance_type(db: &dyn salsa::Database, ty: Type<'_>) -> bool {
    matches!(
        ty.get_attr(db, ATTR_IS_VARIANT()),
        Some(Attribute::Bool(true))
    )
}

/// Get the variant tag from a variant instance type.
pub fn get_variant_tag<'db>(db: &'db dyn salsa::Database, ty: Type<'db>) -> Option<Symbol> {
    match ty.get_attr(db, ATTR_VARIANT_TAG()) {
        Some(Attribute::Symbol(tag)) => Some(*tag),
        _ => None,
    }
}

/// Get the base enum type from a variant instance type.
pub fn get_base_enum<'db>(db: &'db dyn salsa::Database, ty: Type<'db>) -> Option<Type<'db>> {
    match ty.get_attr(db, ATTR_BASE_ENUM()) {
        Some(Attribute::Type(base_ty)) => Some(*base_ty),
        _ => None,
    }
}

/// Get the field types from a variant instance type.
pub fn get_variant_field_types<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
) -> Option<Vec<Type<'db>>> {
    let Attribute::List(fields) = ty.get_attr(db, ATTR_VARIANT_FIELDS())? else {
        return None;
    };
    Some(
        fields
            .iter()
            .filter_map(|a| match a {
                Attribute::Type(t) => Some(*t),
                _ => None,
            })
            .collect(),
    )
}

/// Get the name from an ADT type (struct, enum, or typeref).
pub fn get_type_name<'db>(db: &'db dyn salsa::Database, ty: Type<'db>) -> Option<Symbol> {
    match ty.get_attr(db, ATTR_NAME()) {
        Some(Attribute::Symbol(name)) => Some(*name),
        _ => None,
    }
}

/// Get the variants from an `adt.enum` type.
///
/// Returns a list of (variant_name, field_types) pairs.
pub fn get_enum_variants<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
) -> Option<Vec<(Symbol, Vec<Type<'db>>)>> {
    if !is_enum_type(db, ty) {
        return None;
    }

    let variants_attr = ty.get_attr(db, ATTR_VARIANTS())?;
    let Attribute::List(variants) = variants_attr else {
        return None;
    };

    let mut result = Vec::new();
    for variant in variants {
        let Attribute::List(pair) = variant else {
            continue;
        };
        if pair.len() < 2 {
            continue;
        }
        let Attribute::Symbol(name) = &pair[0] else {
            continue;
        };
        let Attribute::List(field_attrs) = &pair[1] else {
            continue;
        };

        let field_types: Vec<Type<'db>> = field_attrs
            .iter()
            .filter_map(|a| match a {
                Attribute::Type(t) => Some(*t),
                _ => None,
            })
            .collect();

        result.push((*name, field_types));
    }

    Some(result)
}

/// Get the fields from an `adt.struct` type.
///
/// Returns a list of (field_name, field_type) pairs.
pub fn get_struct_fields<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
) -> Option<Vec<(Symbol, Type<'db>)>> {
    if !is_struct_type(db, ty) {
        return None;
    }

    let fields_attr = ty.get_attr(db, ATTR_FIELDS())?;
    let Attribute::List(fields) = fields_attr else {
        return None;
    };

    let mut result = Vec::new();
    for field in fields {
        let Attribute::List(pair) = field else {
            continue;
        };
        if pair.len() < 2 {
            continue;
        }
        let Attribute::Symbol(name) = &pair[0] else {
            continue;
        };
        let Attribute::Type(ty) = &pair[1] else {
            continue;
        };

        result.push((*name, *ty));
    }

    Some(result)
}

// === Printable interface registrations ===

// adt.struct -> "struct Name" or just the name
inventory::submit! {
    Printable::implement("adt", "struct", |db, ty, f| {
        if let Some(Attribute::Symbol(name)) = ty.get_attr(db, ATTR_NAME()) {
            write!(f, "{}", name)
        } else {
            f.write_str("<struct>")
        }
    })
}

// adt.enum -> "enum Name" or just the name
inventory::submit! {
    Printable::implement("adt", "enum", |db, ty, f| {
        if let Some(Attribute::Symbol(name)) = ty.get_attr(db, ATTR_NAME()) {
            write!(f, "{}", name)
        } else {
            f.write_str("<enum>")
        }
    })
}

// adt.typeref -> "Name" (just the referenced type name)
inventory::submit! {
    Printable::implement("adt", "typeref", |db, ty, f| {
        if let Some(Attribute::Symbol(name)) = ty.get_attr(db, ATTR_NAME()) {
            write!(f, "{}", name)
        } else {
            f.write_str("<typeref>")
        }
    })
}

// adt.* (other types) -> "Name" or "Name(params...)" with capitalized first letter
// This is a catch-all for variant types like "adt.Expr$Num"
inventory::submit! {
    Printable::implement_prefix("adt", "", |db, ty, f| {
        let type_name = ty.name(db);

        // Check if this is one of our special types (handled above)
        if type_name == STRUCT_TYPE() || type_name == ENUM_TYPE() || type_name == TYPEREF() {
            // These are handled by specific implementations above
            // This shouldn't happen, but just in case
            return f.write_str("?");
        }

        let params = ty.params(db);

        // Capitalize first letter
        let name = type_name.to_string();
        let mut chars = name.chars();
        if let Some(c) = chars.next() {
            for ch in c.to_uppercase() {
                f.write_char(ch)?;
            }
            f.write_str(chars.as_str())?;
        }

        if !params.is_empty() {
            f.write_char('(')?;
            for (i, &p) in params.iter().enumerate() {
                if i > 0 {
                    f.write_str(", ")?;
                }
                Printable::print_type(db, p, f)?;
            }
            f.write_char(')')?;
        }

        Ok(())
    })
}
