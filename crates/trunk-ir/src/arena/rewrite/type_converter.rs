//! Arena-based type converter.
//!
//! Provides type conversion infrastructure for arena IR dialect lowering.

use crate::arena::context::IrContext;
use crate::arena::refs::{OpRef, TypeRef, ValueRef};
use crate::arena::types::Location;

/// Result of materializing a type conversion.
pub struct MaterializeResult {
    /// The converted value.
    pub value: ValueRef,
    /// Operations created during materialization (to be inserted).
    pub ops: Vec<OpRef>,
}

/// Type conversion function signature.
type ConversionFn = dyn Fn(&IrContext, TypeRef) -> Option<TypeRef>;

/// Materialization function signature: creates cast ops when needed.
type MaterializerFn =
    dyn Fn(&mut IrContext, Location, ValueRef, TypeRef, TypeRef) -> Option<MaterializeResult>;

/// Arena type converter — maps types during dialect conversion.
///
/// Holds a collection of conversion functions and a materialization callback
/// for inserting cast operations.
pub struct ArenaTypeConverter {
    /// Type conversion functions, tried in order.
    conversions: Vec<Box<ConversionFn>>,
    /// Materialization function: creates cast ops when needed.
    materializer: Option<Box<MaterializerFn>>,
}

impl ArenaTypeConverter {
    /// Create a new empty type converter.
    pub fn new() -> Self {
        Self {
            conversions: Vec::new(),
            materializer: None,
        }
    }

    /// Add a type conversion function.
    pub fn add_conversion(&mut self, f: impl Fn(&IrContext, TypeRef) -> Option<TypeRef> + 'static) {
        self.conversions.push(Box::new(f));
    }

    /// Set the materialization function.
    pub fn set_materializer(
        &mut self,
        f: impl Fn(&mut IrContext, Location, ValueRef, TypeRef, TypeRef) -> Option<MaterializeResult>
        + 'static,
    ) {
        self.materializer = Some(Box::new(f));
    }

    /// Convert a type, trying each conversion function in order.
    ///
    /// Returns `None` if no conversion applies (type is already legal).
    pub fn convert_type(&self, ctx: &IrContext, ty: TypeRef) -> Option<TypeRef> {
        for conv in &self.conversions {
            if let Some(converted) = conv(ctx, ty) {
                return Some(converted);
            }
        }
        None
    }

    /// Convert a type, returning the original if no conversion applies.
    pub fn convert_type_or_identity(&self, ctx: &IrContext, ty: TypeRef) -> TypeRef {
        self.convert_type(ctx, ty).unwrap_or(ty)
    }

    /// Materialize a conversion from one type to another by creating cast ops.
    pub fn materialize(
        &self,
        ctx: &mut IrContext,
        location: Location,
        value: ValueRef,
        from_ty: TypeRef,
        to_ty: TypeRef,
    ) -> Option<MaterializeResult> {
        self.materializer.as_ref()?(ctx, location, value, from_ty, to_ty)
    }

    /// Check if this converter has any conversion functions.
    pub fn is_empty(&self) -> bool {
        self.conversions.is_empty()
    }
}

impl Default for ArenaTypeConverter {
    fn default() -> Self {
        Self::new()
    }
}
