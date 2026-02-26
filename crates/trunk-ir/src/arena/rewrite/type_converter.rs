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

    /// Check if this converter has any conversions or materializer.
    pub fn is_empty(&self) -> bool {
        self.conversions.is_empty() && self.materializer.is_none()
    }
}

impl Default for ArenaTypeConverter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::*;
    use crate::ir::Symbol;
    use crate::location::Span;

    fn test_ctx() -> (IrContext, Location) {
        let mut ctx = IrContext::new();
        let path = ctx.paths.intern("test.trb".to_owned());
        let loc = Location::new(path, Span::new(0, 0));
        (ctx, loc)
    }

    fn i32_type(ctx: &mut IrContext) -> TypeRef {
        ctx.types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build())
    }

    fn i64_type(ctx: &mut IrContext) -> TypeRef {
        ctx.types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i64")).build())
    }

    #[test]
    fn is_empty_true_when_no_conversions_or_materializer() {
        let tc = ArenaTypeConverter::new();
        assert!(tc.is_empty());
    }

    #[test]
    fn is_empty_false_with_conversion() {
        let mut tc = ArenaTypeConverter::new();
        tc.add_conversion(|_, _| None);
        assert!(!tc.is_empty());
    }

    #[test]
    fn is_empty_false_with_materializer_only() {
        let mut tc = ArenaTypeConverter::new();
        tc.set_materializer(|_, _, _, _, _| None);
        assert!(!tc.is_empty());
    }

    #[test]
    fn convert_type_returns_none_when_empty() {
        let (mut ctx, _) = test_ctx();
        let ty = i32_type(&mut ctx);
        let tc = ArenaTypeConverter::new();
        assert!(tc.convert_type(&ctx, ty).is_none());
    }

    #[test]
    fn convert_type_applies_first_match() {
        let (mut ctx, _) = test_ctx();
        let i32_ty = i32_type(&mut ctx);
        let i64_ty = i64_type(&mut ctx);

        let target = i64_ty;
        let mut tc = ArenaTypeConverter::new();
        tc.add_conversion(move |_, _| Some(target));

        assert_eq!(tc.convert_type(&ctx, i32_ty), Some(i64_ty));
    }

    #[test]
    fn convert_type_or_identity_falls_back() {
        let (mut ctx, _) = test_ctx();
        let ty = i32_type(&mut ctx);
        let tc = ArenaTypeConverter::new();
        assert_eq!(tc.convert_type_or_identity(&ctx, ty), ty);
    }
}
