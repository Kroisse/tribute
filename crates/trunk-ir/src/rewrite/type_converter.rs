//! Arena-based type converter.
//!
//! Provides type conversion infrastructure for arena IR dialect lowering.

use crate::context::IrContext;
use crate::refs::{OpRef, TypeRef, ValueRef};
use crate::types::Location;

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

/// Subsumption predicate signature: whether the target accepts a value of the
/// first type where the second is declared.
type SubsumptionFn = dyn Fn(&IrContext, TypeRef, TypeRef) -> bool;

/// Arena type converter — maps types during dialect conversion.
///
/// Holds a collection of conversion functions and a materialization callback
/// for inserting cast operations.
pub struct TypeConverter {
    /// Type conversion functions, tried in order.
    conversions: Vec<Box<ConversionFn>>,
    /// Materialization function: creates cast ops when needed.
    materializer: Option<Box<MaterializerFn>>,
    /// Target subtyping: a value of a subtype is accepted without a conversion.
    subsumption: Option<Box<SubsumptionFn>>,
}

impl TypeConverter {
    /// Create a new empty type converter.
    pub fn new() -> Self {
        Self {
            conversions: Vec::new(),
            materializer: None,
            subsumption: None,
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

    /// Set the target's subtyping relation.
    ///
    /// A target whose type system accepts a value of a subtype where a
    /// supertype is declared (such as WasmGC references) reports it here. A
    /// cast from a subtype needs no conversion and its source is used
    /// directly; a representation match alone is not subsumption.
    pub fn set_subsumption(&mut self, f: impl Fn(&IrContext, TypeRef, TypeRef) -> bool + 'static) {
        self.subsumption = Some(Box::new(f));
    }

    /// Whether the target accepts a value of `from_ty` where `to_ty` is declared.
    pub fn is_subsumed(&self, ctx: &IrContext, from_ty: TypeRef, to_ty: TypeRef) -> bool {
        self.subsumption
            .as_ref()
            .is_some_and(|subsumes| subsumes(ctx, from_ty, to_ty))
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
        self.conversions.is_empty() && self.materializer.is_none() && self.subsumption.is_none()
    }
}

impl Default for TypeConverter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::location::Span;
    use crate::*;

    fn test_ctx() -> (IrContext, Location) {
        let mut ctx = IrContext::new();
        let path = ctx.intern_path("test.trb".to_owned());
        let loc = Location::new(path, Span::new(0, 0));
        (ctx, loc)
    }

    fn i32_type(ctx: &mut IrContext) -> TypeRef {
        ctx.intern_type(TypeDataBuilder::new("core", "i32").build())
    }

    fn i64_type(ctx: &mut IrContext) -> TypeRef {
        ctx.intern_type(TypeDataBuilder::new("core", "i64").build())
    }

    #[test]
    fn is_empty_true_when_no_conversions_or_materializer() {
        let tc = TypeConverter::new();
        assert!(tc.is_empty());
    }

    #[test]
    fn is_empty_false_with_conversion() {
        let mut tc = TypeConverter::new();
        tc.add_conversion(|_, _| None);
        assert!(!tc.is_empty());
    }

    #[test]
    fn is_empty_false_with_materializer_only() {
        let mut tc = TypeConverter::new();
        tc.set_materializer(|_, _, _, _, _| None);
        assert!(!tc.is_empty());
    }

    #[test]
    fn convert_type_returns_none_when_empty() {
        let (mut ctx, _) = test_ctx();
        let ty = i32_type(&mut ctx);
        let tc = TypeConverter::new();
        assert!(tc.convert_type(&ctx, ty).is_none());
    }

    #[test]
    fn convert_type_applies_first_match() {
        let (mut ctx, _) = test_ctx();
        let i32_ty = i32_type(&mut ctx);
        let i64_ty = i64_type(&mut ctx);

        let target = i64_ty;
        let mut tc = TypeConverter::new();
        tc.add_conversion(move |_, _| Some(target));

        assert_eq!(tc.convert_type(&ctx, i32_ty), Some(i64_ty));
    }

    #[test]
    fn convert_type_or_identity_falls_back() {
        let (mut ctx, _) = test_ctx();
        let ty = i32_type(&mut ctx);
        let tc = TypeConverter::new();
        assert_eq!(tc.convert_type_or_identity(&ctx, ty), ty);
    }
}
