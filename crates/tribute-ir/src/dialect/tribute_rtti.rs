//! Native RTTI layout declarations.
//!
//! Native ownership planning declares, for each managed allocation layout,
//! its RTTI index and which of its fields hold managed references. Native
//! RTTI generation and RC header lowering read these module-level
//! declarations, and RC header lowering erases them, so none reaches the
//! backend-ready boundary.

use trunk_ir::adt_layout::{get_enum_variants, get_struct_fields};
use trunk_ir::context::IrContext;
use trunk_ir::types::{Attribute, Location};
use trunk_ir::{Symbol, TypeRef};

#[trunk_ir::dialect]
mod tribute_rtti {
    /// The RTTI index and managed-field bitmap of one allocation layout.
    ///
    /// For an `adt.struct` layout, `managed` lists one bool per field. For an
    /// `adt.enum` layout, it lists one such list per variant.
    #[verify]
    fn layout(r#type: Attr<Type>, index: Attr<u32>, managed: Attr<_>) {}
}

/// Which fields of an allocation layout hold managed references.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ManagedFieldBitmap {
    Struct(Vec<bool>),
    Enum(Vec<Vec<bool>>),
}

impl ManagedFieldBitmap {
    fn to_attribute(&self) -> Attribute {
        fn bools(fields: &[bool]) -> Attribute {
            Attribute::List(fields.iter().copied().map(Attribute::Bool).collect())
        }
        match self {
            Self::Struct(fields) => bools(fields),
            Self::Enum(variants) => {
                Attribute::List(variants.iter().map(|fields| bools(fields)).collect())
            }
        }
    }

    /// Decode `managed` against the shape of `ty`.
    fn from_attribute(ctx: &IrContext, ty: TypeRef, attribute: &Attribute) -> Result<Self, String> {
        fn bools(attribute: &Attribute, expected: usize) -> Result<Vec<bool>, String> {
            let Attribute::List(items) = attribute else {
                return Err("expected a list of bools".into());
            };
            if items.len() != expected {
                return Err(format!(
                    "expected {expected} field flag(s), found {}",
                    items.len()
                ));
            }
            items
                .iter()
                .map(|item| match item {
                    Attribute::Bool(flag) => Ok(*flag),
                    _ => Err("expected a list of bools".to_owned()),
                })
                .collect()
        }

        if let Some(fields) = get_struct_fields(ctx, ty) {
            return bools(attribute, fields.len()).map(Self::Struct);
        }
        if let Some(variants) = get_enum_variants(ctx, ty) {
            let Attribute::List(items) = attribute else {
                return Err("expected one list of bools per variant".into());
            };
            if items.len() != variants.len() {
                return Err(format!(
                    "expected {} variant(s), found {}",
                    variants.len(),
                    items.len()
                ));
            }
            return items
                .iter()
                .zip(&variants)
                .map(|(item, (_, fields))| bools(item, fields.len()))
                .collect::<Result<_, _>>()
                .map(Self::Enum);
        }
        Err("layout type must be an adt.struct or adt.enum".into())
    }
}

impl Layout {
    /// Build a layout declaration for `ty`.
    pub fn declare(
        ctx: &mut IrContext,
        location: Location,
        ty: TypeRef,
        index: u32,
        managed: &ManagedFieldBitmap,
    ) -> Self {
        Self::operands()
            .r#type(ty)
            .index(index)
            .managed(managed.to_attribute())
            .build(ctx, location)
    }

    /// The managed-field bitmap of a verified declaration.
    pub fn managed_fields(self, ctx: &IrContext) -> ManagedFieldBitmap {
        ManagedFieldBitmap::from_attribute(ctx, self.r#type(ctx), &self.managed(ctx))
            .expect("verified tribute_rtti.layout")
    }

    /// Point the declaration at another layout type with the same shape.
    pub fn set_type(self, ctx: &mut IrContext, ty: TypeRef) {
        ctx.op_mut(self.op_ref())
            .attributes
            .insert(Symbol::new("type"), Attribute::Type(ty));
    }
}

impl trunk_ir::ops::Verify for Layout {
    /// `managed` matches the field shape of the declared layout type.
    fn verify(self, ctx: &IrContext) -> Result<(), String> {
        ManagedFieldBitmap::from_attribute(ctx, self.r#type(ctx), &self.managed(ctx)).map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::op_def::OpDef;
    use trunk_ir::parser::parse_test_module;

    #[test]
    fn layout_decodes_struct_and_enum_bitmaps() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !Pair = adt.struct() {name = @Pair, fields = [[@left, tribute_rt.anyref], [@right, core.i64]]}
  !Choice = adt.enum() {name = @Choice, variants = [[@None, []], [@Some, [tribute_rt.anyref]]]}
  tribute_rtti.layout {type = !Pair, index = 32, managed = [true, false]}
  tribute_rtti.layout {type = !Choice, index = 33, managed = [[], [true]]}
}"#,
        );
        let layouts = module
            .ops(&ctx)
            .into_iter()
            .map(|op| Layout::from_op(&ctx, op).expect("layout"))
            .collect::<Vec<_>>();

        assert_eq!(layouts[0].index(&ctx), 32);
        assert_eq!(
            layouts[0].managed_fields(&ctx),
            ManagedFieldBitmap::Struct(vec![true, false])
        );
        assert_eq!(
            layouts[1].managed_fields(&ctx),
            ManagedFieldBitmap::Enum(vec![vec![], vec![true]])
        );
    }

    #[test]
    fn layout_verifier_rejects_a_bitmap_of_another_shape() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !Pair = adt.struct() {name = @Pair, fields = [[@left, tribute_rt.anyref], [@right, core.i64]]}
  tribute_rtti.layout {type = !Pair, index = 32, managed = [true]}
}"#,
        );
        let op = module.ops(&ctx)[0];

        let violations = OpDef::of(&ctx, op).expect("registered").verify(&ctx, op);

        assert!(
            violations
                .iter()
                .any(|violation| violation.to_string().contains("expected 2 field flag(s)")),
            "{violations:?}"
        );
    }
}
