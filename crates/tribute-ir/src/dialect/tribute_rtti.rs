//! Runtime type descriptor declarations.
//!
//! The target boundary that last reads nominal layouts declares, for each
//! runtime type descriptor, its number and how the runtime reads each of its
//! fields. A descriptor is a struct allocation layout, one variant of an
//! enum allocation layout, or on native an array type. The pass that lowers
//! allocations reads these
//! module-level declarations and erases them, so none reaches the
//! backend-ready boundary.

use std::fmt;
use trunk_ir::attr_kind::Type;
use trunk_ir::dialect::core::I32;
use trunk_ir::op_flag::Pure;

use rustc_hash::FxHashMap as HashMap;

use trunk_ir::TypeRef;
use trunk_ir::context::IrContext;
use trunk_ir::ops::DialectOp;
use trunk_ir::rewrite::Module;
use trunk_ir::types::{Attribute, Location};

use crate::dialect::adt;

#[trunk_ir::dialect]
mod tribute_rtti {
    /// The RTTI index and field kinds of one runtime type descriptor.
    ///
    /// For an `adt.struct` layout, `tag` is absent and `fields` lists one
    /// kind per field. For an `adt.enum` layout, `tag` is the variant's
    /// position, as for `adt.variant_new`, and `fields` lists one kind per
    /// field of that variant. For an `adt.array`, `tag` is absent and
    /// `fields` is the kind of its element.
    #[verify]
    #[asm_format(attr_dict_bare)]
    fn layout(
        r#type: Attr<Type>,
        tag: Option<Attr<u32>>,
        index: Attr<u32>,
        fields: Attr<[String]>,
    ) {
    }

    /// The runtime type descriptor number of the allocation `ref` points to.
    ///
    /// Reading a null reference is undefined.
    #[flags(Pure)]
    #[asm_format(r#ref attr_dict_bare ":" type(result))]
    fn descriptor(r#ref: Value<impl ManagedRef>) -> Value<I32> {}

    /// The position, among its enum's variants in declaration order, of
    /// the variant whose runtime type descriptor number is `descriptor`.
    ///
    /// ```text
    /// %position = tribute_rtti.variant_position %descriptor
    /// ```
    ///
    /// The native RTTI table records each variant's position, so this reads
    /// it there; the pass that emits the table lowers this operation. The
    /// result is undefined when `descriptor` is not the number of a variant.
    #[flags(Pure)]
    #[asm_format(descriptor attr_dict_bare)]
    fn variant_position(descriptor: Value<I32>) -> Value<I32> {}
}

/// A managed reference: a value that points to an allocation carrying a
/// runtime type descriptor. Unmanaged pointers and scalars are not.
///
/// A type is a managed reference when it implements [`ManagedRefType`].
pub struct ManagedRef;

/// Marks a type as a managed reference. A `#[dialect]` type registers its
/// impl with `#[implements(ManagedRefType)]`.
#[trunk_ir::type_interface]
pub trait ManagedRefType: trunk_ir::ops::DialectType {}

/// Marks a managed reference type as the layout of the allocation its values
/// point to: that allocation's runtime type descriptor is keyed by the type
/// itself, as `tribute_rtti.layout` names it. A struct layout, an enum layout
/// (one descriptor per variant), and on native an array type are.
///
/// `tribute_rt.anyref`, `tribute_rt.intref`, and
/// `closure.closure` are managed references but not descriptor layouts of
/// their own, so this is narrower than [`ManagedRefType`]. A nominal
/// layout's body is no allocation layout of its own; its key is.
#[trunk_ir::type_interface]
pub trait DescriptorLayoutType: trunk_ir::ops::DialectType {
    /// The field types of this layout's descriptor `tag`: a struct's fields
    /// without a tag, the fields of the enum variant at position `tag`, or
    /// an array's element without a tag.
    fn descriptor_fields<'a>(
        &self,
        ctx: &'a IrContext,
        tag: Option<u32>,
    ) -> Result<&'a [TypeRef], DescriptorFieldsError>;
}

impl DynDescriptorLayoutType {
    /// The field types of the descriptor `(ty, tag)`.
    pub fn fields_of(
        ctx: &IrContext,
        ty: TypeRef,
        tag: Option<u32>,
    ) -> Result<&[TypeRef], DescriptorFieldsError> {
        Self::get(ctx, ty)
            .ok_or(DescriptorFieldsError::NotLayout)?
            .descriptor_fields(ctx, tag)
    }
}

/// Why a type and a tag name no runtime type descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, derive_more::Display, derive_more::Error)]
pub enum DescriptorFieldsError {
    #[display("the type is not a well-formed descriptor layout")]
    NotLayout,
    #[display("an {layout} descriptor has no tag")]
    UnexpectedTag { layout: &'static str },
    #[display("an adt.enum descriptor needs a variant tag")]
    MissingTag,
    #[display("adt.enum has no variant at position {tag}")]
    NoVariant { tag: u32 },
}

impl trunk_ir::type_constraint::TypeConstraint for ManagedRef {
    const DESC: &'static trunk_ir::type_constraint::ConstraintDesc =
        &trunk_ir::type_constraint::ConstraintDesc {
            name: "ManagedRef",
            exact: false,
            projections: &[],
            matches: Self::matches,
            project: |_, _, _| None,
            fixed: None,
        };
}

impl ManagedRef {
    pub fn matches(ctx: &IrContext, ty: TypeRef) -> bool {
        DynManagedRefType::is_implemented(ctx, ty)
    }
}

/// How the runtime reads one field of an allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FieldKind {
    /// A managed reference to an allocation with a static layout.
    Managed,
    /// A managed value whose descriptor is read from the value itself.
    Dynamic,
    /// An unmanaged pointer or other raw bits, never followed.
    Raw,
    Int {
        width: u8,
        signed: bool,
    },
    Float {
        width: u8,
    },
    Bool,
}

impl FieldKind {
    /// The kind of a scalar field of semantic type `ty`, or `None` when `ty`
    /// is not a scalar.
    pub fn scalar(ctx: &IrContext, ty: TypeRef) -> Option<Self> {
        let data = ctx.get_type(ty);
        use crate::dialect::tribute_rt;
        use trunk_ir::dialect::core;
        use trunk_ir::ops::DialectType;

        let is = |name: trunk_ir::TypeName| data.name == name;
        if is(core::I1::NAME) || is(tribute_rt::Bool::NAME) {
            return Some(Self::Bool);
        }
        if is(tribute_rt::Int::NAME) {
            return Some(Self::Int {
                width: 32,
                signed: true,
            });
        }
        if is(tribute_rt::Nat::NAME) {
            return Some(Self::Int {
                width: 32,
                signed: false,
            });
        }
        if is(tribute_rt::Float::NAME) || is(core::F64::NAME) {
            return Some(Self::Float { width: 64 });
        }
        if is(core::F32::NAME) {
            return Some(Self::Float { width: 32 });
        }
        // A `core` integer carries no sign, so it records as unsigned.
        let width = (data.name.dialect(ctx) == core::DIALECT_NAME)
            .then(|| {
                data.name
                    .strip_dialect(ctx)
                    .strip_prefix('i')
                    .and_then(|width| width.parse().ok())
            })
            .flatten()?;
        Some(Self::Int {
            width,
            signed: false,
        })
    }

    /// Whether releasing the allocation releases this field.
    pub fn is_released(self) -> bool {
        matches!(self, Self::Managed | Self::Dynamic)
    }

    /// The `field_kind` word of a native descriptor record: the class in the
    /// low byte and a scalar's bit width in the next.
    pub fn record_code(self) -> u32 {
        let (class, width) = match self {
            Self::Managed => (0, 0),
            Self::Dynamic => (1, 0),
            Self::Raw => (2, 0),
            Self::Int {
                width,
                signed: true,
            } => (3, width),
            Self::Int {
                width,
                signed: false,
            } => (4, width),
            Self::Float { width } => (5, width),
            Self::Bool => (6, 0),
        };
        class | (u32::from(width) << 8)
    }

    fn parse(text: &str) -> Option<Self> {
        Some(match text {
            "managed" => Self::Managed,
            "dynamic" => Self::Dynamic,
            "raw" => Self::Raw,
            "bool" => Self::Bool,
            _ => {
                let (class, width) = text.split_at_checked(1)?;
                // Only the canonical spelling `Display` prints.
                if width.starts_with('0') || !width.bytes().all(|byte| byte.is_ascii_digit()) {
                    return None;
                }
                let width = width.parse().ok()?;
                match class {
                    "i" => Self::Int {
                        width,
                        signed: true,
                    },
                    "u" => Self::Int {
                        width,
                        signed: false,
                    },
                    "f" => Self::Float { width },
                    _ => return None,
                }
            }
        })
    }
}

impl fmt::Display for FieldKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Managed => f.write_str("managed"),
            Self::Dynamic => f.write_str("dynamic"),
            Self::Raw => f.write_str("raw"),
            Self::Int {
                width,
                signed: true,
            } => write!(f, "i{width}"),
            Self::Int {
                width,
                signed: false,
            } => write!(f, "u{width}"),
            Self::Float { width } => write!(f, "f{width}"),
            Self::Bool => f.write_str("bool"),
        }
    }
}

/// The runtime type descriptor an allocation operation creates: its layout
/// type, and the variant's position for an `adt.variant_new`.
pub fn allocation_descriptor(
    ctx: &IrContext,
    op: trunk_ir::OpRef,
) -> Option<(TypeRef, Option<u32>)> {
    if let Ok(new) = adt::StructNew::from_op(ctx, op) {
        return Some((new.layout(ctx), None));
    }
    let new = adt::VariantNew::from_op(ctx, op).ok()?;
    Some((new.layout(ctx), Some(new.variant(ctx))))
}

/// The runtime type descriptor a native allocation operation creates: an
/// [`allocation_descriptor`], or the array type an array construction
/// allocates.
pub fn native_allocation_descriptor(
    ctx: &IrContext,
    op: trunk_ir::OpRef,
) -> Option<(TypeRef, Option<u32>)> {
    let array_ty = if let Ok(new) = adt::ArrayNew::from_op(ctx, op) {
        new.r#type(ctx)
    } else if let Ok(of) = adt::ArrayOf::from_op(ctx, op) {
        of.r#type(ctx)
    } else if let Ok(splice) = adt::ArraySplice::from_op(ctx, op) {
        splice.r#type(ctx)
    } else {
        return allocation_descriptor(ctx, op);
    };
    Some((array_ty, None))
}

impl Layout {
    /// Build a layout declaration for the descriptor `(ty, tag)`.
    pub fn declare(
        ctx: &mut IrContext,
        location: Location,
        ty: TypeRef,
        tag: Option<u32>,
        index: u32,
        fields: &[FieldKind],
    ) -> Self {
        let mut builder = Self::operands()
            .r#type(ty)
            .index(index)
            .fields(fields.iter().map(ToString::to_string));
        if let Some(tag) = tag {
            builder = builder.tag(tag);
        }
        builder.build(ctx, location)
    }

    /// The field kinds of a verified declaration.
    pub fn field_kinds(self, ctx: &IrContext) -> Vec<FieldKind> {
        self.decode(ctx).expect("verified tribute_rtti.layout")
    }

    fn decode(self, ctx: &IrContext) -> Result<Vec<FieldKind>, String> {
        let expected = DynDescriptorLayoutType::fields_of(ctx, self.r#type(ctx), self.tag(ctx))
            .map_err(|error| error.to_string())?
            .len();
        let fields = self.fields(ctx);
        if fields.len() != expected {
            return Err(format!(
                "expected {expected} field kind(s), found {}",
                fields.len()
            ));
        }
        fields
            .map(|text| {
                FieldKind::parse(text).ok_or_else(|| format!("unknown field kind \"{text}\""))
            })
            .collect()
    }

    /// The declarations of a module, in module order.
    pub fn declared(ctx: &IrContext, module: Module) -> Vec<Self> {
        module
            .ops(ctx)
            .iter()
            .filter_map(|&op| Self::from_op(ctx, op).ok())
            .collect()
    }

    /// The declared number of each descriptor of a module: a struct layout,
    /// or an enum layout and one of its variant tags.
    pub fn declared_indices(
        ctx: &IrContext,
        module: Module,
    ) -> HashMap<(TypeRef, Option<u32>), u32> {
        Self::declared(ctx, module)
            .into_iter()
            .map(|layout| ((layout.r#type(ctx), layout.tag(ctx)), layout.index(ctx)))
            .collect()
    }

    /// Point the declaration at another layout type with the same shape.
    pub fn set_type(self, ctx: &mut IrContext, ty: TypeRef) {
        ctx.op_attrs_mut(self.op_ref())
            .insert("type", Attribute::Type(ty));
    }
}

impl trunk_ir::ops::Verify for Layout {
    /// `fields` matches the field shape of the declared descriptor.
    fn verify(&self, ctx: &IrContext) -> Result<(), trunk_ir::ops::VerificationError> {
        Ok(self.decode(ctx).map(|_| ())?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::op_def::OpDef;
    use trunk_ir::ops::DialectOp;
    use trunk_ir::parser::parse_test_module;

    #[test]
    fn managed_references_are_the_types_registered_at_their_definitions() {
        let mut ctx = IrContext::new();
        parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !point = adt.struct<"Point", (core.i32)>
  !point_ref = adt.struct<"Point">
  !choice = adt.enum<"Choice", [()]>
  !any = tribute_rt.anyref
  !small = tribute_rt.intref
  !raw = core.ptr
  !int = tribute_rt.int
  !callable = closure.closure<func.func_sig<() -> core.nil>>
  !array = adt.array<core.i32>
  !linear = core.array<core.i32>
}"#,
        );
        let alias = |name| ctx.type_alias_by_text(name).expect("fixture alias");

        for name in [
            "point",
            "point_ref",
            "choice",
            "any",
            "small",
            "callable",
            "array",
        ] {
            assert!(ManagedRef::matches(&ctx, alias(name)), "{name}");
        }
        for name in ["raw", "int", "linear"] {
            assert!(!ManagedRef::matches(&ctx, alias(name)), "{name}");
        }
    }

    #[test]
    fn layout_decodes_struct_and_variant_field_kinds() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !Pair = adt.struct<"Pair", (tribute_rt.anyref, core.i64)>
  !Choice = adt.enum<"Choice", [(), (!Pair, core.f64)]>
  tribute_rtti.layout {type = !Pair, index = 32, fields = ["dynamic", "u64"]}
  tribute_rtti.layout {type = !Choice, tag = 1, index = 33, fields = ["managed", "f64"]}
}"#,
        );
        let layouts = module
            .ops(&ctx)
            .iter()
            .copied()
            .map(|op| Layout::from_op(&ctx, op).expect("layout"))
            .collect::<Vec<_>>();

        assert_eq!(layouts[0].index(&ctx), 32);
        assert_eq!(
            layouts[0].field_kinds(&ctx),
            [
                FieldKind::Dynamic,
                FieldKind::Int {
                    width: 64,
                    signed: false
                }
            ]
        );
        assert_eq!(layouts[1].tag(&ctx), Some(1));
        assert_eq!(
            layouts[1].field_kinds(&ctx),
            [FieldKind::Managed, FieldKind::Float { width: 64 }]
        );
    }

    #[test]
    fn field_kind_record_codes_put_the_width_above_the_class() {
        assert_eq!(FieldKind::Managed.record_code(), 0);
        assert_eq!(FieldKind::Bool.record_code(), 6);
        assert_eq!(
            FieldKind::Int {
                width: 32,
                signed: true
            }
            .record_code(),
            3 | (32 << 8)
        );
        assert_eq!(FieldKind::Float { width: 64 }.record_code(), 5 | (64 << 8));
    }

    #[test]
    fn layout_verifier_rejects_fields_of_another_shape() {
        for (layout, expected) in [
            (
                r#"tribute_rtti.layout {type = !Pair, index = 32, fields = ["dynamic"]}"#,
                "expected 2 field kind(s)",
            ),
            (
                r#"tribute_rtti.layout {type = !Pair, index = 32, fields = ["dynamic", "word"]}"#,
                "unknown field kind \"word\"",
            ),
            (
                r#"tribute_rtti.layout {type = !Pair, index = 32, fields = ["dynamic", "i064"]}"#,
                "unknown field kind \"i064\"",
            ),
            (
                r#"tribute_rtti.layout {type = !Pair, index = 32, fields = ["dynamic", "u+8"]}"#,
                "unknown field kind \"u+8\"",
            ),
            (
                r#"tribute_rtti.layout {type = !Choice, index = 33, fields = []}"#,
                "needs a variant tag",
            ),
            (
                r#"tribute_rtti.layout {type = !Choice, tag = 2, index = 33, fields = []}"#,
                "has no variant at position 2",
            ),
        ] {
            let mut ctx = IrContext::new();
            let module = parse_test_module(
                &mut ctx,
                &format!(
                    r#"core.module @test {{
  !Pair = adt.struct<"Pair", (tribute_rt.anyref, core.i64)>
  !Choice = adt.enum<"Choice", [()]>
  {layout}
}}"#
                ),
            );
            let op = module.ops(&ctx)[0];

            let violations = OpDef::of(&ctx, op).expect("registered").verify(&ctx, op);

            assert!(
                violations
                    .iter()
                    .any(|violation| violation.to_string().contains(expected)),
                "{expected}: {violations:?}"
            );
        }
    }
}
