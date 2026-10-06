//! Runtime type descriptor declarations.
//!
//! The target boundary that last reads nominal layouts declares, for each
//! runtime type descriptor, its number and how the runtime reads each of its
//! fields. A descriptor is a struct allocation layout, or one variant of an
//! enum allocation layout. The pass that lowers allocations reads these
//! module-level declarations and erases them, so none reaches the
//! backend-ready boundary.

use std::fmt;
use trunk_ir::attr_kind::Type;
use trunk_ir::dialect::core::I32;

use rustc_hash::FxHashMap as HashMap;

use trunk_ir::TypeRef;
use trunk_ir::context::IrContext;
use trunk_ir::ops::DialectOp;
use trunk_ir::rewrite::Module;
use trunk_ir::types::{Attribute, Location, StringRef};

use crate::dialect::adt;
use crate::dialect::adt::layout::{get_enum_variants, get_struct_fields};

trunk_ir::register_pure_op!(Descriptor);

#[trunk_ir::dialect]
mod tribute_rtti {
    /// The RTTI index and field kinds of one runtime type descriptor.
    ///
    /// For an `adt.struct` layout, `tag` is absent and `fields` lists one
    /// kind per field. For an `adt.enum` layout, `tag` names the variant and
    /// `fields` lists one kind per field of that variant.
    #[verify]
    fn layout(
        r#type: Attr<Type>,
        tag: Option<Attr<String>>,
        index: Attr<u32>,
        fields: Attr<[String]>,
    ) {
    }

    /// The runtime type descriptor number of the allocation `ref` points to.
    ///
    /// Reading a null reference is undefined.
    fn descriptor(r#ref: Value<impl ManagedRef>) -> Value<I32> {}
}

/// A managed reference: a value that points to an allocation carrying a
/// runtime type descriptor. Unmanaged pointers and scalars are not.
pub struct ManagedRef;

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
        let data = ctx.get_type(ty);
        let is = |dialect: &'static str, name: &'static str| {
            data.dialect == trunk_ir::Symbol::new(dialect)
                && data.name == trunk_ir::Symbol::new(name)
        };
        is("adt", "typeref")
            || is("adt", "struct")
            || is("adt", "enum")
            || is("tribute_rt", "anyref")
            || is("tribute_rt", "intref")
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
        let is = |dialect: &'static str, name: &'static str| {
            data.dialect == trunk_ir::Symbol::new(dialect)
                && data.name == trunk_ir::Symbol::new(name)
        };
        if is("core", "i1") || is("tribute_rt", "bool") {
            return Some(Self::Bool);
        }
        if is("tribute_rt", "int") {
            return Some(Self::Int {
                width: 32,
                signed: true,
            });
        }
        if is("tribute_rt", "nat") {
            return Some(Self::Int {
                width: 32,
                signed: false,
            });
        }
        if is("tribute_rt", "float") || is("core", "f64") {
            return Some(Self::Float { width: 64 });
        }
        if is("core", "f32") {
            return Some(Self::Float { width: 32 });
        }
        // A `core` integer carries no sign, so it records as unsigned.
        let width = (data.dialect == trunk_ir::Symbol::new("core"))
            .then(|| {
                data.name
                    .with_str(|name| name.strip_prefix('i').and_then(|width| width.parse().ok()))
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
/// type, and the variant tag for an `adt.variant_new`.
pub fn allocation_descriptor(
    ctx: &IrContext,
    op: trunk_ir::OpRef,
) -> Option<(TypeRef, Option<StringRef>)> {
    if let Ok(new) = adt::StructNew::from_op(ctx, op) {
        return Some((new.r#type(ctx), None));
    }
    let new = adt::VariantNew::from_op(ctx, op).ok()?;
    Some((new.r#type(ctx), Some(new.tag_ref(ctx))))
}

/// The field types of the descriptor `(ty, tag)`: the struct's fields, or
/// the fields of the enum variant `tag`.
pub fn descriptor_field_types(
    ctx: &IrContext,
    ty: TypeRef,
    tag: Option<StringRef>,
) -> Result<Vec<TypeRef>, String> {
    match tag {
        None => get_struct_fields(ctx, ty)
            .map(|fields| fields.into_iter().map(|(_, ty)| ty).collect())
            .ok_or_else(|| "a layout without a tag must be an adt.struct".into()),
        Some(tag) => {
            let variants = get_enum_variants(ctx, ty)
                .ok_or_else(|| "a layout with a tag must be an adt.enum".to_owned())?;
            variants
                .into_iter()
                .find(|(name, _)| *name == tag)
                .map(|(_, fields)| fields)
                .ok_or_else(|| format!("adt.enum has no variant \"{}\"", ctx.str(tag)))
        }
    }
}

impl Layout {
    /// Build a layout declaration for the descriptor `(ty, tag)`.
    pub fn declare(
        ctx: &mut IrContext,
        location: Location,
        ty: TypeRef,
        tag: Option<StringRef>,
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
        let expected = descriptor_field_types(ctx, self.r#type(ctx), self.tag_ref(ctx))?.len();
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
    ) -> HashMap<(TypeRef, Option<StringRef>), u32> {
        Self::declared(ctx, module)
            .into_iter()
            .map(|layout| ((layout.r#type(ctx), layout.tag_ref(ctx)), layout.index(ctx)))
            .collect()
    }

    /// Point the declaration at another layout type with the same shape.
    pub fn set_type(self, ctx: &mut IrContext, ty: TypeRef) {
        ctx.op_mut(self.op_ref())
            .attributes
            .insert("type", Attribute::Type(ty));
    }
}

impl trunk_ir::ops::Verify for Layout {
    /// `fields` matches the field shape of the declared descriptor.
    fn verify(self, ctx: &IrContext) -> Result<(), String> {
        self.decode(ctx).map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::op_def::OpDef;
    use trunk_ir::ops::DialectOp;
    use trunk_ir::parser::parse_test_module;

    #[test]
    fn layout_decodes_struct_and_variant_field_kinds() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !Pair = adt.struct<Pair(left: tribute_rt.anyref, right: core.i64)>
  !Choice = adt.enum<Choice { None(), Some(!Pair, core.f64) }>
  tribute_rtti.layout {type = !Pair, index = 32, fields = ["dynamic", "u64"]}
  tribute_rtti.layout {type = !Choice, tag = "Some", index = 33, fields = ["managed", "f64"]}
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
        assert_eq!(layouts[1].tag(&ctx), Some("Some"));
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
                "must be an adt.struct",
            ),
            (
                r#"tribute_rtti.layout {type = !Choice, tag = "Other", index = 33, fields = []}"#,
                "has no variant \"Other\"",
            ),
        ] {
            let mut ctx = IrContext::new();
            let module = parse_test_module(
                &mut ctx,
                &format!(
                    r#"core.module @test {{
  !Pair = adt.struct<Pair(left: tribute_rt.anyref, right: core.i64)>
  !Choice = adt.enum<Choice {{ None() }}>
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
