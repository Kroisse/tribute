//! Registered type definitions generated from `#[dialect]` or written by
//! hand.

use std::sync::LazyLock;

use crate::asm_format::TypeAsmHooks;
use crate::name::{Definition, DefinitionTable, Name, NameArg, QualifiedName};
use crate::type_verifier::TypeVerifierFn;
use crate::{IrContext, TypeRef};

/// The name of a type.
pub type TypeName = Name<TypeDef>;

/// A type name given to a builder.
pub type TypeNameArg = NameArg<TypeDef>;

/// What `#[dialect]` registers for one type: its name and the hooks its
/// declaration opts into, such as `#[verify]` and `#[asm_format]`.
///
/// `#[dialect]` emits each definition as a `static`, so its address
/// identifies the type: a [`TypeName`] of a registered type points to it.
#[derive(Debug)]
pub struct TypeDef {
    /// The type's `dialect.name`.
    pub name: QualifiedName,
    /// Calls the wrapper's [`VerifyType`](crate::type_verifier::VerifyType)
    /// impl for types with a verifier.
    pub verifier: Option<TypeVerifierFn>,
    /// Calls the wrapper's [`TypeAsmFormat`](crate::asm_format::TypeAsmFormat)
    /// impl for types with a custom assembly format.
    pub asm_format: Option<TypeAsmHooks>,
}

impl TypeDef {
    /// A definition without hooks.
    pub const fn new(name: &'static str) -> Self {
        Self {
            name: QualifiedName::new(name),
            verifier: None,
            asm_format: None,
        }
    }

    /// The registered definition written `dialect.name`.
    pub fn lookup(name: &str) -> Option<&'static TypeDef> {
        REGISTRY.get(name)
    }

    /// The registered definition of a type.
    pub fn of(ctx: &IrContext, ty: TypeRef) -> Option<&'static TypeDef> {
        ctx.get_type(ty).name.def()
    }
}

impl Definition for TypeDef {
    fn qualified_name(&self) -> &QualifiedName {
        &self.name
    }

    fn lookup(name: &str) -> Option<&'static Self> {
        REGISTRY.get(name)
    }
}

/// Inventory entry registering a [`TypeDef`].
///
/// Emitted by the `#[dialect]` macro for every declared type.
pub struct TypeDefRegistration(pub &'static TypeDef);

inventory::collect!(TypeDefRegistration);

static REGISTRY: LazyLock<DefinitionTable<TypeDef>> = LazyLock::new(|| {
    DefinitionTable::collect(
        "type",
        inventory::iter::<TypeDefRegistration>
            .into_iter()
            .map(|TypeDefRegistration(def)| *def),
    )
});

#[cfg(test)]
mod tests {
    use super::*;

    #[crate::dialect]
    mod type_hooks_test {
        #[verify]
        struct Checked;

        #[asm_format]
        struct Custom;
    }

    impl crate::type_verifier::VerifyType for Checked {
        fn verify(ctx: &IrContext, ty: TypeRef) -> Result<(), String> {
            match ctx.type_attrs(ty).contains_key("bad") {
                true => Err("bad checked type".into()),
                false => Ok(()),
            }
        }
    }

    impl crate::asm_format::TypeAsmFormat for Custom {
        fn print(self, h: &mut crate::printer::TypePrintHelper<'_, '_>) -> std::fmt::Result {
            use std::fmt::Write;
            h.write_str("type_hooks_test.custom<!>")
        }

        fn parse<'a>(
            input: &mut &'a str,
            name: &'a str,
        ) -> winnow::ModalResult<crate::parser::raw::RawType<'a>> {
            use winnow::prelude::*;
            "!>".parse_next(input)?;
            Ok(crate::parser::raw::RawType::Concrete {
                name,
                params: Vec::new(),
                attrs: Default::default(),
            })
        }
    }

    #[test]
    fn declared_type_hooks_reach_their_definitions() {
        use crate::ops::DialectType;

        let checked = TypeDef::lookup("type_hooks_test.checked").unwrap();
        assert!(checked.verifier.is_some() && checked.asm_format.is_none());
        let custom = TypeDef::lookup("type_hooks_test.custom").unwrap();
        assert!(custom.verifier.is_none() && custom.asm_format.is_some());

        let mut ctx = IrContext::new();
        let bad = crate::TypeDataBuilder::new(Checked::NAME)
            .attr("bad", crate::Attribute::Bool(true))
            .intern(&mut ctx);
        let verifier = TypeDef::of(&ctx, bad).and_then(|def| def.verifier).unwrap();
        assert_eq!(verifier(&ctx, bad), Err("bad checked type".into()));

        let ty = crate::TypeDataBuilder::new(Custom::NAME).intern(&mut ctx);
        assert_eq!(
            crate::printer::print_type(&ctx, ty),
            "type_hooks_test.custom<!>"
        );
        let mut input = "type_hooks_test.custom<!>";
        let raw = crate::parser::raw::raw_type(&mut input).unwrap();
        assert!(input.is_empty());
        assert!(matches!(
            raw,
            crate::parser::raw::RawType::Concrete {
                name: "type_hooks_test.custom",
                ..
            }
        ));
    }

    #[test]
    fn every_registered_type_name_has_one_handle() {
        let mut ctx = IrContext::new();
        for TypeDefRegistration(def) in inventory::iter::<TypeDefRegistration> {
            assert_eq!(
                ctx.intern_type_name(def.name.as_str()),
                TypeName::registered(def)
            );
        }
        let i32_ty = crate::TypeDataBuilder::new("core.i32").intern(&mut ctx);
        assert!(ctx.get_type(i32_ty).name.is_registered());
        assert!(crate::dialect::core::I32::matches(&ctx, i32_ty));
        assert_eq!(
            i32_ty,
            crate::TypeDataBuilder::new(crate::dialect::core::I32::NAME).intern(&mut ctx)
        );
    }
}
