//! Registered type definitions generated from `#[dialect]`.

use std::sync::LazyLock;

use crate::name::{Definition, DefinitionTable, Name, NameArg, QualifiedName};

/// The name of a type.
pub type TypeName = Name<TypeDef>;

/// A type name given to a builder.
pub type TypeNameArg = NameArg<TypeDef>;

/// What `#[dialect]` registers for one type.
///
/// `#[dialect]` emits each definition as a `static`, so its address
/// identifies the type: a [`TypeName`] of a registered type points to it.
#[derive(Debug)]
pub struct TypeDef {
    /// The type's `dialect.name`.
    pub name: QualifiedName,
}

impl TypeDef {
    /// The registered definition written `dialect.name`.
    pub fn lookup(name: &str) -> Option<&'static TypeDef> {
        REGISTRY.get(name)
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
    use crate::IrContext;

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
