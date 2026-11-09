//! Registered type definitions generated from `#[dialect]` or written by
//! hand.

use std::sync::LazyLock;

use crate::asm_format::{
    TypeAliasHint, TypeAliasHook, TypeAsmFormat, TypeAsmHooks, erased_alias_hint,
};
use crate::name::{Definition, DefinitionTable, Name, NameArg, QualifiedName};
use crate::ops::{DialectType, VerificationError};
use crate::type_verifier::{VerifyType, erased_verifier};
use crate::{IrContext, TypeRef};

/// The name of a type.
pub type TypeName = Name<TypeDef>;

/// A type name given to a builder.
pub type TypeNameArg = NameArg<TypeDef>;

/// What `#[dialect]` registers for one type: its name and the hooks its
/// declaration opts into, such as `#[verify]`, `#[asm_format]`, and
/// `#[alias_hint]`. A hand-written definition is registered with
/// [`register_type_def!`](crate::register_type_def).
///
/// `#[dialect]` emits each definition as a `static`, so its address
/// identifies the type: a [`TypeName`] of a registered type points to it.
#[derive(Debug)]
pub struct TypeDef {
    /// The type's `dialect.name`.
    pub name: QualifiedName,
    /// Calls the wrapper's [`VerifyType`]
    /// impl for types with a verifier.
    pub verifier: Option<TypeVerifier>,
    /// Calls the wrapper's [`TypeAsmFormat`]
    /// impl for types with a custom assembly format.
    pub asm_format: Option<TypeAsmHooks>,
    /// Calls the wrapper's [`TypeAliasHint`]
    /// impl for types that suggest printer alias names.
    pub alias_hint: Option<TypeAliasHook>,
    /// Calls the wrapper's [`IdentifiedType`] impl for types whose instances
    /// may be identified. An identified type registers an assembly format,
    /// which writes its body.
    pub identifier: Option<IdentifierHook>,
}

/// A type verifier, calling the wrapper's [`VerifyType`] impl.
pub type TypeVerifier = fn(&IrContext, TypeRef) -> Result<(), VerificationError>;

/// A type's identifier, calling the wrapper's [`IdentifiedType`] impl.
pub type IdentifierHook = for<'a> fn(&'a IrContext, TypeRef) -> Option<&'a str>;

/// A type whose instances may be identified (`design/ir.md` § "식별 타입"):
/// an instance with an identifier is a key, and the context binds its body
/// once ([`IrContext::bind_type_body`]). An instance without one is an
/// ordinary type.
///
/// `#[dialect]` opts in with `#[identified]`, and a hand-written definition
/// adds it with [`TypeDef::with_identifier`].
#[diagnostic::on_unimplemented(
    message = "`{Self}` is declared with `#[identified]` but does not implement `IdentifiedType`",
    label = "`#[identified]` requires this implementation",
    note = "add `impl IdentifiedType for {Self} {{ fn identifier<'a>(&self, ctx: &'a IrContext) -> Option<&'a str> {{ .. }} }}`"
)]
pub trait IdentifiedType: DialectType {
    /// The identifier this instance's data holds, or `None` when the
    /// instance is not identified. The printer names the alias of an
    /// identified type by it.
    fn identifier<'a>(&self, ctx: &'a IrContext) -> Option<&'a str>;
}

/// `T`'s identifier as a [`TypeDef`] stores it.
pub(crate) fn erased_identifier<T: IdentifiedType>(ctx: &IrContext, ty: TypeRef) -> Option<&str> {
    T::from_type_ref(ctx, ty)?.identifier(ctx)
}

impl TypeDef {
    /// A definition without hooks.
    pub const fn new(name: &'static str) -> Self {
        Self {
            name: QualifiedName::new(name),
            verifier: None,
            asm_format: None,
            alias_hint: None,
            identifier: None,
        }
    }

    /// This definition with `T`'s [`VerifyType`] impl as its verifier.
    pub const fn with_verifier<T: VerifyType>(self) -> Self {
        Self {
            verifier: Some(erased_verifier::<T>),
            ..self
        }
    }

    /// This definition with `T`'s [`TypeAsmFormat`] impl as its assembly
    /// format.
    pub const fn with_asm_format<T: TypeAsmFormat>(self) -> Self {
        Self {
            asm_format: Some(TypeAsmHooks::of::<T>()),
            ..self
        }
    }

    /// This definition with `T`'s [`TypeAliasHint`] impl as its alias name
    /// hint.
    pub const fn with_alias_hint<T: TypeAliasHint>(self) -> Self {
        Self {
            alias_hint: Some(erased_alias_hint::<T>),
            ..self
        }
    }

    /// This definition with `T`'s [`IdentifiedType`] impl as its
    /// identifier. Its assembly format must write the body.
    pub const fn with_identifier<T: IdentifiedType>(self) -> Self {
        Self {
            identifier: Some(erased_identifier::<T>),
            ..self
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

/// The identifier of `ty`, if its definition reads one from it
/// ([`IdentifiedType`]).
pub fn identifier(ctx: &IrContext, ty: TypeRef) -> Option<&str> {
    let identifier = TypeDef::of(ctx, ty)?.identifier?;
    identifier(ctx, ty)
}

/// Whether `ty` is an identified type: it has an [`identifier`].
pub fn is_identified(ctx: &IrContext, ty: TypeRef) -> bool {
    identifier(ctx, ty).is_some()
}

/// Inventory entry registering a [`TypeDef`].
///
/// Emitted by the `#[dialect]` macro for every declared type, and by
/// [`register_type_def!`](crate::register_type_def) for a hand-written one.
#[doc(hidden)]
pub struct TypeDefRegistration(pub &'static TypeDef);

/// Register a hand-written [`TypeDef`], given as a `&'static TypeDef`.
///
/// ```text
/// static FUNC_SIG_DEF: TypeDef = TypeDef::new("func.func_sig").with_verifier::<FuncSig>();
/// register_type_def!(&FUNC_SIG_DEF);
/// ```
///
/// `#[dialect]` registers the definitions of the types it declares.
#[macro_export]
macro_rules! register_type_def {
    ($def:expr) => {
        $crate::__private::inventory::submit! {
            $crate::type_def::TypeDefRegistration($def)
        }
    };
}

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
        struct Checked {
            bad: Option<Attr<bool>>,
        }

        #[asm_format]
        struct Custom;

        struct Tagged<Elem> {
            tag: Attr<String>,
        }
    }

    impl crate::type_verifier::VerifyType for Checked {
        fn verify(ctx: &IrContext, ty: TypeRef) -> Result<Self, VerificationError> {
            match ctx.type_attrs(ty).contains_key("bad") {
                true => Err("bad checked type".into()),
                false => Ok(Self(ty)),
            }
        }
    }

    impl crate::asm_format::TypeAsmFormat for Custom {
        fn print(&self, p: &mut crate::printer::AsmPrinter<'_>) -> std::fmt::Result {
            use std::fmt::Write;
            p.write_str("type_hooks_test.custom<!>")
        }

        fn parse(
            p: &mut crate::parser::AsmParser<'_>,
            name: &str,
        ) -> crate::parser::ParseResult<Option<crate::TypeRef>> {
            p.parse_punct("!")?;
            p.parse_punct(">")?;
            p.build_type(name, Vec::new(), Default::default()).map(Some)
        }
    }

    #[test]
    fn wrappers_require_the_declared_shape() {
        use crate::ops::DialectType;
        use crate::type_constraint::TypeConstraint;

        let mut ctx = IrContext::new();
        let elem = crate::TypeDataBuilder::new(Custom::NAME).intern(&mut ctx);
        let tag = ctx.string_attr("t");
        let tagged = crate::TypeDataBuilder::new(Tagged::NAME)
            .param(elem)
            .attr("tag", tag.clone())
            .intern(&mut ctx);
        assert!(Tagged::from_type_ref(&ctx, tagged).is_some());
        let missing_param = crate::TypeDataBuilder::new(Tagged::NAME)
            .attr("tag", tag)
            .intern(&mut ctx);
        let wrong_kind = crate::TypeDataBuilder::new(Tagged::NAME)
            .param(elem)
            .attr("tag", crate::Attribute::Bool(true))
            .intern(&mut ctx);
        for ty in [missing_param, wrong_kind] {
            assert!(Tagged::from_type_ref(&ctx, ty).is_none());
            assert!(!(<Tagged as TypeConstraint>::DESC.matches)(&ctx, ty));
        }

        assert!(<Custom as TypeConstraint>::DESC.fixed.is_some());
        assert!(<Checked as TypeConstraint>::DESC.fixed.is_none());
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
        let error = verifier(&ctx, bad).unwrap_err();
        assert_eq!(error.to_string(), "bad checked type");
        assert!(Checked::from_type_ref(&ctx, bad).is_none());
        let good = crate::TypeDataBuilder::new(Checked::NAME).intern(&mut ctx);
        assert!(Checked::from_type_ref(&ctx, good).is_some());

        let ty = crate::TypeDataBuilder::new(Custom::NAME).intern(&mut ctx);
        assert_eq!(ty.display(&ctx).to_string(), "type_hooks_test.custom<!>");
        let parsed = crate::parser::parse_type(&mut ctx, "type_hooks_test.custom<!>").unwrap();
        assert_eq!(parsed, ty);
    }

    #[test]
    fn builders_intern_the_declared_shape() {
        use crate::ops::DialectType;

        let mut ctx = IrContext::new();
        let elem = Custom::params().intern(&mut ctx).as_type_ref();
        let marked =
            crate::AttributeMap::from_entries(&mut ctx, [("k", crate::Attribute::Bool(true))]);
        let tagged = Tagged::params(elem)
            .tag("t")
            .param_attrs([marked])
            .intern(&mut ctx);
        assert_eq!(
            Tagged::from_type_ref(&ctx, tagged.as_type_ref()),
            Some(tagged)
        );
        assert_eq!(tagged.elem(&ctx), elem);
        assert_eq!(tagged.tag(&ctx), "t");
        assert_eq!(
            ctx.get_type(tagged.as_type_ref())
                .param_attrs(&ctx, 0)
                .view(&ctx)
                .get("k"),
            Some(&crate::Attribute::Bool(true))
        );

        let good = Checked::params().intern(&mut ctx).unwrap();
        assert_eq!(good.bad(&ctx), None);
        let error = Checked::params().bad(true).intern(&mut ctx).unwrap_err();
        assert_eq!(error.to_string(), "bad checked type");
    }

    #[test]
    #[should_panic(expected = "type_hooks_test.tagged: missing attribute `tag`")]
    fn builders_require_their_attributes() {
        let mut ctx = IrContext::new();
        let elem = Custom::params().intern(&mut ctx).as_type_ref();
        let _ = Tagged::params(elem).intern(&mut ctx);
    }

    /// A hand-written definition with every hook.
    #[derive(Clone, Copy)]
    struct HandWritten(TypeRef);

    static HAND_WRITTEN_DEF: TypeDef = TypeDef::new("type_hooks_test.hand_written")
        .with_verifier::<HandWritten>()
        .with_asm_format::<Custom>()
        .with_alias_hint::<HandWritten>();

    crate::register_type_def!(&HAND_WRITTEN_DEF);

    impl crate::ops::DialectType for HandWritten {
        const DEF: &'static TypeDef = &HAND_WRITTEN_DEF;

        fn from_type_ref(ctx: &IrContext, ty: TypeRef) -> Option<Self> {
            Self::matches(ctx, ty).then_some(Self(ty))
        }

        fn as_type_ref(&self) -> TypeRef {
            self.0
        }
    }

    impl crate::type_verifier::VerifyType for HandWritten {
        fn verify(_ctx: &IrContext, ty: TypeRef) -> Result<Self, VerificationError> {
            Ok(Self(ty))
        }
    }

    impl crate::asm_format::TypeAliasHint for HandWritten {
        fn alias_hint<'a>(&self, _ctx: &'a IrContext) -> Option<&'a str> {
            Some("handy")
        }
    }

    #[test]
    fn hand_written_definitions_register_their_hooks() {
        let def = TypeDef::lookup("type_hooks_test.hand_written").unwrap();
        assert!(std::ptr::eq(def, &HAND_WRITTEN_DEF));
        assert!(def.verifier.is_some() && def.asm_format.is_some());

        let mut ctx = IrContext::new();
        let ty = crate::TypeDataBuilder::new("type_hooks_test.hand_written").intern(&mut ctx);
        assert!(ctx.get_type(ty).name.is_registered());
        assert_eq!(
            crate::asm_format::suggest_type_alias_name(&ctx, ty),
            Some("handy")
        );
        let verifier = def.verifier.unwrap();
        assert!(verifier(&ctx, ty).is_ok());
        let wrapper = <HandWritten as crate::ops::DialectType>::from_type_ref(&ctx, ty).unwrap();
        assert_eq!(crate::ops::DialectType::as_type_ref(&wrapper), ty);

        // The builders also run outside constant evaluation.
        let built = TypeDef::new("type_hooks_test.unregistered").with_alias_hint::<HandWritten>();
        assert!(built.verifier.is_none() && built.asm_format.is_none());
        assert_eq!((built.alias_hint.unwrap())(&ctx, ty), Some("handy"));
    }

    #[test]
    fn every_identified_type_registers_an_assembly_format() {
        // Generic syntax cannot write a body.
        for TypeDefRegistration(def) in inventory::iter::<TypeDefRegistration> {
            assert!(
                def.identifier.is_none() || def.asm_format.is_some(),
                "{} is identified without an assembly format",
                def.name.as_str()
            );
        }
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

#[cfg(test)]
mod identified_tests {
    use std::fmt::Write as _;

    use crate::asm_format::TypeAsmFormat;
    use crate::ops::DialectType;
    use crate::parser::{AsmParser, ParseResult, lexer, parse_module, parse_test_module};
    use crate::printer::AsmPrinter;
    use crate::types::Attribute;
    use crate::validation::validate_type_bodies;
    use crate::{IrContext, TypeBodyError, TypeDataBuilder, TypeRef};

    use super::{IdentifiedType, TypeDef, is_identified};

    /// `test.ident<"Key">`, or `test.ident<"Key", BODY>` where it writes its
    /// body: an identified type whose key is its `key` attribute.
    #[derive(Clone, Copy)]
    struct Ident(TypeRef);

    static IDENT_DEF: TypeDef = TypeDef::new("test.ident")
        .with_asm_format::<Ident>()
        .with_identifier::<Ident>();

    crate::register_type_def!(&IDENT_DEF);

    impl Ident {
        fn key(ctx: &mut IrContext, key: &str) -> TypeRef {
            let key = ctx.string_attr(key);
            TypeDataBuilder::new("test.ident")
                .attr("key", key)
                .intern(ctx)
        }

        fn key_text(self, ctx: &IrContext) -> Option<&str> {
            ctx.type_attrs(self.0).get_str("key")
        }
    }

    impl DialectType for Ident {
        const DEF: &'static TypeDef = &IDENT_DEF;

        fn from_type_ref(ctx: &IrContext, ty: TypeRef) -> Option<Self> {
            Self::matches(ctx, ty).then_some(Self(ty))
        }

        fn as_type_ref(&self) -> TypeRef {
            self.0
        }
    }

    impl IdentifiedType for Ident {
        fn identifier<'a>(&self, ctx: &'a IrContext) -> Option<&'a str> {
            self.key_text(ctx)
        }
    }

    impl TypeAsmFormat for Ident {
        fn print(&self, p: &mut AsmPrinter<'_>) -> std::fmt::Result {
            let key = self.key_text(p.ctx()).ok_or(std::fmt::Error)?;
            write!(p, "test.ident<{key:?}")?;
            if let Some(body) = p.identified_body(self.0) {
                p.write_str(", ")?;
                p.print_type(body.ty())?;
            }
            p.write_char('>')
        }

        fn parse(p: &mut AsmParser<'_>, _name: &str) -> ParseResult<Option<TypeRef>> {
            let key = p.parse_string()?;
            let ty = Ident::key(p.ctx_mut(), &key);
            if p.parse_optional_punct(",") {
                let offset = p.offset();
                let body = p.parse_type()?;
                p.bind_type_body(ty, body, offset)?;
            }
            p.parse_punct(">")?;
            Ok(Some(ty))
        }
    }

    const LIST: &str = r#"core.module @test {
  !List = test.ident<"List", test.cell<core.i32, test.ident<"List">>>

  func.func @f(%0: !List) -> !List {
      func.return %0
  }
}
"#;

    #[test]
    fn a_key_and_a_written_body_name_one_type_with_that_body() {
        let mut ctx = IrContext::new();
        parse_test_module(&mut ctx, LIST);
        let list = Ident::key(&mut ctx, "List");
        assert_eq!(ctx.type_alias_by_text("List"), Some(list));
        let body = ctx.type_body(list).expect("the alias line binds the body");
        // The body refers to the identified type itself, by its key.
        assert_eq!(ctx.get_type(body).params[1], list);
    }

    #[test]
    fn a_body_is_bound_once() {
        let mut ctx = IrContext::new();
        let list = Ident::key(&mut ctx, "List");
        let i32_ty = TypeDataBuilder::new("core.i32").intern(&mut ctx);
        let i64_ty = TypeDataBuilder::new("core.i64").intern(&mut ctx);
        assert_eq!(ctx.bind_type_body(list, i32_ty), Ok(()));
        assert_eq!(ctx.bind_type_body(list, i32_ty), Ok(()));
        assert_eq!(
            ctx.bind_type_body(list, i64_ty),
            Err(TypeBodyError::Conflicting { bound: i32_ty })
        );
        assert_eq!(ctx.type_body(list), Some(i32_ty));
        // Only an identified type takes a body.
        assert_eq!(
            ctx.bind_type_body(i32_ty, i64_ty),
            Err(TypeBodyError::NotIdentified)
        );
    }

    #[test]
    fn a_differing_body_is_a_parse_error() {
        let mut ctx = IrContext::new();
        let error = parse_module(
            &mut ctx,
            r#"core.module @test {
  !A = test.ident<"A", core.i32>
  !B = test.ident<"A", core.i64>
}"#,
        )
        .expect_err("a second body for one key");
        assert!(error.message.contains("differs"), "{}", error.message);
    }

    #[test]
    fn identified_types_round_trip_with_their_bodies_on_alias_lines() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, LIST);
        assert_eq!(module.op().display(&ctx).to_string(), LIST);

        // Mutual recursion: each body writes the other type as its key, so
        // the alias lines need no particular order.
        let text = r#"core.module @test {
  !A = test.ident<"A", test.cell<test.ident<"B">>>
  !B = test.ident<"B", test.cell<test.ident<"A">>>

  func.func @f(%0: !A, %1: !B) {
      func.return
  }
}
"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, text);
        assert_eq!(module.op().display(&ctx).to_string(), text);
    }

    #[test]
    fn the_printer_aliases_every_identified_type_a_module_reaches() {
        // Built without aliases: `Node` is reached only through the body of
        // `Tree`, and still gets the alias line that writes its body.
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @f(%0: core.i32) {
      func.return
  }
}"#,
        );
        let tree = Ident::key(&mut ctx, "Tree");
        let node = Ident::key(&mut ctx, "Node");
        let i32_ty = TypeDataBuilder::new("core.i32").intern(&mut ctx);
        let node_body = TypeDataBuilder::new("test.cell")
            .param(i32_ty)
            .param(tree)
            .intern(&mut ctx);
        let tree_body = TypeDataBuilder::new("test.cell")
            .param(node)
            .intern(&mut ctx);
        ctx.bind_type_body(node, node_body).unwrap();
        ctx.bind_type_body(tree, tree_body).unwrap();
        let entry = ctx.region(module.body(&ctx).unwrap()).blocks[0];
        let func = ctx.block(entry).ops[0];
        ctx.op_attrs_mut(func).insert("tree", Attribute::Type(tree));

        let printed = module.op().display(&ctx).to_string();
        assert!(
            printed
                .contains(r#"!Node = test.ident<"Node", test.cell<core.i32, test.ident<"Tree">>>"#),
            "{printed}"
        );
        assert!(
            printed.contains(r#"!Tree = test.ident<"Tree", test.cell<test.ident<"Node">>>"#),
            "{printed}"
        );
        // The printed text binds both bodies again.
        let mut reparsed = IrContext::new();
        let reparsed_module = parse_test_module(&mut reparsed, &printed);
        assert!(validate_type_bodies(&reparsed, reparsed_module).is_ok());
        let reprinted = reparsed_module.op().display(&reparsed).to_string();
        let mut again = IrContext::new();
        let again_module = parse_test_module(&mut again, &reprinted);
        assert_eq!(again_module.op().display(&again).to_string(), reprinted);

        // Without aliases, the outermost occurrence writes the body and the
        // types inside it are keys.
        assert_eq!(
            tree.display(&ctx).to_string(),
            r#"test.ident<"Tree", test.cell<test.ident<"Node">>>"#
        );
    }

    #[test]
    fn a_type_reached_only_through_an_alias_definition_keeps_its_body() {
        // `B` appears only inside the body of the context alias `A`, where it
        // is written as its key, so it needs an alias line of its own.
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !A = test.ident<"A", test.cell<test.ident<"B">>>
}"#,
        );
        let b = Ident::key(&mut ctx, "B");
        let i32_ty = TypeDataBuilder::new("core.i32").intern(&mut ctx);
        ctx.bind_type_body(b, i32_ty).unwrap();

        let printed = module.op().display(&ctx).to_string();
        assert!(
            printed.contains(r#"!B = test.ident<"B", core.i32>"#),
            "{printed}"
        );
        let mut reparsed = IrContext::new();
        let reparsed_module = parse_test_module(&mut reparsed, &printed);
        let b = Ident::key(&mut reparsed, "B");
        assert!(reparsed.type_body(b).is_some(), "{printed}");
        assert!(validate_type_bodies(&reparsed, reparsed_module).is_ok());
    }

    #[test]
    fn a_generated_alias_in_an_identified_body_round_trips() {
        // The body of `A` is a generated alias that writes `A` again, so the
        // two definitions depend on each other.
        let cell = r#"test.cell<test.ident<"A">, core.i64, core.i64, core.i64>"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            &format!(
                "core.module @test {{\n  func.func @f(%0: {cell}, %1: {cell}) {{\n      func.return\n  }}\n}}"
            ),
        );
        let a = Ident::key(&mut ctx, "A");
        let entry = ctx.region(module.body(&ctx).unwrap()).blocks[0];
        let func = ctx.block(entry).ops[0];
        let body = ctx
            .get_type(ctx.op_attrs(func).get_type("type").unwrap())
            .params[0];
        ctx.bind_type_body(a, body).unwrap();

        let printed = module.op().display(&ctx).to_string();
        assert!(printed.contains("!t0 = "), "{printed}");
        // Reparsed, the definitions are the context's aliases, whose cycle
        // the printer breaks the same way.
        let mut reparsed = IrContext::new();
        let reparsed_module = parse_test_module(&mut reparsed, &printed);
        assert!(validate_type_bodies(&reparsed, reparsed_module).is_ok());
        let reprinted = reparsed_module.op().display(&reparsed).to_string();
        let mut again = IrContext::new();
        let again_module = parse_test_module(&mut again, &reprinted);
        assert_eq!(again_module.op().display(&again).to_string(), reprinted);
    }

    #[test]
    fn a_module_must_bind_every_identified_type_it_reaches() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @f(%0: test.cell<test.ident<"Orphan">>) {
      func.return
  }
}"#,
        );
        let result = validate_type_bodies(&ctx, module);
        assert_eq!(result.errors.len(), 1, "{result}");
        assert!(
            result
                .to_string()
                .contains(r#"identified type "Orphan" has no body"#),
            "{result}"
        );

        let orphan = Ident::key(&mut ctx, "Orphan");
        let i32_ty = TypeDataBuilder::new("core.i32").intern(&mut ctx);
        ctx.bind_type_body(orphan, i32_ty).unwrap();
        assert!(validate_type_bodies(&ctx, module).is_ok());
    }

    #[test]
    fn attribute_types_reach_identified_types() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @f() attributes {t = test.ident<"Loose">} {
      func.return
  }
}"#,
        );
        assert!(!validate_type_bodies(&ctx, module).is_ok());
    }

    #[crate::dialect]
    mod layout_test {
        /// `layout_test.layout<"Name">` and `layout_test.layout<"Name", BODY>`
        /// name an identified layout; `layout_test.layout<(A, B)>` is a
        /// literal one.
        #[identified]
        #[asm_format]
        struct Layout<#[rest] Fields> {
            name: Option<Attr<String>>,
        }
    }

    impl IdentifiedType for Layout {
        fn identifier<'a>(&self, ctx: &'a IrContext) -> Option<&'a str> {
            self.name(ctx)
        }
    }

    impl TypeAsmFormat for Layout {
        fn print(&self, p: &mut AsmPrinter<'_>) -> std::fmt::Result {
            let ty = self.as_type_ref();
            p.write_str("layout_test.layout<")?;
            if let Some(name) = self.name(p.ctx()) {
                write!(p, "{name:?}")?;
                if let Some(body) = p.identified_body(ty) {
                    p.write_str(", ")?;
                    p.print_type(body.ty())?;
                }
            } else {
                p.write_char('(')?;
                for (i, &field) in p.ctx().get_type(ty).params.iter().enumerate() {
                    if i > 0 {
                        p.write_str(", ")?;
                    }
                    p.print_type(field)?;
                }
                p.write_char(')')?;
            }
            p.write_char('>')
        }

        fn parse(p: &mut AsmParser<'_>, _name: &str) -> ParseResult<Option<TypeRef>> {
            let ty = if let Some(name) = p.parse_optional_token("a string", lexer::string_lit)? {
                let name = p.ctx_mut().string_attr(&name);
                let ty = TypeDataBuilder::new("layout_test.layout")
                    .attr("name", name)
                    .intern(p.ctx_mut());
                if p.parse_optional_punct(",") {
                    let offset = p.offset();
                    let body = p.parse_type()?;
                    p.bind_type_body(ty, body, offset)?;
                }
                ty
            } else {
                let fields = p.parse_delimited("(", ")", |p| p.parse_type())?;
                TypeDataBuilder::new("layout_test.layout")
                    .params(fields)
                    .intern(p.ctx_mut())
            };
            p.parse_punct(">")?;
            Ok(Some(ty))
        }
    }

    #[test]
    fn instances_of_one_definition_are_identified_by_their_identifier() {
        let text = r#"core.module @test {
  !Pair = layout_test.layout<"Pair", layout_test.layout<(core.i32, core.i64)>>

  func.func @f(%0: !Pair, %1: layout_test.layout<(!Pair)>) {
      func.return
  }
}
"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, text);
        // The literal instance gets no alias; the identified one is aliased
        // by its identifier, with its body on the alias line.
        assert_eq!(module.op().display(&ctx).to_string(), text);
        assert!(validate_type_bodies(&ctx, module).is_ok());

        let pair = ctx.type_alias_by_text("Pair").unwrap();
        assert!(is_identified(&ctx, pair));
        let body = ctx.type_body(pair).unwrap();
        assert!(Layout::from_type_ref(&ctx, body).is_some());
        assert!(!is_identified(&ctx, body));
        assert_eq!(
            ctx.bind_type_body(body, pair),
            Err(TypeBodyError::NotIdentified)
        );
    }
}
