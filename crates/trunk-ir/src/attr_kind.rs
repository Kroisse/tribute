//! Attribute kinds for declarative operation and type schemas.
//!
//! `Attr<K>` in a `#[dialect]` declaration names a Rust type `K` implementing
//! [`AttrKind`]. The kind defines the schema domain of the attribute, the
//! value its accessor returns, the value its builder setter takes, and how a
//! declarative assembly format spells its values; the macro emits
//! `<K as AttrKind>` without interpreting `K`.

use std::fmt;
use std::marker::PhantomData;

use smallvec::SmallVec;

use crate::context::IrContext;
use crate::op_schema::AttributeKind;
use crate::parser::{AsmParser, ParseResult};
use crate::printer::AsmPrinter;
use crate::refs::TypeRef;
use crate::symbol::SymbolPath;
use crate::types::{Attribute, AttributeIter, AttributeKey, AttributeMap, StringArg, StringRef};

/// The value domain and typed access of a declared attribute.
pub trait AttrKind {
    /// The domain the schema checks.
    const KIND: AttributeKind;
    /// What the generated accessor returns.
    type Out<'ctx>;
    /// What the generated builder setter stores until `build`.
    type In;

    /// Read an attribute of this kind. Panics if `attr` is outside the
    /// kind's domain, which the schema rejects.
    fn read<'ctx>(ctx: &'ctx IrContext, attr: &'ctx Attribute) -> Self::Out<'ctx>;

    fn write(ctx: &mut IrContext, value: Self::In) -> Attribute;

    /// The values `accepts` admits, as a schema violation names them, when
    /// they are narrower than `KIND`'s domain.
    const DOMAIN: Option<&'static str> = None;

    /// Whether `attr` is a value of this kind, which the schema checks: by
    /// default, whether it is in `KIND`'s domain. A narrower kind, such as a
    /// keyword, also checks the rest and names it in `DOMAIN`.
    fn accepts(ctx: &IrContext, attr: &Attribute) -> bool {
        let _ = ctx;
        Self::KIND.accepts(attr)
    }

    /// Write `attr`, a value `accepts` admits, as a declarative assembly
    /// format spells it: by default in generic attribute syntax. A keyword
    /// kind writes its keyword (`design/ir.md`, "선언적 어셈블리 형식").
    fn print_text(p: &mut AsmPrinter<'_>, attr: &Attribute) -> fmt::Result {
        p.print_attribute(attr)
    }

    /// Read a value as [`Self::print_text`] writes it, as an unverified
    /// attribute that the schema checks later.
    fn parse_text(p: &mut AsmParser<'_>) -> ParseResult<Attribute> {
        p.parse_attribute()
    }

    /// Whether two values read from attributes are the same value, which
    /// tells a stored `#[default(v)]` from another value. Floating-point
    /// kinds compare bits, so `-0.0` is not the default `0.0`.
    fn same<'ctx>(a: &Self::Out<'ctx>, b: &Self::Out<'ctx>) -> bool
    where
        Self::Out<'ctx>: PartialEq,
    {
        a == b
    }
}

/// A kind whose value is a pooled handle, read without borrowing the context.
pub trait AttrHandle: AttrKind {
    type Handle<'a>;

    fn handle(attr: &Attribute) -> Self::Handle<'_>;
}

/// `Attr<_>`: any attribute value.
impl AttrKind for Attribute {
    const KIND: AttributeKind = AttributeKind::Any;
    type Out<'ctx> = Attribute;
    type In = Attribute;

    fn read<'ctx>(_: &'ctx IrContext, attr: &'ctx Attribute) -> Attribute {
        attr.clone()
    }

    fn write(_: &mut IrContext, value: Attribute) -> Attribute {
        value
    }
}

impl AttrKind for bool {
    const KIND: AttributeKind = AttributeKind::Bool;
    type Out<'ctx> = bool;
    type In = bool;

    fn read<'ctx>(_: &'ctx IrContext, attr: &'ctx Attribute) -> bool {
        match attr {
            Attribute::Bool(value) => *value,
            _ => panic!("expected Bool attribute"),
        }
    }

    fn write(_: &mut IrContext, value: bool) -> Attribute {
        Attribute::Bool(value)
    }
}

macro_rules! int_kind {
    ($($ty:ident => $kind:ident),* $(,)?) => {$(
        impl AttrKind for $ty {
            const KIND: AttributeKind = AttributeKind::$kind;
            type Out<'ctx> = $ty;
            type In = $ty;

            fn read<'ctx>(_: &'ctx IrContext, attr: &'ctx Attribute) -> $ty {
                match attr {
                    Attribute::Int(value) => $ty::try_from(*value).expect(concat!(
                        "Int attribute is out of range for ",
                        stringify!($ty)
                    )),
                    _ => panic!("expected Int attribute"),
                }
            }

            fn write(_: &mut IrContext, value: $ty) -> Attribute {
                Attribute::Int(i128::from(value))
            }
        }
    )*};
}

int_kind!(i32 => I32, i64 => I64, u32 => U32, u64 => U64);

/// Declare a keyword attribute kind: an enum stored as a string attribute
/// holding its keyword, which a declarative format writes bare, as MLIR
/// writes `arith.cmpi slt, %a, %b`.
macro_rules! keyword_kind {
    (
        $(#[$meta:meta])*
        $vis:vis enum $name:ident {
            $($(#[$variant_meta:meta])* $variant:ident = $word:literal),+ $(,)?
        }
    ) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        $vis enum $name {
            $($(#[$variant_meta])* $variant),+
        }

        impl $name {
            /// The keyword that spells this value.
            pub fn keyword(self) -> &'static str {
                match self {
                    $(Self::$variant => $word),+
                }
            }

            /// The value a keyword spells; `None` for any other word.
            pub fn from_keyword(word: &str) -> Option<Self> {
                match word {
                    $($word => Some(Self::$variant),)+
                    _ => None,
                }
            }

            fn from_attribute(
                ctx: &$crate::context::IrContext,
                attr: &$crate::types::Attribute,
            ) -> Option<Self> {
                match attr {
                    $crate::types::Attribute::String(word) => Self::from_keyword(ctx.str(*word)),
                    _ => None,
                }
            }
        }

        impl ::std::fmt::Display for $name {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                f.write_str(self.keyword())
            }
        }

        impl $crate::attr_kind::AttrKind for $name {
            const KIND: $crate::op_schema::AttributeKind = $crate::op_schema::AttributeKind::String;
            type Out<'ctx> = $name;
            type In = $name;
            const DOMAIN: Option<&'static str> = Some(concat!("one of:" $(, " ", $word)+));

            fn accepts(ctx: &$crate::context::IrContext, attr: &$crate::types::Attribute) -> bool {
                Self::from_attribute(ctx, attr).is_some()
            }

            fn read<'ctx>(
                ctx: &'ctx $crate::context::IrContext,
                attr: &'ctx $crate::types::Attribute,
            ) -> $name {
                Self::from_attribute(ctx, attr)
                    .expect(concat!("unverified ", stringify!($name), " keyword"))
            }

            fn write(ctx: &mut $crate::context::IrContext, value: $name) -> $crate::types::Attribute {
                ctx.string_attr(value.keyword())
            }

            fn print_text(
                p: &mut $crate::printer::AsmPrinter<'_>,
                attr: &$crate::types::Attribute,
            ) -> ::std::fmt::Result {
                ::std::fmt::Write::write_str(p, Self::read(p.ctx(), attr).keyword())
            }

            fn parse_text(
                p: &mut $crate::parser::AsmParser<'_>,
            ) -> $crate::parser::ParseResult<$crate::types::Attribute> {
                $crate::asm_format::parse_keyword(p, |ctx, word| {
                    Self::from_keyword(word).map(|value| ctx.string_attr(value.keyword()))
                })
            }
        }
    };
}

pub(crate) use keyword_kind;

impl AttrKind for f32 {
    const KIND: AttributeKind = AttributeKind::F32;
    type Out<'ctx> = f32;
    type In = f32;

    fn read<'ctx>(_: &'ctx IrContext, attr: &'ctx Attribute) -> f32 {
        match attr {
            Attribute::FloatBits(bits) => f64::from_bits(*bits) as f32,
            _ => panic!("expected FloatBits attribute"),
        }
    }

    fn write(_: &mut IrContext, value: f32) -> Attribute {
        Attribute::FloatBits(f64::from(value).to_bits())
    }

    fn same<'ctx>(a: &f32, b: &f32) -> bool
    where
        Self::Out<'ctx>: PartialEq,
    {
        a.to_bits() == b.to_bits()
    }
}

impl AttrKind for f64 {
    const KIND: AttributeKind = AttributeKind::F64;
    type Out<'ctx> = f64;
    type In = f64;

    fn read<'ctx>(_: &'ctx IrContext, attr: &'ctx Attribute) -> f64 {
        match attr {
            Attribute::FloatBits(bits) => f64::from_bits(*bits),
            _ => panic!("expected FloatBits attribute"),
        }
    }

    fn write(_: &mut IrContext, value: f64) -> Attribute {
        Attribute::FloatBits(value.to_bits())
    }

    fn same<'ctx>(a: &f64, b: &f64) -> bool
    where
        Self::Out<'ctx>: PartialEq,
    {
        a.to_bits() == b.to_bits()
    }
}

/// A type attribute.
pub struct Type;

impl AttrKind for Type {
    const KIND: AttributeKind = AttributeKind::Type;
    type Out<'ctx> = TypeRef;
    type In = TypeRef;

    fn read<'ctx>(_: &'ctx IrContext, attr: &'ctx Attribute) -> TypeRef {
        match attr {
            Attribute::Type(ty) => *ty,
            _ => panic!("expected Type attribute"),
        }
    }

    fn write(_: &mut IrContext, value: TypeRef) -> Attribute {
        Attribute::Type(value)
    }
}

/// A string attribute. The accessor borrows the text from the context's
/// string pool, and the setter takes a handle or text to intern.
impl AttrKind for String {
    const KIND: AttributeKind = AttributeKind::String;
    type Out<'ctx> = &'ctx str;
    type In = StringArg;

    fn read<'ctx>(ctx: &'ctx IrContext, attr: &'ctx Attribute) -> &'ctx str {
        ctx.str(Self::handle(attr))
    }

    fn write(ctx: &mut IrContext, value: StringArg) -> Attribute {
        Attribute::String(ctx.intern_string_arg(value))
    }
}

impl AttrHandle for String {
    type Handle<'a> = StringRef;

    fn handle(attr: &Attribute) -> StringRef {
        match attr {
            Attribute::String(value) => *value,
            _ => panic!("expected String attribute"),
        }
    }
}

/// A reference to a symbol table definition, by its qualified name.
pub struct SymbolRef;

impl AttrKind for SymbolRef {
    const KIND: AttributeKind = AttributeKind::SymbolRef;
    type Out<'ctx> = &'ctx SymbolPath;
    type In = SymbolPath;

    fn read<'ctx>(_: &'ctx IrContext, attr: &'ctx Attribute) -> &'ctx SymbolPath {
        match attr {
            Attribute::SymbolRef(symbol) => symbol,
            _ => panic!("expected SymbolRef attribute"),
        }
    }

    fn write(_: &mut IrContext, value: SymbolPath) -> Attribute {
        Attribute::SymbolRef(value)
    }

    /// Only `@` begins a symbol reference, so another value is not read as one.
    fn parse_text(p: &mut AsmParser<'_>) -> ParseResult<Attribute> {
        if !p.peek().starts_with('@') {
            return Err(p.expected("`@name`"));
        }
        p.parse_attribute()
    }
}

/// A byte string attribute.
pub struct Bytes;

impl AttrKind for Bytes {
    const KIND: AttributeKind = AttributeKind::Bytes;
    type Out<'ctx> = SmallVec<[u8; 16]>;
    type In = SmallVec<[u8; 16]>;

    fn read<'ctx>(_: &'ctx IrContext, attr: &'ctx Attribute) -> SmallVec<[u8; 16]> {
        match attr {
            Attribute::Bytes(bytes) => bytes.clone(),
            _ => panic!("expected Bytes attribute"),
        }
    }

    fn write(_: &mut IrContext, value: SmallVec<[u8; 16]>) -> Attribute {
        Attribute::Bytes(value)
    }
}

fn list_items(attr: &Attribute) -> &[Attribute] {
    match attr {
        Attribute::List(items) => items,
        _ => panic!("expected List attribute"),
    }
}

/// `Attr<[K]>`: a list whose every element has kind `K`.
impl<K: AttrKind> AttrKind for [K] {
    const KIND: AttributeKind = AttributeKind::List(&K::KIND);
    type Out<'ctx> = ListIter<'ctx, K>;
    type In = Vec<K::In>;
    const DOMAIN: Option<&'static str> = K::DOMAIN;

    fn accepts(ctx: &IrContext, attr: &Attribute) -> bool {
        match attr {
            Attribute::List(items) => items.iter().all(|item| K::accepts(ctx, item)),
            _ => false,
        }
    }

    fn read<'ctx>(ctx: &'ctx IrContext, attr: &'ctx Attribute) -> ListIter<'ctx, K> {
        ListIter {
            ctx,
            items: list_items(attr).iter(),
            kind: PhantomData,
        }
    }

    fn write(ctx: &mut IrContext, value: Vec<K::In>) -> Attribute {
        Attribute::List(
            value
                .into_iter()
                .map(|element| K::write(ctx, element))
                .collect(),
        )
    }
}

impl<K: AttrHandle> AttrHandle for [K] {
    type Handle<'a> = HandleIter<'a, K>;

    fn handle(attr: &Attribute) -> HandleIter<'_, K> {
        HandleIter {
            items: list_items(attr).iter(),
            kind: PhantomData,
        }
    }
}

/// `Attr<Dict<V>>`: a dictionary whose every value has kind `V`.
pub struct Dict<V: ?Sized>(PhantomData<fn() -> V>);

impl<V: AttrKind + ?Sized> AttrKind for Dict<V> {
    const KIND: AttributeKind = AttributeKind::Dict(&V::KIND);
    type Out<'ctx> = DictView<'ctx, V>;
    type In = Vec<(StringArg, V::In)>;
    const DOMAIN: Option<&'static str> = V::DOMAIN;

    fn accepts(ctx: &IrContext, attr: &Attribute) -> bool {
        match attr {
            Attribute::Dict(entries) => entries.values().all(|value| V::accepts(ctx, value)),
            _ => false,
        }
    }

    fn read<'ctx>(ctx: &'ctx IrContext, attr: &'ctx Attribute) -> DictView<'ctx, V> {
        match attr {
            Attribute::Dict(entries) => DictView {
                ctx,
                entries,
                kind: PhantomData,
            },
            _ => panic!("expected Dict attribute"),
        }
    }

    fn write(ctx: &mut IrContext, value: Vec<(StringArg, V::In)>) -> Attribute {
        let mut entries = AttributeMap::new();
        for (key, entry) in value {
            let entry = V::write(ctx, entry);
            entries.view_mut(ctx).insert(key, entry);
        }
        Attribute::Dict(entries)
    }
}

/// The entries of a dictionary attribute, read as kind `V`.
pub struct DictView<'ctx, V: ?Sized> {
    ctx: &'ctx IrContext,
    entries: &'ctx AttributeMap,
    kind: PhantomData<fn() -> V>,
}

impl<'ctx, V: AttrKind + ?Sized> DictView<'ctx, V> {
    pub fn get(&self, key: impl AttributeKey) -> Option<V::Out<'ctx>> {
        self.entries
            .view(self.ctx)
            .get(key)
            .map(|attr| V::read(self.ctx, attr))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The entries in key order.
    pub fn iter(&self) -> DictIter<'ctx, V> {
        DictIter {
            ctx: self.ctx,
            entries: self.entries.iter(),
            kind: PhantomData,
        }
    }
}

impl<'ctx, V: AttrKind + ?Sized> IntoIterator for DictView<'ctx, V> {
    type Item = (StringRef, V::Out<'ctx>);
    type IntoIter = DictIter<'ctx, V>;

    fn into_iter(self) -> DictIter<'ctx, V> {
        self.iter()
    }
}

/// The entries of a dictionary attribute in key order.
pub struct DictIter<'ctx, V: ?Sized> {
    ctx: &'ctx IrContext,
    entries: AttributeIter<'ctx>,
    kind: PhantomData<fn() -> V>,
}

impl<'ctx, V: AttrKind + ?Sized> Iterator for DictIter<'ctx, V> {
    type Item = (StringRef, V::Out<'ctx>);

    fn next(&mut self) -> Option<Self::Item> {
        self.entries
            .next()
            .map(|(key, attr)| (key, V::read(self.ctx, attr)))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.entries.size_hint()
    }
}

/// The elements of a list attribute, read as kind `K`.
pub struct ListIter<'ctx, K> {
    ctx: &'ctx IrContext,
    items: std::slice::Iter<'ctx, Attribute>,
    kind: PhantomData<fn() -> K>,
}

impl<'ctx, K: AttrKind> Iterator for ListIter<'ctx, K> {
    type Item = K::Out<'ctx>;

    fn next(&mut self) -> Option<Self::Item> {
        self.items.next().map(|attr| K::read(self.ctx, attr))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.items.size_hint()
    }
}

impl<K: AttrKind> ExactSizeIterator for ListIter<'_, K> {}

/// The handles of a list attribute's elements.
pub struct HandleIter<'a, K> {
    items: std::slice::Iter<'a, Attribute>,
    kind: PhantomData<fn() -> K>,
}

impl<'a, K: AttrHandle> Iterator for HandleIter<'a, K> {
    type Item = K::Handle<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.items.next().map(K::handle)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.items.size_hint()
    }
}

impl<K: AttrHandle> ExactSizeIterator for HandleIter<'_, K> {}
