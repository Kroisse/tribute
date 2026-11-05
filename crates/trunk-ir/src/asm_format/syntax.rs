//! Syntax descriptions that print and parse from one definition.
//!
//! A [`Syntax`] value describes a piece of an operation's custom assembly
//! once; the printer, the parser, and the naming pre-pass all read the same
//! description, so the text a directive prints is the text it reads back.
//! Pieces combine as in a parser combinator library: [`lit`] for literal
//! text, [`int`], [`operand`], [`ty`], [`symbol_ref`], [`symbol_name`], and
//! [`keyword_of`] for single tokens, [`successor`] for successors, [`region`] and [`region_with_entry`]
//! for regions, tuples, [`preceded`], [`terminated`], and [`delimited`] for
//! sequences, [`many`], [`separated`], and [`optional`] for repetition, and [`map`] to
//! change a piece's value in both directions.
//!
//! An [`OpAsmDirective`](super::OpAsmDirective) built from one implements
//! [`SyntaxDirective`] and delegates its hooks with [`syntax_directive!`](crate::syntax_directive).

use std::fmt::{self, Write as _};

use crate::parser::{OpAsmParser, ParseResult};
use crate::printer::{AsmNamer, OpAsmPrinter, display_symbol_ref, display_symbol_text};
use crate::refs::{BlockRef, RegionRef, TypeRef, ValueRef};
use crate::types::StringRef;
use crate::{IrContext, SymbolPath};

use super::{CustomEntity, ParsedEntity};

/// A piece of custom assembly that prints a value and reads it back.
pub trait Syntax {
    /// What the piece prints and parses.
    type Value;

    /// Write `value`.
    fn print(&self, p: &mut OpAsmPrinter<'_>, value: &Self::Value) -> fmt::Result;

    /// Read the value. Text that does not start this piece is an error.
    fn parse(&self, p: &mut OpAsmParser<'_>) -> ParseResult<Self::Value>;

    /// Read the value, or `None`, leaving the text as it was, when the text
    /// does not start this piece. By default a piece is always present;
    /// [`many`] and [`separated`] need an item whose first non-[silent]
    /// piece can decline.
    ///
    /// [silent]: Self::is_silent
    fn parse_optional(&self, p: &mut OpAsmParser<'_>) -> ParseResult<Option<Self::Value>> {
        self.parse(p).map(Some)
    }

    /// Whether the piece reads nothing, as whitespace does, so that a
    /// sequence asks the next piece whether it is present.
    fn is_silent(&self) -> bool {
        false
    }

    /// Name the values and blocks [`Self::print`] defines, in the order it
    /// writes them.
    fn name(&self, _namer: &mut AsmNamer<'_>, _value: &Self::Value) {}
}

// =============================================================================
// Tokens
// =============================================================================

/// Literal text, printed as written. Parsing skips its whitespace and reads
/// each run of identifier characters as a keyword and each run of other
/// characters as punctuation; text that is only whitespace reads nothing.
pub fn lit(text: &'static str) -> Lit {
    Lit(text)
}

#[derive(Clone, Copy, Debug)]
pub struct Lit(&'static str);

impl Syntax for Lit {
    type Value = ();

    fn print(&self, p: &mut OpAsmPrinter<'_>, (): &()) -> fmt::Result {
        p.write_str(self.0)
    }

    fn parse(&self, p: &mut OpAsmParser<'_>) -> ParseResult<()> {
        for token in literal_tokens(self.0) {
            p.parse_literal(token)?;
        }
        Ok(())
    }

    fn parse_optional(&self, p: &mut OpAsmParser<'_>) -> ParseResult<Option<()>> {
        let mut tokens = literal_tokens(self.0);
        if let Some(first) = tokens.next()
            && !p.parse_optional_literal(first)
        {
            return Ok(None);
        }
        for token in tokens {
            p.parse_literal(token)?;
        }
        Ok(Some(()))
    }

    fn is_silent(&self) -> bool {
        self.0.trim_ascii().is_empty()
    }
}

/// The tokens of literal text: runs of identifier characters, which read
/// as keywords, and runs of other characters, which read as punctuation,
/// split at whitespace.
fn literal_tokens(text: &str) -> impl Iterator<Item = &str> {
    let is_word = |c: char| c.is_ascii_alphanumeric() || c == '_';
    text.split_ascii_whitespace().flat_map(move |word| {
        let mut rest = word;
        std::iter::from_fn(move || {
            let first = rest.chars().next()?;
            let end = rest
                .find(|c: char| is_word(c) != is_word(first))
                .unwrap_or(rest.len());
            let (token, tail) = rest.split_at(end);
            rest = tail;
            Some(token)
        })
    })
}

/// An integer literal.
pub fn int() -> Int {
    Int
}

#[derive(Clone, Copy, Debug)]
pub struct Int;

impl Syntax for Int {
    type Value = i128;

    fn print(&self, p: &mut OpAsmPrinter<'_>, value: &i128) -> fmt::Result {
        write!(p, "{value}")
    }

    fn parse(&self, p: &mut OpAsmParser<'_>) -> ParseResult<i128> {
        p.parse_integer()
    }

    fn parse_optional(&self, p: &mut OpAsmParser<'_>) -> ParseResult<Option<i128>> {
        p.parse_optional_token("an integer", crate::parser::lexer::integer_lit)
    }
}

/// An operand, `%name`.
pub fn operand() -> Operand {
    Operand
}

#[derive(Clone, Copy, Debug)]
pub struct Operand;

impl Syntax for Operand {
    type Value = ValueRef;

    fn print(&self, p: &mut OpAsmPrinter<'_>, value: &ValueRef) -> fmt::Result {
        let s = p.state();
        write!(p, "{}", s.value(*value))
    }

    fn parse(&self, p: &mut OpAsmParser<'_>) -> ParseResult<ValueRef> {
        p.parse_operand()
    }

    fn parse_optional(&self, p: &mut OpAsmParser<'_>) -> ParseResult<Option<ValueRef>> {
        if p.peek().starts_with('%') {
            p.parse_operand().map(Some)
        } else {
            Ok(None)
        }
    }
}

/// A type.
pub fn ty() -> Ty {
    Ty
}

#[derive(Clone, Copy, Debug)]
pub struct Ty;

impl Syntax for Ty {
    type Value = TypeRef;

    fn print(&self, p: &mut OpAsmPrinter<'_>, value: &TypeRef) -> fmt::Result {
        let s = p.state();
        write!(p, "{}", s.ty(*value))
    }

    fn parse(&self, p: &mut OpAsmParser<'_>) -> ParseResult<TypeRef> {
        p.parse_type()
    }
}

/// A symbol reference, `@outer::@name`.
pub fn symbol_ref() -> SymbolRefSyntax {
    SymbolRefSyntax
}

#[derive(Clone, Copy, Debug)]
pub struct SymbolRefSyntax;

impl Syntax for SymbolRefSyntax {
    type Value = SymbolPath;

    fn print(&self, p: &mut OpAsmPrinter<'_>, value: &SymbolPath) -> fmt::Result {
        write!(p, "{}", display_symbol_ref(value))
    }

    fn parse(&self, p: &mut OpAsmParser<'_>) -> ParseResult<SymbolPath> {
        p.parse_symbol_ref()
    }

    fn parse_optional(&self, p: &mut OpAsmParser<'_>) -> ParseResult<Option<SymbolPath>> {
        if p.peek().starts_with('@') {
            p.parse_symbol_ref().map(Some)
        } else {
            Ok(None)
        }
    }
}

/// A name held as text, written `@name`, such as an operation name.
pub fn symbol_name() -> SymbolName {
    SymbolName
}

#[derive(Clone, Copy, Debug)]
pub struct SymbolName;

impl Syntax for SymbolName {
    type Value = StringRef;

    fn print(&self, p: &mut OpAsmPrinter<'_>, value: &StringRef) -> fmt::Result {
        let ctx = p.ctx();
        write!(p, "{}", display_symbol_text(ctx.str(*value)))
    }

    fn parse(&self, p: &mut OpAsmParser<'_>) -> ParseResult<StringRef> {
        let name = p.parse_symbol_name()?;
        Ok(p.ctx_mut().intern_str(&name))
    }

    fn parse_optional(&self, p: &mut OpAsmParser<'_>) -> ParseResult<Option<StringRef>> {
        if p.peek().starts_with('@') {
            self.parse(p).map(Some)
        } else {
            Ok(None)
        }
    }
}

/// One of a set of keywords, each standing for a value of `T`.
pub fn keyword_of<T>(
    what: &'static str,
    from_keyword: fn(&str) -> Option<T>,
    keyword: fn(&T) -> &'static str,
) -> KeywordOf<T> {
    KeywordOf {
        what,
        from_keyword,
        keyword,
    }
}

#[derive(Clone, Copy, Debug)]
pub struct KeywordOf<T> {
    what: &'static str,
    from_keyword: fn(&str) -> Option<T>,
    keyword: fn(&T) -> &'static str,
}

impl<T> Syntax for KeywordOf<T> {
    type Value = T;

    fn print(&self, p: &mut OpAsmPrinter<'_>, value: &T) -> fmt::Result {
        p.write_str((self.keyword)(value))
    }

    fn parse(&self, p: &mut OpAsmParser<'_>) -> ParseResult<T> {
        match self.parse_optional(p)? {
            Some(value) => Ok(value),
            None => Err(p.expected(self.what)),
        }
    }

    fn parse_optional(&self, p: &mut OpAsmParser<'_>) -> ParseResult<Option<T>> {
        use winnow::Parser as _;

        let from_keyword = self.from_keyword;
        p.parse_optional_token(self.what, |input| {
            crate::parser::lexer::ident
                .verify_map(from_keyword)
                .parse_next(input)
        })
    }
}

/// A successor, `^label`.
pub fn successor() -> Successor {
    Successor
}

#[derive(Clone, Copy, Debug)]
pub struct Successor;

impl Syntax for Successor {
    type Value = BlockRef;

    fn print(&self, p: &mut OpAsmPrinter<'_>, value: &BlockRef) -> fmt::Result {
        let s = p.state();
        write!(p, "{}", s.block(*value))
    }

    fn parse(&self, p: &mut OpAsmParser<'_>) -> ParseResult<BlockRef> {
        p.parse_successor()
    }

    fn parse_optional(&self, p: &mut OpAsmParser<'_>) -> ParseResult<Option<BlockRef>> {
        if p.peek().starts_with('^') {
            p.parse_successor().map(Some)
        } else {
            Ok(None)
        }
    }
}

// =============================================================================
// Regions
// =============================================================================

/// A region in generic form, `{ .. }`, block labels included.
pub fn region() -> Region {
    Region
}

#[derive(Clone, Copy, Debug)]
pub struct Region;

impl Syntax for Region {
    type Value = RegionRef;

    fn print(&self, p: &mut OpAsmPrinter<'_>, value: &RegionRef) -> fmt::Result {
        p.print_region(*value)
    }

    fn parse(&self, p: &mut OpAsmParser<'_>) -> ParseResult<RegionRef> {
        p.parse_region()
    }

    fn parse_optional(&self, p: &mut OpAsmParser<'_>) -> ParseResult<Option<RegionRef>> {
        if p.peek().starts_with('{') {
            p.parse_region().map(Some)
        } else {
            Ok(None)
        }
    }

    fn name(&self, namer: &mut AsmNamer<'_>, value: &RegionRef) {
        namer.region(*value);
    }
}

/// A region whose entry block arguments are written first, as
/// `(%a: A, ..)`, then `middle`, then the region with its entry block
/// elided. The region always has an entry block: a region without one
/// cannot be written this way.
pub fn region_with_entry<M: Syntax>(middle: M) -> RegionWithEntry<M> {
    RegionWithEntry(middle)
}

#[derive(Clone, Copy, Debug)]
pub struct RegionWithEntry<M>(M);

impl<M: Syntax> Syntax for RegionWithEntry<M> {
    type Value = (RegionRef, M::Value);

    fn print(&self, p: &mut OpAsmPrinter<'_>, (region, middle): &Self::Value) -> fmt::Result {
        let s = p.state();
        write!(p, "{}", s.entry_args(*region))?;
        self.0.print(p, middle)?;
        p.print_region_without_entry(*region)
    }

    fn parse(&self, p: &mut OpAsmParser<'_>) -> ParseResult<Self::Value> {
        let args = p.parse_entry_args()?;
        let middle = self.0.parse(p)?;
        Ok((p.parse_region_with_entry(&args)?, middle))
    }

    fn parse_optional(&self, p: &mut OpAsmParser<'_>) -> ParseResult<Option<Self::Value>> {
        if p.peek().starts_with('(') {
            self.parse(p).map(Some)
        } else {
            Ok(None)
        }
    }

    fn name(&self, namer: &mut AsmNamer<'_>, (region, middle): &Self::Value) {
        namer.entry_args(*region);
        self.0.name(namer, middle);
        namer.region_without_entry(*region);
    }
}

// =============================================================================
// Sequences
// =============================================================================

/// Read the pieces of a sequence: optionally while every piece so far is
/// silent, then each as required.
macro_rules! sequence {
    ($($piece:ident $value:ident $index:tt),+) => {
        impl<$($piece: Syntax),+> Syntax for ($($piece,)+) {
            type Value = ($($piece::Value,)+);

            fn print(&self, p: &mut OpAsmPrinter<'_>, value: &Self::Value) -> fmt::Result {
                $(self.$index.print(p, &value.$index)?;)+
                Ok(())
            }

            fn parse(&self, p: &mut OpAsmParser<'_>) -> ParseResult<Self::Value> {
                Ok(($(self.$index.parse(p)?,)+))
            }

            fn parse_optional(
                &self,
                p: &mut OpAsmParser<'_>,
            ) -> ParseResult<Option<Self::Value>> {
                let mut deciding = true;
                $(
                    let $value = if deciding {
                        match self.$index.parse_optional(p)? {
                            Some(value) => {
                                deciding = self.$index.is_silent();
                                value
                            }
                            None => return Ok(None),
                        }
                    } else {
                        self.$index.parse(p)?
                    };
                )+
                let _ = deciding;
                Ok(Some(($($value,)+)))
            }

            fn is_silent(&self) -> bool {
                true $(&& self.$index.is_silent())+
            }

            fn name(&self, namer: &mut AsmNamer<'_>, value: &Self::Value) {
                $(self.$index.name(namer, &value.$index);)+
            }
        }
    };
}

sequence!(A a 0, B b 1);
sequence!(A a 0, B b 1, C c 2);
sequence!(A a 0, B b 1, C c 2, D d 3);
sequence!(A a 0, B b 1, C c 2, D d 3, E e 4);

/// `open`, `inner`, then `close`, keeping only the value of `inner`.
pub fn delimited<A, B, C>(open: A, inner: B, close: C) -> Delimited<A, B, C>
where
    A: Syntax<Value = ()>,
    B: Syntax,
    C: Syntax<Value = ()>,
{
    Delimited { open, inner, close }
}

/// `first`, then `second`, keeping only the value of `second`.
pub fn preceded<A: Syntax<Value = ()>, B: Syntax>(first: A, second: B) -> Delimited<A, B, Lit> {
    delimited(first, second, lit(""))
}

/// `first`, then `second`, keeping only the value of `first`.
pub fn terminated<A: Syntax, B: Syntax<Value = ()>>(first: A, second: B) -> Delimited<Lit, A, B> {
    delimited(lit(""), first, second)
}

#[derive(Clone, Copy, Debug)]
pub struct Delimited<A, B, C> {
    open: A,
    inner: B,
    close: C,
}

impl<A, B, C> Syntax for Delimited<A, B, C>
where
    A: Syntax<Value = ()>,
    B: Syntax,
    C: Syntax<Value = ()>,
{
    type Value = B::Value;

    fn print(&self, p: &mut OpAsmPrinter<'_>, value: &B::Value) -> fmt::Result {
        self.open.print(p, &())?;
        self.inner.print(p, value)?;
        self.close.print(p, &())
    }

    fn parse(&self, p: &mut OpAsmParser<'_>) -> ParseResult<B::Value> {
        self.open.parse(p)?;
        let value = self.inner.parse(p)?;
        self.close.parse(p)?;
        Ok(value)
    }

    fn parse_optional(&self, p: &mut OpAsmParser<'_>) -> ParseResult<Option<B::Value>> {
        if self.open.parse_optional(p)?.is_none() {
            return Ok(None);
        }
        let value = if self.open.is_silent() {
            match self.inner.parse_optional(p)? {
                Some(value) => value,
                None => return Ok(None),
            }
        } else {
            self.inner.parse(p)?
        };
        if self.open.is_silent() && self.inner.is_silent() {
            if self.close.parse_optional(p)?.is_none() {
                return Ok(None);
            }
        } else {
            self.close.parse(p)?;
        }
        Ok(Some(value))
    }

    fn is_silent(&self) -> bool {
        self.open.is_silent() && self.inner.is_silent() && self.close.is_silent()
    }

    fn name(&self, namer: &mut AsmNamer<'_>, value: &B::Value) {
        self.inner.name(namer, value);
    }
}

// =============================================================================
// Repetition
// =============================================================================

/// `item` as many times as it is present. The item's first non-silent
/// piece must be able to decline.
pub fn many<S: Syntax>(item: S) -> Many<S> {
    Many(item)
}

#[derive(Clone, Copy, Debug)]
pub struct Many<S>(S);

impl<S: Syntax> Syntax for Many<S> {
    type Value = Vec<S::Value>;

    fn print(&self, p: &mut OpAsmPrinter<'_>, values: &Self::Value) -> fmt::Result {
        values.iter().try_for_each(|value| self.0.print(p, value))
    }

    fn parse(&self, p: &mut OpAsmParser<'_>) -> ParseResult<Self::Value> {
        let mut values = Vec::new();
        while let Some(value) = self.0.parse_optional(p)? {
            values.push(value);
        }
        Ok(values)
    }

    fn name(&self, namer: &mut AsmNamer<'_>, values: &Self::Value) {
        for value in values {
            self.0.name(namer, value);
        }
    }
}

/// `item`s separated by `separator`, possibly none. The item's first
/// non-silent piece must be able to decline, so that an empty list reads.
pub fn separated<S: Syntax>(item: S, separator: &'static str) -> Separated<S> {
    Separated {
        item,
        separator: lit(separator),
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Separated<S> {
    item: S,
    separator: Lit,
}

impl<S: Syntax> Syntax for Separated<S> {
    type Value = Vec<S::Value>;

    fn print(&self, p: &mut OpAsmPrinter<'_>, values: &Self::Value) -> fmt::Result {
        for (index, value) in values.iter().enumerate() {
            if index > 0 {
                self.separator.print(p, &())?;
            }
            self.item.print(p, value)?;
        }
        Ok(())
    }

    fn parse(&self, p: &mut OpAsmParser<'_>) -> ParseResult<Self::Value> {
        Ok(self.parse_optional(p)?.unwrap_or_default())
    }

    /// Declines when the first item does, so that an [`optional`] list is
    /// absent rather than empty.
    fn parse_optional(&self, p: &mut OpAsmParser<'_>) -> ParseResult<Option<Self::Value>> {
        let Some(first) = self.item.parse_optional(p)? else {
            return Ok(None);
        };
        let mut values = vec![first];
        while self.separator.parse_optional(p)?.is_some() {
            values.push(self.item.parse(p)?);
        }
        Ok(Some(values))
    }

    fn name(&self, namer: &mut AsmNamer<'_>, values: &Self::Value) {
        for value in values {
            self.item.name(namer, value);
        }
    }
}

/// `inner` when it is present, and nothing otherwise.
pub fn optional<S: Syntax>(inner: S) -> Optional<S> {
    Optional(inner)
}

#[derive(Clone, Copy, Debug)]
pub struct Optional<S>(S);

impl<S: Syntax> Syntax for Optional<S> {
    type Value = Option<S::Value>;

    fn print(&self, p: &mut OpAsmPrinter<'_>, value: &Self::Value) -> fmt::Result {
        match value {
            Some(value) => self.0.print(p, value),
            None => Ok(()),
        }
    }

    fn parse(&self, p: &mut OpAsmParser<'_>) -> ParseResult<Self::Value> {
        self.0.parse_optional(p)
    }

    fn name(&self, namer: &mut AsmNamer<'_>, value: &Self::Value) {
        if let Some(value) = value {
            self.0.name(namer, value);
        }
    }
}

// =============================================================================
// Values
// =============================================================================

/// `inner` with its value changed in both directions: `parsed` turns what
/// `inner` reads into the value, and `printed` turns the value back into
/// what `inner` writes. The two must be inverse.
pub fn map<S: Syntax, U>(
    inner: S,
    parsed: fn(S::Value) -> U,
    printed: fn(&U) -> S::Value,
) -> Map<S, U> {
    Map {
        inner,
        parsed,
        printed,
    }
}

pub struct Map<S: Syntax, U> {
    inner: S,
    parsed: fn(S::Value) -> U,
    printed: fn(&U) -> S::Value,
}

impl<S: Syntax, U> Syntax for Map<S, U> {
    type Value = U;

    fn print(&self, p: &mut OpAsmPrinter<'_>, value: &U) -> fmt::Result {
        self.inner.print(p, &(self.printed)(value))
    }

    fn parse(&self, p: &mut OpAsmParser<'_>) -> ParseResult<U> {
        self.inner.parse(p).map(self.parsed)
    }

    fn parse_optional(&self, p: &mut OpAsmParser<'_>) -> ParseResult<Option<U>> {
        Ok(self.inner.parse_optional(p)?.map(self.parsed))
    }

    fn is_silent(&self) -> bool {
        self.inner.is_silent()
    }

    fn name(&self, namer: &mut AsmNamer<'_>, value: &U) {
        self.inner.name(namer, &(self.printed)(value));
    }
}

// =============================================================================
// Directives
// =============================================================================

/// An [`OpAsmDirective`](super::OpAsmDirective) whose syntax is one
/// [`Syntax`] description over a value it reads from the entities and
/// turns back into them. [`syntax_directive!`](crate::syntax_directive) implements the directive's
/// hooks from it.
pub trait SyntaxDirective {
    /// What the syntax prints and parses.
    type Value;

    /// The syntax.
    fn syntax() -> impl Syntax<Value = Self::Value>;

    /// The value the entities hold, or `None` when the syntax cannot write
    /// them; the operation then prints generically.
    fn from_entities(ctx: &IrContext, entities: &[CustomEntity<'_>]) -> Option<Self::Value>;

    /// The entities a parsed value stands for, one for each the element
    /// names, in order. The inverse of [`Self::from_entities`].
    fn into_entities(ctx: &mut IrContext, value: Self::Value) -> Vec<ParsedEntity>;
}

/// [`OpAsmDirective::assign_names`](super::OpAsmDirective::assign_names)
/// for a [`SyntaxDirective`].
pub fn assign_names<D: SyntaxDirective>(namer: &mut AsmNamer<'_>, entities: &[CustomEntity<'_>]) {
    if let Some(value) = D::from_entities(namer.ctx(), entities) {
        D::syntax().name(namer, &value);
    }
}

/// [`OpAsmDirective::parse`](super::OpAsmDirective::parse) for a
/// [`SyntaxDirective`].
pub fn parse<D: SyntaxDirective>(p: &mut OpAsmParser<'_>) -> ParseResult<Vec<ParsedEntity>> {
    let value = D::syntax().parse(p)?;
    Ok(D::into_entities(p.ctx_mut(), value))
}

/// Implement [`OpAsmDirective`](crate::asm_format::OpAsmDirective) for a
/// [`SyntaxDirective`] by delegating each hook to its syntax:
/// `syntax_directive!(SwitchArms)`, or with `spaced = false` for a
/// directive that writes no space before itself.
#[macro_export]
macro_rules! syntax_directive {
    ($directive:ty) => {
        $crate::syntax_directive!($directive, spaced = true);
    };
    ($directive:ty, spaced = $spaced:expr) => {
        impl $crate::asm_format::OpAsmDirective for $directive {
            type Parts<'a> = <$directive as $crate::asm_format::syntax::SyntaxDirective>::Value;

            const SPACED: bool = $spaced;

            fn decompose<'a>(
                ctx: &'a $crate::IrContext,
                entities: &'a [$crate::asm_format::CustomEntity<'a>],
            ) -> Option<Self::Parts<'a>> {
                <$directive as $crate::asm_format::syntax::SyntaxDirective>::from_entities(
                    ctx, entities,
                )
            }

            fn print(
                p: &mut $crate::printer::OpAsmPrinter<'_>,
                parts: Self::Parts<'_>,
            ) -> ::std::fmt::Result {
                $crate::asm_format::syntax::Syntax::print(
                    &<$directive as $crate::asm_format::syntax::SyntaxDirective>::syntax(),
                    p,
                    &parts,
                )
            }

            fn assign_names(
                namer: &mut $crate::printer::AsmNamer<'_>,
                entities: &[$crate::asm_format::CustomEntity<'_>],
            ) {
                $crate::asm_format::syntax::assign_names::<$directive>(namer, entities)
            }

            fn parse(
                p: &mut $crate::parser::OpAsmParser<'_>,
            ) -> $crate::parser::ParseResult<::std::vec::Vec<$crate::asm_format::ParsedEntity>>
            {
                $crate::asm_format::syntax::parse::<$directive>(p)
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::OpAsmParser;
    use crate::printer::AsmState;

    fn print<S: Syntax>(syntax: &S, value: &S::Value) -> String {
        let ctx = IrContext::new();
        let state = AsmState::without_aliases(&ctx);
        let mut out = String::new();
        syntax
            .print(&mut OpAsmPrinter::new(&state, &mut out), value)
            .unwrap();
        out
    }

    /// The value `syntax` reads from `text`, and the text it left.
    fn parse<S: Syntax>(syntax: &S, text: &str) -> (S::Value, String) {
        let mut ctx = IrContext::new();
        let mut p = OpAsmParser::new(&mut ctx, text);
        let value = syntax.parse(&mut p).unwrap();
        let rest = p.peek().to_owned();
        (value, rest)
    }

    #[test]
    fn literals_read_words_and_punctuation_apart() {
        let tokens = |text| literal_tokens(text).collect::<Vec<_>>();
        assert_eq!(tokens("default: "), ["default", ":"]);
        assert_eq!(tokens("\ncase "), ["case"]);
        assert_eq!(tokens(" -> "), ["->"]);
        assert!(tokens(" ").is_empty());
        assert!(lit(" ").is_silent());
        assert!(!lit(" : ").is_silent());
    }

    #[test]
    fn a_list_prints_and_reads_back_each_pair() {
        let syntax = delimited(
            lit("["),
            separated((terminated(int(), lit(": ")), int()), ", "),
            lit("]"),
        );
        for (value, text) in [(vec![(1, 2), (-3, 4)], "[1: 2, -3: 4]"), (vec![], "[]")] {
            assert_eq!(print(&syntax, &value), text);
            assert_eq!(parse(&syntax, text), (value, String::new()));
        }
    }

    #[test]
    fn repetition_asks_past_silent_pieces_whether_an_item_follows() {
        let syntax = many(preceded(lit("\n"), preceded(lit("case "), int())));
        assert_eq!(print(&syntax, &vec![1, 2]), "\ncase 1\ncase 2");
        assert_eq!(
            parse(&syntax, "case 1 case 2 default"),
            (vec![1, 2], "default".into())
        );
    }

    #[test]
    fn an_optional_list_is_absent_rather_than_empty() {
        let syntax = optional(terminated(separated(int(), ","), lit(";")));
        assert_eq!(print(&syntax, &None), "");
        assert_eq!(parse(&syntax, "1, 2;"), (Some(vec![1, 2]), String::new()));
        assert_eq!(parse(&syntax, "]"), (None, "]".into()));
    }
}
