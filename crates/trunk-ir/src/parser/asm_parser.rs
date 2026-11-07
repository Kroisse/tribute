//! [`AsmParser`]: the reader of types, attributes, and tokens.

use rustc_hash::FxHashMap as HashMap;

use winnow::error::ErrMode;
use winnow::prelude::*;

use super::lexer;
use super::{ParseError, ParseResult};
use crate::attr_kind::AttrKind;
use crate::context::IrContext;
use crate::refs::TypeRef;
use crate::symbol::{Symbol, SymbolPath};
use crate::types::{Attribute, AttributeMap, Location, PARAM_ATTRS_ATTR, TypeDataBuilder};

/// Reads the text format into IR, as MLIR's `AsmParser` does: types,
/// attributes, symbols, and tokens, each built in the context as it is
/// read. It is the parsing counterpart of
/// [`AsmPrinter`](crate::printer::AsmPrinter), and
/// [`OpAsmParser`](super::OpAsmParser) extends it with operations.
///
/// Every piece skips the whitespace before it. A `parse_*` piece reads its
/// construct or fails with an error at the offset where it failed; a
/// `parse_optional_*` piece returns `None`, leaving the text as it was,
/// when the construct is absent, and fails like `parse_*` once it has
/// begun. Errors do not backtrack: a format tells its alternatives apart
/// with optional pieces before it reads further, so a construct that fails
/// is never retried as another.
///
/// A piece creates operations, blocks, regions, and values only once it
/// has committed to its construct, so a construct an optional piece finds
/// absent leaves the context as it was. Types, strings, and names are
/// interned as they are read; interning is idempotent and adds nothing to
/// the IR.
pub struct AsmParser<'a> {
    ctx: &'a mut IrContext,
    source: &'a str,
    rest: &'a str,
    /// The type aliases of each enclosing module region, innermost last.
    aliases: Vec<HashMap<String, TypeRef>>,
    location: Location,
}

impl<'a> AsmParser<'a> {
    /// A parser of `source`, a type, an attribute, or a part of one, outside
    /// any module.
    pub fn new(ctx: &'a mut IrContext, source: &'a str) -> Self {
        let path = ctx.intern_path("textual-ir");
        let location = Location::new(path, crate::location::Span::new(0, 0));
        Self {
            ctx,
            source,
            rest: source,
            aliases: Vec::new(),
            location,
        }
    }

    pub fn ctx(&self) -> &IrContext {
        self.ctx
    }

    pub fn ctx_mut(&mut self) -> &mut IrContext {
        self.ctx
    }

    /// The location every parsed operation, block, and region gets.
    pub fn location(&self) -> Location {
        self.location
    }

    /// The byte offset of the next character in the parsed text.
    pub fn offset(&self) -> usize {
        self.source.len() - self.rest.len()
    }

    /// An error at the current offset.
    pub fn error(&self, message: impl Into<String>) -> ParseError {
        self.error_at(self.offset(), message)
    }

    /// An error at `offset`.
    pub fn error_at(&self, offset: usize, message: impl Into<String>) -> ParseError {
        ParseError {
            message: message.into(),
            offset,
        }
    }

    /// Bind `body`, which the text wrote at `offset`, to the identified type
    /// `ty`, as an identified type's assembly format does for an occurrence
    /// that writes a body (`new-plans/ir.md` § "식별 타입").
    pub fn bind_type_body(&mut self, ty: TypeRef, body: TypeRef, offset: usize) -> ParseResult<()> {
        self.ctx
            .bind_type_body(ty, body)
            .map_err(|error| match error {
                crate::TypeBodyError::NotIdentified => self.error_at(
                    offset,
                    "a body is written for a type that is not identified",
                ),
                crate::TypeBodyError::Conflicting { .. } => self.error_at(
                    offset,
                    "the body differs from the one this identified type already has",
                ),
            })
    }

    /// An error saying what the text should have held at the current offset.
    pub fn expected(&self, what: &str) -> ParseError {
        self.error(format!("expected {what}"))
    }

    /// Skip whitespace, and tell whether it held a line break.
    pub(crate) fn skip_ws(&mut self) -> bool {
        let trimmed = self
            .rest
            .trim_start_matches(|c: char| c.is_ascii_whitespace());
        let skipped = &self.rest[..self.rest.len() - trimmed.len()];
        self.rest = trimmed;
        skipped.contains('\n')
    }

    /// The text after the whitespace at the cursor, which a format looks at
    /// to tell its alternatives apart.
    pub fn peek(&mut self) -> &'a str {
        self.skip_ws();
        self.rest
    }

    /// The cursor, to return to with [`Self::reset`].
    pub(crate) fn checkpoint(&self) -> &'a str {
        self.rest
    }

    pub(crate) fn reset(&mut self, checkpoint: &'a str) {
        self.rest = checkpoint;
    }

    /// Whether only whitespace is left.
    pub fn at_end(&mut self) -> bool {
        self.peek().is_empty()
    }

    // ------------------------------------------------------------------
    // Tokens
    // ------------------------------------------------------------------

    /// Read one token with `token`, a [`lexer`] function or another winnow
    /// parser of text; `what` names it in the error when it is absent.
    pub fn parse_token<T>(
        &mut self,
        what: &str,
        token: impl FnOnce(&mut &'a str) -> ModalResult<T>,
    ) -> ParseResult<T> {
        self.skip_ws();
        let start = self.rest;
        let mut input = self.rest;
        match token(&mut input) {
            Ok(value) => {
                self.rest = input;
                Ok(value)
            }
            Err(_) => {
                self.rest = start;
                Err(self.expected(what))
            }
        }
    }

    /// Read one token with `token` if it is there, or leave the text as it
    /// was. A token that fails after it has begun, with a cut, is an error.
    pub fn parse_optional_token<T>(
        &mut self,
        what: &str,
        token: impl FnOnce(&mut &'a str) -> ModalResult<T>,
    ) -> ParseResult<Option<T>> {
        let checkpoint = self.rest;
        self.skip_ws();
        let mut input = self.rest;
        match token(&mut input) {
            Ok(value) => {
                self.rest = input;
                Ok(Some(value))
            }
            Err(ErrMode::Backtrack(_)) => {
                self.rest = checkpoint;
                Ok(None)
            }
            Err(_) => Err(self.expected(what)),
        }
    }

    /// Read `keyword` as a whole identifier, not followed by `.`, as an
    /// operation name would be.
    pub fn parse_keyword(&mut self, keyword: &str) -> ParseResult<()> {
        if self.parse_optional_keyword(keyword) {
            Ok(())
        } else {
            self.skip_ws();
            Err(self.expected(&format!("`{keyword}`")))
        }
    }

    /// Read `keyword` if it is next.
    pub fn parse_optional_keyword(&mut self, keyword: &str) -> bool {
        let checkpoint = self.rest;
        self.skip_ws();
        let mut input = self.rest;
        match lexer::ident.parse_next(&mut input) {
            Ok(word) if word == keyword && !input.starts_with('.') => {
                self.rest = input;
                true
            }
            _ => {
                self.rest = checkpoint;
                false
            }
        }
    }

    /// Read `punct`, such as `(` or `->`, as is.
    pub fn parse_punct(&mut self, punct: &str) -> ParseResult<()> {
        if self.parse_optional_punct(punct) {
            Ok(())
        } else {
            self.skip_ws();
            Err(self.expected(&format!("`{punct}`")))
        }
    }

    /// Read `punct` if it is next.
    pub fn parse_optional_punct(&mut self, punct: &str) -> bool {
        let checkpoint = self.rest;
        self.skip_ws();
        match self.rest.strip_prefix(punct) {
            Some(rest) => {
                self.rest = rest;
                true
            }
            None => {
                self.rest = checkpoint;
                false
            }
        }
    }

    /// Read a format's literal: a keyword when it begins as an identifier
    /// does, and punctuation otherwise.
    pub fn parse_literal(&mut self, text: &str) -> ParseResult<()> {
        if is_keyword(text) {
            self.parse_keyword(text)
        } else {
            self.parse_punct(text)
        }
    }

    /// Read a format's literal if it is next.
    pub fn parse_optional_literal(&mut self, text: &str) -> bool {
        if is_keyword(text) {
            self.parse_optional_keyword(text)
        } else {
            self.parse_optional_punct(text)
        }
    }

    /// Read an identifier.
    pub fn parse_identifier(&mut self) -> ParseResult<&'a str> {
        self.parse_token("an identifier", lexer::ident)
    }

    /// Read an identifier if one is next.
    pub fn parse_optional_identifier(&mut self) -> Option<&'a str> {
        self.parse_optional_token("an identifier", lexer::ident)
            .ok()
            .flatten()
    }

    /// Read an integer literal.
    pub fn parse_integer(&mut self) -> ParseResult<i128> {
        self.parse_token("an integer", lexer::integer_lit)
    }

    /// Read a string literal.
    pub fn parse_string(&mut self) -> ParseResult<String> {
        self.parse_token("a string", lexer::string_lit)
    }

    /// Read a definition name, `@name`, as the name's text.
    pub fn parse_symbol_name(&mut self) -> ParseResult<String> {
        self.parse_token("`@name`", lexer::symbol_ref)
    }

    /// Read a definition name if one is next.
    pub fn parse_optional_symbol_name(&mut self) -> ParseResult<Option<String>> {
        let checkpoint = self.rest;
        if !self.peek().starts_with('@') {
            self.rest = checkpoint;
            return Ok(None);
        }
        self.parse_symbol_name().map(Some)
    }

    /// Read a symbol reference, `@a::@b`.
    pub fn parse_symbol_ref(&mut self) -> ParseResult<SymbolPath> {
        let components = self.parse_token("a symbol reference", lexer::symbol_path)?;
        Ok(symbol_path(components))
    }

    /// Read `item`s separated by commas: none when `close` is next, which
    /// this does not read.
    pub fn parse_list<T>(
        &mut self,
        close: &str,
        mut item: impl FnMut(&mut Self) -> ParseResult<T>,
    ) -> ParseResult<Vec<T>> {
        let mut items = Vec::new();
        if self.peek().starts_with(close) {
            return Ok(items);
        }
        loop {
            items.push(item(self)?);
            if !self.parse_optional_punct(",") {
                return Ok(items);
            }
        }
    }

    /// Read `open`, `item`s separated by commas, and `close`.
    pub fn parse_delimited<T>(
        &mut self,
        open: &str,
        close: &str,
        item: impl FnMut(&mut Self) -> ParseResult<T>,
    ) -> ParseResult<Vec<T>> {
        self.parse_punct(open)?;
        let items = self.parse_list(close, item)?;
        self.parse_punct(close)?;
        Ok(items)
    }

    // ------------------------------------------------------------------
    // Types
    // ------------------------------------------------------------------

    /// Read a type.
    ///
    /// Types are self-delimiting: everything a type owns is inside its angle
    /// brackets, so a `{` after a type never belongs to it.
    ///
    /// ```text
    /// dialect.name
    /// dialect.name<param {param attrs}, param, {type attrs}>
    /// !alias
    /// ```
    ///
    /// An element that starts with `{` holds the type's own attributes and
    /// must come last. A dictionary directly after a parameter holds that
    /// parameter's attributes. A type whose definition registers a
    /// [`TypeAsmFormat`](crate::asm_format::TypeAsmFormat) parses its own
    /// syntax inside the brackets first.
    pub fn parse_type(&mut self) -> ParseResult<TypeRef> {
        self.skip_ws();
        if self.rest.starts_with('!') {
            let start = self.offset();
            let name = self.parse_token("a type alias", |input| {
                '!'.parse_next(input)?;
                lexer::type_alias_name(input)
            })?;
            return self
                .alias(&name)
                .ok_or_else(|| self.error_at(start, format!("undefined type alias '!{name}'")));
        }
        let name = self.parse_token("a type", lexer::qualified_name)?;
        if !self.rest.starts_with('<') {
            return self.build_type(name, Vec::new(), AttributeMap::new());
        }
        self.rest = &self.rest[1..];
        self.skip_ws();

        if let Some(hooks) = crate::TypeDef::lookup(name).and_then(|def| def.asm_format) {
            let checkpoint = self.rest;
            match (hooks.parse)(self, name)? {
                Some(ty) => return Ok(ty),
                None => self.rest = checkpoint,
            }
        }

        let mut params = Vec::new();
        let mut attrs = AttributeMap::new();
        if !self.peek().starts_with('>') {
            loop {
                if self.peek().starts_with('{') {
                    attrs = self.parse_attr_dict()?;
                    break;
                }
                params.push(self.parse_param()?);
                if !self.parse_optional_punct(",") {
                    break;
                }
            }
        }
        self.parse_punct(">")?;
        self.build_type(name, params, attrs)
    }

    /// Read a type parameter and its own attributes, `T {k = v}`.
    pub fn parse_param(&mut self) -> ParseResult<(TypeRef, AttributeMap)> {
        let ty = self.parse_type()?;
        let attrs = self.parse_optional_attr_dict()?.unwrap_or_default();
        Ok((ty, attrs))
    }

    /// Read a parenthesized parameter list, `(T {k = v}, U)`.
    pub fn parse_param_list(&mut self) -> ParseResult<Vec<(TypeRef, AttributeMap)>> {
        self.parse_delimited("(", ")", Self::parse_param)
    }

    /// Intern the type `name` with `params`, each with its own attributes,
    /// and the type attributes `attrs`, as the generic form writes it.
    ///
    /// The textual form writes parameter attributes only beside each
    /// parameter, so an explicit [`PARAM_ATTRS_ATTR`] key is an error.
    pub fn build_type(
        &mut self,
        name: &str,
        params: Vec<(TypeRef, AttributeMap)>,
        attrs: AttributeMap,
    ) -> ParseResult<TypeRef> {
        if attrs.view(self.ctx).contains_key(PARAM_ATTRS_ATTR) {
            return Err(self.error(format!(
                "{name}: `{PARAM_ATTRS_ATTR}` is reserved; write parameter attributes after each parameter"
            )));
        }
        let name = crate::TypeName::intern(self.ctx.strings_mut(), name);
        let mut builder = TypeDataBuilder::new(name);
        for (ty, param_attrs) in params {
            builder = builder.param_with_attrs(ty, param_attrs);
        }
        for (key, value) in attrs {
            builder = builder.attr(key, value);
        }
        Ok(builder.intern(self.ctx))
    }

    fn alias(&self, name: &str) -> Option<TypeRef> {
        self.aliases
            .iter()
            .rev()
            .find_map(|scope| scope.get(name).copied())
    }

    pub(crate) fn push_alias_scope(&mut self) {
        self.aliases.push(HashMap::default());
    }

    pub(crate) fn pop_alias_scope(&mut self) {
        self.aliases.pop();
    }

    /// Read a type alias definition, `!name = type`, and register it.
    pub(crate) fn parse_alias_def(&mut self) -> ParseResult<()> {
        let name = self.parse_token("a type alias", |input| {
            '!'.parse_next(input)?;
            lexer::type_alias_name(input)
        })?;
        self.parse_punct("=")?;
        let ty = self.parse_type()?;
        if let Some(scope) = self.aliases.last_mut() {
            scope.insert(name.clone(), ty);
        }
        self.ctx.register_type_alias(Symbol::new(&name), ty);
        Ok(())
    }

    // ------------------------------------------------------------------
    // Attributes
    // ------------------------------------------------------------------

    /// Read an attribute value in generic attribute syntax.
    pub fn parse_attribute(&mut self) -> ParseResult<Attribute> {
        self.skip_ws();
        for (keyword, value) in [
            ("true", Attribute::Bool(true)),
            ("false", Attribute::Bool(false)),
            ("unit", Attribute::Unit),
        ] {
            if let Some(rest) = self.rest.strip_prefix(keyword) {
                self.rest = rest;
                return Ok(value);
            }
        }
        if let Some((path, start, end)) =
            self.parse_optional_token("a location", lexer::location)?
        {
            let path = self.ctx.intern_path(&path);
            return Ok(Attribute::Location(Location::new(
                path,
                crate::location::Span::new(start, end),
            )));
        }
        if let Some(bytes) = self.parse_optional_token("a byte string", lexer::bytes_lit)? {
            return Ok(Attribute::Bytes(bytes.into_iter().collect()));
        }
        if self.rest.starts_with('"') {
            let text = self.parse_string()?;
            return Ok(self.ctx.string_attr(&text));
        }
        if self.rest.starts_with('@') {
            return Ok(Attribute::SymbolRef(self.parse_symbol_ref()?));
        }
        if self.rest.starts_with('[') {
            return Ok(Attribute::List(self.parse_delimited(
                "[",
                "]",
                Self::parse_attribute,
            )?));
        }
        if self.rest.starts_with('{') {
            return self.parse_dict_attribute();
        }
        if let Some(value) = self.parse_optional_token("a float", lexer::float_with_dot)? {
            return Ok(Attribute::FloatBits(value.to_bits()));
        }
        if let Some(value) = self.parse_optional_token("an integer", lexer::integer_lit)? {
            return Ok(Attribute::Int(value));
        }
        if self.rest.starts_with('!') || lexer::qualified_name.parse_peek(self.rest).is_ok() {
            return Ok(Attribute::Type(self.parse_type()?));
        }
        Err(self.expected("attribute value"))
    }

    /// Read an attribute of kind `K` as the kind spells it
    /// ([`AttrKind::parse_text`]).
    pub fn parse_attr<K: AttrKind>(&mut self) -> ParseResult<Attribute> {
        K::parse_text(self)
    }

    /// A dictionary attribute, `{k = v}`, whose keys are distinct.
    fn parse_dict_attribute(&mut self) -> ParseResult<Attribute> {
        let mut dict = AttributeMap::new();
        self.parse_entries(|p, key, offset, value| {
            let key = p.ctx.intern_str(&key);
            if dict.view_mut(p.ctx).insert(key, value).is_some() {
                let key = p.ctx.str(key).to_owned();
                return Err(p.error_at(
                    offset,
                    format!("duplicate dictionary attribute key '{key}'"),
                ));
            }
            Ok(())
        })?;
        Ok(Attribute::Dict(dict))
    }

    /// Read an attribute dictionary, `{k = v, ..}`. A key written twice
    /// keeps its last value.
    pub fn parse_attr_dict(&mut self) -> ParseResult<AttributeMap> {
        let mut map = AttributeMap::new();
        self.parse_entries(|p, key, _, value| {
            let key = p.ctx.intern_str(&key);
            map.view_mut(p.ctx).insert(key, value);
            Ok(())
        })?;
        Ok(map)
    }

    /// Read an attribute dictionary if one is next.
    pub fn parse_optional_attr_dict(&mut self) -> ParseResult<Option<AttributeMap>> {
        if self.peek_dict() {
            self.parse_attr_dict().map(Some)
        } else {
            Ok(None)
        }
    }

    /// Whether a `{` is next, without moving the cursor.
    fn peek_dict(&mut self) -> bool {
        let checkpoint = self.rest;
        let found = self.peek().starts_with('{');
        self.rest = checkpoint;
        found
    }

    fn parse_entries(
        &mut self,
        mut entry: impl FnMut(&mut Self, String, usize, Attribute) -> ParseResult<()>,
    ) -> ParseResult<()> {
        self.parse_punct("{")?;
        let entries = self.parse_list("}", |p| {
            p.skip_ws();
            let offset = p.offset();
            let key = p.parse_token("an attribute key", lexer::attribute_key)?;
            p.parse_punct("=")?;
            let value = p.parse_attribute()?;
            Ok((key, offset, value))
        })?;
        self.parse_punct("}")?;
        for (key, offset, value) in entries {
            entry(self, key, offset, value)?;
        }
        Ok(())
    }
}

/// A symbol path from its components' text.
pub(crate) fn symbol_path(components: Vec<String>) -> SymbolPath {
    SymbolPath::new(components.iter().map(|name| Symbol::new(name)))
}

/// Whether a literal is a keyword, read as a whole identifier.
pub(crate) fn is_keyword(text: &str) -> bool {
    text.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
}
