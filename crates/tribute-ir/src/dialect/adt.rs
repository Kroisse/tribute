//! Arena-based adt dialect.

use crate::dialect::tribute_rtti::ManagedRefType;
use trunk_ir::asm_format::TypeAliasHint;
use trunk_ir::dialect::core::Index;
use trunk_ir::op_flag::Pure;

#[trunk_ir::dialect]
mod adt {
    /// Validated wrapper for a nominal `adt.struct` layout type.
    ///
    /// Field types are the type's parameters, each field's name is its parameter's
    /// `name` attribute, and the struct's name is the type's `name` attribute.
    #[verify]
    #[asm_format]
    #[alias_hint]
    #[implements(ManagedRefType)]
    struct Struct<#[rest] FieldTypes> {
        name: Attr<String>,
    }

    /// Validated wrapper for a nominal `adt.enum` layout type.
    ///
    /// Each variant is one type parameter: an `adt.variant` whose parameters are
    /// the variant's field types and whose `name` attribute is the variant's
    /// name. A field may carry a `name` parameter attribute. The enum's name is
    /// the type's `name` attribute.
    #[verify]
    #[asm_format]
    #[alias_hint]
    #[implements(ManagedRefType)]
    struct Enum<#[rest] VariantTypes> {
        name: Attr<String>,
    }

    /// `adt.typeref`: a reference to a nominal layout by its `name`.
    #[alias_hint]
    #[implements(ManagedRefType)]
    struct Typeref {
        name: Attr<String>,
    }

    /// `adt.array`: a managed reference to an array object whose elements
    /// are `Element` values in order. An element is a scalar, a pointer, or a
    /// managed reference; see `new-plans/ir.md` § "`adt` 배열". The unmanaged
    /// linear array is `core.array`, which no `adt` array operation accepts.
    #[implements(ManagedRefType)]
    struct Array<Element>;

    /// `adt.variant`: one variant of an `adt.enum`, named by its `name`.
    #[alias_hint]
    struct Variant<#[rest] Fields> {
        name: Attr<String>,
    }

    #[flags(Pure)]
    fn struct_new(r#type: Attr<Type>, fields: Variadic<_>) -> Value<_> {}

    #[flags(Pure)]
    fn struct_get(r#type: Attr<Type>, field: Attr<u32>, r#ref: Value<_>) -> Value<_> {}

    fn struct_set(r#type: Attr<Type>, field: Attr<u32>, r#ref: Value<_>, value: Value<_>) {}

    #[flags(Pure)]
    fn variant_new(r#type: Attr<Type>, tag: Attr<String>, fields: Variadic<_>) -> Value<_> {}

    #[flags(Pure)]
    fn variant_is(r#type: Attr<Type>, tag: Attr<String>, r#ref: Value<_>) -> Value<_> {}

    #[flags(Pure)]
    fn variant_cast(r#type: Attr<Type>, tag: Attr<String>, r#ref: Value<_>) -> Value<_> {}

    #[flags(Pure)]
    fn variant_get(
        r#type: Attr<Type>,
        tag: Attr<String>,
        field: Attr<u32>,
        r#ref: Value<_>,
    ) -> Value<_> {
    }

    /// A `length`-element array whose every element is `initial`.
    #[flags(Pure)]
    fn array_new<A: Array>(
        r#type: Attr<TypeOf<A>>,
        length: Value<Index>,
        initial: Value<A::Element>,
    ) -> Value<A> {
    }

    /// An array of `elements` in order. Its length is the number of
    /// operands, which may be zero.
    #[flags(Pure)]
    fn array_of<A: Array>(r#type: Attr<TypeOf<A>>, elements: Variadic<A::Element>) -> Value<A> {}

    /// A new array of `source[0, start)`, `elements`, and `source[end, len)`.
    ///
    /// It builds the array by copying ranges and leaves `source` unchanged.
    /// Inserting one element (`start = end`), removing one (no `elements` and
    /// `end = start + 1`), replacing elements, and trimming either end are
    /// all written with this operation. It requires `start ≤ end ≤ len`; the
    /// target traps an execution that violates it.
    #[flags(Pure)]
    fn array_splice<A: Array>(
        r#type: Attr<TypeOf<A>>,
        source: Value<A>,
        start: Value<Index>,
        end: Value<Index>,
        elements: Variadic<A::Element>,
    ) -> Value<A> {
    }

    /// The element at `index`. It requires `index < len`; the target traps
    /// an execution that violates it.
    #[flags(Pure)]
    fn array_get<A: Array>(r#ref: Value<A>, index: Value<Index>) -> Value<A::Element> {}

    /// Replace the element at `index` in place. Every value that refers to
    /// the same array sees the new element; a producer that must not change
    /// a shared array builds the changed array with `adt.array_splice`
    /// instead. It requires `index < len`; the target traps an execution that
    /// violates it.
    fn array_set<A: Array>(r#ref: Value<A>, index: Value<Index>, value: Value<A::Element>) {}

    /// The number of elements.
    #[flags(Pure)]
    fn array_len(r#ref: Value<impl Array>) -> Value<Index> {}

    #[flags(Pure)]
    fn ref_null(r#type: Attr<Type>) -> Value<_> {}

    #[flags(Pure)]
    fn ref_is_null(r#ref: Value<_>) -> Value<_> {}

    #[implements(Fold)]
    #[flags(Pure)]
    fn ref_cast(r#type: Attr<Type>, r#ref: Value<_>) -> Value<_> {}

    #[flags(Pure)]
    fn string_const(value: Attr<String>) -> Value<_> {}

    #[flags(Pure)]
    fn bytes_const(value: Attr<Bytes>) -> Value<_> {}
}

// A value of a nominal layout type, or a reference to one by name, points to
// an allocation of that layout.
impl ManagedRefType for Struct {}
impl ManagedRefType for Enum {}
impl ManagedRefType for Typeref {}
impl ManagedRefType for Array {}

// A named ADT type suggests its `name` as its alias.
impl TypeAliasHint for Struct {
    fn alias_hint<'a>(&self, ctx: &'a IrContext) -> Option<&'a str> {
        Some(self.name(ctx))
    }
}

impl TypeAliasHint for Enum {
    fn alias_hint<'a>(&self, ctx: &'a IrContext) -> Option<&'a str> {
        Some(self.name(ctx))
    }
}

impl TypeAliasHint for Typeref {
    fn alias_hint<'a>(&self, ctx: &'a IrContext) -> Option<&'a str> {
        Some(self.name(ctx))
    }
}

impl TypeAliasHint for Variant {
    fn alias_hint<'a>(&self, ctx: &'a IrContext) -> Option<&'a str> {
        Some(self.name(ctx))
    }
}

// === Canonicalization folds ===

use rustc_hash::FxHashSet as HashSet;
use std::ops::ControlFlow;
use trunk_ir::analysis::AnalysisCache;
use trunk_ir::ops::DialectOp;
use trunk_ir::pass::{Pass, PassRunResult};
use trunk_ir::refs::{TypeRef, ValueDef};
use trunk_ir::rewrite::{
    Module, OpRewritePattern, PatternApplicator, PatternRewriter, TypeConverter,
};
use trunk_ir::transforms::canonicalize::{Fold, FoldResult};
use trunk_ir::walk::{WalkAction, walk_op};

/// `adt.struct_get(adt.struct_new(.., %f, ..))` → `%f`, over the whole module.
///
/// Both ops must name the same layout type, the field index must be in
/// range, and the stored operand must already have the result type: a fold
/// never forwards a value of another type. A layout type that any
/// `adt.struct_set` in the module writes is mutable (compiler-generated cells
/// reach their writers through casts and closure environments, so use lists
/// cannot prove a particular struct unwritten) and is never folded. The pass
/// also skips a struct that any `tribute_rt` ownership operation already
/// consumes, since forwarding the field would then bypass an explicit
/// ownership unit. Reference counting materializes after this pass runs; the
/// ownership plan then sees the forwarded use directly.
///
/// This is a module-level pass rather than a per-op canonicalize fold so the
/// written layouts are collected once per run instead of once per candidate.
pub struct ForwardStructGets;

impl Pass for ForwardStructGets {
    type Target = trunk_ir::dialect::core::Module;

    fn name(&self) -> &'static str {
        "forward-struct-gets"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: trunk_ir::dialect::core::Module,
        _analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        let root = Module::from(target);
        let mut written = HashSet::default();
        let _ = walk_op::<()>(ctx, root.op(), &mut |op| {
            if let Ok(set) = StructSet::from_op(ctx, op) {
                written.insert(set.r#type(ctx));
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
        PatternApplicator::new(TypeConverter::new())
            .add_pattern(ForwardStructGet { written })
            .apply_partial(ctx, root);
        Ok(())
    }
}

struct ForwardStructGet {
    written: HashSet<TypeRef>,
}

impl OpRewritePattern for ForwardStructGet {
    type Op = StructGet;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        get: Self::Op,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        if self.written.contains(&get.r#type(ctx)) {
            return None;
        }
        let source = get.r#ref(ctx);
        let ValueDef::OpResult(producer, _) = ctx.value_def(source) else {
            return None;
        };
        let new = StructNew::from_op(ctx, producer).ok()?;
        if new.r#type(ctx) != get.r#type(ctx) {
            return None;
        }
        let &stored = ctx.op_operands(producer).get(get.field(ctx) as usize)?;
        let [result_ty] = ctx.op_result_types(get.op_ref()).as_array()?;
        if ctx.value_ty(stored) != *result_ty {
            return None;
        }
        let owned = ctx.uses(source).iter().any(|usage| {
            crate::dialect::tribute_rt::Retain::matches(ctx, usage.user)
                || crate::dialect::tribute_rt::Release::matches(ctx, usage.user)
                || crate::dialect::tribute_rt::IntoRaw::matches(ctx, usage.user)
        });
        if owned {
            return None;
        }
        rewriter.erase_op(vec![stored]);
        Some(())
    }

    fn name(&self) -> &'static str {
        "ForwardStructGet"
    }
}

/// `adt.ref_cast {type = T} (%x : T)` → `%x`.
///
/// The cast is the identity when its operand already has the result type.
impl Fold for RefCast {
    fn fold(&self, ctx: &IrContext) -> Option<FoldResult> {
        let op = self.op_ref();
        let cast = self;
        let operand = cast.r#ref(ctx);
        let [result_ty] = ctx.op_result_types(op).as_array()?;
        (ctx.value_ty(operand) == *result_ty && cast.r#type(ctx) == *result_ty)
            .then_some(FoldResult::Forward(operand))
    }
}

// === Nominal struct layout type ===

use trunk_ir::attr_kind::{Bytes, Type};
use trunk_ir::context::IrContext;
use trunk_ir::ops::DialectType;
use trunk_ir::types::{
    Attribute, AttributeMap, PARAM_ATTRS_ATTR, StringArg, StringRef, TypeDataBuilder,
};

pub mod layout;

/// Type attribute holding an `adt.struct`'s name, and the parameter attribute
/// holding each field's name. Both are strings.
pub const STRUCT_NAME_ATTR: &str = "name";

/// Whether `ty` is an `adt.struct`, `adt.enum` or `adt.typeref`: a type
/// named by its `name` attribute.
pub fn is_nominal(ctx: &IrContext, ty: TypeRef) -> bool {
    let name = ctx.get_type(ty).name;
    name == <Struct as DialectType>::NAME
        || name == <Enum as DialectType>::NAME
        || name == Typeref::NAME
}

/// The nominal name of an `adt.struct`, `adt.enum` or `adt.typeref`: its
/// `name` type attribute.
pub fn nominal_name_ref(ctx: &IrContext, ty: TypeRef) -> Option<StringRef> {
    is_nominal(ctx, ty)
        .then(|| ctx.type_attrs(ty).get_string_ref(STRUCT_NAME_ATTR))
        .flatten()
}

/// The text of [`nominal_name_ref`].
pub fn nominal_name(ctx: &IrContext, ty: TypeRef) -> Option<&str> {
    nominal_name_ref(ctx, ty).map(|name| ctx.str(name))
}

/// A malformed `adt.struct` type.
#[derive(Clone, Debug, PartialEq, Eq, derive_more::Display, derive_more::Error)]
pub enum StructTypeError {
    #[display("missing `{STRUCT_NAME_ATTR}` string")]
    MissingName,
    #[display("field {_0} has no `{STRUCT_NAME_ATTR}` string")]
    MissingFieldName(#[error(not(source))] usize),
    #[display("duplicate field name {_0:?}")]
    DuplicateFieldName(#[error(not(source))] String),
    #[display("`fields` is not an `adt.struct` attribute; fields are type parameters")]
    FieldsAttribute,
    #[display("closure base fields are not a prefix of the closure capture layout")]
    ClosureBasePrefix,
}

impl trunk_ir::type_verifier::VerifyType for Struct {
    /// Validate a name-matching `adt.struct`.
    fn verify(ctx: &IrContext, ty: TypeRef) -> Result<Self, trunk_ir::ops::VerificationError> {
        let data = ctx.get_type(ty);
        debug_assert!(Self::matches(ctx, ty));
        if ctx.type_attrs(ty).contains_key("fields") {
            return Err(StructTypeError::FieldsAttribute.into());
        }
        if ctx
            .type_attrs(ty)
            .get_string_ref(STRUCT_NAME_ATTR)
            .is_none()
        {
            return Err(StructTypeError::MissingName.into());
        }
        let mut names: smallvec::SmallVec<[StringRef; 8]> = smallvec::SmallVec::new();
        for index in 0..data.params.len() {
            let name = data
                .param_attrs(ctx, index)
                .view(ctx)
                .get_string_ref(STRUCT_NAME_ATTR)
                .ok_or(StructTypeError::MissingFieldName(index))?;
            if names.contains(&name) {
                return Err(StructTypeError::DuplicateFieldName(ctx.str(name).to_owned()).into());
            }
            names.push(name);
        }
        if let Some(base) = ctx
            .type_attrs(ty)
            .get_type(crate::runtime_layout::CLOSURE_BASE_ATTR)
        {
            let base_data = ctx.get_type(base);
            let prefix = base_data.name == data.name
                && base_data.params.len() <= data.params.len()
                && (0..base_data.params.len()).all(|index| {
                    base_data.params[index] == data.params[index]
                        && base_data.param_attrs(ctx, index) == data.param_attrs(ctx, index)
                });
            if !prefix {
                return Err(StructTypeError::ClosureBasePrefix.into());
            }
        }
        Ok(Self(ty))
    }
}

impl Struct {
    pub fn field_count(self, ctx: &IrContext) -> usize {
        ctx.get_type(self.0).params.len()
    }

    /// The type of field `index`, if it exists.
    pub fn field_type(self, ctx: &IrContext, index: usize) -> Option<TypeRef> {
        ctx.get_type(self.0).params.get(index).copied()
    }

    /// The name of field `index`, if it exists.
    pub fn field_name(self, ctx: &IrContext, index: usize) -> Option<&str> {
        self.field_name_ref(ctx, index).map(|name| ctx.str(name))
    }

    /// The name of field `index` as a pooled handle, if it exists.
    pub fn field_name_ref(self, ctx: &IrContext, index: usize) -> Option<StringRef> {
        let data = ctx.get_type(self.0);
        (index < data.params.len()).then(|| field_name(ctx, data.param_attrs(ctx, index)))
    }

    /// The index of the field named `name`.
    pub fn field_index(self, ctx: &IrContext, name: &str) -> Option<usize> {
        self.fields(ctx).position(|(field, _)| field == name)
    }

    /// Each field's name and type, in declaration order.
    pub fn fields(self, ctx: &IrContext) -> impl ExactSizeIterator<Item = (&str, TypeRef)> + '_ {
        let data = ctx.get_type(self.0);
        data.params
            .iter()
            .enumerate()
            .map(|(index, &ty)| (ctx.str(field_name(ctx, data.param_attrs(ctx, index))), ty))
    }

    /// The attributes of field `index` other than its name.
    pub fn field_attrs(
        self,
        ctx: &IrContext,
        index: usize,
    ) -> impl Iterator<Item = (StringRef, &Attribute)> + '_ {
        ctx.get_type(self.0)
            .param_attrs(ctx, index)
            .iter()
            .filter(|&(key, _)| ctx.str(key) != STRUCT_NAME_ATTR)
    }

    /// The type attributes other than the name and the field attributes.
    pub fn extra_attrs(
        self,
        ctx: &IrContext,
    ) -> impl Iterator<Item = (StringRef, &Attribute)> + '_ {
        ctx.get_type(self.0)
            .attrs
            .iter()
            .filter(|&(key, _)| ![STRUCT_NAME_ATTR, PARAM_ATTRS_ATTR].contains(&ctx.str(key)))
    }

    /// Each field as an owned `(name, type, attributes)` triple, for rebuilding
    /// the struct with [`try_from_fields_with_attrs`](Self::try_from_fields_with_attrs).
    pub fn fields_with_attrs(self, ctx: &IrContext) -> Vec<(StringRef, TypeRef, AttributeMap)> {
        (0..self.field_count(ctx))
            .map(|index| {
                let name = self
                    .field_name_ref(ctx, index)
                    .expect("field index is in range");
                let ty = self
                    .field_type(ctx, index)
                    .expect("field index is in range");
                let attrs = AttributeMap::from_refs(
                    ctx,
                    self.field_attrs(ctx, index)
                        .map(|(key, value)| (key, value.clone())),
                );
                (name, ty, attrs)
            })
            .collect()
    }
}

fn field_name(ctx: &IrContext, attrs: &AttributeMap) -> StringRef {
    attrs
        .view(ctx)
        .get_string_ref(STRUCT_NAME_ATTR)
        .expect("validated adt.struct fields must retain their names")
}

impl Struct {
    /// Construct an `adt.struct` named `name` with `fields` in declaration
    /// order.
    ///
    /// `attrs` holds the remaining type attributes, such as `layout`.
    ///
    /// # Panics
    ///
    /// If the result is not a valid `adt.struct`; see
    /// [`try_from_fields_with_attrs`](Self::try_from_fields_with_attrs).
    pub fn from_fields<N: Into<StringArg>>(
        ctx: &mut IrContext,
        name: impl Into<StringArg>,
        fields: impl IntoIterator<Item = (N, TypeRef)>,
        attrs: AttributeMap,
    ) -> Self {
        let fields = fields
            .into_iter()
            .map(|(field, ty)| (field, ty, AttributeMap::new()));
        Self::try_from_fields_with_attrs(ctx, name, fields, attrs)
            .unwrap_or_else(|error| panic!("adt.struct: {error}"))
    }

    /// Construct an `adt.struct` whose fields carry their own attributes,
    /// reporting a duplicate field name or a `fields` attribute instead of
    /// panicking.
    ///
    /// A `name` in `attrs` or in a field's attributes is replaced by `name` or
    /// the field's name, so a caller may pass a source struct's attributes
    /// unchanged.
    pub fn try_from_fields_with_attrs<N: Into<StringArg>>(
        ctx: &mut IrContext,
        name: impl Into<StringArg>,
        fields: impl IntoIterator<Item = (N, TypeRef, AttributeMap)>,
        attrs: AttributeMap,
    ) -> Result<Self, StructTypeError> {
        let mut builder = TypeDataBuilder::new(<Struct as DialectType>::NAME);
        for (field, ty, mut field_attrs) in fields {
            let field = ctx.intern_string_arg(field.into());
            field_attrs.view_mut(ctx).insert(STRUCT_NAME_ATTR, field);
            builder = builder.param_with_attrs(ty, field_attrs);
        }
        let name = ctx.intern_string_arg(name.into());
        finish_struct_type(ctx, name, builder, attrs)
    }
}

/// The non-generic rest of [`Struct::try_from_fields_with_attrs`], once the
/// fields are added.
fn finish_struct_type(
    ctx: &mut IrContext,
    name: StringRef,
    mut builder: TypeDataBuilder,
    mut attrs: AttributeMap,
) -> Result<Struct, StructTypeError> {
    attrs.view_mut(ctx).remove(PARAM_ATTRS_ATTR);
    attrs.view_mut(ctx).insert(STRUCT_NAME_ATTR, name);
    for (key, value) in attrs {
        builder = builder.attr(key, value);
    }
    let data = builder.build(ctx);
    if data.attrs.view(ctx).contains_key("fields") {
        return Err(StructTypeError::FieldsAttribute);
    }
    let mut seen: smallvec::SmallVec<[StringRef; 8]> = smallvec::SmallVec::new();
    for index in 0..data.params.len() {
        let field = field_name(ctx, data.param_attrs(ctx, index));
        if seen.contains(&field) {
            return Err(StructTypeError::DuplicateFieldName(
                ctx.str(field).to_owned(),
            ));
        }
        seen.push(field);
    }
    Ok(Struct(ctx.intern_type(data)))
}

// === Textual syntax: `adt.struct<Name(field: type {attrs}, ...), {attrs}>` ===
//
// Names are bare identifiers, or quoted strings when they are not identifiers.

impl trunk_ir::asm_format::TypeAsmFormat for Struct {
    fn print(&self, h: &mut trunk_ir::printer::TypePrintHelper<'_, '_>) -> std::fmt::Result {
        write_struct_type(h, *self)
    }

    fn parse<'a>(
        input: &mut &'a str,
        name: &'a str,
    ) -> winnow::ModalResult<trunk_ir::parser::raw::RawType<'a>> {
        parse_struct_type(input, name)
    }
}

fn write_struct_type(
    h: &mut trunk_ir::printer::TypePrintHelper<'_, '_>,
    adt_struct: Struct,
) -> std::fmt::Result {
    use std::fmt::Write;

    let ctx = h.ctx();
    h.write_str("adt.struct<")?;
    h.write_name(adt_struct.name(ctx))?;
    h.write_char('(')?;
    for (index, (name, ty)) in adt_struct.fields(ctx).enumerate() {
        if index > 0 {
            h.write_str(", ")?;
        }
        h.write_name(name)?;
        h.write_str(": ")?;
        h.write_type(ty)?;
        let mut attrs = adt_struct.field_attrs(ctx, index).peekable();
        if attrs.peek().is_some() {
            h.write_char(' ')?;
            h.write_attr_dict(attrs)?;
        }
    }
    h.write_char(')')?;
    let mut attrs = adt_struct.extra_attrs(ctx).peekable();
    if attrs.peek().is_some() {
        h.write_str(", ")?;
        h.write_attr_dict(attrs)?;
    }
    h.write_char('>')
}

/// Parse the rest of `adt.struct<Name(field: type {attrs}, ...), {attrs}>`
/// after its opening bracket into the generic form: each field's name becomes
/// its parameter's `name` attribute, and the struct's name the type's.
///
/// The syntax owns both names, so `name` may not appear in either dictionary.
/// Anything else, including a generic spelling, backtracks to generic parsing.
fn parse_struct_type<'a>(
    input: &mut &'a str,
    name: &'a str,
) -> winnow::ModalResult<trunk_ir::parser::raw::RawType<'a>> {
    use trunk_ir::parser::raw::{
        RawAttribute, RawParam, RawType, name_token, raw_attr_dict, raw_param, ws,
    };
    use winnow::combinator::{delimited, opt, preceded, separated};
    use winnow::prelude::*;

    let backtrack = || winnow::error::ErrMode::Backtrack(winnow::error::ContextError::new());
    let struct_name = name_token.parse_next(input)?;
    ws.parse_next(input)?;
    let fields: Vec<(String, RawParam<'a>)> = delimited(
        ('(', ws),
        separated(
            0..,
            (ws, name_token, ws, ':', ws, raw_param, ws)
                .map(|(_, field, _, _, _, param, _)| (field, param)),
            ',',
        ),
        (ws, ')'),
    )
    .parse_next(input)?;
    ws.parse_next(input)?;
    let mut attrs = opt(preceded((',', ws), raw_attr_dict))
        .parse_next(input)?
        .unwrap_or_default();
    ws.parse_next(input)?;
    '>'.parse_next(input)?;

    let is_name = |key: &str| key == STRUCT_NAME_ATTR;
    if attrs.iter().any(|(key, _)| is_name(key)) {
        return Err(backtrack());
    }
    attrs.push((STRUCT_NAME_ATTR.into(), RawAttribute::String(struct_name)));
    let mut params = Vec::with_capacity(fields.len());
    for (field, mut param) in fields {
        if param.attrs.iter().any(|(key, _)| is_name(key)) {
            return Err(backtrack());
        }
        param
            .attrs
            .push((STRUCT_NAME_ATTR.into(), RawAttribute::String(field)));
        params.push(param);
    }
    Ok(RawType::Concrete {
        name,
        params,
        attrs,
    })
}

// === Nominal enum layout type ===

/// A malformed `adt.enum` type.
#[derive(Clone, Debug, PartialEq, Eq, derive_more::Display, derive_more::Error)]
pub enum EnumTypeError {
    #[display("missing `{STRUCT_NAME_ATTR}` string")]
    MissingName,
    #[display("variant {_0} is not an `adt.variant` with a `{STRUCT_NAME_ATTR}` string")]
    MalformedVariant(#[error(not(source))] usize),
    #[display("duplicate variant name {_0:?}")]
    DuplicateVariantName(#[error(not(source))] String),
    #[display("`variants` is not an `adt.enum` attribute; variants are type parameters")]
    VariantsAttribute,
    #[display("an `adt.enum` variant parameter carries no parameter attributes")]
    VariantAttributes,
}

impl trunk_ir::type_verifier::VerifyType for Enum {
    /// Validate a name-matching `adt.enum`.
    fn verify(ctx: &IrContext, ty: TypeRef) -> Result<Self, trunk_ir::ops::VerificationError> {
        let data = ctx.get_type(ty);
        debug_assert!(Self::matches(ctx, ty));
        let attrs = ctx.type_attrs(ty);
        if attrs.contains_key("variants") {
            return Err(EnumTypeError::VariantsAttribute.into());
        }
        // The syntax has no place for them, so printing would drop them.
        if attrs.contains_key(PARAM_ATTRS_ATTR) {
            return Err(EnumTypeError::VariantAttributes.into());
        }
        if attrs.get_string_ref(STRUCT_NAME_ATTR).is_none() {
            return Err(EnumTypeError::MissingName.into());
        }
        let mut names: smallvec::SmallVec<[StringRef; 8]> = smallvec::SmallVec::new();
        for (index, &variant) in data.params.iter().enumerate() {
            let name = variant_name(ctx, variant).ok_or(EnumTypeError::MalformedVariant(index))?;
            if names.contains(&name) {
                return Err(EnumTypeError::DuplicateVariantName(ctx.str(name).to_owned()).into());
            }
            names.push(name);
        }
        Ok(Self(ty))
    }
}

impl Enum {
    pub fn variant_count(self, ctx: &IrContext) -> usize {
        ctx.get_type(self.0).params.len()
    }

    /// Each variant's name and field types, in declaration order.
    pub fn variants(
        self,
        ctx: &IrContext,
    ) -> impl ExactSizeIterator<Item = (StringRef, &[TypeRef])> + '_ {
        ctx.get_type(self.0).params.iter().map(|&variant| {
            let name = variant_name(ctx, variant).expect("validated adt.enum variant");
            (name, &ctx.get_type(variant).params[..])
        })
    }

    /// The field types of the variant named `tag`.
    pub fn variant_fields(self, ctx: &IrContext, tag: StringRef) -> Option<&[TypeRef]> {
        self.variants(ctx)
            .find_map(|(name, fields)| (name == tag).then_some(fields))
    }

    /// The type attributes other than the name.
    pub fn extra_attrs(
        self,
        ctx: &IrContext,
    ) -> impl Iterator<Item = (StringRef, &Attribute)> + '_ {
        ctx.get_type(self.0)
            .attrs
            .iter()
            .filter(|&(key, _)| ctx.str(key) != STRUCT_NAME_ATTR)
    }
}

/// The name of an `adt.variant` type.
fn variant_name(ctx: &IrContext, variant: TypeRef) -> Option<StringRef> {
    let data = ctx.get_type(variant);
    (data.name == Variant::NAME)
        .then(|| ctx.type_attrs(variant).get_string_ref(STRUCT_NAME_ATTR))
        .flatten()
}

impl Enum {
    /// Construct an `adt.enum` named `name` with `variants` in declaration
    /// order, each a name and its field types.
    ///
    /// `attrs` holds the remaining type attributes.
    ///
    /// # Panics
    ///
    /// If two variants have the same name, or `attrs` has a `variants` or
    /// parameter attribute entry.
    pub fn from_variants<N: Into<StringArg>, F: IntoIterator<Item = TypeRef>>(
        ctx: &mut IrContext,
        name: impl Into<StringArg>,
        variants: impl IntoIterator<Item = (N, F)>,
        mut attrs: AttributeMap,
    ) -> Self {
        let mut builder = TypeDataBuilder::new(<Enum as DialectType>::NAME);
        for (variant, fields) in variants {
            let variant = Variant::params(fields).name(variant).intern(ctx);
            builder = builder.param(variant.as_type_ref());
        }
        let name = ctx.intern_string_arg(name.into());
        attrs.view_mut(ctx).insert(STRUCT_NAME_ATTR, name);
        for (key, value) in attrs {
            builder = builder.attr(key, value);
        }
        let ty = builder.intern(ctx);
        <Enum as trunk_ir::type_verifier::VerifyType>::verify(ctx, ty)
            .unwrap_or_else(|error| panic!("adt.enum: {error}"))
    }
}

// === Textual syntax: `adt.enum<Name { Variant(field: type, ...), ... }, {attrs}>` ===
//
// A field is `name: type` or, without a name, `type`. Either may be followed
// by the field's other attributes.

impl trunk_ir::asm_format::TypeAsmFormat for Enum {
    fn print(&self, h: &mut trunk_ir::printer::TypePrintHelper<'_, '_>) -> std::fmt::Result {
        write_enum_type(h, *self)
    }

    fn parse<'a>(
        input: &mut &'a str,
        name: &'a str,
    ) -> winnow::ModalResult<trunk_ir::parser::raw::RawType<'a>> {
        parse_enum_type(input, name)
    }
}

fn write_enum_type(
    h: &mut trunk_ir::printer::TypePrintHelper<'_, '_>,
    adt_enum: Enum,
) -> std::fmt::Result {
    use std::fmt::Write;

    let ctx = h.ctx();
    let data = ctx.get_type(adt_enum.as_type_ref());
    h.write_str("adt.enum<")?;
    h.write_name(adt_enum.name(ctx))?;
    h.write_str(" {")?;
    for (index, &variant) in data.params.iter().enumerate() {
        h.write_str(if index > 0 { ", " } else { " " })?;
        let variant_data = ctx.get_type(variant);
        let name = variant_name(ctx, variant).expect("validated adt.enum variant");
        h.write_name(ctx.str(name))?;
        h.write_char('(')?;
        for (field, &field_ty) in variant_data.params.iter().enumerate() {
            if field > 0 {
                h.write_str(", ")?;
            }
            let field_attrs = variant_data.param_attrs(ctx, field);
            if let Some(field_name) = field_attrs.view(ctx).get_string_ref(STRUCT_NAME_ATTR) {
                h.write_name(ctx.str(field_name))?;
                h.write_str(": ")?;
            }
            h.write_type(field_ty)?;
            let mut attrs = field_attrs
                .iter()
                .filter(|&(key, _)| ctx.str(key) != STRUCT_NAME_ATTR)
                .peekable();
            if attrs.peek().is_some() {
                h.write_char(' ')?;
                h.write_attr_dict(attrs)?;
            }
        }
        h.write_char(')')?;
    }
    h.write_str(if data.params.is_empty() { "}" } else { " }" })?;
    let mut attrs = adt_enum.extra_attrs(ctx).peekable();
    if attrs.peek().is_some() {
        h.write_str(", ")?;
        h.write_attr_dict(attrs)?;
    }
    h.write_char('>')
}

/// Parse the rest of `adt.enum<Name { Variant(field: type, ...), ... }, {attrs}>`
/// after its opening bracket into the generic form: each variant becomes an
/// `adt.variant` parameter named by its `name` attribute.
///
/// The syntax owns the names, so `name` may not appear in a dictionary.
/// Anything else, including a generic spelling, backtracks to generic parsing.
fn parse_enum_type<'a>(
    input: &mut &'a str,
    name: &'a str,
) -> winnow::ModalResult<trunk_ir::parser::raw::RawType<'a>> {
    use trunk_ir::parser::raw::{
        RawAttribute, RawParam, RawType, name_token, raw_attr_dict, raw_param, ws,
    };
    use winnow::combinator::{delimited, opt, preceded, separated, terminated};
    use winnow::prelude::*;

    type Field<'a> = (Option<String>, RawParam<'a>);

    let backtrack = || winnow::error::ErrMode::Backtrack(winnow::error::ContextError::new());
    let field = |input: &mut &'a str| -> winnow::ModalResult<Field<'a>> {
        (
            ws,
            opt(terminated(name_token, (ws, ':', ws))),
            raw_param,
            ws,
        )
            .map(|(_, field, param, _)| (field, param))
            .parse_next(input)
    };
    let variant = |input: &mut &'a str| -> winnow::ModalResult<(String, Vec<Field<'a>>)> {
        (
            ws,
            name_token,
            ws,
            delimited('(', separated(0.., field, ','), (ws, ')')),
            ws,
        )
            .map(|(_, variant, _, fields, _)| (variant, fields))
            .parse_next(input)
    };

    let enum_name = name_token.parse_next(input)?;
    ws.parse_next(input)?;
    let variants: Vec<(String, Vec<Field<'a>>)> =
        delimited('{', separated(0.., variant, ','), (ws, '}')).parse_next(input)?;
    ws.parse_next(input)?;
    let mut attrs = opt(preceded((',', ws), raw_attr_dict))
        .parse_next(input)?
        .unwrap_or_default();
    ws.parse_next(input)?;
    '>'.parse_next(input)?;

    let is_name = |key: &str| key == STRUCT_NAME_ATTR;
    if attrs.iter().any(|(key, _)| is_name(key)) {
        return Err(backtrack());
    }
    attrs.push((STRUCT_NAME_ATTR.into(), RawAttribute::String(enum_name)));
    let mut params = Vec::with_capacity(variants.len());
    for (variant, fields) in variants {
        let mut field_params = Vec::with_capacity(fields.len());
        for (field, mut param) in fields {
            if param.attrs.iter().any(|(key, _)| is_name(key)) {
                return Err(backtrack());
            }
            if let Some(field) = field {
                param
                    .attrs
                    .push((STRUCT_NAME_ATTR.into(), RawAttribute::String(field)));
            }
            field_params.push(param);
        }
        params.push(RawParam::from(RawType::Concrete {
            name: "adt.variant",
            params: field_params,
            attrs: vec![(STRUCT_NAME_ATTR.into(), RawAttribute::String(variant))],
        }));
    }
    Ok(RawType::Concrete {
        name,
        params,
        attrs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustc_hash::FxHashMap as HashMap;
    use trunk_ir::parser::{parse_module, parse_test_module};
    use trunk_ir::printer::print_module;
    use trunk_ir::rewrite::Module;
    use trunk_ir::validation::validate_operation_verifiers;

    /// Print `module`, parse the output, and check that it prints the same.
    fn assert_roundtrip(ctx: &IrContext, module: trunk_ir::refs::OpRef) {
        let printed = print_module(ctx, module);
        let mut reparsed = IrContext::new();
        let module = parse_module(&mut reparsed, &printed).expect("printed IR must parse");
        assert_eq!(printed, print_module(&reparsed, module));
    }

    fn array_schema_ok(body: &str) -> bool {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            &format!(
                "core.module @test {{\n  !A = adt.array<core.i32>\n  !B = adt.array<core.i64>\n  func.func @f(%a: !A, %i: core.index, %n: core.i32, %w: core.i64) -> core.nil {{\n    {body}\n    func.return\n  }}\n}}"
            ),
        );
        trunk_ir::validation::validate_op_schemas(&ctx, module.op()).is_ok()
    }

    #[test]
    fn array_operations_reject_the_unmanaged_core_array() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            "core.module @test {\n  !L = core.array<core.i32>\n  func.func @f(%l: !L, %i: core.index, %n: core.i32) -> core.nil {\n    %len = adt.array_len %l : core.index\n    %v = adt.array_new %i, %n {type = !L} : !L\n    func.return\n  }\n}",
        );
        let result = trunk_ir::validation::validate_op_schemas(&ctx, module.op());
        // `array_len` rejects its operand; `array_new` its type and result.
        assert_eq!(result.errors.len(), 3, "{result}");
    }

    #[test]
    fn array_construction_binds_the_element_type_and_index_operands() {
        assert!(array_schema_ok(
            "%v = adt.array_new %i, %n {type = !A} : !A"
        ));
        assert!(!array_schema_ok(
            "%v = adt.array_new %n, %n {type = !A} : !A"
        ));
        assert!(!array_schema_ok(
            "%v = adt.array_new %i, %w {type = !A} : !A"
        ));
        assert!(!array_schema_ok(
            "%v = adt.array_new %i, %n {type = !A} : !B"
        ));
        assert!(array_schema_ok("%v = adt.array_of {type = !A} : !A"));
        assert!(array_schema_ok("%v = adt.array_of %n, %n {type = !A} : !A"));
        assert!(!array_schema_ok(
            "%v = adt.array_of %n, %w {type = !A} : !A"
        ));
        assert!(array_schema_ok(
            "%v = adt.array_splice %a, %i, %i, %n {type = !A} : !A"
        ));
        assert!(array_schema_ok(
            "%v = adt.array_splice %a, %i, %i {type = !A} : !A"
        ));
        assert!(!array_schema_ok(
            "%v = adt.array_splice %a, %n, %i {type = !A} : !A"
        ));
        assert!(!array_schema_ok(
            "%v = adt.array_splice %a, %i, %i, %w {type = !A} : !A"
        ));
        assert!(!array_schema_ok(
            "%v = adt.array_splice %a, %i, %i {type = !B} : !B"
        ));
    }

    #[test]
    fn named_types_suggest_their_name_as_alias() {
        use trunk_ir::TypeDataBuilder;
        use trunk_ir::asm_format::suggest_type_alias_name;

        let mut ctx = IrContext::new();
        let i32_ty = trunk_ir::dialect::core::I32::type_ref(&mut ctx);
        let named = |ctx: &mut IrContext, name: trunk_ir::TypeName, alias: &str| {
            let alias = ctx.string_attr(alias);
            TypeDataBuilder::new(name).attr("name", alias).intern(ctx)
        };
        let types = [
            named(&mut ctx, Struct::NAME, "Point"),
            named(&mut ctx, Enum::NAME, "Option"),
            named(&mut ctx, Typeref::NAME, "Point"),
            named(&mut ctx, Variant::NAME, "Some"),
        ];
        let names = types.map(|ty| suggest_type_alias_name(&ctx, ty));
        assert_eq!(
            names,
            [Some("Point"), Some("Option"), Some("Point"), Some("Some")]
        );
        assert_eq!(suggest_type_alias_name(&ctx, i32_ty), None);
    }

    #[test]
    fn string_attribute_accessors_return_text_and_handle() {
        use trunk_ir::ops::DialectOp;

        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  %0 = adt.string_const {value = "hi"} : core.ptr
}"#,
        );
        let op = module.ops(&ctx)[0];
        let string_const = StringConst::from_op(&ctx, op).unwrap();
        assert_eq!(string_const.value(&ctx), "hi");
        assert_eq!(string_const.value_ref(&ctx), ctx.lookup_str("hi").unwrap());

        let loc = ctx.op(op).location;
        let ptr = ctx.op_result_types(op)[0];
        let copy = StringConst::operands()
            .value(string_const.value_ref(&ctx))
            .results(ptr)
            .build(&mut ctx, loc);
        assert_eq!(copy.value(&ctx), "hi");
    }

    #[test]
    fn test_roundtrip_adt_struct() {
        let input = r#"core.module @test {
  !point = adt.struct<Point(x: core.i32, y: core.i32 {k = @v})>
  !closure = adt.struct<"Nested::Closure"(func_ptr: func.func_sig<(core.i32) -> core.i32, {k = 1}> {}, env: core.tuple<core.ptr>), {layout = "closure"}>
  !empty = adt.struct<Empty()>
  !nested = adt.struct<Outer(inner: adt.struct<Inner(a: core.i32), {layout = "closure"}> {m = 1})>
}"#;
        let mut ctx = IrContext::new();
        let module = parse_module(&mut ctx, input).expect("adt.struct syntax should parse");
        let aliases: HashMap<_, _> = ctx
            .type_aliases()
            .iter()
            .map(|(name, ty)| (name.to_string(), *ty))
            .collect();
        let point = Struct::from_type_ref(&ctx, aliases["point"]).unwrap();
        assert_eq!(point.name(&ctx), "Point");
        itertools::assert_equal(point.fields(&ctx).map(|(name, _)| name), ["x", "y"]);
        assert_eq!(point.field_attrs(&ctx, 1).count(), 1);
        let empty = Struct::from_type_ref(&ctx, aliases["empty"]).unwrap();
        assert_eq!(empty.field_count(&ctx), 0);
        let printed = print_module(&ctx, module);
        for expected in [
            "!point = adt.struct<Point(x: core.i32, y: core.i32 {k = @v})>",
            "adt.struct<\"Nested::Closure\"(func_ptr: func.func_sig<(core.i32) -> core.i32, {k = 1}>, env: core.tuple<core.ptr>), {layout = \"closure\"}>",
            "!empty = adt.struct<Empty()>",
            "(inner: adt.struct<Inner(a: core.i32), {layout = \"closure\"}> {m = 1})>",
        ] {
            assert!(printed.contains(expected), "{expected}\n{printed}");
        }
        assert_roundtrip(&ctx, module);
    }

    #[test]
    fn repeated_struct_is_aliased_by_its_name() {
        let input = r#"core.module @test {
  func.func @f1(%x: adt.struct<_Marker(a: core.i32)>) -> adt.struct<_Marker(a: core.i32)> {
    func.return %x
  }
  func.func @f2(%x: adt.struct<_Marker(a: core.i32)>) -> adt.struct<_Marker(a: core.i32)> {
    func.return %x
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_module(&mut ctx, input).expect("struct types should parse");
        let printed = print_module(&ctx, module);
        assert!(
            printed.contains("!_Marker = adt.struct<_Marker(a: core.i32)>"),
            "{printed}"
        );
        assert!(
            printed.contains("func.func @f1(%0: !_Marker) -> !_Marker"),
            "{printed}"
        );
        assert_roundtrip(&ctx, module);
    }

    #[test]
    fn test_roundtrip_adt_enum() {
        let input = r#"core.module @test {
  !option = adt.enum<Option { None(), Some(value: core.i32) }>
  !shape = adt.enum<"geo::Shape" { Dot(), Rect(core.f64, core.f64 {k = 1}), Named(label: core.ptr {k = 2}) }, {layout = "x"}>
  !never = adt.enum<Never {}>
  !list = adt.enum<List { Empty(), Cons(core.i32, adt.typeref<{name = "List"}>) }>
}"#;
        let mut ctx = IrContext::new();
        let module = parse_module(&mut ctx, input).expect("adt.enum syntax should parse");
        let aliases: HashMap<_, _> = ctx
            .type_aliases()
            .iter()
            .map(|(name, ty)| (name.to_string(), *ty))
            .collect();
        let i32_ty = TypeDataBuilder::new("core.i32").intern(&mut ctx);

        let option = Enum::from_type_ref(&ctx, aliases["option"]).unwrap();
        assert_eq!(option.name(&ctx), "Option");
        let variants: Vec<_> = option
            .variants(&ctx)
            .map(|(name, fields)| (ctx.str(name), fields.to_vec()))
            .collect();
        assert_eq!(variants, [("None", vec![]), ("Some", vec![i32_ty])]);
        let some = ctx.intern_str("Some");
        assert_eq!(option.variant_fields(&ctx, some), Some(&[i32_ty][..]));
        let missing = ctx.intern_str("Missing");
        assert_eq!(option.variant_fields(&ctx, missing), None);
        // A variant is one type parameter, so type traversal reaches fields.
        assert_eq!(ctx.get_type(aliases["option"]).params.len(), 2);

        let never = Enum::from_type_ref(&ctx, aliases["never"]).unwrap();
        assert_eq!(never.variant_count(&ctx), 0);

        // The constructor builds the type the syntax denotes, apart from the
        // field name the syntax adds.
        let built = Enum::from_variants(
            &mut ctx,
            "List",
            [("Empty", vec![]), ("Cons", vec![i32_ty])],
            AttributeMap::new(),
        );
        assert_eq!(built.variant_count(&ctx), 2);

        let printed = print_module(&ctx, module);
        for expected in [
            "!option = adt.enum<Option { None(), Some(value: core.i32) }>",
            "adt.enum<\"geo::Shape\" { Dot(), Rect(core.f64, core.f64 {k = 1}), Named(label: core.ptr {k = 2}) }, {layout = \"x\"}>",
            "!never = adt.enum<Never {}>",
            "adt.enum<List { Empty(), Cons(core.i32, adt.typeref<{name = \"List\"}>) }>",
        ] {
            assert!(printed.contains(expected), "{expected}\n{printed}");
        }
        assert_roundtrip(&ctx, module);
    }

    #[test]
    fn test_adt_enum_reserved_names_are_parse_errors() {
        for spelling in [
            "adt.enum<E { A() }, {name = @Q}>",
            "adt.enum<E { A(core.i32 {name = @x}) }>",
        ] {
            let mut ctx = IrContext::new();
            let input = format!("core.module @test {{ !bad = {spelling} }}");
            parse_module(&mut ctx, &input).expect_err(spelling);
        }
    }

    #[test]
    fn test_malformed_adt_enum_parses_but_fails_validation() {
        for spelling in [
            "adt.enum<E { A(), A(core.i32) }>",
            "adt.enum<{name = \"E\", variants = [[\"A\", []]]}>",
            "adt.enum<core.i32, {name = \"E\"}>",
            "adt.enum<adt.variant<core.i32>, {name = \"E\"}>",
            "adt.enum<adt.variant<{name = \"A\"}>>",
            "adt.enum<adt.variant<{name = \"A\"}>, {name = @E}>",
            // The syntax cannot print a variant parameter's attributes.
            "adt.enum<adt.variant<{name = \"A\"}> {k = 1}, {name = \"E\"}>",
        ] {
            let mut ctx = IrContext::new();
            let input = format!("core.module @test {{ !bad = {spelling} }}");
            let module = parse_module(&mut ctx, &input).expect(spelling);
            let alias = ctx.type_aliases()[0].1;
            assert!(Enum::from_type_ref(&ctx, alias).is_none(), "{spelling}");
            let result = validate_operation_verifiers(&ctx, Module::new(&ctx, module).unwrap());
            assert!(!result.is_ok(), "{spelling}");
        }
    }

    #[test]
    fn test_adt_struct_reserved_names_are_parse_errors() {
        for spelling in [
            "adt.struct<P(x: core.i32), {name = @Q}>",
            "adt.struct<P(x: core.i32), {param_attrs = [{}]}>",
            "adt.struct<P(x: core.i32 {name = @y})>",
        ] {
            let mut ctx = IrContext::new();
            let input = format!("core.module @test {{ !bad = {spelling} }}");
            parse_module(&mut ctx, &input).expect_err(spelling);
        }
    }

    #[test]
    fn test_malformed_adt_struct_parses_but_fails_validation() {
        for spelling in [
            "adt.struct<P(x: core.i32, x: core.i64)>",
            "adt.struct<P(), {fields = []}>",
            "adt.struct<core.i32, {name = @P}>",
        ] {
            let mut ctx = IrContext::new();
            let input = format!("core.module @test {{ !bad = {spelling} }}");
            let module = parse_module(&mut ctx, &input).expect(spelling);
            let alias = ctx.type_aliases()[0].1;
            assert!(Struct::from_type_ref(&ctx, alias).is_none(), "{spelling}");
            let result = validate_operation_verifiers(&ctx, Module::new(&ctx, module).unwrap());
            assert!(!result.is_ok(), "{spelling}");
        }
    }

    #[test]
    fn malformed_adt_struct_types_are_rejected_by_type_validation() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, "core.module @test {}");
        let i32_ty = TypeDataBuilder::new("core.i32").intern(&mut ctx);
        let x = ctx.string_attr("x");
        let p = ctx.string_attr("P");
        let named_field = AttributeMap::from_entries(&mut ctx, [("name", x.clone())]);
        let cases = [
            (
                "missing `name` string",
                TypeDataBuilder::new("adt.struct")
                    .param_with_attrs(i32_ty, named_field.clone())
                    .build(&mut ctx),
            ),
            (
                "field 1 has no `name` string",
                TypeDataBuilder::new("adt.struct")
                    .param_with_attrs(i32_ty, named_field.clone())
                    .param(i32_ty)
                    .attr("name", p.clone())
                    .build(&mut ctx),
            ),
            (
                "duplicate field name \"x\"",
                TypeDataBuilder::new("adt.struct")
                    .param_with_attrs(i32_ty, named_field.clone())
                    .param_with_attrs(i32_ty, named_field.clone())
                    .attr("name", p.clone())
                    .build(&mut ctx),
            ),
            (
                "`fields` is not an `adt.struct` attribute",
                TypeDataBuilder::new("adt.struct")
                    .attr("name", p)
                    .attr("fields", Attribute::List(vec![]))
                    .build(&mut ctx),
            ),
        ];
        for (expected, data) in cases {
            let ty = ctx.intern_type(data);
            assert!(Struct::from_type_ref(&ctx, ty).is_none(), "{expected}");
            let result = validate_operation_verifiers(&ctx, module);
            let text = result.to_string();
            assert!(
                text.contains("adt.struct") && text.contains(expected),
                "{expected}: {text}"
            );
        }
        let valid = Struct::from_fields(&mut ctx, "Q", [("x", i32_ty)], AttributeMap::new());
        assert!(
            ctx.type_attrs(valid.as_type_ref())
                .contains_key(PARAM_ATTRS_ATTR)
        );
    }
}

#[cfg(test)]
mod canonicalize_tests {
    use super::ForwardStructGets;
    use trunk_ir::IrContext;
    use trunk_ir::dialect::{core, func};
    use trunk_ir::ops::DialectOp;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::pass::PassManager;
    use trunk_ir::printer::print_module;
    use trunk_ir::rewrite::Module;
    use trunk_ir::transforms::Canonicalize;

    fn count(ctx: &IrContext, module: Module, name: &str) -> usize {
        print_module(ctx, module.op()).matches(name).count()
    }

    /// Forward struct reads and canonicalize; the flag tells whether the IR
    /// changed.
    fn fold(input: &str) -> (IrContext, Module, bool) {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        let before = print_module(&ctx, module.op());
        let root = core::Module::from_op(&ctx, module.op()).expect("core.module");
        let mut pm = PassManager::new();
        pm.add_pass(ForwardStructGets)
            .nest::<func::Func>()
            .add_pass(Canonicalize);
        pm.run(&mut ctx, root, &mut Default::default())
            .expect("fold passes");
        let changed = print_module(&ctx, module.op()) != before;
        (ctx, module, changed)
    }

    const PAIR: &str = r#"core.module @test {
  !pair = adt.struct<Pair(a: core.i32, b: core.i64)>
  func.func @f(%x: core.i32, %y: core.i64) -> core.i64 {
    %p = adt.struct_new %x, %y {type = !pair} : !pair
    %r = adt.struct_get %p {type = !pair, field = 1} : core.i64
    func.return %r
  }
}"#;

    #[test]
    fn struct_get_of_struct_new_forwards_the_field() {
        let (ctx, module, changed) = fold(PAIR);
        assert!(changed);
        assert_eq!(count(&ctx, module, "adt.struct_get"), 0);
        assert!(print_module(&ctx, module.op()).contains("func.return %1"));
    }

    #[test]
    fn struct_get_of_a_block_argument_is_not_folded() {
        let (ctx, module, changed) = fold(
            r#"core.module @test {
  !pair = adt.struct<Pair(a: core.i32, b: core.i64)>
  func.func @f(%p: !pair) -> core.i64 {
    %r = adt.struct_get %p {type = !pair, field = 1} : core.i64
    func.return %r
  }
}"#,
        );
        assert!(!changed);
        assert_eq!(count(&ctx, module, "adt.struct_get"), 1);
    }

    #[test]
    fn struct_get_with_a_different_layout_type_is_not_folded() {
        let (ctx, module, changed) = fold(
            r#"core.module @test {
  !pair = adt.struct<Pair(a: core.i32, b: core.i64)>
  !other = adt.struct<Other(a: core.i32, b: core.i64)>
  func.func @f(%x: core.i32, %y: core.i64) -> core.i64 {
    %p = adt.struct_new %x, %y {type = !pair} : !pair
    %r = adt.struct_get %p {type = !other, field = 1} : core.i64
    func.return %r
  }
}"#,
        );
        assert!(!changed);
        assert_eq!(count(&ctx, module, "adt.struct_get"), 1);
    }

    #[test]
    fn struct_get_with_a_mismatched_result_type_is_not_folded() {
        let (ctx, module, changed) = fold(
            r#"core.module @test {
  !pair = adt.struct<Pair(a: core.i32, b: core.i64)>
  func.func @f(%x: core.i32, %y: core.i64) -> core.i32 {
    %p = adt.struct_new %x, %y {type = !pair} : !pair
    %r = adt.struct_get %p {type = !pair, field = 1} : core.i32
    func.return %r
  }
}"#,
        );
        assert!(!changed);
        assert_eq!(count(&ctx, module, "adt.struct_get"), 1);
    }

    #[test]
    fn struct_get_out_of_range_field_is_not_folded() {
        let (ctx, module, changed) = fold(
            r#"core.module @test {
  !pair = adt.struct<Pair(a: core.i32, b: core.i64)>
  func.func @f(%x: core.i32, %y: core.i64) -> core.i64 {
    %p = adt.struct_new %x, %y {type = !pair} : !pair
    %r = adt.struct_get %p {type = !pair, field = 2} : core.i64
    func.return %r
  }
}"#,
        );
        assert!(!changed);
        assert_eq!(count(&ctx, module, "adt.struct_get"), 1);
    }

    #[test]
    fn struct_get_of_a_struct_mutated_through_an_alias_is_not_folded() {
        let (ctx, module, changed) = fold(
            r#"core.module @test {
  !pair = adt.struct<Pair(a: core.i32, b: core.i64)>
  func.func @f(%x: core.i32, %y: core.i64, %z: core.i64) -> core.i64 {
    %p = adt.struct_new %x, %y {type = !pair} : !pair
    %any = adt.ref_cast %p {type = tribute_rt.anyref} : tribute_rt.anyref
    %alias = adt.ref_cast %any {type = !pair} : !pair
    adt.struct_set %alias, %z {type = !pair, field = 1}
    %r = adt.struct_get %p {type = !pair, field = 1} : core.i64
    func.return %r
  }
}"#,
        );
        assert!(!changed);
        assert_eq!(count(&ctx, module, "adt.struct_get"), 1);
    }

    #[test]
    fn struct_get_is_not_folded_when_another_function_writes_the_layout() {
        let (ctx, module, changed) = fold(
            r#"core.module @test {
  !pair = adt.struct<Pair(a: core.i32, b: core.i64)>
  func.func @write(%p: !pair, %z: core.i64) {
    adt.struct_set %p, %z {type = !pair, field = 1}
    func.return
  }
  func.func @f(%x: core.i32, %y: core.i64) -> core.i64 {
    %p = adt.struct_new %x, %y {type = !pair} : !pair
    %r = adt.struct_get %p {type = !pair, field = 1} : core.i64
    func.return %r
  }
}"#,
        );
        assert!(!changed);
        assert_eq!(count(&ctx, module, "adt.struct_get"), 1);
    }

    #[test]
    fn struct_get_of_a_mutated_struct_is_not_folded() {
        let (ctx, module, changed) = fold(
            r#"core.module @test {
  !pair = adt.struct<Pair(a: core.i32, b: core.i64)>
  func.func @f(%x: core.i32, %y: core.i64, %z: core.i64) -> core.i64 {
    %p = adt.struct_new %x, %y {type = !pair} : !pair
    adt.struct_set %p, %z {type = !pair, field = 1}
    %r = adt.struct_get %p {type = !pair, field = 1} : core.i64
    func.return %r
  }
}"#,
        );
        assert!(!changed);
        assert_eq!(count(&ctx, module, "adt.struct_get"), 1);
    }

    #[test]
    fn struct_get_of_a_struct_with_explicit_ownership_is_not_folded() {
        let (ctx, module, changed) = fold(
            r#"core.module @test {
  !pair = adt.struct<Pair(a: core.i32, b: core.i64)>
  func.func @f(%x: core.i32, %y: core.i64) -> core.i64 {
    %p = adt.struct_new %x, %y {type = !pair} : !pair
    %r = adt.struct_get %p {type = !pair, field = 1} : core.i64
    tribute_rt.release %p {alloc_size = 24}
    func.return %r
  }
}"#,
        );
        assert!(!changed);
        assert_eq!(count(&ctx, module, "adt.struct_get"), 1);
    }

    #[test]
    fn ref_cast_to_the_operand_type_is_forwarded() {
        let (ctx, module, changed) = fold(
            r#"core.module @test {
  !pair = adt.struct<Pair(a: core.i32, b: core.i64)>
  func.func @f(%p: !pair) -> !pair {
    %r = adt.ref_cast %p {type = !pair} : !pair
    func.return %r
  }
}"#,
        );
        assert!(changed);
        assert_eq!(count(&ctx, module, "adt.ref_cast"), 0);
    }

    #[test]
    fn ref_cast_to_another_type_is_not_folded() {
        let (ctx, module, changed) = fold(
            r#"core.module @test {
  !pair = adt.struct<Pair(a: core.i32, b: core.i64)>
  func.func @f(%p: !pair) -> tribute_rt.anyref {
    %r = adt.ref_cast %p {type = tribute_rt.anyref} : tribute_rt.anyref
    func.return %r
  }
}"#,
        );
        assert!(!changed);
        assert_eq!(count(&ctx, module, "adt.ref_cast"), 1);
    }
}
