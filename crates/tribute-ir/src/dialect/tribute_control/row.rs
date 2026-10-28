//! Ability rows of source-logical callable types, the row binders a definition
//! declares, and the substitution a reference applies to its callee's type.

use rustc_hash::FxHashMap as HashMap;
use trunk_ir::ops::DialectType;
use trunk_ir::types::{Attribute, AttributeMap, TypeData};
use trunk_ir::{IrContext, Symbol, SymbolPath, TypeRef};

use crate::dialect::adt;

/// Type attribute of `tribute_control.func_sig` holding its ability row.
pub const ABILITY_ROW_ATTR: &str = "ability_row";
/// `tribute_control.func` attribute declaring its row binders.
pub const ROW_BINDERS_ATTR: &str = "row_binders";
/// `tribute_control.func` attribute declaring its row tail unions.
pub const ROW_UNIONS_ATTR: &str = "row_unions";
/// `tribute_control.func` attribute marking a definition whose row was
/// inferred rather than annotated.
pub const INFERRED_ROW_ATTR: &str = "inferred_row";
/// Attribute of a reference holding one row for each quantified binder of
/// its callee.
pub const ROW_ARGS_ATTR: &str = "row_args";
/// `tribute_control.lambda` attribute listing the local binders it
/// quantifies, in instantiation order.
pub const QUANTIFIED_ROWS_ATTR: &str = "quantified_rows";

const LOGICAL_TUPLE_PREFIX: &str = "__logical_tuple_";

/// An ability row: ability instances, sorted by path with repetitions kept,
/// and an optional tail naming a row binder of the enclosing definition.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Row {
    pub instances: Vec<SymbolPath>,
    pub tail: Option<u32>,
}

impl Row {
    /// A row of `instances`, in canonical order, over `tail`.
    pub fn new(instances: impl IntoIterator<Item = SymbolPath>, tail: Option<u32>) -> Self {
        let mut instances: Vec<_> = instances.into_iter().collect();
        instances.sort();
        Self { instances, tail }
    }

    /// Whether this is the closed empty row.
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty() && self.tail.is_none()
    }

    /// Decode a row, or say why it is malformed.
    pub fn from_attribute(ctx: &IrContext, attr: &Attribute) -> Result<Self, String> {
        let Attribute::Dict(entries) = attr else {
            return Err("a row must be a dictionary".into());
        };
        let mut row = Self::default();
        for (key, value) in entries.view(ctx).iter() {
            (match (key, value) {
                ("instances", Attribute::List(instances)) => {
                    row.instances = instances
                        .iter()
                        .map(|instance| match instance {
                            Attribute::SymbolRef(path) => Ok(path.clone()),
                            _ => Err("row instances must name ability instances".to_owned()),
                        })
                        .collect::<Result<_, _>>()?;
                    Ok(())
                }
                ("tail", Attribute::Int(tail)) => {
                    row.tail =
                        Some(u32::try_from(*tail).map_err(|_| "row tail must be a binder index")?);
                    Ok(())
                }
                ("instances" | "tail", _) => Err(format!("row {key} is malformed")),
                _ => Err(format!("a row has no entry '{key}'")),
            })?;
        }
        if !row.instances.is_sorted() {
            return Err("row instances must be sorted by path".into());
        }
        Ok(row)
    }

    pub fn to_attribute(&self, ctx: &mut IrContext) -> Attribute {
        let (instances, tail) = self.entries();
        let entries = instances
            .map(|value| ("instances", value))
            .into_iter()
            .chain(tail.map(|value| ("tail", value)));
        Attribute::Dict(AttributeMap::from_entries(ctx, entries))
    }

    /// [`Self::to_attribute`] without interning, `None` when no attribute of
    /// the context can hold this row.
    pub fn find_attribute(&self, ctx: &IrContext) -> Option<Attribute> {
        let (instances, tail) = self.entries();
        let instances = match instances {
            Some(value) => Some((ctx.lookup_str("instances")?, value)),
            None => None,
        };
        let tail = match tail {
            Some(value) => Some((ctx.lookup_str("tail")?, value)),
            None => None,
        };
        Some(Attribute::Dict(AttributeMap::from_refs(
            ctx,
            instances.into_iter().chain(tail),
        )))
    }

    fn entries(&self) -> (Option<Attribute>, Option<Attribute>) {
        let instances = (!self.instances.is_empty()).then(|| {
            Attribute::List(
                self.instances
                    .iter()
                    .cloned()
                    .map(Attribute::SymbolRef)
                    .collect(),
            )
        });
        (
            instances,
            self.tail.map(|tail| Attribute::Int(i128::from(tail))),
        )
    }

    /// Whether a callable of this row may stand for one of `other`: it names
    /// every instance `other` names, as often, over the same tail if
    /// `other` has one. Both rows' instances are in canonical order.
    pub fn includes(&self, other: &Row) -> bool {
        if other.tail.is_some() && other.tail != self.tail {
            return false;
        }
        let mut remaining = self.instances.iter();
        other
            .instances
            .iter()
            .all(|instance| remaining.any(|candidate| candidate == instance))
    }

    /// This row with its tail replaced by the row argument it names, or
    /// `None` when the tail names no argument.
    pub fn substitute<'a>(&self, args: impl Into<RowArgs<'a>>) -> Option<Self> {
        let Some(tail) = self.tail else {
            return Some(self.clone());
        };
        let Some(arg) = args.into().get(tail)? else {
            return Some(self.clone());
        };
        Some(Self::new(
            self.instances.iter().chain(&arg.instances).cloned(),
            arg.tail,
        ))
    }
}

/// The rows a substitution puts in place of row tails.
#[derive(Clone, Copy, Debug)]
pub enum RowArgs<'a> {
    /// One row for each quantified binder of a callee, by binder position.
    /// Every tail must name one.
    Positional(&'a [Row]),
    /// The row for each of the given binders. Other tails stay.
    Binders(&'a [u32], &'a [Row]),
}

impl<'a> RowArgs<'a> {
    /// The row for `tail`: `Some(None)` when the tail stays, `None` when it
    /// names no argument.
    fn get(self, tail: u32) -> Option<Option<&'a Row>> {
        match self {
            Self::Positional(args) => args.get(usize::try_from(tail).ok()?).map(Some),
            Self::Binders(binders, args) => Some(
                binders
                    .iter()
                    .position(|binder| *binder == tail)
                    .and_then(|at| args.get(at)),
            ),
        }
    }
}

impl<'a> From<&'a [Row]> for RowArgs<'a> {
    fn from(args: &'a [Row]) -> Self {
        Self::Positional(args)
    }
}

impl<'a> From<&'a Vec<Row>> for RowArgs<'a> {
    fn from(args: &'a Vec<Row>) -> Self {
        Self::Positional(args)
    }
}

impl<'a, const N: usize> From<&'a [Row; N]> for RowArgs<'a> {
    fn from(args: &'a [Row; N]) -> Self {
        Self::Positional(args)
    }
}

impl trunk_ir::attr_kind::AttrKind for Row {
    const KIND: trunk_ir::op_schema::AttributeKind =
        trunk_ir::op_schema::AttributeKind::Dict(&trunk_ir::op_schema::AttributeKind::Any);
    type Out<'ctx> = Row;
    type In = Row;

    fn read<'ctx>(ctx: &'ctx IrContext, attr: &'ctx Attribute) -> Row {
        Self::from_attribute(ctx, attr).unwrap_or_else(|error| panic!("unverified row: {error}"))
    }

    fn write(ctx: &mut IrContext, value: Row) -> Attribute {
        value.to_attribute(ctx)
    }
}

/// What kind of row variable a binder of a definition stands for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RowBinder {
    /// Quantified by the definition's scheme.
    Param,
    /// Quantified by the definition's scheme and named inside an ability
    /// argument.
    AbilityArg,
    /// Quantified by the local scheme of a `let`-bound lambda.
    Local,
    /// Quantified by no scheme.
    Body,
}

impl RowBinder {
    pub fn keyword(self) -> &'static str {
        match self {
            Self::Param => "param",
            Self::AbilityArg => "ability_arg",
            Self::Local => "local",
            Self::Body => "body",
        }
    }

    pub fn from_keyword(keyword: &str) -> Option<Self> {
        match keyword {
            "param" => Some(Self::Param),
            "ability_arg" => Some(Self::AbilityArg),
            "local" => Some(Self::Local),
            "body" => Some(Self::Body),
            _ => None,
        }
    }

    /// Whether a reference passes a row argument for this binder.
    pub fn is_quantified(self) -> bool {
        matches!(self, Self::Param | Self::AbilityArg)
    }
}

impl trunk_ir::attr_kind::AttrKind for RowBinder {
    const KIND: trunk_ir::op_schema::AttributeKind = trunk_ir::op_schema::AttributeKind::String;
    type Out<'ctx> = RowBinder;
    type In = RowBinder;

    fn read<'ctx>(ctx: &'ctx IrContext, attr: &'ctx Attribute) -> RowBinder {
        let Attribute::String(keyword) = attr else {
            panic!("unverified row binder");
        };
        Self::from_keyword(ctx.str(*keyword)).expect("unverified row binder")
    }

    fn write(ctx: &mut IrContext, value: RowBinder) -> Attribute {
        ctx.string_attr(value.keyword())
    }
}

/// A row tail declared as the union of other tails. `sources` lists the open
/// sources' tails in declaration order and never holds `result`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RowUnion {
    pub result: u32,
    pub sources: Vec<u32>,
}

impl RowUnion {
    pub fn from_attribute(ctx: &IrContext, attr: &Attribute) -> Result<Self, String> {
        let Attribute::Dict(entries) = attr else {
            return Err("a row union must be a dictionary".into());
        };
        let entries = entries.view(ctx);
        let index = |value: &Attribute| match value {
            Attribute::Int(index) => u32::try_from(*index).ok(),
            _ => None,
        };
        let result = entries
            .get("result")
            .and_then(index)
            .ok_or("a row union needs a result binder index")?;
        let sources = match entries.get("sources") {
            Some(Attribute::List(sources)) => sources
                .iter()
                .map(|source| index(source).ok_or("row union sources must be binder indices"))
                .collect::<Result<Vec<_>, _>>()?,
            _ => return Err("a row union needs a list of sources".into()),
        };
        if entries.len() != 2 {
            return Err("a row union has only result and sources".into());
        }
        Ok(Self { result, sources })
    }

    pub fn to_attribute(&self, ctx: &mut IrContext) -> Attribute {
        Attribute::Dict(AttributeMap::from_entries(
            ctx,
            [
                ("result", Attribute::Int(i128::from(self.result))),
                (
                    "sources",
                    Attribute::List(
                        self.sources
                            .iter()
                            .map(|source| Attribute::Int(i128::from(*source)))
                            .collect(),
                    ),
                ),
            ],
        ))
    }
}

impl trunk_ir::attr_kind::AttrKind for RowUnion {
    const KIND: trunk_ir::op_schema::AttributeKind =
        trunk_ir::op_schema::AttributeKind::Dict(&trunk_ir::op_schema::AttributeKind::Any);
    type Out<'ctx> = RowUnion;
    type In = RowUnion;

    fn read<'ctx>(ctx: &'ctx IrContext, attr: &'ctx Attribute) -> RowUnion {
        Self::from_attribute(ctx, attr)
            .unwrap_or_else(|error| panic!("unverified row union: {error}"))
    }

    fn write(ctx: &mut IrContext, value: RowUnion) -> Attribute {
        value.to_attribute(ctx)
    }
}

/// Decode a list attribute of `key` on `op`, which is empty when absent.
fn read_list<T>(
    ctx: &IrContext,
    op: trunk_ir::OpRef,
    key: &str,
    decode: impl Fn(&IrContext, &Attribute) -> Result<T, String>,
) -> Result<Option<Vec<T>>, String> {
    match ctx.op_attrs(op).get(key) {
        None => Ok(None),
        Some(Attribute::List(items)) => items
            .iter()
            .map(|item| decode(ctx, item))
            .collect::<Result<_, _>>()
            .map(Some),
        Some(_) => Err(format!("{key} must be a list")),
    }
}

/// The row binders `op` declares, or why they are malformed.
pub fn row_binders(ctx: &IrContext, op: trunk_ir::OpRef) -> Result<Vec<RowBinder>, String> {
    let binders = read_list(ctx, op, ROW_BINDERS_ATTR, |ctx, binder| match binder {
        Attribute::String(keyword) => RowBinder::from_keyword(ctx.str(*keyword))
            .ok_or_else(|| format!("unknown row binder kind '{}'", ctx.str(*keyword))),
        _ => Err("row binders must be strings".into()),
    })?
    .unwrap_or_default();
    if binders[quantified_count(&binders)..]
        .iter()
        .any(|binder| binder.is_quantified())
    {
        return Err("quantified row binders must come before the others".into());
    }
    Ok(binders)
}

/// The row tail unions `op` declares, or why they are malformed.
pub fn row_unions(ctx: &IrContext, op: trunk_ir::OpRef) -> Result<Vec<RowUnion>, String> {
    Ok(read_list(ctx, op, ROW_UNIONS_ATTR, RowUnion::from_attribute)?.unwrap_or_default())
}

/// The local binders a lambda quantifies, or why they are malformed.
pub fn quantified_rows(ctx: &IrContext, op: trunk_ir::OpRef) -> Result<Vec<u32>, String> {
    Ok(
        read_list(ctx, op, QUANTIFIED_ROWS_ATTR, |_, binder| match binder {
            Attribute::Int(index) => {
                u32::try_from(*index).map_err(|_| "quantified rows must be binder indices".into())
            }
            _ => Err("quantified rows must be binder indices".into()),
        })?
        .unwrap_or_default(),
    )
}

/// The row arguments a reference passes, or why they are malformed.
pub fn row_args(ctx: &IrContext, op: trunk_ir::OpRef) -> Result<Vec<Row>, String> {
    Ok(read_list(ctx, op, ROW_ARGS_ATTR, Row::from_attribute)?.unwrap_or_default())
}

/// The quantified binders of `binders`, which a reference passes rows for.
pub fn quantified_count(binders: &[RowBinder]) -> usize {
    binders
        .iter()
        .take_while(|binder| binder.is_quantified())
        .count()
}

fn is_func_sig(data: &TypeData) -> bool {
    data.name == <super::FuncSig as trunk_ir::ops::DialectType>::NAME
}

/// The ability row a type stores, if it is a callable type with a well-formed
/// row.
pub(crate) fn stored_row(ctx: &IrContext, data: &TypeData) -> Option<Result<Row, String>> {
    if !is_func_sig(data) {
        return None;
    }
    Some(
        data.attrs
            .view(ctx)
            .get(ABILITY_ROW_ATTR)
            .map_or_else(|| Ok(Row::default()), |row| Row::from_attribute(ctx, row)),
    )
}

/// The name of the compiler-generated tuple layout whose fields have the
/// logical types `elements`.
pub fn logical_tuple_name(ctx: &IrContext, elements: &[TypeRef]) -> String {
    let mut name = LOGICAL_TUPLE_PREFIX.to_owned();
    name.push_str(&key(
        "tuple",
        elements.iter().map(|element| type_key(ctx, *element)),
    ));
    name
}

/// `tag_N` followed by each of the `N` parts as `_length_part`; just `tag`
/// without parts. A tag holds no `_` and every part is a key or [`text`], so
/// a key reads back in only one way.
fn key(tag: &str, parts: impl IntoIterator<Item = String>) -> String {
    let parts: Vec<_> = parts.into_iter().collect();
    if parts.is_empty() {
        return tag.to_owned();
    }
    let mut key = format!("{tag}_{}", parts.len());
    for part in parts {
        key.push('_');
        key.push_str(&part.len().to_string());
        key.push('_');
        key.push_str(&part);
    }
    key
}

/// `text` in identifier characters without `_`: a letter or digit other than
/// `Z` stays, and every other byte is `Z` and its two hexadecimal digits.
fn text(text: &str) -> String {
    let mut encoded = String::with_capacity(text.len());
    for byte in text.bytes() {
        if byte.is_ascii_alphanumeric() && byte != b'Z' {
            encoded.push(char::from(byte));
        } else {
            encoded.push_str(&format!("Z{byte:02x}"));
        }
    }
    encoded
}

/// A symbol path, one part per component.
fn path_key(path: &SymbolPath) -> String {
    key(
        "path",
        path.iter().map(|component| component.with_str(text)),
    )
}

fn type_key(ctx: &IrContext, ty: TypeRef) -> String {
    if let Some(elements) = logical_tuple_elements(ctx, ty) {
        return key(
            "tuple",
            elements.iter().map(|element| type_key(ctx, *element)),
        );
    }
    let data = ctx.get_type(ty);
    if data.name == crate::dialect::adt::Typeref::NAME
        && let Some(name) = data.attrs.view(ctx).get_str("name")
    {
        return key("named", [text(name)]);
    }
    let attributes = |reserved: &[&str]| -> Vec<String> {
        data.attrs
            .view(ctx)
            .iter()
            .filter(|(name, _)| !reserved.contains(name))
            .map(|(name, attr)| key("attr", [text(name), attribute_key(ctx, attr)]))
            .collect()
    };
    if let Some(signature) = <super::FuncSig as trunk_ir::ops::DialectType>::from_type_ref(ctx, ty)
    {
        let row = signature.row(ctx);
        let row = key(
            "row",
            row.instances.iter().map(path_key).chain([row
                .tail
                .map_or_else(|| "closed".to_owned(), |tail| tail.to_string())]),
        );
        return key(
            "callable",
            signature
                .inputs(ctx)
                .iter()
                .chain([&signature.result(ctx)])
                .map(|ty| type_key(ctx, *ty))
                .chain([row])
                .chain(attributes(&super::RESERVED_ATTRS)),
        );
    }
    let tag = text(&data.name.display(ctx).to_string());
    key(
        &tag,
        data.params
            .iter()
            .map(|param| type_key(ctx, *param))
            .chain(attributes(&[])),
    )
}

fn attribute_key(ctx: &IrContext, attr: &Attribute) -> String {
    match attr {
        Attribute::Bool(value) => key("bool", [value.to_string()]),
        Attribute::Int(value) => key("int", [text(&value.to_string())]),
        Attribute::FloatBits(bits) => key("float", [format!("{bits:x}")]),
        Attribute::String(value) => key("string", [text(ctx.str(*value))]),
        Attribute::SymbolRef(path) => key("symbol", [path_key(path)]),
        Attribute::Type(ty) => key("type", [type_key(ctx, *ty)]),
        Attribute::List(items) => key("list", items.iter().map(|item| attribute_key(ctx, item))),
        Attribute::Dict(entries) => key(
            "dict",
            entries
                .view(ctx)
                .iter()
                .map(|(name, value)| key("entry", [text(name), attribute_key(ctx, value)])),
        ),
        Attribute::Bytes(bytes) => key(
            "bytes",
            [bytes.iter().map(|byte| format!("{byte:02x}")).collect()],
        ),
        Attribute::Unit => "unit".to_owned(),
        Attribute::Location(_) => "location".to_owned(),
    }
}

fn tuple_layout_data(ctx: &mut IrContext, name: &str, elements: &[TypeRef]) -> TypeRef {
    adt::Struct::from_fields(
        ctx,
        name.to_owned(),
        elements
            .iter()
            .enumerate()
            .map(|(index, element)| (index.to_string(), *element)),
        AttributeMap::new(),
    )
    .as_type_ref()
}

/// The compiler-generated tuple layout of `elements`: an `adt.typeref` whose
/// layout is registered under its name.
pub fn logical_tuple_type(ctx: &mut IrContext, elements: &[TypeRef]) -> TypeRef {
    let name = logical_tuple_name(ctx, elements);
    let layout = tuple_layout_data(ctx, &name, elements);
    ctx.register_type_alias(Symbol::new(&name), layout);
    let name = ctx.intern_str(&name);
    adt::Typeref::params().name(name).intern(ctx).as_type_ref()
}

/// The element types of a compiler-generated tuple layout.
pub fn logical_tuple_elements(ctx: &IrContext, ty: TypeRef) -> Option<&[TypeRef]> {
    let data = ctx.get_type(ty);
    if data.name != crate::dialect::adt::Typeref::NAME {
        return None;
    }
    let name = data.attrs.view(ctx).get_str("name")?;
    if !name.starts_with(LOGICAL_TUPLE_PREFIX) {
        return None;
    }
    let layout = ctx.type_alias_by_text(name)?;
    if !ctx.get_type(layout).name.is(ctx, "adt.struct") {
        return None;
    }
    adt::Struct::from_type_ref(ctx, layout).map(|layout| layout.field_types(ctx))
}

/// Where substitution finds or makes the types it produces.
trait TypeSink {
    fn ctx(&self) -> &IrContext;
    fn make(&mut self, data: TypeData) -> Option<TypeRef>;
    fn tuple(&mut self, elements: &[TypeRef]) -> Option<TypeRef>;
    /// The attribute key `text`.
    fn key(&mut self, text: &str) -> Option<trunk_ir::types::StringRef>;
    /// `row` encoded as an attribute value.
    fn row(&mut self, row: &Row) -> Option<Attribute>;
}

struct Intern<'a>(&'a mut IrContext);

impl TypeSink for Intern<'_> {
    fn ctx(&self) -> &IrContext {
        self.0
    }

    fn make(&mut self, data: TypeData) -> Option<TypeRef> {
        Some(self.0.intern_type(data))
    }

    fn tuple(&mut self, elements: &[TypeRef]) -> Option<TypeRef> {
        Some(logical_tuple_type(self.0, elements))
    }

    fn key(&mut self, text: &str) -> Option<trunk_ir::types::StringRef> {
        Some(self.0.intern_str(text))
    }

    fn row(&mut self, row: &Row) -> Option<Attribute> {
        Some(row.to_attribute(self.0))
    }
}

/// Finds only types that already exist, so a type it cannot find differs
/// from every type of the IR.
struct Lookup<'a>(&'a IrContext);

impl TypeSink for Lookup<'_> {
    fn ctx(&self) -> &IrContext {
        self.0
    }

    fn make(&mut self, data: TypeData) -> Option<TypeRef> {
        self.0.types().lookup(&data)
    }

    fn tuple(&mut self, elements: &[TypeRef]) -> Option<TypeRef> {
        let name = logical_tuple_name(self.0, elements);
        let entry = (
            self.0.lookup_str("name")?,
            Attribute::String(self.0.lookup_str(&name)?),
        );
        let ty = self.0.types().lookup(&TypeData {
            name: crate::dialect::adt::Typeref::NAME,
            params: Default::default(),
            attrs: AttributeMap::from_refs(self.0, [entry]),
        })?;
        (logical_tuple_elements(self.0, ty)? == elements).then_some(ty)
    }

    fn key(&mut self, text: &str) -> Option<trunk_ir::types::StringRef> {
        self.0.lookup_str(text)
    }

    fn row(&mut self, row: &Row) -> Option<Attribute> {
        row.find_attribute(self.0)
    }
}

/// Rewrites every callable type inside a type, rebuilding the types and
/// tuple layouts that hold it.
struct Substitution<'a, S> {
    sink: S,
    args: RowArgs<'a>,
    done: HashMap<TypeRef, Option<TypeRef>>,
}

impl<S: TypeSink> Substitution<'_, S> {
    fn ty(&mut self, ty: TypeRef) -> Option<TypeRef> {
        if let Some(&done) = self.done.get(&ty) {
            return done;
        }
        let substituted = self.compute(ty);
        self.done.insert(ty, substituted);
        substituted
    }

    fn compute(&mut self, ty: TypeRef) -> Option<TypeRef> {
        if let Some(elements) = logical_tuple_elements(self.sink.ctx(), ty) {
            let elements = elements.to_vec();
            let substituted = elements
                .iter()
                .map(|element| self.ty(*element))
                .collect::<Option<Vec<_>>>()?;
            return if substituted == elements {
                Some(ty)
            } else {
                self.sink.tuple(&substituted)
            };
        }
        let data = self.sink.ctx().get_type(ty);
        if data.name == crate::dialect::adt::Typeref::NAME {
            return Some(ty);
        }
        let mut data = data.clone();
        let row = stored_row(self.sink.ctx(), &data).transpose().ok()?;
        let mut changed = false;
        for param in data.params.iter_mut() {
            let substituted = self.ty(*param)?;
            changed |= substituted != *param;
            *param = substituted;
        }
        if let Some(row) = row {
            let substituted = row.substitute(self.args)?;
            if substituted != row {
                changed = true;
                let key = self.sink.key(ABILITY_ROW_ATTR)?;
                let value = self.sink.row(&substituted)?;
                let entries: Vec<_> = data
                    .attrs
                    .iter()
                    .filter(|(existing, _)| *existing != key)
                    .map(|(existing, attr)| (existing, attr.clone()))
                    .chain((!substituted.is_empty()).then_some((key, value)))
                    .collect();
                data.attrs = AttributeMap::from_refs(self.sink.ctx(), entries);
            }
        }
        if changed {
            self.sink.make(data)
        } else {
            Some(ty)
        }
    }
}

/// Replace, in every callable type inside `ty`, a tail that `args` names
/// with the row argument for it.
///
/// Returns `None` when a tail names no argument.
pub fn substitute_rows<'a>(
    ctx: &mut IrContext,
    ty: TypeRef,
    args: impl Into<RowArgs<'a>>,
) -> Option<TypeRef> {
    Substitution {
        sink: Intern(ctx),
        args: args.into(),
        done: HashMap::default(),
    }
    .ty(ty)
}

/// [`substitute_rows`] without creating types: `None` also when the
/// substituted type does not exist, which no IR value can then have.
pub fn find_substituted_rows<'a>(
    ctx: &IrContext,
    ty: TypeRef,
    args: impl Into<RowArgs<'a>>,
) -> Option<TypeRef> {
    Substitution {
        sink: Lookup(ctx),
        args: args.into(),
        done: HashMap::default(),
    }
    .ty(ty)
}

/// Visit every ability row inside `ty`, including those of the tuple layouts
/// it names. Visits a malformed row as an error.
pub fn visit_rows(ctx: &IrContext, ty: TypeRef, visit: &mut impl FnMut(Result<Row, String>)) {
    let mut seen = rustc_hash::FxHashSet::default();
    visit_rows_in(ctx, ty, &mut seen, visit);
}

fn visit_rows_in(
    ctx: &IrContext,
    ty: TypeRef,
    seen: &mut rustc_hash::FxHashSet<TypeRef>,
    visit: &mut impl FnMut(Result<Row, String>),
) {
    if !seen.insert(ty) {
        return;
    }
    if let Some(elements) = logical_tuple_elements(ctx, ty) {
        for element in elements {
            visit_rows_in(ctx, *element, seen, visit);
        }
        return;
    }
    let data = ctx.get_type(ty);
    if data.attrs.view(ctx).contains_key(ABILITY_ROW_ATTR)
        && let Some(row) = stored_row(ctx, data)
    {
        visit(row);
    }
    for param in data.params.iter() {
        visit_rows_in(ctx, *param, seen, visit);
    }
}

#[cfg(test)]
mod tests;
