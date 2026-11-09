//! Arena-based adt dialect.

use crate::dialect::tribute_control::{
    ROW_ARGS_ATTR, ROW_PARAMS_ATTR, Row, display_layout_binders, parse_layout_binders,
};
use crate::dialect::tribute_rtti::ManagedRefType;
use trunk_ir::asm_format::TypeAliasHint;
use trunk_ir::dialect::core::Index;
use trunk_ir::op_flag::Pure;

#[trunk_ir::dialect]
mod adt {
    /// Validated wrapper for an `adt.struct` layout type.
    ///
    /// ```text
    /// adt.struct<"Point", (core.i32, core.i32)>
    /// adt.struct<"Point">
    /// adt.struct<"Callback"['a0], (tribute_control.func_sig<(core.i64) -> core.i64 row(| 'a0)>)>
    /// adt.struct<(core.ptr), {layout = "closure"}>
    /// ```
    ///
    /// A nominal layout is an identified type whose identifier is its
    /// `name`: its key has only type attributes, which an `adt.typeref` to
    /// it also has, and its body is a literal `adt.struct` without
    /// attributes holding its fields. A literal layout has no name and is
    /// its fields, its type parameters in declaration order, and its
    /// attributes (`design/ir.md` § "`adt.struct` nominal layout type").
    /// [`Struct::fields`] reads either; `param_fields` reads only the type's
    /// own parameters, which a key has none of. The text writes a key as its name,
    /// and the fields of its body where it writes the body. Fields have no
    /// names; the source names are the layout's `adt.struct_decl`. A layout
    /// whose declaration has row parameters counts its row binders in
    /// `row_params`, a count without kinds; a row tail inside its field types,
    /// in a field's callable type or in the row arguments of a nested
    /// `adt.typeref`, is the index of one of them, and the binders do not
    /// leave the layout. The text writes the binders right after the name,
    /// before the fields, as `['a0, 'a1, ...]`, numbered from `'a0` in order
    /// with no kind keyword or union; the layout is one for every row argument
    /// (`design/tribute-dialects.md` § "Nominal row 인자").
    #[verify]
    #[asm_format]
    #[alias_hint]
    #[identified]
    #[implements(ManagedRefType)]
    struct Struct<#[rest] ParamFields> {
        name: Option<Attr<String>>,
        row_params: Option<Attr<u32>>,
    }

    /// Validated wrapper for an `adt.enum` layout type.
    ///
    /// ```text
    /// adt.enum<"Option", [(), (core.i32)]>
    /// adt.enum<"Option">
    /// adt.enum<"Step"['a0], [(tribute_control.func_sig<() -> core.i64 row(| 'a0)>), ()]>
    /// adt.enum<[(), (core.i32)]>
    /// ```
    ///
    /// A named enum is an identified type like a nominal [`Struct`], whose
    /// body is a literal `adt.enum`. A literal enum has no attributes; each
    /// of its variants is one type parameter, in declaration order: an
    /// `adt.variant` whose parameters are the variant's field types.
    /// [`Enum::variants`] reads either; `param_variants` reads only the
    /// type's own parameters, which a key has none of. Variants and fields have no names;
    /// the source names are the layout's `adt.enum_decl`. Its `row_params`
    /// are those of [`Struct`].
    #[verify]
    #[asm_format]
    #[alias_hint]
    #[identified]
    #[implements(ManagedRefType)]
    struct Enum<#[rest] ParamVariants> {
        name: Option<Attr<String>>,
        row_params: Option<Attr<u32>>,
    }

    /// `adt.typeref`: a nominal managed reference to the layout named
    /// `name`.
    ///
    /// A valid value is managed by this type alone; no pass decides whether
    /// it is managed from pointer provenance. `adt.ref_null` makes its null
    /// inhabitant and `adt.ref_cast` converts between references of the
    /// same nominal identity.
    ///
    /// When the layout has row parameters, `row_args` holds one row for each
    /// of them in declaration order, each in the form of a callable's
    /// `ability_row`. A tail names a row binder of the definition that writes
    /// the type, or of the layout inside a layout. A typeref of a declaration
    /// without row parameters omits the attribute. Row arguments are part of
    /// the type's identity: two typerefs of one layout with different row
    /// arguments are different types, while the layout's name and published
    /// alias carry no row argument, so they share the one layout. The text
    /// writes the arguments right after the name as `[row(...), ...]`, each
    /// in the `row(..)` form of a callable's row except that the empty row
    /// is written as `row()` and never omitted; without row parameters the
    /// `[...]` is absent. An attribute value this syntax cannot write prints
    /// in the generic form. How row arguments reach the field types and when
    /// legalization drops them is in `design/tribute-dialects.md` § "Nominal
    /// row 인자".
    ///
    /// ```text
    /// adt.typeref<"Point">
    /// adt.typeref<"Callback"[row(@"<State(Int)>")]>
    /// adt.typeref<"Pair"[row(), row(@A | 'a0)]>
    /// ```
    #[verify]
    #[asm_format]
    #[alias_hint]
    #[implements(ManagedRefType)]
    struct Typeref {
        name: Attr<String>,
        row_args: Option<Attr<[Row]>>,
    }

    /// `adt.array`: a managed reference to an array object whose elements
    /// are `Element` values in order. An element is a scalar, a pointer, or a
    /// managed reference, one machine value each; an aggregate such as a
    /// struct, variant, or closure is held as a managed reference to its
    /// object, never inlined. The unmanaged linear array is `core.array`,
    /// which no `adt` array operation accepts. The index and trap rules the
    /// array operations share are in `design/tribute-dialects.md` § "`adt`
    /// 배열".
    #[implements(ManagedRefType)]
    struct Array<Element>;

    /// `adt.variant`: the field types of one variant of an `adt.enum`, which
    /// addresses it by its position.
    struct Variant<#[rest] Fields>;

    /// A new struct of layout `type` whose fields are `fields` in order.
    ///
    /// When the layout declares row parameters, each field operand has the
    /// layout's field type with the layout's row binders replaced by the
    /// `row_args` of the result's `adt.typeref`, using call-site row
    /// substitution; see `design/tribute-dialects.md` § "Nominal row 인자".
    #[flags(Pure)]
    #[asm_format(fields attr_dict_bare ":" type(result))]
    fn struct_new(r#type: Attr<Type>, fields: Variadic<_>) -> Value<_> {}

    /// Field `field` of the struct `ref` of layout `type`.
    ///
    /// When the layout declares row parameters, the result has the layout's
    /// field type with the layout's row binders replaced by the `row_args` of
    /// the `ref` operand's `adt.typeref`, using call-site row substitution; see
    /// `design/tribute-dialects.md` § "Nominal row 인자".
    #[flags(Pure)]
    #[asm_format(r#ref attr_dict_bare ":" type(result))]
    fn struct_get(r#type: Attr<Type>, field: Attr<u32>, r#ref: Value<_>) -> Value<_> {}

    #[asm_format(r#ref "," value attr_dict_bare)]
    fn struct_set(r#type: Attr<Type>, field: Attr<u32>, r#ref: Value<_>, value: Value<_>) {}

    /// A new value of variant `tag` of enum layout `type` whose fields are
    /// `fields` in order.
    ///
    /// ```text
    /// %dot = adt.variant_new %x, %y {tag = 1, type = !Shape} : !Shape
    /// ```
    ///
    /// `tag` is the variant's position among the layout's variants, in
    /// declaration order, as `adt.variant_tag` returns it; the verifier
    /// rejects a position the layout does not have.
    ///
    /// When the layout declares row parameters, each field operand has the
    /// layout's field type with the layout's row binders replaced by the
    /// `row_args` of the result's `adt.typeref`, using call-site row
    /// substitution; see `design/tribute-dialects.md` § "Nominal row 인자".
    #[flags(Pure)]
    #[verify]
    #[asm_format(fields attr_dict_bare ":" type(result))]
    fn variant_new(r#type: Attr<Type>, tag: Attr<u32>, fields: Variadic<_>) -> Value<_> {}

    /// Whether `ref`, a value of enum layout `type`, is of variant `tag`, the
    /// variant's position as for `adt.variant_new`.
    #[flags(Pure)]
    #[verify]
    #[asm_format(r#ref attr_dict_bare ":" type(result))]
    fn variant_is(r#type: Attr<Type>, tag: Attr<u32>, r#ref: Value<_>) -> Value<_> {}

    /// The position of `ref`'s variant among the variants of the
    /// `adt.enum` `type`, in declaration order.
    ///
    /// ```text
    /// %tag = adt.variant_tag %value {type = !Shape}
    /// ```
    ///
    /// - **Shape:** `type` is the enum layout of `ref`, and the result is a
    ///   `core.index`.
    /// - **Meaning:** `0` for the first declared variant, `1` for the next,
    ///   and so on. A variant the program never allocates still keeps its
    ///   position, so the result never depends on which variants a program
    ///   allocates. `ref` is an enum value and is not null; reading a null
    ///   reference is undefined, as for `adt.variant_is`.
    /// - **Ownership:** borrows `ref`.
    /// - **Location:** a case that selects by variant switches on this
    ///   position (`design/types.md` § "망라성과 도달 불가 arm"); each target
    ///   computes it from the value's runtime type descriptor
    ///   (`design/runtime-types.md` § "Variant 판별").
    #[flags(Pure)]
    #[asm_format(r#ref attr_dict_bare)]
    fn variant_tag(r#type: Attr<Type>, r#ref: Value<_>) -> Value<Index> {}

    /// Field `field` of `ref`, a value of variant `tag` of enum layout `type`,
    /// the variant's position as for `adt.variant_new`.
    ///
    /// When the layout declares row parameters, the result has the layout's
    /// field type with the layout's row binders replaced by the `row_args` of
    /// the `ref` operand's `adt.typeref`, using call-site row substitution; see
    /// `design/tribute-dialects.md` § "Nominal row 인자".
    #[flags(Pure)]
    #[verify]
    #[asm_format(r#ref attr_dict_bare ":" type(result))]
    fn variant_get(
        r#type: Attr<Type>,
        tag: Attr<u32>,
        field: Attr<u32>,
        r#ref: Value<_>,
    ) -> Value<_> {
    }

    /// The source names of the struct layout `type`.
    ///
    /// ```text
    /// adt.struct_decl "Point" !Point ["x", "y"]
    /// ```
    ///
    /// - **Shape:** a module-level operation. `name` is the struct's source
    ///   name, `type` is a nominal `adt.struct` layout, and `fields` has one
    ///   name per field of that layout, in declaration order.
    /// - **Meaning:** names only; no value depends on it. Native RTTI
    ///   records the struct's name and field names from it. It belongs to
    ///   the layout that is its `type`, not to a layout of the same name.
    /// - **Ownership and value flow:** a pass that converts layouts converts
    ///   `type` as it converts the types in operation attributes, and keeps
    ///   `name` when the converted layout's name has a tag.
    /// - **Location:** the frontend emits it with the layout of each source
    ///   struct; the target pass that lowers `adt` operations erases it
    ///   (`design/ir.md` § "Nominal 선언").
    #[verify]
    #[asm_format(name r#type fields attr_dict)]
    fn struct_decl(name: Attr<String>, r#type: Attr<Type>, fields: Attr<[String]>) {}

    /// The source names of the enum layout `type`.
    ///
    /// ```text
    /// adt.enum_decl "Shape" !Shape ["Dot", "Rect"]
    /// ```
    ///
    /// - **Shape:** a module-level operation. `name` is the enum's source
    ///   name, `type` is a named `adt.enum` layout, and `variants` has one
    ///   name per variant of that layout, in declaration order.
    /// - **Meaning:** names only; no value depends on it. Native RTTI
    ///   records the enum's name and its variants' names from it. It belongs
    ///   to the layout that is its `type`, not to a layout of the same name.
    ///   `tribute.definition`, when present, is the declaration's source
    ///   location.
    /// - **Ownership and value flow:** a pass that converts layouts converts
    ///   `type` as it converts the types in operation attributes, and keeps
    ///   `name` when the converted layout's name has a tag.
    /// - **Location:** the frontend emits it with the layout of each source
    ///   enum and each specialized enum instance; the target pass that
    ///   lowers `adt` operations erases it (`design/ir.md` § "Nominal 선언").
    #[verify]
    #[asm_format(name r#type variants attr_dict)]
    fn enum_decl(name: Attr<String>, r#type: Attr<Type>, variants: Attr<[String]>) {}

    /// A `length`-element array whose every element is `initial`.
    #[flags(Pure)]
    #[asm_format(length "," initial attr_dict_bare ":" type(result))]
    fn array_new<A: Array>(
        r#type: Attr<TypeOf<A>>,
        length: Value<Index>,
        initial: Value<A::Element>,
    ) -> Value<A> {
    }

    /// An array of `elements` in order. Its length is the number of
    /// operands, which may be zero.
    #[flags(Pure)]
    #[asm_format(elements attr_dict_bare ":" type(result))]
    fn array_of<A: Array>(r#type: Attr<TypeOf<A>>, elements: Variadic<A::Element>) -> Value<A> {}

    /// A new array of `source[0, start)`, `elements`, and `source[end, len)`.
    ///
    /// It builds the array by copying ranges and leaves `source` unchanged.
    /// Inserting one element (`start = end`), removing one (no `elements` and
    /// `end = start + 1`), replacing elements, and trimming either end are
    /// all written with this operation. It requires `start ≤ end ≤ len`; the
    /// target traps an execution that violates it.
    ///
    /// ```text
    /// %r = adt.array_splice %source[%start, %end](%element0, ...) attributes {type = !A} : !A
    /// ```
    #[flags(Pure)]
    #[asm_format(source "[" start "," end "]" "(" elements ")" attr_dict ":" type(result))]
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
    #[asm_format(r#ref "," index attr_dict_bare ":" type(result))]
    fn array_get<A: Array>(r#ref: Value<A>, index: Value<Index>) -> Value<A::Element> {}

    /// Replace the element at `index` in place. Every value that refers to
    /// the same array sees the new element; a producer that must not change
    /// a shared array builds the changed array with `adt.array_splice`
    /// instead. It requires `index < len`; the target traps an execution that
    /// violates it.
    #[asm_format(r#ref "," index "," value attr_dict_bare)]
    fn array_set<A: Array>(r#ref: Value<A>, index: Value<Index>, value: Value<A::Element>) {}

    /// The number of elements.
    #[flags(Pure)]
    #[asm_format(r#ref attr_dict_bare ":" type(result))]
    fn array_len(r#ref: Value<impl Array>) -> Value<Index> {}

    /// The null inhabitant of exactly the `adt.typeref` `type`, which is
    /// also the result type. Retaining or releasing it is a no-op.
    #[flags(Pure)]
    #[asm_format(attr_dict_bare ":" type(result))]
    fn ref_null(r#type: Attr<Type>) -> Value<_> {}

    #[flags(Pure)]
    #[asm_format(r#ref attr_dict_bare ":" type(result))]
    fn ref_is_null(r#ref: Value<_>) -> Value<_> {}

    /// `ref` as the `adt.typeref` `type`, which is also the result type. It
    /// is allowed only between managed references whose nominal identity has
    /// been established to be the same.
    #[implements(Fold)]
    #[flags(Pure)]
    #[asm_format(r#ref attr_dict_bare ":" type(result))]
    fn ref_cast(r#type: Attr<Type>, r#ref: Value<_>) -> Value<_> {}

    #[flags(Pure)]
    #[asm_format(attr_dict_bare ":" type(result))]
    fn string_const(value: Attr<String>) -> Value<_> {}

    #[flags(Pure)]
    #[asm_format(attr_dict_bare ":" type(result))]
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
    /// The layout's name, or a literal layout's runtime `layout` identifier.
    fn alias_hint<'a>(&self, ctx: &'a IrContext) -> Option<&'a str> {
        self.name(ctx).or_else(|| {
            ctx.type_attrs(self.as_type_ref())
                .get_str(crate::runtime_layout::LAYOUT_ATTR)
        })
    }
}

impl TypeAliasHint for Enum {
    fn alias_hint<'a>(&self, ctx: &'a IrContext) -> Option<&'a str> {
        self.name(ctx)
    }
}

impl trunk_ir::type_def::IdentifiedType for Struct {
    fn identifier<'a>(&self, ctx: &'a IrContext) -> Option<&'a str> {
        self.name(ctx)
    }
}

impl trunk_ir::type_def::IdentifiedType for Enum {
    fn identifier<'a>(&self, ctx: &'a IrContext) -> Option<&'a str> {
        self.name(ctx)
    }
}

// Typerefs that differ only in their row arguments would share the name, so
// one with row arguments suggests none.
impl TypeAliasHint for Typeref {
    fn alias_hint<'a>(&self, ctx: &'a IrContext) -> Option<&'a str> {
        (!ctx.type_attrs(self.0).contains_key(ROW_ARGS_ATTR)).then(|| self.name(ctx))
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

use rustc_hash::FxHashMap as HashMap;
use trunk_ir::attr_kind::{Bytes, Type};
use trunk_ir::context::IrContext;
use trunk_ir::ops::DialectType;
use trunk_ir::types::{
    Attribute, AttributeMap, PARAM_ATTRS_ATTR, StringArg, StringRef, TypeDataBuilder,
};

pub mod convert;
pub mod layout;

/// Type attribute holding a nominal layout's or an `adt.typeref`'s name, a
/// string.
pub const STRUCT_NAME_ATTR: &str = "name";

/// Type attribute of a row instance holding the key of the layout it
/// applies (`design/ir.md` § "Nominal 값 타입").
pub const DEF_ATTR: &str = "def";

/// Whether `ty` is an `adt.struct`, `adt.enum` or `adt.typeref`. A nominal
/// one is named by its `name` attribute, or by its `def`'s for a row
/// instance; a literal `adt.struct` has none.
pub fn is_nominal(ctx: &IrContext, ty: TypeRef) -> bool {
    let name = ctx.get_type(ty).name;
    name == <Struct as DialectType>::NAME
        || name == <Enum as DialectType>::NAME
        || name == Typeref::NAME
}

/// Whether `ty` is a nominal value type: a nominal layout's key, a row
/// instance, or an `adt.typeref`, rather than a literal layout.
pub fn is_nominal_value(ctx: &IrContext, ty: TypeRef) -> bool {
    Typeref::matches(ctx, ty)
        || row_instance_def(ctx, ty).is_some()
        || Struct::from_type_ref(ctx, ty).is_some_and(|layout| layout.is_nominal(ctx))
        || Enum::from_type_ref(ctx, ty).is_some_and(|layout| layout.is_nominal(ctx))
}

/// The nominal name of an `adt.struct`, `adt.enum` or `adt.typeref`: its
/// `name` type attribute, or the name of a row instance's `def`.
pub fn nominal_name_ref(ctx: &IrContext, ty: TypeRef) -> Option<StringRef> {
    if !is_nominal(ctx, ty) {
        return None;
    }
    let ty = row_instance_def(ctx, ty).unwrap_or(ty);
    ctx.type_attrs(ty).get_string_ref(STRUCT_NAME_ATTR)
}

/// The text of [`nominal_name_ref`].
pub fn nominal_name(ctx: &IrContext, ty: TypeRef) -> Option<&str> {
    nominal_name_ref(ctx, ty).map(|name| ctx.str(name))
}

/// The `adt.struct` or `adt.enum` layout of a value of type `ty`: `ty`
/// itself when it is a layout, a row instance's `def`, or the layout an
/// `adt.typeref` refers to ([`Typeref::layout`]). `None` for any other type
/// or a typeref without a layout.
pub fn layout_of(ctx: &IrContext, ty: TypeRef) -> Option<TypeRef> {
    if let Some(def) = row_instance_def(ctx, ty) {
        return Some(def);
    }
    if Struct::matches(ctx, ty) || Enum::matches(ctx, ty) {
        return Some(ty);
    }
    Typeref::from_type_ref(ctx, ty)?.layout(ctx)
}

/// The layout key a row instance `ty` applies, its `def`; `None` for any
/// other type (`design/ir.md` § "Nominal 값 타입").
pub fn row_instance_def(ctx: &IrContext, ty: TypeRef) -> Option<TypeRef> {
    let name = ctx.get_type(ty).name;
    if name != <Struct as DialectType>::NAME && name != <Enum as DialectType>::NAME {
        return None;
    }
    ctx.type_attrs(ty).get_type(DEF_ATTR)
}

/// The value type of the nominal layout `key` applied to `rows`, one for
/// each of its row parameters: the key itself without rows, or its row
/// instance (`design/ir.md` § "Nominal 값 타입").
///
/// # Panics
///
/// If `rows` does not have one row for each row parameter of `key`.
pub fn value_type(
    ctx: &mut IrContext,
    key: TypeRef,
    rows: impl IntoIterator<Item = Row>,
) -> TypeRef {
    let rows: Vec<_> = rows.into_iter().map(|row| row.to_attribute(ctx)).collect();
    if rows.is_empty() {
        return key;
    }
    let name = ctx.get_type(key).name;
    let ty = TypeDataBuilder::new(name)
        .attr(DEF_ATTR, Attribute::Type(key))
        .attr(ROW_ARGS_ATTR, Attribute::List(rows))
        .intern(ctx);
    verify_row_instance(ctx, ty).unwrap_or_else(|error| panic!("row instance: {error}"));
    ty
}

/// The row arguments a value of type `ty` passes to its layout's row
/// parameters: those of a row instance or an `adt.typeref`; empty for any
/// other type.
pub fn value_rows(ctx: &IrContext, ty: TypeRef) -> Vec<Row> {
    if !is_nominal(ctx, ty) {
        return Vec::new();
    }
    match ctx.type_attrs(ty).get(ROW_ARGS_ATTR) {
        Some(Attribute::List(rows)) => rows
            .iter()
            .map(|row| Row::from_attribute(ctx, row).expect("verified row argument"))
            .collect(),
        _ => Vec::new(),
    }
}

/// A malformed row instance.
#[derive(Clone, Debug, PartialEq, Eq, derive_more::Display, derive_more::Error)]
pub enum RowInstanceError {
    #[display("a row instance has only `{DEF_ATTR}` and `{ROW_ARGS_ATTR}`")]
    Shape,
    #[display("`{DEF_ATTR}` is not a nominal layout key of the same kind with row parameters")]
    Def,
    #[display("`{ROW_ARGS_ATTR}` must be a list of rows")]
    MalformedRowArgs,
    #[display("row argument {index} is malformed: {reason}")]
    MalformedRowArg { index: usize, reason: String },
    #[display("{args} row argument(s) for a layout with {params} row parameter(s)")]
    Arity { args: usize, params: i128 },
}

/// Check the row instance `ty`: no name or parameters, a `def` that is a
/// key of the same kind named only by its name and row binders, and one
/// row argument for each binder.
fn verify_row_instance(ctx: &IrContext, ty: TypeRef) -> Result<(), RowInstanceError> {
    let data = ctx.get_type(ty);
    let only_instance_attrs = data
        .attrs
        .iter()
        .all(|(key, _)| [DEF_ATTR, ROW_ARGS_ATTR].contains(&ctx.str(key)));
    if !data.params.is_empty() || !only_instance_attrs {
        return Err(RowInstanceError::Shape);
    }
    let def = ctx
        .type_attrs(ty)
        .get_type(DEF_ATTR)
        .ok_or(RowInstanceError::Shape)?;
    let def_data = ctx.get_type(def);
    let only_name_and_binders = def_data
        .attrs
        .iter()
        .all(|(key, _)| [STRUCT_NAME_ATTR, ROW_PARAMS_ATTR].contains(&ctx.str(key)));
    let params = match ctx.type_attrs(def).get(ROW_PARAMS_ATTR) {
        Some(Attribute::Int(count)) if *count > 0 => *count,
        _ => return Err(RowInstanceError::Def),
    };
    if def_data.name != data.name
        || !def_data.params.is_empty()
        || ctx.type_attrs(def).get_str(STRUCT_NAME_ATTR).is_none()
        || !only_name_and_binders
    {
        return Err(RowInstanceError::Def);
    }
    let Some(Attribute::List(rows)) = ctx.type_attrs(ty).get(ROW_ARGS_ATTR) else {
        return Err(RowInstanceError::MalformedRowArgs);
    };
    for (index, row) in rows.iter().enumerate() {
        Row::from_attribute(ctx, row)
            .map_err(|reason| RowInstanceError::MalformedRowArg { index, reason })?;
    }
    if i128::try_from(rows.len()).ok() != Some(params) {
        return Err(RowInstanceError::Arity {
            args: rows.len(),
            params,
        });
    }
    Ok(())
}

// === Nominal reference type ===

/// A malformed `adt.typeref` type.
#[derive(Clone, Debug, PartialEq, Eq, derive_more::Display, derive_more::Error)]
pub enum TyperefTypeError {
    #[display("`{ROW_ARGS_ATTR}` must be a nonempty list of rows")]
    MalformedRowArgs,
    #[display("row argument {index} is malformed: {reason}")]
    MalformedRowArg { index: usize, reason: String },
}

impl trunk_ir::type_verifier::VerifyType for Typeref {
    fn verify(ctx: &IrContext, ty: TypeRef) -> Result<Self, trunk_ir::ops::VerificationError> {
        if let Some(args) = ctx.type_attrs(ty).get(ROW_ARGS_ATTR) {
            let Attribute::List(rows) = args else {
                return Err(TyperefTypeError::MalformedRowArgs.into());
            };
            if rows.is_empty() {
                return Err(TyperefTypeError::MalformedRowArgs.into());
            }
            for (index, row) in rows.iter().enumerate() {
                Row::from_attribute(ctx, row)
                    .map_err(|reason| TyperefTypeError::MalformedRowArg { index, reason })?;
            }
        }
        Ok(Self(ty))
    }
}

impl Typeref {
    /// A reference to the layout named `name`, passing `row_args` for its
    /// row parameters in order.
    pub fn named(
        ctx: &mut IrContext,
        name: impl Into<StringArg>,
        row_args: impl IntoIterator<Item = Row>,
    ) -> Self {
        let row_args: Vec<_> = row_args.into_iter().collect();
        let builder = Self::params().name(name);
        let builder = if row_args.is_empty() {
            builder
        } else {
            builder.row_args(row_args)
        };
        builder
            .intern(ctx)
            .unwrap_or_else(|error| panic!("adt.typeref: {error}"))
    }

    /// The nominal layout this typeref refers to: the `adt.struct` or
    /// `adt.enum` published as the type alias of its `name`, which a pass
    /// that gives the layout a new key publishes again, or without such an
    /// alias the layout whose key this typeref's attributes spell, with
    /// `row_args` replaced by their number as `row_params`
    /// (`design/ir.md` § "`adt.typeref` nominal reference type").
    pub fn layout(self, ctx: &IrContext) -> Option<TypeRef> {
        let is_layout = |ty: TypeRef| Struct::matches(ctx, ty) || Enum::matches(ctx, ty);
        if let Some(layout) = ctx
            .type_alias_by_text(self.name(ctx))
            .filter(|&alias| is_layout(alias))
        {
            return Some(layout);
        }
        let mut entries = Vec::new();
        for (key, value) in ctx.get_type(self.0).attrs.iter() {
            if ctx.str(key) != ROW_ARGS_ATTR {
                entries.push((key, value.clone()));
            }
        }
        if let Some(Attribute::List(rows)) = ctx.type_attrs(self.0).get(ROW_ARGS_ATTR) {
            // A layout with binders has interned the key.
            let key = ctx.lookup_str(ROW_PARAMS_ATTR)?;
            entries.push((key, Attribute::Int(rows.len() as i128)));
        }
        let attrs = AttributeMap::from_refs(ctx, entries);
        [<Struct as DialectType>::NAME, <Enum as DialectType>::NAME]
            .into_iter()
            .find_map(|name| {
                ctx.types().lookup(&trunk_ir::types::TypeData {
                    name,
                    params: Default::default(),
                    attrs: attrs.clone(),
                })
            })
            .filter(|&layout| ctx.type_body(layout).is_some())
    }

    /// The row arguments, one for each row parameter of the layout; empty
    /// when it has none.
    pub fn rows(self, ctx: &IrContext) -> Vec<Row> {
        match ctx.type_attrs(self.0).get(ROW_ARGS_ATTR) {
            Some(Attribute::List(rows)) => rows
                .iter()
                .map(|row| Row::from_attribute(ctx, row).expect("verified adt.typeref row"))
                .collect(),
            _ => Vec::new(),
        }
    }

    /// The type attributes other than the name and the row arguments.
    pub fn extra_attrs(
        self,
        ctx: &IrContext,
    ) -> impl Iterator<Item = (StringRef, &Attribute)> + '_ {
        ctx.get_type(self.0)
            .attrs
            .iter()
            .filter(|&(key, _)| ![STRUCT_NAME_ATTR, ROW_ARGS_ATTR].contains(&ctx.str(key)))
    }
}

/// The type of field `field` of a value of type `value_ty`, a nominal struct:
/// the layout's field type, with the layout's row binders replaced by the
/// value type's row arguments (`design/tribute-dialects.md` § "Nominal row
/// 인자").
/// `None` when `value_ty` has no struct layout or no such field.
pub fn field_type(ctx: &mut IrContext, value_ty: TypeRef, field: usize) -> Option<TypeRef> {
    let layout = Struct::from_type_ref(ctx, layout_of(ctx, value_ty)?)?;
    let ty = layout.field_type(ctx, field)?;
    substitute_layout_binders(ctx, layout.row_params(ctx), value_ty, ty)
}

/// `ty`, written in a layout with `row_params` binders, as seen through a
/// value of type `value_ty`: the layout's binders replaced by the value
/// type's row arguments. `None` when the layout has binders and `value_ty`
/// carries no row arguments.
pub fn substitute_layout_binders(
    ctx: &mut IrContext,
    row_params: Option<u32>,
    value_ty: TypeRef,
    ty: TypeRef,
) -> Option<TypeRef> {
    if row_params.is_none() {
        return Some(ty);
    }
    let rows = value_rows(ctx, value_ty);
    if rows.is_empty() {
        return None;
    }
    crate::dialect::tribute_control::substitute_rows(ctx, ty, &rows)
}

/// [`field_type`] for a value of type `value_ty` whose struct layout is
/// `layout`, without creating types: `None` also when the field type does
/// not exist, which no IR value can then have.
pub fn find_field_type(
    ctx: &IrContext,
    layout: Struct,
    value_ty: TypeRef,
    field: usize,
) -> Option<TypeRef> {
    let ty = layout.field_type(ctx, field)?;
    find_layout_binders(ctx, layout.row_params(ctx), value_ty, ty)
}

/// The type of field `field` of variant `tag` of a value of type
/// `value_ty` whose enum layout is `layout`, as [`find_field_type`] finds a
/// struct field's.
pub fn find_variant_field_type(
    ctx: &IrContext,
    layout: Enum,
    value_ty: TypeRef,
    tag: u32,
    field: usize,
) -> Option<TypeRef> {
    let ty = *layout.variant_fields(ctx, tag)?.get(field)?;
    find_layout_binders(ctx, layout.row_params(ctx), value_ty, ty)
}

/// [`substitute_layout_binders`] without creating types.
fn find_layout_binders(
    ctx: &IrContext,
    row_params: Option<u32>,
    value_ty: TypeRef,
    ty: TypeRef,
) -> Option<TypeRef> {
    if row_params.is_none() {
        return Some(ty);
    }
    let rows = value_rows(ctx, value_ty);
    if rows.is_empty() {
        return None;
    }
    crate::dialect::tribute_control::find_substituted_rows(ctx, ty, &rows)
}

/// `adt.typeref<"Name"[row(..), ..], {attrs}>`.
impl trunk_ir::asm_format::TypeAsmFormat for Typeref {
    fn print(&self, p: &mut trunk_ir::printer::AsmPrinter<'_>) -> std::fmt::Result {
        use itertools::Itertools as _;
        use std::fmt::Write;
        use trunk_ir::printer::display_string;

        let s = p.state();
        let ctx = s.ctx();
        write!(p, "adt.typeref<{}", display_string(self.name(ctx)))?;
        let rows = self.rows(ctx);
        if !rows.is_empty() {
            let rows = rows
                .iter()
                .map(crate::dialect::tribute_control::display_row_argument)
                .format(", ");
            write!(p, "[{rows}]")?;
        }
        let mut attrs = self.extra_attrs(ctx).peekable();
        if attrs.peek().is_some() {
            write!(p, ", {}", s.attr_dict(attrs))?;
        }
        p.write_char('>')
    }

    fn parse(
        p: &mut trunk_ir::parser::AsmParser<'_>,
        name: &str,
    ) -> trunk_ir::parser::ParseResult<Option<TypeRef>> {
        if !p.peek().starts_with('"') {
            return Ok(None);
        }
        let layout = p.parse_token("a layout name", trunk_ir::parser::lexer::string_lit)?;
        let rows = if p.parse_optional_punct("[") {
            let mut rows = Vec::new();
            loop {
                rows.push(crate::dialect::tribute_control::parse_row_argument(p)?);
                if !p.parse_optional_punct(",") {
                    break;
                }
            }
            p.parse_punct("]")?;
            rows
        } else {
            Vec::new()
        };
        let attrs = parse_layout_attrs(p)?;
        let mut attrs = named(p, attrs, &layout)?;
        if !rows.is_empty() {
            if attrs.view(p.ctx()).contains_key(ROW_ARGS_ATTR) {
                return Err(p.expected("attributes without `row_args`, which the syntax writes"));
            }
            let ctx = p.ctx_mut();
            let rows = rows.iter().map(|row| row.to_attribute(ctx)).collect();
            attrs
                .view_mut(ctx)
                .insert(ROW_ARGS_ATTR, Attribute::List(rows));
        }
        p.build_type(name, Vec::new(), attrs).map(Some)
    }
}

/// The nominal declarations of a module, by the layout each one declares
/// (`design/ir.md` § "Nominal 선언").
#[derive(Debug, Default)]
pub struct NominalDecls {
    structs: HashMap<TypeRef, StructDecl>,
    enums: HashMap<TypeRef, EnumDecl>,
}

impl NominalDecls {
    /// The declarations at the top level of `module`.
    pub fn collect(ctx: &IrContext, module: trunk_ir::rewrite::Module) -> Self {
        let mut decls = Self::default();
        let Some(block) = module.first_block(ctx) else {
            return decls;
        };
        for &op in &ctx.block(block).ops {
            if let Ok(decl) = StructDecl::from_op(ctx, op) {
                decls.structs.insert(decl.r#type(ctx), decl);
            } else if let Ok(decl) = EnumDecl::from_op(ctx, op) {
                decls.enums.insert(decl.r#type(ctx), decl);
            }
        }
        decls
    }

    /// The declaration of the struct layout `layout`.
    pub fn of_struct(&self, layout: Struct) -> Option<StructDecl> {
        self.structs.get(&layout.as_type_ref()).copied()
    }

    /// The declaration of the enum layout `layout`.
    pub fn of_enum(&self, layout: Enum) -> Option<EnumDecl> {
        self.enums.get(&layout.as_type_ref()).copied()
    }

    /// The position of the variant the declaration of `layout` names
    /// `variant`, for a pass that builds a variant of a compiler-owned enum
    /// by its source name.
    pub fn variant_position(&self, ctx: &IrContext, layout: Enum, variant: &str) -> Option<u32> {
        self.of_enum(layout)?.position_of(ctx, variant)
    }
}

/// A nominal declaration that does not fit its layout.
#[derive(Debug, derive_more::Display, derive_more::Error)]
pub enum NominalDeclError {
    #[display("`type` is not a nominal `adt.struct` layout with a body")]
    NotANamedStruct,
    #[display("`type` is not a named `adt.enum` layout with a body")]
    NotAnEnum,
    #[display("{names} name(s) for a layout with {positions} position(s)")]
    CountMismatch { names: usize, positions: usize },
}

impl trunk_ir::ops::Verify for StructDecl {
    fn verify(&self, ctx: &IrContext) -> Result<(), trunk_ir::ops::VerificationError> {
        let layout = Struct::from_type_ref(ctx, self.r#type(ctx))
            .filter(|layout| layout.body(ctx).is_some())
            .ok_or(NominalDeclError::NotANamedStruct)?;
        let (names, positions) = (self.fields(ctx).len(), layout.field_count(ctx));
        if names != positions {
            return Err(NominalDeclError::CountMismatch { names, positions }.into());
        }
        Ok(())
    }
}

impl trunk_ir::ops::Verify for EnumDecl {
    fn verify(&self, ctx: &IrContext) -> Result<(), trunk_ir::ops::VerificationError> {
        let layout = Enum::from_type_ref(ctx, self.r#type(ctx))
            .filter(|layout| layout.body(ctx).is_some())
            .ok_or(NominalDeclError::NotAnEnum)?;
        let (names, positions) = (self.variants(ctx).len(), layout.variant_count(ctx));
        if names != positions {
            return Err(NominalDeclError::CountMismatch { names, positions }.into());
        }
        Ok(())
    }
}

impl EnumDecl {
    /// The position of the variant this declaration names `variant`.
    pub fn position_of(self, ctx: &IrContext, variant: &str) -> Option<u32> {
        let position = self.variants(ctx).position(|name| name == variant)?;
        Some(u32::try_from(position).expect("an enum has fewer than 2^32 variants"))
    }
}

/// A variant operation whose `tag` is not a variant of its `type`.
#[derive(Debug, derive_more::Display, derive_more::Error)]
pub enum VariantTagError {
    #[display("`type` is not an `adt.enum` layout")]
    NotAnEnum,
    #[display("tag {tag} is not a variant of an enum with {count} variant(s)")]
    OutOfRange { tag: u32, count: usize },
}

/// Check that `tag` is the position of a variant of the enum layout `ty`.
fn verify_variant_tag(
    ctx: &IrContext,
    ty: TypeRef,
    tag: u32,
) -> Result<(), trunk_ir::ops::VerificationError> {
    let count = Enum::from_type_ref(ctx, ty)
        .ok_or(VariantTagError::NotAnEnum)?
        .variant_count(ctx);
    if usize::try_from(tag).is_ok_and(|tag| tag < count) {
        Ok(())
    } else {
        Err(VariantTagError::OutOfRange { tag, count }.into())
    }
}

impl trunk_ir::ops::Verify for VariantNew {
    fn verify(&self, ctx: &IrContext) -> Result<(), trunk_ir::ops::VerificationError> {
        verify_variant_tag(ctx, self.r#type(ctx), self.tag(ctx))
    }
}

impl trunk_ir::ops::Verify for VariantIs {
    fn verify(&self, ctx: &IrContext) -> Result<(), trunk_ir::ops::VerificationError> {
        verify_variant_tag(ctx, self.r#type(ctx), self.tag(ctx))
    }
}

impl trunk_ir::ops::Verify for VariantGet {
    fn verify(&self, ctx: &IrContext) -> Result<(), trunk_ir::ops::VerificationError> {
        verify_variant_tag(ctx, self.r#type(ctx), self.tag(ctx))
    }
}

/// Erases `adt.struct_decl` and `adt.enum_decl`, for the target pass that
/// lowers `adt` operations.
pub struct EraseNominalDecls;

impl trunk_ir::rewrite::RewritePattern for EraseNominalDecls {
    fn roots(&self) -> Option<&[trunk_ir::OpName]> {
        Some(&[StructDecl::NAME, EnumDecl::NAME])
    }

    fn match_and_rewrite(
        &self,
        _ctx: &mut IrContext,
        _op: trunk_ir::OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        rewriter.erase_op(vec![]);
        Some(())
    }
}

/// A malformed `adt.struct` type.
#[derive(Clone, Debug, PartialEq, Eq, derive_more::Display, derive_more::Error)]
pub enum StructTypeError {
    #[display("`{STRUCT_NAME_ATTR}` is not a string")]
    MalformedName,
    #[display("a literal `adt.struct` has no `{ROW_PARAMS_ATTR}`")]
    LiteralRowParams,
    #[display("a nominal `adt.struct` has no type parameters; its fields are in its body")]
    NominalParams,
    #[display("`fields` is not an `adt.struct` attribute; fields are type parameters")]
    FieldsAttribute,
    #[display("closure base fields are not a prefix of the closure capture layout")]
    ClosureBasePrefix,
}

impl trunk_ir::type_verifier::VerifyType for Struct {
    /// Validate a name-matching `adt.struct`: a nominal layout's key, or a
    /// literal layout. A key's body is checked when it is bound.
    fn verify(ctx: &IrContext, ty: TypeRef) -> Result<Self, trunk_ir::ops::VerificationError> {
        let data = ctx.get_type(ty);
        debug_assert!(Self::matches(ctx, ty));
        if ctx.type_attrs(ty).contains_key("fields") {
            return Err(StructTypeError::FieldsAttribute.into());
        }
        let attrs = ctx.type_attrs(ty);
        match attrs.get(STRUCT_NAME_ATTR) {
            Some(Attribute::String(_)) if !data.params.is_empty() => {
                return Err(StructTypeError::NominalParams.into());
            }
            Some(Attribute::String(_)) => {
                row_binders(ctx, ty)?;
                return Ok(Self(ty));
            }
            Some(_) => return Err(StructTypeError::MalformedName.into()),
            None if attrs.contains_key(DEF_ATTR) => {
                verify_row_instance(ctx, ty)?;
                return Ok(Self(ty));
            }
            // A row binder is bound by the declaration a nominal layout has.
            None if attrs.contains_key(ROW_PARAMS_ATTR) => {
                return Err(StructTypeError::LiteralRowParams.into());
            }
            None => {}
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

/// A layout's `row_params` that do not bind its field types' tails.
#[derive(Clone, Debug, PartialEq, Eq, derive_more::Display, derive_more::Error)]
pub enum LayoutRowError {
    #[display("`{ROW_PARAMS_ATTR}` counts at least one row binder")]
    NoBinders,
    #[display("row tail 'a{_0} names no binder of the layout")]
    UnboundTail(#[error(not(source))] u32),
    #[display("a field row is malformed: {_0}")]
    MalformedRow(#[error(not(source))] String),
}

/// The number of row binders the nominal layout `layout` declares, `0`
/// without `row_params`.
fn row_binders(ctx: &IrContext, layout: TypeRef) -> Result<i128, LayoutRowError> {
    match ctx.type_attrs(layout).get(ROW_PARAMS_ATTR) {
        None => Ok(0),
        Some(Attribute::Int(count)) if *count > 0 => Ok(*count),
        Some(_) => Err(LayoutRowError::NoBinders),
    }
}

/// A body that cannot be bound to a nominal layout.
#[derive(
    Clone, Debug, PartialEq, Eq, derive_more::Display, derive_more::Error, derive_more::From,
)]
pub enum LayoutBodyError {
    #[display(
        "the body of a nominal layout is a literal layout of the same kind without attributes"
    )]
    NotALiteral,
    #[display("{_0}")]
    Rows(LayoutRowError),
    #[display("the layout already has a different body")]
    Conflicting,
}

/// Bind `body` to the nominal layout `key` after checking that it is a
/// literal layout of `key`'s kind without attributes whose row tails name
/// binders of `key` (`design/ir.md` § "`adt.struct` nominal layout type").
fn bind_layout_body(
    ctx: &mut IrContext,
    key: TypeRef,
    body: TypeRef,
) -> Result<(), LayoutBodyError> {
    check_layout_body(ctx, key, body)?;
    ctx.bind_type_body(key, body)
        .map_err(|_| LayoutBodyError::Conflicting)
}

fn check_layout_body(ctx: &IrContext, key: TypeRef, body: TypeRef) -> Result<(), LayoutBodyError> {
    let (key_data, body_data) = (ctx.get_type(key), ctx.get_type(body));
    let literal = body_data.name == key_data.name
        && body_data
            .attrs
            .iter()
            .all(|(name, _)| ctx.str(name) == PARAM_ATTRS_ATTR);
    if !literal {
        return Err(LayoutBodyError::NotALiteral);
    }
    let fields = if Enum::matches(ctx, body) {
        body_data
            .params
            .iter()
            .flat_map(|&variant| ctx.get_type(variant).params.iter().copied())
            .collect()
    } else {
        body_data.params.to_vec()
    };
    // Without binders, a tail names a binder of the definition that writes
    // the layout, as in a tuple layout.
    let binders = row_binders(ctx, key)?;
    if binders > 0 {
        verify_row_params(ctx, binders, fields)?;
    }
    Ok(())
}

/// Check that a layout with `binders` row binders writes only those binders
/// as tails in `fields` (`design/tribute-dialects.md` § "Nominal row 인자").
fn verify_row_params(
    ctx: &IrContext,
    binders: i128,
    fields: Vec<TypeRef>,
) -> Result<(), LayoutRowError> {
    let mut error = None;
    for field in fields {
        crate::dialect::tribute_control::visit_rows(ctx, field, &mut |row| {
            let checked =
                row.map_err(LayoutRowError::MalformedRow)
                    .and_then(|row| match row.tail {
                        Some(tail) if i128::from(tail) >= binders => {
                            Err(LayoutRowError::UnboundTail(tail))
                        }
                        _ => Ok(()),
                    });
            if let Err(found) = checked {
                error.get_or_insert(found);
            }
        });
    }
    error.map_or(Ok(()), Err)
}

impl Struct {
    /// Whether this is a nominal layout, an identified type, rather than a
    /// literal one.
    pub fn is_nominal(self, ctx: &IrContext) -> bool {
        self.name(ctx).is_some()
    }

    /// The literal layout holding the fields: the body of a nominal layout,
    /// which must have one, that of a row instance's `def`, or a literal
    /// layout itself.
    pub fn literal_layout(self, ctx: &IrContext) -> Struct {
        if let Some(def) = row_instance_def(ctx, self.0) {
            return Struct(def).literal_layout(ctx);
        }
        if self.is_nominal(ctx) {
            let body = ctx.type_body(self.0).unwrap_or_else(|| {
                panic!(
                    "nominal adt.struct {:?} has no body",
                    self.name(ctx).unwrap_or_default()
                )
            });
            Struct(body)
        } else {
            self
        }
    }

    /// The body of a nominal layout, if it has one.
    pub fn body(self, ctx: &IrContext) -> Option<Struct> {
        ctx.type_body(self.0).map(Struct)
    }

    pub fn field_count(self, ctx: &IrContext) -> usize {
        self.fields(ctx).len()
    }

    /// The type of field `index`, if it exists.
    pub fn field_type(self, ctx: &IrContext, index: usize) -> Option<TypeRef> {
        self.fields(ctx).get(index).copied()
    }

    /// Each field's type, in declaration order.
    pub fn fields(self, ctx: &IrContext) -> &[TypeRef] {
        &ctx.get_type(self.literal_layout(ctx).0).params
    }

    /// The attributes of field `index`.
    pub fn field_attrs(self, ctx: &IrContext, index: usize) -> &AttributeMap {
        ctx.get_type(self.literal_layout(ctx).0)
            .param_attrs(ctx, index)
    }

    /// The type attributes other than the name, the row binders, and the
    /// field attributes: a nominal layout's other key attributes, or a
    /// literal layout's attributes.
    pub fn extra_attrs(
        self,
        ctx: &IrContext,
    ) -> impl Iterator<Item = (StringRef, &Attribute)> + '_ {
        ctx.get_type(self.0).attrs.iter().filter(|&(key, _)| {
            ![STRUCT_NAME_ATTR, ROW_PARAMS_ATTR, PARAM_ATTRS_ATTR].contains(&ctx.str(key))
        })
    }
}

impl Struct {
    /// Construct the nominal `adt.struct` named `name` with `fields` in
    /// declaration order: the key with `attrs` as its other key attributes,
    /// bound to the literal body of `fields`.
    ///
    /// # Panics
    ///
    /// If the key is malformed or already has another body.
    pub fn from_fields(
        ctx: &mut IrContext,
        name: impl Into<StringArg>,
        fields: impl IntoIterator<Item = TypeRef>,
        attrs: AttributeMap,
    ) -> Self {
        let key = Self::key(ctx, name, attrs);
        let body = Self::literal(ctx, fields, AttributeMap::new());
        key.bind_body(ctx, body).unwrap_or_else(|error| {
            panic!(
                "adt.struct {:?}: {error}",
                key.name(ctx).unwrap_or_default()
            )
        });
        key
    }

    /// The key of the nominal `adt.struct` named `name` with `attrs` as its
    /// other key attributes, without binding a body.
    ///
    /// # Panics
    ///
    /// If the key is malformed.
    pub fn key(ctx: &mut IrContext, name: impl Into<StringArg>, attrs: AttributeMap) -> Self {
        let name = ctx.intern_string_arg(name.into());
        Self::build(ctx, Some(name), [], attrs)
    }

    /// Bind the literal layout `body` to this nominal layout's key.
    pub fn bind_body(self, ctx: &mut IrContext, body: Struct) -> Result<(), LayoutBodyError> {
        bind_layout_body(ctx, self.0, body.0)
    }

    /// Construct the literal `adt.struct` of `fields` in declaration order,
    /// for a compiler storage layout no `adt.typeref` names.
    ///
    /// `attrs` holds the type attributes, such as `layout`.
    ///
    /// # Panics
    ///
    /// If `attrs` has a `fields` or `name` attribute.
    pub fn literal(
        ctx: &mut IrContext,
        fields: impl IntoIterator<Item = TypeRef>,
        attrs: AttributeMap,
    ) -> Self {
        assert!(
            !attrs.view(ctx).contains_key(STRUCT_NAME_ATTR),
            "a literal adt.struct has no name"
        );
        Self::build(ctx, None, fields, attrs)
    }

    fn build(
        ctx: &mut IrContext,
        name: Option<StringRef>,
        fields: impl IntoIterator<Item = TypeRef>,
        mut attrs: AttributeMap,
    ) -> Self {
        let mut builder = TypeDataBuilder::new(<Struct as DialectType>::NAME);
        for ty in fields {
            builder = builder.param(ty);
        }
        attrs.view_mut(ctx).remove(PARAM_ATTRS_ATTR);
        if let Some(name) = name {
            attrs
                .view_mut(ctx)
                .insert(STRUCT_NAME_ATTR, Attribute::String(name));
        }
        for (key, value) in attrs {
            builder = builder.attr(key, value);
        }
        let ty = builder.intern(ctx);
        <Struct as trunk_ir::type_verifier::VerifyType>::verify(ctx, ty)
            .unwrap_or_else(|error| panic!("adt.struct: {error}"))
    }
}

// === Textual syntax: `adt.struct<"Name", (type {attrs}, ...), {attrs}>` ===
//
// The name is always a quoted string. A nominal layout's key writes no field
// list where its body is not written: `adt.struct<"Name", {attrs}>`. A
// literal layout writes no name: `adt.struct<(type {attrs}, ...), {attrs}>`.

impl trunk_ir::asm_format::TypeAsmFormat for Struct {
    fn print(&self, p: &mut trunk_ir::printer::AsmPrinter<'_>) -> std::fmt::Result {
        write_struct_type(p, *self)
    }

    fn parse(
        p: &mut trunk_ir::parser::AsmParser<'_>,
        name: &str,
    ) -> trunk_ir::parser::ParseResult<Option<TypeRef>> {
        parse_struct_type(p, name)
    }
}

fn write_struct_type(
    p: &mut trunk_ir::printer::AsmPrinter<'_>,
    adt_struct: Struct,
) -> std::fmt::Result {
    use std::fmt::Write;
    use trunk_ir::printer::display_string;

    let s = p.state();
    let ctx = s.ctx();
    p.write_str("adt.struct<")?;
    if row_instance_def(ctx, adt_struct.as_type_ref()).is_some() {
        return write_row_instance(p, adt_struct.as_type_ref());
    }
    if let Some(name) = adt_struct.name(ctx) {
        write!(
            p,
            "{}{}",
            display_string(name),
            display_layout_binders(adt_struct.row_params(ctx).unwrap_or(0))
        )?;
        if let Some(body) = p.identified_body(adt_struct.as_type_ref()) {
            write!(p, ", ({})", param_list(s, ctx.get_type(body.ty())))?;
        }
    } else {
        write!(
            p,
            "({})",
            param_list(s, ctx.get_type(adt_struct.as_type_ref()))
        )?;
    }
    let mut attrs = adt_struct.extra_attrs(ctx).peekable();
    if attrs.peek().is_some() {
        write!(p, ", {}", s.attr_dict(attrs))?;
    }
    p.write_char('>')
}

/// The parameters of `data` as a layout writes a field list: each type,
/// followed by its parameter attributes when it has any.
fn param_list<'a>(
    s: &'a trunk_ir::printer::AsmState<'a>,
    data: &'a trunk_ir::types::TypeData,
) -> impl std::fmt::Display + 'a {
    use itertools::Itertools as _;

    let ctx = s.ctx();
    data.params
        .iter()
        .enumerate()
        .format_with(", ", move |(index, &ty), f| {
            f(&s.ty(ty))?;
            let attrs = data.param_attrs(ctx, index);
            if !attrs.is_empty() {
                f(&format_args!(" {}", s.attr_dict(attrs.iter())))?;
            }
            Ok(())
        })
}

/// Parse a layout's field list, `(type {attrs}, ...)`.
fn parse_field_list(
    p: &mut trunk_ir::parser::AsmParser<'_>,
) -> trunk_ir::parser::ParseResult<Vec<(TypeRef, AttributeMap)>> {
    p.parse_delimited("(", ")", |p| p.parse_param())
}

/// What follows a layout's quoted name: its binder list, possibly empty, or
/// the row arguments of a row instance.
enum NameSuffix {
    Binders(u32),
    RowArgs(Vec<Row>),
}

/// Parse a layout's quoted name and what follows it, when the text after
/// the opening bracket begins with a string; `None`, reading nothing,
/// otherwise. The generic spelling begins with a type or a dictionary. A
/// list whose first element begins with `row(` is a row instance's row
/// arguments; any other list is the binder list.
fn parse_layout_name(
    p: &mut trunk_ir::parser::AsmParser<'_>,
    what: &'static str,
) -> trunk_ir::parser::ParseResult<Option<(String, NameSuffix)>> {
    if !p.peek().starts_with('"') {
        return Ok(None);
    }
    let layout_name = p.parse_token(what, trunk_ir::parser::lexer::string_lit)?;
    let rest = p.peek();
    let row_args = rest
        .strip_prefix('[')
        .is_some_and(|rest| rest.trim_start().starts_with("row("));
    if !row_args {
        let binders = parse_layout_binders(p)?;
        return Ok(Some((layout_name, NameSuffix::Binders(binders))));
    }
    p.parse_punct("[")?;
    let mut rows = Vec::new();
    loop {
        rows.push(crate::dialect::tribute_control::parse_row_argument(p)?);
        if !p.parse_optional_punct(",") {
            break;
        }
    }
    p.parse_punct("]")?;
    Ok(Some((layout_name, NameSuffix::RowArgs(rows))))
}

/// Intern the row instance of type `name` applying the layout named
/// `layout_name` to `rows`, after its `>`: its `def` is the key of that
/// name with one row binder for each row.
fn build_row_instance(
    p: &mut trunk_ir::parser::AsmParser<'_>,
    name: &str,
    layout_name: &str,
    rows: Vec<Row>,
) -> trunk_ir::parser::ParseResult<TypeRef> {
    p.parse_punct(">")?;
    let binders = u32::try_from(rows.len()).map_err(|_| p.expected("fewer row arguments"))?;
    let attrs = named(p, AttributeMap::new(), layout_name)?;
    let attrs = with_binders(p, attrs, binders)?;
    let key = p.build_type(name, Vec::new(), attrs)?;
    let ctx = p.ctx_mut();
    let rows = rows.iter().map(|row| row.to_attribute(ctx)).collect();
    let attrs = AttributeMap::from_entries(
        ctx,
        [
            (DEF_ATTR, Attribute::Type(key)),
            (ROW_ARGS_ATTR, Attribute::List(rows)),
        ],
    );
    p.build_type(name, Vec::new(), attrs)
}

/// `"Name"[row(..), ..]>`, the rest of a row instance after its opening
/// bracket: the name of its `def` and its row arguments.
fn write_row_instance(
    p: &mut trunk_ir::printer::AsmPrinter<'_>,
    instance: TypeRef,
) -> std::fmt::Result {
    use itertools::Itertools as _;
    use std::fmt::Write;
    use trunk_ir::printer::display_string;

    let ctx = p.state().ctx();
    let name = nominal_name(ctx, instance).expect("a row instance's def is named");
    let rows = value_rows(ctx, instance);
    let rows = rows
        .iter()
        .map(crate::dialect::tribute_control::display_row_argument)
        .format(", ");
    write!(p, "{}[{rows}]>", display_string(name))
}

/// Parse the rest of a nominal layout after its name and binders: the
/// parameters of its body, which `parse_body` reads where the text begins
/// with `open`, if the text writes the body, and then its key attributes
/// and `>`. Returns the parameters with the offset they begin at.
#[expect(clippy::type_complexity)]
fn parse_nominal_rest(
    p: &mut trunk_ir::parser::AsmParser<'_>,
    open: &str,
    parse_body: impl FnOnce(
        &mut trunk_ir::parser::AsmParser<'_>,
    ) -> trunk_ir::parser::ParseResult<Vec<(TypeRef, AttributeMap)>>,
) -> trunk_ir::parser::ParseResult<(Option<(usize, Vec<(TypeRef, AttributeMap)>)>, AttributeMap)> {
    if !p.parse_optional_punct(",") {
        p.parse_punct(">")?;
        return Ok((None, AttributeMap::new()));
    }
    if !p.peek().starts_with(open) {
        let attrs = p.parse_attr_dict()?;
        p.parse_punct(">")?;
        return Ok((None, attrs));
    }
    let offset = p.offset();
    let params = parse_body(p)?;
    let attrs = parse_layout_attrs(p)?;
    Ok((Some((offset, params)), attrs))
}

/// Intern the key of the nominal layout of type `name` named `layout_name`
/// with `binders` and `attrs`, and bind the literal body of `body`, written
/// at its offset, when the text writes one.
fn build_nominal_layout(
    p: &mut trunk_ir::parser::AsmParser<'_>,
    name: &str,
    layout_name: &str,
    binders: u32,
    body: Option<(usize, Vec<(TypeRef, AttributeMap)>)>,
    attrs: AttributeMap,
) -> trunk_ir::parser::ParseResult<TypeRef> {
    let attrs = named(p, attrs, layout_name)?;
    let attrs = with_binders(p, attrs, binders)?;
    let key = p.build_type(name, Vec::new(), attrs)?;
    if let Some((offset, params)) = body {
        let body = p.build_type(name, params, AttributeMap::new())?;
        check_layout_body(p.ctx(), key, body)
            .map_err(|error| p.error_at(offset, error.to_string()))?;
        p.bind_type_body(key, body, offset)?;
    }
    Ok(key)
}

/// Parse the rest of `adt.struct<"Name"['a0], (type {attrs}, ...), {attrs}>`
/// after its opening bracket. The struct's name becomes the key's `name`
/// attribute and its binder list, if any, its `row_params`; the type
/// dictionary may hold neither. The field list, when written, is the body
/// bound to the key. A literal layout begins with its field list and its
/// dictionary may hold no `name`. Text that begins with neither a string
/// nor `(` is the generic spelling.
fn parse_struct_type(
    p: &mut trunk_ir::parser::AsmParser<'_>,
    name: &str,
) -> trunk_ir::parser::ParseResult<Option<TypeRef>> {
    if p.peek().starts_with('(') {
        let params = parse_field_list(p)?;
        let attrs = parse_layout_attrs(p)?;
        if attrs.view(p.ctx()).contains_key(STRUCT_NAME_ATTR) {
            return Err(p.expected("a literal layout's attributes without `name`"));
        }
        return p.build_type(name, params, attrs).map(Some);
    }
    let Some((struct_name, suffix)) = parse_layout_name(p, "a struct name")? else {
        return Ok(None);
    };
    let binders = match suffix {
        NameSuffix::Binders(binders) => binders,
        NameSuffix::RowArgs(rows) => {
            return build_row_instance(p, name, &struct_name, rows).map(Some);
        }
    };
    let (body, attrs) = parse_nominal_rest(p, "(", parse_field_list)?;
    build_nominal_layout(p, name, &struct_name, binders, body, attrs).map(Some)
}

/// `attrs` with the `row_params` a layout's binder list writes, which
/// `attrs` may not hold.
fn with_binders(
    p: &mut trunk_ir::parser::AsmParser<'_>,
    mut attrs: AttributeMap,
    binders: u32,
) -> trunk_ir::parser::ParseResult<AttributeMap> {
    if attrs.view(p.ctx()).contains_key(ROW_PARAMS_ATTR) {
        return Err(p.expected("attributes without `row_params`, which the syntax writes"));
    }
    if binders > 0 {
        attrs
            .view_mut(p.ctx_mut())
            .insert(ROW_PARAMS_ATTR, Attribute::Int(i128::from(binders)));
    }
    Ok(attrs)
}

/// The type attributes after a layout's body, `, {attrs}`, and its `>`.
fn parse_layout_attrs(
    p: &mut trunk_ir::parser::AsmParser<'_>,
) -> trunk_ir::parser::ParseResult<AttributeMap> {
    let attrs = if p.parse_optional_punct(",") {
        p.parse_attr_dict()?
    } else {
        AttributeMap::new()
    };
    p.parse_punct(">")?;
    Ok(attrs)
}

/// `attrs` with the `name` the syntax writes, which `attrs` may not hold.
fn named(
    p: &mut trunk_ir::parser::AsmParser<'_>,
    mut attrs: AttributeMap,
    name: &str,
) -> trunk_ir::parser::ParseResult<AttributeMap> {
    if attrs.view(p.ctx()).contains_key(STRUCT_NAME_ATTR) {
        return Err(p.expected("attributes without `name`, which the syntax writes"));
    }
    let name = p.ctx_mut().string_attr(name);
    attrs.view_mut(p.ctx_mut()).insert(STRUCT_NAME_ATTR, name);
    Ok(attrs)
}

// === Nominal enum layout type ===

/// A malformed `adt.enum` type.
#[derive(Clone, Debug, PartialEq, Eq, derive_more::Display, derive_more::Error)]
pub enum EnumTypeError {
    #[display("`{STRUCT_NAME_ATTR}` is not a string")]
    MalformedName,
    #[display("a literal `adt.enum` has no type attributes")]
    LiteralAttributes,
    #[display("a named `adt.enum` has no type parameters; its variants are in its body")]
    NominalParams,
    #[display("variant {_0} is not an `adt.variant`")]
    MalformedVariant(#[error(not(source))] usize),
    #[display("`variants` is not an `adt.enum` attribute; variants are type parameters")]
    VariantsAttribute,
    #[display("an `adt.enum` variant parameter carries no parameter attributes")]
    VariantAttributes,
}

impl trunk_ir::type_verifier::VerifyType for Enum {
    /// Validate a name-matching `adt.enum`: a named enum's key, or a
    /// literal enum. A key's body is checked when it is bound.
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
        match attrs.get(STRUCT_NAME_ATTR) {
            Some(Attribute::String(_)) if !data.params.is_empty() => {
                return Err(EnumTypeError::NominalParams.into());
            }
            Some(Attribute::String(_)) => {
                row_binders(ctx, ty)?;
                return Ok(Self(ty));
            }
            Some(_) => return Err(EnumTypeError::MalformedName.into()),
            None if attrs.contains_key(DEF_ATTR) => {
                verify_row_instance(ctx, ty)?;
                return Ok(Self(ty));
            }
            None if !data.attrs.is_empty() => {
                return Err(EnumTypeError::LiteralAttributes.into());
            }
            None => {}
        }
        for (index, &variant) in data.params.iter().enumerate() {
            // The syntax writes a variant as its fields and their attributes
            // alone.
            if Variant::from_type_ref(ctx, variant).is_none()
                || ctx
                    .type_attrs(variant)
                    .iter()
                    .any(|(key, _)| key != PARAM_ATTRS_ATTR)
            {
                return Err(EnumTypeError::MalformedVariant(index).into());
            }
        }
        Ok(Self(ty))
    }
}

impl Enum {
    /// Whether this is a named enum, an identified type, rather than a
    /// literal one.
    pub fn is_nominal(self, ctx: &IrContext) -> bool {
        self.name(ctx).is_some()
    }

    /// The literal enum holding the variants: the body of a named enum,
    /// which must have one, that of a row instance's `def`, or a literal
    /// enum itself.
    pub fn literal_layout(self, ctx: &IrContext) -> Enum {
        if let Some(def) = row_instance_def(ctx, self.0) {
            return Enum(def).literal_layout(ctx);
        }
        if self.is_nominal(ctx) {
            let body = ctx.type_body(self.0).unwrap_or_else(|| {
                panic!(
                    "named adt.enum {:?} has no body",
                    self.name(ctx).unwrap_or_default()
                )
            });
            Enum(body)
        } else {
            self
        }
    }

    /// The body of a named enum, if it has one.
    pub fn body(self, ctx: &IrContext) -> Option<Enum> {
        ctx.type_body(self.0).map(Enum)
    }

    /// The `adt.variant` types, in declaration order.
    fn variant_types(self, ctx: &IrContext) -> &[TypeRef] {
        &ctx.get_type(self.literal_layout(ctx).0).params
    }

    pub fn variant_count(self, ctx: &IrContext) -> usize {
        self.variant_types(ctx).len()
    }

    /// Each variant's field types, in declaration order.
    pub fn variants(self, ctx: &IrContext) -> impl ExactSizeIterator<Item = &[TypeRef]> + '_ {
        self.variant_types(ctx)
            .iter()
            .map(|&variant| &ctx.get_type(variant).params[..])
    }

    /// The field types of the variant at position `tag`.
    pub fn variant_fields(self, ctx: &IrContext, tag: u32) -> Option<&[TypeRef]> {
        let variant = *self.variant_types(ctx).get(usize::try_from(tag).ok()?)?;
        Some(&ctx.get_type(variant).params[..])
    }

    /// The type attributes other than the name and the row binders.
    pub fn extra_attrs(
        self,
        ctx: &IrContext,
    ) -> impl Iterator<Item = (StringRef, &Attribute)> + '_ {
        ctx.get_type(self.0)
            .attrs
            .iter()
            .filter(|&(key, _)| ![STRUCT_NAME_ATTR, ROW_PARAMS_ATTR].contains(&ctx.str(key)))
    }
}

impl Enum {
    /// Construct the `adt.enum` named `name` with `variants` in declaration
    /// order, each its field types: the key with `attrs` as its other key
    /// attributes, bound to the literal body of `variants`.
    ///
    /// # Panics
    ///
    /// If the key is malformed or already has another body.
    pub fn from_variants<F: IntoIterator<Item = TypeRef>>(
        ctx: &mut IrContext,
        name: impl Into<StringArg>,
        variants: impl IntoIterator<Item = F>,
        attrs: AttributeMap,
    ) -> Self {
        let key = Self::key(ctx, name, attrs);
        let body = Self::literal(ctx, variants);
        key.bind_body(ctx, body).unwrap_or_else(|error| {
            panic!("adt.enum {:?}: {error}", key.name(ctx).unwrap_or_default())
        });
        key
    }

    /// The key of the `adt.enum` named `name` with `attrs` as its other key
    /// attributes, without binding a body.
    ///
    /// # Panics
    ///
    /// If the key is malformed.
    pub fn key(ctx: &mut IrContext, name: impl Into<StringArg>, mut attrs: AttributeMap) -> Self {
        let name = ctx.intern_string_arg(name.into());
        attrs.view_mut(ctx).insert(STRUCT_NAME_ATTR, name);
        let mut builder = TypeDataBuilder::new(<Enum as DialectType>::NAME);
        for (key, value) in attrs {
            builder = builder.attr(key, value);
        }
        let ty = builder.intern(ctx);
        <Enum as trunk_ir::type_verifier::VerifyType>::verify(ctx, ty)
            .unwrap_or_else(|error| panic!("adt.enum: {error}"))
    }

    /// The literal `adt.enum` of `variants` in declaration order, each its
    /// field types: the body of a named enum.
    pub fn literal<F: IntoIterator<Item = TypeRef>>(
        ctx: &mut IrContext,
        variants: impl IntoIterator<Item = F>,
    ) -> Self {
        let mut builder = TypeDataBuilder::new(<Enum as DialectType>::NAME);
        for fields in variants {
            let variant = Variant::params(fields).intern(ctx);
            builder = builder.param(variant.as_type_ref());
        }
        Enum(builder.intern(ctx))
    }

    /// Bind the literal enum `body` to this named enum's key.
    pub fn bind_body(self, ctx: &mut IrContext, body: Enum) -> Result<(), LayoutBodyError> {
        bind_layout_body(ctx, self.0, body.0)
    }
}

// === Textual syntax: `adt.enum<"Name", [(type {attrs}, ...), ...], {attrs}>` ===
//
// Each variant is its field list, as an `adt.struct` writes its fields. A
// named enum's key writes no variant list where its body is not written:
// `adt.enum<"Name", {attrs}>`. A literal enum writes no name:
// `adt.enum<[(type {attrs}, ...), ...]>`.

impl trunk_ir::asm_format::TypeAsmFormat for Enum {
    fn print(&self, p: &mut trunk_ir::printer::AsmPrinter<'_>) -> std::fmt::Result {
        write_enum_type(p, *self)
    }

    fn parse(
        p: &mut trunk_ir::parser::AsmParser<'_>,
        name: &str,
    ) -> trunk_ir::parser::ParseResult<Option<TypeRef>> {
        parse_enum_type(p, name)
    }
}

fn write_enum_type(p: &mut trunk_ir::printer::AsmPrinter<'_>, adt_enum: Enum) -> std::fmt::Result {
    use itertools::Itertools as _;
    use std::fmt::Write;
    use trunk_ir::printer::display_string;

    let s = p.state();
    let ctx = s.ctx();
    let variants = |literal: TypeRef| {
        ctx.get_type(literal)
            .params
            .iter()
            .format_with(", ", |&variant, f| {
                f(&format_args!("({})", param_list(s, ctx.get_type(variant))))
            })
    };
    p.write_str("adt.enum<")?;
    if row_instance_def(ctx, adt_enum.as_type_ref()).is_some() {
        return write_row_instance(p, adt_enum.as_type_ref());
    }
    if let Some(name) = adt_enum.name(ctx) {
        write!(
            p,
            "{}{}",
            display_string(name),
            display_layout_binders(adt_enum.row_params(ctx).unwrap_or(0)),
        )?;
        if let Some(body) = p.identified_body(adt_enum.as_type_ref()) {
            write!(p, ", [{}]", variants(body.ty()))?;
        }
    } else {
        write!(p, "[{}]", variants(adt_enum.as_type_ref()))?;
    }
    let mut attrs = adt_enum.extra_attrs(ctx).peekable();
    if attrs.peek().is_some() {
        write!(p, ", {}", s.attr_dict(attrs))?;
    }
    p.write_char('>')
}

/// Parse a variant list, `[(type {attrs}, ...), ...]`: each field list
/// becomes an `adt.variant` parameter.
fn parse_variant_list(
    p: &mut trunk_ir::parser::AsmParser<'_>,
) -> trunk_ir::parser::ParseResult<Vec<(TypeRef, AttributeMap)>> {
    let variants = p.parse_delimited("[", "]", parse_field_list)?;
    let mut params = Vec::with_capacity(variants.len());
    for fields in variants {
        let ty = p.build_type("adt.variant", fields, AttributeMap::new())?;
        params.push((ty, AttributeMap::new()));
    }
    Ok(params)
}

/// Parse the rest of `adt.enum<"Name", [(type {attrs}, ...), ...], {attrs}>`
/// after its opening bracket. The enum's name becomes the key's `name`
/// attribute, which the type dictionary may not hold, and the variant list,
/// when written, is the body bound to the key. A literal enum begins with
/// its variant list. Text that begins with neither a string nor `[` is the
/// generic spelling.
fn parse_enum_type(
    p: &mut trunk_ir::parser::AsmParser<'_>,
    name: &str,
) -> trunk_ir::parser::ParseResult<Option<TypeRef>> {
    if p.peek().starts_with('[') {
        let params = parse_variant_list(p)?;
        p.parse_punct(">")?;
        return p.build_type(name, params, AttributeMap::new()).map(Some);
    }
    let Some((enum_name, suffix)) = parse_layout_name(p, "an enum name")? else {
        return Ok(None);
    };
    let binders = match suffix {
        NameSuffix::Binders(binders) => binders,
        NameSuffix::RowArgs(rows) => {
            return build_row_instance(p, name, &enum_name, rows).map(Some);
        }
    };
    let (body, attrs) = parse_nominal_rest(p, "[", parse_variant_list)?;
    build_nominal_layout(p, name, &enum_name, binders, body, attrs).map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustc_hash::FxHashMap as HashMap;
    use trunk_ir::SymbolPath;
    use trunk_ir::parser::{parse_module, parse_test_module};
    use trunk_ir::rewrite::Module;
    use trunk_ir::validation::validate_operation_verifiers;

    /// Print `module`, parse the output, and check that it prints the same.
    fn assert_roundtrip(ctx: &IrContext, module: trunk_ir::refs::OpRef) {
        let printed = module.display(ctx).to_string();
        let mut reparsed = IrContext::new();
        let module = parse_module(&mut reparsed, &printed).expect("printed IR must parse");
        assert_eq!(printed, module.display(&reparsed).to_string());
    }

    #[test]
    fn nominal_declarations_are_found_by_their_layouts() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !Point = adt.struct<"Point", (core.i32, core.i32)>
  !Empty = adt.struct<"Empty", ()>
  !Never = adt.enum<"Never", []>
  !Quoted = adt.struct<"Quoted", (core.i32, core.i32)>
  !"Shape$abi" = adt.struct<"Shape$abi", (core.i32)>
  adt.struct_decl "Point" !Point ["x", "y"]
  adt.struct_decl "Empty" !Empty []
  adt.enum_decl "Never" !Never []
  adt.struct_decl "Quoted" !Quoted ["0", "a b"]
  adt.struct_decl "Shape" !"Shape$abi" ["x"]
}"#,
        );
        assert_roundtrip(&ctx, module.op());
        assert!(validate_operation_verifiers(&ctx, module).is_ok());
        let decls = NominalDecls::collect(&ctx, module);
        let alias = |ctx: &IrContext, name| ctx.type_alias_by_text(name).expect("fixture alias");
        let point = Struct::from_type_ref(&ctx, alias(&ctx, "Point")).unwrap();
        let point = decls.of_struct(point).expect("Point is declared");
        assert_eq!(point.name(&ctx), "Point");
        assert_eq!(point.fields(&ctx).collect::<Vec<_>>(), ["x", "y"]);
        let empty = Struct::from_type_ref(&ctx, alias(&ctx, "Empty")).unwrap();
        let empty = decls.of_struct(empty).expect("Empty is declared");
        assert_eq!(empty.fields(&ctx).len(), 0);
        let never = Enum::from_type_ref(&ctx, alias(&ctx, "Never")).unwrap();
        let never = decls.of_enum(never).expect("Never is declared");
        assert_eq!(never.variants(&ctx).len(), 0);

        // A converted layout keeps its source name in its declaration.
        let shape = Struct::key(&mut ctx, "Shape$abi", AttributeMap::new());
        let shape = decls.of_struct(shape).expect("Shape is declared");
        assert_eq!(shape.name(&ctx), "Shape");

        // A layout that only shares the name of a declared one has no
        // declaration.
        let mut attrs = AttributeMap::new();
        attrs.view_mut(&mut ctx).insert("k", Attribute::Int(1));
        let other = Struct::from_fields(&mut ctx, "Point", [], attrs);
        assert!(decls.of_struct(other).is_none());
    }

    #[test]
    fn nominal_declarations_must_fit_their_layouts() {
        for decl in [
            r#"adt.struct_decl "P" adt.struct<"P", (core.i32)> ["x", "y"]"#,
            r#"adt.struct_decl "P" adt.struct<(core.i32)> ["x"]"#,
            r#"adt.enum_decl "P" adt.struct<"P", ()> []"#,
            r#"adt.enum_decl "E" adt.enum<"E", [()]> []"#,
        ] {
            let mut ctx = IrContext::new();
            let module =
                parse_test_module(&mut ctx, &format!("core.module @test {{\n  {decl}\n}}"));
            assert!(
                !validate_operation_verifiers(&ctx, module).is_ok(),
                "{decl} should be rejected"
            );
        }
    }

    #[test]
    fn layout_of_resolves_a_typeref_to_its_published_layout() {
        let mut ctx = IrContext::new();
        parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !Point = adt.struct<"Point", (core.i32)>
  !PointRef = adt.typeref<{name = "Point"}>
  !Shape = adt.enum<"Shape", [(), (core.i32)]>
  !ShapeRef = adt.typeref<{name = "Shape"}>
  !Number = core.i32
  !NumberRef = adt.typeref<{name = "Number"}>
  !MissingRef = adt.typeref<{name = "Missing"}>
}"#,
        );
        let alias = |ctx: &IrContext, name| ctx.type_alias_by_text(name).expect("fixture alias");
        let point = alias(&ctx, "Point");
        let shape = alias(&ctx, "Shape");

        assert_eq!(layout_of(&ctx, alias(&ctx, "PointRef")), Some(point));
        assert_eq!(layout_of(&ctx, alias(&ctx, "ShapeRef")), Some(shape));
        // A layout used as a value type is its own layout.
        assert_eq!(layout_of(&ctx, point), Some(point));
        // A name whose alias is not a layout, an unpublished name, and a
        // type that is not nominal have none.
        assert_eq!(layout_of(&ctx, alias(&ctx, "NumberRef")), None);
        assert_eq!(layout_of(&ctx, alias(&ctx, "MissingRef")), None);
        assert_eq!(layout_of(&ctx, alias(&ctx, "Number")), None);
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
            "%v = adt.array_splice %a[%i, %i](%n) attributes {type = !A} : !A"
        ));
        assert!(array_schema_ok(
            "%v = adt.array_splice %a[%i, %i]() attributes {type = !A} : !A"
        ));
        assert!(!array_schema_ok(
            "%v = adt.array_splice %a[%n, %i]() attributes {type = !A} : !A"
        ));
        assert!(!array_schema_ok(
            "%v = adt.array_splice %a[%i, %i](%w) attributes {type = !A} : !A"
        ));
        assert!(!array_schema_ok(
            "%v = adt.array_splice %a[%i, %i]() attributes {type = !B} : !B"
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
        ];
        let names = types.map(|ty| suggest_type_alias_name(&ctx, ty));
        assert_eq!(names, [Some("Point"), Some("Option"), Some("Point")]);
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
  !point = adt.struct<"Point", (core.i32, core.i32 {k = @v})>
  !closure = adt.struct<"Nested::Closure", (func.func_sig<(core.i32) -> core.i32, {k = 1}> {}, core.tuple<core.ptr>), {layout = "closure"}>
  !empty = adt.struct<"Empty", ()>
  !Inner = adt.struct<"Inner", (core.i32), {layout = "closure"}>
  !nested = adt.struct<"Outer", (adt.struct<"Inner", {layout = "closure"}> {m = 1})>
  !escaped = adt.struct<"say \"hi\"", ()>
  !literal = adt.struct<(core.ptr, core.i32 {k = 1}), {layout = "closure"}>
  !literal_empty = adt.struct<()>
  !literal_nested = adt.struct<(adt.struct<(core.i32)>)>
}"#;
        let mut ctx = IrContext::new();
        let module = parse_module(&mut ctx, input).expect("adt.struct syntax should parse");
        let aliases: HashMap<_, _> = ctx
            .type_aliases()
            .iter()
            .map(|(name, ty)| (name.to_string(), *ty))
            .collect();
        let i32_ty = TypeDataBuilder::new("core.i32").intern(&mut ctx);
        let point = Struct::from_type_ref(&ctx, aliases["point"]).unwrap();
        assert_eq!(point.name(&ctx), Some("Point"));
        assert_eq!(point.fields(&ctx), [i32_ty, i32_ty]);
        assert!(point.field_attrs(&ctx, 0).is_empty());
        assert_eq!(point.field_attrs(&ctx, 1).len(), 1);
        let empty = Struct::from_type_ref(&ctx, aliases["empty"]).unwrap();
        assert_eq!(empty.field_count(&ctx), 0);
        let literal = Struct::from_type_ref(&ctx, aliases["literal"]).unwrap();
        assert_eq!(literal.name(&ctx), None);
        assert_eq!(literal.field_count(&ctx), 2);
        // A literal layout is its fields and attributes.
        let ptr_ty = TypeDataBuilder::new("core.ptr").intern(&mut ctx);
        assert_eq!(
            Struct::literal(&mut ctx, [i32_ty], AttributeMap::new()).as_type_ref(),
            ctx.get_type(aliases["literal_nested"]).params[0]
        );
        assert_ne!(
            Struct::literal(&mut ctx, [ptr_ty], AttributeMap::new()).as_type_ref(),
            Struct::from_fields(&mut ctx, "P", [ptr_ty], AttributeMap::new()).as_type_ref()
        );
        let printed = module.display(&ctx).to_string();
        for expected in [
            "!point = adt.struct<\"Point\", (core.i32, core.i32 {k = @v})>",
            "adt.struct<\"Nested::Closure\", (func.func_sig<(core.i32) -> core.i32, {k = 1}>, core.tuple<core.ptr>), {layout = \"closure\"}>",
            "!empty = adt.struct<\"Empty\", ()>",
            // A body writes the nominal layouts in it as keys; each one's
            // own alias writes its body.
            ", (adt.struct<\"Inner\", {layout = \"closure\"}> {m = 1})>",
            "!Inner = adt.struct<\"Inner\", (core.i32), {layout = \"closure\"}>",
            "adt.struct<\"say \\\"hi\\\"\", ()>",
            "adt.struct<(core.ptr, core.i32 {k = 1}), {layout = \"closure\"}>",
            "adt.struct<()>",
            "adt.struct<(adt.struct<(core.i32)>)>",
        ] {
            assert!(printed.contains(expected), "{expected}\n{printed}");
        }
        assert_roundtrip(&ctx, module);
    }

    #[test]
    fn repeated_struct_is_aliased_by_its_name() {
        let input = r#"core.module @test {
  func.func @f1(%x: adt.struct<"_Marker", (core.i32)>) -> adt.struct<"_Marker", (core.i32)> {
    func.return %x
  }
  func.func @f2(%x: adt.struct<"_Marker", (core.i32)>) -> adt.struct<"_Marker", (core.i32)> {
    func.return %x
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_module(&mut ctx, input).expect("struct types should parse");
        let printed = module.display(&ctx).to_string();
        assert!(
            printed.contains("!_Marker = adt.struct<\"_Marker\", (core.i32)>"),
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
  !option = adt.enum<"Option", [(), (core.i32)]>
  !shape = adt.enum<"geo::Shape", [(), (core.f64, core.f64 {k = 1}), (core.ptr {k = 2})], {layout = "x"}>
  !never = adt.enum<"Never", []>
  !list = adt.enum<"List", [(), (core.i32, adt.typeref<{name = "List"}>)]>
  !same = adt.enum<"Same", [(core.i32), (core.i32)]>
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
        assert_eq!(option.name(&ctx), Some("Option"));
        let variants: Vec<_> = option.variants(&ctx).map(<[TypeRef]>::to_vec).collect();
        assert_eq!(variants, [vec![], vec![i32_ty]]);
        assert_eq!(option.variant_fields(&ctx, 1), Some(&[i32_ty][..]));
        assert_eq!(option.variant_fields(&ctx, 2), None);
        // The key has no parameters; a variant is one type parameter of the
        // literal body.
        assert!(ctx.get_type(aliases["option"]).params.is_empty());
        let body = option.body(&ctx).expect("the alias binds the body");
        assert_eq!(ctx.get_type(body.as_type_ref()).params.len(), 2);

        let never = Enum::from_type_ref(&ctx, aliases["never"]).unwrap();
        assert_eq!(never.variant_count(&ctx), 0);

        // Variants with the same fields are one `adt.variant` type, told apart
        // by position.
        let same = Enum::from_type_ref(&ctx, aliases["same"]).unwrap();
        let same = ctx.get_type(same.body(&ctx).unwrap().as_type_ref());
        assert_eq!(same.params[0], same.params[1]);
        assert_eq!(
            Enum::from_type_ref(&ctx, aliases["same"])
                .unwrap()
                .variant_count(&ctx),
            2
        );

        // The constructor builds the type the syntax denotes.
        let built = Enum::from_variants(
            &mut ctx,
            "Option",
            [vec![], vec![i32_ty]],
            AttributeMap::new(),
        );
        assert_eq!(built.as_type_ref(), aliases["option"]);

        let printed = module.display(&ctx).to_string();
        for expected in [
            "!option = adt.enum<\"Option\", [(), (core.i32)]>",
            "adt.enum<\"geo::Shape\", [(), (core.f64, core.f64 {k = 1}), (core.ptr {k = 2})], {layout = \"x\"}>",
            "!never = adt.enum<\"Never\", []>",
            "adt.enum<\"List\", [(), (core.i32, adt.typeref<\"List\">)]>",
        ] {
            assert!(printed.contains(expected), "{expected}\n{printed}");
        }
        assert_roundtrip(&ctx, module);
    }

    #[test]
    fn test_adt_enum_reserved_names_are_parse_errors() {
        for spelling in [
            "adt.enum<\"E\", [()], {name = @Q}>",
            "adt.enum<\"E\", [()], {name = \"Q\"}>",
            // The name is always a quoted string.
            "adt.enum<E, [()]>",
        ] {
            let mut ctx = IrContext::new();
            let input = format!("core.module @test {{ !bad = {spelling} }}");
            parse_module(&mut ctx, &input).expect_err(spelling);
        }
    }

    #[test]
    fn test_malformed_adt_enum_parses_but_fails_validation() {
        for spelling in [
            "adt.enum<{name = \"E\", variants = [[\"A\", []]]}>",
            "adt.enum<core.i32, {name = \"E\"}>",
            "adt.enum<adt.variant<{name = \"A\"}>, {name = \"E\"}>",
            "adt.enum<adt.variant<core.i32>, {name = @E}>",
            // A named enum's variants are in its body; a literal enum has no
            // attributes.
            "adt.enum<adt.variant<core.i32>, {name = \"E\"}>",
            "adt.enum<adt.variant<core.i32>, {k = 1}>",
            // The syntax cannot print a variant parameter's attributes.
            "adt.enum<adt.variant<core.i32> {k = 1}, {name = \"E\"}>",
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

    const ROW_LAYOUTS: &str = r#"core.module @test {
  !Callback = adt.struct<"Callback"['a0], (tribute_control.func_sig<(core.i64) -> core.i64 row(@Log | 'a0)>)>
  !Step = adt.enum<"Step"['a0], [(tribute_control.func_sig<() -> adt.typeref<"Step"[row(| 'a0)]> row(| 'a0)>), ()]>
  !Pair = adt.struct<"Pair"['a0, 'a1], (tribute_control.func_sig<() -> core.i64 row(| 'a0)>, tribute_control.func_sig<() -> core.i64 row(| 'a1)>)>
  !Pure = adt.typeref<"Callback"[row()]>
  !Ticked = adt.typeref<"Callback"[row(@Tick)]>
  !Open = adt.typeref<"Pair"[row(), row(@A | 'a0)]>
}"#;

    #[test]
    fn row_binders_and_row_arguments_round_trip() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, ROW_LAYOUTS);
        let alias = |ctx: &IrContext, name| ctx.type_alias_by_text(name).expect("fixture alias");
        let callback = Struct::from_type_ref(&ctx, alias(&ctx, "Callback")).unwrap();
        assert_eq!(callback.row_params(&ctx), Some(1));
        let step = Enum::from_type_ref(&ctx, alias(&ctx, "Step")).unwrap();
        assert_eq!(step.row_params(&ctx), Some(1));
        let open = Typeref::from_type_ref(&ctx, alias(&ctx, "Open")).unwrap();
        assert_eq!(
            open.rows(&ctx),
            [Row::default(), Row::new([SymbolPath::new(["A"])], Some(0))]
        );
        // Typerefs that differ only in their row arguments are distinct
        // types over one layout.
        assert_ne!(alias(&ctx, "Pure"), alias(&ctx, "Ticked"));
        assert_eq!(
            layout_of(&ctx, alias(&ctx, "Pure")),
            Some(callback.as_type_ref())
        );
        assert_eq!(
            layout_of(&ctx, alias(&ctx, "Ticked")),
            Some(callback.as_type_ref())
        );

        let printed = module.op().display(&ctx).to_string();
        for expected in [
            "adt.struct<\"Callback\"['a0], (tribute_control.func_sig",
            "adt.enum<\"Step\"['a0], [(tribute_control.func_sig",
            "adt.typeref<\"Step\"[row(| 'a0)]>",
            "adt.typeref<\"Callback\"[row()]>",
            "adt.typeref<\"Callback\"[row(@Tick)]>",
            "adt.typeref<\"Pair\"[row(), row(@A | 'a0)]>",
        ] {
            assert!(printed.contains(expected), "{expected}\n{printed}");
        }
        assert_roundtrip(&ctx, module.op());
    }

    #[test]
    fn row_instances_apply_a_layout_to_its_row_arguments() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !Callback = adt.struct<"Callback"['a0], (tribute_control.func_sig<(core.i64) -> core.i64 row(@Log | 'a0)>)>
  !Step = adt.enum<"Step"['a0], [(tribute_control.func_sig<() -> adt.enum<"Step"[row(| 'a0)]> row(| 'a0)>), ()]>
  !Node = adt.struct<"Node", (core.i32, adt.struct<"Node">)>
  !Ticked = adt.struct<"Callback"[row(@Tick)]>
  !Open = adt.enum<"Step"[row(@A | 'a0)]>
}"#,
        );
        let alias = |ctx: &IrContext, name| ctx.type_alias_by_text(name).expect("fixture alias");
        let [callback, step, node, ticked, open] =
            ["Callback", "Step", "Node", "Ticked", "Open"].map(|name| alias(&ctx, name));

        // An instance is a type of the layout's kind with no identifier,
        // whose layout is its `def`.
        assert!(Struct::from_type_ref(&ctx, ticked).is_some());
        assert!(!trunk_ir::type_def::is_identified(&ctx, ticked));
        assert_eq!(layout_of(&ctx, ticked), Some(callback));
        assert_eq!(layout_of(&ctx, open), Some(step));
        assert_eq!(nominal_name(&ctx, open), Some("Step"));
        assert_eq!(
            value_rows(&ctx, ticked),
            [Row::new([SymbolPath::new(["Tick"])], None)]
        );
        assert_eq!(
            Struct::from_type_ref(&ctx, ticked).unwrap().fields(&ctx),
            Struct::from_type_ref(&ctx, callback).unwrap().fields(&ctx)
        );
        // A field reads through the instance with its rows substituted.
        let field = field_type(&mut ctx, ticked, 0).unwrap();
        let row = crate::dialect::tribute_control::FuncSig::from_type_ref(&ctx, field)
            .unwrap()
            .row(&ctx);
        assert_eq!(
            row,
            Row::new([SymbolPath::new(["Log"]), SymbolPath::new(["Tick"])], None)
        );
        // A layout holds its own key.
        let node_body = Struct::from_type_ref(&ctx, node).unwrap().fields(&ctx)[1];
        assert_eq!(node_body, node);
        // The constructor makes the instance the text spells.
        let rows = value_rows(&ctx, ticked);
        assert_eq!(value_type(&mut ctx, callback, rows), ticked);
        assert_eq!(value_type(&mut ctx, node, []), node);

        let printed = module.op().display(&ctx).to_string();
        for expected in [
            "adt.struct<\"Callback\"[row(@Tick)]>",
            "adt.enum<\"Step\"[row(@A | 'a0)]>",
            "adt.enum<\"Step\"[row(| 'a0)]>",
            "!Node = adt.struct<\"Node\", (core.i32, adt.struct<\"Node\">)>",
        ] {
            assert!(printed.contains(expected), "{expected}\n{printed}");
        }
        assert_roundtrip(&ctx, module.op());
    }

    #[test]
    fn malformed_row_instances_fail_validation() {
        for spelling in [
            // Too many or too few row arguments.
            "adt.struct<{def = adt.struct<\"C\"['a0], ()>, row_args = [{}, {}]}>",
            "adt.struct<{def = adt.struct<\"C\"['a0], ()>, row_args = []}>",
            // A `def` without binders, of another kind, or with a field.
            "adt.struct<{def = adt.struct<\"P\", ()>, row_args = [{}]}>",
            "adt.enum<{def = adt.struct<\"C\"['a0], ()>, row_args = [{}]}>",
            "adt.struct<core.i32, {def = adt.struct<\"C\"['a0], ()>, row_args = [{}]}>",
        ] {
            let mut ctx = IrContext::new();
            let input = format!("core.module @test {{ !bad = {spelling} }}");
            let module = parse_module(&mut ctx, &input).expect(spelling);
            let alias = ctx.type_aliases()[0].1;
            assert!(
                Struct::from_type_ref(&ctx, alias).is_none()
                    && Enum::from_type_ref(&ctx, alias).is_none(),
                "{spelling}"
            );
            let result = validate_operation_verifiers(&ctx, Module::new(&ctx, module).unwrap());
            assert!(!result.is_ok(), "{spelling}");
        }
    }

    #[test]
    fn a_typeref_with_row_arguments_suggests_no_alias() {
        let mut ctx = IrContext::new();
        let plain = Typeref::named(&mut ctx, "Point", []);
        let rowed = Typeref::named(&mut ctx, "Callback", [Row::default()]);
        assert_eq!(plain.alias_hint(&ctx), Some("Point"));
        assert_eq!(rowed.alias_hint(&ctx), None);
    }

    #[test]
    fn field_types_take_the_value_types_row_arguments() {
        let mut ctx = IrContext::new();
        parse_test_module(&mut ctx, ROW_LAYOUTS);
        let alias = |ctx: &IrContext, name| ctx.type_alias_by_text(name).expect("fixture alias");
        let row = |ctx: &IrContext, ty| {
            crate::dialect::tribute_control::FuncSig::from_type_ref(ctx, ty)
                .expect("field is a callable")
                .row(ctx)
        };
        let path = |name| SymbolPath::new([name]);

        let [pure, ticked, open] = ["Pure", "Ticked", "Open"].map(|name| alias(&ctx, name));

        let field = field_type(&mut ctx, pure, 0).unwrap();
        assert_eq!(row(&ctx, field), Row::new([path("Log")], None));
        let field = field_type(&mut ctx, ticked, 0).unwrap();
        assert_eq!(
            row(&ctx, field),
            Row::new([path("Log"), path("Tick")], None)
        );
        let field = field_type(&mut ctx, open, 1).unwrap();
        assert_eq!(row(&ctx, field), Row::new([path("A")], Some(0)));
        assert_eq!(field_type(&mut ctx, open, 2), None);
    }

    #[test]
    fn malformed_row_binders_and_row_arguments_fail_validation() {
        // A tail inside a layout's body names one of its binders, which the
        // body is checked against when it is bound.
        let mut ctx = IrContext::new();
        let error = parse_module(
            &mut ctx,
            "core.module @test { !bad = adt.struct<\"C\"['a0], (tribute_control.func_sig<() -> core.i64 row(| 'a1)>)> }",
        )
        .expect_err("an unbound tail");
        assert!(error.message.contains("'a1"), "{}", error.message);

        for spelling in [
            "adt.struct<{name = \"C\", row_params = 0}>",
            "adt.enum<{name = \"E\", row_params = 0}>",
            "adt.typeref<{name = \"C\", row_args = []}>",
            "adt.typeref<{name = \"C\", row_args = [1]}>",
        ] {
            let mut ctx = IrContext::new();
            let input = format!("core.module @test {{ !bad = {spelling} }}");
            let module = parse_module(&mut ctx, &input).expect(spelling);
            let alias = ctx.type_aliases()[0].1;
            assert!(
                Struct::from_type_ref(&ctx, alias).is_none()
                    && Enum::from_type_ref(&ctx, alias).is_none()
                    && Typeref::from_type_ref(&ctx, alias).is_none(),
                "{spelling}"
            );
            let result = validate_operation_verifiers(&ctx, Module::new(&ctx, module).unwrap());
            assert!(!result.is_ok(), "{spelling}");
        }
    }

    #[test]
    fn row_binders_and_row_arguments_in_a_dictionary_are_parse_errors() {
        for spelling in [
            "adt.struct<\"C\"['a0], (), {row_params = 1}>",
            "adt.enum<\"E\"['a0], [()], {row_params = 1}>",
            "adt.typeref<\"C\"[row()], {row_args = [{}]}>",
            "adt.struct<\"C\"['a1], ()>",
            "adt.typeref<\"C\"[]>",
            // The name is always a quoted string.
            "adt.typeref<C>",
            "adt.typeref<C[row()]>",
        ] {
            let mut ctx = IrContext::new();
            let input = format!("core.module @test {{ !bad = {spelling} }}");
            parse_module(&mut ctx, &input).expect_err(spelling);
        }
    }

    #[test]
    fn test_adt_struct_reserved_names_are_parse_errors() {
        for spelling in [
            "adt.struct<\"P\", (core.i32), {name = @Q}>",
            "adt.struct<\"P\", (core.i32), {param_attrs = [{}]}>",
            // The name is always a quoted string.
            "adt.struct<P, (core.i32)>",
        ] {
            let mut ctx = IrContext::new();
            let input = format!("core.module @test {{ !bad = {spelling} }}");
            parse_module(&mut ctx, &input).expect_err(spelling);
        }
    }

    #[test]
    fn test_malformed_adt_struct_parses_but_fails_validation() {
        for spelling in [
            "adt.struct<\"P\", {fields = []}>",
            "adt.struct<core.i32, {name = @P}>",
            // A nominal layout's fields are in its body.
            "adt.struct<core.i32, {name = \"P\"}>",
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
        let p = ctx.string_attr("P");
        let cases = [
            (
                "`name` is not a string",
                TypeDataBuilder::new("adt.struct")
                    .param(i32_ty)
                    .attr("name", Attribute::Int(1))
                    .build(&mut ctx),
            ),
            (
                "a literal `adt.struct` has no `row_params`",
                TypeDataBuilder::new("adt.struct")
                    .param(i32_ty)
                    .attr("row_params", Attribute::Int(1))
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
        let valid = Struct::from_fields(&mut ctx, "Q", [i32_ty], AttributeMap::new());
        assert!(
            !ctx.type_attrs(valid.as_type_ref())
                .contains_key(PARAM_ATTRS_ATTR)
        );
    }

    #[test]
    fn variant_operations_take_a_position_of_their_enum() {
        let valid = r#"core.module @test {
  !Shape = adt.enum<"Shape", [(), (core.f64, core.f64)]>
  func.func @f(%x: core.f64, %shape: !Shape) -> core.f64 {
    %dot = adt.variant_new {tag = 0, type = !Shape} : !Shape
    %rect = adt.variant_new %x, %x {tag = 1, type = !Shape} : !Shape
    %is_rect = adt.variant_is %shape {tag = 1, type = !Shape} : core.i1
    %width = adt.variant_get %shape {field = 0, tag = 1, type = !Shape} : core.f64
    func.return %width
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, valid);
        assert_roundtrip(&ctx, module.op());
        assert!(validate_operation_verifiers(&ctx, module).is_ok());

        for (invalid, expected) in [
            (
                "adt.variant_new {tag = 2, type = !Shape} : !Shape",
                "tag 2 is not a variant of an enum with 2 variant(s)",
            ),
            (
                "adt.variant_is %shape {tag = 2, type = !Shape} : core.i1",
                "tag 2 is not a variant",
            ),
            (
                "adt.variant_get %shape {field = 0, tag = 5, type = !Shape} : core.f64",
                "tag 5 is not a variant",
            ),
            (
                "adt.variant_is %shape {tag = 0, type = core.f64} : core.i1",
                "`type` is not an `adt.enum` layout",
            ),
        ] {
            let mut ctx = IrContext::new();
            let text = valid.replace(
                "%is_rect = adt.variant_is %shape {tag = 1, type = !Shape} : core.i1",
                &format!("%bad = {invalid}"),
            );
            let module = parse_test_module(&mut ctx, &text);
            let result = validate_operation_verifiers(&ctx, module).to_string();
            assert!(result.contains(expected), "{invalid}: {result}");
        }
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
    use trunk_ir::rewrite::Module;
    use trunk_ir::transforms::Canonicalize;

    fn count(ctx: &IrContext, module: Module, name: &str) -> usize {
        module.op().display(ctx).to_string().matches(name).count()
    }

    /// Forward struct reads and canonicalize; the flag tells whether the IR
    /// changed.
    fn fold(input: &str) -> (IrContext, Module, bool) {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        let before = module.op().display(&ctx).to_string();
        let root = core::Module::from_op(&ctx, module.op()).expect("core.module");
        let mut pm = PassManager::new();
        pm.add_pass(ForwardStructGets)
            .nest::<func::Func>()
            .add_pass(Canonicalize);
        pm.run(&mut ctx, root, &mut Default::default())
            .expect("fold passes");
        let changed = module.op().display(&ctx).to_string() != before;
        (ctx, module, changed)
    }

    const PAIR: &str = r#"core.module @test {
  !pair = adt.struct<"Pair", (core.i32, core.i64)>
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
        assert!(
            module
                .op()
                .display(&ctx)
                .to_string()
                .contains("func.return %1")
        );
    }

    #[test]
    fn struct_get_of_a_block_argument_is_not_folded() {
        let (ctx, module, changed) = fold(
            r#"core.module @test {
  !pair = adt.struct<"Pair", (core.i32, core.i64)>
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
  !pair = adt.struct<"Pair", (core.i32, core.i64)>
  !other = adt.struct<"Other", (core.i32, core.i64)>
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
  !pair = adt.struct<"Pair", (core.i32, core.i64)>
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
  !pair = adt.struct<"Pair", (core.i32, core.i64)>
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
  !pair = adt.struct<"Pair", (core.i32, core.i64)>
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
  !pair = adt.struct<"Pair", (core.i32, core.i64)>
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
  !pair = adt.struct<"Pair", (core.i32, core.i64)>
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
  !pair = adt.struct<"Pair", (core.i32, core.i64)>
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
  !pair = adt.struct<"Pair", (core.i32, core.i64)>
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
  !pair = adt.struct<"Pair", (core.i32, core.i64)>
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
