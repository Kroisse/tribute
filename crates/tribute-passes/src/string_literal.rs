//! The `adt` operations a byte or string constant becomes on every target.
//!
//! An `adt.bytes_const` is the representation of `Bytes` viewing all of an
//! `adt.array_const` of its bytes, and an `adt.string_const` is the `Leaf` of
//! the prelude `String` holding such a value (`design/string.md` § "Constant
//! lowering"). Each target's constant lowering expands both with the
//! patterns here and then lowers the `adt.array_const` among the results as
//! it lowers any other array constant.

use tribute_ir::dialect::adt;
use tribute_ir::metadata::{BytesRepr, StringLeaf};
use trunk_ir::context::IrContext;
use trunk_ir::dialect::core;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{OpRef, TypeRef, ValueRef};
use trunk_ir::rewrite::{Module, OpRewritePattern, PatternRewriter};
use trunk_ir::types::Location;

/// The layouts the constants of a module build, from its nominal
/// declarations.
#[derive(Clone, Debug, Default)]
pub(crate) struct LiteralLayouts {
    /// The representation of each `Bytes` value type a constant builds.
    bytes: Vec<BytesRepr>,
    /// The `Leaf` of the prelude `String`, when the module has string
    /// constants and declares it.
    string: Option<StringLeaf>,
}

impl LiteralLayouts {
    /// The layouts of the constants in `module`: the representation of each
    /// `adt.bytes_const` result type and, when `strings` is set, the
    /// `String` leaf and the representation of its `Bytes`.
    pub(crate) fn collect(ctx: &IrContext, module: Module, strings: bool) -> Self {
        let mut bytes_types = Vec::new();
        let _ = trunk_ir::walk::walk_op::<()>(ctx, module.op(), &mut |op| {
            if adt::BytesConst::matches(ctx, op) {
                let ty = ctx.op_result_types(op)[0];
                if !bytes_types.contains(&ty) {
                    bytes_types.push(ty);
                }
            }
            std::ops::ControlFlow::Continue(trunk_ir::walk::WalkAction::Advance)
        });
        if bytes_types.is_empty() && !strings {
            return Self::default();
        }
        let decls = adt::NominalDecls::collect(ctx, module);
        let string = strings
            .then(|| tribute_ir::metadata::string_leaf(ctx, module, &decls))
            .flatten();
        if let Some(leaf) = string
            && !bytes_types.contains(&leaf.bytes)
        {
            bytes_types.push(leaf.bytes);
        }
        let bytes = bytes_types
            .into_iter()
            .filter_map(|ty| BytesRepr::of(ctx, &decls, ty))
            .collect();
        Self { bytes, string }
    }

    /// The representation of the `Bytes` value type `ty`.
    pub(crate) fn bytes_repr(&self, ty: TypeRef) -> Option<&BytesRepr> {
        self.bytes.iter().find(|repr| repr.ty == ty)
    }

    /// The `Leaf` of the prelude `String`, when string constants need it.
    pub(crate) fn string_leaf(&self) -> Option<StringLeaf> {
        self.string
    }
}

/// Builds the target's constant of a `Nat` field type at a location: the
/// operation and its result.
pub(crate) type NatConstant<'a> =
    &'a dyn Fn(&mut IrContext, TypeRef, i64, Location) -> (OpRef, ValueRef);

/// The operations that build a `repr` value viewing all of `bytes`, in
/// order. The last one produces the value.
fn build_bytes(
    ctx: &mut IrContext,
    repr: &BytesRepr,
    bytes: &[u8],
    loc: Location,
    nat: NatConstant<'_>,
) -> Option<Vec<OpRef>> {
    let nat_ty = adt::field_type(ctx, repr.ty, repr.start as usize)?;
    let array_ty = tribute_ir::runtime_layout::byte_array(ctx);
    let data = adt::ArrayConst::operands()
        .r#type(array_ty)
        .value(bytes.into())
        .build(ctx, loc);
    let (start_op, start) = nat(ctx, nat_ty, 0, loc);
    let (len_op, len) = nat(ctx, nat_ty, bytes.len() as i64, loc);
    let value = adt::StructNew::operands(repr.fields(data.result(ctx), start, len))
        .results(repr.ty)
        .build(ctx, loc);
    Some(vec![data.op_ref(), start_op, len_op, value.op_ref()])
}

/// Insert all of `ops` but the last, which replaces the matched operation.
fn replace_with(rewriter: &mut PatternRewriter<'_>, mut ops: Vec<OpRef>) {
    let last = ops.pop().expect("a constant builds its value");
    for op in ops {
        rewriter.insert_op(op);
    }
    rewriter.replace_op(last);
}

/// Expands `adt.bytes_const` to its representation over an
/// `adt.array_const`.
pub(crate) struct ExpandBytesConst<'a> {
    pub(crate) layouts: &'a LiteralLayouts,
    pub(crate) nat: NatConstant<'a>,
}

impl OpRewritePattern for ExpandBytesConst<'_> {
    type Op = adt::BytesConst;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        bytes_const: Self::Op,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let op = bytes_const.op_ref();
        let result_ty = ctx.op_result_types(op)[0];
        let repr = *rewriter.require(
            self.layouts.bytes_repr(result_ty),
            "the result type is not a declared `Bytes` representation",
        )?;
        let bytes = bytes_const.value(ctx).to_vec();
        let loc = ctx.op(op).location;
        let ops = build_bytes(ctx, &repr, &bytes, loc, self.nat);
        let ops = rewriter.require(ops, "the `start` field has no type")?;
        replace_with(rewriter, ops);
        Some(())
    }

    fn name(&self) -> &'static str {
        "ExpandBytesConst"
    }
}

/// Expands `adt.string_const` to `String::Leaf` of a `Bytes` representation
/// over an `adt.array_const`. The value has the `String` layout's type,
/// cast back to the constant's result type when `cast_to_result` is set.
pub(crate) struct ExpandStringConst<'a> {
    pub(crate) layouts: &'a LiteralLayouts,
    pub(crate) nat: NatConstant<'a>,
    pub(crate) cast_to_result: bool,
}

impl OpRewritePattern for ExpandStringConst<'_> {
    type Op = adt::StringConst;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        string_const: Self::Op,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let op = string_const.op_ref();
        let leaf = rewriter.require(self.layouts.string, "the prelude `String` is not declared")?;
        let repr = *rewriter.require(
            self.layouts.bytes_repr(leaf.bytes),
            "the `String` leaf holds no declared `Bytes` representation",
        )?;
        let text = string_const.value(ctx).as_bytes().to_vec();
        let result_ty = ctx.op_result_types(op)[0];
        let loc = ctx.op(op).location;
        let mut ops = rewriter.require(
            build_bytes(ctx, &repr, &text, loc, self.nat),
            "the `start` field has no type",
        )?;
        let bytes = ctx.op_result(*ops.last().expect("bytes value"), 0);
        let string = adt::VariantNew::operands([bytes])
            .tag(u64::from(leaf.leaf))
            .results(leaf.string)
            .build(ctx, loc);
        ops.push(string.op_ref());
        if self.cast_to_result && result_ty != leaf.string {
            let cast = core::UnrealizedConversionCast::operands(string.result(ctx))
                .results(result_ty)
                .build(ctx, loc);
            ops.push(cast.op_ref());
        }
        replace_with(rewriter, ops);
        Some(())
    }

    fn name(&self) -> &'static str {
        "ExpandStringConst"
    }
}
