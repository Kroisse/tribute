//! Arena-based core dialect.

use crate::IrContext;

// === Operation registrations ===
crate::register_isolated_op!(Module);

#[trunk_ir::dialect]
mod core {
    fn module(sym_name: Attr<Symbol>) {
        #[region(body)]
        {}
    }

    fn unrealized_conversion_cast(value: Value<_>) -> Value<_> {}

    struct Nil;
    struct Never;
    struct Bytes;
    struct Ptr;
    struct Array<Element>;
    #[attr(nullable: bool)]
    struct Ref<Pointee>;
    struct Tuple<#[rest] Elements>;
}

// =========================================================================
// Scalar type categories
//
// Closed predicates over `core` scalar types, shared by operation
// verification and target lowering instead of comparing type names.
// =========================================================================

use crate::Symbol;
use crate::refs::TypeRef;

/// Parse a bare `core.<prefix>{N}` scalar type name.
fn core_scalar_width(ctx: &IrContext, ty: TypeRef, prefix: char) -> Option<u32> {
    let data = ctx.get_type(ty);
    if data.dialect != Symbol::new("core") || !data.params.is_empty() || !data.attrs.is_empty() {
        return None;
    }
    data.name.with_str(|s| {
        let digits = s.strip_prefix(prefix)?;
        // Accept only canonical decimal widths: `u32::from_str` alone would
        // also accept `i+32` and `i032`.
        let canonical = !digits.is_empty()
            && digits.bytes().all(|b| b.is_ascii_digit())
            && !digits.starts_with('0');
        if !canonical {
            return None;
        }
        digits.parse().ok()
    })
}

/// Fixed-width signless integer types `core.i{N}` with `1 <= N <= 128`.
///
/// The upper bound is what an `i128` constant can represent. Signedness is
/// a property of each operation, not of the type.
pub struct IntegerLike;

impl crate::type_constraint::TypeConstraint for IntegerLike {
    const DESC: &'static crate::type_constraint::ConstraintDesc =
        &crate::type_constraint::ConstraintDesc {
            name: "IntegerLike",
            exact: false,
            projections: &[],
            matches: Self::matches,
            project: |_, _, _| None,
            fixed: None,
        };
}

impl IntegerLike {
    /// The bit width of `ty`, or `None` if it is not an integer type.
    pub fn width(ctx: &IrContext, ty: TypeRef) -> Option<u32> {
        core_scalar_width(ctx, ty, 'i').filter(|width| (1..=128).contains(width))
    }

    pub fn matches(ctx: &IrContext, ty: TypeRef) -> bool {
        Self::width(ctx, ty).is_some()
    }
}

/// The boolean integer type `core.i1`.
pub struct BoolLike;

impl crate::type_constraint::TypeConstraint for BoolLike {
    const DESC: &'static crate::type_constraint::ConstraintDesc =
        &crate::type_constraint::ConstraintDesc {
            name: "BoolLike",
            exact: false,
            projections: &[],
            matches: Self::matches,
            project: |_, _, _| None,
            fixed: None,
        };
}

impl BoolLike {
    pub fn matches(ctx: &IrContext, ty: TypeRef) -> bool {
        IntegerLike::width(ctx, ty) == Some(1)
    }
}

/// IEEE floating-point types `core.f32` and `core.f64`.
pub struct FloatLike;

impl crate::type_constraint::TypeConstraint for FloatLike {
    const DESC: &'static crate::type_constraint::ConstraintDesc =
        &crate::type_constraint::ConstraintDesc {
            name: "FloatLike",
            exact: false,
            projections: &[],
            matches: Self::matches,
            project: |_, _, _| None,
            fixed: None,
        };
}

impl FloatLike {
    /// The bit width of `ty`, or `None` if it is not a float type.
    pub fn width(ctx: &IrContext, ty: TypeRef) -> Option<u32> {
        core_scalar_width(ctx, ty, 'f').filter(|width| matches!(width, 32 | 64))
    }

    pub fn matches(ctx: &IrContext, ty: TypeRef) -> bool {
        Self::width(ctx, ty).is_some()
    }
}

/// Exact bounds for individual `core` scalar types, e.g. `Value<core::I32>`.
macro_rules! scalar_types {
    ($($(#[$meta:meta])* $wrapper:ident = $name:literal via $category:ident($width:literal);)*) => {$(
        $(#[$meta])*
        pub struct $wrapper;

        impl $wrapper {
            pub fn matches(ctx: &IrContext, ty: TypeRef) -> bool {
                $category::width(ctx, ty) == Some($width)
            }

            /// Intern this scalar type.
            pub fn type_ref(ctx: &mut IrContext) -> TypeRef {
                ctx.intern_type(crate::TypeDataBuilder::new("core", $name).build())
            }
        }

        impl crate::type_constraint::TypeConstraint for $wrapper {
            const DESC: &'static crate::type_constraint::ConstraintDesc =
                &crate::type_constraint::ConstraintDesc {
                    name: concat!("core.", $name),
                    exact: true,
                    projections: &[],
                    matches: Self::matches,
                    project: |_, _, _| None,
                    fixed: Some(Self::type_ref),
                };
        }
    )*};
}

scalar_types! {
    /// `core.i1`.
    I1 = "i1" via IntegerLike(1);
    /// `core.i8`.
    I8 = "i8" via IntegerLike(8);
    /// `core.i16`.
    I16 = "i16" via IntegerLike(16);
    /// `core.i32`.
    I32 = "i32" via IntegerLike(32);
    /// `core.i64`.
    I64 = "i64" via IntegerLike(64);
    /// `core.f32`.
    F32 = "f32" via FloatLike(32);
    /// `core.f64`.
    F64 = "f64" via FloatLike(64);
}

// =========================================================================
// Canonicalization folds
//
// Owned by this dialect and aggregated by `transforms::canonicalize` via
// [`folds`]. Folds are looked up by (dialect, op_name) so they don't
// self-filter — they assume the dispatcher already decided this op is
// `core.unrealized_conversion_cast`.
// =========================================================================

use crate::ops::DialectOp;
use crate::refs::{OpRef, ValueDef};
use crate::transforms::canonicalize::FoldResult;

/// `core.unrealized_conversion_cast` folds:
///
/// - **Identity** (`%x : T → T`): drop the cast and forward `%x`.
/// - **Round-trip** (`cast<A → B>(cast<B → A>(%x))`): forward the inner
///   cast's input; the now-dead inner cast falls to DCE.
///
/// Safe *specifically* because both ops are
/// `core.unrealized_conversion_cast` — dialect-conversion placeholders
/// that carry no value-level conversion semantics. A resolved cast pair
/// like `arith.trunc` followed by `arith.extend` is *not* safe to collapse
/// the same way (narrower intermediate types lose information). Once
/// `resolve_unrealized_casts` has run, no `unrealized_conversion_cast`
/// ops remain and this fold is a no-op.
#[trunk_ir::canonicalize_fold(UnrealizedConversionCast)]
pub(crate) fn fold_unrealized_conversion_cast(ctx: &IrContext, op: OpRef) -> Option<FoldResult> {
    let operands = ctx.op_operands(op);
    let result_types = ctx.op_result_types(op);
    if operands.len() != 1 || result_types.len() != 1 {
        return None;
    }
    let input = operands[0];
    let result_ty = result_types[0];

    // Identity: T → T
    if ctx.value_ty(input) == result_ty {
        return Some(FoldResult::Forward(input));
    }

    // Round-trip: A → B → A
    if let ValueDef::OpResult(producer, _) = ctx.value_def(input)
        && UnrealizedConversionCast::matches(ctx, producer)
        && let Some(&inner_input) = ctx.op_operands(producer).first()
        && ctx.value_ty(inner_input) == result_ty
    {
        return Some(FoldResult::Forward(inner_input));
    }

    None
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod canonicalize_tests {
    use super::*;
    use crate::dialect::func::{FuncSig, NUM_INPUTS_ATTR, NUM_RESULTS_ATTR};
    use crate::parser::parse_test_module;
    use crate::printer::print_module;
    use crate::rewrite::{ApplyResult, Module, PatternApplicator, TypeConverter};
    use crate::walk::{WalkAction, walk_op};
    use crate::{Attribute, AttributeMap, Symbol};
    use std::ops::ControlFlow;

    use crate::transforms::canonicalize::{FoldDispatchPattern, folds_for_dialect};

    fn run_core_patterns(ctx: &mut IrContext, module: Module) -> ApplyResult {
        let dispatcher = FoldDispatchPattern::from_folds(folds_for_dialect("core"));
        PatternApplicator::new(TypeConverter::new())
            .add_pattern_box(Box::new(dispatcher))
            .apply_partial(ctx, module)
    }

    fn count_ops(ctx: &IrContext, module: Module, dialect: &str, name: &str) -> usize {
        let dialect_sym = Symbol::from_dynamic(dialect);
        let name_sym = Symbol::from_dynamic(name);
        let mut count = 0usize;
        let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
            let data = ctx.op(op);
            if data.dialect == dialect_sym && data.name == name_sym {
                count += 1;
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
        count
    }

    #[test]
    fn remove_reserved_attrs_preserves_metadata_and_is_idempotent() {
        let mut attrs = AttributeMap::new();
        FuncSig::remove_reserved_attrs(&mut attrs);
        assert!(attrs.is_empty());

        let mut ctx = IrContext::new();
        let nil = nil(&mut ctx).as_type_ref();
        let metadata = AttributeMap::from_iter([
            (Symbol::new("tag"), Attribute::Symbol(Symbol::new("kept"))),
            (
                Symbol::new("nested"),
                Attribute::List(vec![Attribute::Type(nil)]),
            ),
        ]);
        attrs = metadata.clone();
        attrs.insert(Symbol::new(NUM_INPUTS_ATTR), Attribute::from(2u32));
        attrs.insert(Symbol::new(NUM_RESULTS_ATTR), Attribute::from(1u32));
        FuncSig::remove_reserved_attrs(&mut attrs);
        assert_eq!(attrs, metadata);
        FuncSig::remove_reserved_attrs(&mut attrs);
        assert_eq!(attrs, metadata);
    }

    #[test]
    fn unrealized_cast_identity_drops_same_type_cast() {
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %r = core.unrealized_conversion_cast %x : core.i32
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_core_patterns(&mut ctx, module);
        assert!(result.total_changes >= 1);
        assert_eq!(
            count_ops(&ctx, module, "core", "unrealized_conversion_cast"),
            0
        );
        insta::assert_snapshot!(print_module(&ctx, module.op()));
    }

    #[test]
    fn unrealized_cast_identity_does_not_match_when_types_differ() {
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i64 {
    %r = core.unrealized_conversion_cast %x : core.i64
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_core_patterns(&mut ctx, module);
        assert_eq!(result.total_changes, 0);
        assert_eq!(
            count_ops(&ctx, module, "core", "unrealized_conversion_cast"),
            1
        );
    }

    #[test]
    fn unrealized_cast_round_trip_collapses_pair() {
        let input = r#"core.module @test {
  func.func @f(%x: core.i64) -> core.i64 {
    %a = core.unrealized_conversion_cast %x : core.i32
    %b = core.unrealized_conversion_cast %a : core.i64
    func.return %b
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_core_patterns(&mut ctx, module);
        assert!(result.total_changes >= 1);
        // Inner cast remains (now dead); outer is gone.
        assert_eq!(
            count_ops(&ctx, module, "core", "unrealized_conversion_cast"),
            1
        );
        insta::assert_snapshot!(print_module(&ctx, module.op()));
    }

    #[test]
    fn unrealized_cast_round_trip_does_not_match_three_step_chain() {
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i64 {
    %a = core.unrealized_conversion_cast %x : core.i16
    %b = core.unrealized_conversion_cast %a : core.i64
    func.return %b
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_core_patterns(&mut ctx, module);
        assert_eq!(result.total_changes, 0);
        assert_eq!(
            count_ops(&ctx, module, "core", "unrealized_conversion_cast"),
            2
        );
    }
}

#[cfg(test)]
mod scalar_category_tests {
    use super::*;
    use crate::types::{Attribute, TypeDataBuilder};

    fn ty(ctx: &mut IrContext, dialect: &'static str, name: &'static str) -> TypeRef {
        ctx.intern_type(TypeDataBuilder::new(dialect, name).build())
    }

    #[test]
    fn integer_like_accepts_bare_core_integer_widths() {
        let mut ctx = IrContext::new();
        for (name, width) in [("i1", Some(1)), ("i32", Some(32)), ("i128", Some(128))] {
            let t = ty(&mut ctx, "core", name);
            assert_eq!(IntegerLike::width(&ctx, t), width, "{name}");
        }
        for (dialect, name) in [
            ("core", "i0"),
            ("core", "i129"),
            ("core", "i"),
            ("core", "i-1"),
            ("core", "i+32"),
            ("core", "i032"),
            ("core", "int"),
            ("core", "f32"),
            ("tribute_rt", "i32"),
        ] {
            let t = ty(&mut ctx, dialect, name);
            assert!(!IntegerLike::matches(&ctx, t), "{dialect}.{name}");
        }
        let with_attr = ctx.intern_type(
            TypeDataBuilder::new("core", "i32")
                .attr("x", Attribute::Int(0))
                .build(),
        );
        assert!(!IntegerLike::matches(&ctx, with_attr));
    }

    #[test]
    fn bool_and_float_categories() {
        let mut ctx = IrContext::new();
        let i1 = ty(&mut ctx, "core", "i1");
        let i8 = ty(&mut ctx, "core", "i8");
        let f32 = ty(&mut ctx, "core", "f32");
        let f64 = ty(&mut ctx, "core", "f64");
        let f16 = ty(&mut ctx, "core", "f16");
        assert!(BoolLike::matches(&ctx, i1));
        assert!(!BoolLike::matches(&ctx, i8));
        assert_eq!(FloatLike::width(&ctx, f32), Some(32));
        assert_eq!(FloatLike::width(&ctx, f64), Some(64));
        assert!(!FloatLike::matches(&ctx, f16));
        let f064 = ty(&mut ctx, "core", "f064");
        assert!(!FloatLike::matches(&ctx, f064));
        assert!(!FloatLike::matches(&ctx, i1));
    }
}
