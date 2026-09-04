//! Arena-based core dialect.

// === Operation registrations ===
crate::register_isolated_op!(core.module);

#[trunk_ir::dialect]
mod core {
    #[attr(sym_name: Symbol)]
    fn module() {
        #[region(body)]
        {}
    }

    fn unrealized_conversion_cast(value: ()) -> result {}

    struct Nil;
    struct Never;
    struct Bytes;
    struct Ptr;
    struct Array<Element>;
    #[attr(nullable: bool)]
    struct Ref<Pointee>;
    struct Tuple<#[rest] Elements>;
}

use crate::ops::DialectType;
use crate::{Attribute, AttributeMap, IrContext, Symbol, TypeDataBuilder, TypeRef};

/// Reserved delimiter attribute for the number of function inputs.
pub const NUM_INPUTS_ATTR: &str = "num_inputs";

/// Reserved delimiter attribute for the number of function results.
pub const NUM_RESULTS_ATTR: &str = "num_results";

/// Return the interned name of the `core.func` type.
#[allow(non_snake_case)]
#[inline]
pub fn FUNC() -> Symbol {
    Symbol::new("func")
}

/// Why a name-matching `core.func` does not satisfy its storage invariant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FuncTypeError {
    MissingCount(&'static str),
    InvalidCount(&'static str),
    CountOverflow,
    LengthMismatch {
        num_inputs: u32,
        num_results: u32,
        params: usize,
    },
    UnsupportedResultCount(u32),
}

impl std::fmt::Display for FuncTypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingCount(name) => write!(f, "missing required `{name}` u32 attribute"),
            Self::InvalidCount(name) => write!(f, "`{name}` must be a u32 attribute"),
            Self::CountOverflow => write!(f, "input and result counts overflow u32"),
            Self::LengthMismatch {
                num_inputs,
                num_results,
                params,
            } => write!(
                f,
                "num_inputs ({num_inputs}) + num_results ({num_results}) must equal params length ({params})"
            ),
            Self::UnsupportedResultCount(count) => {
                write!(f, "currently supports at most one result, found {count}")
            }
        }
    }
}

impl std::error::Error for FuncTypeError {}

/// Validated wrapper for an input-first, zero-or-one-result `core.func` type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Func(TypeRef);

impl Func {
    /// Validate a name-matching `core.func`, including both delimiter counts.
    pub(crate) fn validate(ctx: &IrContext, ty: TypeRef) -> Result<Self, FuncTypeError> {
        let data = ctx.types.get(ty);
        debug_assert!(data.dialect == DIALECT_NAME() && data.name == FUNC());

        let num_inputs = read_count(&data.attrs, NUM_INPUTS_ATTR)?;
        let num_results = read_count(&data.attrs, NUM_RESULTS_ATTR)?;
        if num_results > 1 {
            return Err(FuncTypeError::UnsupportedResultCount(num_results));
        }
        let total = num_inputs
            .checked_add(num_results)
            .ok_or(FuncTypeError::CountOverflow)?;
        if usize::try_from(total).ok() != Some(data.params.len()) {
            return Err(FuncTypeError::LengthMismatch {
                num_inputs,
                num_results,
                params: data.params.len(),
            });
        }
        Ok(Self(ty))
    }

    fn counts(self, ctx: &IrContext) -> (usize, usize) {
        let data = ctx.types.get(self.0);
        let num_inputs = read_count(&data.attrs, NUM_INPUTS_ATTR)
            .expect("validated core.func must retain a valid num_inputs attribute");
        let num_results = read_count(&data.attrs, NUM_RESULTS_ATTR)
            .expect("validated core.func must retain a valid num_results attribute");
        (
            usize::try_from(num_inputs).expect("u32 must fit usize"),
            usize::try_from(num_results).expect("u32 must fit usize"),
        )
    }

    pub fn as_type_ref(&self) -> TypeRef {
        self.0
    }

    pub fn inputs<'a>(&self, ctx: &'a IrContext) -> &'a [TypeRef] {
        let (num_inputs, _) = self.counts(ctx);
        &ctx.types.get(self.0).params[..num_inputs]
    }

    pub fn results<'a>(&self, ctx: &'a IrContext) -> &'a [TypeRef] {
        let (num_inputs, num_results) = self.counts(ctx);
        &ctx.types.get(self.0).params[num_inputs..num_inputs + num_results]
    }

    pub fn single_result(&self, ctx: &IrContext) -> Option<TypeRef> {
        self.results(ctx).first().copied()
    }

    pub fn is_resultless(&self, ctx: &IrContext) -> bool {
        self.results(ctx).is_empty()
    }

    /// Compatibility accessor for the one-result code migrated in stage 2.
    #[doc(hidden)]
    pub fn r#return(&self, ctx: &IrContext) -> TypeRef {
        self.single_result(ctx)
            .expect("one-result core.func expected during staged migration")
    }

    /// Compatibility accessor for the input list, migrated in stage 2.
    #[doc(hidden)]
    pub fn params<'a>(&self, ctx: &'a IrContext) -> &'a [TypeRef] {
        self.inputs(ctx)
    }
}

impl DialectType for Func {
    const DIALECT_NAME: &'static str = "core";
    const TYPE_NAME: &'static str = "func";

    fn from_type_ref(ctx: &IrContext, ty: TypeRef) -> Option<Self> {
        if !Self::matches(ctx, ty) {
            return None;
        }
        Self::validate(ctx, ty).ok()
    }

    fn as_type_ref(&self) -> TypeRef {
        self.0
    }
}

impl From<Func> for TypeRef {
    fn from(ty: Func) -> Self {
        ty.0
    }
}

fn read_count(attrs: &AttributeMap, name: &'static str) -> Result<u32, FuncTypeError> {
    match attrs.get(name) {
        None => Err(FuncTypeError::MissingCount(name)),
        Some(Attribute::Int(value)) => {
            u32::try_from(*value).map_err(|_| FuncTypeError::InvalidCount(name))
        }
        Some(_) => Err(FuncTypeError::InvalidCount(name)),
    }
}

/// Construct a canonical `core.func` with zero or one result.
pub fn func(
    ctx: &mut IrContext,
    inputs: impl IntoIterator<Item = TypeRef>,
    results: impl IntoIterator<Item = TypeRef>,
) -> Func {
    func_with_attrs(ctx, inputs, results, AttributeMap::new())
}

/// Construct a canonical `core.func` while preserving non-reserved attributes.
pub fn func_with_attrs(
    ctx: &mut IrContext,
    inputs: impl IntoIterator<Item = TypeRef>,
    results: impl IntoIterator<Item = TypeRef>,
    attrs: AttributeMap,
) -> Func {
    assert!(
        !attrs.contains_key(NUM_INPUTS_ATTR) && !attrs.contains_key(NUM_RESULTS_ATTR),
        "core.func count attributes are reserved"
    );

    let inputs: Vec<_> = inputs.into_iter().collect();
    let results: Vec<_> = results.into_iter().collect();
    assert!(
        results.len() <= 1,
        "core.func currently supports at most one result"
    );
    let num_inputs = u32::try_from(inputs.len()).expect("core.func input count exceeds u32");
    let num_results = u32::try_from(results.len()).expect("core.func result count exceeds u32");

    let mut builder = TypeDataBuilder::new(DIALECT_NAME(), FUNC())
        .params(inputs)
        .params(results);
    for (key, value) in attrs {
        builder = builder.attr(key, value);
    }
    let ty = ctx.types.intern(
        builder
            .attr(NUM_INPUTS_ATTR, Attribute::from(num_inputs))
            .attr(NUM_RESULTS_ATTR, Attribute::from(num_results))
            .build(),
    );
    Func::validate(ctx, ty).expect("core.func constructor must produce a valid type")
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
#[trunk_ir::canonicalize_fold(core.unrealized_conversion_cast)]
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
    use crate::parser::parse_test_module;
    use crate::printer::print_module;
    use crate::rewrite::{ApplyResult, Module, PatternApplicator, TypeConverter};
    use crate::symbol::Symbol;
    use crate::walk::{WalkAction, walk_op};
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
