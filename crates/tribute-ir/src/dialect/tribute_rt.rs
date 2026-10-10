//! Tribute runtime dialect — boxing, unboxing, and reference counting.

// === Managed reference registrations ===
inventory::submit!(crate::dialect::tribute_rtti::ManagedRefType::new::<Intref>());
inventory::submit!(crate::dialect::tribute_rtti::ManagedRefType::new::<Anyref>());

#[trunk_ir::dialect]
mod tribute_rt {
    // Types
    struct Int;
    struct Nat;
    struct Float;
    struct Bool;
    struct Intref;
    struct Anyref;

    fn box_int(value: Value<_>) -> Value<_> {}
    fn unbox_int(value: Value<_>) -> Value<_> {}
    fn box_nat(value: Value<_>) -> Value<_> {}
    fn unbox_nat(value: Value<_>) -> Value<_> {}
    fn box_float(value: Value<_>) -> Value<_> {}
    fn unbox_float(value: Value<_>) -> Value<_> {}
    fn box_bool(value: Value<_>) -> Value<_> {}
    fn unbox_bool(value: Value<_>) -> Value<_> {}

    fn retain(ptr: Value<_>) -> Value<_> {}

    fn release(alloc_size: Attr<u64>, ptr: Value<_>) {}

    /// Consume one managed ownership unit while crossing a native raw-pointer
    /// representation boundary. This is deliberately not a pure operation.
    fn into_raw(value: Value<_>) -> Value<_> {}

    /// Receive one managed ownership unit from a raw pointer to a
    /// reference-counted object. The inverse of `into_raw`, and like it not a
    /// pure operation: dropping it would leak the unit.
    fn from_raw(ptr: Value<_>) -> Value<_> {}
}

// === RC Header Layout ===
// Re-exported from `tribute-rc`. See `tribute_rc::RcBox` for the layout definition.

pub use tribute_rc::HEADER_SIZE as RC_HEADER_SIZE;
pub use tribute_rc::REFCOUNT_OFFSET;
pub use tribute_rc::RTTI_IDX_OFFSET;

use trunk_ir::ops::DialectOp;

// === Pure operation registrations ===
// Boxing and unboxing operations are pure (no side effects)

inventory::submit! { trunk_ir::op_interface::PureOps::register::<BoxInt>() }
inventory::submit! { trunk_ir::op_interface::PureOps::register::<UnboxInt>() }
inventory::submit! { trunk_ir::op_interface::PureOps::register::<BoxNat>() }
inventory::submit! { trunk_ir::op_interface::PureOps::register::<UnboxNat>() }
inventory::submit! { trunk_ir::op_interface::PureOps::register::<BoxFloat>() }
inventory::submit! { trunk_ir::op_interface::PureOps::register::<UnboxFloat>() }
inventory::submit! { trunk_ir::op_interface::PureOps::register::<BoxBool>() }
inventory::submit! { trunk_ir::op_interface::PureOps::register::<UnboxBool>() }

// === Canonicalization folds ===

use trunk_ir::context::IrContext;
use trunk_ir::refs::{OpRef, ValueDef};
use trunk_ir::transforms::canonicalize::FoldResult;

/// `unbox_*(box_*(%x))` → `%x` for one box/unbox pair with the same contract,
/// provided the boxed value already has the unboxed result type.
fn fold_unbox_of_box<Box: DialectOp>(ctx: &IrContext, op: OpRef) -> Option<FoldResult> {
    let [boxed] = *<&[_; 1]>::try_from(ctx.op_operands(op)).ok()?;
    let [result_ty] = *<&[_; 1]>::try_from(ctx.op_result_types(op)).ok()?;
    let producer = match ctx.value_def(boxed) {
        ValueDef::OpResult(producer, _) => producer,
        ValueDef::BlockArg(..) => return None,
    };
    let [inner] = *<&[_; 1]>::try_from(ctx.op_operands(producer)).ok()?;
    (Box::matches(ctx, producer) && ctx.value_ty(inner) == result_ty)
        .then_some(FoldResult::Forward(inner))
}

#[trunk_ir::canonicalize_fold(UnboxInt)]
pub(crate) fn fold_unbox_int(ctx: &IrContext, op: OpRef) -> Option<FoldResult> {
    fold_unbox_of_box::<BoxInt>(ctx, op)
}

#[trunk_ir::canonicalize_fold(UnboxNat)]
pub(crate) fn fold_unbox_nat(ctx: &IrContext, op: OpRef) -> Option<FoldResult> {
    fold_unbox_of_box::<BoxNat>(ctx, op)
}

#[trunk_ir::canonicalize_fold(UnboxFloat)]
pub(crate) fn fold_unbox_float(ctx: &IrContext, op: OpRef) -> Option<FoldResult> {
    fold_unbox_of_box::<BoxFloat>(ctx, op)
}

#[trunk_ir::canonicalize_fold(UnboxBool)]
pub(crate) fn fold_unbox_bool(ctx: &IrContext, op: OpRef) -> Option<FoldResult> {
    fold_unbox_of_box::<BoxBool>(ctx, op)
}

#[cfg(test)]
mod tests {
    use trunk_ir::Span;
    use trunk_ir::Symbol;
    use trunk_ir::ops::DialectOp;
    use trunk_ir::refs::PathRef;
    use trunk_ir::types::Location;
    use trunk_ir::{Attribute, IrContext, TypeDataBuilder};

    fn dummy_location() -> Location {
        Location::new(PathRef::from_u32(0), Span::default())
    }

    fn make_i32_type(ctx: &mut IrContext) -> trunk_ir::TypeRef {
        ctx.intern_type(TypeDataBuilder::new("core", "i32").build())
    }

    fn make_ptr_type(ctx: &mut IrContext) -> trunk_ir::TypeRef {
        ctx.intern_type(TypeDataBuilder::new("core", "ptr").build())
    }

    #[test]
    fn test_single_value_ops_round_trip() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let i32_ty = make_i32_type(&mut ctx);
        let ptr_ty = make_ptr_type(&mut ctx);
        let f64_ty = ctx.intern_type(TypeDataBuilder::new("core", "f64").build());
        let bool_ty = ctx.intern_type(TypeDataBuilder::new("core", "i1").build());
        let name_attr = ctx.string_attr("Box");
        let managed_ty = ctx.intern_type(
            TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("typeref"))
                .attr("name", name_attr)
                .build(),
        );

        // Build `op` over a fresh operand of `operand_ty`, then check that
        // `from_op` matches it and that its operand accessor and result type
        // round-trip.
        macro_rules! check {
            ($op:ident, $accessor:ident, $operand_ty:expr, $result_ty:expr) => {{
                let name = stringify!($op);
                let value = trunk_ir::dialect::arith::Const::operands()
                    .value(Attribute::Int(0))
                    .results($operand_ty)
                    .build(&mut ctx, loc)
                    .result(&ctx);
                let op = super::$op::operands(value)
                    .results($result_ty)
                    .build(&mut ctx, loc);
                let round_trip = super::$op::from_op(&ctx, op.op_ref())
                    .unwrap_or_else(|_| panic!("{name}: from_op must match"));
                assert_eq!(round_trip.op_ref(), op.op_ref(), "{name}");
                assert_eq!(op.$accessor(&ctx), value, "{name}");
                assert_eq!(ctx.value_ty(op.result(&ctx)), $result_ty, "{name}");
            }};
        }

        check!(BoxInt, value, i32_ty, ptr_ty);
        check!(UnboxInt, value, ptr_ty, i32_ty);
        check!(BoxFloat, value, f64_ty, ptr_ty);
        check!(BoxBool, value, bool_ty, ptr_ty);
        check!(Retain, ptr, ptr_ty, ptr_ty);
        check!(IntoRaw, value, managed_ty, ptr_ty);
        check!(FromRaw, ptr, ptr_ty, managed_ty);
    }

    #[test]
    fn test_release_round_trip() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();

        // Create a ptr value
        let c = trunk_ir::dialect::mem::Null::operands().build(&mut ctx, loc);
        let ptr_val = c.result(&ctx);

        // Create tribute_rt.release (no result, has alloc_size attr)
        let op = super::Release::operands(ptr_val)
            .alloc_size(16u64)
            .build(&mut ctx, loc);

        // Verify from_op round-trip
        let op2 =
            super::Release::from_op(&ctx, op.op_ref()).expect("should match tribute_rt.release");
        assert_eq!(op.op_ref(), op2.op_ref());

        // Verify operand
        assert_eq!(op.ptr(&ctx), ptr_val);

        // Verify alloc_size attribute
        assert_eq!(op.alloc_size(&ctx), 16u64);
    }

    #[test]
    fn test_from_op_wrong_dialect() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let i32_ty = make_i32_type(&mut ctx);

        // Create an arith.const — should not match tribute_rt ops
        let c = trunk_ir::dialect::arith::Const::operands()
            .value(Attribute::Int(1))
            .results(i32_ty)
            .build(&mut ctx, loc);
        assert!(super::BoxInt::from_op(&ctx, c.op_ref()).is_err());
        assert!(super::UnboxInt::from_op(&ctx, c.op_ref()).is_err());
        assert!(super::Retain::from_op(&ctx, c.op_ref()).is_err());
        assert!(super::Release::from_op(&ctx, c.op_ref()).is_err());
    }

    mod fold {
        use trunk_ir::IrContext;
        use trunk_ir::parser::parse_test_module;
        use trunk_ir::printer::print_module;
        use trunk_ir::transforms::canonicalize::canonicalize;

        fn run(input: &str) -> (String, usize) {
            let mut ctx = IrContext::new();
            let module = parse_test_module(&mut ctx, input);
            let changes = canonicalize(&mut ctx, module).total_changes;
            (print_module(&ctx, module.op()), changes)
        }

        #[test]
        fn unbox_of_box_forwards_each_pair() {
            for (boxed, unboxed, ty) in [
                ("box_int", "unbox_int", "core.i64"),
                ("box_nat", "unbox_nat", "core.i64"),
                ("box_float", "unbox_float", "core.f64"),
                ("box_bool", "unbox_bool", "core.i1"),
            ] {
                let (text, changes) = run(&format!(
                    r#"core.module @test {{
  func.func @f(%x: {ty}) -> {ty} {{
    %b = tribute_rt.{boxed} %x : tribute_rt.anyref
    %u = tribute_rt.{unboxed} %b : {ty}
    func.return %u
  }}
}}"#
                ));
                assert!(changes >= 1, "{unboxed}");
                assert!(!text.contains(unboxed), "{unboxed}: {text}");
            }
        }

        #[test]
        fn unbox_of_a_different_box_kind_is_not_folded() {
            let (text, changes) = run(r#"core.module @test {
  func.func @f(%x: core.i64) -> core.f64 {
    %b = tribute_rt.box_int %x : tribute_rt.anyref
    %u = tribute_rt.unbox_float %b : core.f64
    func.return %u
  }
}"#);
            assert_eq!(changes, 0);
            assert!(text.contains("unbox_float"));
        }

        #[test]
        fn unbox_of_a_mistyped_box_is_not_folded() {
            let (text, changes) = run(r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i64 {
    %b = tribute_rt.box_int %x : tribute_rt.anyref
    %u = tribute_rt.unbox_int %b : core.i64
    func.return %u
  }
}"#);
            assert_eq!(changes, 0);
            assert!(text.contains("unbox_int"));
        }

        #[test]
        fn unbox_of_a_block_argument_is_not_folded() {
            let (text, changes) = run(r#"core.module @test {
  func.func @f(%b: tribute_rt.anyref) -> core.i64 {
    %u = tribute_rt.unbox_int %b : core.i64
    func.return %u
  }
}"#);
            assert_eq!(changes, 0);
            assert!(text.contains("unbox_int"));
        }
    }
}
