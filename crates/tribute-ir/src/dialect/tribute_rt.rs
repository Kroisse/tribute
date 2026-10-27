//! Tribute runtime dialect — boxing, unboxing, reference counting, and closure convention tables.

use crate::dialect::tribute_rtti::{ManagedRef, ManagedRefType};
use trunk_ir::attr_kind::SymbolRef;
use trunk_ir::dialect::core::{Bytes, Ptr};
use trunk_ir::op_flag::Pure;

#[trunk_ir::dialect]
mod tribute_rt {
    // Types
    struct Int;
    struct Nat;
    struct Float;
    struct Bool;
    #[implements(ManagedRefType)]
    struct Intref;
    #[implements(ManagedRefType)]
    struct Anyref;
    /// A reference to a static convention table, the type of a closure's
    /// convention table field. It is unmanaged: the table is never released
    /// and holds no managed reference. Each target converts it to its own
    /// table reference, the table's address on native targets and its first
    /// slot in the function table on Wasm, before it lowers the
    /// `convention_table` and `convention_entry` operations that use it.
    struct ConventionTableRef;

    #[flags(Pure)]
    fn box_int(value: Value<_>) -> Value<_> {}
    #[implements(Fold)]
    #[flags(Pure)]
    fn unbox_int(value: Value<_>) -> Value<_> {}
    #[flags(Pure)]
    fn box_nat(value: Value<_>) -> Value<_> {}
    #[implements(Fold)]
    #[flags(Pure)]
    fn unbox_nat(value: Value<_>) -> Value<_> {}
    #[flags(Pure)]
    fn box_float(value: Value<_>) -> Value<_> {}
    #[implements(Fold)]
    #[flags(Pure)]
    fn unbox_float(value: Value<_>) -> Value<_> {}
    #[flags(Pure)]
    fn box_bool(value: Value<_>) -> Value<_> {}
    #[implements(Fold)]
    #[flags(Pure)]
    fn unbox_bool(value: Value<_>) -> Value<_> {}

    fn retain(ptr: Value<_>) -> Value<_> {}

    fn release(alloc_size: Attr<u64>, ptr: Value<_>) {}

    /// Release one unit of `ptr` and move the units of the fields `moved`
    /// were read from into `moved`. A unique owner releases only the managed
    /// fields at `release_offsets` before it is freed; a shared one acquires
    /// a unit for each moved value before it is released.
    fn release_moving(
        alloc_size: Attr<u64>,
        release_offsets: Attr<[i32]>,
        ptr: Value<_>,
        moved: Variadic<_>,
    ) {
    }

    /// Consume one managed ownership unit while crossing a native raw-pointer
    /// representation boundary. This is deliberately not a pure operation.
    fn into_raw(value: Value<impl ManagedRef>) -> Value<Ptr> {}

    /// Receive one managed ownership unit from a raw pointer to a
    /// reference-counted object. The inverse of `into_raw`, and like it not a
    /// pure operation: dropping it would leak the unit.
    fn from_raw(ptr: Value<Ptr>) -> Value<impl ManagedRef> {}

    /// A reference to the static convention table holding the entry
    /// functions `entries` in order, the value a closure stores in its
    /// convention table field. The reference is unmanaged and never
    /// released. Target lowering builds one table per distinct entry list:
    /// read-only data of function addresses on native targets, and
    /// consecutive slots of the function table on Wasm, whose start index is
    /// the reference. See the closure storage layout in `new-plans/ir.md`.
    ///
    /// ```text
    /// %table = tribute_rt.convention_table {entries = [@f]} : tribute_rt.convention_table_ref
    /// ```
    ///
    /// Its result is a `convention_table_ref` until the target converts that
    /// type, so the schema leaves it unconstrained.
    #[flags(Pure)]
    fn convention_table(entries: Attr<[SymbolRef]>) -> Value<_> {}

    /// The entry function at `index` of the convention table `table`, as an
    /// indirect callee: a load of the function address on native targets,
    /// and the function table index `table + index` on Wasm.
    ///
    /// ```text
    /// %entry = tribute_rt.convention_entry %table {index = 0} : core.i32
    /// ```
    ///
    /// `table` is a `convention_table_ref` until the target converts that
    /// type, so the schema leaves it unconstrained.
    #[flags(Pure)]
    fn convention_entry(index: Attr<u32>, table: Value<_>) -> Value<_> {}
}

impl ManagedRefType for Intref {}
impl ManagedRefType for Anyref {}

// A native `Bytes` value points to a reference-counted object.
impl ManagedRefType for Bytes {}
trunk_ir::register_interfaces!(Bytes: ManagedRefType);

// === RC Header Layout ===
// Re-exported from `tribute-rc`. See `tribute_rc::RcBox` for the layout definition.

pub use tribute_rc::HEADER_SIZE as RC_HEADER_SIZE;
pub use tribute_rc::REFCOUNT_OFFSET;
pub use tribute_rc::RTTI_IDX_OFFSET;

use trunk_ir::ops::{DialectOp, DialectType};

// === Representation changes ===

use trunk_ir::dialect::core::{self, F64, I1, I32};
use trunk_ir::refs::{TypeRef, ValueRef};
use trunk_ir::types::Location;

/// The scalar families that cross to `tribute_rt.anyref` through a box.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Scalar {
    Int,
    Nat,
    Bool,
    Float,
}

impl Scalar {
    fn of(ctx: &IrContext, ty: TypeRef) -> Option<Self> {
        if Int::matches(ctx, ty) || I32::matches(ctx, ty) {
            Some(Self::Int)
        } else if Nat::matches(ctx, ty) {
            Some(Self::Nat)
        } else if Bool::matches(ctx, ty) || I1::matches(ctx, ty) {
            Some(Self::Bool)
        } else if Float::matches(ctx, ty) || F64::matches(ctx, ty) {
            Some(Self::Float)
        } else {
            None
        }
    }
}

/// Whether converting `from` to `to` changes representation: a scalar to
/// `tribute_rt.anyref` or back.
pub fn changes_representation(ctx: &IrContext, from: TypeRef, to: TypeRef) -> bool {
    (Anyref::matches(ctx, to) && Scalar::of(ctx, from).is_some())
        || (Anyref::matches(ctx, from) && Scalar::of(ctx, to).is_some())
}

/// The box or unbox that converts `value` to `to`, if the conversion changes
/// representation: a scalar to `tribute_rt.anyref` or back.
pub fn box_or_unbox(
    ctx: &mut IrContext,
    location: Location,
    value: ValueRef,
    to: TypeRef,
) -> Option<OpRef> {
    let from = ctx.value_ty(value);
    let op = if Anyref::matches(ctx, to) {
        match Scalar::of(ctx, from)? {
            Scalar::Int => BoxInt::operands(value)
                .results(to)
                .build(ctx, location)
                .op_ref(),
            Scalar::Nat => BoxNat::operands(value)
                .results(to)
                .build(ctx, location)
                .op_ref(),
            Scalar::Bool => BoxBool::operands(value)
                .results(to)
                .build(ctx, location)
                .op_ref(),
            Scalar::Float => BoxFloat::operands(value)
                .results(to)
                .build(ctx, location)
                .op_ref(),
        }
    } else if Anyref::matches(ctx, from) {
        match Scalar::of(ctx, to)? {
            Scalar::Int => UnboxInt::operands(value)
                .results(to)
                .build(ctx, location)
                .op_ref(),
            Scalar::Nat => UnboxNat::operands(value)
                .results(to)
                .build(ctx, location)
                .op_ref(),
            Scalar::Bool => UnboxBool::operands(value)
                .results(to)
                .build(ctx, location)
                .op_ref(),
            Scalar::Float => UnboxFloat::operands(value)
                .results(to)
                .build(ctx, location)
                .op_ref(),
        }
    } else {
        return None;
    };
    Some(op)
}

/// The operation that converts `value` to `to` for a pass that erases or
/// recovers a value's type.
///
/// A representation change is boxed or unboxed here, by the pass that
/// creates the conversion. Any other pair is a
/// `core.unrealized_conversion_cast` that only retypes the value until target
/// type conversion resolves it.
pub fn convert_value(
    ctx: &mut IrContext,
    location: Location,
    value: ValueRef,
    to: TypeRef,
) -> OpRef {
    box_or_unbox(ctx, location, value, to).unwrap_or_else(|| {
        core::UnrealizedConversionCast::operands(value)
            .results(to)
            .build(ctx, location)
            .op_ref()
    })
}

// === Canonicalization folds ===

use trunk_ir::context::IrContext;
use trunk_ir::refs::{OpRef, ValueDef};
use trunk_ir::transforms::canonicalize::{Fold, FoldResult};

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

impl Fold for UnboxInt {
    fn fold(&self, ctx: &IrContext) -> Option<FoldResult> {
        fold_unbox_of_box::<BoxInt>(ctx, self.op_ref())
    }
}

impl Fold for UnboxNat {
    fn fold(&self, ctx: &IrContext) -> Option<FoldResult> {
        fold_unbox_of_box::<BoxNat>(ctx, self.op_ref())
    }
}

impl Fold for UnboxFloat {
    fn fold(&self, ctx: &IrContext) -> Option<FoldResult> {
        fold_unbox_of_box::<BoxFloat>(ctx, self.op_ref())
    }
}

impl Fold for UnboxBool {
    fn fold(&self, ctx: &IrContext) -> Option<FoldResult> {
        fold_unbox_of_box::<BoxBool>(ctx, self.op_ref())
    }
}

#[cfg(test)]
mod tests {
    use trunk_ir::Span;

    use trunk_ir::ops::DialectOp;
    use trunk_ir::refs::PathRef;
    use trunk_ir::types::Location;
    use trunk_ir::{Attribute, IrContext, TypeDataBuilder};

    fn dummy_location() -> Location {
        Location::new(PathRef::from_u32(0), Span::default())
    }

    fn make_i32_type(ctx: &mut IrContext) -> trunk_ir::TypeRef {
        TypeDataBuilder::new("core.i32").intern(ctx)
    }

    fn make_ptr_type(ctx: &mut IrContext) -> trunk_ir::TypeRef {
        trunk_ir::dialect::core::Ptr::params()
            .intern(ctx)
            .as_type_ref()
    }

    #[test]
    fn test_single_value_ops_round_trip() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let i32_ty = make_i32_type(&mut ctx);
        let ptr_ty = make_ptr_type(&mut ctx);
        let f64_ty = TypeDataBuilder::new("core.f64").intern(&mut ctx);
        let bool_ty = TypeDataBuilder::new("core.i1").intern(&mut ctx);
        let managed_ty = crate::dialect::adt::Typeref::params()
            .name("Box")
            .intern(&mut ctx)
            .as_type_ref();

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
        check!(FromRaw, ptr, ptr_ty, managed_ty);

        // `into_raw` infers its `core.ptr` result.
        let managed = trunk_ir::dialect::arith::Const::operands()
            .value(Attribute::Int(0))
            .results(managed_ty)
            .build(&mut ctx, loc)
            .result(&ctx);
        let into_raw = super::IntoRaw::operands(managed).build(&mut ctx, loc);
        assert_eq!(into_raw.value(&ctx), managed);
        assert_eq!(ctx.value_ty(into_raw.result(&ctx)), ptr_ty);
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

    #[test]
    fn convert_value_boxes_only_scalars_crossing_anyref() {
        let mut ctx = IrContext::new();
        let module = trunk_ir::parser::parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @f(%i: core.i32, %n: tribute_rt.nat, %b: core.i1, %f: core.f64, %any: tribute_rt.anyref, %p: core.ptr) {
    func.return
  }
}"#,
        );
        let function = module.ops(&ctx)[0];
        let body = ctx.op_region(function, 0).expect("function body");
        let entry = ctx.region(body).blocks[0];
        let [i, n, b, f, any, p]: [trunk_ir::refs::ValueRef; 6] =
            ctx.block_args(entry).try_into().expect("six parameters");
        let (i32_ty, nat_ty, i1_ty, f64_ty, anyref_ty) = (
            ctx.value_ty(i),
            ctx.value_ty(n),
            ctx.value_ty(b),
            ctx.value_ty(f),
            ctx.value_ty(any),
        );
        for (value, to, expected) in [
            (i, anyref_ty, "tribute_rt.box_int"),
            (n, anyref_ty, "tribute_rt.box_nat"),
            (b, anyref_ty, "tribute_rt.box_bool"),
            (f, anyref_ty, "tribute_rt.box_float"),
            (any, i32_ty, "tribute_rt.unbox_int"),
            (any, nat_ty, "tribute_rt.unbox_nat"),
            (any, i1_ty, "tribute_rt.unbox_bool"),
            (any, f64_ty, "tribute_rt.unbox_float"),
            (p, anyref_ty, "core.unrealized_conversion_cast"),
            (i, f64_ty, "core.unrealized_conversion_cast"),
        ] {
            let op = super::convert_value(&mut ctx, dummy_location(), value, to);
            assert!(ctx.op(op).name.is(&ctx, expected), "{expected}");
            assert_eq!(ctx.op_operands(op), [value], "{expected}");
            assert_eq!(ctx.op_result_types(op), [to], "{expected}");
        }
    }

    mod fold {
        use trunk_ir::IrContext;
        use trunk_ir::dialect::{core, func};
        use trunk_ir::ops::DialectOp;
        use trunk_ir::parser::parse_test_module;
        use trunk_ir::pass::PassManager;
        use trunk_ir::printer::print_module;
        use trunk_ir::transforms::Canonicalize;

        /// Canonicalize every function; the flag tells whether the IR
        /// changed.
        fn run(input: &str) -> (String, bool) {
            let mut ctx = IrContext::new();
            let module = parse_test_module(&mut ctx, input);
            let before = print_module(&ctx, module.op());
            let root = core::Module::from_op(&ctx, module.op()).expect("core.module");
            let mut pm = PassManager::new();
            pm.nest::<func::Func>().add_pass(Canonicalize);
            pm.run(&mut ctx, root, &mut Default::default())
                .expect("canonicalize");
            let after = print_module(&ctx, module.op());
            let changed = after != before;
            (after, changed)
        }

        #[test]
        fn unbox_of_box_forwards_each_pair() {
            for (boxed, unboxed, ty) in [
                ("box_int", "unbox_int", "core.i64"),
                ("box_nat", "unbox_nat", "core.i64"),
                ("box_float", "unbox_float", "core.f64"),
                ("box_bool", "unbox_bool", "core.i1"),
            ] {
                let (text, changed) = run(&format!(
                    r#"core.module @test {{
  func.func @f(%x: {ty}) -> {ty} {{
    %b = tribute_rt.{boxed} %x : tribute_rt.anyref
    %u = tribute_rt.{unboxed} %b : {ty}
    func.return %u
  }}
}}"#
                ));
                assert!(changed, "{unboxed}");
                assert!(!text.contains(unboxed), "{unboxed}: {text}");
            }
        }

        #[test]
        fn unbox_of_a_different_box_kind_is_not_folded() {
            let (text, changed) = run(r#"core.module @test {
  func.func @f(%x: core.i64) -> core.f64 {
    %b = tribute_rt.box_int %x : tribute_rt.anyref
    %u = tribute_rt.unbox_float %b : core.f64
    func.return %u
  }
}"#);
            assert!(!changed);
            assert!(text.contains("unbox_float"));
        }

        #[test]
        fn unbox_of_a_mistyped_box_is_not_folded() {
            let (text, changed) = run(r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i64 {
    %b = tribute_rt.box_int %x : tribute_rt.anyref
    %u = tribute_rt.unbox_int %b : core.i64
    func.return %u
  }
}"#);
            assert!(!changed);
            assert!(text.contains("unbox_int"));
        }

        #[test]
        fn unbox_of_a_block_argument_is_not_folded() {
            let (text, changed) = run(r#"core.module @test {
  func.func @f(%b: tribute_rt.anyref) -> core.i64 {
    %u = tribute_rt.unbox_int %b : core.i64
    func.return %u
  }
}"#);
            assert!(!changed);
            assert!(text.contains("unbox_int"));
        }
    }
}
