//! Lower the `std::runtime::Slots` intrinsics to `adt` array operations.
//!
//! Each intrinsic is one `adt.array_*` operation (`new-plans/types.md`,
//! "Slots"). Positions and lengths are `Nat` at the call and `core.index` at
//! the array, converted with `arith.index_castui`; a length that does not
//! fit `Nat` is truncated, as `Nat` bounds every array the runtime builds. A call is selected through
//! the declaration its callee names and that declaration's verified compiler
//! intrinsic identity, never by the callee's spelling. The declarations,
//! generic and specialized, are removed afterwards, so a declaration used
//! other than as the callee of a call of its shape, such as a function
//! value, is rejected before anything is rewritten.

use rustc_hash::FxHashMap as HashMap;
use std::ops::ControlFlow;

use tribute_ir::dialect::adt;
use tribute_ir::dialect::tribute_control::COMPILER_INTRINSIC_ATTR;
use trunk_ir::analysis::AnalysisCache;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::{arith, core, func};
use trunk_ir::ops::DialectOp;
use trunk_ir::pass::{Pass, PassRunResult};
use trunk_ir::refs::{OpRef, TypeRef, ValueRef};
use trunk_ir::rewrite::{
    Module, PatternApplicator, PatternRewriter, RewritePattern, TypeConverter,
};
use trunk_ir::types::{Attribute, Location};
use trunk_ir::walk::{WalkAction, walk_op};
use trunk_ir::{Symbol, SymbolPath};

/// A `Slots` intrinsic used other than as the callee of a call of its shape.
#[derive(Debug, derive_more::Display, derive_more::Error)]
#[display("the Slots intrinsic `{intrinsic}` is used by {user}, which is not a call of its shape")]
pub struct SlotsIntrinsicError {
    pub intrinsic: String,
    pub user: String,
}

/// One `Slots` operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlotsOp {
    Empty,
    Len,
    Get,
    Insert,
    Remove,
    Replace,
}

impl SlotsOp {
    fn from_identity(identity: &str) -> Option<Self> {
        Some(match identity.strip_prefix("std::runtime::Slots::")? {
            "empty" => Self::Empty,
            "len" => Self::Len,
            "get" => Self::Get,
            "insert" => Self::Insert,
            "remove" => Self::Remove,
            "replace" => Self::Replace,
            _ => return None,
        })
    }

    fn arity(self) -> usize {
        match self {
            Self::Empty => 0,
            Self::Len => 1,
            Self::Get | Self::Remove => 2,
            Self::Insert | Self::Replace => 3,
        }
    }
}

/// Lowers every call to a `Slots` intrinsic and removes the declarations.
pub struct LowerSlotsIntrinsics;

impl Pass for LowerSlotsIntrinsics {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "lower-slots-intrinsics"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: core::Module,
        _analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        let module = Module::from(target);
        let declarations: HashMap<SymbolPath, SlotsOp> = module
            .ops(ctx)
            .iter()
            .filter(|&&op| func::Func::matches(ctx, op))
            .filter_map(|&op| {
                let identity = ctx.op_attrs(op).get_str(COMPILER_INTRINSIC_ATTR)?;
                let slots_op = SlotsOp::from_identity(identity)?;
                let name = ctx.op_attrs(op).get_str("sym_name")?;
                Some((SymbolPath::new([Symbol::new(name)]), slots_op))
            })
            .collect();
        if declarations.is_empty() {
            return Ok(());
        }
        reject_other_references(ctx, module, &declarations)?;
        PatternApplicator::new(TypeConverter::new())
            .add_pattern(SlotsCallPattern {
                declarations: &declarations,
            })
            .add_pattern(SlotsDeclarationPattern {
                declarations: &declarations,
            })
            .apply_partial(ctx, module);
        Ok(())
    }
}

/// Fail if an operation other than a call of the intrinsic's shape refers
/// to a declaration, which erasing it would leave dangling.
fn reject_other_references(
    ctx: &IrContext,
    module: Module,
    declarations: &HashMap<SymbolPath, SlotsOp>,
) -> Result<(), SlotsIntrinsicError> {
    let mut found = None;
    let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
        let lowered_callee = func::Call::from_op(ctx, op).ok().and_then(|call| {
            let callee = call.callee(ctx);
            let slots_op = declarations.get(callee)?;
            (ctx.op_operands(op).len() == slots_op.arity() && ctx.op_results(op).len() == 1)
                .then(|| callee.clone())
        });
        ctx.op(op).attributes.visit_symbol_refs(&mut |path| {
            if found.is_none()
                && declarations.contains_key(path)
                && lowered_callee.as_ref() != Some(path)
            {
                found = Some(SlotsIntrinsicError {
                    intrinsic: path.to_string(),
                    user: ctx.op(op).name.display(ctx).to_string(),
                });
            }
        });
        ControlFlow::Continue(WalkAction::Advance)
    });
    found.map_or(Ok(()), Err)
}

struct SlotsCallPattern<'a> {
    declarations: &'a HashMap<SymbolPath, SlotsOp>,
}

impl RewritePattern for SlotsCallPattern<'_> {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(call) = func::Call::from_op(ctx, op) else {
            return false;
        };
        let Some(&slots_op) = self.declarations.get(call.callee(ctx)) else {
            return false;
        };
        // `reject_other_references` checked the call's shape.
        let args = ctx.op_operands(op).to_vec();
        let result_ty = ctx.op_result_types(op)[0];
        let location = ctx.op(op).location;
        let mut lower = Lowering {
            ctx,
            rewriter,
            location,
        };
        let lowered = match slots_op {
            SlotsOp::Empty => adt::ArrayOf::operands([])
                .r#type(result_ty)
                .build(lower.ctx, location)
                .op_ref(),
            SlotsOp::Len => {
                let len = adt::ArrayLen::operands(args[0]).build(lower.ctx, location);
                lower.insert(len.op_ref());
                arith::IndexCastui::operands(len.result(lower.ctx))
                    .results(result_ty)
                    .build(lower.ctx, location)
                    .op_ref()
            }
            SlotsOp::Get => {
                let at = lower.position(args[1]);
                adt::ArrayGet::operands(args[0], at)
                    .build(lower.ctx, location)
                    .op_ref()
            }
            SlotsOp::Insert => {
                let at = lower.position(args[1]);
                lower.splice(result_ty, args[0], at, at, Some(args[2]))
            }
            SlotsOp::Remove => {
                let at = lower.position(args[1]);
                let after = lower.successor(at);
                lower.splice(result_ty, args[0], at, after, None)
            }
            SlotsOp::Replace => {
                let at = lower.position(args[1]);
                let after = lower.successor(at);
                lower.splice(result_ty, args[0], at, after, Some(args[2]))
            }
        };
        rewriter.replace_op(lowered);
        true
    }
}

/// Builds the operations one call lowers to, inserting all but the last.
struct Lowering<'a, 'r> {
    ctx: &'a mut IrContext,
    rewriter: &'a mut PatternRewriter<'r>,
    location: Location,
}

impl Lowering<'_, '_> {
    fn insert(&mut self, op: OpRef) {
        self.rewriter.insert_op(op);
    }

    fn index_type(&mut self) -> TypeRef {
        core::Index::type_ref(self.ctx)
    }

    /// A `Nat` position as an array index. A `Nat` is never negative.
    fn position(&mut self, at: ValueRef) -> ValueRef {
        let index_ty = self.index_type();
        let cast = arith::IndexCastui::operands(at)
            .results(index_ty)
            .build(self.ctx, self.location);
        self.insert(cast.op_ref());
        cast.result(self.ctx)
    }

    /// The index after `at`.
    fn successor(&mut self, at: ValueRef) -> ValueRef {
        let index_ty = self.index_type();
        let one = arith::Const::operands()
            .value(Attribute::Int(1))
            .results(index_ty)
            .build(self.ctx, self.location);
        self.insert(one.op_ref());
        let after = arith::Addi::operands(at, one.result(self.ctx)).build(self.ctx, self.location);
        self.insert(after.op_ref());
        after.result(self.ctx)
    }

    fn splice(
        &mut self,
        array_ty: TypeRef,
        source: ValueRef,
        start: ValueRef,
        end: ValueRef,
        element: Option<ValueRef>,
    ) -> OpRef {
        adt::ArraySplice::operands(source, start, end, element)
            .r#type(array_ty)
            .build(self.ctx, self.location)
            .op_ref()
    }
}

struct SlotsDeclarationPattern<'a> {
    declarations: &'a HashMap<SymbolPath, SlotsOp>,
}

impl RewritePattern for SlotsDeclarationPattern<'_> {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(function) = func::Func::from_op(ctx, op) else {
            return false;
        };
        let name = SymbolPath::new([Symbol::new(function.sym_name(ctx))]);
        if !self.declarations.contains_key(&name) {
            return false;
        }
        rewriter.erase_op(vec![]);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;

    fn lower(input: &str) -> String {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        let core = core::Module::from_op(&ctx, module.op()).expect("core.module");
        LowerSlotsIntrinsics
            .run(&mut ctx, core, &mut Default::default())
            .unwrap();
        print_module(&ctx, module.op())
    }

    #[test]
    fn lowers_each_operation_to_one_array_operation() {
        let output = lower(
            r#"core.module @test {
  func.func @"std::runtime::Slots::empty$Nat"() -> adt.array<core.i32> attributes {abi = "intrinsic", tribute.compiler_intrinsic = "std::runtime::Slots::empty"}
  func.func @"std::runtime::Slots::len$Nat"(%s: adt.array<core.i32>) -> core.i32 attributes {abi = "intrinsic", tribute.compiler_intrinsic = "std::runtime::Slots::len"}
  func.func @"std::runtime::Slots::get$Nat"(%s: adt.array<core.i32>, %at: core.i32) -> core.i32 attributes {abi = "intrinsic", tribute.compiler_intrinsic = "std::runtime::Slots::get"}
  func.func @"std::runtime::Slots::insert$Nat"(%s: adt.array<core.i32>, %at: core.i32, %v: core.i32) -> adt.array<core.i32> attributes {abi = "intrinsic", tribute.compiler_intrinsic = "std::runtime::Slots::insert"}
  func.func @"std::runtime::Slots::remove$Nat"(%s: adt.array<core.i32>, %at: core.i32) -> adt.array<core.i32> attributes {abi = "intrinsic", tribute.compiler_intrinsic = "std::runtime::Slots::remove"}
  func.func @"std::runtime::Slots::replace$Nat"(%s: adt.array<core.i32>, %at: core.i32, %v: core.i32) -> adt.array<core.i32> attributes {abi = "intrinsic", tribute.compiler_intrinsic = "std::runtime::Slots::replace"}
  func.func @"std::runtime::Slots::empty"() -> adt.array<tribute_rt.anyref> attributes {abi = "intrinsic", tribute.compiler_intrinsic = "std::runtime::Slots::empty"}
  func.func @run(%at: core.i32, %v: core.i32) -> core.i32 {
    %0 = func.call {callee = @"std::runtime::Slots::empty$Nat"} : adt.array<core.i32>
    %1 = func.call %0, %at, %v {callee = @"std::runtime::Slots::insert$Nat"} : adt.array<core.i32>
    %2 = func.call %1, %at, %v {callee = @"std::runtime::Slots::replace$Nat"} : adt.array<core.i32>
    %3 = func.call %2, %at {callee = @"std::runtime::Slots::remove$Nat"} : adt.array<core.i32>
    %4 = func.call %1, %at {callee = @"std::runtime::Slots::get$Nat"} : core.i32
    %5 = func.call %3 {callee = @"std::runtime::Slots::len$Nat"} : core.i32
    func.return %5
  }
}"#,
        );
        assert!(!output.contains("Slots"), "{output}");
        assert!(!output.contains("func.call"), "{output}");
        for op in [
            "adt.array_of",
            "adt.array_splice",
            "adt.array_get",
            "adt.array_len",
        ] {
            assert!(output.contains(op), "{op}: {output}");
        }
        assert_eq!(output.matches("adt.array_splice").count(), 3, "{output}");
        assert_eq!(output.matches("arith.addi").count(), 2, "{output}");
    }

    #[test]
    fn handles_every_registered_identity() {
        for identity in tribute_ir::runtime_items::SLOTS_INTRINSICS {
            assert!(SlotsOp::from_identity(identity).is_some(), "{identity}");
        }
        assert_eq!(SlotsOp::from_identity("std::runtime::Slots::set"), None);
        assert_eq!(SlotsOp::from_identity("std::Int::+"), None);
        assert_eq!(LowerSlotsIntrinsics.name(), "lower-slots-intrinsics");
    }

    #[test]
    fn rejects_a_function_value_before_rewriting() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @"std::runtime::Slots::len$Nat"(%s: adt.array<core.i32>) -> core.i32 attributes {abi = "intrinsic", tribute.compiler_intrinsic = "std::runtime::Slots::len"}
  func.func @run(%s: adt.array<core.i32>) -> core.i32 {
    %0 = func.call %s {callee = @"std::runtime::Slots::len$Nat"} : core.i32
    %f = func.constant {func_ref = @"std::runtime::Slots::len$Nat"} : func.func_sig<(adt.array<core.i32>) -> core.i32>
    func.return %0
  }
}"#,
        );
        let before = print_module(&ctx, module.op());
        let core = core::Module::from_op(&ctx, module.op()).expect("core.module");
        let error = LowerSlotsIntrinsics
            .run(&mut ctx, core, &mut Default::default())
            .expect_err("a function value is rejected");
        assert!(error.to_string().contains("func.constant"), "{error}");
        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn rejects_a_call_of_another_shape_before_rewriting() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @"std::runtime::Slots::len$Nat"(%s: adt.array<core.i32>) -> core.i32 attributes {abi = "intrinsic", tribute.compiler_intrinsic = "std::runtime::Slots::len"}
  func.func @run(%s: adt.array<core.i32>, %n: core.i32) -> core.i32 {
    %0 = func.call %s {callee = @"std::runtime::Slots::len$Nat"} : core.i32
    %1 = func.call %s, %n {callee = @"std::runtime::Slots::len$Nat"} : core.i32
    func.return %0
  }
}"#,
        );
        let before = print_module(&ctx, module.op());
        let core = core::Module::from_op(&ctx, module.op()).expect("core.module");
        let error = LowerSlotsIntrinsics
            .run(&mut ctx, core, &mut Default::default())
            .expect_err("a call of another shape is rejected");
        assert!(error.to_string().contains("func.call"), "{error}");
        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn leaves_a_same_spelled_function_without_the_identity() {
        let output = lower(
            r#"core.module @test {
  func.func @"std::runtime::Slots::len"(%n: core.i32) -> core.i32 {
    func.return %n
  }
  func.func @run(%n: core.i32) -> core.i32 {
    %0 = func.call %n {callee = @"std::runtime::Slots::len"} : core.i32
    func.return %0
  }
}"#,
        );
        assert!(output.contains("func.call"), "{output}");
        assert!(!output.contains("adt.array_len"), "{output}");
    }
}
