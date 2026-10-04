//! Lower intrinsic operator calls to arith dialect operations.
//!
//! This pass transforms `func.call @"std::Int::(+)"(a, b)` → `arith.addi(a, b)`, etc.
//! It runs in the shared pipeline before backend-specific lowering, handling
//! arithmetic and comparison intrinsics declared in the prelude for Int, Nat, and Float.

use rustc_hash::FxHashMap;
use std::collections::HashSet;
use std::ops::ControlFlow;
use std::rc::Rc;

use tribute_ir::dialect::tribute_control::COMPILER_INTRINSIC_ATTR;
use trunk_ir::analysis::AnalysisCache;
use trunk_ir::context::{BlockArgData, BlockData, IrContext, RegionData};
use trunk_ir::dialect::arith;
use trunk_ir::dialect::core;
use trunk_ir::dialect::func;
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::pass::{Pass, PassRunResult};
use trunk_ir::refs::{OpRef, TypeRef, ValueRef};
use trunk_ir::rewrite::{
    Module, PatternApplicator, PatternRewriter, RewritePattern, TypeConverter,
};
use trunk_ir::symbol_table::SymbolTable;
use trunk_ir::types::{Attribute, Location};
use trunk_ir::walk::{WalkAction, walk_op};
use trunk_ir::{Symbol, SymbolPath};

/// Lower intrinsic arithmetic/comparison calls to arith dialect operations.
///
/// Direct calls are rewritten inline (e.g. `func.call @"std::Int::+"(a,b)` →
/// `arith.addi`). Intrinsic `func.func` declarations — which originally
/// contain only `func.unreachable` — are given a real body so they remain
/// valid when used as first-class values (closures, `func.constant`, etc.).
pub(crate) fn lower_intrinsic_to_arith(ctx: &mut IrContext, module: Module) {
    let pattern = ArithIntrinsicPattern::new();
    let intrinsic_map: FxHashMap<SymbolPath, ArithMapping> = pattern.map.clone();
    let eligible: Rc<HashSet<SymbolPath>> = Rc::new(
        module
            .ops(ctx)
            .iter()
            .copied()
            .filter_map(|op| {
                let function = func::Func::from_op(ctx, op).ok()?;
                let symbol = SymbolPath::from(function.sym_name(ctx));
                (ctx.op(op)
                    .attributes
                    .get_str(ctx, COMPILER_INTRINSIC_ATTR)
                    .is_some_and(|identity| symbol == identity)
                    && intrinsic_map.get(&symbol).is_some_and(|mapping| {
                        exact_signature(ctx, function.r#type(ctx), &symbol, mapping)
                    }))
                .then_some(symbol)
            })
            .collect(),
    );

    let mut applicator = PatternApplicator::new(TypeConverter::new());
    applicator = applicator
        .add_pattern(pattern.with_eligible(Rc::clone(&eligible)))
        .add_pattern(ArithIntrinsicFuncDeclPattern {
            intrinsic_map,
            eligible: Rc::clone(&eligible),
        });
    applicator.apply_partial(ctx, module);

    // Every call was rewritten against the authenticated identity, so this
    // pass is its last reader. A bodyless declaration nothing references any
    // more is removed; one still referenced as a value keeps only its
    // `abi = "intrinsic"` binding. A declaration given a body above is an
    // ordinary definition now.
    let declarations: Vec<OpRef> = module
        .ops(ctx)
        .iter()
        .copied()
        .filter(|&op| {
            func::Func::from_op(ctx, op).is_ok_and(|function| {
                eligible.contains(&SymbolPath::from(&Symbol::from_dynamic(
                    function.sym_name(ctx),
                )))
            })
        })
        .collect();
    if declarations.is_empty() {
        return;
    }
    let symbols = SymbolTable::collect(ctx, module);
    let mut referenced = HashSet::new();
    let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
        for value in ctx.op(op).attributes.values() {
            if let Attribute::SymbolRef(reference) = value
                && let Some(target) = symbols.resolve(reference)
                && target != op
            {
                referenced.insert(target);
            }
        }
        ControlFlow::Continue(WalkAction::Advance)
    });
    for declaration in declarations {
        if ctx.op_has_regions(declaration) || referenced.contains(&declaration) {
            ctx.op_mut(declaration)
                .attributes
                .remove(COMPILER_INTRINSIC_ATTR);
        } else {
            ctx.detach_op(declaration);
            ctx.remove_op(declaration);
        }
    }
}

/// PassManager-friendly wrapper for [`lower_intrinsic_to_arith`].
pub struct LowerIntrinsicToArith;

impl Pass for LowerIntrinsicToArith {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "lower-intrinsic-to-arith"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: core::Module,
        _analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        lower_intrinsic_to_arith(ctx, target.into());
        Ok(())
    }
}

/// What kind of arith operation to emit.
#[derive(Clone)]
enum ArithMapping {
    /// Binary arithmetic: addi, addf, subi, etc.
    BinaryOp(fn(&mut IrContext, Location, ValueRef, ValueRef) -> OpRef),
    /// Integer comparison with predicate.
    CmpI(&'static str),
    /// Float comparison with predicate.
    CmpF(&'static str),
}

/// Pattern that matches `func.call` to known arithmetic intrinsics and
/// rewrites them to the corresponding `arith.*` dialect operations.
struct ArithIntrinsicPattern {
    map: FxHashMap<SymbolPath, ArithMapping>,
    eligible: Rc<HashSet<SymbolPath>>,
}

impl ArithIntrinsicPattern {
    fn new() -> Self {
        let mut map = FxHashMap::default();

        macro_rules! binary {
            ($name:expr, $op_fn:expr) => {
                map.insert(SymbolPath::from($name), ArithMapping::BinaryOp($op_fn));
            };
        }
        macro_rules! cmpi {
            ($name:expr, $pred:expr) => {
                map.insert(SymbolPath::from($name), ArithMapping::CmpI($pred));
            };
        }
        macro_rules! cmpf {
            ($name:expr, $pred:expr) => {
                map.insert(SymbolPath::from($name), ArithMapping::CmpF($pred));
            };
        }

        // --- Int (signed) ---
        binary!("std::Int::+", |ctx, loc, l, r| arith::Addi::operands(l, r)
            .build(ctx, loc)
            .op_ref());
        binary!("std::Int::-", |ctx, loc, l, r| arith::Subi::operands(l, r)
            .build(ctx, loc)
            .op_ref());
        binary!("std::Int::*", |ctx, loc, l, r| arith::Muli::operands(l, r)
            .build(ctx, loc)
            .op_ref());
        binary!("std::Int::/", |ctx, loc, l, r| arith::Divsi::operands(l, r)
            .build(ctx, loc)
            .op_ref());
        binary!("std::Int::%", |ctx, loc, l, r| arith::Remsi::operands(l, r)
            .build(ctx, loc)
            .op_ref());
        cmpi!("std::Int::==", "eq");
        cmpi!("std::Int::!=", "ne");
        cmpi!("std::Int::<", "slt");
        cmpi!("std::Int::<=", "sle");
        cmpi!("std::Int::>", "sgt");
        cmpi!("std::Int::>=", "sge");

        // --- Nat (unsigned) ---
        binary!("std::Nat::+", |ctx, loc, l, r| arith::Addi::operands(l, r)
            .build(ctx, loc)
            .op_ref());
        binary!("std::Nat::-", |ctx, loc, l, r| arith::Subi::operands(l, r)
            .build(ctx, loc)
            .op_ref());
        binary!("std::Nat::*", |ctx, loc, l, r| arith::Muli::operands(l, r)
            .build(ctx, loc)
            .op_ref());
        binary!("std::Nat::/", |ctx, loc, l, r| arith::Divui::operands(l, r)
            .build(ctx, loc)
            .op_ref());
        binary!("std::Nat::%", |ctx, loc, l, r| arith::Remui::operands(l, r)
            .build(ctx, loc)
            .op_ref());
        cmpi!("std::Nat::==", "eq");
        cmpi!("std::Nat::!=", "ne");
        cmpi!("std::Nat::<", "ult");
        cmpi!("std::Nat::<=", "ule");
        cmpi!("std::Nat::>", "ugt");
        cmpi!("std::Nat::>=", "uge");

        // --- Float ---
        binary!("std::Float::+", |ctx, loc, l, r| arith::Addf::operands(
            l, r
        )
        .build(ctx, loc)
        .op_ref());
        binary!("std::Float::-", |ctx, loc, l, r| arith::Subf::operands(
            l, r
        )
        .build(ctx, loc)
        .op_ref());
        binary!("std::Float::*", |ctx, loc, l, r| arith::Mulf::operands(
            l, r
        )
        .build(ctx, loc)
        .op_ref());
        binary!("std::Float::/", |ctx, loc, l, r| arith::Divf::operands(
            l, r
        )
        .build(ctx, loc)
        .op_ref());
        cmpf!("std::Float::==", "oeq");
        cmpf!("std::Float::!=", "une");
        cmpf!("std::Float::<", "olt");
        cmpf!("std::Float::<=", "ole");
        cmpf!("std::Float::>", "ogt");
        cmpf!("std::Float::>=", "oge");

        Self {
            map,
            eligible: Rc::new(HashSet::new()),
        }
    }

    fn with_eligible(mut self, eligible: Rc<HashSet<SymbolPath>>) -> Self {
        self.eligible = eligible;
        self
    }
}

impl RewritePattern for ArithIntrinsicPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(call_op) = func::Call::from_op(ctx, op) else {
            return false;
        };
        let callee = call_op.callee(ctx);

        if !self.eligible.contains(callee) {
            return false;
        }
        let Some(mapping) = self.map.get(callee) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let operands = ctx.op_operands(op).to_vec();
        let lhs = operands[0];
        let rhs = operands[1];

        match mapping {
            ArithMapping::BinaryOp(op_fn) => {
                let new_op = op_fn(ctx, loc, lhs, rhs);
                rewriter.replace_op(new_op);
            }
            ArithMapping::CmpI(predicate) => {
                let cmp = arith::Cmpi::operands(lhs, rhs)
                    .predicate(*predicate)
                    .build(ctx, loc);
                rewriter.replace_op(cmp.op_ref());
            }
            ArithMapping::CmpF(predicate) => {
                let cmp = arith::Cmpf::operands(lhs, rhs)
                    .predicate(*predicate)
                    .build(ctx, loc);
                rewriter.replace_op(cmp.op_ref());
            }
        }

        true
    }
}

/// Pattern that replaces `func.unreachable` bodies in intrinsic operator
/// declarations with real arith-dialect implementations.
///
/// This allows intrinsic operators to work as first-class values (closures,
/// `func.constant` references) while also removing the `abi = "intrinsic"`
/// marker so the backend treats them as normal functions.
struct ArithIntrinsicFuncDeclPattern {
    intrinsic_map: FxHashMap<SymbolPath, ArithMapping>,
    eligible: Rc<HashSet<SymbolPath>>,
}

impl RewritePattern for ArithIntrinsicFuncDeclPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(func_op) = func::Func::from_op(ctx, op) else {
            return false;
        };

        // Check if this is one of our known arithmetic intrinsics
        let sym_name = Symbol::from_dynamic(func_op.sym_name(ctx));
        if !self.eligible.contains(&SymbolPath::from(&sym_name)) {
            return false;
        }
        let Some(mapping) = self.intrinsic_map.get(&SymbolPath::from(&sym_name)) else {
            return false;
        };

        // External declarations have no body region and must remain
        // untouched. `Func::body` asserts that the region exists.
        let Some(old_body) = ctx.op_region(op, 0) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let func_ty = func_op.r#type(ctx);

        // All mapped intrinsics are binary (lhs, rhs) -> result.
        // This will panic if a non-binary intrinsic is ever added to the map.
        let function = func::FuncSig::from_type_ref(ctx, func_ty)
            .expect("mapped intrinsic declaration must have a valid func.func_sig type");
        if function.single_result(ctx).is_none() {
            return false;
        }
        let param_tys: Vec<TypeRef> = function.inputs(ctx).to_vec();

        // Build a new body: entry block with params → arith op → func.return
        let block_args: Vec<BlockArgData> = param_tys
            .iter()
            .map(|&ty| BlockArgData {
                ty,
                attrs: Default::default(),
            })
            .collect();
        let body_block = ctx.create_block(BlockData {
            location: loc,
            args: block_args,
            ops: Default::default(),
            parent_region: None,
        });
        let lhs = ctx.block_args(body_block)[0];
        let rhs = ctx.block_args(body_block)[1];

        let result_op = match mapping {
            ArithMapping::BinaryOp(op_fn) => op_fn(ctx, loc, lhs, rhs),
            ArithMapping::CmpI(predicate) => arith::Cmpi::operands(lhs, rhs)
                .predicate(*predicate)
                .build(ctx, loc)
                .op_ref(),
            ArithMapping::CmpF(predicate) => arith::Cmpf::operands(lhs, rhs)
                .predicate(*predicate)
                .build(ctx, loc)
                .op_ref(),
        };
        ctx.push_op(body_block, result_op);

        let result_val = ctx.op_results(result_op)[0];
        let ret_op = func::Return::operands([result_val]).build(ctx, loc);
        ctx.push_op(body_block, ret_op.op_ref());

        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: trunk_ir::smallvec::smallvec![body_block],
            parent_op: None,
        });

        // Detach old body region before replacing
        ctx.detach_region(old_body);

        let new_func = func::Func::operands()
            .sym_name(sym_name)
            .r#type(func_ty)
            .regions(body)
            .build(ctx, loc)
            .op_ref();
        // Do NOT copy the "intrinsic" abi — this is now a real function
        rewriter.replace_op(new_func);
        true
    }
}

fn exact_signature(
    ctx: &IrContext,
    ty: TypeRef,
    symbol: &SymbolPath,
    mapping: &ArithMapping,
) -> bool {
    let Some(function) = func::FuncSig::from_type_ref(ctx, ty) else {
        return false;
    };
    let [left, right] = function.inputs(ctx) else {
        return false;
    };
    let Some(result) = function.single_result(ctx) else {
        return false;
    };
    if left != right {
        return false;
    }
    let operand = ctx.get_type(*left);
    let result = ctx.get_type(result);
    let operand_is_i32 =
        operand.dialect == Symbol::new("core") && operand.name == Symbol::new("i32");
    let operand_is_f64 =
        operand.dialect == Symbol::new("core") && operand.name == Symbol::new("f64");
    match mapping {
        ArithMapping::BinaryOp(_) => {
            result.dialect == operand.dialect
                && result.name == operand.name
                && if symbol
                    .as_simple()
                    .is_some_and(|name| name.as_str().starts_with("std::Float::"))
                {
                    operand_is_f64
                } else {
                    operand_is_i32
                }
        }
        ArithMapping::CmpI(_) => {
            operand_is_i32
                && result.dialect == Symbol::new("core")
                && result.name == Symbol::new("i1")
        }
        ArithMapping::CmpF(_) => {
            operand_is_f64
                && result.dialect == Symbol::new("core")
                && result.name == Symbol::new("i1")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;

    #[test]
    fn intrinsic_decl_gets_real_body() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"
            core.module @test {
                func.func @"std::Nat::+"(%0: core.i32, %1: core.i32) -> core.i32
                    attributes {abi = "intrinsic", tribute.compiler_intrinsic = "std::Nat::+"} {
                ^bb0:
                    func.unreachable
                }
            }
        "#,
        );

        lower_intrinsic_to_arith(&mut ctx, module);

        let output = print_module(&ctx, module.op());
        // The declaration should still exist with a real body (not erased)
        assert!(
            output.contains(r#"@"std::Nat::+""#),
            "func decl should not be erased:\n{output}"
        );
        // Body should contain arith.addi, not func.unreachable
        assert!(
            output.contains("arith.addi"),
            "body should have arith.addi:\n{output}"
        );
        assert!(
            !output.contains("func.unreachable"),
            "func.unreachable should be gone:\n{output}"
        );
        // The intrinsic abi attribute should be removed
        assert!(
            !output.contains(r#"abi = "intrinsic""#),
            "intrinsic abi attribute should be removed:\n{output}"
        );
    }

    #[test]
    fn intrinsic_cmpi_gets_real_body() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"
            core.module @test {
                func.func @"std::Int::=="(%0: core.i32, %1: core.i32) -> core.i1
                    attributes {abi = "intrinsic", tribute.compiler_intrinsic = "std::Int::=="} {
                ^bb0:
                    func.unreachable
                }
            }
        "#,
        );

        lower_intrinsic_to_arith(&mut ctx, module);

        let output = print_module(&ctx, module.op());
        assert!(
            output.contains("arith.cmpi"),
            "body should have arith.cmpi:\n{output}"
        );
        assert!(
            !output.contains("func.unreachable"),
            "func.unreachable should be gone:\n{output}"
        );
    }

    #[test]
    fn unreferenced_bodyless_intrinsic_decl_is_removed() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"
            core.module @test {
                func.func @"std::Nat::+"(%0: core.i32, %1: core.i32) -> core.i32
                    attributes {abi = "intrinsic", tribute.compiler_intrinsic = "std::Nat::+"}
                func.func @caller(%0: core.i32, %1: core.i32) -> core.i32 {
                ^bb0:
                    %2 = func.call %0, %1 {callee = @"std::Nat::+"} : core.i32
                    func.return %2
                }
            }
        "#,
        );

        lower_intrinsic_to_arith(&mut ctx, module);
        let after = print_module(&ctx, module.op());

        assert!(after.contains("arith.addi"), "{after}");
        assert!(!after.contains(r#"@"std::Nat::+""#), "{after}");
    }

    #[test]
    fn referenced_bodyless_intrinsic_decl_keeps_binding_and_consumes_identity() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"
            core.module @test {
                func.func @"std::Nat::+"(%0: core.i32, %1: core.i32) -> core.i32
                    attributes {abi = "intrinsic", tribute.compiler_intrinsic = "std::Nat::+"}
                func.func @user() -> func.func_sig<(core.i32, core.i32) -> core.i32> {
                ^bb0:
                    %f = func.constant {func_ref = @"std::Nat::+"} : func.func_sig<(core.i32, core.i32) -> core.i32>
                    func.return %f
                }
            }
        "#,
        );

        lower_intrinsic_to_arith(&mut ctx, module);
        let after = print_module(&ctx, module.op());

        assert!(
            after.contains(r#"func.func @"std::Nat::+"(%arg0: core.i32, %arg1: core.i32) -> core.i32 attributes {abi = "intrinsic"}"#)
                && !after.contains(COMPILER_INTRINSIC_ATTR),
            "a declaration referenced as a value keeps its binding without the identity:\n{after}"
        );
    }

    #[test]
    fn same_identity_with_wrong_complete_signature_is_not_lowered() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"
            core.module @test {
                func.func @"std::Nat::+"(%0: core.f64, %1: core.f64) -> core.f64
                    attributes {abi = "intrinsic", tribute.compiler_intrinsic = "std::Nat::+"} {
                ^bb0:
                    func.unreachable
                }
            }
        "#,
        );

        let before = print_module(&ctx, module.op());
        lower_intrinsic_to_arith(&mut ctx, module);
        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn same_spelled_unregistered_declaration_is_not_lowered() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"
            core.module @test {
                func.func @"std::Nat::+"(%0: core.i32, %1: core.i32) -> core.i32
                    attributes {abi = "intrinsic"} {
                ^bb0:
                    func.return %0
                }
                func.func @caller(%0: core.i32, %1: core.i32) -> core.i32 {
                ^bb0:
                    %2 = func.call %0, %1 {callee = @"std::Nat::+"} : core.i32
                    func.return %2
                }
            }
        "#,
        );

        let before = print_module(&ctx, module.op());
        lower_intrinsic_to_arith(&mut ctx, module);
        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn direct_calls_still_rewritten() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"
            core.module @test {
                func.func @"std::Nat::+"(%0: core.i32, %1: core.i32) -> core.i32
                    attributes {abi = "intrinsic", tribute.compiler_intrinsic = "std::Nat::+"} {
                ^bb0:
                    func.unreachable
                }
                func.func @caller(%0: core.i32, %1: core.i32) -> core.i32 {
                ^bb0:
                    %2 = func.call %0, %1 {callee = @"std::Nat::+"} : core.i32
                    func.return %2
                }
            }
        "#,
        );

        lower_intrinsic_to_arith(&mut ctx, module);

        let output = print_module(&ctx, module.op());
        // Direct call should be replaced by arith.addi in @caller
        assert!(
            output.contains("arith.addi"),
            "direct call should be rewritten to arith.addi:\n{output}"
        );
        // The intrinsic decl should still exist (for first-class usage)
        assert!(
            output.contains(r#"@"std::Nat::+""#),
            "intrinsic decl should still exist:\n{output}"
        );
    }
}
