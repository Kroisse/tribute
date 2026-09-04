//! Lower intrinsic operator calls to arith dialect operations.
//!
//! This pass transforms `func.call @"Int::(+)"(a, b)` → `arith.addi(a, b)`, etc.
//! It runs in the shared pipeline before backend-specific lowering, handling
//! arithmetic and comparison intrinsics declared in the prelude for Int, Nat, and Float.

use std::collections::HashMap;
use std::collections::HashSet;
use std::rc::Rc;

use tribute_ir::dialect::tribute_control::COMPILER_INTRINSIC_ATTR;
use trunk_ir::Symbol;
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
use trunk_ir::types::Location;

/// Lower intrinsic arithmetic/comparison calls to arith dialect operations.
///
/// Direct calls are rewritten inline (e.g. `func.call @"Int::+"(a,b)` →
/// `arith.addi`). Intrinsic `func.func` declarations — which originally
/// contain only `func.unreachable` — are given a real body so they remain
/// valid when used as first-class values (closures, `func.constant`, etc.).
pub(crate) fn lower_intrinsic_to_arith(ctx: &mut IrContext, module: Module) {
    let pattern = ArithIntrinsicPattern::new();
    let intrinsic_map: HashMap<Symbol, ArithMapping> = pattern.map.clone();
    let eligible: Rc<HashSet<Symbol>> = Rc::new(
        module
            .ops(ctx)
            .into_iter()
            .filter_map(|op| {
                let function = func::Func::from_op(ctx, op).ok()?;
                let symbol = function.sym_name(ctx);
                (ctx.op(op).attributes.get_symbol(COMPILER_INTRINSIC_ATTR) == Some(symbol)
                    && intrinsic_map.get(&symbol).is_some_and(|mapping| {
                        exact_signature(ctx, function.r#type(ctx), symbol, mapping)
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
            eligible,
        });
    applicator.apply_partial(ctx, module);
}

/// PassManager-friendly wrapper for [`lower_intrinsic_to_arith`].
pub struct LowerIntrinsicToArith;

impl Pass for LowerIntrinsicToArith {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "lower-intrinsic-to-arith"
    }

    fn run(&mut self, ctx: &mut IrContext, target: core::Module) -> PassRunResult {
        lower_intrinsic_to_arith(ctx, target.into());
        Ok(())
    }
}

/// What kind of arith operation to emit.
#[derive(Clone)]
enum ArithMapping {
    /// Binary arithmetic: addi, addf, subi, etc.
    BinaryOp(fn(&mut IrContext, Location, ValueRef, ValueRef, TypeRef) -> OpRef),
    /// Integer comparison with predicate.
    CmpI(&'static str),
    /// Float comparison with predicate.
    CmpF(&'static str),
}

/// Pattern that matches `func.call` to known arithmetic intrinsics and
/// rewrites them to the corresponding `arith.*` dialect operations.
struct ArithIntrinsicPattern {
    map: HashMap<Symbol, ArithMapping>,
    eligible: Rc<HashSet<Symbol>>,
}

impl ArithIntrinsicPattern {
    fn new() -> Self {
        let mut map = HashMap::new();

        macro_rules! binary {
            ($name:expr, $op_fn:expr) => {
                map.insert(Symbol::from_dynamic($name), ArithMapping::BinaryOp($op_fn));
            };
        }
        macro_rules! cmpi {
            ($name:expr, $pred:expr) => {
                map.insert(Symbol::from_dynamic($name), ArithMapping::CmpI($pred));
            };
        }
        macro_rules! cmpf {
            ($name:expr, $pred:expr) => {
                map.insert(Symbol::from_dynamic($name), ArithMapping::CmpF($pred));
            };
        }

        // --- Int (signed) ---
        binary!("Int::+", |ctx, loc, l, r, ty| arith::addi(
            ctx, loc, l, r, ty
        )
        .op_ref());
        binary!("Int::-", |ctx, loc, l, r, ty| arith::subi(
            ctx, loc, l, r, ty
        )
        .op_ref());
        binary!("Int::*", |ctx, loc, l, r, ty| arith::muli(
            ctx, loc, l, r, ty
        )
        .op_ref());
        binary!("Int::/", |ctx, loc, l, r, ty| arith::divsi(
            ctx, loc, l, r, ty
        )
        .op_ref());
        binary!("Int::%", |ctx, loc, l, r, ty| arith::remsi(
            ctx, loc, l, r, ty
        )
        .op_ref());
        cmpi!("Int::==", "eq");
        cmpi!("Int::!=", "ne");
        cmpi!("Int::<", "slt");
        cmpi!("Int::<=", "sle");
        cmpi!("Int::>", "sgt");
        cmpi!("Int::>=", "sge");

        // --- Nat (unsigned) ---
        binary!("Nat::+", |ctx, loc, l, r, ty| arith::addi(
            ctx, loc, l, r, ty
        )
        .op_ref());
        binary!("Nat::-", |ctx, loc, l, r, ty| arith::subi(
            ctx, loc, l, r, ty
        )
        .op_ref());
        binary!("Nat::*", |ctx, loc, l, r, ty| arith::muli(
            ctx, loc, l, r, ty
        )
        .op_ref());
        binary!("Nat::/", |ctx, loc, l, r, ty| arith::divui(
            ctx, loc, l, r, ty
        )
        .op_ref());
        binary!("Nat::%", |ctx, loc, l, r, ty| arith::remui(
            ctx, loc, l, r, ty
        )
        .op_ref());
        cmpi!("Nat::==", "eq");
        cmpi!("Nat::!=", "ne");
        cmpi!("Nat::<", "ult");
        cmpi!("Nat::<=", "ule");
        cmpi!("Nat::>", "ugt");
        cmpi!("Nat::>=", "uge");

        // --- Float ---
        binary!("Float::+", |ctx, loc, l, r, ty| arith::addf(
            ctx, loc, l, r, ty
        )
        .op_ref());
        binary!("Float::-", |ctx, loc, l, r, ty| arith::subf(
            ctx, loc, l, r, ty
        )
        .op_ref());
        binary!("Float::*", |ctx, loc, l, r, ty| arith::mulf(
            ctx, loc, l, r, ty
        )
        .op_ref());
        binary!("Float::/", |ctx, loc, l, r, ty| arith::divf(
            ctx, loc, l, r, ty
        )
        .op_ref());
        cmpf!("Float::==", "oeq");
        cmpf!("Float::!=", "une");
        cmpf!("Float::<", "olt");
        cmpf!("Float::<=", "ole");
        cmpf!("Float::>", "ogt");
        cmpf!("Float::>=", "oge");

        Self {
            map,
            eligible: Rc::new(HashSet::new()),
        }
    }

    fn with_eligible(mut self, eligible: Rc<HashSet<Symbol>>) -> Self {
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

        if !self.eligible.contains(&callee) {
            return false;
        }
        let Some(mapping) = self.map.get(&callee) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let result_ty = ctx.op_result_types(op)[0];
        let operands = ctx.op_operands(op).to_vec();
        let lhs = operands[0];
        let rhs = operands[1];

        match mapping {
            ArithMapping::BinaryOp(op_fn) => {
                let new_op = op_fn(ctx, loc, lhs, rhs, result_ty);
                rewriter.replace_op(new_op);
            }
            ArithMapping::CmpI(predicate) => {
                let cmp = arith::cmpi(ctx, loc, lhs, rhs, result_ty, Symbol::new(predicate));
                rewriter.replace_op(cmp.op_ref());
            }
            ArithMapping::CmpF(predicate) => {
                let cmp = arith::cmpf(ctx, loc, lhs, rhs, result_ty, Symbol::new(predicate));
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
    intrinsic_map: HashMap<Symbol, ArithMapping>,
    eligible: Rc<HashSet<Symbol>>,
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
        let sym_name = func_op.sym_name(ctx);
        if !self.eligible.contains(&sym_name) {
            return false;
        }
        let Some(mapping) = self.intrinsic_map.get(&sym_name) else {
            return false;
        };

        // External declarations have no body region and must remain
        // untouched. `Func::body` asserts that the region exists.
        let Some(old_body) = ctx.op(op).regions.first().copied() else {
            return false;
        };

        let loc = ctx.op(op).location;
        let func_ty = func_op.r#type(ctx);

        // All mapped intrinsics are binary (lhs, rhs) -> result.
        // This will panic if a non-binary intrinsic is ever added to the map.
        let function = core::Func::from_type_ref(ctx, func_ty)
            .expect("mapped intrinsic declaration must have a valid core.func type");
        let return_ty = function.r#return(ctx);
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
            ArithMapping::BinaryOp(op_fn) => op_fn(ctx, loc, lhs, rhs, return_ty),
            ArithMapping::CmpI(predicate) => {
                arith::cmpi(ctx, loc, lhs, rhs, return_ty, Symbol::new(predicate)).op_ref()
            }
            ArithMapping::CmpF(predicate) => {
                arith::cmpf(ctx, loc, lhs, rhs, return_ty, Symbol::new(predicate)).op_ref()
            }
        };
        ctx.push_op(body_block, result_op);

        let result_val = ctx.op_results(result_op)[0];
        let ret_op = func::r#return(ctx, loc, [result_val]);
        ctx.push_op(body_block, ret_op.op_ref());

        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: trunk_ir::smallvec::smallvec![body_block],
            parent_op: None,
        });

        // Detach old body region before replacing
        ctx.detach_region(old_body);

        let new_func = func::func(ctx, loc, sym_name, func_ty, body).op_ref();
        // Do NOT copy the "intrinsic" abi — this is now a real function
        rewriter.replace_op(new_func);
        true
    }
}

fn exact_signature(ctx: &IrContext, ty: TypeRef, symbol: Symbol, mapping: &ArithMapping) -> bool {
    let Some(function) = core::Func::from_type_ref(ctx, ty) else {
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
    let operand = ctx.types.get(*left);
    let result = ctx.types.get(result);
    let operand_is_i32 =
        operand.dialect == Symbol::new("core") && operand.name == Symbol::new("i32");
    let operand_is_f64 =
        operand.dialect == Symbol::new("core") && operand.name == Symbol::new("f64");
    match mapping {
        ArithMapping::BinaryOp(_) => {
            result.dialect == operand.dialect
                && result.name == operand.name
                && symbol.with_str(|symbol| {
                    if symbol.starts_with("Float::") {
                        operand_is_f64
                    } else {
                        operand_is_i32
                    }
                })
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
                func.func @"Nat::+"(%0: core.i32, %1: core.i32) -> core.i32
                    attributes {abi = "intrinsic", tribute.compiler_intrinsic = @"Nat::+"} {
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
            output.contains(r#"@"Nat::+""#),
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
                func.func @"Int::=="(%0: core.i32, %1: core.i32) -> core.i1
                    attributes {abi = "intrinsic", tribute.compiler_intrinsic = @"Int::=="} {
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
    fn bodyless_intrinsic_decl_is_unchanged() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"
            core.module @test {
                func.func @"Nat::+"(%0: core.i32, %1: core.i32) -> core.i32
                    attributes {abi = "intrinsic", tribute.compiler_intrinsic = @"Nat::+"}
            }
        "#,
        );

        let before = print_module(&ctx, module.op());
        lower_intrinsic_to_arith(&mut ctx, module);
        let after = print_module(&ctx, module.op());

        assert_eq!(after, before, "bodyless declaration must remain unchanged");
    }

    #[test]
    fn same_identity_with_wrong_complete_signature_is_not_lowered() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"
            core.module @test {
                func.func @"Nat::+"(%0: core.f64, %1: core.f64) -> core.f64
                    attributes {abi = "intrinsic", tribute.compiler_intrinsic = @"Nat::+"} {
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
                func.func @"Nat::+"(%0: core.i32, %1: core.i32) -> core.i32
                    attributes {abi = "intrinsic"} {
                ^bb0:
                    func.return %0
                }
                func.func @caller(%0: core.i32, %1: core.i32) -> core.i32 {
                ^bb0:
                    %2 = func.call %0, %1 {callee = @"Nat::+"} : core.i32
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
                func.func @"Nat::+"(%0: core.i32, %1: core.i32) -> core.i32
                    attributes {abi = "intrinsic", tribute.compiler_intrinsic = @"Nat::+"} {
                ^bb0:
                    func.unreachable
                }
                func.func @caller(%0: core.i32, %1: core.i32) -> core.i32 {
                ^bb0:
                    %2 = func.call %0, %1 {callee = @"Nat::+"} : core.i32
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
            output.contains(r#"@"Nat::+""#),
            "intrinsic decl should still exist:\n{output}"
        );
    }
}
