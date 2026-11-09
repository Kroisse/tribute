//! Assign Wasm locals and write every local access as an operation.
//!
//! Before this pass, an SSA value of a `wasm.func` body names no storage.
//! The `Stackify` analysis first decides which results stay on the stack
//! from their producer to their one user and which nothing uses. This pass
//! gives every other block argument and result a local, numbered after the
//! parameters, and lists their types in the function's `locals` attribute.
//! Each operation then reads its other stack operands with `wasm.local_get`,
//! in its `StackOperandOrder`, either right before it or where `Stackify`
//! moved the read; each result it leaves on the stack is stored with
//! `wasm.local_set`, stays there, or is dropped with `wasm.drop`.
//! Loop-carried values are written to the loop body's argument locals, and a
//! `wasm.yield` that passes a value takes it from the top of the stack. The
//! result satisfies the stack contract of `design/wasm-backend.md`, "명시적
//! 지역 변수와 stack 계약".

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use trunk_ir::analysis::AnalysisCache;
use trunk_ir::callable::{CallableBody, classify_callable_body};
use trunk_ir::dialect::wasm;
use trunk_ir::ops::DialectOp;
use trunk_ir::pass::{Pass, PassRunResult};
use trunk_ir::{
    Attribute, BlockRef, IrContext, Location, OpList, OpRef, RegionList, RegionRef, TypeRef,
    ValueRef,
};

use crate::stack_effect::{
    YieldTransfer, pushed_results, stack_operand_slots, structured_result_type, yield_transfer,
};
use crate::stackify::Stackify;
use crate::{CompilationError, CompilationResult};

/// Assign the locals of each `wasm.func` body and write their accesses as
/// `wasm.local_get` and `wasm.local_set`.
pub struct ExplicitLocals;

impl Pass for ExplicitLocals {
    type Target = wasm::Func;

    fn name(&self) -> &'static str {
        "wasm-explicit-locals"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        func: wasm::Func,
        analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        let stackify = analyses.require::<Stackify>(ctx, func.op_ref());
        explicit_locals(ctx, func, &stackify)?;
        Ok(())
    }
}

/// The enclosing structured operations of a region, innermost last: the
/// locals of a loop's body block arguments, or `None`.
#[derive(Default)]
struct Labels {
    loop_locals: Vec<Option<Vec<u32>>>,
    loop_args: Vec<Option<usize>>,
}

impl Labels {
    fn push(&mut self, loop_locals: Option<Vec<u32>>) {
        self.loop_args.push(loop_locals.as_ref().map(Vec::len));
        self.loop_locals.push(loop_locals);
    }

    fn pop(&mut self) {
        self.loop_args.pop();
        self.loop_locals.pop();
    }

    /// The argument locals of the loop the `wasm.br` after `position`
    /// continues.
    fn continued_loop(&self, ctx: &IrContext, ops: &[OpRef], position: usize) -> &[u32] {
        ops.get(position + 1)
            .and_then(|&next| wasm::Br::from_op(ctx, next).ok())
            .and_then(|br| {
                self.loop_locals
                    .len()
                    .checked_sub(br.target(ctx) as usize + 1)
            })
            .and_then(|label| self.loop_locals[label].as_deref())
            .unwrap_or(&[])
    }
}

struct Locals {
    index: HashMap<ValueRef, u32>,
    types: Vec<TypeRef>,
    param_count: u32,
}

impl Locals {
    fn assign(&mut self, ctx: &IrContext, value: ValueRef) {
        let index = self.param_count + self.types.len() as u32;
        self.index.insert(value, index);
        self.types.push(ctx.value_ty(value));
    }

    fn of(&self, ctx: &IrContext, value: ValueRef, user: OpRef) -> CompilationResult<u32> {
        self.index.get(&value).copied().ok_or_else(|| {
            CompilationError::invalid_module(format!(
                "{} uses a value defined outside its function",
                ctx.op(user).name.display(ctx)
            ))
        })
    }
}

fn explicit_locals(
    ctx: &mut IrContext,
    func: wasm::Func,
    stackify: &Stackify,
) -> CompilationResult<()> {
    // A declaration has no locals, and emission rejects a malformed body
    // with its own diagnostic.
    let Ok(CallableBody::Definition { region: body, .. }) =
        classify_callable_body(ctx, func.op_ref())
    else {
        return Ok(());
    };
    let entry = single_block(ctx, body)?;
    let params = ctx.block_args(entry).to_vec();
    let mut locals = Locals {
        index: params
            .iter()
            .enumerate()
            .map(|(index, &param)| (param, index as u32))
            .collect(),
        types: Vec::new(),
        param_count: params.len() as u32,
    };
    assign_region(ctx, body, stackify, &mut locals);

    let mut rewrite = Rewrite {
        locals: &locals,
        stackify,
        hoisted: HashSet::default(),
    };
    rewrite.region(ctx, body, &mut Labels::default(), false)?;

    if !locals.types.is_empty() {
        let types = locals.types.iter().map(|&ty| Attribute::Type(ty)).collect();
        ctx.op_attrs_mut(func.op_ref())
            .insert("locals", Attribute::List(types));
    }
    Ok(())
}

fn single_block(ctx: &IrContext, region: RegionRef) -> CompilationResult<BlockRef> {
    match ctx.region(region).blocks[..] {
        [block] => Ok(block),
        _ => Err(CompilationError::unsupported_feature("multi-block regions")),
    }
}

/// Number each block argument, then each operation's nested values before
/// its own results, leaving out the results that need no local.
fn assign_region(ctx: &IrContext, region: RegionRef, stackify: &Stackify, locals: &mut Locals) {
    for &block in &ctx.region(region).blocks {
        for &arg in ctx.block_args(block) {
            if !locals.index.contains_key(&arg) {
                locals.assign(ctx, arg);
            }
        }
        for &op in &ctx.block(block).ops {
            for nested in ctx.op_regions(op) {
                assign_region(ctx, nested, stackify, locals);
            }
            for &result in ctx.op_results(op) {
                if !stackify.is_stacked(result) && !stackify.is_dropped(result) {
                    locals.assign(ctx, result);
                }
            }
        }
    }
}

/// Rewrites the regions of one function body.
struct Rewrite<'a> {
    locals: &'a Locals,
    stackify: &'a Stackify,
    /// Operand slots a `wasm.local_get` placed earlier already reads.
    hoisted: HashSet<(OpRef, u32)>,
}

impl Rewrite<'_> {
    /// Rewrite one region. `passes_result` tells whether the parent takes
    /// this region's value.
    fn region(
        &mut self,
        ctx: &mut IrContext,
        region: RegionRef,
        labels: &mut Labels,
        passes_result: bool,
    ) -> CompilationResult<()> {
        let block = single_block(ctx, region)?;
        let ops = ctx.block(block).ops.clone();
        let mut rewritten = OpList::new();
        let mut erased = Vec::new();

        for (position, &op) in ops.iter().enumerate() {
            let location = ctx.op(op).location;
            self.read_hoisted(ctx, op, location, &mut rewritten)?;

            if let Ok(yield_op) = wasm::Yield::from_op(ctx, op) {
                let value = yield_op.value(ctx);
                match yield_transfer(ctx, &ops, position, &labels.loop_args, passes_result) {
                    YieldTransfer::Stack => {
                        self.read_stack_operands(ctx, op, location, &mut rewritten)?;
                        rewritten.push(op);
                    }
                    YieldTransfer::LoopArgument | YieldTransfer::LoopArguments(_) => {
                        let arg_locals = labels.continued_loop(ctx, &ops, position).to_vec();
                        if self.stackify.is_stacked(value) {
                            for &arg_local in &arg_locals {
                                rewritten.push(local_set(ctx, arg_local, value, location));
                            }
                        } else {
                            let local = self.locals.of(ctx, value, op)?;
                            for &arg_local in &arg_locals {
                                let get = local_get(ctx, local, ctx.value_ty(value), location);
                                rewritten.push(get);
                                let got = ctx.op_result(get, 0);
                                rewritten.push(local_set(ctx, arg_local, got, location));
                            }
                        }
                        erased.push(op);
                    }
                    YieldTransfer::None => erased.push(op),
                }
                continue;
            }

            if let Ok(loop_op) = wasm::Loop::from_op(ctx, op) {
                let arg_locals = loop_arg_locals(ctx, loop_op, self.locals)?;
                let init = ctx.op_operands(op).to_vec();
                if init.len() != arg_locals.len() {
                    return Err(CompilationError::invalid_module(
                        "wasm.loop init operands and body block arguments differ in number",
                    ));
                }
                for (&value, &arg_local) in init.iter().zip(&arg_locals) {
                    let get = local_get(
                        ctx,
                        self.locals.of(ctx, value, op)?,
                        ctx.value_ty(value),
                        location,
                    );
                    rewritten.push(get);
                    let got = ctx.op_result(get, 0);
                    rewritten.push(local_set(ctx, arg_local, got, location));
                }
                for index in (0..init.len()).rev() {
                    ctx.remove_op_operand(op, index as u32);
                }
            } else {
                self.read_stack_operands(ctx, op, location, &mut rewritten)?;
            }

            let pushed = pushed_results(ctx, op);
            let label = match wasm::Loop::from_op(ctx, op) {
                Ok(loop_op) => Some(loop_arg_locals(ctx, loop_op, self.locals)?),
                Err(_) => None,
            };
            let region_passes_result = structured_result_type(ctx, op).is_some();
            rewritten.push(op);

            let regions: RegionList = ctx.op_regions(op).collect();
            if !regions.is_empty() {
                labels.push(label);
                for nested in regions {
                    self.region(ctx, nested, labels, region_passes_result)?;
                }
                labels.pop();
            }

            for &result in pushed.iter().rev() {
                if self.stackify.is_stacked(result) {
                    continue;
                }
                let consume = if self.stackify.is_dropped(result) {
                    wasm::Drop::operands(result).build(ctx, location).op_ref()
                } else {
                    let local = self.locals.of(ctx, result, op)?;
                    local_set(ctx, local, result, location)
                };
                rewritten.push(consume);
            }
        }

        // A region whose last operation is not a yield passes that
        // operation's first result.
        if passes_result
            && let Some(&last) = ops.last()
            && !wasm::Yield::matches(ctx, last)
            && let Some(&result) = ctx.op_results(last).first()
        {
            let location = ctx.op(last).location;
            let get = local_get(
                ctx,
                self.locals.of(ctx, result, last)?,
                ctx.value_ty(result),
                location,
            );
            rewritten.push(get);
            let yield_op = wasm::Yield::operands(ctx.op_result(get, 0)).build(ctx, location);
            rewritten.push(yield_op.op_ref());
        }

        ctx.set_block_ops(block, rewritten);
        for op in erased {
            ctx.remove_op(op);
        }
        Ok(())
    }

    /// Emit the local reads `Stackify` placed before `op`.
    fn read_hoisted(
        &mut self,
        ctx: &mut IrContext,
        op: OpRef,
        location: Location,
        rewritten: &mut OpList,
    ) -> CompilationResult<()> {
        for read in self.stackify.hoisted_before(op) {
            let value = ctx.op_operands(read.user)[read.slot as usize];
            let get = local_get(
                ctx,
                self.locals.of(ctx, value, read.user)?,
                ctx.value_ty(value),
                location,
            );
            rewritten.push(get);
            ctx.set_op_operand(read.user, read.slot, ctx.op_result(get, 0));
            self.hoisted.insert((read.user, read.slot));
        }
        Ok(())
    }

    /// Read each stack operand of `op` that is neither on the stack nor read
    /// earlier from its local, right before `op`.
    fn read_stack_operands(
        &mut self,
        ctx: &mut IrContext,
        op: OpRef,
        location: Location,
        rewritten: &mut OpList,
    ) -> CompilationResult<()> {
        for (value, slot) in stack_operand_slots(ctx, op) {
            if self.hoisted.contains(&(op, slot)) || self.stackify.is_stacked(value) {
                continue;
            }
            let get = local_get(
                ctx,
                self.locals.of(ctx, value, op)?,
                ctx.value_ty(value),
                location,
            );
            rewritten.push(get);
            ctx.set_op_operand(op, slot, ctx.op_result(get, 0));
        }
        Ok(())
    }
}

fn loop_arg_locals(
    ctx: &IrContext,
    loop_op: wasm::Loop,
    locals: &Locals,
) -> CompilationResult<Vec<u32>> {
    let body = single_block(ctx, loop_op.body(ctx))?;
    ctx.block_args(body)
        .iter()
        .map(|&arg| locals.of(ctx, arg, loop_op.op_ref()))
        .collect()
}

fn local_get(ctx: &mut IrContext, index: u32, ty: TypeRef, location: Location) -> OpRef {
    wasm::LocalGet::operands()
        .index(index)
        .results(ty)
        .build(ctx, location)
        .op_ref()
}

fn local_set(ctx: &mut IrContext, index: u32, value: ValueRef, location: Location) -> OpRef {
    wasm::LocalSet::operands(value)
        .index(index)
        .build(ctx, location)
        .op_ref()
}

#[cfg(test)]
mod tests {
    use trunk_ir::parser::parse_test_module;

    use crate::translate::assign_explicit_locals;
    use crate::validation::validate_wasm_stack;

    use super::*;

    fn explicit(source: &str) -> String {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, source);
        assign_explicit_locals(&mut ctx, module).expect("ExplicitLocals runs");
        validate_wasm_stack(&ctx, module).expect("the result satisfies the stack contract");
        module.op().display(&ctx).to_string()
    }

    fn explicit_error(source: &str) -> String {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, source);
        assign_explicit_locals(&mut ctx, module)
            .expect_err("ExplicitLocals rejects the body")
            .to_string()
    }

    #[test]
    fn results_that_leave_no_value_are_not_stored() {
        let ir = explicit(
            r#"core.module @test {
  wasm.func @g() -> core.nil {
    wasm.return
  }
  wasm.func @f() -> core.i32 {
    %unit = wasm.call @g() : () -> core.nil
    %opaque = wasm.nop : core.i32
    wasm.return %opaque
  }
}"#,
        );
        assert_eq!(
            ir,
            r#"core.module @test {
  wasm.func @g() -> core.nil {
      wasm.return
  }
  wasm.func @f() -> core.i32 attributes {locals = [core.nil, core.i32]} {
      %0 = wasm.call @g() : () -> core.nil
      %1 = wasm.nop : core.i32
      %2 = wasm.local_get {index = 1} : core.i32
      wasm.return %2
  }
}
"#
        );
    }

    #[test]
    fn rejects_a_loop_whose_init_and_arguments_differ() {
        let error = explicit_error(
            r#"core.module @test {
  wasm.func @f() -> core.nil {
    wasm.loop() {
      ^body(%i: core.i32):
        wasm.return
    }
    wasm.return
  }
}"#,
        );
        assert!(error.contains("differ in number"), "{error}");
    }

    #[test]
    fn rejects_a_multi_block_region() {
        let error = explicit_error(
            r#"core.module @test {
  wasm.func @f() -> core.nil {
    wasm.block {
      ^first:
        wasm.br {target = 0}
      ^second:
        wasm.br {target = 0}
    }
    wasm.return
  }
}"#,
        );
        assert!(error.contains("multi-block regions"), "{error}");
    }

    #[test]
    fn keeps_single_use_results_on_the_stack_and_reads_a_local_operand_first() {
        let ir = explicit(
            r#"core.module @test {
  wasm.func @f(%x: core.i32) -> core.i32 {
    %one = wasm.i32_const 1 : core.i32
    %sum = wasm.i32_add %x, %one : core.i32
    wasm.return %sum
  }
}"#,
        );
        assert_eq!(
            ir,
            r#"core.module @test {
  wasm.func @f(%0: core.i32) -> core.i32 {
      %1 = wasm.local_get {index = 0} : core.i32
      %2 = wasm.i32_const 1 : core.i32
      %3 = wasm.i32_add %1, %2 : core.i32
      wasm.return %3
  }
}
"#
        );
    }

    /// `%i + %k` reads `%i` before `%k`'s constant, past an `if` whose then
    /// region continues the loop with a new `%i`. The continue never reaches
    /// the add, so the earlier read is the value the add uses.
    const LOOP_ARGUMENT_READ_PAST_A_CONTINUE: &str = r#"core.module @test {
  wasm.func @count(%n: core.i32) -> core.i32 {
    %r = wasm.block -> (core.i32) {
      wasm.loop(%n) {
        ^body(%i: core.i32):
          %k = wasm.i32_const 100 : core.i32
          %limit = wasm.i32_const 5 : core.i32
          %small = wasm.i32_lt_s %i, %limit : core.i32
          wasm.if %small {
            %one = wasm.i32_const 1 : core.i32
            %next = wasm.i32_add %i, %one : core.i32
            wasm.yield %next
            wasm.br {target = 1}
          } else {
            %zero = wasm.i32_const 0 : core.i32
          }
          %sum = wasm.i32_add %i, %k : core.i32
          wasm.yield %sum
          wasm.br {target = 1}
      }
      wasm.unreachable
    }
    wasm.return %r
  }
  wasm.export_func @count as "count"
}"#;

    #[test]
    fn reads_a_loop_argument_before_a_branch_that_continues_the_loop() {
        let ir = explicit(LOOP_ARGUMENT_READ_PAST_A_CONTINUE);
        assert_eq!(
            ir,
            r#"core.module @test {
  wasm.func @count(%0: core.i32) -> core.i32 attributes {locals = [core.i32]} {
      %1 = wasm.block -> (core.i32) {
          %2 = wasm.local_get {index = 0} : core.i32
          wasm.local_set %2 {index = 1}
          wasm.loop() {
            ^bb2(%3: core.i32):
              %4 = wasm.local_get {index = 1} : core.i32
              %5 = wasm.i32_const 100 : core.i32
              %6 = wasm.local_get {index = 1} : core.i32
              %7 = wasm.i32_const 5 : core.i32
              %8 = wasm.i32_lt_s %6, %7 : core.i32
              wasm.if %8 {
                  %9 = wasm.local_get {index = 1} : core.i32
                  %10 = wasm.i32_const 1 : core.i32
                  %11 = wasm.i32_add %9, %10 : core.i32
                  wasm.local_set %11 {index = 1}
                  wasm.br {target = 1}
              } else {
                  %12 = wasm.i32_const 0 : core.i32
                  wasm.drop %12
              }
              %13 = wasm.i32_add %4, %5 : core.i32
              wasm.yield %13
              wasm.br {target = 1}
          }
          wasm.unreachable
      }
      wasm.return %1
  }
  wasm.export_func @count as "count"
}
"#
        );
    }

    #[test]
    fn a_loop_argument_read_past_a_continue_executes_in_wasmtime() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, LOOP_ARGUMENT_READ_PAST_A_CONTINUE);
        let bytes = crate::emit_module_to_wasm(&mut ctx, module)
            .expect("the loop emits")
            .bytes;
        let file = tempfile::NamedTempFile::new().expect("temporary Wasm file");
        std::fs::write(file.path(), bytes).expect("write Wasm module");
        for (n, expected) in [("2", "105\n"), ("7", "107\n")] {
            let output = std::process::Command::new("wasmtime")
                .args(["-W", "gc=y", "--invoke", "count"])
                .arg(file.path())
                .arg(n)
                .output()
                .expect("Wasmtime CLI is required");
            assert!(
                output.status.success(),
                "Wasmtime failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert_eq!(String::from_utf8_lossy(&output.stdout), expected);
        }
    }

    #[test]
    fn stores_a_value_used_twice_and_drops_an_unused_one() {
        let ir = explicit(
            r#"core.module @test {
  wasm.func @f() -> core.i32 {
    %two = wasm.i32_const 2 : core.i32
    %unused = wasm.i32_const 7 : core.i32
    %square = wasm.i32_mul %two, %two : core.i32
    wasm.return %square
  }
}"#,
        );
        assert_eq!(
            ir,
            r#"core.module @test {
  wasm.func @f() -> core.i32 attributes {locals = [core.i32]} {
      %0 = wasm.i32_const 2 : core.i32
      wasm.local_set %0 {index = 0}
      %1 = wasm.i32_const 7 : core.i32
      wasm.drop %1
      %2 = wasm.local_get {index = 0} : core.i32
      %3 = wasm.local_get {index = 0} : core.i32
      %4 = wasm.i32_mul %2, %3 : core.i32
      wasm.return %4
  }
}
"#
        );
    }

    #[test]
    fn stores_values_whose_stack_order_does_not_match_their_use() {
        let ir = explicit(
            r#"core.module @test {
  wasm.func @f() -> core.i32 {
    %a = wasm.i32_const 1 : core.i32
    %b = wasm.i32_const 2 : core.i32
    %diff = wasm.i32_sub %b, %a : core.i32
    wasm.return %diff
  }
}"#,
        );
        assert_eq!(
            ir,
            r#"core.module @test {
  wasm.func @f() -> core.i32 attributes {locals = [core.i32, core.i32]} {
      %0 = wasm.i32_const 1 : core.i32
      wasm.local_set %0 {index = 0}
      %1 = wasm.i32_const 2 : core.i32
      wasm.local_set %1 {index = 1}
      %2 = wasm.local_get {index = 1} : core.i32
      %3 = wasm.local_get {index = 0} : core.i32
      %4 = wasm.i32_sub %2, %3 : core.i32
      wasm.return %4
  }
}
"#
        );
    }

    #[test]
    fn leaves_a_value_under_a_side_effect_it_does_not_cross() {
        let ir = explicit(
            r#"core.module @test {
  wasm.func @g() -> core.nil {
    wasm.return
  }
  wasm.func @f() -> core.i32 {
    %a = wasm.i32_const 1 : core.i32
    %unit = wasm.call @g() : () -> core.nil
    %b = wasm.i32_const 2 : core.i32
    %sum = wasm.i32_add %a, %b : core.i32
    wasm.return %sum
  }
}"#,
        );
        assert_eq!(
            ir,
            r#"core.module @test {
  wasm.func @g() -> core.nil {
      wasm.return
  }
  wasm.func @f() -> core.i32 attributes {locals = [core.nil]} {
      %0 = wasm.i32_const 1 : core.i32
      %1 = wasm.call @g() : () -> core.nil
      %2 = wasm.i32_const 2 : core.i32
      %3 = wasm.i32_add %0, %2 : core.i32
      wasm.return %3
  }
}
"#
        );
    }

    #[test]
    fn takes_call_indirect_arguments_before_the_index() {
        let ir = explicit(
            r#"core.module @test {
  wasm.func @f(%index: core.i32, %value: core.i32) -> core.i32 {
    %result = wasm.call_indirect %index(%value) : (core.i32) -> core.i32
    wasm.return %result
  }
}"#,
        );
        assert_eq!(
            ir,
            r#"core.module @test {
  wasm.func @f(%0: core.i32, %1: core.i32) -> core.i32 {
      %2 = wasm.local_get {index = 1} : core.i32
      %3 = wasm.local_get {index = 0} : core.i32
      %4 = wasm.call_indirect %3(%2) : (core.i32) -> core.i32
      wasm.return %4
  }
}
"#
        );
    }

    #[test]
    fn writes_loop_carried_values_to_the_argument_locals() {
        let ir = explicit(
            r#"core.module @test {
  wasm.func @f(%n: core.i32) -> core.nil {
    wasm.block {
      wasm.loop(%n) {
        ^body(%i: core.i32):
          %next = wasm.i32_sub %i, %n : core.i32
          wasm.yield %next
          wasm.br {target = 0}
      }
    }
    wasm.return
  }
}"#,
        );
        assert_eq!(
            ir,
            r#"core.module @test {
  wasm.func @f(%0: core.i32) -> core.nil attributes {locals = [core.i32]} {
      wasm.block {
          %1 = wasm.local_get {index = 0} : core.i32
          wasm.local_set %1 {index = 1}
          wasm.loop() {
            ^bb2(%2: core.i32):
              %3 = wasm.local_get {index = 1} : core.i32
              %4 = wasm.local_get {index = 0} : core.i32
              %5 = wasm.i32_sub %3, %4 : core.i32
              wasm.local_set %5 {index = 1}
              wasm.br {target = 0}
          }
      }
      wasm.return
  }
}
"#
        );
    }

    #[test]
    fn passes_a_region_result_from_a_local_right_before_the_yield() {
        let ir = explicit(
            r#"core.module @test {
  wasm.func @f(%c: core.i32, %a: core.i32, %b: core.i32) -> core.i32 {
    %r = wasm.if %c -> (core.i32) {
      wasm.yield %a
    } else {
      wasm.yield %b
    }
    wasm.return %r
  }
}"#,
        );
        assert_eq!(
            ir,
            r#"core.module @test {
  wasm.func @f(%0: core.i32, %1: core.i32, %2: core.i32) -> core.i32 {
      %3 = wasm.local_get {index = 0} : core.i32
      %4 = wasm.if %3 -> (core.i32) {
          %5 = wasm.local_get {index = 1} : core.i32
          wasm.yield %5
      } else {
          %6 = wasm.local_get {index = 2} : core.i32
          wasm.yield %6
      }
      wasm.return %4
  }
}
"#
        );
    }
}
