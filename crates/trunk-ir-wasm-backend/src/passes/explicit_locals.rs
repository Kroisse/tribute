//! Assign Wasm locals and write every local access as an operation.
//!
//! Before this pass, an SSA value of a `wasm.func` body names no storage.
//! This pass gives each block argument and each result a local, numbered
//! after the parameters, and lists their types in the function's `locals`
//! attribute. Each operation then reads its stack operands with
//! `wasm.local_get`, in its `StackOperandOrder`, and each result it leaves on
//! the stack is stored with `wasm.local_set`. Loop-carried values are written
//! to the loop body's argument locals, and a `wasm.yield` that passes a value
//! takes it from a `wasm.local_get` right before it. The result satisfies the
//! stack contract of `design/wasm-backend.md`, "명시적 지역 변수와 stack
//! 계약".
//!
//! The pass performs no stack optimization: the emitted code keeps every
//! value in its own local.

use rustc_hash::FxHashMap as HashMap;
use trunk_ir::analysis::AnalysisCache;
use trunk_ir::callable::{CallableBody, classify_callable_body};
use trunk_ir::dialect::wasm::{self, DynStackOperandOrder};
use trunk_ir::ops::DialectOp;
use trunk_ir::pass::{Pass, PassRunResult};
use trunk_ir::{
    Attribute, BlockRef, IrContext, Location, OpList, OpRef, RegionList, RegionRef, TypeRef,
    ValueRef,
};

use crate::stack_effect::{pushed_results, structured_result_type};
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
        _analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        explicit_locals(ctx, func)?;
        Ok(())
    }
}

/// The label a branch depth reaches: a loop carries the locals of its body
/// block arguments.
enum Label {
    Other,
    Loop(Vec<u32>),
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

fn explicit_locals(ctx: &mut IrContext, func: wasm::Func) -> CompilationResult<()> {
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
    assign_region(ctx, body, &mut locals);

    rewrite_region(ctx, body, &locals, &mut Vec::new(), false)?;

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
/// its own results.
fn assign_region(ctx: &IrContext, region: RegionRef, locals: &mut Locals) {
    for &block in &ctx.region(region).blocks {
        for &arg in ctx.block_args(block) {
            if !locals.index.contains_key(&arg) {
                locals.assign(ctx, arg);
            }
        }
        for &op in &ctx.block(block).ops {
            for nested in ctx.op_regions(op) {
                assign_region(ctx, nested, locals);
            }
            for &result in ctx.op_results(op) {
                locals.assign(ctx, result);
            }
        }
    }
}

/// Rewrite one region. `labels` are the enclosing structured operations,
/// innermost last; `passes_result` tells whether the parent takes this
/// region's value.
fn rewrite_region(
    ctx: &mut IrContext,
    region: RegionRef,
    locals: &Locals,
    labels: &mut Vec<Label>,
    passes_result: bool,
) -> CompilationResult<()> {
    let block = single_block(ctx, region)?;
    let ops = ctx.block(block).ops.clone();
    let mut rewritten = OpList::new();
    let mut erased = Vec::new();

    for (position, &op) in ops.iter().enumerate() {
        let location = ctx.op(op).location;
        if let Ok(yield_op) = wasm::Yield::from_op(ctx, op) {
            let value = yield_op.value(ctx);
            let branch_depth = ops
                .get(position + 1)
                .and_then(|&next| wasm::Br::from_op(ctx, next).ok())
                .map(|br| br.target(ctx) as usize);
            let label = branch_depth
                .and_then(|depth| labels.len().checked_sub(depth + 1))
                .map(|index| &labels[index]);
            if let Some(Label::Loop(arg_locals)) = label {
                let local = locals.of(ctx, value, op)?;
                for &arg_local in arg_locals {
                    let get = local_get(ctx, local, ctx.value_ty(value), location);
                    rewritten.push(get);
                    let got = ctx.op_result(get, 0);
                    rewritten.push(local_set(ctx, arg_local, got, location));
                }
                erased.push(op);
            } else if branch_depth.is_some() || (passes_result && position + 1 == ops.len()) {
                let get = local_get(
                    ctx,
                    locals.of(ctx, value, op)?,
                    ctx.value_ty(value),
                    location,
                );
                rewritten.push(get);
                ctx.set_op_operand(op, 0, ctx.op_result(get, 0));
                rewritten.push(op);
            } else {
                erased.push(op);
            }
            continue;
        }

        if let Ok(loop_op) = wasm::Loop::from_op(ctx, op) {
            let arg_locals = loop_arg_locals(ctx, loop_op, locals)?;
            let init = ctx.op_operands(op).to_vec();
            if init.len() != arg_locals.len() {
                return Err(CompilationError::invalid_module(
                    "wasm.loop init operands and body block arguments differ in number",
                ));
            }
            for (&value, &arg_local) in init.iter().zip(&arg_locals) {
                let get = local_get(
                    ctx,
                    locals.of(ctx, value, op)?,
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
            read_stack_operands(ctx, op, locals, location, &mut rewritten)?;
        }

        let pushed = pushed_results(ctx, op);
        let label = match wasm::Loop::from_op(ctx, op) {
            Ok(loop_op) => Label::Loop(loop_arg_locals(ctx, loop_op, locals)?),
            Err(_) => Label::Other,
        };
        let region_passes_result = structured_result_type(ctx, op).is_some();
        rewritten.push(op);

        let regions: RegionList = ctx.op_regions(op).collect();
        if !regions.is_empty() {
            labels.push(label);
            for nested in regions {
                rewrite_region(ctx, nested, locals, labels, region_passes_result)?;
            }
            labels.pop();
        }

        for &result in pushed.iter().rev() {
            let local = locals.of(ctx, result, op)?;
            rewritten.push(local_set(ctx, local, result, location));
        }
    }

    // A region whose last operation is not a yield passes that operation's
    // first result.
    if passes_result
        && let Some(&last) = ops.last()
        && !wasm::Yield::matches(ctx, last)
        && let Some(&result) = ctx.op_results(last).first()
    {
        let location = ctx.op(last).location;
        let get = local_get(
            ctx,
            locals.of(ctx, result, last)?,
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

/// Read each stack operand of `op` from its local, right before `op`.
fn read_stack_operands(
    ctx: &mut IrContext,
    op: OpRef,
    locals: &Locals,
    location: Location,
    rewritten: &mut OpList,
) -> CompilationResult<()> {
    let operands = ctx.op_operands(op).to_vec();
    let mut replaced = vec![false; operands.len()];
    for value in DynStackOperandOrder::of(ctx, op) {
        let index = operands
            .iter()
            .enumerate()
            .position(|(index, &operand)| operand == value && !replaced[index])
            .expect("a stack operand is one of the operation's operands");
        replaced[index] = true;
        let get = local_get(
            ctx,
            locals.of(ctx, value, op)?,
            ctx.value_ty(value),
            location,
        );
        rewritten.push(get);
        ctx.set_op_operand(op, index as u32, ctx.op_result(get, 0));
    }
    Ok(())
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

    #[test]
    fn reads_operands_and_stores_results_through_locals() {
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
  wasm.func @f(%0: core.i32) -> core.i32 attributes {locals = [core.i32, core.i32]} {
      %1 = wasm.i32_const 1 : core.i32
      wasm.local_set %1 {index = 1}
      %2 = wasm.local_get {index = 0} : core.i32
      %3 = wasm.local_get {index = 1} : core.i32
      %4 = wasm.i32_add %2, %3 : core.i32
      wasm.local_set %4 {index = 2}
      %5 = wasm.local_get {index = 2} : core.i32
      wasm.return %5
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
  wasm.func @f(%0: core.i32, %1: core.i32) -> core.i32 attributes {locals = [core.i32]} {
      %2 = wasm.local_get {index = 1} : core.i32
      %3 = wasm.local_get {index = 0} : core.i32
      %4 = wasm.call_indirect %3(%2) : (core.i32) -> core.i32
      wasm.local_set %4 {index = 2}
      %5 = wasm.local_get {index = 2} : core.i32
      wasm.return %5
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
  wasm.func @f(%0: core.i32) -> core.nil attributes {locals = [core.i32, core.i32]} {
      wasm.block {
          %1 = wasm.local_get {index = 0} : core.i32
          wasm.local_set %1 {index = 1}
          wasm.loop() {
            ^bb2(%2: core.i32):
              %3 = wasm.local_get {index = 1} : core.i32
              %4 = wasm.local_get {index = 0} : core.i32
              %5 = wasm.i32_sub %3, %4 : core.i32
              wasm.local_set %5 {index = 2}
              %6 = wasm.local_get {index = 2} : core.i32
              wasm.local_set %6 {index = 1}
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
  wasm.func @f(%0: core.i32, %1: core.i32, %2: core.i32) -> core.i32 attributes {locals = [core.i32]} {
      %3 = wasm.local_get {index = 0} : core.i32
      %4 = wasm.if %3 -> (core.i32) {
          %5 = wasm.local_get {index = 1} : core.i32
          wasm.yield %5
      } else {
          %6 = wasm.local_get {index = 2} : core.i32
          wasm.yield %6
      }
      wasm.local_set %4 {index = 3}
      %7 = wasm.local_get {index = 3} : core.i32
      wasm.return %7
  }
}
"#
        );
    }
}
