//! Address sanitizer access checks for native code.
//!
//! Inserts a runtime check before every `clif.load`, `clif.store` and
//! `clif.atomic_rmw`. The check receives the effective address and the access
//! width; the runtime decides whether the range is valid. See
//! `new-plans/cranelift-backend.md` (Sanitizer).
//!
//! ## Pipeline Position
//!
//! Runs last in native lowering, after RC lowering has produced the refcount
//! accesses, and only for a sanitized build. [`DeclareAccessChecks`] declares
//! the runtime checks in the module, and [`InstrumentMemoryAccesses`] rewrites
//! each `clif.func` with a pattern that reads no provenance, ownership or
//! liveness.
//!
//! ```text
//! %v = clif.load %p {offset = 8} : core.i32
//! // becomes
//! %o = clif.iconst {value = 8} : core.i64
//! %a = clif.iadd %p, %o : core.ptr
//! %n = clif.iconst {value = 4} : core.i64
//! %_ = clif.call %a, %n {callee = @__tribute_asan_load} : core.nil
//! %v = clif.load %p {offset = 8} : core.i32
//! ```

use std::ops::ControlFlow;

use trunk_ir::context::IrContext;
use trunk_ir::dialect::{clif, core};
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::rewrite::{
    Module, PatternApplicator, PatternRewriter, RewritePattern, TypeConverter,
};
use trunk_ir::types::{Attribute, Location};
use trunk_ir::walk::{WalkAction, walk_op};
use trunk_ir::{OpRef, OperationDataBuilder, Symbol, SymbolPath, TypeRef, ValueRef};

/// Runtime check for a read; see `tribute-runtime`'s `asan` module.
const LOAD_CHECK_FN: &str = "__tribute_asan_load";
/// Runtime check for a write or an atomic read-modify-write.
const STORE_CHECK_FN: &str = "__tribute_asan_store";

/// One memory access of generated code.
struct Access {
    addr: ValueRef,
    offset: i32,
    ty: TypeRef,
    check: &'static str,
}

fn access_of(ctx: &IrContext, op: OpRef) -> Option<Access> {
    if let Ok(load) = clif::Load::from_op(ctx, op) {
        return Some(Access {
            addr: load.addr(ctx),
            offset: load.offset(ctx),
            ty: ctx.value_ty(load.result(ctx)),
            check: LOAD_CHECK_FN,
        });
    }
    if let Ok(store) = clif::Store::from_op(ctx, op) {
        return Some(Access {
            addr: store.addr(ctx),
            offset: store.offset(ctx),
            ty: ctx.value_ty(store.value(ctx)),
            check: STORE_CHECK_FN,
        });
    }
    let rmw = clif::AtomicRmw::from_op(ctx, op).ok()?;
    Some(Access {
        addr: rmw.addr(ctx),
        offset: rmw.offset(ctx),
        ty: ctx.value_ty(rmw.value(ctx)),
        check: STORE_CHECK_FN,
    })
}

/// Bytes an access of `ty` touches on the supported 64-bit native targets.
fn access_width(ctx: &IrContext, ty: TypeRef) -> Result<i64, String> {
    let data = ctx.get_type(ty);
    if data.dialect == "core" {
        let width = data.name.with_str(|name| match name {
            "nil" => Some(0),
            "i1" | "i8" => Some(1),
            "i16" => Some(2),
            "i32" | "f32" => Some(4),
            "i64" | "f64" | "ptr" => Some(8),
            _ => None,
        });
        if let Some(width) = width {
            return Ok(width);
        }
    }
    Err(format!("memory access of type {ty} has no known width"))
}

/// Whether the operation right before `op` is already its check, which makes
/// the rewrite idempotent.
fn is_checked(ctx: &IrContext, op: OpRef, access: &Access) -> bool {
    let Some(block) = ctx.op(op).parent_block else {
        return false;
    };
    let ops = &ctx.block(block).ops;
    let previous = ops
        .iter()
        .position(|&other| other == op)
        .and_then(|position| position.checked_sub(1))
        .map(|position| ops[position]);
    previous
        .and_then(|previous| clif::Call::from_op(ctx, previous).ok())
        .is_some_and(|call| *call.callee(ctx) == access.check)
}

/// Puts a runtime check before each memory access.
struct CheckMemoryAccess;

impl RewritePattern for CheckMemoryAccess {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Some(access) = access_of(ctx, op) else {
            return false;
        };
        // The pass rejects an access of unknown width before it rewrites.
        let Ok(width) = access_width(ctx, access.ty) else {
            return false;
        };
        if width == 0 || is_checked(ctx, op, &access) {
            return false;
        }
        let loc = ctx.op(op).location;
        let i64_ty = ctx.intern_type(trunk_ir::TypeDataBuilder::new("core", "i64").build());
        let ptr_ty = core::ptr(ctx).as_type_ref();
        let nil_ty = core::nil(ctx).as_type_ref();

        let addr = if access.offset == 0 {
            access.addr
        } else {
            let offset = clif::Iconst::operands()
                .value(i64::from(access.offset))
                .results(i64_ty)
                .build(ctx, loc);
            rewriter.insert_op(offset.op_ref());
            let addr = clif::Iadd::operands(access.addr, offset.result(ctx))
                .results(ptr_ty)
                .build(ctx, loc);
            rewriter.insert_op(addr.op_ref());
            addr.result(ctx)
        };
        let width = clif::Iconst::operands()
            .value(width)
            .results(i64_ty)
            .build(ctx, loc);
        rewriter.insert_op(width.op_ref());
        let check = clif::Call::operands([addr, width.result(ctx)])
            .callee(SymbolPath::from(access.check))
            .results([nil_ty])
            .build(ctx, loc);
        rewriter.insert_op(check.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "CheckMemoryAccess"
    }
}

/// Insert an access check before every memory access of `function`.
fn instrument_function(ctx: &mut IrContext, function: clif::Func) -> Result<(), String> {
    let unsupported = walk_op(ctx, function.op_ref(), &mut |op| match access_of(ctx, op)
        .map(|access| access_width(ctx, access.ty))
    {
        Some(Err(error)) => ControlFlow::Break(error),
        _ => ControlFlow::Continue(WalkAction::Advance),
    });
    if let ControlFlow::Break(error) = unsupported {
        return Err(error);
    }
    PatternApplicator::new(TypeConverter::new())
        .add_pattern(CheckMemoryAccess)
        .apply_partial(ctx, function);
    Ok(())
}

fn declare_check(ctx: &mut IrContext, loc: Location, name: &str) -> OpRef {
    let i64_ty = ctx.intern_type(trunk_ir::TypeDataBuilder::new("core", "i64").build());
    let ptr_ty = core::ptr(ctx).as_type_ref();
    let nil_ty = core::nil(ctx).as_type_ref();
    let signature = clif::func_sig(ctx, [ptr_ty, i64_ty], [nil_ty]).as_type_ref();
    let data = OperationDataBuilder::new(loc, Symbol::new("clif"), Symbol::new("func"))
        .attr("sym_name", Attribute::String(ctx.intern_str(name)))
        .attr("type", Attribute::Type(signature))
        .attr("abi", ctx.string_attr("C"))
        .build(ctx);
    ctx.create_op(data)
}

/// Declare the runtime checks `module` does not declare yet.
fn declare_checks(ctx: &mut IrContext, module: Module) {
    let Some(first_block) = module.first_block(ctx) else {
        return;
    };
    let Some(&first_op) = ctx.block(first_block).ops.first() else {
        return;
    };
    let loc = ctx.op(first_op).location;
    for name in [LOAD_CHECK_FN, STORE_CHECK_FN] {
        let declared = ctx.block(first_block).ops.iter().any(|&op| {
            clif::Func::from_op(ctx, op).is_ok_and(|function| function.sym_name(ctx) == name)
        });
        if !declared {
            let declaration = declare_check(ctx, loc, name);
            ctx.insert_op_before(first_block, first_op, declaration);
        }
    }
}

/// Declare the runtime checks and instrument every `clif.func` of `module`.
pub fn instrument_accesses(ctx: &mut IrContext, module: Module) -> Result<(), String> {
    declare_checks(ctx, module);
    let mut functions = Vec::new();
    let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
        if let Ok(function) = clif::Func::from_op(ctx, op) {
            functions.push(function);
        }
        ControlFlow::Continue(WalkAction::Advance)
    });
    functions
        .into_iter()
        .try_for_each(|function| instrument_function(ctx, function))
}

/// Declares the runtime access checks in the module.
pub struct DeclareAccessChecks;

impl trunk_ir::pass::Pass for DeclareAccessChecks {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "sanitize-declare-access-checks"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: core::Module,
        _analyses: &mut trunk_ir::analysis::AnalysisCache,
    ) -> trunk_ir::pass::PassRunResult {
        declare_checks(ctx, target.into());
        Ok(())
    }
}

/// Checks every memory access of one `clif.func`.
pub struct InstrumentMemoryAccesses;

impl trunk_ir::pass::Pass for InstrumentMemoryAccesses {
    type Target = clif::Func;

    fn name(&self) -> &'static str {
        "sanitize-memory-accesses"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: clif::Func,
        _analyses: &mut trunk_ir::analysis::AnalysisCache,
    ) -> trunk_ir::pass::PassRunResult {
        instrument_function(ctx, target).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;

    fn run_pass(ir: &str) -> String {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, ir);
        instrument_accesses(&mut ctx, module).expect("instrumentation");
        print_module(&ctx, module.op())
    }

    #[test]
    fn every_access_is_checked_with_its_effective_address_and_width() {
        let result = run_pass(
            r#"core.module @test {
  clif.func @f(%p: core.ptr, %v: core.i64) -> core.i32 {
    %field = clif.load %p {offset = 8} : core.i32
    clif.store %v, %p {offset = 0}
    %one = clif.iconst {value = 1} : core.i32
    %old = clif.atomic_rmw %p, %one {bin_op = "add", offset = -8} : core.i32
    clif.return %field
  }
}"#,
        );
        insta::assert_snapshot!(result);
    }

    #[test]
    fn a_function_without_accesses_is_unchanged_apart_from_the_declarations() {
        let result = run_pass(
            r#"core.module @test {
  clif.func @f(%p: core.ptr) -> core.ptr {
    clif.return %p
  }
}"#,
        );
        assert_eq!(result.matches("clif.call").count(), 0);
        assert_eq!(result.matches("__tribute_asan_").count(), 2);
    }

    #[test]
    fn instrumenting_twice_changes_nothing() {
        let once = run_pass(
            r#"core.module @test {
  clif.func @f(%p: core.ptr) -> core.i64 {
    %field = clif.load %p {offset = 0} : core.i64
    clif.return %field
  }
}"#,
        );
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, &once);
        instrument_accesses(&mut ctx, module).expect("instrumentation");
        assert_eq!(print_module(&ctx, module.op()), once);
    }

    #[test]
    fn an_access_of_unknown_width_is_rejected() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  clif.func @f(%p: core.ptr) -> core.nil {
    %field = clif.load %p {offset = 0} : core.bytes
    clif.return
  }
}"#,
        );
        let error = instrument_accesses(&mut ctx, module).unwrap_err();
        assert!(error.contains("no known width"), "{error}");
    }
}
