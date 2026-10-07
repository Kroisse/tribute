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
//! accesses, and only for a sanitized build. Each access is rewritten in
//! place; the pass reads no provenance, ownership or liveness.
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

use trunk_ir::context::IrContext;
use trunk_ir::dialect::{clif, core};
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::rewrite::Module;
use trunk_ir::types::{Attribute, Location};
use trunk_ir::{
    BlockRef, OpList, OpRef, OperationDataBuilder, RegionList, Symbol, SymbolPath, TypeRef,
    ValueRef,
};

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

fn insert_check(ctx: &mut IrContext, block: BlockRef, op: OpRef, access: &Access, width: i64) {
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
        ctx.insert_op_before(block, op, offset.op_ref());
        let addr = clif::Iadd::operands(access.addr, offset.result(ctx))
            .results(ptr_ty)
            .build(ctx, loc);
        ctx.insert_op_before(block, op, addr.op_ref());
        addr.result(ctx)
    };
    let width = clif::Iconst::operands()
        .value(width)
        .results(i64_ty)
        .build(ctx, loc);
    ctx.insert_op_before(block, op, width.op_ref());
    let call = clif::Call::operands([addr, width.result(ctx)])
        .callee(SymbolPath::from(access.check))
        .results([nil_ty])
        .build(ctx, loc);
    ctx.insert_op_before(block, op, call.op_ref());
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

/// Insert an access check before every memory access of every defined
/// `clif.func` in `module`.
pub fn instrument_accesses(ctx: &mut IrContext, module: Module) -> Result<(), String> {
    let Some(first_block) = module.first_block(ctx) else {
        return Ok(());
    };
    let module_ops: OpList = ctx.block(first_block).ops.clone();
    let mut declared = [LOAD_CHECK_FN, STORE_CHECK_FN].map(|name| (name, false));
    for &op in &module_ops {
        let Ok(function) = clif::Func::from_op(ctx, op) else {
            continue;
        };
        for (name, found) in &mut declared {
            *found |= function.sym_name(ctx) == *name;
        }
        let regions: RegionList = ctx.op_regions(op).collect();
        for region in regions {
            for block in ctx.region(region).blocks.clone() {
                let ops: OpList = ctx.block(block).ops.clone();
                for op in ops {
                    let Some(access) = access_of(ctx, op) else {
                        continue;
                    };
                    let width = access_width(ctx, access.ty)?;
                    if width != 0 {
                        insert_check(ctx, block, op, &access, width);
                    }
                }
            }
        }
    }
    let Some(&first_op) = module_ops.first() else {
        return Ok(());
    };
    let loc = ctx.op(first_op).location;
    for (name, found) in declared {
        if !found {
            let declaration = declare_check(ctx, loc, name);
            ctx.insert_op_before(first_block, first_op, declaration);
        }
    }
    Ok(())
}

/// Pass form of the access-check instrumentation.
pub struct InstrumentMemoryAccesses;

impl trunk_ir::pass::Pass for InstrumentMemoryAccesses {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "sanitize-memory-accesses"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: core::Module,
        _analyses: &mut trunk_ir::analysis::AnalysisCache,
    ) -> trunk_ir::pass::PassRunResult {
        instrument_accesses(ctx, target.into()).map_err(Into::into)
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
    fn existing_declarations_are_not_repeated() {
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
        let twice = print_module(&ctx, module.op());
        assert_eq!(twice.matches("sym_name = \"__tribute_asan_").count(), 2);
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
