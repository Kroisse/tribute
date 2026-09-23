//! Native lowering passes for Tribute.
//!
//! This module contains Tribute-specific passes that lower high-level Tribute IR
//! to native (Cranelift) dialect operations.
//!
//! ## Passes
//!
//! - `entrypoint`: Generate C ABI `main` wrapper for native binaries
//! - `type_converter`: Native type converter for IR-level type transformations
//! - `adt_rc_header`: Lower `adt.struct_new` to clif alloc + RC header init + field stores
//! - `tribute_rt_to_clif`: Lower `tribute_rt.box_*`/`unbox_*` to clif alloc + load/store
//! - `rc_optimization`: Eliminate redundant local retain/release pairs
//! - `rc_lowering`: Lower `tribute_rt.retain`/`release` to inline `clif.*` ops

pub mod adt_rc_header;
pub mod const_to_native;
pub mod entrypoint;
pub mod evidence;
pub mod intrinsic_to_native;
pub mod io;
pub mod list;
pub mod ownership_plan;
pub mod rc_lowering;
pub mod rc_materialization;
pub mod rc_optimization;
pub mod rtti;
pub mod tribute_rt_to_clif;
pub mod type_converter;

use trunk_ir::OperationDataBuilder;
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::func;
use trunk_ir::refs::{OpRef, TypeRef};
use trunk_ir::types::{Attribute, Location};

/// Build a bodyless extern `func.func` declaration with `abi = "C"`.
///
/// This is the common pattern for declaring external runtime functions
/// that are linked at native code generation time.
pub(crate) fn build_extern_func(
    ctx: &mut IrContext,
    loc: Location,
    name: &str,
    params: &[TypeRef],
    result: TypeRef,
) -> OpRef {
    let func_ty = func::func_sig(ctx, params.iter().copied(), [result]).as_type_ref();

    let data = OperationDataBuilder::new(loc, Symbol::new("func"), Symbol::new("func"))
        .attr("sym_name", Attribute::Symbol(Symbol::from_dynamic(name)))
        .attr("type", Attribute::Type(func_ty))
        .attr("abi", Attribute::String("C".to_owned()))
        .build(ctx);
    ctx.create_op(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::Span;
    use trunk_ir::callable::{CallableBody, classify_callable_body};
    use trunk_ir::dialect::core;
    use trunk_ir::ops::DialectType;

    #[test]
    fn runtime_externs_have_a_signature_and_binding_but_no_body() {
        let mut ctx = IrContext::new();
        let path = ctx.intern_path("test.ir".to_owned());
        let loc = Location::new(path, Span::new(0, 0));
        let ptr = core::ptr(&mut ctx).as_type_ref();
        let i32 = ctx.intern_type(
            trunk_ir::TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build(),
        );
        let op = build_extern_func(&mut ctx, loc, "runtime", &[ptr, i32], ptr);
        assert_eq!(
            classify_callable_body(&ctx, op),
            Ok(CallableBody::Declaration)
        );
        assert_eq!(ctx.op(op).attributes.get_str("abi"), Some("C"));
        let signature =
            func::FuncSig::from_type_ref(&ctx, ctx.op(op).attributes.get_type("type").unwrap())
                .unwrap();
        assert_eq!(signature.inputs(&ctx), &[ptr, i32]);
        assert_eq!(signature.results(&ctx), &[ptr]);
    }
}
