//! Arena-based func dialect.

use crate::op_interface::{IndirectCallLikeModel, IndirectCallLikeOps};
use crate::ops::{DialectOp, DialectType};

/// The optional exact callable signature retained after a typed indirect callee
/// becomes a runtime function or table index.
const INDIRECT_CALL_SIGNATURE_ATTR: &str = "signature";

/// Shared accessors for function calls with ordinary results or arguments.
///
/// This is a static Rust interface over the four `func.*call*` operations;
/// it does not participate in dynamic operation dispatch.
pub trait CallLike: crate::ops::DialectOp {
    /// Arguments supplied to the callable, excluding an indirect callee.
    fn call_args<'a>(&self, ctx: &'a crate::IrContext) -> &'a [crate::ValueRef];

    /// Result types produced by this call.
    fn call_result_types<'a>(&self, ctx: &'a crate::IrContext) -> &'a [crate::TypeRef] {
        ctx.op_result_types(self.op_ref())
    }
}

/// Marker and resultless query for proper-tail transfers.
///
/// Exact callable signatures belong only to `IndirectCallLike`, keeping
/// CPS/tail legality independent from indirect-call metadata.
pub trait TailCallLike: CallLike {
    /// Proper tail transfers cannot produce SSA results.
    fn is_resultless(&self, ctx: &crate::IrContext) -> bool {
        self.call_result_types(ctx).is_empty()
    }
}

// === Operation registrations ===
crate::register_pure_op!(func.constant);
crate::register_isolated_op!(func.func);

#[trunk_ir::dialect]
mod func {
    #[attr(sym_name: Symbol, r#type: Type)]
    fn func() {
        #[region(body)]
        {}
    }

    #[attr(callee: Symbol)]
    fn call(#[rest] args: ()) -> result {}

    #[attr(signature?: Type)]
    fn call_indirect(callee: (), #[rest] args: ()) -> result {}

    #[attr(callee: Symbol)]
    fn tail_call(#[rest] args: ()) {}

    #[attr(signature?: Type)]
    fn tail_call_indirect(callee: (), #[rest] args: ()) {}

    fn r#return(#[rest] values: ()) {}

    #[attr(func_ref: Symbol)]
    fn constant() -> result {}

    fn unreachable() {}
}

impl CallLike for Call {
    fn call_args<'a>(&self, ctx: &'a crate::IrContext) -> &'a [crate::ValueRef] {
        self.args(ctx)
    }
}

impl CallLike for CallIndirect {
    fn call_args<'a>(&self, ctx: &'a crate::IrContext) -> &'a [crate::ValueRef] {
        self.args(ctx)
    }
}

impl IndirectCallLikeModel for CallIndirect {
    fn exact_signature(self, ctx: &crate::IrContext) -> Option<crate::TypeRef> {
        CallIndirect::signature(&self, ctx)
    }

    fn set_exact_signature(self, ctx: &mut crate::IrContext, signature: crate::TypeRef) -> bool {
        set_indirect_call_signature(ctx, self.op_ref(), signature)
    }
}

impl CallLike for TailCall {
    fn call_args<'a>(&self, ctx: &'a crate::IrContext) -> &'a [crate::ValueRef] {
        self.args(ctx)
    }
}

impl TailCallLike for TailCall {}

impl CallLike for TailCallIndirect {
    fn call_args<'a>(&self, ctx: &'a crate::IrContext) -> &'a [crate::ValueRef] {
        self.args(ctx)
    }
}

impl IndirectCallLikeModel for TailCallIndirect {
    fn exact_signature(self, ctx: &crate::IrContext) -> Option<crate::TypeRef> {
        TailCallIndirect::signature(&self, ctx)
    }

    fn set_exact_signature(self, ctx: &mut crate::IrContext, signature: crate::TypeRef) -> bool {
        set_indirect_call_signature(ctx, self.op_ref(), signature)
    }
}

impl TailCallLike for TailCallIndirect {}

inventory::submit! {
    IndirectCallLikeOps::register::<CallIndirect>()
}

inventory::submit! {
    IndirectCallLikeOps::register::<TailCallIndirect>()
}

/// Attach an exact callable contract to a `func` indirect transfer.
///
/// Returns `false` without mutation when the operation is not a `func`
/// indirect call or the supplied type is not a `core.func` contract.
pub fn set_indirect_call_signature(
    ctx: &mut crate::IrContext,
    op: crate::OpRef,
    signature: crate::TypeRef,
) -> bool {
    if crate::dialect::core::Func::from_type_ref(ctx, signature).is_none()
        || (CallIndirect::from_op(ctx, op).is_err() && TailCallIndirect::from_op(ctx, op).is_err())
    {
        return false;
    }
    set_indirect_call_signature_attribute(&mut ctx.op_mut(op).attributes, signature);
    true
}

fn set_indirect_call_signature_attribute(
    attributes: &mut crate::AttributeMap,
    signature: crate::TypeRef,
) {
    attributes.insert(
        crate::Symbol::new(INDIRECT_CALL_SIGNATURE_ATTR),
        crate::Attribute::Type(signature),
    );
}

/// Remove the `func`-owned exact-signature attribute from copied metadata.
pub fn remove_indirect_call_signature(attributes: &mut crate::AttributeMap) {
    attributes.remove(INDIRECT_CALL_SIGNATURE_ATTR);
}

impl Func {
    /// Return the function body when this declaration has one.
    pub fn body_if_present(&self, ctx: &crate::IrContext) -> Option<crate::RegionRef> {
        ctx.op(self.op_ref()).regions.first().copied()
    }
}

// === Custom assembly format for func.func ===

/// Print func.func with decomposed signature:
/// `func.func @name(%arg: type, ...) -> return_type effects eff_type { body }`
fn print_func(
    h: &mut crate::printer::OpPrintHelper<'_, '_>,
    op: crate::OpRef,
    indent: usize,
) -> std::fmt::Result {
    use std::fmt::Write;

    let indent_str = " ".repeat(indent);

    // Extract sym_name before mutable operations
    let sym_name = {
        let data = h.ctx().op(op);
        data.attributes.get_symbol("sym_name")
    };

    write!(h, "{indent_str}func.func")?;

    // Function name
    if let Some(name) = sym_name {
        write!(h, " @")?;
        name.with_str(|s| {
            let needs_quoting =
                s.is_empty() || !s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
            if needs_quoting {
                write!(h, "\"")?;
                crate::printer::write_escaped_string(h, s)?;
                write!(h, "\"")
            } else {
                write!(h, "{s}")
            }
        })?;
    }

    // Reset numbering for function body
    h.reset_numbering();

    // Extract region ref and func type info before mutable operations
    let region = {
        let data = h.ctx().op(op);
        assert!(
            data.regions.len() <= 1,
            "print_func: expected at most one region, found {}",
            data.regions.len(),
        );
        data.regions.first().copied()
    };

    // Extract type decomposition: return type from core.func type attribute
    let type_info = {
        let data = h.ctx().op(op);
        if let Some(func_ty) = data.attributes.get_type("type") {
            let ty_data = h.ctx().types.get(func_ty);
            let is_core_func = ty_data.dialect == crate::Symbol::new("core")
                && ty_data.name == crate::Symbol::new("func");
            if is_core_func && !ty_data.params.is_empty() {
                let result_ty = ty_data.params[0];
                let param_tys: Vec<crate::TypeRef> = ty_data.params[1..].to_vec();
                Some((result_ty, param_tys))
            } else {
                // Non-standard or empty core.func type — skip signature
                None
            }
        } else {
            None
        }
    };

    if let Some(region) = region {
        // Print entry block args as function signature
        let entry_args: Vec<_> = {
            let blocks = &h.ctx().region(region).blocks;
            blocks
                .first()
                .map(|&b| h.ctx().block_args(b).to_vec())
                .unwrap_or_default()
        };

        write!(h, "(")?;
        for (i, &arg) in entry_args.iter().enumerate() {
            if i > 0 {
                write!(h, ", ")?;
            }
            let name = h.assign_value_name(arg);
            let ty = h.ctx().value_ty(arg);
            write!(h, "{name}: ")?;
            h.write_type(ty)?;
        }
        write!(h, ")")?;
    } else if let Some((_, ref param_tys)) = type_info {
        // Body-less declaration: synthesize params from type
        write!(h, "(")?;
        for (i, &ty) in param_tys.iter().enumerate() {
            if i > 0 {
                write!(h, ", ")?;
            }
            write!(h, "%arg{i}: ")?;
            h.write_type(ty)?;
        }
        write!(h, ")")?;
    } else {
        write!(h, "()")?;
    }

    if let Some((result_ty, _)) = type_info {
        write!(h, " -> ")?;
        h.write_type(result_ty)?;
    }

    // Extra attributes (everything except sym_name and type, which are
    // already encoded in the signature).  Clone to avoid borrow conflicts
    // with the mutable write helpers.
    let extra_attrs: Vec<_> = {
        let data = h.ctx().op(op);
        data.attributes
            .iter()
            .filter(|(k, _)| {
                **k != crate::Symbol::new("sym_name") && **k != crate::Symbol::new("type")
            })
            .map(|(k, v)| (*k, v.clone()))
            .collect()
    };
    if !extra_attrs.is_empty() {
        write!(h, " attributes {{")?;
        for (i, (key, val)) in extra_attrs.iter().enumerate() {
            if i > 0 {
                write!(h, ", ")?;
            }
            write!(h, "{key} = ")?;
            h.write_attribute(val)?;
        }
        write!(h, "}}")?;
    }

    if let Some(region) = region {
        // Body
        writeln!(h, " {{")?;
        h.print_region_eliding_entry(region, indent + 2)?;
        writeln!(h, "{indent_str}}}")?;
    } else {
        writeln!(h)?;
    }

    Ok(())
}

/// Parse func.func custom format back into a RawOperation.
fn parse_func<'a>(
    input: &mut &'a str,
    results: Vec<&'a str>,
    sym_name: Option<String>,
) -> winnow::ModalResult<crate::parser::raw::RawOperation<'a>> {
    use crate::parser::raw::*;
    use winnow::combinator::opt;
    use winnow::prelude::*;

    // "(%arg: type, ...)" or "()"
    ws.parse_next(input)?;
    let params = if input.starts_with('(') {
        func_params.parse_next(input)?
    } else {
        vec![]
    };

    // "-> return_type" (optional)
    let ret_ty = opt(return_type).parse_next(input)?;

    // "attributes { key = value, ... }" (optional extra attributes)
    let attributes = opt((ws, "attributes", ws, raw_attr_dict))
        .parse_next(input)?
        .map(|(_, _, _, attrs)| attrs)
        .unwrap_or_default();

    // "{ body }"
    ws.parse_next(input)?;
    let mut regions = Vec::new();
    if input.starts_with('{') {
        let region = raw_region.parse_next(input)?;
        regions.push(region);
    }

    Ok(RawOperation {
        results,
        dialect: "func",
        op_name: "func",
        sym_name,
        func_params: params,
        return_type: ret_ty,
        operands: vec![],
        attributes,
        result_types: vec![],
        regions,
        successors: vec![],
    })
}

inventory::submit! {
    crate::op_interface::OpAsmFormat {
        dialect: "func",
        op_name: "func",
        print_fn: print_func,
        parse_fn: parse_func,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op_interface::IndirectCallLikeOps;
    use crate::ops::DialectOp;
    use crate::parser::parse_test_module;
    use crate::printer::print_module;

    #[test]
    fn indirect_signature_is_declared_and_round_trips() {
        let input = r#"core.module @test {
  func.func @run(%callee: core.func(core.i32, core.i32), %value: core.i32) -> core.i32 {
    %result = func.call_indirect %callee, %value {signature = core.func(core.i32, core.i32)} : core.i32
    func.return %result
  }
}"#;
        let mut ctx = crate::IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        let function = Func::from_op(&ctx, ctx.block(module.first_block(&ctx).unwrap()).ops[0])
            .expect("function");
        let body = function.body_if_present(&ctx).expect("function body");
        let call = CallIndirect::from_op(&ctx, ctx.block(ctx.region(body).blocks[0]).ops[0])
            .expect("indirect call");
        assert!(call.signature(&ctx).is_some(), "declared signature");

        let printed = print_module(&ctx, module.op());
        assert!(printed.contains("signature = core.func(core.i32, core.i32)"));
        assert!(!printed.contains("func.indirect_call_signature"));
    }

    #[test]
    fn indirect_call_signature_recognizes_only_indirect_calls() {
        let mut ctx = crate::IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @ordinary(%callee: core.func(core.i32, core.i32), %value: core.i32) -> core.i32 {
    %result = func.call_indirect %callee, %value {signature = core.func(core.i32, core.i32)} : core.i32
    func.return %result
  }
  func.func @tail(%callee: core.func(core.nil, core.i32), %value: core.i32) -> core.nil {
    func.tail_call_indirect %callee, %value {signature = core.func(core.nil, core.i32)}
  }
  func.func @direct() -> core.nil {
    func.return
  }
}"#,
        );

        let functions = module.ops(&ctx);
        let ordinary = Func::from_op(&ctx, functions[0]).expect("ordinary function");
        let tail = Func::from_op(&ctx, functions[1]).expect("tail function");
        let direct = Func::from_op(&ctx, functions[2]).expect("direct function");
        let ordinary_op = ctx
            .block(ctx.region(ordinary.body_if_present(&ctx).unwrap()).blocks[0])
            .ops[0];
        let tail_op = ctx
            .block(ctx.region(tail.body_if_present(&ctx).unwrap()).blocks[0])
            .ops[0];
        let direct_op = ctx
            .block(ctx.region(direct.body_if_present(&ctx).unwrap()).blocks[0])
            .ops[0];

        assert!(IndirectCallLikeOps::get(&ctx, ordinary_op).is_some());
        assert!(IndirectCallLikeOps::get(&ctx, tail_op).is_some());
        assert!(IndirectCallLikeOps::exact_signature(&ctx, ordinary_op).is_some());
        assert!(IndirectCallLikeOps::exact_signature(&ctx, tail_op).is_some());
        for op in [ordinary_op, tail_op] {
            let operands = ctx.op_operands(op);
            assert_eq!(IndirectCallLikeOps::callee(&ctx, op), Some(operands[0]));
            assert_eq!(
                IndirectCallLikeOps::arguments(&ctx, op),
                Some(&operands[1..])
            );
        }
        assert!(IndirectCallLikeOps::get(&ctx, direct_op).is_none());
        assert_eq!(IndirectCallLikeOps::callee(&ctx, direct_op), None);
        assert_eq!(IndirectCallLikeOps::arguments(&ctx, direct_op), None);
    }
}
