//! Function-level code generation for Cranelift backend.
//!
//! Translates `clif.*` dialect operations within a single function body
//! to Cranelift IR instructions using `FunctionBuilder`.

use std::collections::HashMap;

use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{self as cl_ir, InstBuilder, TrapCode};
use cranelift_codegen::isa::CallConv;
use cranelift_frontend::FunctionBuilder;
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::clif as arena_clif;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{BlockRef, OpRef, TypeRef, ValueRef};

use crate::{CompilationError, CompilationResult};

/// Parse a condition symbol into a Cranelift integer condition code.
fn parse_int_cc(sym: Symbol) -> CompilationResult<cl_ir::condcodes::IntCC> {
    use cl_ir::condcodes::IntCC;
    sym.with_str(|s| match s {
        "eq" => Ok(IntCC::Equal),
        "ne" => Ok(IntCC::NotEqual),
        "slt" => Ok(IntCC::SignedLessThan),
        "sle" => Ok(IntCC::SignedLessThanOrEqual),
        "sgt" => Ok(IntCC::SignedGreaterThan),
        "sge" => Ok(IntCC::SignedGreaterThanOrEqual),
        "ult" => Ok(IntCC::UnsignedLessThan),
        "ule" => Ok(IntCC::UnsignedLessThanOrEqual),
        "ugt" => Ok(IntCC::UnsignedGreaterThan),
        "uge" => Ok(IntCC::UnsignedGreaterThanOrEqual),
        other => Err(CompilationError::codegen(format!(
            "unknown integer comparison condition: {other}"
        ))),
    })
}

/// Parse a symbol into a Cranelift atomic RMW operation.
fn parse_atomic_rmw_op(sym: Symbol) -> CompilationResult<cl_ir::AtomicRmwOp> {
    sym.with_str(|s| match s {
        "add" => Ok(cl_ir::AtomicRmwOp::Add),
        "sub" => Ok(cl_ir::AtomicRmwOp::Sub),
        "and" => Ok(cl_ir::AtomicRmwOp::And),
        "or" => Ok(cl_ir::AtomicRmwOp::Or),
        "xor" => Ok(cl_ir::AtomicRmwOp::Xor),
        "nand" => Ok(cl_ir::AtomicRmwOp::Nand),
        "xchg" => Ok(cl_ir::AtomicRmwOp::Xchg),
        "umin" => Ok(cl_ir::AtomicRmwOp::Umin),
        "umax" => Ok(cl_ir::AtomicRmwOp::Umax),
        "smin" => Ok(cl_ir::AtomicRmwOp::Smin),
        "smax" => Ok(cl_ir::AtomicRmwOp::Smax),
        other => Err(CompilationError::codegen(format!(
            "unknown atomic RMW operation: {other}"
        ))),
    })
}

/// Parse a condition symbol into a Cranelift float condition code.
fn parse_float_cc(sym: Symbol) -> CompilationResult<cl_ir::condcodes::FloatCC> {
    use cl_ir::condcodes::FloatCC;
    sym.with_str(|s| match s {
        "eq" => Ok(FloatCC::Equal),
        "ne" => Ok(FloatCC::NotEqual),
        "lt" => Ok(FloatCC::LessThan),
        "le" => Ok(FloatCC::LessThanOrEqual),
        "gt" => Ok(FloatCC::GreaterThan),
        "ge" => Ok(FloatCC::GreaterThanOrEqual),
        other => Err(CompilationError::codegen(format!(
            "unknown float comparison condition: {other}"
        ))),
    })
}

/// Translate a TrunkIR type to a Cranelift IR type.
///
/// `ptr_ty` is the platform pointer type (e.g. I64 on 64-bit, I32 on 32-bit),
/// obtained from `target_config().pointer_type()`.
pub(crate) fn translate_type(
    ctx: &IrContext,
    ty: TypeRef,
    ptr_ty: cl_types::Type,
) -> CompilationResult<cl_types::Type> {
    let td = ctx.types.get(ty);
    let core_dialect = Symbol::new("core");
    if td.dialect == core_dialect {
        return td.name.with_str(|n| match n {
            "i1" => Ok(cl_types::I8),
            "i8" => Ok(cl_types::I8),
            "i16" => Ok(cl_types::I16),
            "i32" => Ok(cl_types::I32),
            "i64" => Ok(cl_types::I64),
            "f32" => Ok(cl_types::F32),
            "f64" => Ok(cl_types::F64),
            "ptr" => Ok(ptr_ty),
            other => Err(CompilationError::type_error(format!(
                "unsupported type for Cranelift: core.{other}"
            ))),
        });
    }
    Err(CompilationError::type_error(format!(
        "unsupported type for Cranelift: {}.{}",
        td.dialect, td.name,
    )))
}

/// Translate a TrunkIR `core.func` type to a Cranelift `Signature`.
///
/// `ptr_ty` is the platform pointer type, obtained from `target_config().pointer_type()`.
pub(crate) fn translate_signature(
    ctx: &IrContext,
    func_ty_ref: TypeRef,
    call_conv: CallConv,
    ptr_ty: cl_types::Type,
) -> CompilationResult<cl_ir::Signature> {
    let td = ctx.types.get(func_ty_ref);
    if td.dialect != Symbol::new("core") || td.name != Symbol::new("func") {
        return Err(CompilationError::type_error(
            "expected core.func type for signature translation",
        ));
    }

    let mut sig = cl_ir::Signature::new(call_conv);

    // core.func type layout: params[0] = return type, params[1..] = parameter types
    if td.params.is_empty() {
        return Err(CompilationError::type_error(
            "core.func type must have at least a return type param",
        ));
    }

    let ret_type = td.params[0];
    let param_types = &td.params[1..];

    for &param_ty in param_types {
        let cl_ty = translate_type(ctx, param_ty, ptr_ty)?;
        sig.params.push(cl_ir::AbiParam::new(cl_ty));
    }

    // Nil return type means void — no return values.
    let ret_td = ctx.types.get(ret_type);
    if !(ret_td.dialect == Symbol::new("core") && ret_td.name == Symbol::new("nil")) {
        let cl_ty = translate_type(ctx, ret_type, ptr_ty)?;
        sig.returns.push(cl_ir::AbiParam::new(cl_ty));
    }

    Ok(sig)
}

/// Translates `clif.*` operations within a single function body
/// to Cranelift IR instructions.
pub(crate) struct FunctionTranslator<'a> {
    ctx: &'a IrContext,
    pub(crate) builder: FunctionBuilder<'a>,
    /// Maps TrunkIR arena values to Cranelift IR values.
    pub(crate) values: HashMap<ValueRef, cl_ir::Value>,
    /// Maps function symbols to Cranelift FuncRefs.
    func_refs: &'a HashMap<Symbol, cl_ir::FuncRef>,
    /// Maps data symbols to Cranelift GlobalValues.
    data_refs: &'a HashMap<Symbol, cl_ir::GlobalValue>,
    /// Maps TrunkIR block refs to Cranelift blocks.
    pub(crate) block_map: HashMap<BlockRef, cl_ir::Block>,
    /// The ISA calling convention (used for indirect calls).
    call_conv: CallConv,
    /// The platform pointer type (e.g. I64 on 64-bit).
    ptr_ty: cl_types::Type,
}

impl<'a> FunctionTranslator<'a> {
    pub(crate) fn new(
        ctx: &'a IrContext,
        builder: FunctionBuilder<'a>,
        func_refs: &'a HashMap<Symbol, cl_ir::FuncRef>,
        data_refs: &'a HashMap<Symbol, cl_ir::GlobalValue>,
        call_conv: CallConv,
        ptr_ty: cl_types::Type,
    ) -> Self {
        Self {
            ctx,
            builder,
            values: HashMap::new(),
            func_refs,
            data_refs,
            block_map: HashMap::new(),
            call_conv,
            ptr_ty,
        }
    }

    fn lookup(&self, ir_val: ValueRef) -> CompilationResult<cl_ir::Value> {
        self.values.get(&ir_val).copied().ok_or_else(|| {
            let val_def = self.ctx.value_def(ir_val);
            let val_ty = self.ctx.value_ty(ir_val);
            let ty_data = self.ctx.types.get(val_ty);
            // Check if the defining op is in any of the blocks we know about
            let def_info = match val_def {
                trunk_ir::refs::ValueDef::OpResult(op, idx) => {
                    let op_data = self.ctx.op(op);
                    let parent_block = op_data.parent_block;
                    let in_block_map = parent_block.map(|b| self.block_map.contains_key(&b)).unwrap_or(false);
                    format!("op={:?} idx={} dialect={} name={} parent_block={:?} in_block_map={}",
                        op, idx, op_data.dialect, op_data.name, parent_block, in_block_map)
                }
                trunk_ir::refs::ValueDef::BlockArg(block, idx) => {
                    let in_block_map = self.block_map.contains_key(&block);
                    format!("block_arg block={:?} idx={} in_block_map={}", block, idx, in_block_map)
                }
            };
            CompilationError::codegen(format!(
                "TrunkIR value not found in Cranelift mapping (mapped {} values total, type: {}.{}, def: {})",
                self.values.len(),
                ty_data.dialect,
                ty_data.name,
                def_info,
            ))
        })
    }

    /// Check if a value has `core.nil` type.
    fn is_nil_typed(&self, val: ValueRef) -> bool {
        let ty = self.ctx.value_ty(val);
        let td = self.ctx.types.get(ty);
        td.dialect == Symbol::new("core") && td.name == Symbol::new("nil")
    }

    pub(crate) fn lookup_block(&self, ir_block: BlockRef) -> CompilationResult<cl_ir::Block> {
        self.block_map.get(&ir_block).copied().ok_or_else(|| {
            CompilationError::codegen("TrunkIR block not found in Cranelift block mapping")
        })
    }

    /// Translate a single `clif.*` arena operation to Cranelift IR.
    pub(crate) fn translate_op(&mut self, op: OpRef) -> CompilationResult<()> {
        let ctx = self.ctx;

        // === Constants ===
        if let Ok(c) = arena_clif::Iconst::from_op(ctx, op) {
            let result_ty = ctx.op_result_types(op)[0];
            // Nil constants have no runtime representation — skip emission.
            let td = ctx.types.get(result_ty);
            if td.dialect == Symbol::new("core") && td.name == Symbol::new("nil") {
                return Ok(());
            }
            let ty = translate_type(ctx, result_ty, self.ptr_ty)?;
            let val = self.builder.ins().iconst(ty, c.value(ctx));
            let result = ctx.op_result(op, 0);
            self.values.insert(result, val);
            return Ok(());
        }
        if let Ok(c) = arena_clif::F32const::from_op(ctx, op) {
            let val = self.builder.ins().f32const(c.value(ctx));
            let result = ctx.op_result(op, 0);
            self.values.insert(result, val);
            return Ok(());
        }
        if let Ok(c) = arena_clif::F64const::from_op(ctx, op) {
            let val = self.builder.ins().f64const(c.value(ctx));
            let result = ctx.op_result(op, 0);
            self.values.insert(result, val);
            return Ok(());
        }

        // === Integer Arithmetic ===
        if arena_clif::Iadd::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary(ops[0], ops[1], op, |b, a, c| b.ins().iadd(a, c));
        }
        if arena_clif::Isub::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary(ops[0], ops[1], op, |b, a, c| b.ins().isub(a, c));
        }
        if arena_clif::Imul::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary(ops[0], ops[1], op, |b, a, c| b.ins().imul(a, c));
        }
        if arena_clif::Sdiv::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary(ops[0], ops[1], op, |b, a, c| b.ins().sdiv(a, c));
        }
        if arena_clif::Udiv::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary(ops[0], ops[1], op, |b, a, c| b.ins().udiv(a, c));
        }
        if arena_clif::Srem::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary(ops[0], ops[1], op, |b, a, c| b.ins().srem(a, c));
        }
        if arena_clif::Urem::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary(ops[0], ops[1], op, |b, a, c| b.ins().urem(a, c));
        }
        if arena_clif::Ineg::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_unary(ops[0], op, |b, v| b.ins().ineg(v));
        }

        // === Floating Point Arithmetic ===
        if arena_clif::Fadd::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary(ops[0], ops[1], op, |b, a, c| b.ins().fadd(a, c));
        }
        if arena_clif::Fsub::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary(ops[0], ops[1], op, |b, a, c| b.ins().fsub(a, c));
        }
        if arena_clif::Fmul::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary(ops[0], ops[1], op, |b, a, c| b.ins().fmul(a, c));
        }
        if arena_clif::Fdiv::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary(ops[0], ops[1], op, |b, a, c| b.ins().fdiv(a, c));
        }
        if arena_clif::Fneg::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_unary(ops[0], op, |b, v| b.ins().fneg(v));
        }

        // === Call ===
        if let Ok(call) = arena_clif::Call::from_op(ctx, op) {
            let callee_sym = call.callee(ctx);
            let func_ref = self
                .func_refs
                .get(&callee_sym)
                .copied()
                .ok_or_else(|| CompilationError::function_not_found(&callee_sym.to_string()))?;

            let operands = ctx.op_operands(op);
            let args: Vec<cl_ir::Value> = operands
                .iter()
                .map(|v| self.lookup(*v))
                .collect::<CompilationResult<_>>()?;

            let inst = self.builder.ins().call(func_ref, &args);
            let results = self.builder.inst_results(inst);
            if !results.is_empty() {
                let ir_result = ctx.op_result(op, 0);
                self.values.insert(ir_result, results[0]);
            }
            return Ok(());
        }

        // === Return ===
        if arena_clif::Return::from_op(ctx, op).is_ok() {
            let operands = ctx.op_operands(op);
            let mut vals = Vec::new();
            for &v in operands {
                if let Some(&cl_val) = self.values.get(&v) {
                    vals.push(cl_val);
                } else if !self.is_nil_typed(v) {
                    return Err(CompilationError::codegen(
                        "return operand has no Cranelift value mapping and is not Nil-typed",
                    ));
                }
            }
            self.builder.ins().return_(&vals);
            return Ok(());
        }

        // === Control Flow ===
        if arena_clif::Jump::from_op(ctx, op).is_ok() {
            let op_data = ctx.op(op);
            let ir_dest = op_data.successors[0];
            let cl_dest = self.lookup_block(ir_dest)?;
            let operands = ctx.op_operands(op);
            let args: Vec<cl_ir::BlockArg> = operands
                .iter()
                .map(|v| self.lookup(*v).map(cl_ir::BlockArg::from))
                .collect::<CompilationResult<_>>()?;
            let dest_param_count = self.builder.block_params(cl_dest).len();
            if args.len() != dest_param_count {
                return Err(CompilationError::codegen(format!(
                    "clif.jump: argument count ({}) does not match destination block parameter count ({})",
                    args.len(),
                    dest_param_count,
                )));
            }
            self.builder.ins().jump(cl_dest, &args);
            return Ok(());
        }
        if arena_clif::Brif::from_op(ctx, op).is_ok() {
            let operands = ctx.op_operands(op);
            let cond = self.lookup(operands[0])?;
            let op_data = ctx.op(op);
            let cl_then = self.lookup_block(op_data.successors[0])?;
            let cl_else = self.lookup_block(op_data.successors[1])?;
            self.builder.ins().brif(cond, cl_then, &[], cl_else, &[]);
            return Ok(());
        }

        // === Comparisons ===
        if let Ok(o) = arena_clif::Icmp::from_op(ctx, op) {
            let cond = parse_int_cc(o.cond(ctx))?;
            let operands = ctx.op_operands(op);
            let lhs = self.lookup(operands[0])?;
            let rhs = self.lookup(operands[1])?;
            let val = self.builder.ins().icmp(cond, lhs, rhs);
            let result = ctx.op_result(op, 0);
            self.values.insert(result, val);
            return Ok(());
        }
        if let Ok(o) = arena_clif::Fcmp::from_op(ctx, op) {
            let cond = parse_float_cc(o.cond(ctx))?;
            let operands = ctx.op_operands(op);
            let lhs = self.lookup(operands[0])?;
            let rhs = self.lookup(operands[1])?;
            let val = self.builder.ins().fcmp(cond, lhs, rhs);
            let result = ctx.op_result(op, 0);
            self.values.insert(result, val);
            return Ok(());
        }

        // === Memory ===
        if let Ok(load) = arena_clif::Load::from_op(ctx, op) {
            let result_ty = ctx.op_result_types(op)[0];
            let td = ctx.types.get(result_ty);
            if td.dialect == Symbol::new("core") && td.name == Symbol::new("nil") {
                return Ok(());
            }
            let operands = ctx.op_operands(op);
            let addr = self.lookup(operands[0])?;
            let ty = translate_type(ctx, result_ty, self.ptr_ty)?;
            let val = self
                .builder
                .ins()
                .load(ty, cl_ir::MemFlags::new(), addr, load.offset(ctx));
            let result = ctx.op_result(op, 0);
            self.values.insert(result, val);
            return Ok(());
        }
        if let Ok(store) = arena_clif::Store::from_op(ctx, op) {
            let operands = ctx.op_operands(op);
            if self.is_nil_typed(operands[0]) {
                return Ok(());
            }
            let value = self.lookup(operands[0])?;
            let addr = self.lookup(operands[1])?;
            self.builder
                .ins()
                .store(cl_ir::MemFlags::new(), value, addr, store.offset(ctx));
            return Ok(());
        }

        if let Ok(armw) = arena_clif::AtomicRmw::from_op(ctx, op) {
            let result_ty = ctx.op_result_types(op)[0];
            let ty = translate_type(ctx, result_ty, self.ptr_ty)?;
            let operands = ctx.op_operands(op);
            let mut addr = self.lookup(operands[0])?;
            let value = self.lookup(operands[1])?;
            let offset = armw.offset(ctx);
            if offset != 0 {
                addr = self.builder.ins().iadd_imm(addr, i64::from(offset));
            }
            let rmw_op = parse_atomic_rmw_op(armw.op(ctx))?;
            let val =
                self.builder
                    .ins()
                    .atomic_rmw(ty, cl_ir::MemFlags::new(), rmw_op, addr, value);
            let result = ctx.op_result(op, 0);
            self.values.insert(result, val);
            return Ok(());
        }

        // === Symbol Address ===
        if let Ok(sym_addr) = arena_clif::SymbolAddr::from_op(ctx, op) {
            let sym = sym_addr.sym(ctx);
            // Check function refs first, then data refs
            let val = if let Some(&func_ref) = self.func_refs.get(&sym) {
                self.builder.ins().func_addr(self.ptr_ty, func_ref)
            } else if let Some(&gv) = self.data_refs.get(&sym) {
                self.builder.ins().global_value(self.ptr_ty, gv)
            } else {
                return Err(CompilationError::codegen(format!(
                    "symbol not found in function or data refs: {}",
                    sym
                )));
            };
            let result = ctx.op_result(op, 0);
            self.values.insert(result, val);
            return Ok(());
        }

        // === Trap ===
        if arena_clif::Trap::from_op(ctx, op).is_ok() {
            self.builder.ins().trap(TrapCode::unwrap_user(1));
            return Ok(());
        }

        // === Return Call (tail call) ===
        if let Ok(rc) = arena_clif::ReturnCall::from_op(ctx, op) {
            let callee_sym = rc.callee(ctx);
            let func_ref = self
                .func_refs
                .get(&callee_sym)
                .copied()
                .ok_or_else(|| CompilationError::function_not_found(&callee_sym.to_string()))?;

            let operands = ctx.op_operands(op);
            let args: Vec<cl_ir::Value> = operands
                .iter()
                .map(|v| self.lookup(*v))
                .collect::<CompilationResult<_>>()?;

            self.builder.ins().return_call(func_ref, &args);
            return Ok(());
        }

        // === Return Call Indirect (indirect tail call) ===
        if let Ok(rci) = arena_clif::ReturnCallIndirect::from_op(ctx, op) {
            let operands = ctx.op_operands(op);
            let callee = self.lookup(operands[0])?;
            let args: Vec<cl_ir::Value> = operands[1..]
                .iter()
                .map(|v| self.lookup(*v))
                .collect::<CompilationResult<_>>()?;

            let sig_ty = rci.sig(ctx);
            let sig = translate_signature(ctx, sig_ty, self.call_conv, self.ptr_ty)?;
            let sig_ref = self.builder.import_signature(sig);

            self.builder
                .ins()
                .return_call_indirect(sig_ref, callee, &args);
            return Ok(());
        }

        // === Indirect Call ===
        if let Ok(call_ind) = arena_clif::CallIndirect::from_op(ctx, op) {
            let operands = ctx.op_operands(op);
            let callee = self.lookup(operands[0])?;
            let args: Vec<cl_ir::Value> = operands[1..]
                .iter()
                .map(|v| self.lookup(*v))
                .collect::<CompilationResult<_>>()?;

            let sig_ty = call_ind.sig(ctx);
            let sig = translate_signature(ctx, sig_ty, self.call_conv, self.ptr_ty)?;
            let sig_ref = self.builder.import_signature(sig);

            let inst = self.builder.ins().call_indirect(sig_ref, callee, &args);
            let results = self.builder.inst_results(inst);
            if !results.is_empty() {
                let ir_result = ctx.op_result(op, 0);
                self.values.insert(ir_result, results[0]);
            }
            return Ok(());
        }

        // === Type Conversions ===
        if arena_clif::Ireduce::from_op(ctx, op).is_ok() {
            let operands = ctx.op_operands(op);
            let result_ty = ctx.op_result_types(op)[0];
            let ty = translate_type(ctx, result_ty, self.ptr_ty)?;
            let val = self.lookup(operands[0])?;
            let cl_val = self.builder.ins().ireduce(ty, val);
            self.values.insert(ctx.op_result(op, 0), cl_val);
            return Ok(());
        }
        if arena_clif::Uextend::from_op(ctx, op).is_ok() {
            let operands = ctx.op_operands(op);
            let result_ty = ctx.op_result_types(op)[0];
            let ty = translate_type(ctx, result_ty, self.ptr_ty)?;
            let val = self.lookup(operands[0])?;
            let cl_val = self.builder.ins().uextend(ty, val);
            self.values.insert(ctx.op_result(op, 0), cl_val);
            return Ok(());
        }
        if arena_clif::Sextend::from_op(ctx, op).is_ok() {
            let operands = ctx.op_operands(op);
            let result_ty = ctx.op_result_types(op)[0];
            let ty = translate_type(ctx, result_ty, self.ptr_ty)?;
            let val = self.lookup(operands[0])?;
            let cl_val = self.builder.ins().sextend(ty, val);
            self.values.insert(ctx.op_result(op, 0), cl_val);
            return Ok(());
        }
        if arena_clif::Fpromote::from_op(ctx, op).is_ok() {
            let operands = ctx.op_operands(op);
            let result_ty = ctx.op_result_types(op)[0];
            let ty = translate_type(ctx, result_ty, self.ptr_ty)?;
            let val = self.lookup(operands[0])?;
            let cl_val = self.builder.ins().fpromote(ty, val);
            self.values.insert(ctx.op_result(op, 0), cl_val);
            return Ok(());
        }
        if arena_clif::Fdemote::from_op(ctx, op).is_ok() {
            let operands = ctx.op_operands(op);
            let result_ty = ctx.op_result_types(op)[0];
            let ty = translate_type(ctx, result_ty, self.ptr_ty)?;
            let val = self.lookup(operands[0])?;
            let cl_val = self.builder.ins().fdemote(ty, val);
            self.values.insert(ctx.op_result(op, 0), cl_val);
            return Ok(());
        }
        if arena_clif::FcvtToSint::from_op(ctx, op).is_ok() {
            let operands = ctx.op_operands(op);
            let result_ty = ctx.op_result_types(op)[0];
            let ty = translate_type(ctx, result_ty, self.ptr_ty)?;
            let val = self.lookup(operands[0])?;
            let cl_val = self.builder.ins().fcvt_to_sint(ty, val);
            self.values.insert(ctx.op_result(op, 0), cl_val);
            return Ok(());
        }
        if arena_clif::FcvtFromSint::from_op(ctx, op).is_ok() {
            let operands = ctx.op_operands(op);
            let result_ty = ctx.op_result_types(op)[0];
            let ty = translate_type(ctx, result_ty, self.ptr_ty)?;
            let val = self.lookup(operands[0])?;
            let cl_val = self.builder.ins().fcvt_from_sint(ty, val);
            self.values.insert(ctx.op_result(op, 0), cl_val);
            return Ok(());
        }
        if arena_clif::FcvtToUint::from_op(ctx, op).is_ok() {
            let operands = ctx.op_operands(op);
            let result_ty = ctx.op_result_types(op)[0];
            let ty = translate_type(ctx, result_ty, self.ptr_ty)?;
            let val = self.lookup(operands[0])?;
            let cl_val = self.builder.ins().fcvt_to_uint(ty, val);
            self.values.insert(ctx.op_result(op, 0), cl_val);
            return Ok(());
        }
        if arena_clif::FcvtFromUint::from_op(ctx, op).is_ok() {
            let operands = ctx.op_operands(op);
            let result_ty = ctx.op_result_types(op)[0];
            let ty = translate_type(ctx, result_ty, self.ptr_ty)?;
            let val = self.lookup(operands[0])?;
            let cl_val = self.builder.ins().fcvt_from_uint(ty, val);
            self.values.insert(ctx.op_result(op, 0), cl_val);
            return Ok(());
        }

        // === Bitwise Operations ===
        if arena_clif::Band::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary(ops[0], ops[1], op, |b, a, c| b.ins().band(a, c));
        }
        if arena_clif::Bor::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary(ops[0], ops[1], op, |b, a, c| b.ins().bor(a, c));
        }
        if arena_clif::Bxor::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary(ops[0], ops[1], op, |b, a, c| b.ins().bxor(a, c));
        }
        if arena_clif::Ishl::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary(ops[0], ops[1], op, |b, a, c| b.ins().ishl(a, c));
        }
        if arena_clif::Sshr::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary(ops[0], ops[1], op, |b, a, c| b.ins().sshr(a, c));
        }
        if arena_clif::Ushr::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary(ops[0], ops[1], op, |b, a, c| b.ins().ushr(a, c));
        }

        let op_data = ctx.op(op);
        Err(CompilationError::codegen(format!(
            "unsupported operation: {}.{}",
            op_data.dialect, op_data.name,
        )))
    }

    fn emit_binary(
        &mut self,
        lhs_ir: ValueRef,
        rhs_ir: ValueRef,
        op: OpRef,
        f: impl FnOnce(&mut FunctionBuilder<'a>, cl_ir::Value, cl_ir::Value) -> cl_ir::Value,
    ) -> CompilationResult<()> {
        let lhs = self.lookup(lhs_ir)?;
        let rhs = self.lookup(rhs_ir)?;
        let cl_val = f(&mut self.builder, lhs, rhs);
        let result = self.ctx.op_result(op, 0);
        self.values.insert(result, cl_val);
        Ok(())
    }

    fn emit_unary(
        &mut self,
        operand_ir: ValueRef,
        op: OpRef,
        f: impl FnOnce(&mut FunctionBuilder<'a>, cl_ir::Value) -> cl_ir::Value,
    ) -> CompilationResult<()> {
        let operand = self.lookup(operand_ir)?;
        let cl_val = f(&mut self.builder, operand);
        let result = self.ctx.op_result(op, 0);
        self.values.insert(result, cl_val);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::context::IrContext;
    use trunk_ir::types::TypeData;

    fn make_core_type(ctx: &mut IrContext, name: &'static str) -> TypeRef {
        ctx.types.intern(TypeData {
            dialect: Symbol::new("core"),
            name: Symbol::new(name),
            params: Default::default(),
            attrs: Default::default(),
        })
    }

    #[test]
    fn test_translate_type_integers() {
        let mut ctx = IrContext::new();
        let i8_ty = make_core_type(&mut ctx, "i8");
        let i16_ty = make_core_type(&mut ctx, "i16");
        let i32_ty = make_core_type(&mut ctx, "i32");
        let i64_ty = make_core_type(&mut ctx, "i64");

        assert_eq!(
            translate_type(&ctx, i8_ty, cl_types::I64).unwrap(),
            cl_types::I8
        );
        assert_eq!(
            translate_type(&ctx, i16_ty, cl_types::I64).unwrap(),
            cl_types::I16
        );
        assert_eq!(
            translate_type(&ctx, i32_ty, cl_types::I64).unwrap(),
            cl_types::I32
        );
        assert_eq!(
            translate_type(&ctx, i64_ty, cl_types::I64).unwrap(),
            cl_types::I64
        );
    }

    #[test]
    fn test_translate_type_floats() {
        let mut ctx = IrContext::new();
        let f32_ty = make_core_type(&mut ctx, "f32");
        let f64_ty = make_core_type(&mut ctx, "f64");

        assert_eq!(
            translate_type(&ctx, f32_ty, cl_types::I64).unwrap(),
            cl_types::F32
        );
        assert_eq!(
            translate_type(&ctx, f64_ty, cl_types::I64).unwrap(),
            cl_types::F64
        );
    }

    #[test]
    fn test_translate_type_unsupported() {
        let mut ctx = IrContext::new();
        let nil_ty = make_core_type(&mut ctx, "nil");
        assert!(translate_type(&ctx, nil_ty, cl_types::I64).is_err());
    }

    #[test]
    fn test_translate_signature_params_and_return() {
        let mut ctx = IrContext::new();
        let i32_ty = make_core_type(&mut ctx, "i32");
        let i64_ty = make_core_type(&mut ctx, "i64");

        // core.func type layout: params[0] = return type, params[1..] = parameter types
        let func_ty = ctx.types.intern(TypeData {
            dialect: Symbol::new("core"),
            name: Symbol::new("func"),
            params: trunk_ir::smallvec::smallvec![i64_ty, i32_ty, i32_ty],
            attrs: Default::default(),
        });

        let sig = translate_signature(&ctx, func_ty, CallConv::SystemV, cl_types::I64).unwrap();
        assert_eq!(sig.params.len(), 2);
        assert_eq!(sig.params[0].value_type, cl_types::I32);
        assert_eq!(sig.params[1].value_type, cl_types::I32);
        assert_eq!(sig.returns.len(), 1);
        assert_eq!(sig.returns[0].value_type, cl_types::I64);
    }

    #[test]
    fn test_translate_signature_void_return() {
        let mut ctx = IrContext::new();
        let i64_ty = make_core_type(&mut ctx, "i64");
        let nil_ty = make_core_type(&mut ctx, "nil");

        let func_ty = ctx.types.intern(TypeData {
            dialect: Symbol::new("core"),
            name: Symbol::new("func"),
            params: trunk_ir::smallvec::smallvec![nil_ty, i64_ty],
            attrs: Default::default(),
        });

        let sig = translate_signature(&ctx, func_ty, CallConv::SystemV, cl_types::I64).unwrap();
        assert_eq!(sig.params.len(), 1);
        assert_eq!(sig.params[0].value_type, cl_types::I64);
        assert_eq!(sig.returns.len(), 0);
    }

    #[test]
    fn test_translate_signature_no_params() {
        let mut ctx = IrContext::new();
        let i64_ty = make_core_type(&mut ctx, "i64");

        let func_ty = ctx.types.intern(TypeData {
            dialect: Symbol::new("core"),
            name: Symbol::new("func"),
            params: trunk_ir::smallvec::smallvec![i64_ty],
            attrs: Default::default(),
        });

        let sig = translate_signature(&ctx, func_ty, CallConv::SystemV, cl_types::I64).unwrap();
        assert_eq!(sig.params.len(), 0);
        assert_eq!(sig.returns.len(), 1);
        assert_eq!(sig.returns[0].value_type, cl_types::I64);
    }
}
