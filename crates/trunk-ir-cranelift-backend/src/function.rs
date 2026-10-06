//! Function-level code generation for Cranelift backend.
//!
//! Translates `clif.*` dialect operations within a single function body
//! to Cranelift IR instructions using `FunctionBuilder`.

use rustc_hash::FxHashMap as HashMap;

use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{self as cl_ir, InstBuilder, TrapCode};
use cranelift_codegen::isa::CallConv;
use cranelift_frontend::FunctionBuilder;
use cranelift_module::{DataId, FuncId, Module as _};
use cranelift_object::ObjectModule;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::{clif, func};
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{BlockRef, OpRef, TypeRef, ValueRef};
use trunk_ir::{Symbol, SymbolPath};

use crate::{CompilationError, CompilationResult};

pub(crate) fn is_nil_type(ctx: &IrContext, ty: TypeRef) -> bool {
    let ty = ctx.get_type(ty);
    ty.dialect == Symbol::new("core") && ty.name == Symbol::new("nil") && ty.params.is_empty()
}

/// Parse a condition symbol into a Cranelift integer condition code.
fn parse_int_cc(s: &str) -> CompilationResult<cl_ir::condcodes::IntCC> {
    use cl_ir::condcodes::IntCC;
    match s {
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
    }
}

/// Parse a symbol into a Cranelift atomic RMW operation.
fn parse_atomic_rmw_op(s: &str) -> CompilationResult<cl_ir::AtomicRmwOp> {
    match s {
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
    }
}

/// Parse a condition symbol into a Cranelift float condition code.
fn parse_float_cc(s: &str) -> CompilationResult<cl_ir::condcodes::FloatCC> {
    use cl_ir::condcodes::FloatCC;
    match s {
        "eq" => Ok(FloatCC::Equal),
        "ne" => Ok(FloatCC::NotEqual),
        "lt" => Ok(FloatCC::LessThan),
        "le" => Ok(FloatCC::LessThanOrEqual),
        "gt" => Ok(FloatCC::GreaterThan),
        "ge" => Ok(FloatCC::GreaterThanOrEqual),
        other => Err(CompilationError::codegen(format!(
            "unknown float comparison condition: {other}"
        ))),
    }
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
    let td = ctx.get_type(ty);
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

/// Translate a target-owned `clif.func_sig` type to a Cranelift `Signature`.
///
/// The calling convention comes from the signature's `call_conv`; an absent
/// attribute selects `platform_call_conv`. `ptr_ty` is the platform pointer
/// type, obtained from `target_config().pointer_type()`.
pub(crate) fn translate_signature(
    ctx: &IrContext,
    func_ty_ref: TypeRef,
    platform_call_conv: CallConv,
    ptr_ty: cl_types::Type,
) -> CompilationResult<cl_ir::Signature> {
    let function = clif::FuncSig::from_type_ref(ctx, func_ty_ref).ok_or_else(|| {
        CompilationError::type_error("expected valid clif.func_sig type for signature translation")
    })?;
    let call_conv = match function.call_conv(ctx) {
        Some(func::CallConv::Platform) => platform_call_conv,
        Some(func::CallConv::Tail) => CallConv::Tail,
        None => {
            return Err(CompilationError::type_error(
                "clif.func_sig has a malformed call_conv attribute",
            ));
        }
    };

    let mut sig = cl_ir::Signature::new(call_conv);

    let param_types = function.inputs(ctx);

    for &param_ty in param_types {
        if is_nil_type(ctx, param_ty) {
            continue;
        }
        let cl_ty = translate_type(ctx, param_ty, ptr_ty)?;
        sig.params.push(cl_ir::AbiParam::new(cl_ty));
    }

    // Nil results are zero-width; preserve every non-nil result in order.
    for &result_ty in function.results(ctx) {
        if is_nil_type(ctx, result_ty) {
            continue;
        }
        let cl_ty = translate_type(ctx, result_ty, ptr_ty)?;
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
    /// The object module that owns the function and data declarations.
    module: &'a mut ObjectModule,
    /// Module-level functions a body may reference.
    func_ids: &'a HashMap<SymbolPath, FuncId>,
    /// Module-level data objects a body may reference.
    data_ids: &'a HashMap<SymbolPath, DataId>,
    /// Functions this body referenced, declared on first reference.
    func_refs: HashMap<SymbolPath, cl_ir::FuncRef>,
    /// Data objects this body referenced, declared on first reference.
    data_refs: HashMap<SymbolPath, cl_ir::GlobalValue>,
    /// Maps TrunkIR block refs to Cranelift blocks.
    pub(crate) block_map: HashMap<BlockRef, cl_ir::Block>,
    /// The platform's ordinary calling convention for non-CPS indirect calls.
    default_call_conv: CallConv,
    /// The platform pointer type (e.g. I64 on 64-bit).
    ptr_ty: cl_types::Type,
}

impl<'a> FunctionTranslator<'a> {
    pub(crate) fn new(
        ctx: &'a IrContext,
        builder: FunctionBuilder<'a>,
        module: &'a mut ObjectModule,
        func_ids: &'a HashMap<SymbolPath, FuncId>,
        data_ids: &'a HashMap<SymbolPath, DataId>,
        default_call_conv: CallConv,
        ptr_ty: cl_types::Type,
    ) -> Self {
        Self {
            ctx,
            builder,
            values: HashMap::default(),
            module,
            func_ids,
            data_ids,
            func_refs: HashMap::default(),
            data_refs: HashMap::default(),
            block_map: HashMap::default(),
            default_call_conv,
            ptr_ty,
        }
    }

    /// The reference to a module function, declared in this function the
    /// first time the body references it.
    fn func_ref(&mut self, sym: SymbolPath) -> Option<cl_ir::FuncRef> {
        if let Some(&func_ref) = self.func_refs.get(&sym) {
            return Some(func_ref);
        }
        let func_id = *self.func_ids.get(&sym)?;
        let func_ref = self.module.declare_func_in_func(func_id, self.builder.func);
        self.func_refs.insert(sym, func_ref);
        Some(func_ref)
    }

    /// The reference to a module data object, declared in this function the
    /// first time the body references it.
    fn data_ref(&mut self, sym: SymbolPath) -> Option<cl_ir::GlobalValue> {
        if let Some(&gv) = self.data_refs.get(&sym) {
            return Some(gv);
        }
        let data_id = *self.data_ids.get(&sym)?;
        let gv = self.module.declare_data_in_func(data_id, self.builder.func);
        self.data_refs.insert(sym, gv);
        Some(gv)
    }

    fn lookup(&self, ir_val: ValueRef) -> CompilationResult<cl_ir::Value> {
        self.values.get(&ir_val).copied().ok_or_else(|| {
            let val_def = self.ctx.value_def(ir_val);
            let val_ty = self.ctx.value_ty(ir_val);
            let ty_data = self.ctx.get_type(val_ty);
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
        is_nil_type(self.ctx, self.ctx.value_ty(val))
    }

    fn map_call_results(&mut self, op: OpRef, results: &[cl_ir::Value]) -> CompilationResult<()> {
        let mut runtime_results = results.iter().copied();
        for index in 0..self.ctx.op_result_types(op).len() {
            let ir_result = self.ctx.op_result(op, index as u32);
            if self.is_nil_typed(ir_result) {
                continue;
            }
            let Some(result) = runtime_results.next() else {
                return Err(CompilationError::codegen(
                    "call result count does not match non-Nil TrunkIR results",
                ));
            };
            self.values.insert(ir_result, result);
        }
        if runtime_results.next().is_some() {
            return Err(CompilationError::codegen(
                "call produced more Cranelift results than TrunkIR declares",
            ));
        }
        Ok(())
    }

    fn runtime_values(&self, values: &[ValueRef]) -> CompilationResult<Vec<cl_ir::Value>> {
        values
            .iter()
            .filter(|&&value| !self.is_nil_typed(value))
            .map(|&value| self.lookup(value))
            .collect()
    }

    /// The jump table entry for a `clif.br_table` successor, which receives
    /// no arguments.
    fn jump_table_entry(&mut self, ir_block: BlockRef) -> CompilationResult<cl_ir::BlockCall> {
        let block = self.lookup_block(ir_block)?;
        if !self.builder.block_params(block).is_empty() {
            return Err(CompilationError::codegen(
                "clif.br_table: a successor block has parameters",
            ));
        }
        Ok(self.builder.func.dfg.block_call(block, &[]))
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
        if let Ok(c) = clif::Iconst::from_op(ctx, op) {
            let result_ty = ctx.op_result_types(op)[0];
            // Nil constants have no runtime representation — skip emission.
            let td = ctx.get_type(result_ty);
            if td.dialect == Symbol::new("core") && td.name == Symbol::new("nil") {
                return Ok(());
            }
            let ty = translate_type(ctx, result_ty, self.ptr_ty)?;
            let val = self.builder.ins().iconst(ty, c.value(ctx));
            let result = ctx.op_result(op, 0);
            self.values.insert(result, val);
            return Ok(());
        }
        if let Ok(c) = clif::F32const::from_op(ctx, op) {
            let val = self.builder.ins().f32const(c.value(ctx));
            let result = ctx.op_result(op, 0);
            self.values.insert(result, val);
            return Ok(());
        }
        if let Ok(c) = clif::F64const::from_op(ctx, op) {
            let val = self.builder.ins().f64const(c.value(ctx));
            let result = ctx.op_result(op, 0);
            self.values.insert(result, val);
            return Ok(());
        }

        // === Integer Arithmetic ===
        if clif::Iadd::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary(ops[0], ops[1], op, |b, a, c| b.ins().iadd(a, c));
        }
        if clif::Isub::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary(ops[0], ops[1], op, |b, a, c| b.ins().isub(a, c));
        }
        if clif::Imul::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary(ops[0], ops[1], op, |b, a, c| b.ins().imul(a, c));
        }
        if clif::Sdiv::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary(ops[0], ops[1], op, |b, a, c| b.ins().sdiv(a, c));
        }
        if clif::Udiv::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary(ops[0], ops[1], op, |b, a, c| b.ins().udiv(a, c));
        }
        if clif::Srem::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary(ops[0], ops[1], op, |b, a, c| b.ins().srem(a, c));
        }
        if clif::Urem::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary(ops[0], ops[1], op, |b, a, c| b.ins().urem(a, c));
        }
        if clif::Ineg::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_unary(ops[0], op, |b, v| b.ins().ineg(v));
        }

        // === Floating Point Arithmetic ===
        if clif::Fadd::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary(ops[0], ops[1], op, |b, a, c| b.ins().fadd(a, c));
        }
        if clif::Fsub::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary(ops[0], ops[1], op, |b, a, c| b.ins().fsub(a, c));
        }
        if clif::Fmul::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary(ops[0], ops[1], op, |b, a, c| b.ins().fmul(a, c));
        }
        if clif::Fdiv::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary(ops[0], ops[1], op, |b, a, c| b.ins().fdiv(a, c));
        }
        if clif::Fneg::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_unary(ops[0], op, |b, v| b.ins().fneg(v));
        }

        // === Call ===
        if let Ok(call) = clif::Call::from_op(ctx, op) {
            let callee_sym = call.callee(ctx);
            let func_ref = self
                .func_ref(callee_sym.clone())
                .ok_or_else(|| CompilationError::function_not_found(&callee_sym.to_string()))?;

            let operands = ctx.op_operands(op);
            let args = self.runtime_values(operands)?;

            let inst = self.builder.ins().call(func_ref, &args);
            let results = self.builder.inst_results(inst).to_vec();
            return self.map_call_results(op, &results);
        }

        // === Return ===
        if clif::Return::from_op(ctx, op).is_ok() {
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
        if clif::Jump::from_op(ctx, op).is_ok() {
            let ir_dest = clif::Jump::from_op(ctx, op).unwrap().dest(ctx);
            let cl_dest = self.lookup_block(ir_dest)?;
            let operands = ctx.op_operands(op);
            let args: Vec<cl_ir::BlockArg> = self
                .runtime_values(operands)?
                .into_iter()
                .map(cl_ir::BlockArg::from)
                .collect();
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
        if clif::Brif::from_op(ctx, op).is_ok() {
            let operands = ctx.op_operands(op);
            let cond = self.lookup(operands[0])?;
            let brif = clif::Brif::from_op(ctx, op).unwrap();
            let cl_then = self.lookup_block(brif.then_dest(ctx))?;
            let cl_else = self.lookup_block(brif.else_dest(ctx))?;
            self.builder.ins().brif(cond, cl_then, &[], cl_else, &[]);
            return Ok(());
        }

        if let Ok(br_table) = clif::BrTable::from_op(ctx, op) {
            let index = self.lookup(br_table.index(ctx))?;
            let index_ty = self.builder.func.dfg.value_type(index);
            if index_ty != cl_types::I32 {
                return Err(CompilationError::codegen(format!(
                    "clif.br_table: index has type {index_ty}, not i32"
                )));
            }
            let default = self.jump_table_entry(br_table.default(ctx))?;
            let table = br_table
                .table(ctx)
                .map(|block| self.jump_table_entry(block))
                .collect::<CompilationResult<Vec<_>>>()?;
            let jump_table = self
                .builder
                .create_jump_table(cl_ir::JumpTableData::new(default, &table));
            self.builder.ins().br_table(index, jump_table);
            return Ok(());
        }

        // === Comparisons ===
        if let Ok(o) = clif::Icmp::from_op(ctx, op) {
            let cond = parse_int_cc(o.cond(ctx))?;
            let operands = ctx.op_operands(op);
            let lhs = self.lookup(operands[0])?;
            let rhs = self.lookup(operands[1])?;
            let val = self.builder.ins().icmp(cond, lhs, rhs);
            let result = ctx.op_result(op, 0);
            self.values.insert(result, val);
            return Ok(());
        }
        if let Ok(o) = clif::Fcmp::from_op(ctx, op) {
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
        if let Ok(load) = clif::Load::from_op(ctx, op) {
            let result_ty = ctx.op_result_types(op)[0];
            let td = ctx.get_type(result_ty);
            if td.dialect == Symbol::new("core") && td.name == Symbol::new("nil") {
                return Ok(());
            }
            let operands = ctx.op_operands(op);
            let addr = self.lookup(operands[0])?;
            let ty = translate_type(ctx, result_ty, self.ptr_ty)?;
            let val =
                self.builder
                    .ins()
                    .load(ty, cl_ir::MemFlagsData::new(), addr, load.offset(ctx));
            let result = ctx.op_result(op, 0);
            self.values.insert(result, val);
            return Ok(());
        }
        if let Ok(store) = clif::Store::from_op(ctx, op) {
            let operands = ctx.op_operands(op);
            if self.is_nil_typed(operands[0]) {
                return Ok(());
            }
            let value = self.lookup(operands[0])?;
            let addr = self.lookup(operands[1])?;
            self.builder
                .ins()
                .store(cl_ir::MemFlagsData::new(), value, addr, store.offset(ctx));
            return Ok(());
        }

        if let Ok(armw) = clif::AtomicRmw::from_op(ctx, op) {
            let result_ty = ctx.op_result_types(op)[0];
            let ty = translate_type(ctx, result_ty, self.ptr_ty)?;
            let operands = ctx.op_operands(op);
            let mut addr = self.lookup(operands[0])?;
            let value = self.lookup(operands[1])?;
            let offset = armw.offset(ctx);
            if offset != 0 {
                // `iadd_imm_s` sign-extends the immediate, matching the historical
                // `iadd_imm` behaviour and the signed `i32` offset this dialect carries.
                addr = self.builder.ins().iadd_imm_s(addr, i64::from(offset));
            }
            let rmw_op = parse_atomic_rmw_op(armw.bin_op(ctx))?;
            let val =
                self.builder
                    .ins()
                    .atomic_rmw(ty, cl_ir::MemFlagsData::new(), rmw_op, addr, value);
            let result = ctx.op_result(op, 0);
            self.values.insert(result, val);
            return Ok(());
        }

        // === Symbol Address ===
        if let Ok(sym_addr) = clif::SymbolAddr::from_op(ctx, op) {
            let sym = sym_addr.sym(ctx);
            // Check function refs first, then data refs
            let val = if let Some(func_ref) = self.func_ref(sym.clone()) {
                self.builder.ins().func_addr(self.ptr_ty, func_ref)
            } else if let Some(gv) = self.data_ref(sym.clone()) {
                self.builder.ins().symbol_value(self.ptr_ty, gv)
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
        if clif::Trap::from_op(ctx, op).is_ok() {
            self.builder.ins().trap(TrapCode::unwrap_user(1));
            return Ok(());
        }

        // === Return Call (tail call) ===
        if let Ok(rc) = clif::ReturnCall::from_op(ctx, op) {
            let callee_sym = rc.callee(ctx);
            let func_ref = self
                .func_ref(callee_sym.clone())
                .ok_or_else(|| CompilationError::function_not_found(&callee_sym.to_string()))?;

            let operands = ctx.op_operands(op);
            let args = self.runtime_values(operands)?;

            self.builder.ins().return_call(func_ref, &args);
            return Ok(());
        }

        // === Return Call Indirect (indirect tail call) ===
        if let Ok(rci) = clif::ReturnCallIndirect::from_op(ctx, op) {
            let operands = ctx.op_operands(op);
            let callee = self.lookup(operands[0])?;
            let args = self.runtime_values(&operands[1..])?;

            let sig_ty = rci.sig(ctx);
            let sig = translate_signature(ctx, sig_ty, self.default_call_conv, self.ptr_ty)?;
            let sig_ref = self.builder.import_signature(sig);

            self.builder
                .ins()
                .return_call_indirect(sig_ref, callee, &args);
            return Ok(());
        }

        // === Indirect Call ===
        if let Ok(call_ind) = clif::CallIndirect::from_op(ctx, op) {
            let operands = ctx.op_operands(op);
            let callee = self.lookup(operands[0])?;
            let args = self.runtime_values(&operands[1..])?;

            let sig_ty = call_ind.sig(ctx);
            let sig = translate_signature(ctx, sig_ty, self.default_call_conv, self.ptr_ty)?;
            let sig_ref = self.builder.import_signature(sig);

            let inst = self.builder.ins().call_indirect(sig_ref, callee, &args);
            let results = self.builder.inst_results(inst).to_vec();
            return self.map_call_results(op, &results);
        }

        // === Type Conversions ===
        if clif::Ireduce::from_op(ctx, op).is_ok() {
            let operands = ctx.op_operands(op);
            let result_ty = ctx.op_result_types(op)[0];
            let ty = translate_type(ctx, result_ty, self.ptr_ty)?;
            let val = self.lookup(operands[0])?;
            let cl_val = self.builder.ins().ireduce(ty, val);
            self.values.insert(ctx.op_result(op, 0), cl_val);
            return Ok(());
        }
        if clif::Uextend::from_op(ctx, op).is_ok() {
            let operands = ctx.op_operands(op);
            let result_ty = ctx.op_result_types(op)[0];
            let ty = translate_type(ctx, result_ty, self.ptr_ty)?;
            let val = self.lookup(operands[0])?;
            let cl_val = self.builder.ins().uextend(ty, val);
            self.values.insert(ctx.op_result(op, 0), cl_val);
            return Ok(());
        }
        if clif::Sextend::from_op(ctx, op).is_ok() {
            let operands = ctx.op_operands(op);
            let result_ty = ctx.op_result_types(op)[0];
            let ty = translate_type(ctx, result_ty, self.ptr_ty)?;
            let val = self.lookup(operands[0])?;
            let cl_val = self.builder.ins().sextend(ty, val);
            self.values.insert(ctx.op_result(op, 0), cl_val);
            return Ok(());
        }
        if clif::Fpromote::from_op(ctx, op).is_ok() {
            let operands = ctx.op_operands(op);
            let result_ty = ctx.op_result_types(op)[0];
            let ty = translate_type(ctx, result_ty, self.ptr_ty)?;
            let val = self.lookup(operands[0])?;
            let cl_val = self.builder.ins().fpromote(ty, val);
            self.values.insert(ctx.op_result(op, 0), cl_val);
            return Ok(());
        }
        if clif::Fdemote::from_op(ctx, op).is_ok() {
            let operands = ctx.op_operands(op);
            let result_ty = ctx.op_result_types(op)[0];
            let ty = translate_type(ctx, result_ty, self.ptr_ty)?;
            let val = self.lookup(operands[0])?;
            let cl_val = self.builder.ins().fdemote(ty, val);
            self.values.insert(ctx.op_result(op, 0), cl_val);
            return Ok(());
        }
        if clif::FcvtToSint::from_op(ctx, op).is_ok() {
            let operands = ctx.op_operands(op);
            let result_ty = ctx.op_result_types(op)[0];
            let ty = translate_type(ctx, result_ty, self.ptr_ty)?;
            let val = self.lookup(operands[0])?;
            let cl_val = self.builder.ins().fcvt_to_sint(ty, val);
            self.values.insert(ctx.op_result(op, 0), cl_val);
            return Ok(());
        }
        if clif::FcvtFromSint::from_op(ctx, op).is_ok() {
            let operands = ctx.op_operands(op);
            let result_ty = ctx.op_result_types(op)[0];
            let ty = translate_type(ctx, result_ty, self.ptr_ty)?;
            let val = self.lookup(operands[0])?;
            let cl_val = self.builder.ins().fcvt_from_sint(ty, val);
            self.values.insert(ctx.op_result(op, 0), cl_val);
            return Ok(());
        }
        if clif::FcvtToUint::from_op(ctx, op).is_ok() {
            let operands = ctx.op_operands(op);
            let result_ty = ctx.op_result_types(op)[0];
            let ty = translate_type(ctx, result_ty, self.ptr_ty)?;
            let val = self.lookup(operands[0])?;
            let cl_val = self.builder.ins().fcvt_to_uint(ty, val);
            self.values.insert(ctx.op_result(op, 0), cl_val);
            return Ok(());
        }
        if clif::FcvtFromUint::from_op(ctx, op).is_ok() {
            let operands = ctx.op_operands(op);
            let result_ty = ctx.op_result_types(op)[0];
            let ty = translate_type(ctx, result_ty, self.ptr_ty)?;
            let val = self.lookup(operands[0])?;
            let cl_val = self.builder.ins().fcvt_from_uint(ty, val);
            self.values.insert(ctx.op_result(op, 0), cl_val);
            return Ok(());
        }

        // === Bitwise Operations ===
        if clif::Band::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary(ops[0], ops[1], op, |b, a, c| b.ins().band(a, c));
        }
        if clif::Bor::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary(ops[0], ops[1], op, |b, a, c| b.ins().bor(a, c));
        }
        if clif::Bxor::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary(ops[0], ops[1], op, |b, a, c| b.ins().bxor(a, c));
        }
        if clif::Ishl::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary(ops[0], ops[1], op, |b, a, c| b.ins().ishl(a, c));
        }
        if clif::Sshr::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary(ops[0], ops[1], op, |b, a, c| b.ins().sshr(a, c));
        }
        if clif::Ushr::from_op(ctx, op).is_ok() {
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
        ctx.intern_type(TypeData {
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

        let func_ty = clif::func_sig(&mut ctx, [i32_ty, i32_ty], [i64_ty]).as_type_ref();

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

        let func_ty = clif::func_sig(&mut ctx, [i64_ty], [nil_ty]).as_type_ref();

        let sig = translate_signature(&ctx, func_ty, CallConv::SystemV, cl_types::I64).unwrap();
        assert_eq!(sig.params.len(), 1);
        assert_eq!(sig.params[0].value_type, cl_types::I64);
        assert_eq!(sig.returns.len(), 0);
    }

    #[test]
    fn test_translate_signature_omits_nil_parameters_in_order() {
        let mut ctx = IrContext::new();
        let i32_ty = make_core_type(&mut ctx, "i32");
        let i64_ty = make_core_type(&mut ctx, "i64");
        let nil_ty = make_core_type(&mut ctx, "nil");

        let func_ty = clif::func_sig(&mut ctx, [i32_ty, nil_ty, i64_ty], [i32_ty]).as_type_ref();

        let sig = translate_signature(&ctx, func_ty, CallConv::SystemV, cl_types::I64).unwrap();
        assert_eq!(
            sig.params
                .iter()
                .map(|param| param.value_type)
                .collect::<Vec<_>>(),
            vec![cl_types::I32, cl_types::I64]
        );
        assert_eq!(sig.returns.len(), 1);
        assert_eq!(sig.returns[0].value_type, cl_types::I32);
    }

    #[test]
    fn test_translate_signature_no_params() {
        let mut ctx = IrContext::new();
        let i64_ty = make_core_type(&mut ctx, "i64");

        let func_ty = clif::func_sig(&mut ctx, [], [i64_ty]).as_type_ref();

        let sig = translate_signature(&ctx, func_ty, CallConv::SystemV, cl_types::I64).unwrap();
        assert_eq!(sig.params.len(), 0);
        assert_eq!(sig.returns.len(), 1);
        assert_eq!(sig.returns[0].value_type, cl_types::I64);
    }

    #[test]
    fn signature_call_conv_selects_the_cranelift_convention() {
        let mut ctx = IrContext::new();
        let i32_ty = make_core_type(&mut ctx, "i32");
        let platform = clif::func_sig(&mut ctx, [i32_ty], []).as_type_ref();
        let mut attrs = trunk_ir::AttributeMap::new();
        func::CallConv::Tail.set_in(&mut ctx, &mut attrs);
        let tail = clif::func_sig_with_attrs(&mut ctx, [i32_ty], [], attrs).as_type_ref();
        let mut malformed_attrs = trunk_ir::AttributeMap::new();
        malformed_attrs.insert(Symbol::new(func::CALL_CONV_ATTR), ctx.string_attr("fast"));
        let malformed =
            clif::func_sig_with_attrs(&mut ctx, [i32_ty], [], malformed_attrs).as_type_ref();

        let sig = translate_signature(&ctx, platform, CallConv::SystemV, cl_types::I64).unwrap();
        assert_eq!(sig.call_conv, CallConv::SystemV);
        let sig = translate_signature(&ctx, tail, CallConv::SystemV, cl_types::I64).unwrap();
        assert_eq!(sig.call_conv, CallConv::Tail);
        assert!(translate_signature(&ctx, malformed, CallConv::SystemV, cl_types::I64).is_err());
    }
}

#[cfg(test)]
mod result_list_tests {
    use super::*;
    use trunk_ir::types::TypeData;

    fn make_core_type(ctx: &mut IrContext, name: &'static str) -> TypeRef {
        ctx.intern_type(TypeData {
            dialect: Symbol::new("core"),
            name: Symbol::new(name),
            params: Default::default(),
            attrs: Default::default(),
        })
    }

    #[test]
    fn translates_resultless_and_ordered_multi_result_signatures() {
        let mut ctx = IrContext::new();
        let i32 = make_core_type(&mut ctx, "i32");
        let i64 = make_core_type(&mut ctx, "i64");
        let resultless = clif::func_sig(&mut ctx, [], []).as_type_ref();
        let multiple = clif::func_sig(&mut ctx, [i32], [i64, i32]).as_type_ref();
        assert!(
            translate_signature(&ctx, resultless, CallConv::SystemV, cl_types::I64)
                .unwrap()
                .returns
                .is_empty()
        );
        let translated =
            translate_signature(&ctx, multiple, CallConv::SystemV, cl_types::I64).unwrap();
        assert_eq!(translated.returns[0].value_type, cl_types::I64);
        assert_eq!(translated.returns[1].value_type, cl_types::I32);
    }
}
