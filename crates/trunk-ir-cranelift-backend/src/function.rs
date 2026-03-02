//! Function-level code generation for Cranelift backend.
//!
//! Translates `clif.*` dialect operations within a single function body
//! to Cranelift IR instructions using `FunctionBuilder`.

use std::collections::HashMap;

use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{self as cl_ir, InstBuilder, TrapCode};
use cranelift_codegen::isa::CallConv;
use cranelift_frontend::FunctionBuilder;
use trunk_ir::arena::context::IrContext;
use trunk_ir::arena::dialect::clif as arena_clif;
use trunk_ir::arena::ops::ArenaDialectOp;
use trunk_ir::arena::refs::{BlockRef, OpRef, TypeRef, ValueRef};
use trunk_ir::dialect::{clif, core};
use trunk_ir::ir::Symbol;
use trunk_ir::{
    Block as IrBlock, BlockId, DialectOp, DialectType, Operation, Type, Value, ValueDef,
};

use crate::{CompilationError, CompilationResult};

/// Translate a TrunkIR core type to a Cranelift IR type.
pub(crate) fn translate_type(
    db: &dyn salsa::Database,
    ty: Type<'_>,
) -> CompilationResult<cl_types::Type> {
    if core::I8::from_type(db, ty).is_some() {
        return Ok(cl_types::I8);
    }
    if core::I16::from_type(db, ty).is_some() {
        return Ok(cl_types::I16);
    }
    if core::I32::from_type(db, ty).is_some() {
        return Ok(cl_types::I32);
    }
    if core::I64::from_type(db, ty).is_some() {
        return Ok(cl_types::I64);
    }
    if core::F32::from_type(db, ty).is_some() {
        return Ok(cl_types::F32);
    }
    if core::F64::from_type(db, ty).is_some() {
        return Ok(cl_types::F64);
    }
    // core.ptr -> i64 (pointers are 64-bit on the target platform)
    if core::Ptr::from_type(db, ty).is_some() {
        return Ok(cl_types::I64);
    }
    Err(CompilationError::type_error(format!(
        "unsupported type for Cranelift: {}.{}",
        ty.dialect(db),
        ty.name(db),
    )))
}

/// Translate a TrunkIR `core.func` type to a Cranelift `Signature`.
pub(crate) fn translate_signature(
    db: &dyn salsa::Database,
    func_ty: core::Func<'_>,
    call_conv: CallConv,
) -> CompilationResult<cl_ir::Signature> {
    let mut sig = cl_ir::Signature::new(call_conv);

    for &param_ty in func_ty.params(db).iter() {
        let cl_ty = translate_type(db, param_ty)?;
        sig.params.push(cl_ir::AbiParam::new(cl_ty));
    }

    let ret_ty = func_ty.result(db);
    // Nil return type means void — no return values.
    if core::Nil::from_type(db, ret_ty).is_none() {
        let cl_ty = translate_type(db, ret_ty)?;
        sig.returns.push(cl_ir::AbiParam::new(cl_ty));
    }

    Ok(sig)
}

/// Translates `clif.*` operations within a single function body
/// to Cranelift IR instructions.
pub(crate) struct FunctionTranslator<'a, 'db> {
    db: &'db dyn salsa::Database,
    pub(crate) builder: FunctionBuilder<'a>,
    /// Maps TrunkIR values to Cranelift IR values.
    pub(crate) values: HashMap<Value<'db>, cl_ir::Value>,
    /// Maps function symbols to Cranelift FuncRefs for call instructions.
    func_refs: &'a HashMap<Symbol, cl_ir::FuncRef>,
    /// Maps TrunkIR block IDs to Cranelift blocks for multi-block functions.
    ///
    /// Uses `BlockId` rather than `Block` because successor references may
    /// point to different `Block` tracked-struct instances that share the same
    /// `BlockId` (e.g., stub blocks created by scf_to_cf).
    pub(crate) block_map: HashMap<BlockId, cl_ir::Block>,
}

impl<'a, 'db> FunctionTranslator<'a, 'db> {
    pub(crate) fn new(
        db: &'db dyn salsa::Database,
        builder: FunctionBuilder<'a>,
        func_refs: &'a HashMap<Symbol, cl_ir::FuncRef>,
    ) -> Self {
        Self {
            db,
            builder,
            values: HashMap::new(),
            func_refs,
            block_map: HashMap::new(),
        }
    }

    /// Look up a TrunkIR value in the value mapping.
    fn lookup(&self, ir_val: Value<'db>) -> CompilationResult<cl_ir::Value> {
        self.values.get(&ir_val).copied().ok_or_else(|| {
            let db = self.db;
            let detail = match ir_val.def(db) {
                ValueDef::OpResult(op) => {
                    let idx = ir_val.index(db);
                    let results = op.results(db);
                    let result_ty = results
                        .get(idx)
                        .map(|ty| format!("{}.{}", ty.dialect(db), ty.name(db)))
                        .unwrap_or_else(|| "<index out of bounds>".to_string());
                    format!(
                        "result #{} of {}.{} (result type: {}, op has {} results)",
                        idx,
                        op.dialect(db),
                        op.name(db),
                        result_ty,
                        results.len(),
                    )
                }
                ValueDef::BlockArg(block_id) => {
                    format!("block arg #{} of block {:?}", ir_val.index(db), block_id,)
                }
            };
            CompilationError::codegen(format!(
                "TrunkIR value not found in Cranelift mapping: {detail} (mapped {} values total)",
                self.values.len(),
            ))
        })
    }

    /// Check if a value has `core.nil` type by examining its definition.
    fn is_nil_typed(&self, val: Value<'db>) -> bool {
        let db = self.db;
        match val.def(db) {
            ValueDef::OpResult(op) => op
                .results(db)
                .get(val.index(db))
                .is_some_and(|ty| core::Nil::from_type(db, *ty).is_some()),
            ValueDef::BlockArg(_) => false,
        }
    }

    /// Look up a TrunkIR block in the block mapping by its BlockId.
    pub(crate) fn lookup_block(&self, ir_block: IrBlock<'db>) -> CompilationResult<cl_ir::Block> {
        let id = ir_block.id(self.db);
        self.block_map.get(&id).copied().ok_or_else(|| {
            CompilationError::codegen("TrunkIR block not found in Cranelift block mapping")
        })
    }

    /// Translate a single `clif.*` operation to Cranelift IR.
    pub(crate) fn translate_op(&mut self, op: &Operation<'db>) -> CompilationResult<()> {
        let db = self.db;

        // === Constants ===
        if let Ok(c) = clif::Iconst::from_operation(db, *op) {
            // Nil constants have no runtime representation — skip emission.
            if core::Nil::from_type(db, c.result_ty(db)).is_some() {
                return Ok(());
            }
            let ty = translate_type(db, c.result_ty(db))?;
            let val = self.builder.ins().iconst(ty, c.value(db));
            self.values.insert(c.result(db), val);
            return Ok(());
        }
        if let Ok(c) = clif::F32const::from_operation(db, *op) {
            let val = self.builder.ins().f32const(c.value(db));
            self.values.insert(c.result(db), val);
            return Ok(());
        }
        if let Ok(c) = clif::F64const::from_operation(db, *op) {
            let val = self.builder.ins().f64const(c.value(db));
            self.values.insert(c.result(db), val);
            return Ok(());
        }

        // === Integer Arithmetic ===
        if let Ok(o) = clif::Iadd::from_operation(db, *op) {
            return self.emit_binary(o.lhs(db), o.rhs(db), o.result(db), |b, a, c| {
                b.ins().iadd(a, c)
            });
        }
        if let Ok(o) = clif::Isub::from_operation(db, *op) {
            return self.emit_binary(o.lhs(db), o.rhs(db), o.result(db), |b, a, c| {
                b.ins().isub(a, c)
            });
        }
        if let Ok(o) = clif::Imul::from_operation(db, *op) {
            return self.emit_binary(o.lhs(db), o.rhs(db), o.result(db), |b, a, c| {
                b.ins().imul(a, c)
            });
        }
        if let Ok(o) = clif::Sdiv::from_operation(db, *op) {
            return self.emit_binary(o.lhs(db), o.rhs(db), o.result(db), |b, a, c| {
                b.ins().sdiv(a, c)
            });
        }
        if let Ok(o) = clif::Udiv::from_operation(db, *op) {
            return self.emit_binary(o.lhs(db), o.rhs(db), o.result(db), |b, a, c| {
                b.ins().udiv(a, c)
            });
        }
        if let Ok(o) = clif::Srem::from_operation(db, *op) {
            return self.emit_binary(o.lhs(db), o.rhs(db), o.result(db), |b, a, c| {
                b.ins().srem(a, c)
            });
        }
        if let Ok(o) = clif::Urem::from_operation(db, *op) {
            return self.emit_binary(o.lhs(db), o.rhs(db), o.result(db), |b, a, c| {
                b.ins().urem(a, c)
            });
        }
        if let Ok(o) = clif::Ineg::from_operation(db, *op) {
            return self.emit_unary(o.operand(db), o.result(db), |b, v| b.ins().ineg(v));
        }

        // === Floating Point Arithmetic ===
        if let Ok(o) = clif::Fadd::from_operation(db, *op) {
            return self.emit_binary(o.lhs(db), o.rhs(db), o.result(db), |b, a, c| {
                b.ins().fadd(a, c)
            });
        }
        if let Ok(o) = clif::Fsub::from_operation(db, *op) {
            return self.emit_binary(o.lhs(db), o.rhs(db), o.result(db), |b, a, c| {
                b.ins().fsub(a, c)
            });
        }
        if let Ok(o) = clif::Fmul::from_operation(db, *op) {
            return self.emit_binary(o.lhs(db), o.rhs(db), o.result(db), |b, a, c| {
                b.ins().fmul(a, c)
            });
        }
        if let Ok(o) = clif::Fdiv::from_operation(db, *op) {
            return self.emit_binary(o.lhs(db), o.rhs(db), o.result(db), |b, a, c| {
                b.ins().fdiv(a, c)
            });
        }
        if let Ok(o) = clif::Fneg::from_operation(db, *op) {
            return self.emit_unary(o.operand(db), o.result(db), |b, v| b.ins().fneg(v));
        }

        // === Call ===
        if let Ok(call) = clif::Call::from_operation(db, *op) {
            let callee_sym = call.callee(db);
            let func_ref = self
                .func_refs
                .get(&callee_sym)
                .copied()
                .ok_or_else(|| CompilationError::function_not_found(&callee_sym.to_string()))?;

            let args: Vec<cl_ir::Value> = call
                .args(db)
                .iter()
                .map(|v| self.lookup(*v))
                .collect::<CompilationResult<_>>()?;

            let inst = self.builder.ins().call(func_ref, &args);
            let results = self.builder.inst_results(inst);
            if !results.is_empty() {
                self.values.insert(call.result(db), results[0]);
            }
            return Ok(());
        }

        // === Return ===
        if let Ok(ret) = clif::Return::from_operation(db, *op) {
            let mut vals = Vec::new();
            for v in ret.values(db).iter() {
                if let Some(&cl_val) = self.values.get(v) {
                    vals.push(cl_val);
                } else {
                    // Only Nil-typed values may be absent from the value map
                    // (they are intentionally not materialized). Any other
                    // missing mapping is a codegen bug.
                    let is_nil = match v.def(db) {
                        ValueDef::OpResult(def_op) => {
                            let ty = def_op.results(db)[v.index(db)];
                            core::Nil::from_type(db, ty).is_some()
                        }
                        ValueDef::BlockArg(_) => false,
                    };
                    if !is_nil {
                        return Err(CompilationError::codegen(format_args!(
                            "return operand {:?} has no Cranelift value mapping \
                             and is not Nil-typed",
                            v,
                        )));
                    }
                }
            }
            self.builder.ins().return_(&vals);
            return Ok(());
        }

        // === Control Flow ===
        if let Ok(jump) = clif::Jump::from_operation(db, *op) {
            let ir_dest = jump.dest(db);
            let cl_dest = self.lookup_block(ir_dest)?;
            let args: Vec<cl_ir::BlockArg> = jump
                .args(db)
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
        // cf.cond_br carries no block args by design, so empty arg slices are intentional.
        if let Ok(brif) = clif::Brif::from_operation(db, *op) {
            let cond = self.lookup(brif.cond(db))?;
            let cl_then = self.lookup_block(brif.then_dest(db))?;
            let cl_else = self.lookup_block(brif.else_dest(db))?;
            self.builder.ins().brif(cond, cl_then, &[], cl_else, &[]);
            return Ok(());
        }

        // === Comparisons ===
        if let Ok(o) = clif::Icmp::from_operation(db, *op) {
            let cond = parse_int_cc(o.cond(db))?;
            let lhs = self.lookup(o.lhs(db))?;
            let rhs = self.lookup(o.rhs(db))?;
            let val = self.builder.ins().icmp(cond, lhs, rhs);
            self.values.insert(o.result(db), val);
            return Ok(());
        }
        if let Ok(o) = clif::Fcmp::from_operation(db, *op) {
            let cond = parse_float_cc(o.cond(db))?;
            let lhs = self.lookup(o.lhs(db))?;
            let rhs = self.lookup(o.rhs(db))?;
            let val = self.builder.ins().fcmp(cond, lhs, rhs);
            self.values.insert(o.result(db), val);
            return Ok(());
        }

        // === Memory ===
        if let Ok(load) = clif::Load::from_operation(db, *op) {
            // Nil loads have no runtime representation — skip emission.
            if core::Nil::from_type(db, load.result_ty(db)).is_some() {
                return Ok(());
            }
            let addr = self.lookup(load.addr(db))?;
            let ty = translate_type(db, load.result_ty(db))?;
            let val = self
                .builder
                .ins()
                .load(ty, cl_ir::MemFlags::new(), addr, load.offset(db));
            self.values.insert(load.result(db), val);
            return Ok(());
        }
        if let Ok(store) = clif::Store::from_operation(db, *op) {
            // Nil values have no runtime representation — skip the store.
            if self.is_nil_typed(store.value(db)) {
                return Ok(());
            }
            let value = self.lookup(store.value(db))?;
            let addr = self.lookup(store.addr(db))?;
            self.builder
                .ins()
                .store(cl_ir::MemFlags::new(), value, addr, store.offset(db));
            return Ok(());
        }

        // === Symbol Address ===
        if let Ok(sym_addr) = clif::SymbolAddr::from_operation(db, *op) {
            let sym = sym_addr.sym(db);
            let func_ref = self
                .func_refs
                .get(&sym)
                .copied()
                .ok_or_else(|| CompilationError::function_not_found(&sym.to_string()))?;
            let val = self.builder.ins().func_addr(cl_types::I64, func_ref);
            self.values.insert(sym_addr.result(db), val);
            return Ok(());
        }

        // === Trap ===
        if clif::Trap::from_operation(db, *op).is_ok() {
            self.builder.ins().trap(TrapCode::unwrap_user(1));
            return Ok(());
        }

        // === Indirect Call ===
        if let Ok(call_ind) = clif::CallIndirect::from_operation(db, *op) {
            let callee = self.lookup(call_ind.callee(db))?;
            let args: Vec<cl_ir::Value> = call_ind
                .args(db)
                .iter()
                .map(|v| self.lookup(*v))
                .collect::<CompilationResult<_>>()?;

            let sig_ty = call_ind.sig(db);
            let func_ty = core::Func::from_type(db, sig_ty).ok_or_else(|| {
                CompilationError::type_error("call_indirect sig must be core.func type")
            })?;
            let sig = translate_signature(db, func_ty, CallConv::SystemV)?;
            let sig_ref = self.builder.import_signature(sig);

            let inst = self.builder.ins().call_indirect(sig_ref, callee, &args);
            let results = self.builder.inst_results(inst);
            if !results.is_empty() {
                self.values.insert(call_ind.result(db), results[0]);
            }
            return Ok(());
        }

        Err(CompilationError::codegen(format!(
            "unsupported operation: {}.{}",
            op.dialect(db),
            op.name(db),
        )))
    }

    /// Helper: emit a binary operation (two inputs, one output).
    fn emit_binary(
        &mut self,
        lhs_ir: Value<'db>,
        rhs_ir: Value<'db>,
        result_ir: Value<'db>,
        f: impl FnOnce(&mut FunctionBuilder<'a>, cl_ir::Value, cl_ir::Value) -> cl_ir::Value,
    ) -> CompilationResult<()> {
        let lhs = self.lookup(lhs_ir)?;
        let rhs = self.lookup(rhs_ir)?;
        let cl_val = f(&mut self.builder, lhs, rhs);
        self.values.insert(result_ir, cl_val);
        Ok(())
    }

    /// Helper: emit a unary operation (one input, one output).
    fn emit_unary(
        &mut self,
        operand_ir: Value<'db>,
        result_ir: Value<'db>,
        f: impl FnOnce(&mut FunctionBuilder<'a>, cl_ir::Value) -> cl_ir::Value,
    ) -> CompilationResult<()> {
        let operand = self.lookup(operand_ir)?;
        let cl_val = f(&mut self.builder, operand);
        self.values.insert(result_ir, cl_val);
        Ok(())
    }
}

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

// =============================================================================
// Arena IR version
// =============================================================================

/// Check if a type is pointer-like in the native backend.
///
/// Translate an arena TypeRef to a Cranelift IR type.
pub(crate) fn translate_type_arena(
    ctx: &IrContext,
    ty: TypeRef,
) -> CompilationResult<cl_types::Type> {
    let td = ctx.types.get(ty);
    let core_dialect = Symbol::new("core");
    if td.dialect == core_dialect {
        return td.name.with_str(|n| match n {
            "i8" => Ok(cl_types::I8),
            "i16" => Ok(cl_types::I16),
            "i32" => Ok(cl_types::I32),
            "i64" => Ok(cl_types::I64),
            "f32" => Ok(cl_types::F32),
            "f64" => Ok(cl_types::F64),
            "ptr" => Ok(cl_types::I64),
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

/// Translate an arena func type (TypeRef) to a Cranelift Signature.
pub(crate) fn translate_signature_arena(
    ctx: &IrContext,
    func_ty_ref: TypeRef,
    call_conv: CallConv,
) -> CompilationResult<cl_ir::Signature> {
    let td = ctx.types.get(func_ty_ref);
    if td.dialect != Symbol::new("core") || td.name != Symbol::new("func") {
        return Err(CompilationError::type_error(
            "expected core.func type for signature translation",
        ));
    }

    let mut sig = cl_ir::Signature::new(call_conv);

    // core.func type layout: params[0] = return type, params[1..] = parameter types
    // (matches Salsa core::Func where result() = params(db)[0], params() = params(db)[1..])
    if td.params.is_empty() {
        return Err(CompilationError::type_error(
            "core.func type must have at least a return type param",
        ));
    }

    let ret_type = td.params[0];
    let param_types = &td.params[1..];

    for &param_ty in param_types {
        let cl_ty = translate_type_arena(ctx, param_ty)?;
        sig.params.push(cl_ir::AbiParam::new(cl_ty));
    }

    // Nil return type means void — no return values.
    let ret_td = ctx.types.get(ret_type);
    if !(ret_td.dialect == Symbol::new("core") && ret_td.name == Symbol::new("nil")) {
        let cl_ty = translate_type_arena(ctx, ret_type)?;
        sig.returns.push(cl_ir::AbiParam::new(cl_ty));
    }

    Ok(sig)
}

/// Arena IR version of FunctionTranslator.
pub(crate) struct ArenaFunctionTranslator<'a> {
    ctx: &'a IrContext,
    pub(crate) builder: FunctionBuilder<'a>,
    /// Maps TrunkIR arena values to Cranelift IR values.
    pub(crate) values: HashMap<ValueRef, cl_ir::Value>,
    /// Maps function symbols to Cranelift FuncRefs.
    func_refs: &'a HashMap<Symbol, cl_ir::FuncRef>,
    /// Maps TrunkIR block refs to Cranelift blocks.
    pub(crate) block_map: HashMap<BlockRef, cl_ir::Block>,
}

impl<'a> ArenaFunctionTranslator<'a> {
    pub(crate) fn new(
        ctx: &'a IrContext,
        builder: FunctionBuilder<'a>,
        func_refs: &'a HashMap<Symbol, cl_ir::FuncRef>,
    ) -> Self {
        Self {
            ctx,
            builder,
            values: HashMap::new(),
            func_refs,
            block_map: HashMap::new(),
        }
    }

    fn lookup(&self, ir_val: ValueRef) -> CompilationResult<cl_ir::Value> {
        self.values.get(&ir_val).copied().ok_or_else(|| {
            CompilationError::codegen(format!(
                "TrunkIR value not found in Cranelift mapping (mapped {} values total)",
                self.values.len(),
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
            let ty = translate_type_arena(ctx, result_ty)?;
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
            return self.emit_binary_arena(ops[0], ops[1], op, |b, a, c| b.ins().iadd(a, c));
        }
        if arena_clif::Isub::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary_arena(ops[0], ops[1], op, |b, a, c| b.ins().isub(a, c));
        }
        if arena_clif::Imul::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary_arena(ops[0], ops[1], op, |b, a, c| b.ins().imul(a, c));
        }
        if arena_clif::Sdiv::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary_arena(ops[0], ops[1], op, |b, a, c| b.ins().sdiv(a, c));
        }
        if arena_clif::Udiv::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary_arena(ops[0], ops[1], op, |b, a, c| b.ins().udiv(a, c));
        }
        if arena_clif::Srem::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary_arena(ops[0], ops[1], op, |b, a, c| b.ins().srem(a, c));
        }
        if arena_clif::Urem::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary_arena(ops[0], ops[1], op, |b, a, c| b.ins().urem(a, c));
        }
        if arena_clif::Ineg::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_unary_arena(ops[0], op, |b, v| b.ins().ineg(v));
        }

        // === Floating Point Arithmetic ===
        if arena_clif::Fadd::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary_arena(ops[0], ops[1], op, |b, a, c| b.ins().fadd(a, c));
        }
        if arena_clif::Fsub::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary_arena(ops[0], ops[1], op, |b, a, c| b.ins().fsub(a, c));
        }
        if arena_clif::Fmul::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary_arena(ops[0], ops[1], op, |b, a, c| b.ins().fmul(a, c));
        }
        if arena_clif::Fdiv::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary_arena(ops[0], ops[1], op, |b, a, c| b.ins().fdiv(a, c));
        }
        if arena_clif::Fneg::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_unary_arena(ops[0], op, |b, v| b.ins().fneg(v));
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
            let ty = translate_type_arena(ctx, result_ty)?;
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

        // === Symbol Address ===
        if let Ok(sym_addr) = arena_clif::SymbolAddr::from_op(ctx, op) {
            let sym = sym_addr.sym(ctx);
            let func_ref = self
                .func_refs
                .get(&sym)
                .copied()
                .ok_or_else(|| CompilationError::function_not_found(&sym.to_string()))?;
            let val = self.builder.ins().func_addr(cl_types::I64, func_ref);
            let result = ctx.op_result(op, 0);
            self.values.insert(result, val);
            return Ok(());
        }

        // === Trap ===
        if arena_clif::Trap::from_op(ctx, op).is_ok() {
            self.builder.ins().trap(TrapCode::unwrap_user(1));
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
            let sig = translate_signature_arena(ctx, sig_ty, CallConv::SystemV)?;
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
            let ty = translate_type_arena(ctx, result_ty)?;
            let val = self.lookup(operands[0])?;
            let cl_val = self.builder.ins().ireduce(ty, val);
            self.values.insert(ctx.op_result(op, 0), cl_val);
            return Ok(());
        }
        if arena_clif::Uextend::from_op(ctx, op).is_ok() {
            let operands = ctx.op_operands(op);
            let result_ty = ctx.op_result_types(op)[0];
            let ty = translate_type_arena(ctx, result_ty)?;
            let val = self.lookup(operands[0])?;
            let cl_val = self.builder.ins().uextend(ty, val);
            self.values.insert(ctx.op_result(op, 0), cl_val);
            return Ok(());
        }
        if arena_clif::Sextend::from_op(ctx, op).is_ok() {
            let operands = ctx.op_operands(op);
            let result_ty = ctx.op_result_types(op)[0];
            let ty = translate_type_arena(ctx, result_ty)?;
            let val = self.lookup(operands[0])?;
            let cl_val = self.builder.ins().sextend(ty, val);
            self.values.insert(ctx.op_result(op, 0), cl_val);
            return Ok(());
        }
        if arena_clif::Fpromote::from_op(ctx, op).is_ok() {
            let operands = ctx.op_operands(op);
            let result_ty = ctx.op_result_types(op)[0];
            let ty = translate_type_arena(ctx, result_ty)?;
            let val = self.lookup(operands[0])?;
            let cl_val = self.builder.ins().fpromote(ty, val);
            self.values.insert(ctx.op_result(op, 0), cl_val);
            return Ok(());
        }
        if arena_clif::Fdemote::from_op(ctx, op).is_ok() {
            let operands = ctx.op_operands(op);
            let result_ty = ctx.op_result_types(op)[0];
            let ty = translate_type_arena(ctx, result_ty)?;
            let val = self.lookup(operands[0])?;
            let cl_val = self.builder.ins().fdemote(ty, val);
            self.values.insert(ctx.op_result(op, 0), cl_val);
            return Ok(());
        }
        if arena_clif::FcvtToSint::from_op(ctx, op).is_ok() {
            let operands = ctx.op_operands(op);
            let result_ty = ctx.op_result_types(op)[0];
            let ty = translate_type_arena(ctx, result_ty)?;
            let val = self.lookup(operands[0])?;
            let cl_val = self.builder.ins().fcvt_to_sint(ty, val);
            self.values.insert(ctx.op_result(op, 0), cl_val);
            return Ok(());
        }
        if arena_clif::FcvtFromSint::from_op(ctx, op).is_ok() {
            let operands = ctx.op_operands(op);
            let result_ty = ctx.op_result_types(op)[0];
            let ty = translate_type_arena(ctx, result_ty)?;
            let val = self.lookup(operands[0])?;
            let cl_val = self.builder.ins().fcvt_from_sint(ty, val);
            self.values.insert(ctx.op_result(op, 0), cl_val);
            return Ok(());
        }
        if arena_clif::FcvtToUint::from_op(ctx, op).is_ok() {
            let operands = ctx.op_operands(op);
            let result_ty = ctx.op_result_types(op)[0];
            let ty = translate_type_arena(ctx, result_ty)?;
            let val = self.lookup(operands[0])?;
            let cl_val = self.builder.ins().fcvt_to_uint(ty, val);
            self.values.insert(ctx.op_result(op, 0), cl_val);
            return Ok(());
        }
        if arena_clif::FcvtFromUint::from_op(ctx, op).is_ok() {
            let operands = ctx.op_operands(op);
            let result_ty = ctx.op_result_types(op)[0];
            let ty = translate_type_arena(ctx, result_ty)?;
            let val = self.lookup(operands[0])?;
            let cl_val = self.builder.ins().fcvt_from_uint(ty, val);
            self.values.insert(ctx.op_result(op, 0), cl_val);
            return Ok(());
        }

        // === Bitwise Operations ===
        if arena_clif::Band::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary_arena(ops[0], ops[1], op, |b, a, c| b.ins().band(a, c));
        }
        if arena_clif::Bor::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary_arena(ops[0], ops[1], op, |b, a, c| b.ins().bor(a, c));
        }
        if arena_clif::Bxor::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary_arena(ops[0], ops[1], op, |b, a, c| b.ins().bxor(a, c));
        }
        if arena_clif::Ishl::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary_arena(ops[0], ops[1], op, |b, a, c| b.ins().ishl(a, c));
        }
        if arena_clif::Sshr::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary_arena(ops[0], ops[1], op, |b, a, c| b.ins().sshr(a, c));
        }
        if arena_clif::Ushr::from_op(ctx, op).is_ok() {
            let ops = ctx.op_operands(op);
            return self.emit_binary_arena(ops[0], ops[1], op, |b, a, c| b.ins().ushr(a, c));
        }

        let op_data = ctx.op(op);
        Err(CompilationError::codegen(format!(
            "unsupported operation: {}.{}",
            op_data.dialect, op_data.name,
        )))
    }

    fn emit_binary_arena(
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

    fn emit_unary_arena(
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
    use salsa_test_macros::salsa_test;
    use trunk_ir::IdVec;

    #[salsa_test]
    fn test_translate_type_integers(db: &salsa::DatabaseImpl) {
        assert_eq!(
            translate_type(db, core::I8::new(db).as_type()).unwrap(),
            cl_types::I8
        );
        assert_eq!(
            translate_type(db, core::I16::new(db).as_type()).unwrap(),
            cl_types::I16
        );
        assert_eq!(
            translate_type(db, core::I32::new(db).as_type()).unwrap(),
            cl_types::I32
        );
        assert_eq!(
            translate_type(db, core::I64::new(db).as_type()).unwrap(),
            cl_types::I64
        );
    }

    #[salsa_test]
    fn test_translate_type_floats(db: &salsa::DatabaseImpl) {
        assert_eq!(
            translate_type(db, core::F32::new(db).as_type()).unwrap(),
            cl_types::F32
        );
        assert_eq!(
            translate_type(db, core::F64::new(db).as_type()).unwrap(),
            cl_types::F64
        );
    }

    #[salsa_test]
    fn test_translate_type_unsupported(db: &salsa::DatabaseImpl) {
        let nil_ty = core::Nil::new(db).as_type();
        assert!(translate_type(db, nil_ty).is_err());
    }

    #[salsa_test]
    fn test_translate_signature_params_and_return(db: &salsa::DatabaseImpl) {
        let i64_ty = core::I64::new(db).as_type();
        let i32_ty = core::I32::new(db).as_type();
        let func_ty = core::Func::new(db, IdVec::from(vec![i32_ty, i32_ty]), i64_ty);

        let sig = translate_signature(db, func_ty, CallConv::SystemV).unwrap();
        assert_eq!(sig.params.len(), 2);
        assert_eq!(sig.params[0].value_type, cl_types::I32);
        assert_eq!(sig.params[1].value_type, cl_types::I32);
        assert_eq!(sig.returns.len(), 1);
        assert_eq!(sig.returns[0].value_type, cl_types::I64);
    }

    #[salsa_test]
    fn test_translate_signature_void_return(db: &salsa::DatabaseImpl) {
        let i64_ty = core::I64::new(db).as_type();
        let nil_ty = core::Nil::new(db).as_type();
        let func_ty = core::Func::new(db, IdVec::from(vec![i64_ty]), nil_ty);

        let sig = translate_signature(db, func_ty, CallConv::SystemV).unwrap();
        assert_eq!(sig.params.len(), 1);
        assert_eq!(sig.params[0].value_type, cl_types::I64);
        assert_eq!(sig.returns.len(), 0);
    }

    #[salsa_test]
    fn test_translate_signature_no_params(db: &salsa::DatabaseImpl) {
        let i64_ty = core::I64::new(db).as_type();
        let func_ty = core::Func::new(db, IdVec::new(), i64_ty);

        let sig = translate_signature(db, func_ty, CallConv::SystemV).unwrap();
        assert_eq!(sig.params.len(), 0);
        assert_eq!(sig.returns.len(), 1);
        assert_eq!(sig.returns[0].value_type, cl_types::I64);
    }
}
