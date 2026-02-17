//! Function-level code generation for Cranelift backend.
//!
//! Translates `clif.*` dialect operations within a single function body
//! to Cranelift IR instructions using `FunctionBuilder`.

use std::collections::HashMap;

use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{self as cl_ir, InstBuilder};
use cranelift_codegen::isa::CallConv;
use cranelift_frontend::FunctionBuilder;
use trunk_ir::dialect::{clif, core};
use trunk_ir::{Block as IrBlock, DialectOp, DialectType, Operation, Symbol, Type, Value};

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
    /// Maps TrunkIR blocks to Cranelift blocks for multi-block functions.
    pub(crate) block_map: HashMap<IrBlock<'db>, cl_ir::Block>,
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
            CompilationError::codegen("TrunkIR value not found in Cranelift mapping")
        })
    }

    /// Look up a TrunkIR block in the block mapping.
    pub(crate) fn lookup_block(&self, ir_block: IrBlock<'db>) -> CompilationResult<cl_ir::Block> {
        self.block_map.get(&ir_block).copied().ok_or_else(|| {
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
            // Skip operands not in the value map (Nil-typed values that were
            // intentionally not materialized).
            let vals: Vec<cl_ir::Value> = ret
                .values(db)
                .iter()
                .filter_map(|v| self.values.get(v).copied())
                .collect();
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

        // === Memory ===
        if let Ok(load) = clif::Load::from_operation(db, *op) {
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
