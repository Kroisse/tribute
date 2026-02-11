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
use trunk_ir::{DialectOp, DialectType, Operation, Symbol, Type, Value};

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
        }
    }

    /// Look up a TrunkIR value in the value mapping.
    fn lookup(&self, ir_val: Value<'db>) -> CompilationResult<cl_ir::Value> {
        self.values.get(&ir_val).copied().ok_or_else(|| {
            CompilationError::codegen("TrunkIR value not found in Cranelift mapping")
        })
    }

    /// Translate a single `clif.*` operation to Cranelift IR.
    pub(crate) fn translate_op(&mut self, op: &Operation<'db>) -> CompilationResult<()> {
        let db = self.db;

        // === Constants ===
        if let Ok(c) = clif::Iconst::from_operation(db, *op) {
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
            let vals: Vec<cl_ir::Value> = ret
                .values(db)
                .iter()
                .map(|v| self.lookup(*v))
                .collect::<CompilationResult<_>>()?;
            self.builder.ins().return_(&vals);
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
