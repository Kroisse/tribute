//! Arena-based arith dialect.
//!
//! Operations are split by type category following MLIR/Cranelift conventions:
//! - Integer arithmetic: `addi`, `subi`, `muli`, `divsi`/`divui`, `remsi`/`remui`, `negi`
//! - Float arithmetic: `addf`, `subf`, `mulf`, `divf`, `negf`
//! - Integer comparison: `cmpi` with predicate attribute (eq, ne, slt, sle, sgt, sge, ult, ule, ugt, uge)
//! - Float comparison: `cmpf` with predicate attribute (oeq, one, olt, ole, ogt, oge)

// === Pure operation registrations ===
crate::register_pure_op!(arith.r#const);

// Integer arithmetic
crate::register_pure_op!(arith.addi);
crate::register_pure_op!(arith.subi);
crate::register_pure_op!(arith.muli);
crate::register_pure_op!(arith.divsi);
crate::register_pure_op!(arith.divui);
crate::register_pure_op!(arith.remsi);
crate::register_pure_op!(arith.remui);
crate::register_pure_op!(arith.negi);

// Float arithmetic
crate::register_pure_op!(arith.addf);
crate::register_pure_op!(arith.subf);
crate::register_pure_op!(arith.mulf);
crate::register_pure_op!(arith.divf);
crate::register_pure_op!(arith.negf);

// Comparisons
crate::register_pure_op!(arith.cmpi);
crate::register_pure_op!(arith.cmpf);

// Bitwise (integer-only, unchanged)
crate::register_pure_op!(arith.and);
crate::register_pure_op!(arith.or);
crate::register_pure_op!(arith.xor);
crate::register_pure_op!(arith.shl);
crate::register_pure_op!(arith.shr);
crate::register_pure_op!(arith.shru);

// Conversions (unchanged)
crate::register_pure_op!(arith.cast);
crate::register_pure_op!(arith.trunc);
crate::register_pure_op!(arith.extend);
crate::register_pure_op!(arith.convert);

#[crate::dialect(crate = crate)]
mod arith {
    #[attr(value: any)]
    fn r#const() -> result {}

    // Integer arithmetic
    fn addi(lhs: (), rhs: ()) -> result {}
    fn subi(lhs: (), rhs: ()) -> result {}
    fn muli(lhs: (), rhs: ()) -> result {}
    fn divsi(lhs: (), rhs: ()) -> result {}
    fn divui(lhs: (), rhs: ()) -> result {}
    fn remsi(lhs: (), rhs: ()) -> result {}
    fn remui(lhs: (), rhs: ()) -> result {}
    fn negi(operand: ()) -> result {}

    // Float arithmetic
    fn addf(lhs: (), rhs: ()) -> result {}
    fn subf(lhs: (), rhs: ()) -> result {}
    fn mulf(lhs: (), rhs: ()) -> result {}
    fn divf(lhs: (), rhs: ()) -> result {}
    fn negf(operand: ()) -> result {}

    // Comparisons
    #[attr(predicate: Symbol)]
    fn cmpi(lhs: (), rhs: ()) -> result {}

    #[attr(predicate: Symbol)]
    fn cmpf(lhs: (), rhs: ()) -> result {}

    // Bitwise (integer-only)
    fn and(lhs: (), rhs: ()) -> result {}
    fn or(lhs: (), rhs: ()) -> result {}
    fn xor(lhs: (), rhs: ()) -> result {}
    fn shl(value: (), amount: ()) -> result {}
    fn shr(value: (), amount: ()) -> result {}
    fn shru(value: (), amount: ()) -> result {}

    // Conversions
    fn cast(operand: ()) -> result {}
    fn trunc(operand: ()) -> result {}
    fn extend(operand: ()) -> result {}
    fn convert(operand: ()) -> result {}
}
