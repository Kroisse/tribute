//! Conversion target for IR transformation passes.
//!
//! Provides a declarative way to specify which dialects are legal (allowed to remain)
//! and which must be fully converted during a pass. Inspired by MLIR's `ConversionTarget`.
//!
//! # Example
//!
//! ```ignore
//! use trunk_ir::rewrite::ConversionTarget;
//!
//! let target = ConversionTarget::new()
//!     .legal_dialect("trampoline")
//!     .legal_dialect("func")
//!     .illegal_dialect("cont");  // All cont.* ops must be converted
//!
//! // After applying patterns, verify no illegal ops remain
//! target.verify(&module)?;
//! ```

use std::collections::HashSet;

use crate::dialect::core::Module;
use crate::{Operation, OperationWalk, Symbol, WalkAction};

/// Specifies legality constraints for IR transformation passes.
///
/// A `ConversionTarget` declares which dialects are legal (allowed to remain
/// after conversion) and which are illegal (must be fully converted).
#[derive(Debug, Clone, Default)]
pub struct ConversionTarget {
    /// Dialects that are legal (allowed to remain after conversion).
    legal_dialects: HashSet<Symbol>,
    /// Dialects that are illegal (must be fully converted).
    illegal_dialects: HashSet<Symbol>,
    /// Operations that are explicitly legal regardless of dialect.
    legal_ops: HashSet<(Symbol, Symbol)>,
    /// Operations that are explicitly illegal regardless of dialect.
    illegal_ops: HashSet<(Symbol, Symbol)>,
}

/// Error returned when verification fails.
#[derive(Debug, Clone, PartialEq, Eq, salsa::Update)]
pub struct ConversionError {
    /// List of illegal operations found.
    pub illegal_ops: Vec<IllegalOp>,
}

/// Information about an illegal operation found during verification.
#[derive(Debug, Clone, PartialEq, Eq, salsa::Update)]
pub struct IllegalOp {
    /// The dialect name of the illegal operation.
    pub dialect: String,
    /// The operation name of the illegal operation.
    pub name: String,
}

impl std::fmt::Display for ConversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Conversion failed: {} illegal operations remain:",
            self.illegal_ops.len()
        )?;
        for op in &self.illegal_ops {
            writeln!(f, "  - {}.{}", op.dialect, op.name)?;
        }
        Ok(())
    }
}

impl std::error::Error for ConversionError {}

impl ConversionTarget {
    /// Create a new empty conversion target.
    ///
    /// By default, all operations are considered legal (no constraints).
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark a dialect as legal (operations from this dialect may remain).
    pub fn legal_dialect(mut self, dialect: impl Into<Symbol>) -> Self {
        let dialect = dialect.into();
        self.legal_dialects.insert(dialect);
        self.illegal_dialects.remove(&dialect);
        self
    }

    /// Mark a dialect as illegal (all operations must be converted).
    pub fn illegal_dialect(mut self, dialect: impl Into<Symbol>) -> Self {
        let dialect = dialect.into();
        self.illegal_dialects.insert(dialect);
        self.legal_dialects.remove(&dialect);
        self
    }

    /// Mark a specific operation as legal regardless of its dialect's legality.
    pub fn legal_op(mut self, dialect: impl Into<Symbol>, name: impl Into<Symbol>) -> Self {
        let key = (dialect.into(), name.into());
        self.legal_ops.insert(key);
        self.illegal_ops.remove(&key);
        self
    }

    /// Mark a specific operation as illegal regardless of its dialect's legality.
    pub fn illegal_op(mut self, dialect: impl Into<Symbol>, name: impl Into<Symbol>) -> Self {
        let key = (dialect.into(), name.into());
        self.illegal_ops.insert(key);
        self.legal_ops.remove(&key);
        self
    }

    /// Check if an operation is legal according to this target.
    ///
    /// Resolution order:
    /// 1. Check explicit op legality (legal_ops/illegal_ops)
    /// 2. Check dialect legality (legal_dialects/illegal_dialects)
    /// 3. If neither specified, operation is considered legal
    pub fn is_legal(&self, dialect: Symbol, name: Symbol) -> bool {
        // 1. Check explicit op-level rules first
        if self.legal_ops.contains(&(dialect, name)) {
            return true;
        }
        if self.illegal_ops.contains(&(dialect, name)) {
            return false;
        }

        // 2. Check dialect-level rules
        if self.legal_dialects.contains(&dialect) {
            return true;
        }
        if self.illegal_dialects.contains(&dialect) {
            return false;
        }

        // 3. Default: legal if no rules specified
        true
    }

    /// Check if an operation is illegal according to this target.
    pub fn is_illegal(&self, dialect: Symbol, name: Symbol) -> bool {
        !self.is_legal(dialect, name)
    }

    /// Verify that a module contains no illegal operations.
    ///
    /// Returns `Ok(())` if all operations are legal, or `Err(ConversionError)`
    /// with details about which illegal operations were found.
    pub fn verify<'db>(
        &self,
        db: &'db dyn salsa::Database,
        module: &Module<'db>,
    ) -> Result<(), ConversionError> {
        let illegal = self.find_illegal_ops(db, module);
        if illegal.is_empty() {
            Ok(())
        } else {
            Err(ConversionError {
                illegal_ops: illegal,
            })
        }
    }

    /// Find all illegal operations in a module.
    pub fn find_illegal_ops<'db>(
        &self,
        db: &'db dyn salsa::Database,
        module: &Module<'db>,
    ) -> Vec<IllegalOp> {
        use std::ops::ControlFlow;

        let mut illegal = Vec::new();
        let body = module.body(db);
        let _ = body.walk_all::<()>(db, |op: Operation<'db>| {
            let dialect = op.dialect(db);
            let name = op.name(db);
            if self.is_illegal(dialect, name) {
                illegal.push(IllegalOp {
                    dialect: dialect.to_string(),
                    name: name.to_string(),
                });
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
        illegal
    }

    /// Check if this target has any constraints defined.
    pub fn has_constraints(&self) -> bool {
        !self.legal_dialects.is_empty()
            || !self.illegal_dialects.is_empty()
            || !self.legal_ops.is_empty()
            || !self.illegal_ops.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialect::{arith, cont, core};
    use crate::ops::DialectOp;
    use crate::types::DialectType;
    use crate::{Block, BlockId, IdVec, Location, PathId, Region, Span, Symbol};
    use salsa_test_macros::salsa_test;

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    #[test]
    fn test_empty_target_allows_all() {
        let target = ConversionTarget::new();
        assert!(target.is_legal(Symbol::new("any"), Symbol::new("op")));
        assert!(target.is_legal(Symbol::new("cont"), Symbol::new("shift")));
    }

    #[test]
    fn test_legal_dialect() {
        let target = ConversionTarget::new()
            .legal_dialect("func")
            .illegal_dialect("cont");

        assert!(target.is_legal(Symbol::new("func"), Symbol::new("call")));
        assert!(target.is_legal(Symbol::new("func"), Symbol::new("return")));
        assert!(!target.is_legal(Symbol::new("cont"), Symbol::new("shift")));
        assert!(!target.is_legal(Symbol::new("cont"), Symbol::new("resume")));
        // Unspecified dialects are legal by default
        assert!(target.is_legal(Symbol::new("arith"), Symbol::new("add")));
    }

    #[test]
    fn test_op_override() {
        let target = ConversionTarget::new()
            .illegal_dialect("cont")
            .legal_op("cont", "drop"); // Allow cont.drop even though cont is illegal

        assert!(!target.is_legal(Symbol::new("cont"), Symbol::new("shift")));
        assert!(!target.is_legal(Symbol::new("cont"), Symbol::new("resume")));
        assert!(target.is_legal(Symbol::new("cont"), Symbol::new("drop"))); // Explicitly legal
    }

    #[test]
    fn test_illegal_op_override() {
        let target = ConversionTarget::new()
            .legal_dialect("func")
            .illegal_op("func", "deprecated_call");

        assert!(target.is_legal(Symbol::new("func"), Symbol::new("call")));
        assert!(!target.is_legal(Symbol::new("func"), Symbol::new("deprecated_call")));
    }

    #[test]
    fn test_dialect_toggle() {
        // Setting legal then illegal should make it illegal
        let target = ConversionTarget::new()
            .legal_dialect("cont")
            .illegal_dialect("cont");
        assert!(!target.is_legal(Symbol::new("cont"), Symbol::new("shift")));

        // Setting illegal then legal should make it legal
        let target = ConversionTarget::new()
            .illegal_dialect("cont")
            .legal_dialect("cont");
        assert!(target.is_legal(Symbol::new("cont"), Symbol::new("shift")));
    }

    #[test]
    fn test_has_constraints() {
        let empty = ConversionTarget::new();
        assert!(!empty.has_constraints());

        let with_legal = ConversionTarget::new().legal_dialect("func");
        assert!(with_legal.has_constraints());

        let with_illegal = ConversionTarget::new().illegal_dialect("cont");
        assert!(with_illegal.has_constraints());
    }

    /// Create a module with arith.const operations only (no illegal ops).
    #[salsa::tracked]
    fn make_legal_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        let op1 =
            arith::r#const(db, location, i32_ty, crate::Attribute::IntBits(42)).as_operation();
        let op2 =
            arith::r#const(db, location, i32_ty, crate::Attribute::IntBits(100)).as_operation();
        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            IdVec::new(),
            IdVec::from(vec![op1, op2]),
        );
        let region = Region::new(db, location, IdVec::from(vec![block]));
        Module::create(db, location, Symbol::new("test"), region)
    }

    /// Create a module with cont.shift operation (illegal op).
    #[salsa::tracked]
    fn make_illegal_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();
        let ability_ref_ty =
            core::AbilityRefType::with_params(db, Symbol::new("State"), IdVec::from(vec![i32_ty]))
                .as_type();

        // Create cont.shift which should be illegal
        let handler_region = Region::new(db, location, IdVec::new());
        let shift_op = cont::shift(
            db,
            location,
            vec![],
            i32_ty,
            0,
            ability_ref_ty,
            Symbol::new("get"),
            handler_region,
        )
        .as_operation();

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            IdVec::new(),
            IdVec::from(vec![shift_op]),
        );
        let region = Region::new(db, location, IdVec::from(vec![block]));
        Module::create(db, location, Symbol::new("test"), region)
    }

    #[salsa_test]
    fn test_verify_legal_module_passes(db: &salsa::DatabaseImpl) {
        let module = make_legal_module(db);
        let target = ConversionTarget::new()
            .legal_dialect("arith")
            .illegal_dialect("cont");

        let result = target.verify(db, &module);
        assert!(
            result.is_ok(),
            "Module with only legal ops should pass verification"
        );
    }

    #[salsa_test]
    fn test_verify_illegal_module_fails(db: &salsa::DatabaseImpl) {
        let module = make_illegal_module(db);
        let target = ConversionTarget::new()
            .legal_dialect("arith")
            .illegal_dialect("cont");

        let result = target.verify(db, &module);
        assert!(
            result.is_err(),
            "Module with illegal ops should fail verification"
        );

        let err = result.unwrap_err();
        assert_eq!(err.illegal_ops.len(), 1);
        assert_eq!(err.illegal_ops[0].dialect, "cont");
        assert_eq!(err.illegal_ops[0].name, "shift");
    }

    #[salsa_test]
    fn test_find_illegal_ops(db: &salsa::DatabaseImpl) {
        let module = make_illegal_module(db);
        let target = ConversionTarget::new().illegal_dialect("cont");

        let illegal = target.find_illegal_ops(db, &module);
        assert_eq!(illegal.len(), 1);
        assert_eq!(illegal[0].dialect, "cont");
        assert_eq!(illegal[0].name, "shift");
    }
}
