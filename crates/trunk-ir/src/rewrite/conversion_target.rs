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
use std::sync::Arc;

use crate::dialect::core::Module;
use crate::{Operation, OperationWalk, Symbol, WalkAction};

/// Result of a dynamic legality check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegalityCheck {
    /// The operation is definitely legal.
    Legal,
    /// The operation is definitely illegal.
    Illegal,
    /// This check cannot determine legality; continue to the next rule.
    Continue,
}

/// Trait for dynamic legality check functions.
///
/// This trait is automatically implemented for closures with the appropriate signature.
pub trait DynamicLegalityCheckFn:
    for<'db> Fn(&'db dyn salsa::Database, Operation<'db>) -> LegalityCheck + Send + Sync
{
}

impl<F> DynamicLegalityCheckFn for F where
    F: for<'db> Fn(&'db dyn salsa::Database, Operation<'db>) -> LegalityCheck + Send + Sync
{
}

/// Specifies legality constraints for IR transformation passes.
///
/// A `ConversionTarget` declares which dialects are legal (allowed to remain
/// after conversion) and which are illegal (must be fully converted).
///
/// Legality is determined in the following order:
/// 1. Dynamic checks (most specific)
/// 2. Explicit operation rules (`legal_ops`, `illegal_ops`)
/// 3. Dialect rules (`legal_dialects`, `illegal_dialects`)
/// 4. Default: legal
#[derive(Clone, Default)]
pub struct ConversionTarget {
    /// Dialects that are legal (allowed to remain after conversion).
    legal_dialects: HashSet<Symbol>,
    /// Dialects that are illegal (must be fully converted).
    illegal_dialects: HashSet<Symbol>,
    /// Operations that are explicitly legal regardless of dialect.
    legal_ops: HashSet<(Symbol, Symbol)>,
    /// Operations that are explicitly illegal regardless of dialect.
    illegal_ops: HashSet<(Symbol, Symbol)>,
    /// Dynamic legality checks that can inspect operation properties.
    /// Uses `Arc` to allow cloning while keeping the trait objects.
    dynamic_checks: Vec<Arc<dyn DynamicLegalityCheckFn>>,
}

impl std::fmt::Debug for ConversionTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConversionTarget")
            .field("legal_dialects", &self.legal_dialects)
            .field("illegal_dialects", &self.illegal_dialects)
            .field("legal_ops", &self.legal_ops)
            .field("illegal_ops", &self.illegal_ops)
            .field(
                "dynamic_checks",
                &format!("[{} checks]", self.dynamic_checks.len()),
            )
            .finish()
    }
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

    /// Add a dynamic legality check that can inspect operation properties.
    ///
    /// Dynamic checks are evaluated first, before static rules. They can return:
    /// - `LegalityCheck::Legal` - operation is definitely legal
    /// - `LegalityCheck::Illegal` - operation is definitely illegal
    /// - `LegalityCheck::Continue` - this check cannot determine, continue to next rule
    ///
    /// # Example
    ///
    /// ```ignore
    /// let target = ConversionTarget::new()
    ///     .add_dynamic_check(|db, op| {
    ///         // Check if any result type is illegal
    ///         for result_ty in op.results(db).iter() {
    ///             if is_illegal_type(db, *result_ty) {
    ///                 return LegalityCheck::Illegal;
    ///             }
    ///         }
    ///         LegalityCheck::Continue
    ///     });
    /// ```
    pub fn add_dynamic_check<F>(mut self, check: F) -> Self
    where
        F: for<'db> Fn(&'db dyn salsa::Database, Operation<'db>) -> LegalityCheck
            + Send
            + Sync
            + 'static,
    {
        self.dynamic_checks.push(Arc::new(check));
        self
    }

    /// Check if an operation is legal, considering dynamic checks.
    ///
    /// Resolution order:
    /// 1. Dynamic checks (first match wins)
    /// 2. Explicit op legality (`legal_ops`/`illegal_ops`)
    /// 3. Dialect legality (`legal_dialects`/`illegal_dialects`)
    /// 4. Default: legal
    pub fn is_legal_op<'db>(&self, db: &'db dyn salsa::Database, op: Operation<'db>) -> bool {
        let dialect = op.dialect(db);
        let name = op.name(db);

        // 1. Check dynamic predicates first
        for check in &self.dynamic_checks {
            match check(db, op) {
                LegalityCheck::Legal => return true,
                LegalityCheck::Illegal => return false,
                LegalityCheck::Continue => continue,
            }
        }

        // 2-4. Fall back to static rules
        self.is_legal(dialect, name)
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
    ///
    /// This method uses `is_legal_op()` to check each operation, which includes
    /// dynamic legality checks.
    pub fn find_illegal_ops<'db>(
        &self,
        db: &'db dyn salsa::Database,
        module: &Module<'db>,
    ) -> Vec<IllegalOp> {
        use std::ops::ControlFlow;

        let mut illegal = Vec::new();
        let body = module.body(db);
        let _ = body.walk_all::<()>(db, |op: Operation<'db>| {
            if !self.is_legal_op(db, op) {
                let dialect = op.dialect(db);
                let name = op.name(db);
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
            || !self.dynamic_checks.is_empty()
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
        let prompt_tag_ty = cont::PromptTag::new(db).as_type();
        let ability_ref_ty =
            core::AbilityRefType::with_params(db, Symbol::new("State"), IdVec::from(vec![i32_ty]))
                .as_type();

        // Create a tag constant
        let tag_const = arith::r#const(db, location, prompt_tag_ty, crate::Attribute::IntBits(0));
        let tag_val = tag_const.result(db);

        // Create cont.shift which should be illegal
        let handler_region = Region::new(db, location, IdVec::new());
        let shift_op = cont::shift(
            db,
            location,
            tag_val,
            vec![],
            i32_ty,
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
            IdVec::from(vec![tag_const.as_operation(), shift_op]),
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

    #[salsa_test]
    fn test_dynamic_check_illegal(db: &salsa::DatabaseImpl) {
        let module = make_legal_module(db);

        // Mark all arith.const with value 42 as illegal using dynamic check
        let target = ConversionTarget::new().add_dynamic_check(|db, op| {
            if let Ok(const_op) = arith::Const::from_operation(db, op)
                && const_op.value(db) == crate::Attribute::IntBits(42)
            {
                return LegalityCheck::Illegal;
            }
            LegalityCheck::Continue
        });

        let illegal = target.find_illegal_ops(db, &module);
        assert_eq!(illegal.len(), 1);
        assert_eq!(illegal[0].dialect, "arith");
        assert_eq!(illegal[0].name, "const");
    }

    #[salsa_test]
    fn test_dynamic_check_legal_overrides_static_illegal(db: &salsa::DatabaseImpl) {
        let module = make_illegal_module(db);

        // Make cont dialect illegal statically, but allow cont.shift dynamically
        let target = ConversionTarget::new()
            .illegal_dialect("cont")
            .add_dynamic_check(|db, op| {
                // Allow cont.shift explicitly
                if op.dialect(db) == Symbol::new("cont") && op.name(db) == Symbol::new("shift") {
                    return LegalityCheck::Legal;
                }
                LegalityCheck::Continue
            });

        let result = target.verify(db, &module);
        assert!(
            result.is_ok(),
            "Dynamic Legal should override static illegal"
        );
    }

    #[salsa_test]
    fn test_dynamic_check_continue_falls_through(db: &salsa::DatabaseImpl) {
        let module = make_illegal_module(db);

        // Dynamic check always returns Continue, so static rules apply
        let target = ConversionTarget::new()
            .illegal_dialect("cont")
            .add_dynamic_check(|_, _| LegalityCheck::Continue);

        let result = target.verify(db, &module);
        assert!(
            result.is_err(),
            "Continue should fall through to static rules"
        );
    }

    #[test]
    fn test_has_constraints_with_dynamic_check() {
        let with_dynamic =
            ConversionTarget::new().add_dynamic_check(|_, _| LegalityCheck::Continue);
        assert!(with_dynamic.has_constraints());
    }

    #[salsa::tracked]
    fn make_const_42_op(db: &dyn salsa::Database) -> Operation<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();
        arith::r#const(db, location, i32_ty, crate::Attribute::IntBits(42)).as_operation()
    }

    #[salsa_test]
    fn test_is_legal_op_with_dynamic_check(db: &salsa::DatabaseImpl) {
        let op = make_const_42_op(db);

        // Dynamic check marks value 42 as illegal
        let target = ConversionTarget::new().add_dynamic_check(|db, op| {
            if let Ok(const_op) = arith::Const::from_operation(db, op)
                && const_op.value(db) == crate::Attribute::IntBits(42)
            {
                return LegalityCheck::Illegal;
            }
            LegalityCheck::Continue
        });

        assert!(!target.is_legal_op(db, op));
    }
}
