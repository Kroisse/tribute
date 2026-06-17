//! Arena-based conversion target.
//!
//! Defines legality rules for dialect conversion: which operations/dialects
//! are legal, illegal, or dynamically checked.

use std::collections::HashSet;

use crate::context::IrContext;
use crate::refs::OpRef;
use crate::symbol::Symbol;
use crate::walk;

/// Result of a legality check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegalityCheck {
    /// The operation is legal (no conversion needed).
    Legal,
    /// The operation is illegal (must be converted).
    Illegal,
    /// The target has no rule for this operation.
    Unknown,
}

/// Dynamic legality check function signature.
type DynamicCheckFn = dyn Fn(&IrContext, OpRef) -> Option<LegalityCheck>;

/// Conversion verification mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversionMode {
    /// Verify that no explicitly illegal operations remain.
    ///
    /// Unknown operations are allowed because another conversion pass may own
    /// them.
    Partial,
    /// Verify that every operation is explicitly legal.
    ///
    /// Illegal and unknown operations both fail full conversion.
    Full,
}

/// Conversion target — defines which ops/dialects are legal or illegal.
///
/// After pattern application, partial verification checks that no illegal
/// operations remain. Full verification additionally rejects unknown operations.
pub struct ConversionTarget {
    /// Entire dialects marked as legal.
    legal_dialects: HashSet<Symbol>,
    /// Entire dialects marked as illegal.
    illegal_dialects: HashSet<Symbol>,
    /// Specific operations marked as legal: (dialect, op_name).
    legal_ops: HashSet<(Symbol, Symbol)>,
    /// Specific operations marked as illegal: (dialect, op_name).
    illegal_ops: HashSet<(Symbol, Symbol)>,
    /// Dynamic legality checks for specific operations.
    dynamic_checks: Vec<Box<DynamicCheckFn>>,
}

impl ConversionTarget {
    /// Create a new empty conversion target.
    ///
    /// Operations that do not match any target rule are reported as
    /// [`LegalityCheck::Unknown`].
    pub fn new() -> Self {
        Self {
            legal_dialects: HashSet::new(),
            illegal_dialects: HashSet::new(),
            legal_ops: HashSet::new(),
            illegal_ops: HashSet::new(),
            dynamic_checks: Vec::new(),
        }
    }

    /// Mark an entire dialect as legal.
    pub fn add_legal_dialect(&mut self, dialect: &str) {
        self.legal_dialects.insert(Symbol::from_dynamic(dialect));
    }

    /// Mark an entire dialect as illegal.
    pub fn add_illegal_dialect(&mut self, dialect: &str) {
        self.illegal_dialects.insert(Symbol::from_dynamic(dialect));
    }

    /// Mark a specific operation as legal.
    pub fn add_legal_op(&mut self, dialect: &str, op_name: &str) {
        self.legal_ops
            .insert((Symbol::from_dynamic(dialect), Symbol::from_dynamic(op_name)));
    }

    /// Mark a specific operation as illegal.
    pub fn add_illegal_op(&mut self, dialect: &str, op_name: &str) {
        self.illegal_ops
            .insert((Symbol::from_dynamic(dialect), Symbol::from_dynamic(op_name)));
    }

    /// Add a dynamic legality check.
    ///
    /// Return `Some(Legal)` or `Some(Illegal)` to override, `None` to defer.
    pub fn add_dynamic_check(
        &mut self,
        f: impl Fn(&IrContext, OpRef) -> Option<LegalityCheck> + 'static,
    ) {
        self.dynamic_checks.push(Box::new(f));
    }

    /// Check if this target has any constraints (legal/illegal dialects/ops/checks).
    pub fn has_constraints(&self) -> bool {
        !self.legal_dialects.is_empty()
            || !self.illegal_dialects.is_empty()
            || !self.legal_ops.is_empty()
            || !self.illegal_ops.is_empty()
            || !self.dynamic_checks.is_empty()
    }

    /// Check if a specific operation is legal.
    ///
    /// Resolution order:
    /// 1. Dynamic checks (first non-None wins)
    /// 2. Specific op rules (legal_ops / illegal_ops)
    /// 3. Dialect rules (legal_dialects / illegal_dialects)
    /// 4. Default: Unknown
    pub fn is_legal(&self, ctx: &IrContext, op: OpRef) -> LegalityCheck {
        // 1. Dynamic checks
        for check in &self.dynamic_checks {
            if let Some(result) = check(ctx, op) {
                return result;
            }
        }

        let data = ctx.op(op);
        let key = (data.dialect, data.name);

        // 2. Specific op rules
        if self.legal_ops.contains(&key) {
            return LegalityCheck::Legal;
        }
        if self.illegal_ops.contains(&key) {
            return LegalityCheck::Illegal;
        }

        // 3. Dialect rules
        if self.legal_dialects.contains(&data.dialect) {
            return LegalityCheck::Legal;
        }
        if self.illegal_dialects.contains(&data.dialect) {
            return LegalityCheck::Illegal;
        }

        // 4. Default
        LegalityCheck::Unknown
    }

    /// Verify that no illegal operations remain in the module.
    ///
    /// This is partial conversion verification: unknown operations are allowed
    /// because they may belong to dialects outside the current conversion step.
    pub fn verify(&self, ctx: &IrContext, module_body: crate::refs::RegionRef) -> Vec<IllegalOp> {
        self.verify_mode(ctx, module_body, ConversionMode::Partial)
    }

    /// Verify that every operation in the module is legal for this target.
    pub fn verify_full(
        &self,
        ctx: &IrContext,
        module_body: crate::refs::RegionRef,
    ) -> Vec<IllegalOp> {
        self.verify_mode(ctx, module_body, ConversionMode::Full)
    }

    /// Verify a module body under the requested conversion mode.
    pub fn verify_mode(
        &self,
        ctx: &IrContext,
        module_body: crate::refs::RegionRef,
        mode: ConversionMode,
    ) -> Vec<IllegalOp> {
        let mut failures = Vec::new();

        let _ = walk::walk_region::<()>(ctx, module_body, &mut |op| {
            let legality = self.is_legal(ctx, op);
            let failed = match mode {
                ConversionMode::Partial => legality == LegalityCheck::Illegal,
                ConversionMode::Full => legality != LegalityCheck::Legal,
            };
            if failed {
                let data = ctx.op(op);
                failures.push(IllegalOp {
                    op,
                    dialect: data.dialect,
                    name: data.name,
                    legality,
                });
            }
            std::ops::ControlFlow::Continue(walk::WalkAction::Advance)
        });

        failures
    }
}

impl Default for ConversionTarget {
    fn default() -> Self {
        Self::new()
    }
}

/// An operation that failed conversion target verification.
#[derive(Debug)]
pub struct IllegalOp {
    pub op: OpRef,
    pub dialect: Symbol,
    pub name: Symbol,
    pub legality: LegalityCheck,
}

impl std::fmt::Display for IllegalOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}.{} ({}, {:?})",
            self.dialect, self.name, self.op, self.legality
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OperationDataBuilder;
    use crate::location::Span;
    use crate::types::Location;

    fn test_ctx() -> (IrContext, Location) {
        let mut ctx = IrContext::new();
        let path = ctx.paths.intern("test.trb".to_owned());
        let loc = Location::new(path, Span::new(0, 0));
        (ctx, loc)
    }

    fn make_op(ctx: &mut IrContext, loc: Location, dialect: Symbol, name: Symbol) -> OpRef {
        let op_data = OperationDataBuilder::new(loc, dialect, name).build(ctx);
        ctx.create_op(op_data)
    }

    fn make_region(ctx: &mut IrContext, loc: Location, ops: Vec<OpRef>) -> crate::refs::RegionRef {
        use crate::context::{BlockData, RegionData};
        use crate::smallvec::smallvec;

        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        for op in ops {
            ctx.push_op(block, op);
        }
        ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block],
            parent_op: None,
        })
    }

    #[test]
    fn unspecified_operations_are_unknown() {
        let (mut ctx, loc) = test_ctx();
        let op = make_op(&mut ctx, loc, Symbol::new("test"), Symbol::new("op"));
        let target = ConversionTarget::new();

        assert_eq!(target.is_legal(&ctx, op), LegalityCheck::Unknown);
    }

    #[test]
    fn partial_verification_allows_unknown_operations() {
        let (mut ctx, loc) = test_ctx();
        let op = make_op(&mut ctx, loc, Symbol::new("test"), Symbol::new("op"));
        let region = make_region(&mut ctx, loc, vec![op]);
        let target = ConversionTarget::new();

        assert!(target.verify(&ctx, region).is_empty());
    }

    #[test]
    fn full_verification_rejects_unknown_operations() {
        let (mut ctx, loc) = test_ctx();
        let op = make_op(&mut ctx, loc, Symbol::new("test"), Symbol::new("op"));
        let region = make_region(&mut ctx, loc, vec![op]);
        let target = ConversionTarget::new();

        let failures = target.verify_full(&ctx, region);
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].legality, LegalityCheck::Unknown);
    }
}
