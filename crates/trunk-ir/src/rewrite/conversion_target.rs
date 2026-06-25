//! Arena-based conversion target.
//!
//! Defines legality rules for dialect conversion: which operations/dialects
//! are legal, illegal, or dynamically checked.

use std::collections::{HashMap, HashSet, hash_map::Entry};

use derive_more::Error;

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

/// Decision returned by one dynamic legality rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegalityDecision {
    /// The operation is legal at this rule tier.
    Legal,
    /// The operation is illegal at this rule tier.
    Illegal,
    /// Defer to the next lower-precedence legality rule.
    Defer,
}

impl LegalityDecision {
    fn into_check(self) -> Option<LegalityCheck> {
        match self {
            Self::Legal => Some(LegalityCheck::Legal),
            Self::Illegal => Some(LegalityCheck::Illegal),
            Self::Defer => None,
        }
    }
}

/// Dynamic legality check function signature.
type DynamicCheckFn = dyn Fn(&IrContext, OpRef) -> LegalityDecision;

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
    /// Dynamic legality checks for specific operations: (dialect, op_name).
    dynamic_ops: HashMap<(Symbol, Symbol), Box<DynamicCheckFn>>,
    /// Dynamic legality checks for entire dialects.
    dynamic_dialects: HashMap<Symbol, Box<DynamicCheckFn>>,
    /// Dynamic fallback for operations not decided by op or dialect rules.
    dynamic_unknown: Option<Box<DynamicCheckFn>>,
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
            dynamic_ops: HashMap::new(),
            dynamic_dialects: HashMap::new(),
            dynamic_unknown: None,
        }
    }

    /// Mark an entire dialect as legal and return the updated target.
    pub fn legal_dialect(mut self, dialect: &str) -> Self {
        self.add_legal_dialect(dialect);
        self
    }

    /// Mark an entire dialect as illegal and return the updated target.
    pub fn illegal_dialect(mut self, dialect: &str) -> Self {
        self.add_illegal_dialect(dialect);
        self
    }

    /// Mark a specific operation as legal and return the updated target.
    pub fn legal_op(mut self, dialect: &str, op_name: &str) -> Self {
        self.add_legal_op(dialect, op_name);
        self
    }

    /// Mark a specific operation as illegal and return the updated target.
    pub fn illegal_op(mut self, dialect: &str, op_name: &str) -> Self {
        self.add_illegal_op(dialect, op_name);
        self
    }

    /// Add a dynamic legality check for a specific operation and return the updated target.
    pub fn dynamic_op(
        mut self,
        dialect: &str,
        op_name: &str,
        f: impl Fn(&IrContext, OpRef) -> LegalityDecision + 'static,
    ) -> Self {
        self.add_dynamic_op(dialect, op_name, f);
        self
    }

    /// Add a dynamic legality check for an entire dialect and return the updated target.
    pub fn dynamic_dialect(
        mut self,
        dialect: &str,
        f: impl Fn(&IrContext, OpRef) -> LegalityDecision + 'static,
    ) -> Self {
        self.add_dynamic_dialect(dialect, f);
        self
    }

    /// Add a dynamic fallback for operations not decided by op or dialect rules.
    pub fn dynamic_unknown(
        mut self,
        f: impl Fn(&IrContext, OpRef) -> LegalityDecision + 'static,
    ) -> Self {
        self.add_dynamic_unknown(f);
        self
    }

    /// Mark an entire dialect as legal.
    pub fn add_legal_dialect(&mut self, dialect: &str) {
        let dialect = Symbol::from_dynamic(dialect);
        assert!(
            !self.illegal_dialects.contains(&dialect),
            "dialect `{dialect}` is already registered as illegal"
        );
        assert!(
            self.legal_dialects.insert(dialect),
            "dialect `{dialect}` is already registered as legal"
        );
    }

    /// Mark an entire dialect as illegal.
    pub fn add_illegal_dialect(&mut self, dialect: &str) {
        let dialect = Symbol::from_dynamic(dialect);
        assert!(
            !self.legal_dialects.contains(&dialect),
            "dialect `{dialect}` is already registered as legal"
        );
        assert!(
            self.illegal_dialects.insert(dialect),
            "dialect `{dialect}` is already registered as illegal"
        );
    }

    /// Mark a specific operation as legal.
    pub fn add_legal_op(&mut self, dialect: &str, op_name: &str) {
        let key = (Symbol::from_dynamic(dialect), Symbol::from_dynamic(op_name));
        assert!(
            !self.illegal_ops.contains(&key),
            "operation `{}.{}` is already registered as illegal",
            key.0,
            key.1
        );
        assert!(
            self.legal_ops.insert(key),
            "operation `{}.{}` is already registered as legal",
            key.0,
            key.1
        );
    }

    /// Mark a specific operation as illegal.
    pub fn add_illegal_op(&mut self, dialect: &str, op_name: &str) {
        let key = (Symbol::from_dynamic(dialect), Symbol::from_dynamic(op_name));
        assert!(
            !self.legal_ops.contains(&key),
            "operation `{}.{}` is already registered as legal",
            key.0,
            key.1
        );
        assert!(
            self.illegal_ops.insert(key),
            "operation `{}.{}` is already registered as illegal",
            key.0,
            key.1
        );
    }

    /// Add a dynamic legality check for a specific operation.
    ///
    /// Return `Legal` or `Illegal` to decide, `Defer` to continue to the static
    /// operation rule and then lower-precedence tiers.
    pub fn add_dynamic_op(
        &mut self,
        dialect: &str,
        op_name: &str,
        f: impl Fn(&IrContext, OpRef) -> LegalityDecision + 'static,
    ) {
        let key = (Symbol::from_dynamic(dialect), Symbol::from_dynamic(op_name));
        match self.dynamic_ops.entry(key) {
            Entry::Vacant(entry) => {
                entry.insert(Box::new(f));
            }
            Entry::Occupied(entry) => {
                let key = entry.key();
                panic!(
                    "operation `{}.{}` already has a dynamic legality rule",
                    key.0, key.1
                );
            }
        }
    }

    /// Add a dynamic legality check for an entire dialect.
    ///
    /// Return `Legal` or `Illegal` to decide, `Defer` to continue to the static
    /// dialect rule and then lower-precedence tiers.
    pub fn add_dynamic_dialect(
        &mut self,
        dialect: &str,
        f: impl Fn(&IrContext, OpRef) -> LegalityDecision + 'static,
    ) {
        let dialect = Symbol::from_dynamic(dialect);
        match self.dynamic_dialects.entry(dialect) {
            Entry::Vacant(entry) => {
                entry.insert(Box::new(f));
            }
            Entry::Occupied(entry) => {
                panic!(
                    "dialect `{}` already has a dynamic legality rule",
                    entry.key()
                );
            }
        }
    }

    /// Add a dynamic fallback for operations not decided by operation or dialect rules.
    ///
    /// Return `Legal` or `Illegal` to decide, `Defer` to leave the operation unknown.
    pub fn add_dynamic_unknown(
        &mut self,
        f: impl Fn(&IrContext, OpRef) -> LegalityDecision + 'static,
    ) {
        assert!(
            self.dynamic_unknown.is_none(),
            "unknown operations already have a dynamic legality rule"
        );
        self.dynamic_unknown = Some(Box::new(f));
    }

    /// Check if this target has any constraints (legal/illegal dialects/ops/checks).
    pub fn has_constraints(&self) -> bool {
        !self.legal_dialects.is_empty()
            || !self.illegal_dialects.is_empty()
            || !self.legal_ops.is_empty()
            || !self.illegal_ops.is_empty()
            || !self.dynamic_ops.is_empty()
            || !self.dynamic_dialects.is_empty()
            || self.dynamic_unknown.is_some()
    }

    /// Check if a specific operation is legal.
    ///
    /// Resolution order:
    /// 1. Operation dynamic rule.
    /// 2. Operation static rule.
    /// 3. Dialect dynamic rule.
    /// 4. Dialect static rule.
    /// 5. Unknown-operation dynamic fallback.
    /// 6. Default: Unknown.
    pub fn is_legal(&self, ctx: &IrContext, op: OpRef) -> LegalityCheck {
        let data = ctx.op(op);
        let key = (data.dialect, data.name);

        // 1. Operation dynamic rule.
        if let Some(check) = self.dynamic_ops.get(&key)
            && let Some(result) = check(ctx, op).into_check()
        {
            return result;
        }

        // 2. Specific op rules
        if self.legal_ops.contains(&key) {
            return LegalityCheck::Legal;
        }
        if self.illegal_ops.contains(&key) {
            return LegalityCheck::Illegal;
        }

        // 3. Dialect dynamic rule.
        if let Some(check) = self.dynamic_dialects.get(&data.dialect)
            && let Some(result) = check(ctx, op).into_check()
        {
            return result;
        }

        // 4. Dialect rules
        if self.legal_dialects.contains(&data.dialect) {
            return LegalityCheck::Legal;
        }
        if self.illegal_dialects.contains(&data.dialect) {
            return LegalityCheck::Illegal;
        }

        // 5. Unknown-operation dynamic fallback.
        if let Some(check) = &self.dynamic_unknown
            && let Some(result) = check(ctx, op).into_check()
        {
            return result;
        }

        // 6. Default
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

/// Error at a named conversion boundary.
#[derive(Debug, Error)]
pub struct ConversionError {
    boundary: &'static str,
    operations: Vec<IllegalOp>,
}

impl ConversionError {
    pub fn new(boundary: &'static str, operations: Vec<IllegalOp>) -> Self {
        Self {
            boundary,
            operations,
        }
    }

    pub fn boundary(&self) -> &'static str {
        self.boundary
    }

    pub fn operations(&self) -> &[IllegalOp] {
        &self.operations
    }
}

impl std::fmt::Display for ConversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "conversion boundary `{}` failed with {} error(s):",
            self.boundary,
            self.operations.len()
        )?;
        for operation in &self.operations {
            writeln!(f, "  - {operation}")?;
        }
        Ok(())
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
    fn fluent_rules_preserve_operation_precedence() {
        let (mut ctx, loc) = test_ctx();
        let allowed = make_op(&mut ctx, loc, Symbol::new("test"), Symbol::new("allowed"));
        let rejected = make_op(&mut ctx, loc, Symbol::new("test"), Symbol::new("rejected"));
        let target = ConversionTarget::new()
            .illegal_dialect("test")
            .legal_op("test", "allowed");

        assert_eq!(target.is_legal(&ctx, allowed), LegalityCheck::Legal);
        assert_eq!(target.is_legal(&ctx, rejected), LegalityCheck::Illegal);
    }

    #[test]
    fn dynamic_registration_order_does_not_change_structural_precedence() {
        let (mut ctx, loc) = test_ctx();
        let special = make_op(&mut ctx, loc, Symbol::new("test"), Symbol::new("special"));
        let ordinary = make_op(&mut ctx, loc, Symbol::new("test"), Symbol::new("ordinary"));

        let op_then_dialect = ConversionTarget::new()
            .dynamic_op("test", "special", |_, _| LegalityDecision::Illegal)
            .dynamic_dialect("test", |_, _| LegalityDecision::Legal);
        let dialect_then_op = ConversionTarget::new()
            .dynamic_dialect("test", |_, _| LegalityDecision::Legal)
            .dynamic_op("test", "special", |_, _| LegalityDecision::Illegal);

        assert_eq!(
            op_then_dialect.is_legal(&ctx, special),
            LegalityCheck::Illegal
        );
        assert_eq!(
            dialect_then_op.is_legal(&ctx, special),
            LegalityCheck::Illegal
        );
        assert_eq!(
            op_then_dialect.is_legal(&ctx, ordinary),
            LegalityCheck::Legal
        );
        assert_eq!(
            dialect_then_op.is_legal(&ctx, ordinary),
            LegalityCheck::Legal
        );
    }

    #[test]
    fn dynamic_operation_rule_precedes_static_operation_rule() {
        let (mut ctx, loc) = test_ctx();
        let op = make_op(&mut ctx, loc, Symbol::new("test"), Symbol::new("op"));
        let target =
            ConversionTarget::new()
                .legal_op("test", "op")
                .dynamic_op("test", "op", |_, _| LegalityDecision::Illegal);

        assert_eq!(target.is_legal(&ctx, op), LegalityCheck::Illegal);
    }

    #[test]
    fn dynamic_operation_rule_can_defer_to_static_operation_rule() {
        let (mut ctx, loc) = test_ctx();
        let op = make_op(&mut ctx, loc, Symbol::new("test"), Symbol::new("op"));
        let target = ConversionTarget::new()
            .illegal_dialect("test")
            .legal_op("test", "op")
            .dynamic_op("test", "op", |_, _| LegalityDecision::Defer);

        assert_eq!(target.is_legal(&ctx, op), LegalityCheck::Legal);
    }

    #[test]
    fn operation_rule_precedes_dynamic_dialect_rule() {
        let (mut ctx, loc) = test_ctx();
        let op = make_op(&mut ctx, loc, Symbol::new("test"), Symbol::new("op"));
        let target = ConversionTarget::new()
            .legal_op("test", "op")
            .dynamic_dialect("test", |_, _| LegalityDecision::Illegal);

        assert_eq!(target.is_legal(&ctx, op), LegalityCheck::Legal);
    }

    #[test]
    fn dynamic_dialect_rule_can_defer_to_static_dialect_rule() {
        let (mut ctx, loc) = test_ctx();
        let op = make_op(&mut ctx, loc, Symbol::new("test"), Symbol::new("op"));
        let target = ConversionTarget::new()
            .illegal_dialect("test")
            .dynamic_dialect("test", |_, _| LegalityDecision::Defer);

        assert_eq!(target.is_legal(&ctx, op), LegalityCheck::Illegal);
    }

    #[test]
    fn unknown_dynamic_fallback_handles_otherwise_unknown_operations() {
        let (mut ctx, loc) = test_ctx();
        let op = make_op(&mut ctx, loc, Symbol::new("test"), Symbol::new("op"));
        let target = ConversionTarget::new().dynamic_unknown(|_, _| LegalityDecision::Legal);

        assert_eq!(target.is_legal(&ctx, op), LegalityCheck::Legal);
    }

    #[test]
    fn unknown_dynamic_fallback_does_not_override_static_rules() {
        let (mut ctx, loc) = test_ctx();
        let op = make_op(&mut ctx, loc, Symbol::new("test"), Symbol::new("op"));
        let target = ConversionTarget::new()
            .illegal_dialect("test")
            .dynamic_unknown(|_, _| LegalityDecision::Legal);

        assert_eq!(target.is_legal(&ctx, op), LegalityCheck::Illegal);
    }

    #[test]
    fn unknown_dynamic_fallback_can_defer_to_unknown() {
        let (mut ctx, loc) = test_ctx();
        let op = make_op(&mut ctx, loc, Symbol::new("test"), Symbol::new("op"));
        let target = ConversionTarget::new().dynamic_unknown(|_, _| LegalityDecision::Defer);

        assert_eq!(target.is_legal(&ctx, op), LegalityCheck::Unknown);
    }

    #[test]
    #[should_panic(expected = "already registered as legal")]
    fn conflicting_static_operation_registration_panics() {
        let mut target = ConversionTarget::new();
        target.add_legal_op("test", "op");
        target.add_illegal_op("test", "op");
    }

    #[test]
    #[should_panic(expected = "already has a dynamic legality rule")]
    fn duplicate_dynamic_operation_registration_panics() {
        let mut target = ConversionTarget::new();
        target.add_dynamic_op("test", "op", |_, _| LegalityDecision::Legal);
        target.add_dynamic_op("test", "op", |_, _| LegalityDecision::Illegal);
    }

    #[test]
    #[should_panic(expected = "unknown operations already have a dynamic legality rule")]
    fn duplicate_unknown_dynamic_registration_panics() {
        let mut target = ConversionTarget::new();
        target.add_dynamic_unknown(|_, _| LegalityDecision::Legal);
        target.add_dynamic_unknown(|_, _| LegalityDecision::Illegal);
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
