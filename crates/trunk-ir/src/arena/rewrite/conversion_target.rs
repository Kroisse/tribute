//! Arena-based conversion target.
//!
//! Defines legality rules for dialect conversion: which operations/dialects
//! are legal, illegal, or dynamically checked.

use std::collections::HashSet;

use crate::arena::context::IrContext;
use crate::arena::refs::OpRef;
use crate::arena::walk;
use crate::ir::Symbol;

/// Result of a legality check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegalityCheck {
    /// The operation is legal (no conversion needed).
    Legal,
    /// The operation is illegal (must be converted).
    Illegal,
}

/// Dynamic legality check function signature.
type DynamicCheckFn = dyn Fn(&IrContext, OpRef) -> Option<LegalityCheck>;

/// Conversion target — defines which ops/dialects are legal or illegal.
///
/// After pattern application, `verify()` walks the module and checks that
/// no illegal operations remain.
pub struct ArenaConversionTarget {
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

impl ArenaConversionTarget {
    /// Create a new empty conversion target (everything is legal by default).
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

    /// Check if a specific operation is legal.
    ///
    /// Resolution order:
    /// 1. Dynamic checks (first non-None wins)
    /// 2. Specific op rules (legal_ops / illegal_ops)
    /// 3. Dialect rules (legal_dialects / illegal_dialects)
    /// 4. Default: Legal
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
        LegalityCheck::Legal
    }

    /// Verify that no illegal operations remain in the module.
    ///
    /// Returns a list of illegal operations found.
    pub fn verify(
        &self,
        ctx: &IrContext,
        module_body: crate::arena::refs::RegionRef,
    ) -> Vec<IllegalOp> {
        let mut illegal = Vec::new();

        let _ = walk::walk_region::<()>(ctx, module_body, &mut |op| {
            if self.is_legal(ctx, op) == LegalityCheck::Illegal {
                let data = ctx.op(op);
                illegal.push(IllegalOp {
                    op,
                    dialect: data.dialect,
                    name: data.name,
                });
            }
            std::ops::ControlFlow::Continue(walk::WalkAction::Advance)
        });

        illegal
    }
}

impl Default for ArenaConversionTarget {
    fn default() -> Self {
        Self::new()
    }
}

/// An illegal operation found during verification.
#[derive(Debug)]
pub struct IllegalOp {
    pub op: OpRef,
    pub dialect: Symbol,
    pub name: Symbol,
}

impl std::fmt::Display for IllegalOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{} ({})", self.dialect, self.name, self.op)
    }
}
