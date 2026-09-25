//! Registered operation definitions generated from `#[dialect]`.
//!
//! An [`OpDef`] bundles everything the dialect macro registers for one
//! operation: its declarative [`OpSchema`] and the hooks the declaration
//! opts into, such as `#[verify]`. Definitions are registered by operation
//! name so that verification and other tooling can look them up for any
//! operation.

use std::collections::HashMap;
use std::sync::LazyLock;

use crate::op_schema::{OpSchema, SchemaViolation};
use crate::{IrContext, OpRef, Symbol};

/// Everything `#[dialect]` registers for one operation.
#[derive(Debug)]
pub struct OpDef {
    /// Declarative constraints on the operation's entities.
    pub schema: OpSchema,
    /// Calls the wrapper's [`Verify`](crate::ops::Verify) impl for
    /// `#[verify]` operations.
    pub verifier: Option<OpVerifier>,
}

/// An operation-local verifier generated from `#[verify]`.
pub type OpVerifier = fn(&IrContext, OpRef) -> Result<(), String>;

impl OpDef {
    /// Run the declarative schema stages and then the `#[verify]` hook,
    /// stopping after the first stage that reports a violation.
    pub fn verify(&self, ctx: &IrContext, op: OpRef) -> Vec<SchemaViolation> {
        let violations = self.schema.verify(ctx, op);
        if !violations.is_empty() {
            return violations;
        }
        match self.verifier.map(|verifier| verifier(ctx, op)) {
            Some(Err(message)) => vec![SchemaViolation::Verifier(message)],
            _ => Vec::new(),
        }
    }

    /// Look up the registered definition for `dialect.name`.
    pub fn lookup(dialect: Symbol, name: Symbol) -> Option<&'static OpDef> {
        REGISTRY.get(&(dialect, name)).copied()
    }

    /// Look up the registered definition for an operation.
    pub fn of(ctx: &IrContext, op: OpRef) -> Option<&'static OpDef> {
        let data = ctx.op(op);
        Self::lookup(data.dialect, data.name)
    }
}

/// Inventory entry registering an [`OpDef`].
///
/// Emitted by the `#[dialect]` macro for every declared operation.
pub struct OpDefRegistration(pub &'static OpDef);

inventory::collect!(OpDefRegistration);

static REGISTRY: LazyLock<HashMap<(Symbol, Symbol), &'static OpDef>> = LazyLock::new(|| {
    let mut registry = HashMap::new();
    for OpDefRegistration(def) in inventory::iter::<OpDefRegistration> {
        let schema = &def.schema;
        let key = (
            Symbol::from_dynamic(schema.dialect),
            Symbol::from_dynamic(schema.name),
        );
        if let Some(previous) = registry.insert(key, *def) {
            panic!(
                "operation {}.{} is registered twice ({previous:p} and {def:p})",
                schema.dialect, schema.name,
            );
        }
    }
    registry
});

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op_schema::Arity;

    #[test]
    fn registry_finds_definitions_across_dialects() {
        let def = OpDef::lookup(Symbol::new("func"), Symbol::new("return"))
            .expect("func.return should be registered");
        assert_eq!((def.schema.dialect, def.schema.name), ("func", "return"));
        assert_eq!(def.schema.operands[0].arity, Arity::Variadic);
        assert!(OpDef::lookup(Symbol::new("func"), Symbol::new("no_such_op")).is_none());
    }
}
