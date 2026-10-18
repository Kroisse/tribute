//! Registered operation definitions generated from `#[dialect]`.
//!
//! An [`OpDef`] bundles everything the dialect macro registers for one
//! operation: its declarative [`OpSchema`] and the hooks the declaration
//! opts into, such as `#[verify]` and `#[asm_format]`. Definitions are registered by operation
//! name so that verification and other tooling can look them up for any
//! operation.

use std::error::Error;
use std::sync::LazyLock;

use crate::asm_format::OpAsmHooks;
use crate::name::{Definition, DefinitionTable, Name, NameArg, QualifiedName};
use crate::op_schema::{OpSchema, SchemaViolation};
use crate::{IrContext, OpRef};

/// The name of an operation.
pub type OpName = Name<OpDef>;

/// An operation name given to a builder.
pub type OpNameArg = NameArg<OpDef>;

/// Everything `#[dialect]` registers for one operation.
///
/// `#[dialect]` emits each definition as a `static`, so its address
/// identifies the operation: an [`OpName`] of a registered operation points
/// to it.
#[derive(Debug)]
pub struct OpDef {
    /// The operation's `dialect.name`.
    pub name: QualifiedName,
    /// Declarative constraints on the operation's entities.
    pub schema: OpSchema,
    /// Calls the wrapper's [`Verify`](crate::ops::Verify) impl for
    /// `#[verify]` operations.
    pub verifier: Option<OpVerifier>,
    /// Calls the wrapper's [`OpAsmFormat`](crate::asm_format::OpAsmFormat) impl
    /// for `#[asm_format]` operations.
    pub asm_format: Option<OpAsmHooks>,
}

/// An operation-local verifier generated from `#[verify]`.
pub type OpVerifier = fn(&IrContext, OpRef) -> Result<(), Box<dyn Error + Send + Sync>>;

impl OpDef {
    /// Run the declarative schema stages and then the `#[verify]` hook,
    /// stopping after the first stage that reports a violation.
    pub fn verify(&self, ctx: &IrContext, op: OpRef) -> Vec<OpViolation> {
        let violations = self.schema.verify(ctx, op);
        if !violations.is_empty() {
            return violations.into_iter().map(OpViolation::Schema).collect();
        }
        match self.verifier.map(|verifier| verifier(ctx, op)) {
            Some(Err(error)) => vec![OpViolation::Verifier(error.to_string())],
            _ => Vec::new(),
        }
    }

    /// The registered definition written `dialect.name`.
    pub fn lookup(name: &str) -> Option<&'static OpDef> {
        REGISTRY.get(name)
    }

    /// The registered definition of an operation.
    pub fn of(ctx: &IrContext, op: OpRef) -> Option<&'static OpDef> {
        ctx.op(op).name.def()
    }
}

impl Definition for OpDef {
    fn qualified_name(&self) -> &QualifiedName {
        &self.name
    }

    fn lookup(name: &str) -> Option<&'static Self> {
        REGISTRY.get(name)
    }
}

/// One way an operation fails its definition.
#[derive(Clone, Debug, PartialEq, Eq, derive_more::Display)]
pub enum OpViolation {
    /// The operation violates its declarative schema.
    #[display("{_0}")]
    Schema(SchemaViolation),
    /// The operation's `#[verify]` hook rejected it.
    #[display("{_0}")]
    Verifier(String),
}

/// Inventory entry registering an [`OpDef`].
///
/// Emitted by the `#[dialect]` macro for every declared operation.
pub struct OpDefRegistration(pub &'static OpDef);

inventory::collect!(OpDefRegistration);

static REGISTRY: LazyLock<DefinitionTable<OpDef>> = LazyLock::new(|| {
    DefinitionTable::collect(
        "operation",
        inventory::iter::<OpDefRegistration>
            .into_iter()
            .map(|OpDefRegistration(def)| *def),
    )
});

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op_schema::Arity;

    #[test]
    fn registry_finds_definitions_across_dialects() {
        let def = OpDef::lookup("func.return").expect("func.return should be registered");
        assert_eq!(def.name.as_str(), "func.return");
        assert_eq!((def.schema.dialect, def.schema.name), ("func", "return"));
        assert_eq!(def.schema.operands[0].arity, Arity::Variadic);
        assert!(OpDef::lookup("func.no_such_op").is_none());
    }

    #[test]
    fn asm_format_operations_carry_their_hooks() {
        let func = OpDef::lookup("func.func").expect("func.func should be registered");
        assert!(func.asm_format.is_some());
        let ret = OpDef::lookup("func.return").expect("func.return should be registered");
        assert!(ret.asm_format.is_none());
    }

    #[test]
    fn every_registered_name_has_one_handle() {
        let mut ctx = IrContext::new();
        for OpDefRegistration(def) in inventory::iter::<OpDefRegistration> {
            let name = def.name;
            assert_eq!(
                (name.dialect(), name.strip_dialect()),
                (def.schema.dialect, def.schema.name)
            );
            assert_eq!(ctx.intern_op_name(name.as_str()), OpName::registered(def));
        }
        assert_eq!(
            <crate::dialect::arith::Addi as crate::ops::DialectOp>::NAME,
            ctx.intern_op_name("arith.addi")
        );
    }

    #[test]
    fn unregistered_names_are_context_strings() {
        let mut ctx = IrContext::new();
        let name = ctx.intern_op_name("test.op");
        assert!(!name.is_registered());
        assert_eq!(name, ctx.intern_op_name("test.op"));
        assert_ne!(name, ctx.intern_op_name("test.other"));
        assert_eq!(ctx.lookup_op_name("test.op"), Some(name));
        assert_eq!(ctx.lookup_op_name("test.missing"), None);
        assert_eq!(name.dialect(&ctx), "test");
        assert_eq!(name.strip_dialect(&ctx), "op");
        assert_eq!(name.display(&ctx).to_string(), "test.op");
        assert!(name.is(&ctx, "test.op"));
        assert!(!name.is(&ctx, "test.opx"));
    }
}
