//! Core IR structures.

use std::collections::BTreeMap;

use crate::{Attribute, Type};
use tribute_core::Location;

// ============================================================================
// Interned Types
// ============================================================================

/// Interned symbol for efficient comparison of names (functions, variables, fields, etc.)
#[salsa::interned(debug)]
#[derive(Ord, PartialOrd)]
pub struct Symbol<'db> {
    #[returns(ref)]
    pub text: String,
}

/// Interned operation name for efficient dialect.operation comparison.
#[salsa::interned(debug)]
pub struct OpNameId<'db> {
    #[returns(ref)]
    pub dialect: String,
    #[returns(ref)]
    pub operation: String,
}

impl<'db> OpNameId<'db> {
    /// Parse "dialect.operation" format.
    pub fn parse(db: &'db dyn salsa::Database, full: &str) -> Option<Self> {
        let (dialect, operation) = full.split_once('.')?;
        Some(Self::new(db, dialect, operation))
    }

    /// Format as "dialect.operation".
    pub fn to_string(&self, db: &'db dyn salsa::Database) -> String {
        format!("{}.{}", self.dialect(db), self.operation(db))
    }
}

// ============================================================================
// SSA Values
// ============================================================================

/// Where a value is defined: either an operation result or a block argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub enum ValueDef<'db> {
    OpResult(Operation<'db>),
    BlockArg(Block<'db>),
}

/// SSA value: a definition point plus an index.
/// Interned so that identical (def, index) pairs yield the same ID.
#[salsa::interned(debug)]
pub struct Value<'db> {
    pub def: ValueDef<'db>,
    pub index: u32,
}

// ============================================================================
// Core IR Structures
// ============================================================================

#[salsa::tracked(debug)]
pub struct Operation<'db> {
    /// Interned operation name (dialect.operation).
    pub name: OpNameId<'db>,
    #[returns(ref)]
    pub operands: Vec<Value<'db>>,
    #[returns(ref)]
    pub results: Vec<Type>,
    #[returns(ref)]
    pub attributes: BTreeMap<Symbol<'db>, Attribute<'db>>,
    #[tracked]
    #[returns(ref)]
    pub regions: Vec<Region<'db>>,
    #[returns(ref)]
    pub successors: Vec<Block<'db>>,
    pub location: Location<'db>,
}

impl<'db> Operation<'db> {
    /// Create a builder for an operation with the given name.
    pub fn of(
        db: &'db dyn salsa::Database,
        name: OpNameId<'db>,
        location: Location<'db>,
    ) -> OperationBuilder<'db> {
        OperationBuilder::new(db, name, location)
    }

    /// Create a builder, parsing "dialect.operation" string.
    pub fn of_name(
        db: &'db dyn salsa::Database,
        name: &str,
        location: Location<'db>,
    ) -> OperationBuilder<'db> {
        let name = OpNameId::parse(db, name).expect("invalid operation name");
        Self::of(db, name, location)
    }

    pub fn result(self, db: &'db dyn salsa::Database, index: u32) -> Value<'db> {
        Value::new(db, ValueDef::OpResult(self), index)
    }
}

#[salsa::tracked(debug)]
pub struct Block<'db> {
    #[returns(deref)]
    pub args: Vec<Type>,
    #[returns(deref)]
    pub operations: Vec<Operation<'db>>,
    pub location: Location<'db>,
}

impl<'db> Block<'db> {
    pub fn arg(self, db: &'db dyn salsa::Database, index: u32) -> Value<'db> {
        Value::new(db, ValueDef::BlockArg(self), index)
    }
}

#[salsa::tracked(debug)]
pub struct Region<'db> {
    #[returns(ref)]
    pub blocks: Vec<Block<'db>>,
    pub location: Location<'db>,
}

// ============================================================================
// Builders
// ============================================================================

/// Builder for constructing Operation instances.
pub struct OperationBuilder<'db> {
    db: &'db dyn salsa::Database,
    name: OpNameId<'db>,
    operands: Vec<Value<'db>>,
    results: Vec<Type>,
    attributes: BTreeMap<Symbol<'db>, Attribute<'db>>,
    regions: Vec<Region<'db>>,
    successors: Vec<Block<'db>>,
    location: Location<'db>,
}

impl<'db> OperationBuilder<'db> {
    pub fn new(db: &'db dyn salsa::Database, name: OpNameId<'db>, location: Location<'db>) -> Self {
        Self {
            db,
            name,
            operands: Vec::new(),
            results: Vec::new(),
            attributes: BTreeMap::new(),
            regions: Vec::new(),
            successors: Vec::new(),
            location,
        }
    }

    pub fn operands(mut self, operands: Vec<Value<'db>>) -> Self {
        self.operands = operands;
        self
    }

    pub fn operand(mut self, operand: Value<'db>) -> Self {
        self.operands.push(operand);
        self
    }

    pub fn results(mut self, results: Vec<Type>) -> Self {
        self.results = results;
        self
    }

    pub fn result(mut self, ty: Type) -> Self {
        self.results.push(ty);
        self
    }

    pub fn attr(mut self, key: &str, value: Attribute<'db>) -> Self {
        let sym = Symbol::new(self.db, key);
        self.attributes.insert(sym, value);
        self
    }

    pub fn regions(mut self, regions: Vec<Region<'db>>) -> Self {
        self.regions = regions;
        self
    }

    pub fn region(mut self, region: Region<'db>) -> Self {
        self.regions.push(region);
        self
    }

    pub fn successors(mut self, successors: Vec<Block<'db>>) -> Self {
        self.successors = successors;
        self
    }

    pub fn build(self) -> Operation<'db> {
        Operation::new(
            self.db,
            self.name,
            self.operands,
            self.results,
            self.attributes,
            self.regions,
            self.successors,
            self.location,
        )
    }
}

/// Builder for constructing Block instances.
pub struct BlockBuilder<'db> {
    db: &'db dyn salsa::Database,
    args: Vec<Type>,
    operations: Vec<Operation<'db>>,
    location: Location<'db>,
}

impl<'db> BlockBuilder<'db> {
    pub fn new(db: &'db dyn salsa::Database, location: Location<'db>) -> Self {
        Self {
            db,
            args: Vec::new(),
            operations: Vec::new(),
            location,
        }
    }

    pub fn args(mut self, args: Vec<Type>) -> Self {
        self.args = args;
        self
    }

    pub fn arg(mut self, ty: Type) -> Self {
        self.args.push(ty);
        self
    }

    /// Add an operation to the block and return it for capturing.
    pub fn op<Op: crate::DialectOp<'db>>(&mut self, operation: Op) -> Op {
        self.operations.push(operation.as_operation());
        operation
    }

    pub fn build(self) -> Block<'db> {
        Block::new(self.db, self.args, self.operations, self.location)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DialectOp, arith, core, func};
    use salsa::Database;
    use std::path::PathBuf;
    use tribute_core::{PathId, Span, TributeDatabaseImpl};

    #[salsa::tracked]
    fn build_sample_module(db: &dyn salsa::Database) -> Operation<'_> {
        let path = PathId::new(db, PathBuf::from("test.tr"));
        let location = Location::new(path, Span::new(0, 0));

        let main_func = func::Func::build(
            db,
            "main",
            vec![],
            vec![Type::I { bits: 32 }],
            location,
            |entry| {
                let c0 = entry.op(arith::Const::i32(db, 40, location));
                let c1 = entry.op(arith::Const::i32(db, 2, location));
                let add = entry.op(arith::Add::new(
                    db,
                    c0.result(db),
                    c1.result(db),
                    Type::I { bits: 32 },
                    location,
                ));
                entry.op(func::Return::value(db, add.result(db), location));
            },
        );

        core::Module::build(db, "main", location, |top| {
            top.op(main_func);
        })
        .as_operation()
    }

    #[test]
    fn can_model_basic_structure() {
        TributeDatabaseImpl::default().attach(|db| {
            let op = build_sample_module(db);
            let module = core::Module::from_operation(db, op).unwrap();
            assert_eq!(module.name(db).as_str(), "main");
        });
    }
}
