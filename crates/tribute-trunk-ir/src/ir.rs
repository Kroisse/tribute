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
    pub location: Location<'db>,
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
}

impl<'db> Operation<'db> {
    /// Create a builder for an operation with the given name.
    pub fn of(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        name: OpNameId<'db>,
    ) -> OperationBuilder<'db> {
        OperationBuilder::new(db, location, name)
    }

    /// Create a builder, parsing "dialect.operation" string.
    pub fn of_name(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        name: &str,
    ) -> OperationBuilder<'db> {
        let name = OpNameId::parse(db, name).expect("invalid operation name");
        Self::of(db, location, name)
    }

    pub fn result(self, db: &'db dyn salsa::Database, index: u32) -> Value<'db> {
        Value::new(db, ValueDef::OpResult(self), index)
    }
}

#[salsa::tracked(debug)]
pub struct Block<'db> {
    pub location: Location<'db>,
    #[returns(deref)]
    pub args: Vec<Type>,
    #[returns(deref)]
    pub operations: Vec<Operation<'db>>,
}

impl<'db> Block<'db> {
    pub fn arg(self, db: &'db dyn salsa::Database, index: u32) -> Value<'db> {
        Value::new(db, ValueDef::BlockArg(self), index)
    }
}

#[salsa::tracked(debug)]
pub struct Region<'db> {
    pub location: Location<'db>,
    #[returns(ref)]
    pub blocks: Vec<Block<'db>>,
}

// ============================================================================
// Builders
// ============================================================================

/// Builder for constructing Operation instances.
pub struct OperationBuilder<'db> {
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    name: OpNameId<'db>,
    operands: Vec<Value<'db>>,
    results: Vec<Type>,
    attributes: BTreeMap<Symbol<'db>, Attribute<'db>>,
    regions: Vec<Region<'db>>,
    successors: Vec<Block<'db>>,
}

impl<'db> OperationBuilder<'db> {
    pub fn new(db: &'db dyn salsa::Database, location: Location<'db>, name: OpNameId<'db>) -> Self {
        Self {
            db,
            location,
            name,
            operands: Vec::new(),
            results: Vec::new(),
            attributes: BTreeMap::new(),
            regions: Vec::new(),
            successors: Vec::new(),
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
            self.location,
            self.name,
            self.operands,
            self.results,
            self.attributes,
            self.regions,
            self.successors,
        )
    }
}

/// Builder for constructing Block instances.
pub struct BlockBuilder<'db> {
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    args: Vec<Type>,
    operations: Vec<Operation<'db>>,
}

impl<'db> BlockBuilder<'db> {
    pub fn new(db: &'db dyn salsa::Database, location: Location<'db>) -> Self {
        Self {
            db,
            location,
            args: Vec::new(),
            operations: Vec::new(),
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
        Block::new(self.db, self.location, self.args, self.operations)
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
            location,
            "main",
            vec![],
            vec![Type::I { bits: 32 }],
            |entry| {
                let c0 = entry.op(arith::Const::i32(db, location, 40));
                let c1 = entry.op(arith::Const::i32(db, location, 2));
                let add = entry.op(arith::Add::new(
                    db,
                    location,
                    c0.result(db),
                    c1.result(db),
                    Type::I { bits: 32 },
                ));
                entry.op(func::Return::value(db, location, add.result(db)));
            },
        );

        core::Module::build(db, location, "main", |top| {
            top.op(main_func);
        })
        .as_operation()
    }

    #[test]
    fn can_model_basic_structure() {
        TributeDatabaseImpl::default().attach(|db| {
            let op = build_sample_module(db);
            let module = core::Module::from_operation(db, op).unwrap();
            assert_eq!(module.name(db), "main");
        });
    }

    // Test the new define_op! macro
    mod define_op_tests {
        use crate::{Attribute, DialectOp, Region, Type, dialect};
        use salsa::Database;
        use std::path::PathBuf;
        use tribute_core::{Location, PathId, Span, TributeDatabaseImpl};

        // Test: dialect! macro for grouping ops
        dialect! {
            test {
                /// Test binary operation.
                pub op binary(lhs, rhs) -> result {};

                /// Test constant operation.
                pub op constant[value]() -> result {};

                /// Test variadic operation.
                pub op variadic(..args) {};

                /// Test region operation.
                pub op container[name]() { body };
            }
        }

        #[salsa::tracked]
        fn test_binary_op(db: &dyn salsa::Database) -> crate::Operation<'_> {
            let path = PathId::new(db, PathBuf::from("test.tr"));
            let location = Location::new(path, Span::new(0, 0));

            // Create dummy values using a helper op
            let dummy_op = crate::Operation::of_name(db, location, "test.dummy")
                .result(Type::I { bits: 32 })
                .result(Type::I { bits: 32 })
                .build();
            let v0 = dummy_op.result(db, 0);
            let v1 = dummy_op.result(db, 1);

            // Test Binary::new
            let binary = Binary::new(db, location, v0, v1, Type::I { bits: 32 });
            binary.as_operation()
        }

        #[test]
        fn test_define_op_binary() {
            TributeDatabaseImpl::default().attach(|db| {
                let op = test_binary_op(db);
                let binary = Binary::from_operation(db, op).unwrap();
                assert_eq!(binary.result_ty(db), Type::I { bits: 32 });

                // Test auto-generated named accessors
                let lhs = binary.lhs(db);
                let rhs = binary.rhs(db);
                assert_ne!(lhs, rhs); // They should be different values
            });
        }

        #[salsa::tracked]
        fn test_constant_op(db: &dyn salsa::Database) -> crate::Operation<'_> {
            let path = PathId::new(db, PathBuf::from("test.tr"));
            let location = Location::new(path, Span::new(0, 0));

            let constant = Constant::new(db, location, Type::I { bits: 64 }, Attribute::Int(42));
            constant.as_operation()
        }

        #[test]
        fn test_define_op_constant() {
            TributeDatabaseImpl::default().attach(|db| {
                let op = test_constant_op(db);
                let constant = Constant::from_operation(db, op).unwrap();
                assert_eq!(constant.result_ty(db), Type::I { bits: 64 });

                // Test auto-generated attribute accessor
                assert_eq!(constant.value(db), &Attribute::Int(42));
            });
        }

        #[salsa::tracked]
        fn test_variadic_op(db: &dyn salsa::Database) -> crate::Operation<'_> {
            let path = PathId::new(db, PathBuf::from("test.tr"));
            let location = Location::new(path, Span::new(0, 0));

            let variadic = Variadic::new(db, location, vec![]);
            variadic.as_operation()
        }

        #[test]
        fn test_define_op_variadic() {
            TributeDatabaseImpl::default().attach(|db| {
                let op = test_variadic_op(db);
                let variadic = Variadic::from_operation(db, op).unwrap();
                assert!(variadic.operands(db).is_empty());
            });
        }

        #[salsa::tracked]
        fn test_container_op(db: &dyn salsa::Database) -> crate::Operation<'_> {
            let path = PathId::new(db, PathBuf::from("test.tr"));
            let location = Location::new(path, Span::new(0, 0));

            let block = crate::Block::new(db, location, vec![], vec![]);
            let region = Region::new(db, location, vec![block]);

            let container =
                Container::new(db, location, Attribute::String("test".to_string()), region);
            container.as_operation()
        }

        #[test]
        fn test_define_op_container() {
            TributeDatabaseImpl::default().attach(|db| {
                let op = test_container_op(db);
                Container::from_operation(db, op).unwrap();
            });
        }
    }
}
