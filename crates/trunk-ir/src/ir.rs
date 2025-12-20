//! Core IR structures.

use std::collections::BTreeMap;
use std::sync::LazyLock;

use lasso::{Rodeo, Spur};
use parking_lot::RwLock;

use crate::Location;
use crate::{Attribute, IdVec, Type};

// ============================================================================
// Interned Types
// ============================================================================

/// Global string interner for symbols.
static INTERNER: LazyLock<RwLock<Rodeo>> = LazyLock::new(|| RwLock::new(Rodeo::default()));

/// Interned symbol for efficient comparison of names (functions, variables, fields, etc.)
///
/// Uses lasso for string interning with 4-byte Spur keys.
/// Significantly smaller than Salsa's interned types.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, salsa::Update)]
pub struct Symbol(Spur);

impl Symbol {
    /// Intern a string and return its symbol.
    pub fn new(text: &str) -> Self {
        Symbol(INTERNER.write().get_or_intern(text))
    }

    /// Access the symbol's text with zero-copy (private - internal use only).
    ///
    /// SAFETY: Do NOT call any Symbol methods (to_string, Display, ==, etc.)
    /// from within the closure, as this will cause a deadlock due to the RwLock.
    fn with_str<R>(&self, f: impl FnOnce(&str) -> R) -> R {
        let interner = INTERNER.read();
        let text = interner.resolve(&self.0);
        f(text)
    }
}

impl From<&str> for Symbol {
    fn from(text: &str) -> Self {
        Symbol::new(text)
    }
}

/// Helper macro for declaring multiple lazy static symbols at once.
///
/// # Example
/// ```
/// use trunk_ir::symbols;
/// use std::sync::LazyLock;
///
/// symbols! {
///     ATTR_NAME => "name",
///     ATTR_TYPE => "type",
///     #[allow(dead_code)]
///     ATTR_UNUSED => "unused",
/// }
/// ```
#[macro_export]
macro_rules! symbols {
    ($($(#[$attr:meta])* $name:ident => $text:expr),* $(,)?) => {
        $(
            $(#[$attr])*
            static $name: std::sync::LazyLock<$crate::Symbol> =
                std::sync::LazyLock::new(|| $crate::Symbol::new($text));
        )*
    };
}

// Convenient comparison with &str
impl PartialEq<str> for Symbol {
    fn eq(&self, other: &str) -> bool {
        self.with_str(|s| s == other)
    }
}

impl PartialEq<&str> for Symbol {
    fn eq(&self, other: &&str) -> bool {
        self.with_str(|s| s == *other)
    }
}

impl PartialEq<Symbol> for str {
    fn eq(&self, other: &Symbol) -> bool {
        other.with_str(|s| s == self)
    }
}

impl PartialEq<Symbol> for &str {
    fn eq(&self, other: &Symbol) -> bool {
        other.with_str(|s| s == *self)
    }
}

// For Display (uses with_str for zero-copy)
impl std::fmt::Display for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.with_str(|s| write!(f, "{}", s))
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
    pub index: usize,
}

// ============================================================================
// Core IR Structures
// ============================================================================

#[salsa::tracked(debug)]
pub struct Operation<'db> {
    pub location: Location<'db>,
    /// Dialect name (e.g., "arith", "func").
    pub dialect: Symbol,
    /// Operation name within the dialect (e.g., "add", "call").
    pub name: Symbol,
    #[returns(ref)]
    pub operands: IdVec<Value<'db>>,
    #[returns(ref)]
    pub results: IdVec<Type<'db>>,
    #[returns(ref)]
    pub attributes: BTreeMap<Symbol, Attribute<'db>>,
    #[tracked]
    #[returns(ref)]
    pub regions: IdVec<Region<'db>>,
    #[returns(ref)]
    pub successors: IdVec<Block<'db>>,
}

impl<'db> Operation<'db> {
    /// Create a builder for an operation with the given dialect and name.
    pub fn of(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        dialect: Symbol,
        name: Symbol,
    ) -> OperationBuilder<'db> {
        OperationBuilder::new(db, location, dialect, name)
    }

    /// Create a builder, parsing "dialect.operation" string.
    pub fn of_name(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        full_name: &str,
    ) -> OperationBuilder<'db> {
        let (dialect, name) = full_name
            .split_once('.')
            .expect("invalid operation name: expected 'dialect.operation'");
        let dialect = Symbol::new(dialect);
        let name = Symbol::new(name);
        Self::of(db, location, dialect, name)
    }

    /// Format as "dialect.operation".
    pub fn full_name(&self, db: &'db dyn salsa::Database) -> String {
        format!("{}.{}", self.dialect(db), self.name(db))
    }

    pub fn result(self, db: &'db dyn salsa::Database, index: usize) -> Value<'db> {
        Value::new(db, ValueDef::OpResult(self), index)
    }

    /// Create a builder initialized from an existing operation.
    pub fn modify(&self, db: &'db dyn salsa::Database) -> OperationBuilder<'db> {
        OperationBuilder {
            db,
            location: self.location(db),
            dialect: self.dialect(db),
            name: self.name(db),
            operands: self.operands(db).clone(),
            results: self.results(db).clone(),
            attributes: self.attributes(db).clone(),
            regions: self.regions(db).clone(),
            successors: self.successors(db).clone(),
        }
    }
}

#[salsa::tracked(debug)]
pub struct Block<'db> {
    pub location: Location<'db>,
    #[returns(ref)]
    pub args: IdVec<Type<'db>>,
    #[returns(ref)]
    pub operations: IdVec<Operation<'db>>,
}

impl<'db> Block<'db> {
    pub fn arg(self, db: &'db dyn salsa::Database, index: usize) -> Value<'db> {
        Value::new(db, ValueDef::BlockArg(self), index)
    }
}

#[salsa::tracked(debug)]
pub struct Region<'db> {
    pub location: Location<'db>,
    #[returns(ref)]
    pub blocks: IdVec<Block<'db>>,
}

// ============================================================================
// Builders
// ============================================================================

/// Builder for constructing Operation instances.
pub struct OperationBuilder<'db> {
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    dialect: Symbol,
    name: Symbol,
    operands: IdVec<Value<'db>>,
    results: IdVec<Type<'db>>,
    attributes: BTreeMap<Symbol, Attribute<'db>>,
    regions: IdVec<Region<'db>>,
    successors: IdVec<Block<'db>>,
}

impl<'db> OperationBuilder<'db> {
    pub fn new(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        dialect: Symbol,
        name: Symbol,
    ) -> Self {
        Self {
            db,
            location,
            dialect,
            name,
            operands: Default::default(),
            results: Default::default(),
            attributes: Default::default(),
            regions: Default::default(),
            successors: Default::default(),
        }
    }

    pub fn operands(mut self, operands: IdVec<Value<'db>>) -> Self {
        self.operands = operands;
        self
    }

    pub fn dialect(mut self, dialect: Symbol) -> Self {
        self.dialect = dialect;
        self
    }

    pub fn dialect_str(mut self, dialect: &str) -> Self {
        self.dialect = Symbol::new(dialect);
        self
    }

    pub fn name(mut self, name: Symbol) -> Self {
        self.name = name;
        self
    }

    pub fn name_str(mut self, name: &str) -> Self {
        self.name = Symbol::new(name);
        self
    }

    pub fn operand(mut self, operand: Value<'db>) -> Self {
        self.operands.push(operand);
        self
    }

    pub fn results(mut self, results: IdVec<Type<'db>>) -> Self {
        self.results = results;
        self
    }

    pub fn result(mut self, ty: Type<'db>) -> Self {
        self.results.push(ty);
        self
    }

    pub fn attr(mut self, key: impl Into<Symbol>, value: Attribute<'db>) -> Self {
        self.attributes.insert(key.into(), value);
        self
    }

    pub fn regions(mut self, regions: IdVec<Region<'db>>) -> Self {
        self.regions = regions;
        self
    }

    pub fn region(mut self, region: Region<'db>) -> Self {
        self.regions.push(region);
        self
    }

    pub fn successors(mut self, successors: IdVec<Block<'db>>) -> Self {
        self.successors = successors;
        self
    }

    pub fn build(self) -> Operation<'db> {
        Operation::new(
            self.db,
            self.location,
            self.dialect,
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
    args: IdVec<Type<'db>>,
    operations: IdVec<Operation<'db>>,
}

impl<'db> BlockBuilder<'db> {
    pub fn new(db: &'db dyn salsa::Database, location: Location<'db>) -> Self {
        Self {
            db,
            location,
            args: Default::default(),
            operations: Default::default(),
        }
    }

    pub fn args(mut self, args: IdVec<Type<'db>>) -> Self {
        self.args = args;
        self
    }

    pub fn arg(mut self, ty: Type<'db>) -> Self {
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
    use crate::test_db::TestDatabase;
    use crate::{
        DialectOp, DialectType, Location, PathId, Span,
        dialect::{arith, core, func},
        idvec,
    };
    use salsa::Database;

    #[salsa::tracked]
    fn build_sample_module(db: &dyn salsa::Database) -> Operation<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        let location = Location::new(path, Span::new(0, 0));

        let main_func = func::Func::build(
            db,
            location,
            "main",
            idvec![],
            core::I32::new(db).as_type(),
            |entry| {
                let c0 = entry.op(arith::Const::i32(db, location, 40));
                let c1 = entry.op(arith::Const::i32(db, location, 2));
                let add = entry.op(arith::add(
                    db,
                    location,
                    c0.result(db),
                    c1.result(db),
                    core::I32::new(db).as_type(),
                ));
                entry.op(func::Return::value(db, location, add.result(db)));
            },
        );

        core::Module::build(db, location, "main".into(), |top| {
            top.op(main_func);
        })
        .as_operation()
    }

    #[test]
    fn can_model_basic_structure() {
        TestDatabase::default().attach(|db| {
            let op = build_sample_module(db);
            let module = core::Module::from_operation(db, op).unwrap();
            assert_eq!(module.name(db), "main");
        });
    }

    // Test the new define_op! macro
    mod define_op_tests {
        use crate::test_db::TestDatabase;
        use crate::{
            Attribute, DialectType, Location, PathId, Region, Span, dialect, dialect::core, idvec,
        };
        use salsa::Database;

        // Test: dialect! macro for grouping ops
        dialect! {
            mod test {
                /// Test binary operation.
                fn binary(lhs, rhs) -> result;

                /// Test constant operation.
                #[attr(value)]
                fn constant() -> result;

                /// Test variadic operation.
                fn variadic(#[rest] args);

                /// Test region operation.
                #[attr(name)]
                fn container() {
                    #[region(body)] {}
                };

                /// Test mixed operands: fixed + variadic.
                fn mixed(first, second, #[rest] rest) -> result;

                /// Test multi-result operation.
                fn multi_result(input) -> (quotient, remainder);
            }
        }

        #[salsa::tracked]
        fn test_binary_op(db: &dyn salsa::Database) -> Binary<'_> {
            let path = PathId::new(db, "file:///test.trb".to_owned());
            let location = Location::new(path, Span::new(0, 0));

            // Create dummy values using a helper op
            let dummy_op = crate::Operation::of_name(db, location, "test.dummy")
                .result(core::I32::new(db).as_type())
                .result(core::I32::new(db).as_type())
                .build();
            let v0 = dummy_op.result(db, 0);
            let v1 = dummy_op.result(db, 1);

            binary(db, location, v0, v1, core::I32::new(db).as_type())
        }

        #[test]
        fn test_define_op_binary() {
            TestDatabase::default().attach(|db| {
                let binary = test_binary_op(db);
                assert_eq!(binary.result_ty(db), core::I32::new(db).as_type());

                // Test auto-generated named accessors
                let lhs = binary.lhs(db);
                let rhs = binary.rhs(db);
                assert_ne!(lhs, rhs); // They should be different values
            });
        }

        #[salsa::tracked]
        fn test_constant_op(db: &dyn salsa::Database) -> Constant<'_> {
            let path = PathId::new(db, "file:///test.trb".to_owned());
            let location = Location::new(path, Span::new(0, 0));

            constant(db, location, core::I64::new(db).as_type(), 42i64.into())
        }

        #[test]
        fn test_define_op_constant() {
            TestDatabase::default().attach(|db| {
                let constant = test_constant_op(db);
                assert_eq!(constant.result_ty(db), core::I64::new(db).as_type());

                // Test auto-generated attribute accessor
                assert_eq!(constant.value(db), &Attribute::IntBits(42));
            });
        }

        #[salsa::tracked]
        fn test_variadic_op(db: &dyn salsa::Database) -> Variadic<'_> {
            let path = PathId::new(db, "file:///test.trb".to_owned());
            let location = Location::new(path, Span::new(0, 0));

            variadic(db, location, vec![])
        }

        #[test]
        fn test_define_op_variadic() {
            TestDatabase::default().attach(|db| {
                let variadic = test_variadic_op(db);
                assert!(variadic.args(db).is_empty());
            });
        }

        #[salsa::tracked]
        fn test_container_op(db: &dyn salsa::Database) -> Container<'_> {
            let path = PathId::new(db, "file:///test.trb".to_owned());
            let location = Location::new(path, Span::new(0, 0));

            let block = crate::Block::new(db, location, idvec![], idvec![]);
            let region = Region::new(db, location, idvec![block]);

            container(db, location, Attribute::String("test".to_string()), region)
        }

        #[test]
        fn test_define_op_container() {
            TestDatabase::default().attach(|db| {
                let container = test_container_op(db);
                assert_eq!(container.body(db).blocks(db).len(), 1);
                assert_eq!(container.regions(db).len(), 1);
            });
        }

        #[salsa::tracked]
        fn test_mixed_op(db: &dyn salsa::Database) -> Mixed<'_> {
            let path = PathId::new(db, "file:///test.trb".to_owned());
            let location = Location::new(path, Span::new(0, 0));

            // Create dummy values
            let dummy_op = crate::Operation::of_name(db, location, "test.dummy")
                .result(core::I32::new(db).as_type())
                .result(core::I32::new(db).as_type())
                .result(core::I32::new(db).as_type())
                .result(core::I32::new(db).as_type())
                .build();
            let v0 = dummy_op.result(db, 0);
            let v1 = dummy_op.result(db, 1);
            let v2 = dummy_op.result(db, 2);
            let v3 = dummy_op.result(db, 3);

            mixed(
                db,
                location,
                v0,
                v1,
                vec![v2, v3],
                core::I32::new(db).as_type(),
            )
        }

        #[test]
        fn test_define_op_mixed() {
            TestDatabase::default().attach(|db| {
                let mixed = test_mixed_op(db);
                assert_eq!(mixed.result_ty(db), core::I32::new(db).as_type());

                // Test named accessors for fixed operands
                let first = mixed.first(db);
                let second = mixed.second(db);
                assert_ne!(first, second);

                // Test variadic accessor
                let rest = mixed.rest(db);
                assert_eq!(rest.len(), 2);
            });
        }

        #[salsa::tracked]
        fn test_multi_result_op(db: &dyn salsa::Database) -> MultiResult<'_> {
            let path = PathId::new(db, "file:///test.trb".to_owned());
            let location = Location::new(path, Span::new(0, 0));

            // Create a dummy input value
            let dummy_op = crate::Operation::of_name(db, location, "test.dummy")
                .result(core::I32::new(db).as_type())
                .build();
            let input = dummy_op.result(db, 0);

            multi_result(
                db,
                location,
                input,
                core::I32::new(db).as_type(), // quotient type
                core::I32::new(db).as_type(), // remainder type
            )
        }

        #[test]
        fn test_define_op_multi_result() {
            TestDatabase::default().attach(|db| {
                let multi = test_multi_result_op(db);

                // Test named result accessors for each result
                assert_eq!(multi.quotient_ty(db), core::I32::new(db).as_type());
                assert_eq!(multi.remainder_ty(db), core::I32::new(db).as_type());

                // Test that we get different Value handles for each result
                let q = multi.quotient(db);
                let r = multi.remainder(db);
                assert_ne!(q, r); // They should be different values (different indices)

                // Verify the underlying operation has 2 results
                assert_eq!(multi.results(db).len(), 2);
            });
        }
    }
}
