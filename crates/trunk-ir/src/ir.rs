//! Core IR structures.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::sync::LazyLock;

use lasso::{Rodeo, Spur};
use parking_lot::RwLock;

use crate::{Attribute, IdVec, Location, Type};

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
    /// Intern a static string and return its symbol. Prefer this over `from_dynamic` when possible.
    pub fn new(text: &'static str) -> Self {
        Self::get_or_else(text, |rodeo| rodeo.get_or_intern_static(text))
    }

    /// Intern a string and return its symbol. Prefer `new` if the text is static.
    pub fn from_dynamic(text: &str) -> Self {
        Self::get_or_else(text, |rodeo| rodeo.get_or_intern(text))
    }

    fn get_or_else(text: &str, f: impl for<'r> FnOnce(&'r mut Rodeo) -> Spur) -> Self {
        let mut lock = INTERNER.upgradable_read();
        Symbol(if let Some(spur) = lock.get(text) {
            spur
        } else {
            lock.with_upgraded(f)
        })
    }

    /// Access the symbol's text with zero-copy.
    ///
    /// Uses `read_recursive()` to allow nested Symbol operations (Display, ==, to_string)
    /// within the closure without risk of deadlock.
    ///
    /// This is useful for optimization: when you need to work with the symbol's text
    /// without allocating a String, use this method. For example:
    ///
    /// ```ignore
    /// // Avoid: symbol.to_string() == "something"
    /// // Prefer:
    /// symbol.with_str(|s| s == "something")
    /// ```
    pub fn with_str<R>(&self, f: impl FnOnce(&str) -> R) -> R {
        let interner = INTERNER.read_recursive();
        let text = interner.resolve(&self.0);
        f(text)
    }
}

impl From<&'static str> for Symbol {
    fn from(text: &'static str) -> Self {
        Symbol::new(text)
    }
}

impl From<Cow<'_, str>> for Symbol {
    fn from(text: Cow<'_, str>) -> Self {
        Symbol::from_dynamic(&text)
    }
}

/// Helper macro for declaring multiple symbol helpers at once.
///
/// # Example
/// ```
/// use trunk_ir::symbols;
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
    ($($(#[$attr:meta])* $name:ident => $text:literal),* $(,)?) => {
        $(
            $(#[$attr])*
            #[allow(non_snake_case)]
            #[inline]
            pub fn $name() -> $crate::Symbol {
                $crate::Symbol::new($text)
            }
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

// Re-export QualifiedName from qualified_name module
pub use crate::qualified_name::QualifiedName;

// ============================================================================
// Block Identity
// ============================================================================

use std::sync::atomic::{AtomicU64, Ordering};

/// Global counter for generating unique block IDs.
static NEXT_BLOCK_ID: AtomicU64 = AtomicU64::new(1);

/// Stable block identifier that survives block recreation.
///
/// Unlike `Block` (which is a Salsa tracked struct with identity tied to creation),
/// `BlockId` is a simple u64 that can be preserved when a block is recreated
/// during IR transformations. This allows block arguments to maintain stable
/// identity across rewrites.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct BlockId(pub u64);

impl BlockId {
    /// Generate a fresh unique block ID.
    pub fn fresh() -> Self {
        BlockId(NEXT_BLOCK_ID.fetch_add(1, Ordering::Relaxed))
    }
}

// ============================================================================
// SSA Values
// ============================================================================

/// Where a value is defined: either an operation result or a block argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub enum ValueDef<'db> {
    OpResult(Operation<'db>),
    BlockArg(BlockId),
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
        full_name: &'static str,
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

/// Block argument with type and optional attributes.
///
/// Attributes can store metadata like binding names for pattern matching,
/// debug information, or other auxiliary data that doesn't affect semantics.
#[salsa::interned(debug)]
pub struct BlockArg<'db> {
    pub ty: Type<'db>,
    #[returns(ref)]
    pub attrs: BTreeMap<Symbol, Attribute<'db>>,
}

impl<'db> BlockArg<'db> {
    /// Create a block argument with just a type (no attributes).
    pub fn of_type(db: &'db dyn salsa::Database, ty: Type<'db>) -> Self {
        Self::new(db, ty, BTreeMap::new())
    }

    /// Create a block argument with a type and a single attribute.
    pub fn with_attr(
        db: &'db dyn salsa::Database,
        ty: Type<'db>,
        key: impl Into<Symbol>,
        value: Attribute<'db>,
    ) -> Self {
        let mut attrs = BTreeMap::new();
        attrs.insert(key.into(), value);
        Self::new(db, ty, attrs)
    }

    /// Get an attribute by key.
    pub fn get_attr(&self, db: &'db dyn salsa::Database, key: Symbol) -> Option<&Attribute<'db>> {
        self.attrs(db).get(&key)
    }
}

#[salsa::tracked(debug)]
pub struct Block<'db> {
    /// Stable identifier that survives block recreation during IR transformations.
    pub id: BlockId,
    pub location: Location<'db>,
    #[returns(ref)]
    pub args: IdVec<BlockArg<'db>>,
    #[returns(ref)]
    pub operations: IdVec<Operation<'db>>,
}

impl<'db> Block<'db> {
    /// Get a Value representing block argument at the given index.
    /// Uses BlockId for stable identity across block recreations.
    pub fn arg(self, db: &'db dyn salsa::Database, index: usize) -> Value<'db> {
        Value::new(db, ValueDef::BlockArg(self.id(db)), index)
    }

    /// Get the type of a block argument at the given index.
    pub fn arg_ty(self, db: &'db dyn salsa::Database, index: usize) -> Type<'db> {
        self.args(db)[index].ty(db)
    }

    /// Get all argument types as a vector (convenience method for compatibility).
    pub fn arg_types(self, db: &'db dyn salsa::Database) -> IdVec<Type<'db>> {
        self.args(db).iter().map(|arg| arg.ty(db)).collect()
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
        self.dialect = Symbol::from_dynamic(dialect);
        self
    }

    pub fn name(mut self, name: Symbol) -> Self {
        self.name = name;
        self
    }

    pub fn name_str(mut self, name: &str) -> Self {
        self.name = Symbol::from_dynamic(name);
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

    /// Set all attributes at once, replacing any existing ones.
    pub fn attrs(mut self, attrs: BTreeMap<Symbol, Attribute<'db>>) -> Self {
        self.attributes = attrs;
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
///
/// Supports fluent API for adding block arguments with attributes:
/// ```ignore
/// BlockBuilder::new(db, location)
///     .arg(ty1).attr(BIND_NAME(), name_sym)  // arg with attribute
///     .arg(ty2)                               // arg without attributes
///     .op(some_op)
///     .build()
/// ```
pub struct BlockBuilder<'db> {
    db: &'db dyn salsa::Database,
    id: BlockId,
    location: Location<'db>,
    args: IdVec<BlockArg<'db>>,
    /// Pending argument being built (type + accumulated attributes).
    /// Flushed when a new arg() is called or when build() is called.
    pending_arg: Option<(Type<'db>, BTreeMap<Symbol, Attribute<'db>>)>,
    operations: IdVec<Operation<'db>>,
}

impl<'db> BlockBuilder<'db> {
    /// Create a new BlockBuilder with a fresh BlockId.
    pub fn new(db: &'db dyn salsa::Database, location: Location<'db>) -> Self {
        Self {
            db,
            id: BlockId::fresh(),
            location,
            args: Default::default(),
            pending_arg: None,
            operations: Default::default(),
        }
    }

    /// Flush any pending argument to the args list.
    fn flush_pending_arg(&mut self) {
        if let Some((ty, attrs)) = self.pending_arg.take() {
            self.args.push(BlockArg::new(self.db, ty, attrs));
        }
    }

    /// Set a specific BlockId (used when recreating a block to preserve identity).
    pub fn id(mut self, id: BlockId) -> Self {
        self.id = id;
        self
    }

    /// Set all block arguments at once.
    ///
    /// Note: This flushes any pending argument and replaces all args.
    pub fn block_args(mut self, args: IdVec<BlockArg<'db>>) -> Self {
        self.flush_pending_arg();
        self.args = args;
        self
    }

    /// Set block arguments from types (convenience for common case with no attributes).
    ///
    /// Note: This flushes any pending argument and replaces all args.
    pub fn args(mut self, types: IdVec<Type<'db>>) -> Self {
        self.flush_pending_arg();
        self.args = types
            .iter()
            .map(|ty| BlockArg::of_type(self.db, *ty))
            .collect();
        self
    }

    /// Add a block argument with the given type.
    ///
    /// Use `.attr()` after this to add attributes to the argument.
    /// The argument is finalized when the next `.arg()` is called or when `.build()` is called.
    pub fn arg(mut self, ty: Type<'db>) -> Self {
        self.flush_pending_arg();
        self.pending_arg = Some((ty, BTreeMap::new()));
        self
    }

    /// Add an attribute to the current pending block argument.
    ///
    /// Must be called after `.arg()`. Panics if no argument is pending.
    ///
    /// # Example
    /// ```ignore
    /// builder
    ///     .arg(ty).attr("bind_name", Symbol::new("x"))
    ///     .arg(ty).attr("flag", true)
    /// ```
    pub fn attr(mut self, key: impl Into<Symbol>, value: impl Into<Attribute<'db>>) -> Self {
        let (_, attrs) = self
            .pending_arg
            .as_mut()
            .expect("attr() called without a pending arg; call arg() first");
        attrs.insert(key.into(), value.into());
        self
    }

    /// Add a block argument with type and attributes (legacy API).
    ///
    /// Prefer using `.arg(ty).attr(...)` for better readability.
    pub fn arg_with_attrs(
        mut self,
        ty: Type<'db>,
        attrs: BTreeMap<Symbol, Attribute<'db>>,
    ) -> Self {
        self.flush_pending_arg();
        self.args.push(BlockArg::new(self.db, ty, attrs));
        self
    }

    /// Get a Value representing block argument at the given index.
    /// This can be used before the block is built to reference block arguments.
    ///
    /// Note: This counts only finalized args, not the pending one.
    pub fn block_arg(&self, db: &'db dyn salsa::Database, index: usize) -> Value<'db> {
        Value::new(db, ValueDef::BlockArg(self.id), index)
    }

    /// Add an operation to the block and return it for capturing.
    pub fn op<Op: crate::DialectOp<'db>>(&mut self, operation: Op) -> Op {
        self.operations.push(operation.as_operation());
        operation
    }

    pub fn build(mut self) -> Block<'db> {
        self.flush_pending_arg();
        Block::new(self.db, self.id, self.location, self.args, self.operations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DialectOp, DialectType, Location, PathId, Span,
        dialect::{arith, core, func},
        idvec,
    };
    use salsa_test_macros::salsa_test;

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

    #[salsa_test]
    fn can_model_basic_structure(db: &salsa::DatabaseImpl) {
        let op = build_sample_module(db);
        let module = core::Module::from_operation(db, op).unwrap();
        assert_eq!(module.name(db), "main");
    }

    #[salsa::tracked]
    fn build_block_with_attrs(db: &dyn salsa::Database) -> Block<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let i32_ty = core::I32::new(db).as_type();

        BlockBuilder::new(db, location)
            .arg(i32_ty)
            .attr("bind_name", Attribute::Symbol(Symbol::new("x")))
            .arg(i32_ty)
            .attr("bind_name", Attribute::Symbol(Symbol::new("y")))
            .attr("other", Attribute::Bool(true))
            .arg(i32_ty) // no attributes
            .build()
    }

    #[salsa_test]
    fn block_builder_fluent_arg_attrs(db: &salsa::DatabaseImpl) {
        let block = build_block_with_attrs(db);
        let i32_ty = core::I32::new(db).as_type();

        let args = block.args(db);
        assert_eq!(args.len(), 3);

        // First arg: bind_name = "x"
        let arg0 = &args[0];
        assert_eq!(arg0.ty(db), i32_ty);
        assert_eq!(
            arg0.get_attr(db, Symbol::new("bind_name")),
            Some(&Attribute::Symbol(Symbol::new("x")))
        );

        // Second arg: bind_name = "y", other = true
        let arg1 = &args[1];
        assert_eq!(
            arg1.get_attr(db, Symbol::new("bind_name")),
            Some(&Attribute::Symbol(Symbol::new("y")))
        );
        assert_eq!(
            arg1.get_attr(db, Symbol::new("other")),
            Some(&Attribute::Bool(true))
        );

        // Third arg: no attributes
        let arg2 = &args[2];
        assert!(arg2.attrs(db).is_empty());
    }

    // Test the new define_op! macro
    mod define_op_tests {
        use crate::{
            Attribute, BlockId, DialectType, Location, PathId, Region, Span, dialect,
            dialect::core, idvec,
        };
        use salsa_test_macros::salsa_test;

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

        #[salsa_test]
        fn test_define_op_binary(db: &salsa::DatabaseImpl) {
            let binary = test_binary_op(db);
            assert_eq!(binary.result_ty(db), core::I32::new(db).as_type());

            // Test auto-generated named accessors
            let lhs = binary.lhs(db);
            let rhs = binary.rhs(db);
            assert_ne!(lhs, rhs); // They should be different values
        }

        #[salsa::tracked]
        fn test_constant_op(db: &dyn salsa::Database) -> Constant<'_> {
            let path = PathId::new(db, "file:///test.trb".to_owned());
            let location = Location::new(path, Span::new(0, 0));

            constant(db, location, core::I64::new(db).as_type(), 42i64.into())
        }

        #[salsa_test]
        fn test_define_op_constant(db: &salsa::DatabaseImpl) {
            let constant = test_constant_op(db);
            assert_eq!(constant.result_ty(db), core::I64::new(db).as_type());

            // Test auto-generated attribute accessor
            assert_eq!(constant.value(db), &Attribute::IntBits(42));
        }

        #[salsa::tracked]
        fn test_variadic_op(db: &dyn salsa::Database) -> Variadic<'_> {
            let path = PathId::new(db, "file:///test.trb".to_owned());
            let location = Location::new(path, Span::new(0, 0));

            variadic(db, location, vec![])
        }

        #[salsa_test]
        fn test_define_op_variadic(db: &salsa::DatabaseImpl) {
            let variadic = test_variadic_op(db);
            assert!(variadic.args(db).is_empty());
        }

        #[salsa::tracked]
        fn test_container_op(db: &dyn salsa::Database) -> Container<'_> {
            let path = PathId::new(db, "file:///test.trb".to_owned());
            let location = Location::new(path, Span::new(0, 0));

            let block = crate::Block::new(db, BlockId::fresh(), location, idvec![], idvec![]);
            let region = Region::new(db, location, idvec![block]);

            container(db, location, Attribute::String("test".to_string()), region)
        }

        #[salsa_test]
        fn test_define_op_container(db: &salsa::DatabaseImpl) {
            let container = test_container_op(db);
            assert_eq!(container.body(db).blocks(db).len(), 1);
            assert_eq!(container.regions(db).len(), 1);
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

        #[salsa_test]
        fn test_define_op_mixed(db: &salsa::DatabaseImpl) {
            let mixed = test_mixed_op(db);
            assert_eq!(mixed.result_ty(db), core::I32::new(db).as_type());

            // Test named accessors for fixed operands
            let first = mixed.first(db);
            let second = mixed.second(db);
            assert_ne!(first, second);

            // Test variadic accessor
            let rest = mixed.rest(db);
            assert_eq!(rest.len(), 2);
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

        #[salsa_test]
        fn test_define_op_multi_result(db: &salsa::DatabaseImpl) {
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
        }
    }
}
