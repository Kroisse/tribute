//! Core IR structures.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::sync::LazyLock;

use smallvec::SmallVec;
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

// ============================================================================
// Qualified Names
// ============================================================================

/// A fully qualified name consisting of path segments.
///
/// Examples: `std::intrinsics::wasi::preview1::fd_write`, `List::map`
///
/// Used for function callees, type references, and other qualified identifiers.
///
/// This is a non-empty structure: every QualifiedName has at least a name.
/// The parent path can be empty (for simple names like `foo`) or contain multiple segments.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct QualifiedName {
    /// Parent path segments (can be empty for simple names).
    /// Uses SmallVec<[Symbol; 4]> to inline up to 4 parent segments.
    parent: SmallVec<[Symbol; 4]>,
    /// The final name segment (guaranteed to exist).
    name: Symbol,
}

impl QualifiedName {
    /// Create a new qualified name from an iterator of symbols.
    /// Returns `None` if the iterator is empty.
    pub fn new(segments: impl IntoIterator<Item = Symbol>) -> Option<Self> {
        let mut parent = SmallVec::from_iter(segments);
        let name = parent.pop()?;
        Some(Self { parent, name })
    }

    /// Create a qualified name from string segments.
    /// Returns `None` if the iterator is empty.
    pub fn from_strs(segments: impl IntoIterator<Item = &'static str>) -> Option<Self> {
        Self::new(segments.into_iter().map(Symbol::new))
    }

    /// Create a simple (single-segment) qualified name.
    pub fn simple(name: Symbol) -> Self {
        Self {
            parent: SmallVec::new(),
            name,
        }
    }

    /// Get all segments of this qualified name (parent + name).
    pub fn segments(&self) -> SmallVec<[Symbol; 6]> {
        let mut result = SmallVec::with_capacity(self.parent.len() + 1);
        result.extend_from_slice(&self.parent);
        result.push(self.name);
        result
    }

    /// Get the parent path (all segments except the last).
    pub fn parent(&self) -> &[Symbol] {
        &self.parent
    }

    /// Get the last segment (the simple name).
    /// This is guaranteed to exist in a non-empty QualifiedName.
    pub fn name(&self) -> Symbol {
        self.name
    }

    /// Check if this is a simple (single-segment) name.
    pub fn is_simple(&self) -> bool {
        self.parent.is_empty()
    }

    /// Get the path relative to a base path.
    ///
    /// Returns `Some` if this path starts with `base`, containing the remaining segments.
    /// Returns `None` if this path does not start with `base`.
    ///
    /// # Example
    /// ```
    /// use trunk_ir::{QualifiedName, Symbol};
    ///
    /// let full = QualifiedName::from_strs(["std", "intrinsics", "wasi", "fd_write"]).unwrap();
    /// let base = QualifiedName::from_strs(["std", "intrinsics", "wasi"]).unwrap();
    ///
    /// let relative = full.relative(&base).unwrap();
    /// assert!(relative.is_simple());
    /// assert_eq!(relative.name(), "fd_write");
    /// ```
    pub fn relative(&self, base: &QualifiedName) -> Option<QualifiedName> {
        let base_len = base.parent.len() + 1;
        let self_len = self.parent.len() + 1;

        // Must be strictly longer to have remaining segments
        if self_len <= base_len {
            return None;
        }

        // Check if our parent starts with base's parent
        if !self.parent.starts_with(&base.parent) {
            return None;
        }

        // Since self_len > base_len, we know base.parent.len() < self.parent.len()
        // base.name should match self.parent[base.parent.len()]
        if self.parent[base.parent.len()] != base.name {
            return None;
        }

        // Extract remaining segments: parent[base_len..] + name
        QualifiedName::new(
            self.parent[base_len..]
                .iter()
                .copied()
                .chain(std::iter::once(self.name)),
        )
    }

    /// Check if this path starts with the given base path.
    pub fn starts_with(&self, base: &QualifiedName) -> bool {
        // First check if our parent starts with base's parent
        if !self.parent.starts_with(&base.parent) {
            return false;
        }

        // Two cases:
        // 1. base.parent is shorter than our parent: base.name matches self.parent[base.parent.len()]
        // 2. base.parent equals our parent: base.name matches our name
        if base.parent.len() < self.parent.len() {
            self.parent[base.parent.len()] == base.name
        } else if base.parent.len() == self.parent.len() {
            self.name == base.name
        } else {
            // base.parent is longer than our parent → can't start with
            false
        }
    }

    /// Get the number of segments (parent + name).
    ///
    /// This is always at least 1, since QualifiedName is non-empty by design.
    pub fn len(&self) -> usize {
        self.parent.len() + 1
    }
}

impl std::fmt::Display for QualifiedName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, seg) in self.parent.iter().enumerate() {
            if i > 0 {
                write!(f, "::")?;
            }
            write!(f, "{seg}")?;
        }
        if !self.parent.is_empty() {
            write!(f, "::")?;
        }
        write!(f, "{}", self.name)
    }
}

impl From<Symbol> for QualifiedName {
    fn from(symbol: Symbol) -> Self {
        QualifiedName::simple(symbol)
    }
}

impl From<&'static str> for QualifiedName {
    fn from(text: &'static str) -> Self {
        QualifiedName::simple(Symbol::new(text))
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

    // Test the new define_op! macro
    mod define_op_tests {
        use crate::{
            Attribute, DialectType, Location, PathId, Region, Span, dialect, dialect::core, idvec,
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

            let block = crate::Block::new(db, location, idvec![], idvec![]);
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

    mod qualified_name_tests {
        use super::*;

        #[test]
        fn test_qualified_name_size() {
            // Ensure QualifiedName stays at 32 bytes
            // This is a regression test to prevent accidental size increases
            assert_eq!(
                std::mem::size_of::<QualifiedName>(),
                32,
                "QualifiedName size changed! Expected 32 bytes (SmallVec<[Symbol; 4]> + Symbol + padding)"
            );
        }

        #[test]
        fn test_simple_name() {
            let name = QualifiedName::simple(Symbol::new("foo"));
            assert!(name.is_simple());
            assert_eq!(name.name(), "foo");
            assert_eq!(name.len(), 1);
            assert_eq!(name.to_string(), "foo");
        }

        #[test]
        fn test_qualified_name() {
            let name = QualifiedName::from_strs(["std", "intrinsics", "wasi", "fd_write"]).unwrap();
            assert!(!name.is_simple());
            assert_eq!(name.name(), "fd_write");
            assert_eq!(name.len(), 4);
            assert_eq!(name.to_string(), "std::intrinsics::wasi::fd_write");
        }

        #[test]
        fn test_relative() {
            let full = QualifiedName::from_strs(["std", "intrinsics", "wasi", "fd_write"]).unwrap();
            let base = QualifiedName::from_strs(["std", "intrinsics", "wasi"]).unwrap();

            let relative = full.relative(&base).unwrap();
            assert!(relative.is_simple());
            assert_eq!(relative.name(), "fd_write");
        }

        #[test]
        fn test_relative_multi_segment() {
            let full =
                QualifiedName::from_strs(["std", "intrinsics", "wasi", "preview1", "fd_write"]).unwrap();
            let base = QualifiedName::from_strs(["std", "intrinsics"]).unwrap();

            let relative = full.relative(&base).unwrap();
            assert_eq!(relative.len(), 3);
            assert_eq!(relative.to_string(), "wasi::preview1::fd_write");
        }

        #[test]
        fn test_relative_no_match() {
            let full = QualifiedName::from_strs(["std", "intrinsics", "wasi"]).unwrap();
            let base = QualifiedName::from_strs(["core", "intrinsics"]).unwrap();

            assert!(full.relative(&base).is_none());
        }

        #[test]
        fn test_relative_same_length() {
            let full = QualifiedName::from_strs(["std", "intrinsics"]).unwrap();
            let base = QualifiedName::from_strs(["std", "intrinsics"]).unwrap();

            // Same length should return None (no remaining segments)
            assert!(full.relative(&base).is_none());
        }

        #[test]
        fn test_starts_with() {
            let name = QualifiedName::from_strs(["std", "intrinsics", "wasi", "fd_write"]).unwrap();
            let prefix = QualifiedName::from_strs(["std", "intrinsics"]).unwrap();
            let other = QualifiedName::from_strs(["core", "intrinsics"]).unwrap();

            assert!(name.starts_with(&prefix));
            assert!(!name.starts_with(&other));
        }

        #[test]
        fn test_from_symbol() {
            let sym = Symbol::new("foo");
            let name: QualifiedName = sym.into();
            assert!(name.is_simple());
            assert_eq!(name.name(), "foo");
        }

        #[test]
        fn test_from_str() {
            let name: QualifiedName = "bar".into();
            assert!(name.is_simple());
            assert_eq!(name.name(), "bar");
        }
    }
}
