//! Core IR structures.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::sync::LazyLock;

use smallvec::SmallVec;
use lasso::{Rodeo, Spur};
use parking_lot::RwLock;

use crate::{Location, SymbolVec};
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
    /// Uses SymbolVec to inline up to 4 parent segments.
    parent: SymbolVec,
    /// The final name segment (guaranteed to exist).
    name: Symbol,
}

impl QualifiedName {
    /// Create a new qualified name with the given parent path and name.
    pub fn new(parent: impl Into<SymbolVec>, name: Symbol) -> Self {
        Self { parent: parent.into(), name }
    }

    /// Create a qualified name from string segments.
    /// Returns `None` if the iterator is empty.
    pub fn from_strs(segments: impl IntoIterator<Item = &'static str>) -> Option<Self> {
        segments.into_iter().map(Symbol::new).collect()
    }

    /// Create a simple (single-segment) qualified name.
    pub fn simple(name: Symbol) -> Self {
        Self {
            parent: SmallVec::new(),
            name,
        }
    }

    /// Get all segments of this qualified name (parent + name).
    pub fn to_segments(&self) -> SmallVec<[Symbol; 6]> {
        let mut result = SmallVec::with_capacity(self.parent.len() + 1);
        result.extend_from_slice(&self.parent);
        result.push(self.name);
        result
    }

    /// Get the parent path as a slice of symbols.
    pub fn as_parent(&self) -> &[Symbol] {
        &self.parent
    }

    /// Get the parent as a QualifiedName, if it exists.
    /// Returns `None` for simple (single-segment) names.
    pub fn to_parent(&self) -> Option<QualifiedName> {
        QualifiedName::try_from(&self.parent[..]).ok()
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
        Some(QualifiedName::new(
            &self.parent[base_len..],self.name,
        ))
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

    /// Returns an iterator over all segments (parent + name).
    pub fn iter(&self) -> <&Self as IntoIterator>::IntoIter {
        self.into_iter()
    }

    /// Join this qualified name with another, creating a new qualified name.
    ///
    /// # Example
    /// ```
    /// # use trunk_ir::ir::{QualifiedName, Symbol};
    /// let base = QualifiedName::from_strs(["std", "io"]).unwrap();
    /// let suffix = QualifiedName::from_strs(["Reader", "new"]).unwrap();
    /// let full = base.join(&suffix);
    /// assert_eq!(full.to_string(), "std::io::Reader::new");
    /// ```
    pub fn join(&self, other: &QualifiedName) -> QualifiedName {
        QualifiedName::new(self.iter().chain(other.parent.iter().copied()).collect::<SymbolVec>(), other.name)
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

impl IntoIterator for QualifiedName {
    type Item = Symbol;
    type IntoIter = std::iter::Chain<
        smallvec::IntoIter<[Symbol; 4]>,
        std::iter::Once<Symbol>,
    >;

    fn into_iter(self) -> Self::IntoIter {
        self.parent.into_iter().chain(std::iter::once(self.name))
    }
}

impl<'a> IntoIterator for &'a QualifiedName {
    type Item = Symbol;
    type IntoIter = std::iter::Chain<
        std::iter::Copied<std::slice::Iter<'a, Symbol>>,
        std::iter::Once<Symbol>,
    >;

    fn into_iter(self) -> Self::IntoIter {
        self.parent.iter().copied().chain(std::iter::once(self.name))
    }
}

impl std::iter::Extend<Symbol> for QualifiedName {
    fn extend<T: IntoIterator<Item = Symbol>>(&mut self, iter: T) {
        // Move current name to parent
        self.parent.push(self.name);

        // Extend parent with all symbols from iterator
        self.parent.extend(iter);

        // Pop the last element as new name (guaranteed non-empty)
        self.name = self.parent.pop().expect("extend maintains non-empty invariant");

        // Shrink to fit to avoid wasting memory
        self.parent.shrink_to_fit();
    }
}

impl std::iter::FromIterator<Symbol> for Option<QualifiedName> {
    fn from_iter<T: IntoIterator<Item = Symbol>>(iter: T) -> Self {
        let mut parent = SymbolVec::from_iter(iter);
        let name = parent.pop()?;
        parent.shrink_to_fit();
        Some(QualifiedName::new(parent, name))
    }
}

impl From<Symbol> for QualifiedName {
    fn from(symbol: Symbol) -> Self {
        QualifiedName::simple(symbol)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmptyQualifiedNameError;

impl std::fmt::Display for EmptyQualifiedNameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cannot create QualifiedName from empty collection")
    }
}

impl std::error::Error for EmptyQualifiedNameError {}

impl<'a> TryFrom<&'a [Symbol]> for QualifiedName {
    type Error = EmptyQualifiedNameError;

    fn try_from(segments: &'a [Symbol]) -> Result<Self, Self::Error> {
        let (name, parent) = segments.split_last().ok_or(EmptyQualifiedNameError)?;
        Ok(QualifiedName::new(SymbolVec::from_slice(parent), *name))
    }
}

impl TryFrom<Vec<Symbol>> for QualifiedName {
    type Error = EmptyQualifiedNameError;

    fn try_from(mut segments: Vec<Symbol>) -> Result<Self, Self::Error> {
        let name = segments.pop().ok_or(EmptyQualifiedNameError)?;
        Ok(QualifiedName::new(SmallVec::from_vec(segments), name))
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
        fn test_extend() {
            let mut name = QualifiedName::from_strs(["std", "io"]).unwrap();
            name.extend([Symbol::new("Read"), Symbol::new("read")]);
            assert_eq!(name.to_string(), "std::io::Read::read");

            // Extending with empty iterator should be a no-op
            let mut name2 = QualifiedName::simple(Symbol::new("foo"));
            name2.extend(std::iter::empty());
            assert_eq!(name2.to_string(), "foo");
        }

        #[test]
        fn test_join() {
            let base = QualifiedName::from_strs(["std", "io"]).unwrap();
            let suffix = QualifiedName::from_strs(["Read", "read"]).unwrap();
            let joined = base.join(&suffix);
            assert_eq!(joined.to_string(), "std::io::Read::read");
        }

        #[test]
        fn test_try_from_vec() {
            let symbols = vec![Symbol::new("std"), Symbol::new("io"), Symbol::new("Read")];
            let name = QualifiedName::try_from(symbols).unwrap();
            assert_eq!(name.to_string(), "std::io::Read");
        }

        #[test]
        fn test_try_from_slice() {
            let symbols = [Symbol::new("std"), Symbol::new("io")];
            let name = QualifiedName::try_from(&symbols[..]).unwrap();
            assert_eq!(name.to_string(), "std::io");
        }

        #[test]
        fn test_try_from_empty_vec_fails() {
            let result = QualifiedName::try_from(Vec::<Symbol>::new());
            assert_eq!(result, Err(EmptyQualifiedNameError));
        }

        #[test]
        fn test_try_from_empty_slice_fails() {
            let result = QualifiedName::try_from(&[][..]);
            assert_eq!(result, Err(EmptyQualifiedNameError));
        }

        #[test]
        fn test_from_iterator_for_option() {
            let symbols = vec![Symbol::new("std"), Symbol::new("io"), Symbol::new("Read")];
            let qn: Option<QualifiedName> = symbols.into_iter().collect();
            assert_eq!(qn.unwrap().to_string(), "std::io::Read");

            // Empty iterator produces None
            let empty: Option<QualifiedName> = std::iter::empty().collect();
            assert!(empty.is_none());
        }

        #[test]
        fn test_from_str() {
            let name: QualifiedName = "bar".into();
            assert!(name.is_simple());
            assert_eq!(name.name(), "bar");
        }

        // Property-based tests
        #[cfg(test)]
        mod proptests {
            use super::*;
            use proptest::prelude::*;

            // Generate arbitrary valid Symbol identifiers
            fn arb_symbol() -> impl Strategy<Value = Symbol> {
                "[a-z][a-z0-9_]{0,15}"
                    .prop_map(|s| Symbol::from_dynamic(&s))
            }

            // Generate arbitrary QualifiedName with specified number of segments
            fn arb_qualified_name_with_len(len: impl Into<prop::collection::SizeRange>) -> impl Strategy<Value = QualifiedName> {
                prop::collection::vec(arb_symbol(), len)
                    .prop_map(|segments| {
                        QualifiedName::try_from(segments)
                            .expect("non-empty by construction")
                    })
            }

            // Generate arbitrary QualifiedName with 1-8 segments
            fn arb_qualified_name() -> impl Strategy<Value = QualifiedName> {
                arb_qualified_name_with_len(1..=8)
            }

            proptest! {
                #[test]
                fn prop_never_empty(qn in arb_qualified_name()) {
                    prop_assert!(qn.len() >= 1);
                    prop_assert!(!qn.to_segments().is_empty());
                }

                #[test]
                fn prop_name_equals_last_segment(qn in arb_qualified_name()) {
                    let segments = qn.to_segments();
                    prop_assert_eq!(qn.name(), *segments.last().unwrap());
                }

                #[test]
                fn prop_segments_roundtrip(qn in arb_qualified_name()) {
                    let segments = qn.to_segments();
                    let reconstructed = segments.into_iter().collect();
                    prop_assert_eq!(Some(qn), reconstructed);
                }

                #[test]
                fn prop_simple_iff_one_segment(qn in arb_qualified_name()) {
                    prop_assert_eq!(qn.is_simple(), qn.len() == 1);
                }

                #[test]
                fn prop_parent_length(qn in arb_qualified_name()) {
                    prop_assert_eq!(qn.as_parent().len(), qn.len() - 1);
                }

                #[test]
                fn prop_display_contains_colons(qn in arb_qualified_name()) {
                    let display = qn.to_string();
                    let colon_count = display.matches("::").count();
                    prop_assert_eq!(colon_count, qn.len() - 1);
                }

                #[test]
                fn prop_starts_with_reflexive(qn in arb_qualified_name()) {
                    prop_assert!(qn.starts_with(&qn));
                }

                #[test]
                fn prop_starts_with_transitive(
                    c in arb_qualified_name(),
                    b_extra in arb_qualified_name(),
                    a_extra in arb_qualified_name(),
                ) {
                    // Build a prefix chain: c ⊆ b ⊆ a
                    let b = c.join(&b_extra);
                    let a = b.join(&a_extra);

                    prop_assert!(a.starts_with(&b));
                    prop_assert!(b.starts_with(&c));
                    prop_assert!(a.starts_with(&c));
                }

                #[test]
                fn prop_relative_some_with_proper_prefix(
                    (a, k) in arb_qualified_name_with_len(2..=8)
                        .prop_flat_map(|a| {
                            let a_len = a.len();
                            (Just(a), 1..a_len)
                        })
                ) {
                    // b is a proper prefix of a (k < a.len())
                    let b = a.iter().take(k).collect::<Option<_>>().unwrap();
                    let rel = a.relative(&b);
                    prop_assert!(rel.is_some());
                    prop_assert_eq!(rel.unwrap().len(), a.len() - b.len());
                }

                #[test]
                fn prop_relative_none_if_equal_or_same_length(
                    (a, b) in arb_qualified_name()
                        .prop_flat_map(|a| {
                            let a_len = a.len();
                            let a_clone1 = a.clone();
                            let a_clone2 = a.clone();
                            prop_oneof![
                                // equal
                                1 => Just((a_clone1.clone(), a_clone1)),
                                // same length but not equal
                                9 => arb_qualified_name_with_len(a_len)
                                    .prop_map(move |b| (a_clone2.clone(), b))
                            ]
                        })
                ) {
                    // Either a == b, or a.len() == b.len() but a != b
                    prop_assert_eq!(a.len(), b.len());
                    prop_assert!(a.relative(&b).is_none());
                }

                #[test]
                fn prop_relative_implies_starts_with(
                    base in arb_qualified_name(),
                    extra in arb_qualified_name(),
                ) {
                    let full = base.join(&extra);

                    let rel = full.relative(&base);
                    prop_assert_eq!(rel, Some(extra.clone()));
                    prop_assert!(full.starts_with(&base));
                }

                #[test]
                fn prop_relative_length(
                    base in arb_qualified_name(),
                    extra in arb_qualified_name(),
                ) {
                    let full = base.join(&extra);

                    let rel = full.relative(&base).unwrap();
                    prop_assert_eq!(rel.len(), extra.len());
                    prop_assert_eq!(rel.len(), full.len() - base.len());
                }
            }
        }
    }
}
