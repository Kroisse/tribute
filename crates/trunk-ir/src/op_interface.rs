//! Operation interface system for querying operation properties.
//!
//! This module provides an interface system similar to `type_interface.rs` but for operations.
//! It uses the `inventory` crate to build a registry of operation properties at compile time.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::LazyLock;

use smallvec::SmallVec;

use crate::Symbol;
use crate::ops::DialectOp;
use crate::{BlockRef, IrContext, OpRef, RegionRef, TypeRef, ValueRef};

/// Marker trait for pure operations (no side effects, safe to remove if unused).
///
/// Operations implementing this trait can be safely eliminated by DCE if their results are unused.
pub trait Pure {}

/// Registration entry for pure operations.
///
/// Use `inventory::submit!` to register pure operations at the dialect definition site.
pub struct PureOpRegistration {
    /// Dialect name (e.g., "arith", "adt")
    pub dialect: &'static str,
    /// Operation name within the dialect (e.g., "add", "const")
    pub op_name: &'static str,
}

inventory::collect!(PureOpRegistration);

/// Internal registry built from inventory at first access.
struct PureOpRegistry {
    /// Lookup: (dialect, op_name) -> is_pure
    pure_ops: HashSet<(Symbol, Symbol)>,
}

impl PureOpRegistry {
    fn new() -> Self {
        Self {
            pure_ops: HashSet::new(),
        }
    }

    fn lookup(&self, dialect: Symbol, op_name: Symbol) -> bool {
        self.pure_ops.contains(&(dialect, op_name))
    }
}

/// Global registry, lazily built from inventory on first access.
static REGISTRY: LazyLock<PureOpRegistry> = LazyLock::new(|| {
    let mut registry = PureOpRegistry::new();

    for reg in inventory::iter::<PureOpRegistration> {
        let dialect = Symbol::from_dynamic(reg.dialect);
        let op_name = Symbol::from_dynamic(reg.op_name);
        registry.pure_ops.insert((dialect, op_name));
    }

    registry
});

/// Interface for querying operation purity.
pub struct PureOps;

impl PureOps {
    /// Register a pure operation (internal use by macro).
    ///
    /// Use the `register_pure_op!` macro instead:
    /// ```text
    /// register_pure_op!(arith.addi);
    /// ```
    #[doc(hidden)]
    pub const fn register(dialect: &'static str, op_name: &'static str) -> PureOpRegistration {
        PureOpRegistration { dialect, op_name }
    }

    /// Check if an arena operation is pure (no side effects, safe to remove if unused).
    pub fn is_pure(ctx: &IrContext, op: OpRef) -> bool {
        let data = ctx.op(op);
        REGISTRY.lookup(data.dialect, data.name)
    }

    /// Check if an arena operation is pure and eligible for DCE removal.
    pub fn is_removable(ctx: &IrContext, op: OpRef) -> bool {
        Self::is_pure(ctx, op)
    }
}

/// Register a pure operation with simplified syntax.
///
/// # Example
/// ```text
/// register_pure_op!(arith.addi);
/// register_pure_op!(adt.struct_new);
/// ```
///
/// This expands to an inventory registration:
/// ```text
/// inventory::submit! {
///     op_interface::PureOps::register("arith", "addi")
/// }
/// ```
#[macro_export]
macro_rules! register_pure_op {
    // Legacy syntax: dialect.op_name
    ($dialect:ident . $op_name:ident) => {
        $crate::paste::paste! {
            ::inventory::submit! {
                $crate::op_interface::PureOps::register(
                    $crate::raw_ident_str!($dialect),
                    $crate::raw_ident_str!($op_name)
                )
            }
        }
    };
}

// =============================================================================
// IsolatedFromAbove Trait
// =============================================================================

/// Marker trait for operations whose regions cannot reference values from above.
///
/// Operations implementing this trait have regions that are "isolated" from the
/// enclosing scope - they cannot directly reference SSA values defined outside
/// the region. This is important for:
///
/// 1. **Verification**: Check that isolated regions don't have stale references
/// 2. **Rewrite passes**: Different handling for isolated vs non-isolated ops
/// 3. **Code generation**: Isolated regions can be compiled independently
///
/// Examples of isolated operations:
/// - `func.func` - function bodies (must receive values via parameters)
/// - `core.module` - module bodies
///
/// Examples of non-isolated operations (can capture outer values):
/// - `scf.if`, `scf.for` - control flow
/// - `closure.new` - closure creation
pub trait IsolatedFromAbove {}

/// Registration entry for isolated operations.
///
/// Use `inventory::submit!` to register isolated operations at the dialect definition site.
pub struct IsolatedFromAboveRegistration {
    /// Dialect name (e.g., "func", "core")
    pub dialect: &'static str,
    /// Operation name within the dialect (e.g., "func", "module")
    pub op_name: &'static str,
}

inventory::collect!(IsolatedFromAboveRegistration);

/// Internal registry built from inventory at first access.
struct IsolatedFromAboveRegistry {
    /// Lookup: (dialect, op_name) -> is_isolated
    isolated_ops: HashSet<(Symbol, Symbol)>,
}

impl IsolatedFromAboveRegistry {
    fn new() -> Self {
        Self {
            isolated_ops: HashSet::new(),
        }
    }

    fn lookup(&self, dialect: Symbol, op_name: Symbol) -> bool {
        self.isolated_ops.contains(&(dialect, op_name))
    }
}

/// Global registry for isolated operations, lazily built from inventory on first access.
static ISOLATED_REGISTRY: LazyLock<IsolatedFromAboveRegistry> = LazyLock::new(|| {
    let mut registry = IsolatedFromAboveRegistry::new();

    for reg in inventory::iter::<IsolatedFromAboveRegistration> {
        let dialect = Symbol::from_dynamic(reg.dialect);
        let op_name = Symbol::from_dynamic(reg.op_name);
        registry.isolated_ops.insert((dialect, op_name));
    }

    registry
});

/// Interface for querying operation isolation.
pub struct IsolatedFromAboveOps;

impl IsolatedFromAboveOps {
    /// Register an isolated operation (internal use by macro).
    ///
    /// Use the `register_isolated_op!` macro instead:
    /// ```text
    /// register_isolated_op!(func.func);
    /// ```
    #[doc(hidden)]
    pub const fn register(
        dialect: &'static str,
        op_name: &'static str,
    ) -> IsolatedFromAboveRegistration {
        IsolatedFromAboveRegistration { dialect, op_name }
    }

    /// Check if an arena operation's regions are isolated from above.
    pub fn is_isolated(ctx: &IrContext, op: OpRef) -> bool {
        let data = ctx.op(op);
        ISOLATED_REGISTRY.lookup(data.dialect, data.name)
    }
}

/// Register an isolated operation with simplified syntax.
///
/// # Example
/// ```text
/// register_isolated_op!(func.func);
/// ```
///
/// This expands to an inventory registration:
/// ```text
/// inventory::submit! {
///     op_interface::IsolatedFromAboveOps::register("func", "func")
/// }
/// ```
#[macro_export]
macro_rules! register_isolated_op {
    // Legacy syntax: dialect.op_name
    ($dialect:ident . $op_name:ident) => {
        $crate::paste::paste! {
            ::inventory::submit! {
                $crate::op_interface::IsolatedFromAboveOps::register(
                    $crate::raw_ident_str!($dialect),
                    $crate::raw_ident_str!($op_name)
                )
            }
        }
    };
}

// =============================================================================
// Control-flow interfaces
// =============================================================================

/// Failure to provide a complete control-flow interface answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ControlFlowInterfaceError {
    kind: ControlFlowInterfaceErrorKind,
    detail: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ControlFlowInterfaceErrorKind {
    Incomplete,
    NotApplicable,
}

impl ControlFlowInterfaceError {
    pub fn new(detail: impl Into<String>) -> Self {
        Self {
            kind: ControlFlowInterfaceErrorKind::Incomplete,
            detail: detail.into(),
        }
    }

    pub fn not_applicable(detail: impl Into<String>) -> Self {
        Self {
            kind: ControlFlowInterfaceErrorKind::NotApplicable,
            detail: detail.into(),
        }
    }

    pub fn is_not_applicable(&self) -> bool {
        self.kind == ControlFlowInterfaceErrorKind::NotApplicable
    }
}

impl fmt::Display for ControlFlowInterfaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.detail)
    }
}

impl std::error::Error for ControlFlowInterfaceError {}

/// Values forwarded along one control-flow edge.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ForwardedValues {
    values: SmallVec<[ValueRef; 4]>,
}

impl ForwardedValues {
    pub fn new(values: impl IntoIterator<Item = ValueRef>) -> Self {
        Self {
            values: values.into_iter().collect(),
        }
    }

    pub fn as_slice(&self) -> &[ValueRef] {
        &self.values
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// One raw block successor and the values forwarded to its block arguments.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BranchSuccessor {
    pub block: BlockRef,
    pub forwarded: ForwardedValues,
}

impl BranchSuccessor {
    pub fn new(block: BlockRef, forwarded: impl IntoIterator<Item = ValueRef>) -> Self {
        Self {
            block,
            forwarded: ForwardedValues::new(forwarded),
        }
    }
}

/// Complete block-successor answer for one branch operation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BranchSuccessors {
    edges: SmallVec<[BranchSuccessor; 2]>,
}

impl BranchSuccessors {
    pub fn new(edges: impl IntoIterator<Item = BranchSuccessor>) -> Self {
        Self {
            edges: edges.into_iter().collect(),
        }
    }

    pub fn as_slice(&self) -> &[BranchSuccessor] {
        &self.edges
    }
}

/// Source of a transfer through a region-holding operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RegionBranchPoint {
    /// First entry into the operation from its parent block.
    Parent,
    /// A terminator in one of the operation's semantic nested regions.
    Terminator(OpRef),
}

/// Target of a transfer through a region-holding operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RegionSuccessor {
    /// Enter a nested region.
    Region(RegionRef),
    /// Leave the operation and make its results available.
    Parent,
}

/// Complete successor answer for one region branch point.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RegionSuccessors {
    successors: SmallVec<[RegionSuccessor; 4]>,
}

impl RegionSuccessors {
    pub fn new(successors: impl IntoIterator<Item = RegionSuccessor>) -> Self {
        Self {
            successors: successors.into_iter().collect(),
        }
    }

    pub fn as_slice(&self) -> &[RegionSuccessor] {
        &self.successors
    }
}

/// Values receiving the forwarded operands of a region transfer.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SuccessorInputs {
    values: SmallVec<[ValueRef; 4]>,
}

impl SuccessorInputs {
    fn new(values: impl IntoIterator<Item = ValueRef>) -> Self {
        Self {
            values: values.into_iter().collect(),
        }
    }

    pub fn as_slice(&self) -> &[ValueRef] {
        &self.values
    }
}

/// Named operand-to-input transfer for one region edge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegionValueTransfer {
    pub successor: RegionSuccessor,
    pub forwarded: ForwardedValues,
    pub inputs: SuccessorInputs,
}

/// Object-safe exact-signature semantics for indirect calls.
///
/// The interface deliberately returns no signature for an ordinary indirect
/// transfer. Consumers must not reconstruct a contract from erased operands or
/// operation result types.
pub trait IndirectCallLike: Sync {
    fn exact_signature(&self, ctx: &IrContext, op: OpRef) -> Option<TypeRef>;
}

/// Typed exact-signature semantics supplied by an indirect-call operation
/// wrapper.
pub trait IndirectCallLikeModel: DialectOp {
    fn exact_signature(self, ctx: &IrContext) -> Option<TypeRef>;
}

fn indirect_call_like_model_signature<T: IndirectCallLikeModel>(
    ctx: &IrContext,
    op: OpRef,
) -> Option<TypeRef> {
    T::from_op(ctx, op)
        .ok()
        .and_then(|model| model.exact_signature(ctx))
}

/// Registry entry for [`IndirectCallLike`].
pub struct IndirectCallLikeRegistration {
    pub dialect: &'static str,
    pub op_name: &'static str,
    pub exact_signature: fn(&IrContext, OpRef) -> Option<TypeRef>,
}

impl IndirectCallLike for IndirectCallLikeRegistration {
    fn exact_signature(&self, ctx: &IrContext, op: OpRef) -> Option<TypeRef> {
        (self.exact_signature)(ctx, op)
    }
}

inventory::collect!(IndirectCallLikeRegistration);

static INDIRECT_CALL_LIKE_REGISTRY: LazyLock<
    HashMap<(Symbol, Symbol), &'static IndirectCallLikeRegistration>,
> = LazyLock::new(|| {
    let mut registry = HashMap::new();
    for registration in inventory::iter::<IndirectCallLikeRegistration> {
        let key = (
            Symbol::from_dynamic(registration.dialect),
            Symbol::from_dynamic(registration.op_name),
        );
        assert!(
            registry.insert(key, registration).is_none(),
            "duplicate IndirectCallLike registration for '{}.{}'",
            registration.dialect,
            registration.op_name,
        );
    }
    registry
});

/// Dynamic query and registration entry point for [`IndirectCallLike`].
pub struct IndirectCallLikeOps;

impl IndirectCallLikeOps {
    #[doc(hidden)]
    pub const fn register<T: IndirectCallLikeModel>() -> IndirectCallLikeRegistration {
        IndirectCallLikeRegistration {
            dialect: T::DIALECT_NAME,
            op_name: T::OP_NAME,
            exact_signature: indirect_call_like_model_signature::<T>,
        }
    }

    pub fn get(ctx: &IrContext, op: OpRef) -> Option<&'static dyn IndirectCallLike> {
        let data = ctx.op(op);
        INDIRECT_CALL_LIKE_REGISTRY
            .get(&(data.dialect, data.name))
            .map(|registration| *registration as &dyn IndirectCallLike)
    }

    /// Read an indirect transfer's exact callable signature, if it has one.
    pub fn exact_signature(ctx: &IrContext, op: OpRef) -> Option<TypeRef> {
        Self::get(ctx, op).and_then(|interface| interface.exact_signature(ctx, op))
    }
}

/// Object-safe block branch semantics.
pub trait Branch: Sync {
    fn successors(
        &self,
        ctx: &IrContext,
        op: OpRef,
    ) -> Result<BranchSuccessors, ControlFlowInterfaceError>;
}

/// Typed block-branch semantics supplied by a generated dialect operation wrapper.
pub trait BranchModel: DialectOp {
    fn successors(self, ctx: &IrContext) -> Result<BranchSuccessors, ControlFlowInterfaceError>;
}

fn branch_model_successors<T: BranchModel>(
    ctx: &IrContext,
    op: OpRef,
) -> Result<BranchSuccessors, ControlFlowInterfaceError> {
    let model = T::from_op(ctx, op).map_err(|error| {
        ControlFlowInterfaceError::new(format!(
            "malformed {}.{}: {error:?}",
            T::DIALECT_NAME,
            T::OP_NAME
        ))
    })?;
    model.successors(ctx)
}

pub type BranchSuccessorsFn =
    fn(&IrContext, OpRef) -> Result<BranchSuccessors, ControlFlowInterfaceError>;

/// Registry entry for [`Branch`].
pub struct BranchRegistration {
    pub dialect: &'static str,
    pub op_name: &'static str,
    pub successors: BranchSuccessorsFn,
}

impl Branch for BranchRegistration {
    fn successors(
        &self,
        ctx: &IrContext,
        op: OpRef,
    ) -> Result<BranchSuccessors, ControlFlowInterfaceError> {
        (self.successors)(ctx, op)
    }
}

inventory::collect!(BranchRegistration);

static BRANCH_REGISTRY: LazyLock<HashMap<(Symbol, Symbol), &'static BranchRegistration>> =
    LazyLock::new(|| {
        let mut registry = HashMap::new();
        for registration in inventory::iter::<BranchRegistration> {
            let key = (
                Symbol::from_dynamic(registration.dialect),
                Symbol::from_dynamic(registration.op_name),
            );
            assert!(
                registry.insert(key, registration).is_none(),
                "duplicate Branch registration for '{}.{}'",
                registration.dialect,
                registration.op_name,
            );
        }
        registry
    });

/// Dyn query and registration entry point for [`Branch`].
pub struct BranchOps;

impl BranchOps {
    #[doc(hidden)]
    pub const fn register<T: BranchModel>() -> BranchRegistration {
        BranchRegistration {
            dialect: T::DIALECT_NAME,
            op_name: T::OP_NAME,
            successors: branch_model_successors::<T>,
        }
    }

    pub fn get(ctx: &IrContext, op: OpRef) -> Option<&'static dyn Branch> {
        let data = ctx.op(op);
        BRANCH_REGISTRY
            .get(&(data.dialect, data.name))
            .map(|registration| *registration as &dyn Branch)
    }
}

/// Object-safe control-flow semantics between an operation and nested regions.
pub trait RegionBranch: Sync {
    fn successors(
        &self,
        ctx: &IrContext,
        op: OpRef,
        point: RegionBranchPoint,
    ) -> Result<RegionSuccessors, ControlFlowInterfaceError>;

    fn entry_successor_operands(
        &self,
        ctx: &IrContext,
        op: OpRef,
        successor: RegionSuccessor,
    ) -> Result<ForwardedValues, ControlFlowInterfaceError>;
}

/// Typed structured-region semantics supplied by a generated dialect operation wrapper.
pub trait RegionBranchModel: DialectOp {
    fn successors(
        self,
        ctx: &IrContext,
        point: RegionBranchPoint,
    ) -> Result<RegionSuccessors, ControlFlowInterfaceError>;

    fn entry_successor_operands(
        self,
        ctx: &IrContext,
        successor: RegionSuccessor,
    ) -> Result<ForwardedValues, ControlFlowInterfaceError>;
}

fn region_branch_model_successors<T: RegionBranchModel>(
    ctx: &IrContext,
    op: OpRef,
    point: RegionBranchPoint,
) -> Result<RegionSuccessors, ControlFlowInterfaceError> {
    let model = T::from_op(ctx, op).map_err(|error| {
        ControlFlowInterfaceError::new(format!(
            "malformed {}.{}: {error:?}",
            T::DIALECT_NAME,
            T::OP_NAME
        ))
    })?;
    model.successors(ctx, point)
}

fn region_branch_model_entry_operands<T: RegionBranchModel>(
    ctx: &IrContext,
    op: OpRef,
    successor: RegionSuccessor,
) -> Result<ForwardedValues, ControlFlowInterfaceError> {
    let model = T::from_op(ctx, op).map_err(|error| {
        ControlFlowInterfaceError::new(format!(
            "malformed {}.{}: {error:?}",
            T::DIALECT_NAME,
            T::OP_NAME
        ))
    })?;
    model.entry_successor_operands(ctx, successor)
}

pub type RegionSuccessorsFn =
    fn(&IrContext, OpRef, RegionBranchPoint) -> Result<RegionSuccessors, ControlFlowInterfaceError>;
pub type EntrySuccessorOperandsFn =
    fn(&IrContext, OpRef, RegionSuccessor) -> Result<ForwardedValues, ControlFlowInterfaceError>;

/// Registry entry for [`RegionBranch`].
pub struct RegionBranchRegistration {
    pub dialect: &'static str,
    pub op_name: &'static str,
    pub successors: RegionSuccessorsFn,
    pub entry_successor_operands: EntrySuccessorOperandsFn,
}

impl RegionBranch for RegionBranchRegistration {
    fn successors(
        &self,
        ctx: &IrContext,
        op: OpRef,
        point: RegionBranchPoint,
    ) -> Result<RegionSuccessors, ControlFlowInterfaceError> {
        (self.successors)(ctx, op, point)
    }

    fn entry_successor_operands(
        &self,
        ctx: &IrContext,
        op: OpRef,
        successor: RegionSuccessor,
    ) -> Result<ForwardedValues, ControlFlowInterfaceError> {
        (self.entry_successor_operands)(ctx, op, successor)
    }
}

inventory::collect!(RegionBranchRegistration);

static REGION_BRANCH_REGISTRY: LazyLock<
    HashMap<(Symbol, Symbol), &'static RegionBranchRegistration>,
> = LazyLock::new(|| {
    let mut registry = HashMap::new();
    for registration in inventory::iter::<RegionBranchRegistration> {
        let key = (
            Symbol::from_dynamic(registration.dialect),
            Symbol::from_dynamic(registration.op_name),
        );
        assert!(
            registry.insert(key, registration).is_none(),
            "duplicate RegionBranch registration for '{}.{}'",
            registration.dialect,
            registration.op_name,
        );
    }
    registry
});

/// Dyn query and transfer helpers for [`RegionBranch`].
pub struct RegionBranchOps;

impl RegionBranchOps {
    #[doc(hidden)]
    pub const fn register<T: RegionBranchModel>() -> RegionBranchRegistration {
        RegionBranchRegistration {
            dialect: T::DIALECT_NAME,
            op_name: T::OP_NAME,
            successors: region_branch_model_successors::<T>,
            entry_successor_operands: region_branch_model_entry_operands::<T>,
        }
    }

    pub fn get(ctx: &IrContext, op: OpRef) -> Option<&'static dyn RegionBranch> {
        let data = ctx.op(op);
        REGION_BRANCH_REGISTRY
            .get(&(data.dialect, data.name))
            .map(|registration| *registration as &dyn RegionBranch)
    }

    /// Build the named value transfer for one already-reported successor.
    pub fn value_transfer(
        ctx: &IrContext,
        op: OpRef,
        point: RegionBranchPoint,
        successor: RegionSuccessor,
    ) -> Result<RegionValueTransfer, ControlFlowInterfaceError> {
        let interface = Self::get(ctx, op).ok_or_else(|| {
            ControlFlowInterfaceError::new("operation has no RegionBranch registration")
        })?;
        let forwarded = match point {
            RegionBranchPoint::Parent => interface.entry_successor_operands(ctx, op, successor)?,
            RegionBranchPoint::Terminator(terminator) => {
                let terminator_interface = RegionBranchTerminatorOps::get(ctx, terminator)
                    .ok_or_else(|| {
                        ControlFlowInterfaceError::new(format!(
                            "terminator {terminator} has no RegionBranchTerminator registration"
                        ))
                    })?;
                terminator_interface.successor_operands(ctx, terminator, successor)?
            }
        };
        let inputs = match successor {
            RegionSuccessor::Parent => SuccessorInputs::new(ctx.op_results(op).iter().copied()),
            RegionSuccessor::Region(region) => {
                let entry = ctx.region(region).blocks.first().copied().ok_or_else(|| {
                    ControlFlowInterfaceError::new(format!(
                        "successor region {region} has no entry block"
                    ))
                })?;
                SuccessorInputs::new(ctx.block_args(entry).iter().copied())
            }
        };
        Ok(RegionValueTransfer {
            successor,
            forwarded,
            inputs,
        })
    }
}

/// Object-safe SSA forwarding semantics for a nested-region terminator.
pub trait RegionBranchTerminator: Sync {
    fn successor_operands(
        &self,
        ctx: &IrContext,
        op: OpRef,
        successor: RegionSuccessor,
    ) -> Result<ForwardedValues, ControlFlowInterfaceError>;
}

/// Typed region-exit forwarding supplied by a generated dialect operation wrapper.
pub trait RegionBranchTerminatorModel: DialectOp {
    fn successor_operands(
        self,
        ctx: &IrContext,
        successor: RegionSuccessor,
    ) -> Result<ForwardedValues, ControlFlowInterfaceError>;
}

fn region_terminator_model_operands<T: RegionBranchTerminatorModel>(
    ctx: &IrContext,
    op: OpRef,
    successor: RegionSuccessor,
) -> Result<ForwardedValues, ControlFlowInterfaceError> {
    let model = T::from_op(ctx, op).map_err(|error| {
        ControlFlowInterfaceError::new(format!(
            "malformed {}.{}: {error:?}",
            T::DIALECT_NAME,
            T::OP_NAME
        ))
    })?;
    model.successor_operands(ctx, successor)
}

pub type RegionSuccessorOperandsFn =
    fn(&IrContext, OpRef, RegionSuccessor) -> Result<ForwardedValues, ControlFlowInterfaceError>;

/// Registry entry for [`RegionBranchTerminator`].
pub struct RegionBranchTerminatorRegistration {
    pub dialect: &'static str,
    pub op_name: &'static str,
    pub successor_operands: RegionSuccessorOperandsFn,
}

impl RegionBranchTerminator for RegionBranchTerminatorRegistration {
    fn successor_operands(
        &self,
        ctx: &IrContext,
        op: OpRef,
        successor: RegionSuccessor,
    ) -> Result<ForwardedValues, ControlFlowInterfaceError> {
        (self.successor_operands)(ctx, op, successor)
    }
}

inventory::collect!(RegionBranchTerminatorRegistration);

static REGION_BRANCH_TERMINATOR_REGISTRY: LazyLock<
    HashMap<(Symbol, Symbol), &'static RegionBranchTerminatorRegistration>,
> = LazyLock::new(|| {
    let mut registry = HashMap::new();
    for registration in inventory::iter::<RegionBranchTerminatorRegistration> {
        let key = (
            Symbol::from_dynamic(registration.dialect),
            Symbol::from_dynamic(registration.op_name),
        );
        assert!(
            registry.insert(key, registration).is_none(),
            "duplicate RegionBranchTerminator registration for '{}.{}'",
            registration.dialect,
            registration.op_name,
        );
    }
    registry
});

/// Dyn query and registration entry point for [`RegionBranchTerminator`].
pub struct RegionBranchTerminatorOps;

impl RegionBranchTerminatorOps {
    #[doc(hidden)]
    pub const fn register<T: RegionBranchTerminatorModel>() -> RegionBranchTerminatorRegistration {
        RegionBranchTerminatorRegistration {
            dialect: T::DIALECT_NAME,
            op_name: T::OP_NAME,
            successor_operands: region_terminator_model_operands::<T>,
        }
    }

    pub fn get(ctx: &IrContext, op: OpRef) -> Option<&'static dyn RegionBranchTerminator> {
        let data = ctx.op(op);
        REGION_BRANCH_TERMINATOR_REGISTRY
            .get(&(data.dialect, data.name))
            .map(|registration| *registration as &dyn RegionBranchTerminator)
    }
}

// =============================================================================
// TypeAliasHint — dialect-provided alias name suggestions for printer
// =============================================================================

/// Dialect-provided hint for suggesting type alias names during printing.
///
/// Each dialect can register a hint that maps its types to suggested alias names.
/// The printer uses these hints when auto-generating type aliases.
pub struct TypeAliasHint {
    /// Dialect name this hint applies to (e.g., "adt").
    pub dialect: &'static str,
    /// Given a type belonging to this dialect, suggest an alias name.
    /// Returns `None` if no name can be suggested.
    pub suggest: fn(&IrContext, TypeRef) -> Option<Symbol>,
}

inventory::collect!(TypeAliasHint);

/// Query all registered `TypeAliasHint`s to find a suggested name for the given type.
pub fn suggest_type_alias_name(ctx: &IrContext, ty: TypeRef) -> Option<Symbol> {
    let data = ctx.types.get(ty);
    let dialect = data.dialect;
    for hint in inventory::iter::<TypeAliasHint> {
        if dialect.with_str(|s| s == hint.dialect)
            && let Some(name) = (hint.suggest)(ctx, ty)
        {
            return Some(name);
        }
    }
    None
}

// =============================================================================
// OpAsmFormat — custom assembly format for operations (print + parse)
// =============================================================================

/// Custom assembly format for an operation — bundles print + parse.
///
/// Modeled after MLIR's `hasCustomAssemblyFormat`. Register via `inventory::submit!`
/// at dialect definition sites. The printer/parser dispatch automatically routes to
/// the registered format.
pub struct OpAsmFormat {
    /// Dialect name (e.g., "func", "closure")
    pub dialect: &'static str,
    /// Operation name within the dialect (e.g., "func", "lambda")
    pub op_name: &'static str,
    /// Custom printer. Called instead of generic printing.
    pub print_fn: fn(&mut crate::printer::OpPrintHelper<'_, '_>, OpRef, usize) -> fmt::Result,
    /// Custom parser. Called after `dialect.op` and optional `@sym_name` are consumed.
    /// `results` and `sym_name` are already parsed by the generic parser.
    pub parse_fn: for<'a> fn(
        input: &mut &'a str,
        results: Vec<&'a str>,
        sym_name: Option<String>,
    ) -> winnow::ModalResult<crate::parser::raw::RawOperation<'a>>,
}

inventory::collect!(OpAsmFormat);

/// Global registry mapping (dialect, op_name) → OpAsmFormat, lazily built from inventory.
static ASM_FORMAT_REGISTRY: LazyLock<HashMap<(Symbol, Symbol), &'static OpAsmFormat>> =
    LazyLock::new(|| {
        let mut map = HashMap::new();
        for fmt in inventory::iter::<OpAsmFormat> {
            let dialect = Symbol::from_dynamic(fmt.dialect);
            let op_name = Symbol::from_dynamic(fmt.op_name);
            if map.contains_key(&(dialect, op_name)) {
                panic!(
                    "duplicate OpAsmFormat registration for '{}.{}'",
                    fmt.dialect, fmt.op_name
                );
            }
            map.insert((dialect, op_name), fmt);
        }
        map
    });

/// Look up a registered custom assembly format for the given operation.
pub fn lookup_asm_format(dialect: Symbol, op_name: Symbol) -> Option<&'static OpAsmFormat> {
    ASM_FORMAT_REGISTRY.get(&(dialect, op_name)).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_is_populated() {
        // Just verify the registry can be accessed without panicking
        let _ = &*REGISTRY;
    }

    #[test]
    fn test_unregistered_ops_are_not_pure() {
        // This test would need a database and operation, so we just verify the struct exists
        let _ = PureOps;
    }

    #[test]
    fn test_isolated_registry_is_populated() {
        // Just verify the registry can be accessed without panicking
        let _ = &*ISOLATED_REGISTRY;
    }

    #[test]
    fn test_unregistered_ops_are_not_isolated() {
        // This test would need a database and operation, so we just verify the struct exists
        let _ = IsolatedFromAboveOps;
    }
}
