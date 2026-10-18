//! Operation interface system for querying operation properties.
//!
//! This module provides an interface system similar to `type_interface.rs` but for operations.
//! It uses the `inventory` crate to build a registry of operation properties at compile time.

use rustc_hash::FxHashSet as HashSet;
use std::fmt;
use std::sync::LazyLock;

use smallvec::SmallVec;

use crate::ops::DialectOp;
use crate::{BlockRef, IrContext, OpRef, RegionRef, TypeRef, ValueRef};
use crate::{OpName, SymbolPath};
use trunk_ir_macros::op_interface;

/// Marker trait for pure operations (no side effects, safe to remove if unused).
///
/// Operations implementing this trait can be safely eliminated by DCE if their results are unused.
pub trait Pure {}

/// Registration entry for pure operations.
///
/// Use `inventory::submit!` to register pure operations at the dialect definition site.
pub struct PureOpRegistration {
    name: OpName,
}

inventory::collect!(PureOpRegistration);

/// Internal registry built from inventory at first access.
struct PureOpRegistry {
    /// Registered pure operations.
    pure_ops: HashSet<OpName>,
}

impl PureOpRegistry {
    fn new() -> Self {
        Self {
            pure_ops: HashSet::default(),
        }
    }

    fn lookup(&self, name: OpName) -> bool {
        self.pure_ops.contains(&name)
    }
}

/// Global registry, lazily built from inventory on first access.
static REGISTRY: LazyLock<PureOpRegistry> = LazyLock::new(|| {
    let mut registry = PureOpRegistry::new();

    for reg in inventory::iter::<PureOpRegistration> {
        registry.pure_ops.insert(reg.name);
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
    /// register_pure_op!(Addi);
    /// ```
    #[doc(hidden)]
    pub const fn register<T: DialectOp>() -> PureOpRegistration {
        PureOpRegistration { name: T::NAME }
    }

    /// Check if an arena operation is pure (no side effects, safe to remove if unused).
    pub fn is_pure(ctx: &IrContext, op: OpRef) -> bool {
        let data = ctx.op(op);
        REGISTRY.lookup(data.name)
    }

    /// Check if an arena operation is pure and eligible for DCE removal.
    pub fn is_removable(ctx: &IrContext, op: OpRef) -> bool {
        Self::is_pure(ctx, op)
    }
}

/// Register pure operations by their typed wrappers.
///
/// # Example
/// ```text
/// register_pure_op!(Addi, Subi);
/// ```
///
/// Each wrapper expands to an inventory registration:
/// ```text
/// inventory::submit! {
///     op_interface::PureOps::register::<Addi>()
/// }
/// ```
#[macro_export]
macro_rules! register_pure_op {
    ($($op:path),+ $(,)?) => {
        $(
            $crate::inventory::submit! {
                $crate::op_interface::PureOps::register::<$op>()
            }
        )+
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
    name: OpName,
}

inventory::collect!(IsolatedFromAboveRegistration);

/// Internal registry built from inventory at first access.
struct IsolatedFromAboveRegistry {
    /// Registered isolated operations.
    isolated_ops: HashSet<OpName>,
}

impl IsolatedFromAboveRegistry {
    fn new() -> Self {
        Self {
            isolated_ops: HashSet::default(),
        }
    }

    fn lookup(&self, name: OpName) -> bool {
        self.isolated_ops.contains(&name)
    }
}

/// Global registry for isolated operations, lazily built from inventory on first access.
static ISOLATED_REGISTRY: LazyLock<IsolatedFromAboveRegistry> = LazyLock::new(|| {
    let mut registry = IsolatedFromAboveRegistry::new();

    for reg in inventory::iter::<IsolatedFromAboveRegistration> {
        registry.isolated_ops.insert(reg.name);
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
    /// register_isolated_op!(Func);
    /// ```
    #[doc(hidden)]
    pub const fn register<T: DialectOp>() -> IsolatedFromAboveRegistration {
        IsolatedFromAboveRegistration { name: T::NAME }
    }

    /// Check if an arena operation's regions are isolated from above.
    pub fn is_isolated(ctx: &IrContext, op: OpRef) -> bool {
        let data = ctx.op(op);
        ISOLATED_REGISTRY.lookup(data.name)
    }
}

/// Register isolated operations by their typed wrappers.
///
/// # Example
/// ```text
/// register_isolated_op!(Func);
/// ```
///
/// Each wrapper expands to an inventory registration:
/// ```text
/// inventory::submit! {
///     op_interface::IsolatedFromAboveOps::register::<Func>()
/// }
/// ```
#[macro_export]
macro_rules! register_isolated_op {
    ($($op:path),+ $(,)?) => {
        $(
            $crate::inventory::submit! {
                $crate::op_interface::IsolatedFromAboveOps::register::<$op>()
            }
        )+
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

/// Exact callable region signature supplied by the owning dialect.
/// A registered owner with an invalid signature is still an ownership boundary.
#[op_interface]
pub trait CallableOwner: DialectOp {
    fn callable_signature(self, ctx: &IrContext) -> Option<crate::dialect::func::FuncSig>;
}

/// Static queries and registration entry point for [`CallableOwner`].
pub struct CallableOwnerOps;

impl CallableOwnerOps {
    #[doc(hidden)]
    pub const fn register<T: CallableOwner>() -> __CallableOwnerVTable {
        T::__TRUNK_IR_INTERFACE
    }

    /// Outer None means unregistered; Some(None) means a malformed owner.
    pub fn signature(ctx: &IrContext, op: OpRef) -> Option<Option<crate::dialect::func::FuncSig>> {
        DynCallableOwner::get(ctx, op).map(|owner| owner.callable_signature(ctx))
    }
}

/// Semantics of operations that leave the current callable.
///
/// Callable exits are distinct from generic structured-region terminators:
/// a yield, break, or continue can transfer control to an enclosing structured
/// operation, while a callable exit cannot execute a following operation in
/// that callable.
///
/// The default contract rejects results, raw block successors, and nested
/// regions. A model may explicitly permit regions and supplies only its
/// dialect-specific checks; registration alone is never proof of termination.
#[op_interface]
pub trait CallableExit: DialectOp {
    /// Whether this callable exit may own structured regions.
    fn allows_nested_regions(&self, _ctx: &IrContext) -> bool {
        false
    }

    /// Check the dialect-specific part of the callable-exit contract.
    fn verify_callable_exit(&self, _ctx: &IrContext) -> Result<(), ControlFlowInterfaceError> {
        Ok(())
    }

    fn exits_callable(self, ctx: &IrContext) -> Result<(), ControlFlowInterfaceError> {
        let op = self.op_ref();
        if !ctx.op_results(op).is_empty() {
            return Err(ControlFlowInterfaceError::new(
                "CallableExit must not produce SSA results",
            ));
        }
        if ctx.op_has_successors(op) {
            return Err(ControlFlowInterfaceError::new(
                "CallableExit must not have block successors",
            ));
        }
        if !self.allows_nested_regions(ctx) && ctx.op_has_regions(op) {
            return Err(ControlFlowInterfaceError::new(
                "CallableExit must not contain nested regions",
            ));
        }
        self.verify_callable_exit(ctx)
    }
}

/// Static queries and registration entry point for [`CallableExit`].
pub struct CallableExitOps;

impl CallableExitOps {
    #[doc(hidden)]
    pub const fn register<T: CallableExit>() -> __CallableExitVTable {
        T::__TRUNK_IR_INTERFACE
    }

    /// Prove that an operation leaves the current callable.
    ///
    /// A missing registration and a failed typed query are both errors so
    /// generic consumers remain conservative.
    pub fn exits_callable(ctx: &IrContext, op: OpRef) -> Result<(), ControlFlowInterfaceError> {
        DynCallableExit::get(ctx, op)
            .ok_or_else(|| {
                ControlFlowInterfaceError::new("operation has no CallableExit registration")
            })?
            .exits_callable(ctx)
    }
}

/// Direct-call semantics supplied by a generated operation wrapper.
///
/// The callee is the one symbol reference of the operation that transfers
/// control to its target. Any other reference takes the target's address.
#[op_interface]
pub trait CallLike: DialectOp {
    /// `None` for a malformed call; read fallibly since unverified IR may be
    /// queried.
    fn direct_callee(self, ctx: &IrContext) -> Option<SymbolPath> {
        ctx.op_attrs(self.op_ref())
            .get_symbol_ref("callee")
            .cloned()
    }
}

/// Static queries and registration entry point for [`CallLike`].
pub struct CallLikeOps;

impl CallLikeOps {
    #[doc(hidden)]
    pub const fn register<T: CallLike>() -> __CallLikeVTable {
        T::__TRUNK_IR_INTERFACE
    }

    /// The callee of a direct call, or `None` if `op` is not one or is
    /// malformed.
    pub fn callee(ctx: &IrContext, op: OpRef) -> Option<SymbolPath> {
        DynCallLike::get(ctx, op).and_then(|call| call.direct_callee(ctx))
    }
}

/// Semantic accessors for indirect calls.
///
/// An exact signature is optional on otherwise valid ordinary indirect
/// transfers. When it is absent, consumers must not reconstruct a contract
/// from erased operands or operation result types.
#[op_interface]
pub trait IndirectCallLike: DialectOp {
    /// The erased runtime callee or table-index value.
    fn indirect_callee(self, ctx: &IrContext) -> Option<ValueRef> {
        ctx.op_operands(self.op_ref()).first().copied()
    }

    /// The arguments supplied to the transfer, excluding its callee.
    fn indirect_arguments(self, ctx: &IrContext) -> Option<&[ValueRef]> {
        ctx.op_operands(self.op_ref()).get(1..)
    }

    fn exact_signature(self, ctx: &IrContext) -> Option<TypeRef>;

    fn set_exact_signature(self, ctx: &mut IrContext, signature: TypeRef) -> bool;
}

/// Static queries and registration entry point for [`IndirectCallLike`].
pub struct IndirectCallLikeOps;

impl IndirectCallLikeOps {
    #[doc(hidden)]
    pub const fn register<T: IndirectCallLike>() -> __IndirectCallLikeVTable {
        T::__TRUNK_IR_INTERFACE
    }

    /// Read an indirect transfer's exact callable signature, if it has one.
    pub fn exact_signature(ctx: &IrContext, op: OpRef) -> Option<TypeRef> {
        DynIndirectCallLike::get(ctx, op).and_then(|call| call.exact_signature(ctx))
    }

    /// Set an indirect transfer's exact callable signature.
    pub fn set_exact_signature(ctx: &mut IrContext, op: OpRef, signature: TypeRef) -> bool {
        DynIndirectCallLike::get(ctx, op)
            .is_some_and(|call| call.set_exact_signature(ctx, signature))
    }

    /// Read an indirect transfer's erased runtime callee or table-index value.
    pub fn callee(ctx: &IrContext, op: OpRef) -> Option<ValueRef> {
        DynIndirectCallLike::get(ctx, op).and_then(|call| call.indirect_callee(ctx))
    }

    /// Read arguments supplied to an indirect transfer, excluding its callee.
    pub fn arguments(ctx: &IrContext, op: OpRef) -> Option<&[ValueRef]> {
        DynIndirectCallLike::get(ctx, op).and_then(|call| call.indirect_arguments(ctx))
    }
}

/// Block branch semantics.
#[op_interface]
pub trait Branch: DialectOp {
    fn successors(self, ctx: &IrContext) -> Result<BranchSuccessors, ControlFlowInterfaceError>;
}

/// Registration entry point for [`Branch`].
pub struct BranchOps;

impl BranchOps {
    #[doc(hidden)]
    pub const fn register<T: Branch>() -> __BranchVTable {
        T::__TRUNK_IR_INTERFACE
    }
}

/// Control-flow semantics between an operation and nested regions.
#[op_interface]
pub trait RegionBranch: DialectOp {
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

/// Transfer helpers and registration entry point for [`RegionBranch`].
pub struct RegionBranchOps;

impl RegionBranchOps {
    #[doc(hidden)]
    pub const fn register<T: RegionBranch>() -> __RegionBranchVTable {
        T::__TRUNK_IR_INTERFACE
    }

    /// Build the named value transfer for one already-reported successor.
    pub fn value_transfer(
        ctx: &IrContext,
        op: OpRef,
        point: RegionBranchPoint,
        successor: RegionSuccessor,
    ) -> Result<RegionValueTransfer, ControlFlowInterfaceError> {
        let interface = DynRegionBranch::get(ctx, op).ok_or_else(|| {
            ControlFlowInterfaceError::new("operation has no RegionBranch registration")
        })?;
        let forwarded = match point {
            RegionBranchPoint::Parent => interface.entry_successor_operands(ctx, successor)?,
            RegionBranchPoint::Terminator(terminator) => {
                DynRegionBranchTerminator::get(ctx, terminator)
                    .ok_or_else(|| {
                        ControlFlowInterfaceError::new(format!(
                            "terminator {terminator} has no RegionBranchTerminator registration"
                        ))
                    })?
                    .successor_operands(ctx, successor)?
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

/// SSA forwarding semantics for a nested-region terminator.
#[op_interface]
pub trait RegionBranchTerminator: DialectOp {
    fn successor_operands(
        self,
        ctx: &IrContext,
        successor: RegionSuccessor,
    ) -> Result<ForwardedValues, ControlFlowInterfaceError>;
}

/// Registration entry point for [`RegionBranchTerminator`].
pub struct RegionBranchTerminatorOps;

impl RegionBranchTerminatorOps {
    #[doc(hidden)]
    pub const fn register<T: RegionBranchTerminator>() -> __RegionBranchTerminatorVTable {
        T::__TRUNK_IR_INTERFACE
    }
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
