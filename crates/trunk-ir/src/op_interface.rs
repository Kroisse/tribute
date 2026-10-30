//! Operation interface system for querying operation properties.
//!
//! Each interface is a trait declared with `#[op_interface]`. An operation
//! implements it with an `impl` and registers it with `#[implements(..)]`
//! on its `#[dialect]` declaration or with [`register_interfaces!`](crate::register_interfaces).
//! Properties without behavior are [operation flags](mod@crate::op_flag) instead.

use std::fmt;

use smallvec::SmallVec;

use crate::SymbolPath;
use crate::ops::DialectOp;
use crate::{BlockRef, IrContext, OpRef, RegionRef, TypeRef, ValueRef};
use trunk_ir_macros::op_interface;

/// Register interface implementations of operations or types that
/// `#[dialect]` does not declare, or of interfaces declared elsewhere.
///
/// ```text
/// register_interfaces!(Exit: CallableExit; Br: Branch);
/// register_interfaces!(core::Bytes: ManagedRefType);
/// ```
///
/// Each entry requires the interface's impl for the operation or type
/// wrapper, as `#[implements(..)]` on a `#[dialect]` declaration does.
#[macro_export]
macro_rules! register_interfaces {
    ($($wrapper:ty : $($interface:path),+ $(,)?);+ $(;)?) => {
        $($(
            $crate::__private::inventory::submit! {
                <$wrapper as $interface>::__TRUNK_IR_INTERFACE
            }
        )+)+
    };
}

// =============================================================================
// Symbol interfaces
// =============================================================================

/// A symbol definition: an operation that a symbol table names by its
/// `sym_name`, the counterpart of MLIR's `SymbolOpInterface`.
///
/// The definition's uses are the symbol references in its own attributes and
/// in its regions. Global DCE removes a discardable definition that no live
/// operation uses; an operation that does not implement this interface is
/// never removed by it, and the references it holds are roots.
#[op_interface]
pub trait SymbolDefinition: DialectOp {
    /// Whether the definition may be removed when nothing live uses it. A
    /// definition that is visible outside the module answers `false`.
    fn is_discardable(&self, _ctx: &IrContext) -> bool {
        true
    }

    /// Whether the definition declares a symbol defined elsewhere, without a
    /// body of its own.
    fn is_declaration(&self, ctx: &IrContext) -> bool {
        !ctx.op_has_regions(self.op_ref())
    }
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
    /// The signature of the callable whose body is `body`, one of this
    /// operation's regions.
    fn callable_signature(
        &self,
        ctx: &IrContext,
        body: crate::refs::RegionRef,
    ) -> Option<crate::dialect::func::FuncSig>;
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

    fn exits_callable(&self, ctx: &IrContext) -> Result<(), ControlFlowInterfaceError> {
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

impl DynCallableExit {
    /// Prove that an operation leaves the current callable.
    ///
    /// A missing registration and a failed typed query are both errors so
    /// generic consumers remain conservative.
    pub fn prove(ctx: &IrContext, op: OpRef) -> Result<(), ControlFlowInterfaceError> {
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
///
/// Queries assume the operation satisfies its schema, so an implementation
/// reads its callee through the wrapper's generated accessor.
#[op_interface]
pub trait CallLike: DialectOp {
    fn direct_callee<'a>(&self, ctx: &'a IrContext) -> &'a SymbolPath;
}

/// Semantic accessors for indirect calls.
///
/// An exact signature is optional on otherwise valid ordinary indirect
/// transfers. When it is absent, consumers must not reconstruct a contract
/// from erased operands or operation result types.
#[op_interface]
pub trait IndirectCallLike: DialectOp {
    /// The erased runtime callee or table-index value.
    fn indirect_callee(&self, ctx: &IrContext) -> Option<ValueRef> {
        ctx.op_operands(self.op_ref()).first().copied()
    }

    /// The arguments supplied to the transfer, excluding its callee.
    fn indirect_arguments<'a>(&self, ctx: &'a IrContext) -> Option<&'a [ValueRef]> {
        ctx.op_operands(self.op_ref()).get(1..)
    }

    fn exact_signature(&self, ctx: &IrContext) -> Option<TypeRef>;

    fn set_exact_signature(&self, ctx: &mut IrContext, signature: TypeRef) -> bool;
}

/// Block branch semantics.
#[op_interface]
pub trait Branch: DialectOp {
    fn successors(&self, ctx: &IrContext) -> Result<BranchSuccessors, ControlFlowInterfaceError>;
}

/// Control-flow semantics between an operation and nested regions.
#[op_interface]
pub trait RegionBranch: DialectOp {
    fn successors(
        &self,
        ctx: &IrContext,
        point: RegionBranchPoint,
    ) -> Result<RegionSuccessors, ControlFlowInterfaceError>;

    fn entry_successor_operands(
        &self,
        ctx: &IrContext,
        successor: RegionSuccessor,
    ) -> Result<ForwardedValues, ControlFlowInterfaceError>;
}

impl DynRegionBranch {
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
        &self,
        ctx: &IrContext,
        successor: RegionSuccessor,
    ) -> Result<ForwardedValues, ControlFlowInterfaceError>;
}
