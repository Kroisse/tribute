//! Verified source-logical callable and direct-style control dialect.
//!
//! `tribute_control` is the target-independent boundary between typed frontend
//! lowering and shared CPS legalization. It deliberately contains no physical
//! evidence, closure-environment, continuation, or backend carrier layout.

use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;
use std::fmt;
use std::ops::ControlFlow;
use trunk_ir::attr_kind::SymbolRef;
use trunk_ir::attr_kind::Type;
use trunk_ir::op_flag::{IsolatedFromAbove, Pure};

use crate::dialect::adt;
use itertools::Itertools;
use trunk_ir::analysis::AnalysisCache;
use trunk_ir::dialect::core;
use trunk_ir::op_def::OpDef;
use trunk_ir::op_interface::{
    CallLike, CallableExit, DynRegionBranch, RegionBranchPoint, RegionSuccessor,
};
use trunk_ir::op_schema::OpSchema;
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{BlockRef, OpRef, RegionRef, TypeRef, ValueDef, ValueRef};
use trunk_ir::rewrite::Module;
use trunk_ir::symbol_table::{SymbolTable, qualified_name};
use trunk_ir::types::{Attribute, AttributeMap, Location, StringRef, TypeDataBuilder};
use trunk_ir::walk::{WalkAction, walk_op};
use trunk_ir::{IrContext, Symbol, SymbolPath};

mod callable_origin;
mod row;

use callable_origin::validate_callable_origins;
pub use callable_origin::{CallableArgumentOrigin, CallableQuery, CallableResultOrigin};

pub use row::{
    ABILITY_ROW_ATTR, INFERRED_ROW_ATTR, QUANTIFIED_ROWS_ATTR, ROW_ARGS_ATTR, ROW_BINDERS_ATTR,
    ROW_UNIONS_ATTR, Row, RowArgs, RowBinder, RowUnion, assign_conventions, find_substituted_rows,
    logical_tuple_elements, logical_tuple_name, logical_tuple_type, quantified_count,
    quantified_rows, row_args, row_binders, row_unions, substitute_rows, visit_rows,
};

/// Exact type-attribute key used by logical callable types.
pub const CALLING_CONVENTION_ATTR: &str = "tribute.calling_convention";

/// Exact semantic identity copied from the trusted compiler-intrinsic registry.
pub const COMPILER_INTRINSIC_ATTR: &str = "tribute.compiler_intrinsic";

/// Source-logical callable convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum CallingConvention {
    Direct = 0,
    EvidenceDirect = 1,
    Cps = 2,
}

impl CallingConvention {
    fn keyword(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::EvidenceDirect => "evidence_direct",
            Self::Cps => "cps",
        }
    }

    fn from_keyword(keyword: &str) -> Option<Self> {
        match keyword {
            "direct" => Some(Self::Direct),
            "evidence_direct" => Some(Self::EvidenceDirect),
            "cps" => Some(Self::Cps),
            _ => None,
        }
    }
}

impl TryFrom<i128> for CallingConvention {
    type Error = i128;

    fn try_from(code: i128) -> Result<Self, Self::Error> {
        match code {
            0 => Ok(Self::Direct),
            1 => Ok(Self::EvidenceDirect),
            2 => Ok(Self::Cps),
            _ => Err(code),
        }
    }
}

#[trunk_ir::dialect]
mod tribute_control {
    // Types
    struct ResumeToken<Input, Answer>;

    // FuncSig operations
    // Only the named function is isolated. Lambda intentionally remains
    // non-isolated so its exact external-reference/capture set can be validated.
    /// A named source function. A parameter is a callable origin when its
    /// type is the parameter type `type` declares.
    ///
    /// ```text
    /// tribute_control.func @id(%x: T) -> R convention(direct) {
    ///   ...
    ///   tribute_control.return %value
    /// }
    /// ```
    ///
    /// - **Shape:** no operands or results. `sym_name: String` and
    ///   `type: tribute_control.func_sig<(Params...) -> Result>` are required.
    ///   A declaration has no region. A definition has one single-block
    ///   `body` that takes only the source parameters as block arguments and
    ///   ends in `tribute_control.return`. Non-control attributes such as a
    ///   foreign ABI are preserved.
    /// - **Meaning:** defines the logical signature and row of a source named
    ///   function, without hidden evidence, environment, or
    ///   `ContinuationFrame<R>`. Class specialization computes the convention
    ///   from the row and writes it.
    /// - **Verification:** the local verifier matches the attributes, region,
    ///   block arguments, and return type against the callable type.
    ///   Whole-IR verification checks symbol uniqueness, that every tail in
    ///   the definition is in the range of `row_binders`, that the indices of
    ///   `row_unions` are in range, and that each row instance resolves to an
    ///   `ability.decl`.
    /// - **Row attributes:** optional `row_binders`, `row_unions`, and
    ///   `inferred_row` (`new-plans/ir.md`, "Ability row").
    /// - **Ownership and value flow:** the body is isolated from above. Only
    ///   source locals and parameters enter as block arguments.
    /// - **Location:** the whole span of the source function or extern
    ///   declaration.
    #[asm_format]
    #[flags(IsolatedFromAbove)]
    #[implements(CallableArgumentOrigin)]
    fn func<S: FuncSig>(
        sym_name: Attr<String>,
        r#type: Attr<TypeOf<S>>,
        row_binders: Option<Attr<[RowBinder]>>,
        row_unions: Option<Attr<[RowUnion]>>,
        inferred_row: Option<Attr<bool>>,
    ) {
        #[region(body?)]
        {}
    }

    /// A source lambda. Its result is a callable origin, and so is a
    /// parameter whose type is the one its result type declares.
    ///
    /// ```text
    /// %f = tribute_control.lambda(%x: T) -> R convention(direct) captures [%capture0, ...] {
    ///   ...
    ///   tribute_control.return %value
    /// }
    /// ```
    ///
    /// - **Shape:** the operands are the typechecked captures in source
    ///   order. The result is one `tribute_control.func_sig`, and no
    ///   attribute is required. The single-block body takes only the source
    ///   parameters as block arguments and ends in `tribute_control.return`.
    /// - **Meaning:** creates a source lambda. The body uses the lexical
    ///   captures and the parameters.
    /// - **Verification:** the local verifier matches the result signature,
    ///   block arguments, and return type. Whole-IR verification checks that
    ///   the body's external SSA references are exactly the captures, that
    ///   each element of `quantified_rows` is a `local` binder of the
    ///   enclosing `func` and each `local` binder is listed by one lambda
    ///   only, and that only `instantiate` uses the result of a quantifying
    ///   lambda. Passing it as a capture of an inner lambda is not a use; the
    ///   use in that lambda's body must be an `instantiate`.
    /// - **Row attributes:** optional `quantified_rows` (`new-plans/ir.md`,
    ///   "지역 람다의 양화와 인스턴스화").
    /// - **Ownership and value flow:** the captures are ordinary SSA uses,
    ///   and there is no other hidden operand.
    /// - **Location:** the whole span of the source lambda.
    #[asm_format]
    #[flags(Pure)]
    #[implements(CallableResultOrigin, CallableArgumentOrigin)]
    fn lambda(quantified_rows: Option<Attr<[u32]>>, captures: Variadic<_>) -> Value<impl FuncSig> {
        #[region(body)]
        {}
    }

    /// One use of a row-polymorphic `let` lambda. Its result is a callable
    /// origin when the lambda is.
    ///
    /// ```text
    /// %s = tribute_control.instantiate %lambda {row_args = [{tail = 0}]} : !Callable
    /// ```
    ///
    /// - **Shape:** the operand is one result of a `tribute_control.lambda`
    ///   with `quantified_rows`. The result is one `tribute_control.func_sig`,
    ///   `row_args` is required, and the operation is a non-terminator with no
    ///   region.
    /// - **Meaning:** one use of a row-polymorphic lambda bound by `let`. For
    ///   each row variable the local scheme quantifies, it fixes the row this
    ///   use passes.
    /// - **Verification:** the local verifier checks the shape. Whole-IR
    ///   verification checks that the operand is the result of a lambda with
    ///   `quantified_rows`, that `row_args` has as many entries as that
    ///   attribute, and that the result is the lambda's type with `row_args`
    ///   substituted (`new-plans/ir.md`, "Row 인자와 치환"), convention
    ///   included. The tails and instances of the row arguments are checked in
    ///   the scope of the enclosing definition, like any other row.
    /// - **Ownership and value flow:** the operand and result are the same
    ///   callable value; no new value is made. CPS legalization replaces the
    ///   result with the operand.
    /// - **Location:** the source span that refers to the generalized
    ///   binding.
    #[verify]
    #[flags(Pure)]
    #[implements(CallableResultOrigin)]
    fn instantiate<S: FuncSig>(row_args: Attr<[Row]>, lambda: Value<S>) -> Value<impl FuncSig> {}

    /// A named function as a value. Its result is a callable origin when the
    /// function has a body or is a registered compiler intrinsic.
    ///
    /// ```text
    /// %f = tribute_control.func_ref {func_ref = @id} : !Callable
    /// ```
    ///
    /// - **Shape:** no operands or regions, and not a terminator.
    ///   `func_ref: Symbol` is required, and the result is one
    ///   `tribute_control.func_sig`.
    /// - **Meaning:** makes a named function a first-class source value,
    ///   which source needs because it can use a named function as a
    ///   higher-order value. The result has the target's source signature;
    ///   once conventions are assigned, its convention may equal or be
    ///   stronger than the target worker's. Its outer row may also be a wider
    ///   row that contains the target's row: this is the subsumption that
    ///   passes a pure function where a callable with an open row is
    ///   expected.
    /// - **Verification:** the local verifier checks the attribute and the
    ///   result shape. Whole-IR verification checks symbol resolution; that
    ///   the result equals the target type with `row_args` substituted
    ///   (`new-plans/ir.md`, "Row 인자와 치환") apart from the outer row and
    ///   convention; that the outer row holds each instance of the
    ///   substituted row at least as many times and, if that row has a tail,
    ///   the same tail; and, once conventions are assigned, that the outer
    ///   convention is not weaker than the target's.
    /// - **Ownership and value flow:** the result is an ordinary SSA value.
    /// - **Location:** the source reference span that uses the named function
    ///   as a value.
    #[verify]
    #[flags(Pure)]
    #[implements(CallableResultOrigin)]
    fn func_ref(func_ref: Attr<SymbolRef>, row_args: Option<Attr<[Row]>>) -> Value<impl FuncSig> {}

    /// A direct call. Its result is a callable origin when the callee has a
    /// body or is a registered compiler intrinsic.
    ///
    /// ```text
    /// %result = tribute_control.call %arg0, ... {callee = @f} : Result
    /// ```
    ///
    /// - **Shape:** the source arguments in declaration order, one
    ///   source-logical result, `callee: Symbol`, and optional
    ///   `evidence_plan` (`new-plans/ir.md`, "Evidence 선택 속성") and
    ///   `row_args` (`new-plans/ir.md`, "Row 인자와 치환"). A non-terminator
    ///   with no region.
    /// - **Meaning:** calls a named source callable directly.
    /// - **Verification:** the local verifier checks the attributes and the
    ///   resolved operands and result. Whole-IR verification checks that
    ///   `row_args` has as many entries as the callee's quantified binders,
    ///   and that the operands and result equal the callee type with
    ///   `row_args` substituted.
    /// - **Ownership and value flow:** the arguments and result are ordinary
    ///   SSA values, and there is no hidden operand.
    /// - **Location:** the source call span, including the callee and the
    ///   arguments.
    #[verify]
    #[implements(CallLike, CallableResultOrigin)]
    fn call(
        callee: Attr<SymbolRef>,
        row_args: Option<Attr<[Row]>>,
        evidence_plan: Option<Attr<[EvidenceStep]>>,
        args: Variadic<_>,
    ) -> Value<_> {
    }

    /// An indirect call. Its result is a callable origin when its callee is.
    ///
    /// ```text
    /// %result = tribute_control.call_indirect %callee, %arg0, ... : Result
    /// ```
    ///
    /// - **Shape:** a `tribute_control.func_sig` callee, the source
    ///   arguments, the one source-logical result of the callee type, and an
    ///   optional `evidence_plan` (`new-plans/ir.md`, "Evidence 선택 속성").
    ///   A non-terminator with no region.
    /// - **Meaning:** calls a callable obtained from a lambda, a `func_ref`,
    ///   a parameter, a capture, a call result, a data projection, or an
    ///   ability operation result.
    /// - **Verification:** the local verifier matches the callee signature
    ///   with the argument and result types. Whole-IR verification follows
    ///   the callee to check that it is a callable origin, a value whose
    ///   source signature comes from a declaration (`new-plans/ir.md`,
    ///   "Callable 출처").
    /// - **Ownership and value flow:** the callee and arguments are ordinary
    ///   SSA uses. There is no environment, evidence, or
    ///   `ContinuationFrame<R>`.
    /// - **Location:** the source indirect-call span, including the callee
    ///   and the arguments.
    #[verify]
    #[implements(CallableResultOrigin)]
    fn call_indirect<S: FuncSig>(
        evidence_plan: Option<Attr<[EvidenceStep]>>,
        callee: Value<S>,
        args: Values<S::Inputs>,
    ) -> Value<S::Result> {
    }

    /// A proper tail call of a named source callable, from source `become`.
    ///
    /// ```text
    /// tribute_control.tail_call %arg0, ... {callee = @f}
    /// ```
    ///
    /// - **Shape:** the source arguments in declaration order,
    ///   `callee: Symbol`, and optional `evidence_plan` (`new-plans/ir.md`,
    ///   "Evidence 선택 속성") and `row_args` (`new-plans/ir.md`, "Row 인자와
    ///   치환"). A terminator with no result or region, registered as a
    ///   `CallableExit`.
    /// - **Placement:** it may end the body of a `tribute_control.func` or
    ///   `lambda`, and in place of `scf.yield` it may end an arm of a
    ///   one-result `scf.if` in tail position. That arm is a terminal region,
    ///   and only the arms that end in `scf.yield` produce the operation's
    ///   result. An `scf.if` is in tail position when its result flows
    ///   directly into the terminator of the enclosing region: the `return`
    ///   of a callable body, or the `scf.yield` of an outer arm that is
    ///   itself in tail position. It is invalid inside a
    ///   `tribute_control.handle` body or handler region.
    /// - **Meaning:** a proper tail call of a named source callable. When the
    ///   call finishes, the callee's result becomes the enclosing callable's
    ///   result, and no caller frame remains.
    /// - **Verification:** the local verifier checks the attributes, the
    ///   placement rule, and that the callee result type equals the enclosing
    ///   callable's result type. Whole-IR verification, as for
    ///   `tribute_control.call`, matches the callee signature with `row_args`
    ///   substituted against the operands and the enclosing callable's
    ///   result, and once conventions are assigned, checks the convention
    ///   order. Typechecking owns the source diagnostics for tail position and
    ///   the handler rule; this check confirms that lowering preserved its
    ///   result.
    /// - **Ownership and value flow:** the arguments are ordinary SSA uses,
    ///   and there is no hidden operand. Shared legalization turns it into a
    ///   proper tail transfer for every combination of conventions
    ///   (`new-plans/cps-effects.md`, "Source proper tail call의 적법화").
    /// - **Location:** the source span including the `become` keyword and
    ///   the call.
    #[verify]
    #[implements(CallLike, CallableExit)]
    fn tail_call(
        callee: Attr<SymbolRef>,
        row_args: Option<Attr<[Row]>>,
        evidence_plan: Option<Attr<[EvidenceStep]>>,
        args: Variadic<_>,
    ) {
    }

    /// A proper tail call of a callable value, from source `become`.
    ///
    /// ```text
    /// tribute_control.tail_call_indirect %callee, %arg0, ...
    /// ```
    ///
    /// - **Shape:** a `tribute_control.func_sig` callee, the source
    ///   arguments, and an optional `evidence_plan` (`new-plans/ir.md`,
    ///   "Evidence 선택 속성"). A terminator with no result or region,
    ///   registered as a `CallableExit`.
    /// - **Placement:** the same as `tribute_control.tail_call`.
    /// - **Meaning:** a proper tail call of a callable value.
    /// - **Verification:** the local verifier checks the callee signature and
    ///   argument types, the placement rule, that the callee result type
    ///   equals the enclosing callable's result type, and the shape of
    ///   `evidence_plan`. Once conventions are assigned, the callee type's
    ///   convention cannot be stronger than the enclosing callable's.
    ///   Typechecking owns whether the selection matches the ability row.
    /// - **Ownership and value flow:** as for `tribute_control.call_indirect`,
    ///   the environment, evidence, and `ContinuationFrame<R>` are not
    ///   operands.
    /// - **Location:** the source span including the `become` keyword and
    ///   the indirect call.
    #[verify]
    #[implements(CallableExit)]
    fn tail_call_indirect<S: FuncSig>(
        evidence_plan: Option<Attr<[EvidenceStep]>>,
        callee: Value<S>,
        args: Values<S::Inputs>,
    ) {
    }

    /// Return the enclosing callable's logical result.
    ///
    /// ```text
    /// tribute_control.return %value
    /// ```
    ///
    /// - **Shape:** takes one source-logical result and has no results,
    ///   attributes, or regions. It is the terminator of a
    ///   `tribute_control.func` or `lambda` body and is invalid anywhere else.
    /// - **Meaning:** returns the enclosing callable's logical result.
    /// - **Local verification:** matches the operand type against the
    ///   enclosing callable's result.
    /// - **Ownership and value flow:** consumes an ordinary SSA value.
    /// - **Location:** the span of the source return or of the implicit
    ///   body-result expression.
    fn r#return(value: Value<_>) {}

    // Direct-style control operations
    /// An ability operation. Its result is a callable origin when it claims
    /// the kind of the declaration it names and has that declaration's
    /// result type.
    ///
    /// ```text
    /// %result = tribute_control.perform %arg0, ... {
    ///   ability = @State,
    ///   op_name = "get",
    ///   operation_kind = "op"
    /// } : ResultType
    /// ```
    ///
    /// - **Operands:** zero or more already evaluated source arguments in
    ///   declaration order. The values keep their logical types; conversion
    ///   handles tuple packing and erasure.
    /// - **Results:** exactly one logical operation result. A source `Never`
    ///   result is `core.never`, and does not select a physical `Never`
    ///   control carrier.
    /// - **Attributes:** `ability: SymbolRef`, `op_name: String`, and
    ///   `operation_kind: String` are required. `operation_kind` is exactly
    ///   `fn` or `op`, copied from the typechecked operation declaration. It
    ///   is source-semantic metadata, not a lowering hint inferred from a body
    ///   or a use site. Every source ability invocation uses this operation,
    ///   and shared conversion does not reclassify the kind.
    /// - **Regions and block arguments:** none.
    /// - **Terminator:** no. In direct-style IR this operation is not a
    ///   terminator.
    /// - **Meaning:** invokes a source operation with its declared kind. With
    ///   `operation_kind = "fn"`, the selected handler's result automatically
    ///   resumes direct-style evaluation; shared conversion captures no
    ///   continuation and uses the existing tail dispatch path. With
    ///   `operation_kind = "op"`, a matching handler may resume, in which case
    ///   execution continues after the operation and `%result` is the result.
    ///   If it does not resume, the selected general handler completes the
    ///   matching handle, and the visible suffix is not evaluated. A source
    ///   `op -> Never` cannot resume, so it neither makes a logical
    ///   resumption nor captures the visible suffix.
    /// - **Local verification:** requires the three attributes, checks the
    ///   `operation_kind` domain, and checks that there is exactly one result,
    ///   no region, and that the operand and result types are resolved rather
    ///   than inference variables. The symbol-aware frontend conformance check
    ///   also confirms that the `fn`/`op` kind, parameter types, and result
    ///   type of the `ability.decl` that `ability` and `op_name` name agree
    ///   with the attributes, operands, and result. No verifier may infer the
    ///   kind from control flow, handlers, result type, or calling convention.
    ///   The `tribute_control.handler` and containing-handle verifiers apply
    ///   the per-kind shape of the handler contract to handler entries.
    /// - **Ownership and value flow:** the operands are ordinary SSA uses.
    ///   This operation makes no source-visible continuation value;
    ///   continuation construction belongs to shared CPS conversion. A `"fn"`
    ///   becomes an `ability.call`/tail dispatch without a continuation, and
    ///   an `"op"` becomes an `ability.perform`/CPS dispatch that takes the
    ///   suffix continuation. For `op -> Never`, conversion instead supplies
    ///   the real zero-capture reject continuation that the existing
    ///   ability/effect ABI requires, whose body is `func.unreachable`. That
    ///   continuation captures no source suffix. A null, an in-band sentinel,
    ///   or an arbitrary `anyref` is not a continuation. The full ABI adapter
    ///   rules are in `new-plans/cps-effects.md`, "Direct-style control
    ///   boundary".
    /// - **Location:** the source location of the ability-operation call,
    ///   including the qualified callee and all arguments where possible.
    #[implements(OperationDeclarationUse, CallableResultOrigin)]
    fn perform(
        ability: Attr<SymbolRef>,
        op_name: Attr<String>,
        operation_kind: Attr<String>,
        args: Variadic<_>,
    ) -> Value<_> {
    }

    /// A handled body. Its result is a callable origin when every value that
    /// becomes it is: the completion region's yield and each general (`op`)
    /// handler's yield. The completion argument is one when the body's yield
    /// is.
    ///
    /// ```text
    /// %answer = tribute_control.handle : AnswerType {
    ///   ...
    ///   tribute_control.yield %body_value
    /// } {
    ///   ^completion(%completed: BodyType):
    ///     ...
    ///     tribute_control.yield %answer_value
    /// } {
    ///   tribute_control.handler {...} { ... }
    ///   ...
    /// }
    /// ```
    ///
    /// - **Operands:** none. Values the executable regions capture use
    ///   ordinary enclosing SSA visibility.
    /// - **Results:** exactly one logical handle result.
    /// - **Attributes:** none required. The optional `evidence_plan`
    ///   (`new-plans/ir.md`, "Evidence 선택 속성") selects the handlers to
    ///   mask in the outer evidence before the body evidence is made. Dynamic
    ///   prompt/owner tags and the physical tail-call representation are
    ///   legalization contracts, not semantic attributes of this operation.
    /// - **Regions:** exactly three, in the fixed order `body`, `completion`,
    ///   `handlers`.
    /// - **Block arguments:** `body` is one block with no arguments.
    ///   `completion` is one block with exactly one argument, whose type is
    ///   the type of the value `body` yields. `handlers` is one block with no
    ///   arguments that holds only `tribute_control.handler` entries.
    /// - **Terminators:** `body` and `completion` end in
    ///   `tribute_control.yield`. The `handlers` block is a declarative table
    ///   with no terminator.
    /// - **Meaning:** when `body` completes normally, `completion` is
    ///   evaluated exactly once and its value is returned. A general handler
    ///   arm that completes without resuming returns the arm value as the
    ///   handle result and skips `completion`. A tail-resumptive `fn` arm
    ///   returns the operation result supplied automatically to the suspended
    ///   computation.
    /// - **Local verification:** enforces the fixed region count, the
    ///   single-block shapes, the block-argument counts, the terminators, and
    ///   yield type equality. It also checks that every direct child of
    ///   `handlers` is a `tribute_control.handler` with a unique
    ///   `(ability, op_name)`. The result type must equal the completion yield
    ///   type and the answer type of every general handler.
    /// - **Ownership and value flow:** this operation owns the delimited
    ///   resumption capability that arises when the body performs a
    ///   resumptive general operation. That capability is exposed only as the
    ///   `resume_token` block argument of a resumptive general handler entry.
    ///   Values leave an executable region only through
    ///   `tribute_control.yield`.
    /// - **Location:** the whole source `handle` expression. Region and block
    ///   locations use the matching body, completion arm, and handler-list
    ///   spans.
    ///
    /// The frontend always materializes the completion region. When the
    /// source has no `do` arm, the region is the identity on the body result.
    /// This removes an optional structural case from conversion without
    /// changing source semantics.
    #[verify]
    #[implements(CallableResultOrigin, CallableArgumentOrigin)]
    fn handle(evidence_plan: Option<Attr<[EvidenceStep]>>) -> Value<_> {
        #[region(body)]
        {}
        #[region(completion)]
        {}
        #[region(handlers)]
        {}
    }

    /// One handler clause. An argument is a callable origin when the clause
    /// claims the kind of the declaration it names and the argument has the
    /// declaration's parameter type at its position.
    ///
    /// ```text
    /// tribute_control.handler {ability = @State, kind = "op", op_name = "get", operation_result_type = ResultType} {
    ///   ^handler(%arg0: Arg0Type, ..., %resume: tribute_control.resume_token<ResultType, AnswerType>):
    ///     ...
    ///     tribute_control.yield %answer
    /// }
    /// ```
    ///
    /// - **Operands and results:** none. It is a declarative entry owned by
    ///   the surrounding `tribute_control.handle`.
    /// - **Attributes:** `ability: SymbolRef`, `op_name: String`,
    ///   `kind: String`, and `operation_result_type: Type` are required.
    ///   `kind` is exactly `fn` or `op`.
    /// - **Regions:** exactly one executable `body` region with one block.
    /// - **Block arguments:** the source operation arguments come first, in
    ///   declaration order and with their logical types. A resumptive `op`
    ///   entry has one last argument,
    ///   `resume_token<operation_result_type, handle-result-type>`. An `op`
    ///   whose `operation_result_type` is source `Never` has no token, and
    ///   neither does a `fn` entry.
    /// - **Terminator:** the body ends in `tribute_control.yield`. For `fn`,
    ///   the yield type equals `operation_result_type` and resumes
    ///   automatically. For `op`, the yield type equals the token's `answer`
    ///   type, and when the arm completes without passing control through
    ///   `resume`, the yield becomes the enclosing handle result. An `op`
    ///   whose `operation_result_type` is source `Never` has no token, and its
    ///   yield type must equal the enclosing `tribute_control.handle` result
    ///   type.
    /// - **Local verification:** checks the required attributes and their
    ///   domains, the single block, the last terminator, the token's position
    ///   and parameters, and the yield rules above. A general handler whose
    ///   `operation_result_type` is `Never` must have no token argument and no
    ///   `tribute_control.resume` anywhere in its body, nested regions
    ///   included. The containing handle's local verifier checks
    ///   `ContinuationFrame` placement, uniqueness, and equality with the
    ///   enclosing handle result, including the `op -> Never` yield; the
    ///   converter checks the same equality and the absence of `resume` before
    ///   rewriting and reports a violation as a conversion failure. The
    ///   symbol-aware frontend conformance check also confirms that the
    ///   referenced declaration's `kind`, argument types, and
    ///   `operation_result_type` are the same. No verifier reclassifies a
    ///   general `op` from the shape of its body.
    /// - **Ownership and value flow:** the last token argument, when present,
    ///   is affine. It may go unused, dropping the continuation, or have one
    ///   static ownership path to a `tribute_control.resume`. When a closure
    ///   captures it, that static path moves into the closure. A copy, store,
    ///   return, yield, or any other escape is invalid. Static SSA validation
    ///   cannot prove that a capturing closure is called at most once
    ///   dynamically, so the lowered resumption must also enforce one-shot
    ///   consumption at run time and reject or trap on a second call. A `fn`
    ///   arm or an `op -> Never` arm has no continuation capability.
    /// - **Location:** the source handler arm, including the operation
    ///   header.
    #[implements(OperationDeclarationUse, CallableArgumentOrigin)]
    fn handler(
        ability: Attr<SymbolRef>,
        op_name: Attr<String>,
        kind: Attr<String>,
        operation_result_type: Attr<Type>,
    ) {
        #[region(body)]
        {}
    }

    /// A resumption. It returns the result of the handle its token belongs
    /// to, so its result is a callable origin when that result is.
    ///
    /// ```text
    /// %answer = tribute_control.resume %resume, %value : AnswerType
    /// ```
    ///
    /// - **Operands:** exactly two. `%resume` is a
    ///   `resume_token<InputType, AnswerType>` and `%value` is an
    ///   `InputType`.
    /// - **Results:** exactly one `AnswerType`.
    /// - **Attributes and regions:** only an optional `evidence_plan`
    ///   (`new-plans/ir.md`, "Evidence 선택 속성"), and no region.
    /// - **Terminator:** no. Strict work after `resume` stays explicitly in
    ///   the enclosing region and runs only after the resumed computation
    ///   returns.
    /// - **Meaning:** consumes the one-shot resumption of the lexically
    ///   nearest enclosing general handler, supplies `%value` to the
    ///   suspended `perform`, and returns the logical result the resumed
    ///   computation reaches at the handle boundary. It is invalid in a `fn`
    ///   or `op -> Never` arm.
    /// - **Local verification:** enforces the operand and result arity and
    ///   the three type equalities that
    ///   `resume_token<InputType, AnswerType>` requires.
    /// - **Ownership and value flow:** consumes the token. The token may
    ///   reach this operation through an explicit closure capture, but it
    ///   must keep a single static use-def path from the handler block
    ///   argument. Affine-use validation follows captures and nested regions,
    ///   so it is a whole-IR check. When a capture allows repeated dynamic
    ///   invocation, the converted continuation's run-time one-shot state is
    ///   the final enforcement boundary.
    /// - **Location:** the source `resume` expression.
    #[verify]
    #[implements(CallableResultOrigin)]
    fn resume<T: ResumeToken>(
        evidence_plan: Option<Attr<[EvidenceStep]>>,
        resume_token: Value<T>,
        value: Value<T::Input>,
    ) -> Value<T::Answer> {
    }

    /// End an executable `tribute_control` region with a logical value.
    ///
    /// ```text
    /// tribute_control.yield %value
    /// ```
    ///
    /// - **Operands:** exactly one logical value.
    /// - **Results, attributes, and regions:** none.
    /// - **Terminator:** of a `handle` body, completion region, or handler
    ///   body. It is invalid anywhere else.
    /// - **Local verification:** enforces its own shape. The owning operation
    ///   checks the placement and the yield type.
    /// - **Ownership and value flow:** passes an ordinary logical value to
    ///   the owning structured operation. A `resume_token` can never be
    ///   yielded.
    /// - **Location:** the source expression that makes the region result. A
    ///   synthesized identity completion uses the owning `handle` location.
    fn r#yield(value: Value<_>) {}
}

/// Attribute that selects the evidence a call, `resume`, or handle body
/// receives from the evidence of the code around it.
pub const EVIDENCE_PLAN_ATTR: &str = "evidence_plan";

/// One element of an `evidence_plan`. `{mask = instance}`, `{dup = instance}`,
/// `{outer = instance}`, and `{push = instance}` name an exact
/// `ability.decl` symbol; `{select = index}` names a row tail of the
/// surrounding callable, and `{tails = [plan, ...]}` holds one plan for each
/// row tail of the callee.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum EvidenceStep {
    /// Hide the top handler of the instance, uncovering the one beneath.
    Mask(SymbolPath),
    /// Stack one more copy of the top handler of the instance.
    Dup(SymbolPath),
    /// Take the evidence the top handler of the instance was installed on.
    /// CPS legalization adds this step; source operations never carry it.
    Outer(SymbolPath),
    /// Take the evidence of the surrounding callable's row tail `index`.
    Select(u32),
    /// Put the surrounding evidence's top handler of the instance on top of
    /// the selected evidence.
    Push(SymbolPath),
    /// Give each row tail of the callee the surrounding evidence after its
    /// plan.
    Tails(Vec<Vec<EvidenceStep>>),
}

impl EvidenceStep {
    /// The ability instance this step changes.
    pub fn instance(&self) -> Option<&SymbolPath> {
        match self {
            Self::Mask(instance)
            | Self::Dup(instance)
            | Self::Outer(instance)
            | Self::Push(instance) => Some(instance),
            Self::Select(_) | Self::Tails(_) => None,
        }
    }

    /// Decode one plan element, or say why it is malformed.
    pub fn from_attribute(ctx: &IrContext, attr: &Attribute) -> Result<Self, String> {
        let Attribute::Dict(entries) = attr else {
            return Err("evidence_plan element must be a dictionary".into());
        };
        let Ok((key, value)) = entries.iter().exactly_one() else {
            return Err(format!(
                "evidence_plan element must have exactly one entry, found {}",
                entries.len()
            ));
        };
        let instance = || match value {
            Attribute::SymbolRef(instance) => Ok(instance.clone()),
            _ => Err(format!(
                "evidence_plan {} must name an ability instance",
                ctx.str(key)
            )),
        };
        match ctx.str(key) {
            "mask" => instance().map(Self::Mask),
            "dup" => instance().map(Self::Dup),
            "outer" => instance().map(Self::Outer),
            "push" => instance().map(Self::Push),
            "select" => match value {
                Attribute::Int(index) => u32::try_from(*index)
                    .map(Self::Select)
                    .map_err(|_| "evidence_plan select must be a row tail index".into()),
                _ => Err("evidence_plan select must be a row tail index".into()),
            },
            "tails" => {
                let Attribute::List(plans) = value else {
                    return Err("evidence_plan tails must be a list of plans".into());
                };
                plans
                    .iter()
                    .map(|plan| {
                        let Attribute::List(steps) = plan else {
                            return Err("evidence_plan tails must be a list of plans".into());
                        };
                        steps
                            .iter()
                            .map(|step| Self::from_attribute(ctx, step))
                            .collect()
                    })
                    .collect::<Result<_, _>>()
                    .map(Self::Tails)
            }
            other => Err(format!(
                "evidence_plan element must be mask, dup, outer, select, push, or tails, \
                 found {other}"
            )),
        }
    }

    pub fn to_attribute(&self, ctx: &mut IrContext) -> Attribute {
        let (keyword, value) = match self {
            Self::Mask(instance) => ("mask", Attribute::SymbolRef(instance.clone())),
            Self::Dup(instance) => ("dup", Attribute::SymbolRef(instance.clone())),
            Self::Outer(instance) => ("outer", Attribute::SymbolRef(instance.clone())),
            Self::Push(instance) => ("push", Attribute::SymbolRef(instance.clone())),
            Self::Select(index) => ("select", Attribute::Int(i128::from(*index))),
            Self::Tails(plans) => (
                "tails",
                Attribute::List(
                    plans
                        .iter()
                        .map(|plan| {
                            Attribute::List(
                                plan.iter().map(|step| step.to_attribute(ctx)).collect(),
                            )
                        })
                        .collect(),
                ),
            ),
        };
        let mut entries = AttributeMap::new();
        entries.view_mut(ctx).insert(keyword, value);
        Attribute::Dict(entries)
    }

    /// The `evidence_plan` value for these steps, or `None` for an empty
    /// plan, which is written by omitting the attribute.
    pub fn plan_attribute(
        ctx: &mut IrContext,
        steps: impl IntoIterator<Item = Self>,
    ) -> Option<Attribute> {
        let steps: Vec<_> = steps
            .into_iter()
            .map(|step| step.to_attribute(ctx))
            .collect();
        (!steps.is_empty()).then_some(Attribute::List(steps))
    }
}

impl trunk_ir::attr_kind::AttrKind for EvidenceStep {
    const KIND: trunk_ir::op_schema::AttributeKind =
        trunk_ir::op_schema::AttributeKind::Dict(&trunk_ir::op_schema::AttributeKind::Any);
    type Out<'ctx> = EvidenceStep;
    type In = EvidenceStep;

    fn read<'ctx>(ctx: &'ctx IrContext, attr: &'ctx Attribute) -> EvidenceStep {
        Self::from_attribute(ctx, attr)
            .unwrap_or_else(|error| panic!("unverified evidence_plan element: {error}"))
    }

    fn write(ctx: &mut IrContext, value: EvidenceStep) -> Attribute {
        value.to_attribute(ctx)
    }
}

/// The operation that carries an `evidence_plan`, which decides the steps it
/// may hold.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum EvidencePlanSite {
    /// A source call or resume: no `outer` step.
    Source,
    /// A handle: `mask` steps only.
    Handle,
    /// An operation CPS legalization built from a source operation, which may
    /// also select the evidence a handle was installed on.
    Legalized,
}

/// Check the shape of an operation's `evidence_plan`. Whether the selection
/// matches the ability rows is typechecking's responsibility.
pub(crate) fn verify_evidence_plan(
    ctx: &IrContext,
    op: OpRef,
    site: EvidencePlanSite,
) -> Result<(), String> {
    let Some(plan) = ctx.op_attrs(op).get(EVIDENCE_PLAN_ATTR) else {
        return Ok(());
    };
    let Attribute::List(items) = plan else {
        return Err("evidence_plan must be a list".into());
    };
    if items.is_empty() {
        return Err("evidence_plan must not be empty; omit it to keep the evidence".into());
    }
    let steps: Vec<_> = items
        .iter()
        .map(|item| EvidenceStep::from_attribute(ctx, item))
        .collect::<Result<_, _>>()?;
    if site == EvidencePlanSite::Handle
        && steps
            .iter()
            .any(|step| !matches!(step, EvidenceStep::Mask(_)))
    {
        return Err("a handle's evidence_plan may only mask".into());
    }
    verify_evidence_steps(&steps, true, site == EvidencePlanSite::Legalized)
}

/// Check one plan: an optional leading `select`, then the steps on ability
/// instances, each instance changed in one way, then an optional `tails`
/// whose plans follow the same rule without `tails` of their own.
fn verify_evidence_steps(
    steps: &[EvidenceStep],
    tails_allowed: bool,
    outer_allowed: bool,
) -> Result<(), String> {
    let mut seen = HashSet::default();
    for (index, step) in steps.iter().enumerate() {
        match step {
            EvidenceStep::Outer(_) if !outer_allowed => {
                return Err("a source operation's evidence_plan may not use outer".into());
            }
            EvidenceStep::Select(_) if index != 0 => {
                return Err("evidence_plan select must be the first step".into());
            }
            EvidenceStep::Select(_) => {}
            EvidenceStep::Push(_) if !matches!(steps[0], EvidenceStep::Select(_)) => {
                return Err("evidence_plan push requires a selected row tail".into());
            }
            EvidenceStep::Tails(_) if !tails_allowed => {
                return Err("an evidence_plan of a row tail may not attach tails".into());
            }
            EvidenceStep::Tails(_) if index + 1 != steps.len() => {
                return Err("evidence_plan tails must be the last step".into());
            }
            EvidenceStep::Tails(plans) => {
                for plan in plans {
                    verify_evidence_steps(plan, false, outer_allowed)?;
                }
            }
            EvidenceStep::Mask(instance)
            | EvidenceStep::Dup(instance)
            | EvidenceStep::Outer(instance) => {
                if !seen.insert((instance, false)) {
                    return Err(format!(
                        "evidence_plan names ability instance {instance} more than once"
                    ));
                }
            }
            EvidenceStep::Push(instance) => {
                if !seen.insert((instance, true)) {
                    return Err(format!(
                        "evidence_plan names ability instance {instance} more than once"
                    ));
                }
            }
        }
    }
    Ok(())
}

impl trunk_ir::ops::Verify for FuncRef {
    fn verify(&self, ctx: &IrContext) -> Result<(), trunk_ir::ops::VerificationError> {
        Ok(row_args(ctx, self.op_ref()).map(drop)?)
    }
}

impl trunk_ir::ops::Verify for Instantiate {
    fn verify(&self, ctx: &IrContext) -> Result<(), trunk_ir::ops::VerificationError> {
        Ok(row_args(ctx, self.op_ref()).map(drop)?)
    }
}

impl trunk_ir::ops::Verify for Call {
    fn verify(&self, ctx: &IrContext) -> Result<(), trunk_ir::ops::VerificationError> {
        row_args(ctx, self.op_ref())?;
        Ok(verify_evidence_plan(
            ctx,
            self.op_ref(),
            EvidencePlanSite::Source,
        )?)
    }
}

impl trunk_ir::ops::Verify for CallIndirect {
    fn verify(&self, ctx: &IrContext) -> Result<(), trunk_ir::ops::VerificationError> {
        Ok(verify_evidence_plan(
            ctx,
            self.op_ref(),
            EvidencePlanSite::Source,
        )?)
    }
}

impl trunk_ir::ops::Verify for TailCall {
    fn verify(&self, ctx: &IrContext) -> Result<(), trunk_ir::ops::VerificationError> {
        row_args(ctx, self.op_ref())?;
        Ok(verify_evidence_plan(
            ctx,
            self.op_ref(),
            EvidencePlanSite::Source,
        )?)
    }
}

impl trunk_ir::ops::Verify for TailCallIndirect {
    fn verify(&self, ctx: &IrContext) -> Result<(), trunk_ir::ops::VerificationError> {
        Ok(verify_evidence_plan(
            ctx,
            self.op_ref(),
            EvidencePlanSite::Source,
        )?)
    }
}

impl trunk_ir::ops::Verify for Resume {
    fn verify(&self, ctx: &IrContext) -> Result<(), trunk_ir::ops::VerificationError> {
        Ok(verify_evidence_plan(
            ctx,
            self.op_ref(),
            EvidencePlanSite::Source,
        )?)
    }
}

impl trunk_ir::ops::Verify for Handle {
    fn verify(&self, ctx: &IrContext) -> Result<(), trunk_ir::ops::VerificationError> {
        Ok(verify_evidence_plan(
            ctx,
            self.op_ref(),
            EvidencePlanSite::Handle,
        )?)
    }
}

/// Why a name-matching source signature does not satisfy its storage contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq, derive_more::Display, derive_more::Error)]
pub enum FuncSigTypeError {
    #[display("missing required `{_0}` u32 attribute")]
    MissingCount(#[error(not(source))] &'static str),
    #[display("`{_0}` must be a u32 attribute")]
    InvalidCount(#[error(not(source))] &'static str),
    #[display("input and result counts overflow u32")]
    CountOverflow,
    #[display("input and result counts must equal the parameter count")]
    LengthMismatch,
    #[display("a source signature has exactly one result, found {_0}")]
    ResultCount(#[error(not(source))] u32),
    #[display("malformed calling convention")]
    Convention,
    /// The ability row is malformed, or written out although closed and empty.
    #[display("malformed ability row")]
    Row,
}

/// Typed wrapper for an input-first, exactly-one-result source signature.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FuncSig(TypeRef);

static FUNC_SIG_DEF: trunk_ir::TypeDef =
    trunk_ir::TypeDef::new("tribute_control.func_sig").with_verifier::<FuncSig>();

trunk_ir::register_type_def!(&FUNC_SIG_DEF);

impl DialectType for FuncSig {
    const DEF: &'static trunk_ir::TypeDef = &FUNC_SIG_DEF;

    fn from_type_ref(ctx: &IrContext, ty: TypeRef) -> Option<Self> {
        Self::matches(ctx, ty)
            .then(|| <Self as trunk_ir::type_verifier::VerifyType>::verify(ctx, ty).ok())
            .flatten()
    }

    fn as_type_ref(&self) -> TypeRef {
        self.0
    }
}

impl trunk_ir::type_constraint::TypeConstraint for FuncSig {
    const DESC: &'static trunk_ir::type_constraint::ConstraintDesc =
        &trunk_ir::type_constraint::ConstraintDesc {
            name: "tribute_control.func_sig",
            exact: true,
            projections: &[
                trunk_ir::type_constraint::ProjectionDesc {
                    name: "Inputs",
                    kind: trunk_ir::type_constraint::ProjectionKind::List,
                },
                trunk_ir::type_constraint::ProjectionDesc {
                    name: "Result",
                    kind: trunk_ir::type_constraint::ProjectionKind::One,
                },
            ],
            matches: |ctx, ty| FuncSig::from_type_ref(ctx, ty).is_some(),
            project: |ctx, ty, index| {
                let signature = FuncSig::from_type_ref(ctx, ty)?;
                match index {
                    0 => Some(trunk_ir::type_constraint::Projected::List(
                        signature.inputs(ctx),
                    )),
                    1 => Some(trunk_ir::type_constraint::Projected::One(
                        signature.result(ctx),
                    )),
                    _ => None,
                }
            },
            fixed: None,
        };
}

impl From<FuncSig> for TypeRef {
    fn from(value: FuncSig) -> Self {
        value.0
    }
}

impl trunk_ir::type_verifier::VerifyType for FuncSig {
    fn verify(ctx: &IrContext, ty: TypeRef) -> Result<Self, trunk_ir::ops::VerificationError> {
        let data = ctx.get_type(ty);
        let inputs = read_count(ctx, &data.attrs, trunk_ir::dialect::func::NUM_INPUTS_ATTR)?;
        let results = read_count(ctx, &data.attrs, trunk_ir::dialect::func::NUM_RESULTS_ATTR)?;
        if results != 1 {
            return Err(FuncSigTypeError::ResultCount(results).into());
        }
        let total = inputs
            .checked_add(results)
            .ok_or(FuncSigTypeError::CountOverflow)?;
        if usize::try_from(total).ok() != Some(data.params.len()) {
            return Err(FuncSigTypeError::LengthMismatch.into());
        }
        // Absent before class specialization assigns it.
        if let Some(convention) = data.attrs.view(ctx).get(CALLING_CONVENTION_ATTR) {
            let Attribute::Int(code) = convention else {
                return Err(FuncSigTypeError::Convention.into());
            };
            CallingConvention::try_from(*code).map_err(|_| FuncSigTypeError::Convention)?;
        }
        if let Some(row) = data.attrs.view(ctx).get(ABILITY_ROW_ATTR)
            && Row::from_attribute(ctx, row).is_ok_and(|row| row.is_empty())
        {
            return Err(FuncSigTypeError::Row.into());
        }
        row::stored_row(ctx, data)
            .expect("a func_sig stores a row")
            .map_err(|_| FuncSigTypeError::Row)?;
        Ok(Self(ty))
    }
}

impl FuncSig {
    pub fn as_type_ref(self) -> TypeRef {
        self.0
    }

    pub fn result(self, ctx: &IrContext) -> TypeRef {
        self.results(ctx)[0]
    }

    pub fn inputs(self, ctx: &IrContext) -> &[TypeRef] {
        let count = read_count(
            ctx,
            &ctx.get_type(self.0).attrs,
            trunk_ir::dialect::func::NUM_INPUTS_ATTR,
        )
        .expect("validated source func_sig retains input count") as usize;
        &ctx.get_type(self.0).params[..count]
    }

    pub fn results(self, ctx: &IrContext) -> &[TypeRef] {
        let inputs = self.inputs(ctx).len();
        &ctx.get_type(self.0).params[inputs..]
    }

    /// Each input paired with its own attributes.
    pub fn inputs_with_attrs(
        self,
        ctx: &IrContext,
    ) -> impl Iterator<Item = (TypeRef, &AttributeMap)> {
        let count = self.inputs(ctx).len();
        ctx.get_type(self.0).params_with_attrs(ctx).take(count)
    }

    /// The result paired with its own attributes.
    pub fn result_with_attrs(self, ctx: &IrContext) -> (TypeRef, &AttributeMap) {
        let index = self.inputs(ctx).len();
        let data = ctx.get_type(self.0);
        (data.params[index], data.param_attrs(ctx, index))
    }

    /// The convention class specialization assigned.
    ///
    /// # Panics
    ///
    /// If the signature has no convention yet.
    pub fn convention(self, ctx: &IrContext) -> CallingConvention {
        self.assigned_convention(ctx)
            .expect("source func_sig has a convention after class specialization")
    }

    /// The convention, or `None` before class specialization assigns it.
    pub fn assigned_convention(self, ctx: &IrContext) -> Option<CallingConvention> {
        let code = ctx.type_attrs(self.0).get_i128(CALLING_CONVENTION_ATTR)?;
        Some(
            CallingConvention::try_from(code)
                .expect("validated source func_sig retains convention domain"),
        )
    }

    /// The callable's ability row; closed and empty when the type omits it.
    pub fn row(self, ctx: &IrContext) -> Row {
        row::stored_row(ctx, ctx.get_type(self.0))
            .expect("a func_sig stores a row")
            .expect("validated source func_sig retains a well-formed row")
    }

    /// Remove the storage delimiters and the source-only convention and row.
    ///
    /// Consumers rebuilding an equivalent signature in another dialect retain
    /// every other attribute through their owning typed constructor.
    pub fn remove_reserved_attrs(ctx: &mut IrContext, attrs: &mut AttributeMap) {
        let mut attrs = attrs.view_mut(ctx);
        for key in RESERVED_ATTRS {
            attrs.remove(key);
        }
    }

    /// Whether this signature carries type-owned metadata beyond storage and
    /// source convention and row fields.
    pub(crate) fn has_nonreserved_attrs(self, ctx: &IrContext) -> bool {
        ctx.type_attrs(self.0)
            .iter()
            .any(|(key, _)| !RESERVED_ATTRS.contains(&key))
    }
}

/// The storage delimiters and source-only convention and row metadata.
const RESERVED_ATTRS: [&str; 4] = [
    trunk_ir::dialect::func::NUM_INPUTS_ATTR,
    trunk_ir::dialect::func::NUM_RESULTS_ATTR,
    CALLING_CONVENTION_ATTR,
    ABILITY_ROW_ATTR,
];

fn read_count(
    ctx: &IrContext,
    attrs: &AttributeMap,
    key: &'static str,
) -> Result<u32, FuncSigTypeError> {
    match attrs.view(ctx).get(key) {
        Some(Attribute::Int(value)) => {
            u32::try_from(*value).map_err(|_| FuncSigTypeError::InvalidCount(key))
        }
        Some(_) => Err(FuncSigTypeError::InvalidCount(key)),
        None => Err(FuncSigTypeError::MissingCount(key)),
    }
}

impl FuncSig {
    /// Start a source signature with `inputs` and `result`: no row, no
    /// convention, and no other attributes until the builder sets them.
    pub fn params(inputs: impl IntoIterator<Item = TypeRef>, result: TypeRef) -> FuncSigBuilder {
        FuncSigBuilder {
            inputs: inputs.into_iter().collect(),
            result,
            row: Row::default(),
            convention: None,
            attrs: AttributeMap::new(),
        }
    }

    /// This signature with its own convention set to `convention`. Callable
    /// types inside it keep theirs.
    pub fn with_convention(self, ctx: &mut IrContext, convention: CallingConvention) -> Self {
        let mut data = ctx.get_type(self.0).clone();
        data.attrs
            .view_mut(ctx)
            .insert(CALLING_CONVENTION_ATTR, Attribute::Int(convention as i128));
        let ty = ctx.intern_type(data);
        Self::from_type_ref(ctx, ty).expect("setting a convention keeps a source signature")
    }
}

/// Builds a [`FuncSig`]; start one with [`FuncSig::params`].
#[must_use]
pub struct FuncSigBuilder {
    inputs: Vec<TypeRef>,
    result: TypeRef,
    row: Row,
    convention: Option<CallingConvention>,
    attrs: AttributeMap,
}

impl FuncSigBuilder {
    /// The ability row; an empty row is the closed pure row and is not stored.
    pub fn row(mut self, row: Row) -> Self {
        self.row = row;
        self
    }

    /// The convention, which class specialization assigns.
    pub fn convention(mut self, convention: CallingConvention) -> Self {
        self.convention = Some(convention);
        self
    }

    /// Metadata other than the reserved delimiters, row, and convention.
    pub fn attrs(mut self, attrs: AttributeMap) -> Self {
        self.attrs = attrs;
        self
    }

    pub fn intern(self, ctx: &mut IrContext) -> FuncSig {
        assert!(
            self.attrs
                .iter()
                .all(|(key, _)| !RESERVED_ATTRS.contains(&ctx.str(key))),
            "source func_sig reserves delimiters, convention, and row metadata"
        );
        let count =
            u32::try_from(self.inputs.len()).expect("source func_sig input count exceeds u32");
        let mut builder = TypeDataBuilder::new("tribute_control.func_sig")
            .params(self.inputs)
            .param(self.result);
        for (key, value) in self.attrs {
            builder = builder.attr(key, value);
        }
        builder = builder
            .attr(
                trunk_ir::dialect::func::NUM_INPUTS_ATTR,
                Attribute::from(count),
            )
            .attr(
                trunk_ir::dialect::func::NUM_RESULTS_ATTR,
                Attribute::from(1_u32),
            );
        if !self.row.is_empty() {
            let row = self.row.to_attribute(ctx);
            builder = builder.attr(ABILITY_ROW_ATTR, row);
        }
        if let Some(convention) = self.convention {
            builder = builder.attr(CALLING_CONVENTION_ATTR, Attribute::Int(convention as i128));
        }
        let ty = builder.intern(ctx);
        FuncSig::from_type_ref(ctx, ty)
            .expect("source func_sig constructor must produce a valid type")
    }
}

/// Read convention only through a validated source signature; `None` for
/// another type and before class specialization assigns one.
pub fn func_sig_convention(ctx: &IrContext, ty: TypeRef) -> Option<CallingConvention> {
    FuncSig::from_type_ref(ctx, ty).and_then(|signature| signature.assigned_convention(ctx))
}

/// Read the exact input and answer types carried by a resume token.
pub fn resume_token_parts(ctx: &IrContext, ty: TypeRef) -> Option<(TypeRef, TypeRef)> {
    if !ResumeToken::matches(ctx, ty) {
        return None;
    }
    let [input, answer] = ctx.get_type(ty).params.as_slice() else {
        return None;
    };
    Some((*input, *answer))
}

/// Build a body-less `tribute_control.func` declaration.
pub fn func_declaration(
    ctx: &mut IrContext,
    location: Location,
    sym_name: &Symbol,
    func_sig_type: TypeRef,
) -> Func {
    let data = trunk_ir::OperationDataBuilder::new(location, "tribute_control.func")
        .attr(
            "sym_name",
            Attribute::String(ctx.intern_symbol_text(sym_name)),
        )
        .attr("type", Attribute::Type(func_sig_type))
        .build(ctx);
    let op = ctx.create_op(data);
    Func::from_op(ctx, op).expect("newly built tribute_control.func")
}

impl Func {
    /// Return the optional semantic identity attached by the trusted frontend registry.
    pub fn compiler_intrinsic_identity<'a>(&self, ctx: &'a IrContext) -> Option<&'a str> {
        ctx.op_attrs(self.op_ref()).get_str(COMPILER_INTRINSIC_ATTR)
    }
}

impl CallLike for Call {
    fn direct_callee<'a>(&self, ctx: &'a IrContext) -> &'a SymbolPath {
        self.callee(ctx)
    }
}
impl CallLike for TailCall {
    fn direct_callee<'a>(&self, ctx: &'a IrContext) -> &'a SymbolPath {
        self.callee(ctx)
    }
}

/// The ability operation declaration an operation names, and the kind it
/// claims through the attribute `kind_attr`.
pub struct DeclarationUse<'a> {
    pub ability: &'a SymbolPath,
    pub op_name: StringRef,
    pub kind: Option<StringRef>,
    pub kind_attr: &'static str,
}

/// An operation that names one ability operation declaration, whose
/// parameter and result types fix the types of values the operation supplies.
///
/// Whole-IR validation queries it on operations whose schema may not hold,
/// so implementations read attributes and regions fallibly.
#[trunk_ir::op_interface]
pub trait OperationDeclarationUse: DialectOp {
    /// The declaration this operation names; `None` without its ability or
    /// operation name.
    fn declaration_use<'a>(&self, ctx: &'a IrContext) -> Option<DeclarationUse<'a>>;

    /// The values that take the declaration's parameter types, in order.
    fn declared_arguments<'a>(&self, ctx: &'a IrContext) -> Option<&'a [ValueRef]>;

    /// The type that the declaration's result type must equal.
    fn declared_result_type(&self, ctx: &IrContext) -> Option<TypeRef>;
}

impl OperationDeclarationUse for Perform {
    fn declaration_use<'a>(&self, ctx: &'a IrContext) -> Option<DeclarationUse<'a>> {
        let attrs = ctx.op_attrs(self.op_ref());
        Some(DeclarationUse {
            ability: attrs.get_symbol_ref("ability")?,
            op_name: attrs.get_string_ref("op_name")?,
            kind: attrs.get_string_ref("operation_kind"),
            kind_attr: "operation_kind",
        })
    }

    fn declared_arguments<'a>(&self, ctx: &'a IrContext) -> Option<&'a [ValueRef]> {
        Some(ctx.op_operands(self.op_ref()))
    }

    fn declared_result_type(&self, ctx: &IrContext) -> Option<TypeRef> {
        ctx.op_result_types(self.op_ref()).first().copied()
    }
}

impl OperationDeclarationUse for Handler {
    fn declaration_use<'a>(&self, ctx: &'a IrContext) -> Option<DeclarationUse<'a>> {
        let attrs = ctx.op_attrs(self.op_ref());
        Some(DeclarationUse {
            ability: attrs.get_symbol_ref("ability")?,
            op_name: attrs.get_string_ref("op_name")?,
            kind: attrs.get_string_ref("kind"),
            kind_attr: "kind",
        })
    }

    /// The clause's block arguments without its trailing resume token.
    fn declared_arguments<'a>(&self, ctx: &'a IrContext) -> Option<&'a [ValueRef]> {
        let region = ctx.op_region(self.op_ref(), 0)?;
        let &block = ctx.region(region).blocks.first()?;
        let args = ctx.block_args(block);
        Some(match args.split_last() {
            Some((token, rest)) if ResumeToken::matches(ctx, ctx.value_ty(*token)) => rest,
            _ => args,
        })
    }

    fn declared_result_type(&self, ctx: &IrContext) -> Option<TypeRef> {
        ctx.op_attrs(self.op_ref())
            .get_type("operation_result_type")
    }
}

// A source `become` ends its callable: the callee's result is the caller's.
impl CallableExit for TailCall {}
impl CallableExit for TailCallIndirect {}

// === Custom assembly: tribute_control.func ===

fn print_row(h: &mut trunk_ir::printer::OpPrintHelper<'_, '_>, row: &Row) -> fmt::Result {
    use fmt::Write;
    if row.is_empty() {
        return Ok(());
    }
    write!(h, " row(")?;
    for (index, instance) in row.instances.iter().enumerate() {
        if index > 0 {
            write!(h, ", ")?;
        }
        h.write_attribute(&Attribute::SymbolRef(instance.clone()))?;
    }
    if let Some(tail) = row.tail {
        if !row.instances.is_empty() {
            write!(h, " ")?;
        }
        write!(h, "| 'a{tail}")?;
    }
    write!(h, ")")
}

/// A definition's row binders as its header declares them, or `None` when
/// the stored attributes cannot be written that way.
struct BinderHeader {
    binders: Vec<RowBinder>,
    unions: Vec<RowUnion>,
    inferred: bool,
}

impl BinderHeader {
    fn of(ctx: &IrContext, op: OpRef) -> Option<Self> {
        let binders = row_binders(ctx, op).ok()?;
        let unions = row_unions(ctx, op).ok()?;
        let in_range = |index: u32| usize::try_from(index).is_ok_and(|index| index < binders.len());
        let mut results = HashSet::default();
        let well_formed = unions.iter().all(|union| {
            in_range(union.result)
                && union.sources.iter().copied().all(in_range)
                && results.insert(union.result)
        });
        let inferred = match ctx.op_attrs(op).get(INFERRED_ROW_ATTR) {
            None => false,
            Some(Attribute::Bool(true)) => true,
            Some(_) => return None,
        };
        // An empty list written out has no header form.
        let attrs = ctx.op_attrs(op);
        let written_empty = (attrs.contains_key(ROW_BINDERS_ATTR) && binders.is_empty())
            || (attrs.contains_key(ROW_UNIONS_ATTR) && unions.is_empty());
        (well_formed && !written_empty).then_some(Self {
            binders,
            unions,
            inferred,
        })
    }

    fn print(&self, h: &mut trunk_ir::printer::OpPrintHelper<'_, '_>) -> fmt::Result {
        use fmt::Write;
        if self.binders.is_empty() {
            return Ok(());
        }
        write!(h, "[")?;
        for (index, binder) in self.binders.iter().enumerate() {
            if index > 0 {
                write!(h, ", ")?;
            }
            if *binder != RowBinder::Param {
                write!(h, "{} ", binder.keyword())?;
            }
            write!(h, "'a{index}")?;
            let union = self
                .unions
                .iter()
                .find(|union| usize::try_from(union.result).ok() == Some(index));
            if let Some(union) = union {
                write!(h, " = ")?;
                for (position, source) in union.sources.iter().enumerate() {
                    if position > 0 {
                        write!(h, " | ")?;
                    }
                    write!(h, "'a{source}")?;
                }
            }
        }
        write!(h, "]")
    }
}

/// `'a<index>`, a row binder named by its position.
fn parse_binder_name(input: &mut &str) -> winnow::ModalResult<u32> {
    use winnow::ascii::digit1;
    use winnow::prelude::*;

    "'a".parse_next(input)?;
    digit1.try_map(str::parse::<u32>).parse_next(input)
}

/// A definition header's `[...]` binder list, as the raw `row_binders` and
/// `row_unions` attributes; nothing when the header has none.
fn parse_binders<'a>(
    input: &mut &'a str,
) -> winnow::ModalResult<
    Vec<(
        std::borrow::Cow<'a, str>,
        trunk_ir::parser::raw::RawAttribute<'a>,
    )>,
> {
    use trunk_ir::parser::raw::{RawAttribute, ident, ws};
    use winnow::combinator::{opt, separated};
    use winnow::prelude::*;

    let reject = || winnow::error::ErrMode::Backtrack(winnow::error::ContextError::new());
    let checkpoint = *input;
    ws.parse_next(input)?;
    if opt('[').parse_next(input)?.is_none() {
        *input = checkpoint;
        return Ok(Vec::new());
    }
    let mut kinds = Vec::new();
    let mut unions = Vec::new();
    loop {
        ws.parse_next(input)?;
        let kind = match opt(ident).parse_next(input)?.map(RowBinder::from_keyword) {
            Some(Some(kind)) if kind != RowBinder::Param => {
                ws.parse_next(input)?;
                kind
            }
            None => RowBinder::Param,
            Some(_) => return Err(reject()),
        };
        let index = parse_binder_name(input)?;
        if usize::try_from(index).ok() != Some(kinds.len()) {
            return Err(reject());
        }
        kinds.push(RawAttribute::String(kind.keyword().to_owned()));
        ws.parse_next(input)?;
        if opt('=').parse_next(input)?.is_some() {
            let sources: Vec<u32> = separated(
                1..,
                (ws, parse_binder_name, ws).map(|(_, source, _)| source),
                '|',
            )
            .parse_next(input)?;
            unions.push(RawAttribute::Dict(vec![
                ("result".into(), RawAttribute::Int(i128::from(index))),
                (
                    "sources".into(),
                    RawAttribute::List(
                        sources
                            .into_iter()
                            .map(|source| RawAttribute::Int(i128::from(source)))
                            .collect(),
                    ),
                ),
            ]));
        }
        ws.parse_next(input)?;
        if opt(',').parse_next(input)?.is_none() {
            break;
        }
    }
    ']'.parse_next(input)?;
    let mut attributes = vec![(ROW_BINDERS_ATTR.into(), RawAttribute::List(kinds))];
    if !unions.is_empty() {
        attributes.push((ROW_UNIONS_ATTR.into(), RawAttribute::List(unions)));
    }
    Ok(attributes)
}

/// The `inferred` keyword before a definition's convention.
fn parse_inferred(input: &mut &str) -> winnow::ModalResult<bool> {
    use trunk_ir::parser::raw::ws;
    use winnow::combinator::opt;
    use winnow::prelude::*;

    let checkpoint = *input;
    ws.parse_next(input)?;
    if opt((
        "inferred",
        winnow::combinator::peek(winnow::ascii::multispace1),
    ))
    .parse_next(input)?
    .is_some()
    {
        return Ok(true);
    }
    *input = checkpoint;
    Ok(false)
}

fn func_sig_parts(
    ctx: &IrContext,
    ty: TypeRef,
) -> Option<(TypeRef, Vec<TypeRef>, Option<CallingConvention>)> {
    let signature = FuncSig::from_type_ref(ctx, ty)?;
    Some((
        signature.result(ctx),
        signature.inputs(ctx).to_vec(),
        signature.assigned_convention(ctx),
    ))
}

/// Whether custom source assembly can represent the complete signature.
///
/// Invalid and metadata-bearing signatures must use generic assembly so
/// printing never discards their storage.
fn has_concise_func_sig(ctx: &IrContext, ty: Option<TypeRef>) -> bool {
    ty.and_then(|ty| FuncSig::from_type_ref(ctx, ty))
        .is_some_and(|signature| !signature.has_nonreserved_attrs(ctx))
}

fn print_extra_attributes(
    h: &mut trunk_ir::printer::OpPrintHelper<'_, '_>,
    op: OpRef,
    excluded: &[&str],
) -> fmt::Result {
    use fmt::Write;
    let ctx = h.ctx();
    let attrs: Vec<_> = ctx
        .op_attrs(op)
        .iter()
        .filter(|(key, _)| !excluded.contains(key))
        .collect();
    if attrs.is_empty() {
        return Ok(());
    }
    write!(h, " attributes {{")?;
    for (index, (key, value)) in attrs.iter().enumerate() {
        if index > 0 {
            write!(h, ", ")?;
        }
        write!(h, "{key} = ")?;
        h.write_attribute(value)?;
    }
    write!(h, "}}")
}

fn print_signature_params(
    h: &mut trunk_ir::printer::OpPrintHelper<'_, '_>,
    region: Option<RegionRef>,
    param_types: &[TypeRef],
) -> fmt::Result {
    use fmt::Write;
    write!(h, "(")?;
    if let Some(region) = region {
        let args = h
            .ctx()
            .region(region)
            .blocks
            .first()
            .map(|block| h.ctx().block_args(*block).to_vec())
            .unwrap_or_default();
        for (index, arg) in args.iter().copied().enumerate() {
            if index > 0 {
                write!(h, ", ")?;
            }
            let name = h.assign_value_name(arg);
            write!(h, "{name}: ")?;
            h.write_type(h.ctx().value_ty(arg))?;
        }
    } else {
        for (index, ty) in param_types.iter().copied().enumerate() {
            if index > 0 {
                write!(h, ", ")?;
            }
            write!(h, "%arg{index}: ")?;
            h.write_type(ty)?;
        }
    }
    write!(h, ")")
}

fn print_func(h: &mut trunk_ir::printer::OpPrintHelper<'_, '_>, op: OpRef) -> fmt::Result {
    use fmt::Write;

    let symbol = h.ctx().op_attrs(op).get_str("sym_name");
    let callable_ty = h.ctx().op_attrs(op).get_type("type");
    let region = h.ctx().op_region(op, 0);
    let header = BinderHeader::of(h.ctx(), op);
    let Some(header) = header.filter(|_| has_concise_func_sig(h.ctx(), callable_ty)) else {
        return h.print_generic(op);
    };
    let parts = callable_ty.and_then(|ty| func_sig_parts(h.ctx(), ty));

    write!(h, "tribute_control.func ")?;
    if let Some(symbol) = symbol {
        h.write_symbol_text(symbol)?;
    } else {
        write!(h, "@<missing>")?;
    }
    header.print(h)?;
    h.reset_numbering();

    let param_types = parts
        .as_ref()
        .map(|(_, params, _)| params.as_slice())
        .unwrap_or(&[]);
    print_signature_params(h, region, param_types)?;

    if let Some((result, _, convention)) = parts {
        write!(h, " -> ")?;
        h.write_type(result)?;
        if let Some(ty) = callable_ty {
            print_row(h, &FuncSig(ty).row(h.ctx()))?;
        }
        if header.inferred {
            write!(h, " inferred")?;
        }
        if let Some(convention) = convention {
            write!(h, " convention({})", convention.keyword())?;
        }
    }
    print_extra_attributes(
        h,
        op,
        &[
            "sym_name",
            "type",
            CALLING_CONVENTION_ATTR,
            ROW_BINDERS_ATTR,
            ROW_UNIONS_ATTR,
            INFERRED_ROW_ATTR,
        ],
    )?;

    match region {
        Some(region) => h.print_body(region),
        None => Ok(()),
    }
}

/// The concise ` convention(...)`; `None` when the signature has none yet.
fn parse_convention(input: &mut &str) -> winnow::ModalResult<Option<CallingConvention>> {
    use trunk_ir::parser::raw::{ident, ws};
    use winnow::combinator::opt;
    use winnow::prelude::*;

    let checkpoint = *input;
    ws.parse_next(input)?;
    if opt("convention").parse_next(input)?.is_none() {
        *input = checkpoint;
        return Ok(None);
    }
    ws.parse_next(input)?;
    '('.parse_next(input)?;
    ws.parse_next(input)?;
    let keyword = ident.parse_next(input)?;
    let Some(convention) = CallingConvention::from_keyword(keyword) else {
        return Err(winnow::error::ErrMode::Backtrack(
            winnow::error::ContextError::new(),
        ));
    };
    ws.parse_next(input)?;
    ')'.parse_next(input)?;
    Ok(Some(convention))
}

/// The concise row of a source signature, `row(@A, @B | 'a1)`, as the raw
/// `ability_row` attribute; `None` when the signature writes no row.
fn parse_row<'a>(
    input: &mut &'a str,
) -> winnow::ModalResult<Option<trunk_ir::parser::raw::RawAttribute<'a>>> {
    use trunk_ir::parser::raw::{RawAttribute, symbol_path, ws};
    use winnow::combinator::{opt, separated};
    use winnow::prelude::*;

    let checkpoint = *input;
    ws.parse_next(input)?;
    if opt("row(").parse_next(input)?.is_none() {
        *input = checkpoint;
        return Ok(None);
    }
    ws.parse_next(input)?;
    let instances: Vec<Vec<String>> =
        separated(0.., (ws, symbol_path, ws).map(|(_, path, _)| path), ',').parse_next(input)?;
    ws.parse_next(input)?;
    let tail = opt(('|', ws, parse_binder_name, ws))
        .parse_next(input)?
        .map(|(_, _, tail, _)| tail);
    ')'.parse_next(input)?;
    if instances.is_empty() && tail.is_none() {
        return Err(winnow::error::ErrMode::Backtrack(
            winnow::error::ContextError::new(),
        ));
    }
    let mut entries = Vec::new();
    if !instances.is_empty() {
        entries.push((
            "instances".into(),
            RawAttribute::List(instances.into_iter().map(RawAttribute::SymbolRef).collect()),
        ));
    }
    if let Some(tail) = tail {
        entries.push(("tail".into(), RawAttribute::Int(i128::from(tail))));
    }
    Ok(Some(RawAttribute::Dict(entries)))
}

fn func_sig_raw_type<'a>(
    result: trunk_ir::parser::raw::RawType<'a>,
    params: &[(&'a str, trunk_ir::parser::raw::RawType<'a>)],
    row: Option<trunk_ir::parser::raw::RawAttribute<'a>>,
    convention: Option<CallingConvention>,
) -> trunk_ir::parser::raw::RawType<'a> {
    use trunk_ir::parser::raw::{RawAttribute, RawType};
    let mut attrs: Vec<_> = convention
        .map(|convention| {
            (
                CALLING_CONVENTION_ATTR.into(),
                RawAttribute::Int(convention as i128),
            )
        })
        .into_iter()
        .collect();
    if let Some(row) = row {
        attrs.push((ABILITY_ROW_ATTR.into(), row));
    }
    RawType::Function {
        name: "tribute_control.func_sig",
        inputs: params.iter().map(|(_, ty)| ty.clone().into()).collect(),
        results: vec![result.into()],
        attrs,
    }
}

/// Whether an operation's attributes repeat metadata its signature type owns.
fn has_duplicate_convention_attr(
    attrs: &[(
        std::borrow::Cow<'_, str>,
        trunk_ir::parser::raw::RawAttribute<'_>,
    )],
) -> bool {
    attrs
        .iter()
        .any(|(key, _)| key == CALLING_CONVENTION_ATTR || key == ABILITY_ROW_ATTR)
}

fn parse_func<'a>(
    input: &mut &'a str,
    results: Vec<&'a str>,
    sym_name: Option<String>,
) -> winnow::ModalResult<trunk_ir::parser::raw::RawOperation<'a>> {
    use trunk_ir::parser::raw::*;
    use winnow::combinator::opt;
    use winnow::prelude::*;

    if !results.is_empty() || sym_name.is_none() {
        return Err(winnow::error::ErrMode::Backtrack(
            winnow::error::ContextError::new(),
        ));
    }

    let binders = parse_binders(input)?;
    ws.parse_next(input)?;
    let params = func_params.parse_next(input)?;
    let result = return_type.parse_next(input)?;
    let row = parse_row(input)?;
    let inferred = parse_inferred(input)?;
    let convention = parse_convention(input)?;
    let attributes = opt((ws, "attributes", ws, raw_attr_dict))
        .parse_next(input)?
        .map(|(_, _, _, attrs)| attrs)
        .unwrap_or_default();
    if has_duplicate_convention_attr(&attributes)
        || attributes.iter().any(|(key, _)| {
            matches!(
                key.as_ref(),
                "sym_name" | "type" | ROW_BINDERS_ATTR | ROW_UNIONS_ATTR | INFERRED_ROW_ATTR
            )
        })
    {
        return Err(winnow::error::ErrMode::Backtrack(
            winnow::error::ContextError::new(),
        ));
    }

    ws.parse_next(input)?;
    let mut regions = Vec::new();
    if input.starts_with('{') {
        let mut region = raw_region.parse_next(input)?;
        if !region.blocks.is_empty() {
            let mut merged = params.clone();
            merged.append(&mut region.blocks[0].args);
            region.blocks[0].args = merged;
        }
        regions.push(region);
    }

    let signature = func_sig_raw_type(result, &params, row, convention);
    let mut attributes = attributes;
    attributes.extend(binders);
    if inferred {
        attributes.push((INFERRED_ROW_ATTR.into(), RawAttribute::Bool(true)));
    }
    attributes.push(("type".into(), RawAttribute::Type(signature)));

    Ok(RawOperation {
        results,
        name: "tribute_control.func",
        sym_name,
        has_func_signature: false,
        func_params: vec![],
        return_type: None,
        operands: vec![],
        attributes,
        result_types: vec![],
        regions,
        successors: vec![],
    })
}

impl trunk_ir::asm_format::OpAsmFormat for Func {
    fn print(&self, h: &mut trunk_ir::printer::OpPrintHelper<'_, '_>) -> std::fmt::Result {
        print_func(h, self.op_ref())
    }

    fn parse<'a>(
        input: &mut &'a str,
        results: Vec<&'a str>,
        sym_name: Option<String>,
    ) -> winnow::ModalResult<trunk_ir::parser::raw::RawOperation<'a>> {
        parse_func(input, results, sym_name)
    }
}

// === Custom assembly: tribute_control.lambda ===

fn print_lambda(h: &mut trunk_ir::printer::OpPrintHelper<'_, '_>, op: OpRef) -> fmt::Result {
    use fmt::Write;

    let result_types = h.ctx().op_result_types(op);
    let callable_ty = result_types.first().copied();
    // An empty list written out has no header form.
    let quantified = quantified_rows(h.ctx(), op).ok().filter(|quantified| {
        !quantified.is_empty() || !h.ctx().op_attrs(op).contains_key(QUANTIFIED_ROWS_ATTR)
    });
    let Some(quantified) = quantified
        .filter(|_| result_types.len() == 1 && has_concise_func_sig(h.ctx(), callable_ty))
    else {
        return h.print_generic(op);
    };

    let result = h.ctx().op_results(op).first().copied();
    let result_name = result.map(|value| h.assign_value_name(value));
    let parts = callable_ty.and_then(|ty| func_sig_parts(h.ctx(), ty));
    let region = h.ctx().op_region(op, 0);

    if let Some(name) = result_name {
        write!(h, "{name} = ")?;
    }
    write!(h, "tribute_control.lambda")?;
    if !quantified.is_empty() {
        write!(h, "[")?;
        for (index, binder) in quantified.iter().enumerate() {
            if index > 0 {
                write!(h, ", ")?;
            }
            write!(h, "'a{binder}")?;
        }
        write!(h, "]")?;
    }

    let param_types = parts
        .as_ref()
        .map(|(_, params, _)| params.as_slice())
        .unwrap_or(&[]);
    print_signature_params(h, region, param_types)?;
    if let Some((result, _, convention)) = parts {
        write!(h, " -> ")?;
        h.write_type(result)?;
        if let Some(ty) = callable_ty {
            print_row(h, &FuncSig(ty).row(h.ctx()))?;
        }
        if let Some(convention) = convention {
            write!(h, " convention({})", convention.keyword())?;
        }
    }
    write!(h, " captures [")?;
    let captures: Vec<String> = h
        .ctx()
        .op_operands(op)
        .iter()
        .map(|value| h.get_value_name(*value).to_owned())
        .collect();
    for (index, capture) in captures.iter().enumerate() {
        if index > 0 {
            write!(h, ", ")?;
        }
        write!(h, "{capture}")?;
    }
    write!(h, "]")?;
    print_extra_attributes(h, op, &[CALLING_CONVENTION_ATTR, QUANTIFIED_ROWS_ATTR])?;

    match region {
        Some(region) => h.print_body(region),
        None => Ok(()),
    }
}

fn parse_captures<'a>(input: &mut &'a str) -> winnow::ModalResult<Vec<&'a str>> {
    use trunk_ir::parser::raw::{value_ref, ws};
    use winnow::combinator::{delimited, separated};
    use winnow::prelude::*;

    ws.parse_next(input)?;
    "captures".parse_next(input)?;
    ws.parse_next(input)?;
    delimited(
        ('[', ws),
        separated(0.., (ws, value_ref, ws).map(|(_, value, _)| value), ','),
        (ws, ']'),
    )
    .parse_next(input)
}

fn parse_lambda<'a>(
    input: &mut &'a str,
    results: Vec<&'a str>,
    sym_name: Option<String>,
) -> winnow::ModalResult<trunk_ir::parser::raw::RawOperation<'a>> {
    use trunk_ir::parser::raw::*;
    use winnow::combinator::{opt, separated};
    use winnow::prelude::*;

    if results.len() != 1 || sym_name.is_some() {
        return Err(winnow::error::ErrMode::Backtrack(
            winnow::error::ContextError::new(),
        ));
    }

    ws.parse_next(input)?;
    let quantified: Option<Vec<u32>> = opt((
        '[',
        separated(
            1..,
            (ws, parse_binder_name, ws).map(|(_, binder, _)| binder),
            ',',
        ),
        ']',
    ))
    .parse_next(input)?
    .map(|(_, binders, _)| binders);
    ws.parse_next(input)?;
    let params = func_params.parse_next(input)?;
    let result = return_type.parse_next(input)?;
    let row = parse_row(input)?;
    let convention = parse_convention(input)?;
    let captures = parse_captures(input)?;
    let mut attributes = opt((ws, "attributes", ws, raw_attr_dict))
        .parse_next(input)?
        .map(|(_, _, _, attrs)| attrs)
        .unwrap_or_default();
    if has_duplicate_convention_attr(&attributes)
        || attributes
            .iter()
            .any(|(key, _)| key == QUANTIFIED_ROWS_ATTR)
    {
        return Err(winnow::error::ErrMode::Backtrack(
            winnow::error::ContextError::new(),
        ));
    }
    if let Some(quantified) = quantified {
        attributes.push((
            QUANTIFIED_ROWS_ATTR.into(),
            RawAttribute::List(
                quantified
                    .into_iter()
                    .map(|binder| RawAttribute::Int(i128::from(binder)))
                    .collect(),
            ),
        ));
    }

    ws.parse_next(input)?;
    let mut region = raw_region.parse_next(input)?;
    if !region.blocks.is_empty() {
        let mut merged = params.clone();
        merged.append(&mut region.blocks[0].args);
        region.blocks[0].args = merged;
    }

    Ok(RawOperation {
        results,
        name: "tribute_control.lambda",
        sym_name,
        has_func_signature: false,
        func_params: vec![],
        return_type: None,
        operands: captures,
        attributes,
        result_types: vec![func_sig_raw_type(result, &params, row, convention)],
        regions: vec![region],
        successors: vec![],
    })
}

impl trunk_ir::asm_format::OpAsmFormat for Lambda {
    fn print(&self, h: &mut trunk_ir::printer::OpPrintHelper<'_, '_>) -> std::fmt::Result {
        print_lambda(h, self.op_ref())
    }

    fn parse<'a>(
        input: &mut &'a str,
        results: Vec<&'a str>,
        sym_name: Option<String>,
    ) -> winnow::ModalResult<trunk_ir::parser::raw::RawOperation<'a>> {
        parse_lambda(input, results, sym_name)
    }
}

// === Explicit Tribute validation entry point ===

/// One operation an `ability.decl` of the module declares, as whole-IR
/// validation reads it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperationDeclaration {
    pub ability: SymbolPath,
    pub op_name: StringRef,
    pub kind: StringRef,
    pub parameter_types: Vec<TypeRef>,
    pub result_type: TypeRef,
}

/// One exact compiler-intrinsic declaration supplied by the typed frontend.
///
/// This metadata is deliberately kept outside textual TrunkIR. A bodyless
/// declaration's operation attribute is accepted only when it matches the
/// identity registered for its symbol. Its signature is the declaration's own;
/// the frontend checks it against the identity before lowering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompilerIntrinsicDeclaration {
    pub symbol: SymbolPath,
    pub identity: Symbol,
}

impl CompilerIntrinsicDeclaration {
    pub fn new(symbol: SymbolPath, identity: Symbol) -> Self {
        Self { symbol, identity }
    }
}

impl OperationDeclaration {
    pub fn new(
        ability: SymbolPath,
        op_name: StringRef,
        kind: StringRef,
        parameter_types: impl IntoIterator<Item = TypeRef>,
        result_type: TypeRef,
    ) -> Self {
        Self {
            ability,
            op_name,
            kind,
            parameter_types: parameter_types.into_iter().collect(),
            result_type,
        }
    }
}

/// Add an `ability.decl` to `module` for each ability instance of
/// `declarations`, holding the operations given for it, in first-use order.
pub fn declare_operations(
    ctx: &mut IrContext,
    module: Module,
    declarations: &[OperationDeclaration],
) {
    let mut instances = Vec::new();
    for declaration in declarations {
        if !instances.contains(&&declaration.ability) {
            instances.push(&declaration.ability);
        }
    }
    declare_abilities(ctx, module, instances, declarations);
}

/// Add an `ability.decl` to `module` for each of `instances`, in order,
/// holding the operations `declarations` gives for it, if any.
pub fn declare_abilities<'a>(
    ctx: &mut IrContext,
    module: Module,
    instances: impl IntoIterator<Item = &'a SymbolPath>,
    declarations: &[OperationDeclaration],
) {
    use trunk_ir::context::{BlockData, RegionData};

    let location = ctx.op(module.op()).location;
    let module_block = module
        .first_block(ctx)
        .expect("a module that declares ability operations has a body block");
    for ability in instances {
        let block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: Default::default(),
            parent_region: None,
        });
        for declaration in declarations
            .iter()
            .filter(|declaration| declaration.ability == *ability)
        {
            let operation = crate::dialect::ability::Operation::operands()
                .op_name(declaration.op_name)
                .kind(declaration.kind)
                .param_types(declaration.parameter_types.iter().copied())
                .result_type(declaration.result_type)
                .build(ctx, location);
            ctx.push_op(block, operation.op_ref());
        }
        let operations = ctx.create_region(RegionData {
            location,
            blocks: trunk_ir::smallvec::smallvec![block],
            parent_op: None,
        });
        let name = ability
            .as_simple()
            .expect("an ability instance is declared in the root module");
        let decl = crate::dialect::ability::Decl::operands()
            .sym_name(name.to_string())
            .regions(operations)
            .build(ctx, location);
        ctx.push_op(module_block, decl.op_ref());
    }
}

/// The operations the `ability.decl` definitions of `module` declare, in
/// module order.
pub fn operation_declarations(ctx: &IrContext, module: Module) -> Vec<OperationDeclaration> {
    declared_operations(ctx, module, &mut Vec::new())
}

fn declared_operations(
    ctx: &IrContext,
    module: Module,
    errors: &mut Vec<ValidationError>,
) -> Vec<OperationDeclaration> {
    use crate::dialect::ability::{Decl, Operation};

    let mut declarations = Vec::new();
    for &op in module.ops(ctx) {
        let Ok(decl) = Decl::from_op(ctx, op) else {
            continue;
        };
        let Some(ability) = qualified_name(ctx, op) else {
            continue;
        };
        let blocks = &ctx.region(decl.operations(ctx)).blocks;
        for &operation in blocks.iter().flat_map(|block| &ctx.block(*block).ops) {
            match Operation::from_op(ctx, operation) {
                Ok(operation) => declarations.push(OperationDeclaration {
                    ability: ability.clone(),
                    op_name: operation.op_name_ref(ctx),
                    kind: operation.kind_ref(ctx),
                    parameter_types: operation.param_types(ctx).collect(),
                    result_type: operation.result_type(ctx),
                }),
                Err(_) => push_op_error(
                    ctx,
                    operation,
                    errors,
                    "ability.decl may hold only ability.operation",
                ),
            }
        }
    }
    declarations
}

/// A focused `tribute_control` validation failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidationError {
    pub op: Option<OpRef>,
    pub location: Option<Location>,
    pub message: String,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(op) = self.op {
            write!(f, "{op}: ")?;
        }
        write!(f, "{}", self.message)
    }
}

/// Result returned by the explicit Tribute validator.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ValidationResult {
    pub errors: Vec<ValidationError>,
}

impl ValidationResult {
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }
}

impl fmt::Display for ValidationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_ok() {
            return write!(f, "tribute_control validation passed");
        }
        writeln!(
            f,
            "{} tribute_control validation error(s):",
            self.errors.len()
        )?;
        for error in &self.errors {
            writeln!(f, "  - {error}")?;
        }
        Ok(())
    }
}

fn push_op_error(
    ctx: &IrContext,
    op: OpRef,
    errors: &mut Vec<ValidationError>,
    message: impl Into<String>,
) {
    errors.push(ValidationError {
        op: Some(op),
        location: Some(ctx.op(op).location),
        message: message.into(),
    });
}

fn push_type_error(errors: &mut Vec<ValidationError>, message: impl Into<String>) {
    errors.push(ValidationError {
        op: None,
        location: None,
        message: message.into(),
    });
}

fn parent_op(ctx: &IrContext, op: OpRef) -> Option<OpRef> {
    let block = ctx.op(op).parent_block?;
    let region = ctx.block(block).parent_region?;
    ctx.region(region).parent_op
}

fn parent_region(ctx: &IrContext, op: OpRef) -> Option<RegionRef> {
    let block = ctx.op(op).parent_block?;
    ctx.block(block).parent_region
}

fn is_never(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.get_type(ty);
    data.name.is(ctx, "core.never")
}

fn is_unresolved_type(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.get_type(ty);
    let unresolved_name = matches!(data.name.strip_dialect(ctx), "var" | "infer" | "unresolved");
    (matches!(data.name.dialect(ctx), "type" | "tribute")) && unresolved_name
}

fn contains_unresolved_type(ctx: &IrContext, ty: TypeRef, visiting: &mut HashSet<TypeRef>) -> bool {
    if !visiting.insert(ty) {
        return false;
    }
    let data = ctx.get_type(ty);
    let unresolved = is_unresolved_type(ctx, ty)
        || data
            .params
            .iter()
            .copied()
            .any(|param| contains_unresolved_type(ctx, param, visiting));
    visiting.remove(&ty);
    unresolved
}

fn contains_forbidden_logical_component(
    ctx: &IrContext,
    ty: TypeRef,
    visiting: &mut HashSet<TypeRef>,
) -> bool {
    if !visiting.insert(ty) {
        return true;
    }
    let data = ctx.get_type(ty);
    let forbidden = (data.name.is(ctx, "func.func_sig"))
        || (data.name.is(ctx, "closure.closure"))
        || ResumeToken::matches(ctx, ty)
        || is_unresolved_type(ctx, ty)
        || data
            .params
            .iter()
            .copied()
            .any(|param| contains_forbidden_logical_component(ctx, param, visiting));
    visiting.remove(&ty);
    forbidden
}

fn validate_control_types(ctx: &IrContext, errors: &mut Vec<ValidationError>) {
    for (ty, data) in ctx.types().iter() {
        if data.name.dialect(ctx) != DIALECT_NAME {
            continue;
        }
        if FuncSig::matches(ctx, ty) {
            if let Err(error) = <FuncSig as trunk_ir::type_verifier::VerifyType>::verify(ctx, ty) {
                push_type_error(
                    errors,
                    format!("{ty}: malformed tribute_control.func_sig: {error:?}"),
                );
                continue;
            }
            for component in data.params.iter().copied() {
                if contains_forbidden_logical_component(ctx, component, &mut HashSet::default()) {
                    push_type_error(
                        errors,
                        format!(
                            "{ty}: callable component {component} is unresolved, recursive, physical, or contains a resume token"
                        ),
                    );
                }
            }
        } else if ResumeToken::matches(ctx, ty) {
            if data.params.len() != 2 {
                push_type_error(
                    errors,
                    format!(
                        "{ty}: tribute_control.resume_token requires exactly input and answer types"
                    ),
                );
            } else {
                for component in data.params.iter().copied() {
                    if contains_forbidden_logical_component(ctx, component, &mut HashSet::default())
                    {
                        push_type_error(
                            errors,
                            format!(
                                "{ty}: resume token component {component} is not a resolved logical value type"
                            ),
                        );
                    }
                }
            }
        } else {
            push_type_error(
                errors,
                format!(
                    "{ty}: unsupported tribute_control type '{}'",
                    data.name.strip_dialect(ctx)
                ),
            );
        }
    }
}

fn validate_attr_keys(
    ctx: &IrContext,
    op: OpRef,
    schema: &OpSchema,
    allowed_extra: bool,
    errors: &mut Vec<ValidationError>,
) {
    let data = ctx.op(op);
    if ctx.op_attrs(op).get(CALLING_CONVENTION_ATTR).is_some() {
        push_op_error(
            ctx,
            op,
            errors,
            format!(
                "{CALLING_CONVENTION_ATTR} belongs only on tribute_control.func_sig, not on the operation"
            ),
        );
    }
    if ctx.op_attrs(op).contains_key(ABILITY_ROW_ATTR) {
        push_op_error(
            ctx,
            op,
            errors,
            format!(
                "{ABILITY_ROW_ATTR} belongs only on tribute_control.func_sig, not on the operation"
            ),
        );
    }
    if allowed_extra {
        return;
    }
    for key in data.attributes.keys() {
        let key = ctx.str(key);
        let declared = schema.attributes.iter().any(|attr| attr.name == key);
        if !declared {
            push_op_error(
                ctx,
                op,
                errors,
                format!("has unsupported attribute '{key}'"),
            );
        }
    }
}

fn single_block(
    ctx: &IrContext,
    owner: OpRef,
    region: RegionRef,
    name: &str,
    errors: &mut Vec<ValidationError>,
) -> Option<BlockRef> {
    let blocks = &ctx.region(region).blocks;
    if let [block] = blocks.as_slice() {
        Some(*block)
    } else {
        push_op_error(
            ctx,
            owner,
            errors,
            format!(
                "{name} region expects exactly one block, found {}",
                blocks.len()
            ),
        );
        None
    }
}

fn terminator(
    ctx: &IrContext,
    owner: OpRef,
    block: BlockRef,
    expected: &'static OpDef,
    errors: &mut Vec<ValidationError>,
) -> Option<OpRef> {
    let Some(&op) = ctx.block(block).ops.last() else {
        push_op_error(
            ctx,
            owner,
            errors,
            format!("region must terminate with {}", expected.name.as_str()),
        );
        return None;
    };
    let data = ctx.op(op);
    if data.name != expected {
        push_op_error(
            ctx,
            owner,
            errors,
            format!("region must terminate with {}", expected.name.as_str()),
        );
        None
    } else {
        Some(op)
    }
}

fn check_types_equal(
    ctx: &IrContext,
    owner: OpRef,
    actual: &[TypeRef],
    expected: &[TypeRef],
    what: &str,
    errors: &mut Vec<ValidationError>,
) {
    if actual != expected {
        push_op_error(
            ctx,
            owner,
            errors,
            format!("{what} types do not match the logical signature"),
        );
    }
}

fn block_arg_types(ctx: &IrContext, block: BlockRef) -> Vec<TypeRef> {
    ctx.block_args(block)
        .iter()
        .map(|value| ctx.value_ty(*value))
        .collect()
}

fn value_types(ctx: &IrContext, values: &[ValueRef]) -> Vec<TypeRef> {
    values.iter().map(|value| ctx.value_ty(*value)).collect()
}

fn validate_callable_body(
    ctx: &IrContext,
    owner: OpRef,
    callable_ty: TypeRef,
    body: RegionRef,
    errors: &mut Vec<ValidationError>,
) {
    let (result_ty, params, _) =
        func_sig_parts(ctx, callable_ty).expect("schema-verified tribute_control.func_sig");
    let Some(block) = single_block(ctx, owner, body, "body", errors) else {
        return;
    };
    check_types_equal(
        ctx,
        owner,
        &block_arg_types(ctx, block),
        &params,
        "entry block argument",
        errors,
    );
    if ctx
        .block(block)
        .ops
        .last()
        .is_some_and(|&last| is_tail_call(ctx, last))
    {
        // The tail call checks its own result against this callable.
        return;
    }
    if let Some(ret) = terminator(ctx, owner, block, Return::DEF, errors)
        && ctx.op_operands(ret).len() == 1
        && ctx.value_ty(ctx.op_operands(ret)[0]) != result_ty
    {
        push_op_error(
            ctx,
            owner,
            errors,
            "return operand type does not match callable result type",
        );
    }
}

fn collect_region_values(ctx: &IrContext, region: RegionRef, values: &mut HashSet<ValueRef>) {
    for block in ctx.region(region).blocks.iter().copied() {
        values.extend(ctx.block_args(block).iter().copied());
        for op in ctx.block(block).ops.iter().copied() {
            values.extend(ctx.op_results(op).iter().copied());
            for nested in ctx.op_regions(op) {
                collect_region_values(ctx, nested, values);
            }
        }
    }
}

fn walk_region_ops(ctx: &IrContext, region: RegionRef, callback: &mut impl FnMut(OpRef)) {
    for block in ctx.region(region).blocks.iter().copied() {
        for op in ctx.block(block).ops.iter().copied() {
            callback(op);
            for nested in ctx.op_regions(op) {
                walk_region_ops(ctx, nested, callback);
            }
        }
    }
}

fn validate_func_isolation(
    ctx: &IrContext,
    op: OpRef,
    body: RegionRef,
    errors: &mut Vec<ValidationError>,
) {
    let mut defined = HashSet::default();
    collect_region_values(ctx, body, &mut defined);
    let mut invalid = Vec::new();
    walk_region_ops(ctx, body, &mut |nested| {
        for operand in ctx.op_operands(nested).iter().copied() {
            if !defined.contains(&operand) {
                invalid.push((nested, operand));
            }
        }
    });
    for (consumer, value) in invalid {
        push_op_error(
            ctx,
            op,
            errors,
            format!("isolated func body operation {consumer} references external value {value}"),
        );
    }
}

fn validate_return_or_yield_shape(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    let Some(block) = ctx.op(op).parent_block else {
        push_op_error(ctx, op, errors, "must be attached to a block");
        return;
    };
    if ctx.block(block).ops.last().copied() != Some(op) {
        push_op_error(ctx, op, errors, "must be the final operation in its block");
    }
}

fn validate_row_declarations(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    let binders = match row_binders(ctx, op) {
        Ok(binders) => binders,
        Err(message) => return push_op_error(ctx, op, errors, message),
    };
    let unions = match row_unions(ctx, op) {
        Ok(unions) => unions,
        Err(message) => return push_op_error(ctx, op, errors, message),
    };
    let mut results = HashSet::default();
    for union in &unions {
        let in_scope = |index: u32| usize::try_from(index).is_ok_and(|index| index < binders.len());
        if !in_scope(union.result) || !union.sources.iter().copied().all(in_scope) {
            push_op_error(
                ctx,
                op,
                errors,
                "row union names a tail that is not a binder",
            );
        } else if union.sources.contains(&union.result) {
            push_op_error(
                ctx,
                op,
                errors,
                "row union lists its result among its sources",
            );
        } else if !results.insert(union.result) {
            push_op_error(
                ctx,
                op,
                errors,
                format!("row tail {} is declared as a union twice", union.result),
            );
        }
    }
}

fn validate_func(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    validate_row_declarations(ctx, op, errors);
    let func = Func::from_op(ctx, op).expect("schema-verified tribute_control.func");
    if let Some(body) = ctx.op_region(op, 0) {
        validate_callable_body(ctx, op, func.r#type(ctx), body, errors);
        validate_func_isolation(ctx, op, body, errors);
    }
}

fn validate_lambda(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    let lambda = Lambda::from_op(ctx, op).expect("schema-verified tribute_control.lambda");
    validate_callable_body(ctx, op, lambda.result_ty(ctx), lambda.body(ctx), errors);
}

fn validate_call(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    let has_unresolved_type = value_types(ctx, ctx.op_operands(op))
        .into_iter()
        .chain(ctx.op_result_types(op).iter().copied())
        .any(|ty| contains_unresolved_type(ctx, ty, &mut HashSet::default()));
    if has_unresolved_type {
        push_op_error(ctx, op, errors, "operands and result must be resolved");
    }
}

fn validate_return(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    validate_return_or_yield_shape(ctx, op, errors);
    let Some(owner) = parent_op(ctx, op) else {
        push_op_error(
            ctx,
            op,
            errors,
            "must terminate a tribute_control.func or lambda body",
        );
        return;
    };
    let callable_ty = if Func::matches(ctx, owner) {
        ctx.op_attrs(owner).get_type("type")
    } else if Lambda::matches(ctx, owner) {
        ctx.op_result_types(owner).first().copied()
    } else {
        None
    };
    let Some((result, _, _)) = callable_ty.and_then(|ty| func_sig_parts(ctx, ty)) else {
        push_op_error(
            ctx,
            op,
            errors,
            "must terminate a tribute_control.func or lambda body",
        );
        return;
    };
    if let Some(&value) = ctx.op_operands(op).first()
        && ctx.value_ty(value) != result
    {
        push_op_error(
            ctx,
            op,
            errors,
            "operand type does not match enclosing callable result",
        );
    }
}

fn is_tail_call(ctx: &IrContext, op: OpRef) -> bool {
    TailCall::matches(ctx, op) || TailCallIndirect::matches(ctx, op)
}

/// The source signature of the callable whose body contains `op`, or `None`
/// when a handle or handler lies between them.
fn tail_call_owner_signature(ctx: &IrContext, op: OpRef) -> Result<TypeRef, &'static str> {
    let mut exit = op;
    loop {
        let Some(owner) = parent_op(ctx, exit) else {
            return Err("must be inside a tribute_control.func or lambda body");
        };
        if Func::matches(ctx, owner) {
            return ctx
                .op_attrs(owner)
                .get_type("type")
                .ok_or("must be inside a tribute_control.func or lambda body");
        }
        if Lambda::matches(ctx, owner) {
            return ctx
                .op_result_types(owner)
                .first()
                .copied()
                .ok_or("must be inside a tribute_control.func or lambda body");
        }
        if Handle::matches(ctx, owner) || Handler::matches(ctx, owner) {
            return Err("must not be inside a tribute_control.handle body or handler");
        }
        // A structured arm may end with a tail call only when the structured
        // operation is a one-result `scf.if` whose value flows straight into
        // its own block's terminator.
        if !trunk_ir::dialect::scf::If::matches(ctx, owner) || ctx.op_results(owner).len() != 1 {
            return Err("must be in tail position of its callable");
        }
        let Some(block) = ctx.op(owner).parent_block else {
            return Err("must be inside a tribute_control.func or lambda body");
        };
        let ops = &ctx.block(block).ops;
        let position = ops
            .iter()
            .position(|&candidate| candidate == owner)
            .expect("an operation is listed in its parent block");
        let mut rest = ops[position + 1..].iter().copied();
        let terminator =
            rest.find(|&candidate| !core::UnrealizedConversionCast::matches(ctx, candidate));
        let Some(terminator) = terminator else {
            return Err("must be in tail position of its callable");
        };
        if ops.last().copied() != Some(terminator) {
            return Err("must be in tail position of its callable");
        }
        if Return::matches(ctx, terminator) {
            exit = terminator;
            continue;
        }
        if trunk_ir::dialect::scf::Yield::matches(ctx, terminator) {
            exit = owner;
            continue;
        }
        return Err("must be in tail position of its callable");
    }
}

fn validate_tail_call(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    validate_return_or_yield_shape(ctx, op, errors);
    let has_unresolved_type = value_types(ctx, ctx.op_operands(op))
        .into_iter()
        .any(|ty| contains_unresolved_type(ctx, ty, &mut HashSet::default()));
    if has_unresolved_type {
        push_op_error(ctx, op, errors, "operands must be resolved");
    }
    let owner_ty = match tail_call_owner_signature(ctx, op) {
        Ok(ty) => ty,
        Err(message) => {
            push_op_error(ctx, op, errors, message);
            return;
        }
    };
    if let Ok(call) = TailCallIndirect::from_op(ctx, op) {
        let callee_ty = ctx.value_ty(call.callee(ctx));
        let (Some((callee_result, _, _)), Some((owner_result, _, _))) = (
            func_sig_parts(ctx, callee_ty),
            func_sig_parts(ctx, owner_ty),
        ) else {
            return;
        };
        if callee_result != owner_result {
            push_op_error(
                ctx,
                op,
                errors,
                "callee result type does not match enclosing callable result",
            );
        }
        if let (Some(callee_cc), Some(owner_cc)) = (
            func_sig_convention(ctx, callee_ty),
            func_sig_convention(ctx, owner_ty),
        ) && callee_cc > owner_cc
        {
            push_op_error(
                ctx,
                op,
                errors,
                "callee convention must not be stronger than the enclosing callable",
            );
        }
    }
}

fn validate_perform(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    match ctx.op_attrs(op).get_str("operation_kind") {
        Some("fn" | "op") => {}
        Some(kind) => push_op_error(
            ctx,
            op,
            errors,
            format!("operation_kind must be \"fn\" or \"op\", found {kind:?}"),
        ),
        None => {}
    }
    let has_unresolved_type = value_types(ctx, ctx.op_operands(op))
        .into_iter()
        .chain(ctx.op_result_types(op).iter().copied())
        .any(|ty| contains_unresolved_type(ctx, ty, &mut HashSet::default()));
    if has_unresolved_type {
        push_op_error(ctx, op, errors, "operands and result must be resolved");
    }
}

fn validate_handle(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    let Some((body_region, completion_region, handlers_region)) =
        ctx.op_regions(op).collect_tuple()
    else {
        return;
    };
    let Some(body_block) = single_block(ctx, op, body_region, "body", errors) else {
        return;
    };
    let Some(completion_block) = single_block(ctx, op, completion_region, "completion", errors)
    else {
        return;
    };
    let Some(handlers_block) = single_block(ctx, op, handlers_region, "handlers", errors) else {
        return;
    };
    if !ctx.block_args(body_block).is_empty() {
        push_op_error(ctx, op, errors, "body block must not have arguments");
    }
    if ctx.block_args(completion_block).len() != 1 {
        push_op_error(
            ctx,
            op,
            errors,
            "completion block must have exactly one argument",
        );
    }
    if !ctx.block_args(handlers_block).is_empty() {
        push_op_error(
            ctx,
            op,
            errors,
            "handlers table block must not have arguments",
        );
    }
    let body_yield = terminator(ctx, op, body_block, Yield::DEF, errors);
    let completion_yield = terminator(ctx, op, completion_block, Yield::DEF, errors);
    if let (Some(body_yield), Some(&completion_arg)) =
        (body_yield, ctx.block_args(completion_block).first())
        && let Some(&body_value) = ctx.op_operands(body_yield).first()
        && ctx.value_ty(body_value) != ctx.value_ty(completion_arg)
    {
        push_op_error(
            ctx,
            op,
            errors,
            "completion argument type must match the body yield type",
        );
    }
    let handle_result = ctx.op_result_types(op).first().copied();
    if let (Some(result), Some(completion_yield)) = (handle_result, completion_yield)
        && let Some(&value) = ctx.op_operands(completion_yield).first()
        && ctx.value_ty(value) != result
    {
        push_op_error(
            ctx,
            op,
            errors,
            "completion yield type must match handle result type",
        );
    }
    let mut clauses = HashSet::default();
    for child in ctx.block(handlers_block).ops.iter().copied() {
        if !Handler::matches(ctx, child) {
            push_op_error(
                ctx,
                op,
                errors,
                format!("handlers block may contain only tribute_control.handler, found {child}"),
            );
            continue;
        }
        let attrs = ctx.op_attrs(child);
        if let (Some(ability), Some(name)) = (
            attrs.get_symbol_ref("ability"),
            attrs.get_string_ref("op_name"),
        ) && !clauses.insert((ability, name))
        {
            push_op_error(
                ctx,
                child,
                errors,
                format!("duplicate handler clause for {ability}::{}", ctx.str(name)),
            );
        }
        if let Some(result) = handle_result
            && let Some(region) = ctx.op_region(child, 0)
            && let Some(&block) = ctx.region(region).blocks.first()
            && let Some(&yield_op) = ctx.block(block).ops.last()
            && Yield::matches(ctx, yield_op)
            && ctx.op_attrs(child).get_str("kind") == Some("op")
            && let Some(&value) = ctx.op_operands(yield_op).first()
            && ctx.value_ty(value) != result
        {
            push_op_error(
                ctx,
                child,
                errors,
                "general handler yield type must match enclosing handle result type",
            );
        }
    }
}

fn region_contains_resume(ctx: &IrContext, region: RegionRef) -> bool {
    let mut found = false;
    walk_region_ops(ctx, region, &mut |op| {
        found |= Resume::matches(ctx, op);
    });
    found
}

fn validate_handler(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    let kind = ctx.op_attrs(op).get_str("kind");
    if let Some(kind) = kind
        && !matches!(kind, "fn" | "op")
    {
        push_op_error(
            ctx,
            op,
            errors,
            format!("kind must be \"fn\" or \"op\", found {kind:?}"),
        );
    }
    let Some(body) = ctx.op_region(op, 0) else {
        return;
    };
    let Some(block) = single_block(ctx, op, body, "body", errors) else {
        return;
    };
    let yield_op = terminator(ctx, op, block, Yield::DEF, errors);
    let result_type = ctx.op_attrs(op).get_type("operation_result_type");
    let handle_result = parent_op(ctx, op)
        .filter(|parent| Handle::matches(ctx, *parent))
        .and_then(|parent| ctx.op_result_types(parent).first().copied());
    let args = ctx.block_args(block);
    let last_token = args
        .last()
        .copied()
        .and_then(|value| resume_token_parts(ctx, ctx.value_ty(value)));
    match (kind, result_type) {
        (Some(kind), Some(operation_result))
            if kind == "op" && !is_never(ctx, operation_result) =>
        {
            let Some((token_input, token_answer)) = last_token else {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "resumptive op handler requires a final resume_token block argument",
                );
                return;
            };
            if token_input != operation_result
                || handle_result.is_some_and(|answer| token_answer != answer)
            {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "resume token input/answer types do not match operation and handle results",
                );
            }
            if let (Some(yield_op), Some(answer)) = (yield_op, handle_result)
                && let Some(&value) = ctx.op_operands(yield_op).first()
                && ctx.value_ty(value) != answer
            {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "general handler yield type must match handle result type",
                );
            }
        }
        (Some(kind), Some(operation_result)) => {
            if last_token.is_some() {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "fn and op -> core.never handlers must not receive a resume token",
                );
            }
            if kind == "op" && is_never(ctx, operation_result) && region_contains_resume(ctx, body)
            {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "op -> core.never handler must not contain tribute_control.resume",
                );
            }
            let expected = if kind == "fn" {
                Some(operation_result)
            } else {
                handle_result
            };
            if let (Some(yield_op), Some(expected)) = (yield_op, expected)
                && let Some(&value) = ctx.op_operands(yield_op).first()
                && ctx.value_ty(value) != expected
            {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "handler yield type does not match its kind-specific result",
                );
            }
        }
        _ => {}
    }

    let placed = parent_op(ctx, op).is_some_and(|parent| {
        Handle::matches(ctx, parent)
            && ctx
                .op_region(parent, 2)
                .is_some_and(|handlers| parent_region(ctx, op) == Some(handlers))
    });
    if !placed {
        push_op_error(
            ctx,
            op,
            errors,
            "must be a direct child of a tribute_control.handle handlers region",
        );
    }
}

fn validate_yield(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    validate_return_or_yield_shape(ctx, op, errors);
    if let Some(&value) = ctx.op_operands(op).first()
        && ResumeToken::matches(ctx, ctx.value_ty(value))
    {
        push_op_error(ctx, op, errors, "must not yield a resume token");
    }
    let valid_owner = parent_op(ctx, op).is_some_and(|owner| {
        if Handler::matches(ctx, owner) {
            true
        } else if Handle::matches(ctx, owner) {
            let region = parent_region(ctx, op);
            ctx.op_region(owner, 0) == region || ctx.op_region(owner, 1) == region
        } else {
            false
        }
    });
    if !valid_owner {
        push_op_error(
            ctx,
            op,
            errors,
            "must terminate a handle body/completion or handler body",
        );
    }
}

fn validate_local_operation(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    let data = ctx.op(op);
    if data.name.dialect(ctx) != DIALECT_NAME {
        return;
    }
    let Some(def) = OpDef::of(ctx, op) else {
        push_op_error(
            ctx,
            op,
            errors,
            format!(
                "unsupported tribute_control operation '{}'",
                data.name.strip_dialect(ctx)
            ),
        );
        return;
    };
    // Counts, attribute kinds, and declared type relations come from the
    // schema; the checks below assume them.
    let violations = def.verify(ctx, op);
    if !violations.is_empty() {
        for violation in violations {
            push_op_error(ctx, op, errors, violation.to_string());
        }
        return;
    }
    validate_attr_keys(
        ctx,
        op,
        &def.schema,
        Func::matches(ctx, op) || Lambda::matches(ctx, op),
        errors,
    );
    let validate: Option<fn(&IrContext, OpRef, &mut Vec<ValidationError>)> =
        if Func::matches(ctx, op) {
            Some(validate_func)
        } else if Lambda::matches(ctx, op) {
            Some(validate_lambda)
        } else if Call::matches(ctx, op) {
            Some(validate_call)
        } else if Return::matches(ctx, op) {
            Some(validate_return)
        } else if TailCall::matches(ctx, op) || TailCallIndirect::matches(ctx, op) {
            Some(validate_tail_call)
        } else if Perform::matches(ctx, op) {
            Some(validate_perform)
        } else if Handle::matches(ctx, op) {
            Some(validate_handle)
        } else if Handler::matches(ctx, op) {
            Some(validate_handler)
        } else if Yield::matches(ctx, op) {
            Some(validate_yield)
        } else {
            None
        };
    if let Some(validate) = validate {
        validate(ctx, op, errors);
    }
}

/// Validate only operation-local and type-local `tribute_control` invariants.
pub fn validate_local(ctx: &IrContext, module: Module) -> ValidationResult {
    let mut errors = Vec::new();
    validate_control_types(ctx, &mut errors);
    if let Some(body) = module.body(ctx) {
        walk_region_ops(ctx, body, &mut |op| {
            validate_local_operation(ctx, op, &mut errors);
        });
    }
    ValidationResult { errors }
}

/// The row arguments a reference passes to `target`, once their count
/// matches the binders `target` quantifies; `None` after reporting why not.
fn reference_row_args(
    ctx: &IrContext,
    op: OpRef,
    target: OpRef,
    errors: &mut Vec<ValidationError>,
) -> Option<Vec<Row>> {
    let args = row_args(ctx, op).ok()?;
    let quantified = quantified_count(&row_binders(ctx, target).ok()?);
    if args.len() != quantified {
        push_op_error(
            ctx,
            op,
            errors,
            format!(
                "passes {} row arguments, but its target quantifies {quantified} row binders",
                args.len()
            ),
        );
        return None;
    }
    Some(args)
}

/// `target`'s signature as a reference with row arguments sees it. A type
/// that cannot be found is `None` and matches no value.
struct SubstitutedSignature {
    result: Option<TypeRef>,
    inputs: Vec<Option<TypeRef>>,
    row: Option<Row>,
}

fn substituted_parts(
    ctx: &IrContext,
    target_ty: TypeRef,
    args: &[Row],
) -> Option<SubstitutedSignature> {
    let signature = FuncSig::from_type_ref(ctx, target_ty)?;
    Some(SubstitutedSignature {
        result: find_substituted_rows(ctx, signature.result(ctx), args),
        inputs: signature
            .inputs(ctx)
            .iter()
            .map(|&input| find_substituted_rows(ctx, input, args))
            .collect(),
        row: signature.row(ctx).substitute(args),
    })
}

fn types_match(actual: &[TypeRef], expected: &[Option<TypeRef>]) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| Some(*actual) == *expected)
}

fn validate_symbol_uses(
    ctx: &IrContext,
    module: Module,
    funcs: &SymbolTable,
    errors: &mut Vec<ValidationError>,
) {
    let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
        validate_symbol_use(ctx, op, funcs, errors);
        ControlFlow::Continue(WalkAction::Advance)
    });
}

fn validate_symbol_use(
    ctx: &IrContext,
    op: OpRef,
    funcs: &SymbolTable,
    errors: &mut Vec<ValidationError>,
) {
    {
        if FuncRef::matches(ctx, op) {
            let Some(symbol) = ctx.op_attrs(op).get_symbol_ref("func_ref") else {
                return;
            };
            let Some(target) = funcs
                .definitions(ctx, op, symbol)
                .first()
                .copied()
                .filter(|&target| Func::matches(ctx, target))
            else {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    format!("unresolved function symbol @{symbol}"),
                );
                return;
            };
            let Some(target_ty) = ctx.op_attrs(target).get_type("type") else {
                return;
            };
            let Some(result_ty) = ctx.op_result_types(op).first().copied() else {
                return;
            };
            let Some(args) = reference_row_args(ctx, op, target, errors) else {
                return;
            };
            let same_signature = substituted_parts(ctx, target_ty, &args)
                .zip(FuncSig::from_type_ref(ctx, result_ty))
                .is_some_and(|(expected, signature)| {
                    Some(signature.result(ctx)) == expected.result
                        && types_match(signature.inputs(ctx), &expected.inputs)
                        && expected
                            .row
                            .is_some_and(|row| signature.row(ctx).includes(&row))
                });
            if !same_signature {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "func_ref result source signature does not match its target",
                );
            } else if let (Some(target_cc), Some(result_cc)) = (
                func_sig_convention(ctx, target_ty),
                func_sig_convention(ctx, result_ty),
            ) && result_cc < target_cc
            {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "func_ref result convention must be at least as strong as its target",
                );
            }
        } else if Call::matches(ctx, op) {
            let Some(symbol) = ctx.op_attrs(op).get_symbol_ref("callee") else {
                return;
            };
            let Some(target) = funcs
                .definitions(ctx, op, symbol)
                .first()
                .copied()
                .filter(|&target| Func::matches(ctx, target))
            else {
                push_op_error(ctx, op, errors, format!("unresolved callee @{symbol}"));
                return;
            };
            let Some(target_ty) = ctx.op_attrs(target).get_type("type") else {
                return;
            };
            let Some(args) = reference_row_args(ctx, op, target, errors) else {
                return;
            };
            let Some(SubstitutedSignature {
                result,
                inputs: params,
                ..
            }) = substituted_parts(ctx, target_ty, &args)
            else {
                return;
            };
            if !types_match(&value_types(ctx, ctx.op_operands(op)), &params) {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "call argument types do not match the logical signature",
                );
            }
            if !types_match(ctx.op_result_types(op), &[result]) {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "call result types do not match the logical signature",
                );
            }
        } else if TailCall::matches(ctx, op) {
            let Some(symbol) = ctx.op_attrs(op).get_symbol_ref("callee") else {
                return;
            };
            let Some(target) = funcs
                .definitions(ctx, op, symbol)
                .first()
                .copied()
                .filter(|&target| Func::matches(ctx, target))
            else {
                push_op_error(ctx, op, errors, format!("unresolved callee @{symbol}"));
                return;
            };
            if ctx.op_attrs(target).get("abi").is_some() {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    format!("cannot tail call foreign function @{symbol}"),
                );
            }
            let Some(target_ty) = ctx.op_attrs(target).get_type("type") else {
                return;
            };
            let Some(args) = reference_row_args(ctx, op, target, errors) else {
                return;
            };
            let Some(SubstitutedSignature {
                result,
                inputs: params,
                ..
            }) = substituted_parts(ctx, target_ty, &args)
            else {
                return;
            };
            if !types_match(&value_types(ctx, ctx.op_operands(op)), &params) {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "tail call argument types do not match the logical signature",
                );
            }
            let Ok(owner_ty) = tail_call_owner_signature(ctx, op) else {
                return;
            };
            if func_sig_parts(ctx, owner_ty)
                .is_some_and(|(owner_result, _, _)| Some(owner_result) != result)
            {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "callee result type does not match enclosing callable result",
                );
            }
            if let (Some(callee_cc), Some(owner_cc)) = (
                func_sig_convention(ctx, target_ty),
                func_sig_convention(ctx, owner_ty),
            ) && callee_cc > owner_cc
            {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "callee convention must not be stronger than the enclosing callable",
                );
            }
        }
    }
}

/// Resolve every function reference in the module tree from its nearest module.
fn validate_module_symbols(
    ctx: &IrContext,
    module: Module,
    funcs: &SymbolTable,
    errors: &mut Vec<ValidationError>,
) {
    for ops in funcs.duplicates() {
        let symbol = qualified_name(ctx, ops[0]).unwrap_or_else(SymbolPath::root);
        for &op in &ops[1..] {
            push_op_error(
                ctx,
                op,
                errors,
                format!("duplicate function symbol @{symbol}"),
            );
        }
    }
    validate_symbol_uses(ctx, module, funcs, errors);
}

fn collect_external_references(
    ctx: &IrContext,
    region: RegionRef,
    defined: &HashSet<ValueRef>,
    external: &mut HashSet<ValueRef>,
) {
    for block in ctx.region(region).blocks.iter().copied() {
        for op in ctx.block(block).ops.iter().copied() {
            for operand in ctx.op_operands(op).iter().copied() {
                if !defined.contains(&operand) {
                    external.insert(operand);
                }
            }
            if Func::matches(ctx, op) {
                continue;
            }
            for nested in ctx.op_regions(op) {
                collect_external_references(ctx, nested, defined, external);
            }
        }
    }
}

fn validate_lambda_captures(ctx: &IrContext, body: RegionRef, errors: &mut Vec<ValidationError>) {
    walk_region_ops(ctx, body, &mut |op| {
        if !Lambda::matches(ctx, op) {
            return;
        }
        let Some(region) = ctx.op_region(op, 0) else {
            return;
        };
        let captures = ctx.op_operands(op);
        let capture_set: HashSet<ValueRef> = captures.iter().copied().collect();
        if capture_set.len() != captures.len() {
            push_op_error(ctx, op, errors, "capture list contains duplicate values");
        }
        let mut defined = HashSet::default();
        collect_region_values(ctx, region, &mut defined);
        let mut external = HashSet::default();
        collect_external_references(ctx, region, &defined, &mut external);
        for missing in external.difference(&capture_set) {
            push_op_error(
                ctx,
                op,
                errors,
                format!("capture list is missing external value {missing}"),
            );
        }
        for excess in capture_set.difference(&external) {
            push_op_error(
                ctx,
                op,
                errors,
                format!("capture list contains unused external value {excess}"),
            );
        }
    });
}

fn declaration_map<'a>(
    ctx: &IrContext,
    declarations: &'a [OperationDeclaration],
    errors: &mut Vec<ValidationError>,
) -> HashMap<(&'a SymbolPath, StringRef), &'a OperationDeclaration> {
    let mut map = HashMap::default();
    for declaration in declarations {
        let key = (&declaration.ability, declaration.op_name);
        if map.insert(key, declaration).is_some() {
            push_type_error(
                errors,
                format!(
                    "duplicate operation declaration for {}::{}",
                    declaration.ability,
                    ctx.str(declaration.op_name)
                ),
            );
        }
    }
    map
}

fn is_adt_typeref(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.get_type(ty);
    data.name == crate::dialect::adt::Typeref::NAME
}

fn contains_adt_typeref(ctx: &IrContext, ty: TypeRef, visiting: &mut HashSet<TypeRef>) -> bool {
    if !visiting.insert(ty) {
        return false;
    }
    let data = ctx.get_type(ty);
    let contains = is_adt_typeref(ctx, ty)
        || data
            .params
            .iter()
            .copied()
            .any(|param| contains_adt_typeref(ctx, param, visiting))
        || data
            .attrs
            .values()
            .any(|attribute| attribute_contains_adt_typeref(ctx, attribute, visiting));
    visiting.remove(&ty);
    contains
}

fn attribute_contains_adt_typeref(
    ctx: &IrContext,
    attribute: &Attribute,
    visiting: &mut HashSet<TypeRef>,
) -> bool {
    let mut contains = false;
    attribute.visit_types(&mut |ty| {
        contains = contains || contains_adt_typeref(ctx, ty, visiting);
    });
    contains
}

fn nominal_identity(ctx: &IrContext, ty: TypeRef) -> Option<StringRef> {
    adt::nominal_name_ref(ctx, ty)
}

fn canonical_nominal_layouts(
    ctx: &IrContext,
    reachable_types: &HashSet<TypeRef>,
    errors: &mut Vec<ValidationError>,
) -> HashMap<StringRef, TypeRef> {
    let mut layouts = HashMap::default();
    let mut referenced_names = HashSet::default();
    let mut sorted_reachable_types = reachable_types.iter().copied().collect::<Vec<_>>();
    sorted_reachable_types.sort_unstable();
    for ty in sorted_reachable_types {
        if is_adt_typeref(ctx, ty) {
            if let Some(name) = ctx.type_attrs(ty).get_string_ref("name") {
                referenced_names.insert(name);
            }
            continue;
        }
        if !adt::Struct::matches(ctx, ty) && !adt::Enum::matches(ctx, ty) {
            continue;
        }
        let Some(name) = ctx.type_attrs(ty).get_string_ref("name") else {
            continue;
        };
        if let Some(previous) = layouts.get(&name).copied() {
            if previous != ty {
                push_type_error(
                    errors,
                    format!(
                        "nominal layout {:?} is declared more than once ({previous} and {ty})",
                        ctx.str(name)
                    ),
                );
            }
        } else {
            layouts.insert(name, ty);
        }
    }
    for name in referenced_names {
        if layouts.contains_key(&name) {
            continue;
        }
        let candidates = ctx
            .types()
            .iter()
            .filter_map(|(ty, data)| {
                ((adt::Struct::matches(ctx, ty) || adt::Enum::matches(ctx, ty))
                    && data.attrs.view(ctx).get_string_ref("name") == Some(name))
                .then_some(ty)
            })
            .collect::<Vec<_>>();
        if let [layout] = candidates.as_slice() {
            layouts.insert(name, *layout);
        } else if let [first, second, ..] = candidates.as_slice() {
            push_type_error(
                errors,
                format!(
                    "nominal layout {:?} is declared more than once ({first} and {second})",
                    ctx.str(name)
                ),
            );
        }
    }
    layouts
}

fn is_core_ptr(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.get_type(ty);
    data.name.is(ctx, "core.ptr")
}

fn collect_attribute_types(ctx: &IrContext, attribute: &Attribute, types: &mut HashSet<TypeRef>) {
    attribute.visit_types(&mut |ty| collect_reachable_type(ctx, ty, types));
}

fn collect_reachable_type(ctx: &IrContext, ty: TypeRef, types: &mut HashSet<TypeRef>) {
    if !types.insert(ty) {
        return;
    }
    let data = ctx.get_type(ty);
    for parameter in data.params.iter().copied() {
        collect_reachable_type(ctx, parameter, types);
    }
    for attribute in data.attrs.values() {
        collect_attribute_types(ctx, attribute, types);
    }
}

/// Collect semantic types reachable from a module operation and its nested IR.
///
/// Context-wide type aliases are printer/parser conveniences, not IR roots:
/// an alias whose type is never used by an operation, operand, result, block
/// argument, or type-bearing attribute must not affect validation of this
/// module. A reachable `adt.typeref` may resolve one unique layout alias; an
/// ambiguous nominal name remains invalid.
/// Whether class specialization has assigned calling conventions in
/// `module`: `Some(true)` when every source signature the module reaches has
/// one, `Some(false)` when none has, and `None` when only some have.
pub fn conventions_assigned(ctx: &IrContext, module: Module) -> Option<bool> {
    let mut assigned = None;
    for ty in collect_reachable_ir_types(ctx, module.op()) {
        let Some(signature) = FuncSig::from_type_ref(ctx, ty) else {
            continue;
        };
        let has = signature.assigned_convention(ctx).is_some();
        if *assigned.get_or_insert(has) != has {
            return None;
        }
    }
    Some(assigned.unwrap_or(true))
}

fn collect_reachable_ir_types(ctx: &IrContext, module: OpRef) -> HashSet<TypeRef> {
    fn collect_op_types(ctx: &IrContext, op: OpRef, types: &mut HashSet<TypeRef>) {
        for value in ctx.op_operands(op).iter().chain(ctx.op_results(op).iter()) {
            collect_reachable_type(ctx, ctx.value_ty(*value), types);
        }
        for attribute in ctx.op(op).attributes.values() {
            collect_attribute_types(ctx, attribute, types);
        }
        for region in ctx.op_regions(op) {
            for block in ctx.region(region).blocks.iter().copied() {
                for argument in ctx.block_args(block) {
                    collect_reachable_type(ctx, ctx.value_ty(*argument), types);
                }
            }
        }
    }

    let mut types = HashSet::default();
    collect_op_types(ctx, module, &mut types);
    for region in ctx.op_regions(module) {
        walk_region_ops(ctx, region, &mut |op| collect_op_types(ctx, op, &mut types));
    }
    types
}

fn raw_pointer_cast_origin(
    ctx: &IrContext,
    value: ValueRef,
    visiting: &mut HashSet<ValueRef>,
) -> bool {
    if !visiting.insert(value) {
        return false;
    }
    if is_core_ptr(ctx, ctx.value_ty(value)) {
        return true;
    }
    let ValueDef::OpResult(producer, _) = ctx.value_def(value) else {
        return false;
    };
    let transparent_cast = core::UnrealizedConversionCast::from_op(ctx, producer).is_ok()
        || adt::RefCast::from_op(ctx, producer).is_ok();
    transparent_cast
        && ctx
            .op_operands(producer)
            .first()
            .copied()
            .is_some_and(|operand| raw_pointer_cast_origin(ctx, operand, visiting))
}

fn validate_managed_reference_boundaries(
    ctx: &IrContext,
    body: RegionRef,
    reachable_types: &HashSet<TypeRef>,
    nominal_layouts: &HashMap<StringRef, TypeRef>,
    errors: &mut Vec<ValidationError>,
) {
    for ty in reachable_types.iter().copied() {
        if !is_adt_typeref(ctx, ty) {
            continue;
        }
        let data = ctx.get_type(ty);
        let Some(name) = ctx.type_attrs(ty).get_string_ref("name") else {
            push_type_error(
                errors,
                format!("{ty}: adt.typeref requires nominal name metadata"),
            );
            continue;
        };
        if !data.params.is_empty() {
            push_type_error(
                errors,
                format!("{ty}: adt.typeref cannot have type parameters"),
            );
        }
        if !nominal_layouts.contains_key(&name) {
            push_type_error(
                errors,
                format!(
                    "{ty}: adt.typeref {:?} has no verified nominal declaration",
                    ctx.str(name)
                ),
            );
        }
    }

    walk_region_ops(ctx, body, &mut |op| {
        let data = ctx.op(op);
        if data.name.is(ctx, "adt.ref_null") {
            let ([], [result], Some(declared)) = (
                ctx.op_operands(op),
                ctx.op_result_types(op),
                ctx.op_attrs(op).get_type("type"),
            ) else {
                push_op_error(ctx, op, errors, "adt.ref_null requires one typed result");
                return;
            };
            if *result != declared || !is_adt_typeref(ctx, *result) {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "adt.ref_null result and type attribute must be the same managed adt.typeref",
                );
            }
        }
        if data.name.is(ctx, "adt.ref_cast") {
            let ([operand], [result], Some(declared)) = (
                ctx.op_operands(op),
                ctx.op_result_types(op),
                ctx.op_attrs(op).get_type("type"),
            ) else {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "adt.ref_cast requires one typed operand and result",
                );
                return;
            };
            let operand_type = ctx.value_ty(*operand);
            let source = nominal_identity(ctx, operand_type);
            let target = nominal_identity(ctx, *result);
            if declared != *result
                || !is_adt_typeref(ctx, operand_type)
                || !is_adt_typeref(ctx, *result)
                || source.is_none()
                || source != target
            {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "adt.ref_cast requires compatible managed nominal reference types",
                );
            }
        }
        for result in ctx.op_results(op) {
            if is_adt_typeref(ctx, ctx.value_ty(*result))
                && raw_pointer_cast_origin(ctx, *result, &mut HashSet::default())
            {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "core.ptr cast chain cannot produce a managed adt.typeref",
                );
            }
        }
    });
}

fn compiler_intrinsic_map<'a>(
    declarations: &'a [CompilerIntrinsicDeclaration],
    errors: &mut Vec<ValidationError>,
) -> HashMap<SymbolPath, &'a CompilerIntrinsicDeclaration> {
    let mut map = HashMap::default();
    let mut previous = None;
    for declaration in declarations {
        let key = (declaration.symbol.clone(), declaration.identity.clone());
        if previous.is_some_and(|previous| previous > key) {
            push_type_error(
                errors,
                "compiler intrinsic declarations are not deterministically ordered",
            );
        }
        previous = Some(key);
        if map
            .insert(declaration.symbol.clone(), declaration)
            .is_some()
        {
            push_type_error(
                errors,
                format!(
                    "duplicate compiler intrinsic declaration for @{}",
                    declaration.symbol
                ),
            );
        }
    }
    map
}

/// `unrealized_conversion_cast` is representation glue, never a source-callable
/// adapter. Convention strengthening must be expressed by `func_ref` and
/// lowered while its declaration provenance is still available. Changing a
/// callable's source shape does not create a physical adapter, regardless of
/// whether its value originates at a named function, lambda, or block argument.
fn validate_callable_conversion_casts(
    ctx: &IrContext,
    body: RegionRef,
    errors: &mut Vec<ValidationError>,
) {
    walk_region_ops(ctx, body, &mut |op| {
        if core::UnrealizedConversionCast::from_op(ctx, op).is_err() {
            return;
        }
        let ([input], [target_ty]) = (ctx.op_operands(op), ctx.op_result_types(op)) else {
            return;
        };
        let source_ty = ctx.value_ty(*input);
        if FuncSig::from_type_ref(ctx, source_ty).is_none()
            || FuncSig::from_type_ref(ctx, *target_ty).is_none()
        {
            return;
        }
        let convention_changes =
            func_sig_convention(ctx, source_ty) != func_sig_convention(ctx, *target_ty);
        if convention_changes {
            push_op_error(
                ctx,
                op,
                errors,
                "core.unrealized_conversion_cast cannot change a source-logical callable calling convention; use tribute_control.func_ref",
            );
        }
    });
}

fn validate_return_contracts(ctx: &IrContext, body: RegionRef, errors: &mut Vec<ValidationError>) {
    walk_region_ops(ctx, body, &mut |op| {
        if !Return::matches(ctx, op) {
            return;
        }
        let mut owner = parent_op(ctx, op);
        let callable = loop {
            let Some(current) = owner else {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "return has no containing semantic callable",
                );
                return;
            };
            if Func::matches(ctx, current) {
                break ctx.op_attrs(current).get_type("type");
            }
            if Lambda::matches(ctx, current) {
                break ctx.op_result_types(current).first().copied();
            }
            owner = parent_op(ctx, current);
        };
        let Some((expected, _, _)) = callable.and_then(|ty| func_sig_parts(ctx, ty)) else {
            return;
        };
        if let [value] = ctx.op_operands(op)
            && ctx.value_ty(*value) != expected
        {
            push_op_error(
                ctx,
                op,
                errors,
                "return operand type does not match containing semantic callable",
            );
        }
    });
}

fn validate_declaration_uses(
    ctx: &IrContext,
    body: RegionRef,
    declarations: &HashMap<(&SymbolPath, StringRef), &OperationDeclaration>,
    errors: &mut Vec<ValidationError>,
) {
    walk_region_ops(ctx, body, &mut |op| {
        let Some(declared) = DynOperationDeclarationUse::get(ctx, op) else {
            return;
        };
        let (Some(claim), Some(args)) = (
            declared.declaration_use(ctx),
            declared.declared_arguments(ctx),
        ) else {
            return;
        };
        let Some(declaration) = declarations.get(&(claim.ability, claim.op_name)) else {
            push_op_error(
                ctx,
                op,
                errors,
                format!(
                    "no resolved operation declaration for {}::{}",
                    claim.ability,
                    ctx.str(claim.op_name)
                ),
            );
            return;
        };
        if claim.kind != Some(declaration.kind) {
            push_op_error(
                ctx,
                op,
                errors,
                format!(
                    "{} does not match the resolved declaration",
                    claim.kind_attr
                ),
            );
        }
        check_types_equal(
            ctx,
            op,
            &value_types(ctx, args),
            &declaration.parameter_types,
            "operation argument",
            errors,
        );
        if declared.declared_result_type(ctx) != Some(declaration.result_type) {
            push_op_error(
                ctx,
                op,
                errors,
                "operation result type does not match the resolved declaration",
            );
        }
    });
}

fn op_is_within_region(ctx: &IrContext, op: OpRef, target: RegionRef) -> bool {
    let mut region = parent_region(ctx, op);
    while let Some(current) = region {
        if current == target {
            return true;
        }
        region = ctx
            .region(current)
            .parent_op
            .and_then(|owner| parent_region(ctx, owner));
    }
    false
}

fn nearest_enclosing_handler(ctx: &IrContext, op: OpRef) -> Option<OpRef> {
    let mut owner = parent_op(ctx, op);
    while let Some(current) = owner {
        if Handler::matches(ctx, current) {
            return Some(current);
        }
        owner = parent_op(ctx, current);
    }
    None
}

fn direct_call_reenters_enclosing_func(ctx: &IrContext, op: OpRef) -> bool {
    if !Call::matches(ctx, op) {
        return false;
    }
    let Some(callee) = ctx.op_attrs(op).get_symbol_ref("callee") else {
        return false;
    };
    let mut owner = parent_op(ctx, op);
    while let Some(current) = owner {
        if Func::matches(ctx, current) {
            // The call is looked up in the module enclosing the function,
            // where the function is a symbol under its own name.
            return parent_op(ctx, current)
                .is_some_and(|module| trunk_ir::dialect::core::Module::matches(ctx, module))
                && ctx
                    .op_attrs(current)
                    .get_str("sym_name")
                    .is_some_and(|name| *callee == [name]);
        }
        owner = parent_op(ctx, current);
    }
    false
}

fn validate_affine_lambda_carrier(
    ctx: &IrContext,
    handler: OpRef,
    carrier: ValueRef,
    visited: &mut HashSet<ValueRef>,
    errors: &mut Vec<ValidationError>,
) {
    if !visited.insert(carrier) {
        push_op_error(
            ctx,
            handler,
            errors,
            "resume-token carrier path contains a cycle",
        );
        return;
    }

    let mut captures = Vec::new();
    let mut terminals = Vec::new();
    for use_ in ctx.uses(carrier) {
        if nearest_enclosing_handler(ctx, use_.user) != Some(handler) {
            push_op_error(
                ctx,
                handler,
                errors,
                format!(
                    "resume-token carrier crosses into a different handler at {}",
                    use_.user
                ),
            );
            continue;
        }
        if Lambda::matches(ctx, use_.user) {
            captures.push(use_.user);
        } else if (CallIndirect::matches(ctx, use_.user) && use_.operand_index == 0)
            // A captured resumption may be passed to the enclosing source
            // function's recursive logical call (for example
            // `run_state(fn() { resume value }, value)`). Other direct calls
            // are rejected: their parameter ownership is not yet proven.
            || direct_call_reenters_enclosing_func(ctx, use_.user)
        {
            terminals.push(use_.user);
        } else {
            push_op_error(
                ctx,
                handler,
                errors,
                format!(
                    "resume-token carrier escapes through {} at operand {}",
                    use_.user, use_.operand_index
                ),
            );
        }
    }

    if has_nonexclusive_pair(ctx, &captures) {
        push_op_error(
            ctx,
            handler,
            errors,
            "resume-token carrier branches into multiple lambda captures",
        );
    }
    if has_nonexclusive_pair(ctx, &terminals) {
        push_op_error(
            ctx,
            handler,
            errors,
            "resume-token carrier has multiple static terminal uses",
        );
    }

    if !captures.is_empty() {
        for terminal in &terminals {
            if CallIndirect::matches(ctx, *terminal)
                && !captures.iter().any(|capture| {
                    ctx.op_region(*capture, 0)
                        .is_some_and(|region| op_is_within_region(ctx, *terminal, region))
                })
            {
                push_op_error(
                    ctx,
                    handler,
                    errors,
                    "resume-token carrier call is outside its capture path",
                );
            }
        }
    }
    for capture in captures {
        let Some(&next_carrier) = ctx.op_results(capture).first() else {
            push_op_error(
                ctx,
                handler,
                errors,
                "lambda capture carrying a resume token must produce one callable value",
            );
            return;
        };
        validate_affine_lambda_carrier(ctx, handler, next_carrier, visited, errors);
    }
}

fn validate_token_path(
    ctx: &IrContext,
    handler: OpRef,
    token_value: ValueRef,
    errors: &mut Vec<ValidationError>,
) {
    let mut resumes = Vec::new();
    let mut captures = Vec::new();
    for use_ in ctx.uses(token_value) {
        if nearest_enclosing_handler(ctx, use_.user) != Some(handler) {
            push_op_error(
                ctx,
                handler,
                errors,
                format!(
                    "resume token crosses into a different handler at {}",
                    use_.user
                ),
            );
            continue;
        }
        if Resume::matches(ctx, use_.user) && use_.operand_index == 0 {
            resumes.push(use_.user);
        } else if Lambda::matches(ctx, use_.user) {
            captures.push(use_.user);
        } else {
            push_op_error(
                ctx,
                handler,
                errors,
                format!(
                    "resume token has forbidden use by {} at operand {}",
                    use_.user, use_.operand_index
                ),
            );
        }
    }
    if has_nonexclusive_pair(ctx, &resumes) {
        push_op_error(
            ctx,
            handler,
            errors,
            "resume token reaches more than one tribute_control.resume",
        );
    }
    for (left, right) in captures.iter().copied().tuple_combinations() {
        let left_region = ctx.op_region(left, 0);
        let right_region = ctx.op_region(right, 0);
        let comparable = left_region.is_some_and(|region| op_is_within_region(ctx, right, region))
            || right_region.is_some_and(|region| op_is_within_region(ctx, left, region));
        if !comparable && !ops_are_mutually_exclusive(ctx, left, right) {
            push_op_error(
                ctx,
                handler,
                errors,
                "resume token is copied into multiple capture paths",
            );
        }
    }
    for capture in &captures {
        let Some(region) = ctx.op_region(*capture, 0) else {
            continue;
        };
        if !resumes
            .iter()
            .any(|resume| op_is_within_region(ctx, *resume, region))
        {
            push_op_error(
                ctx,
                handler,
                errors,
                "resume token capture does not form a single path to resume",
            );
        }
    }
    let mut visited = HashSet::default();
    for capture in captures {
        if let Some(&closure_value) = ctx.op_results(capture).first() {
            validate_affine_lambda_carrier(ctx, handler, closure_value, &mut visited, errors);
        }
    }
}

/// A handler token may have one affine resume on each branch of an ordinary
/// structured conditional: exactly one branch executes at runtime.  Static
/// use counting alone would reject a source `case` whose exhaustive arms each
/// resume the same handler-owned token.
fn has_nonexclusive_pair(ctx: &IrContext, ops: &[OpRef]) -> bool {
    ops.iter()
        .copied()
        .tuple_combinations()
        .any(|(left, right)| !ops_are_mutually_exclusive(ctx, left, right))
}

fn ops_are_mutually_exclusive(ctx: &IrContext, left: OpRef, right: OpRef) -> bool {
    let mut region = parent_region(ctx, left);
    while let Some(current) = region {
        let owner = ctx.region(current).parent_op;
        if let Some(owner) = owner
            && let Some(interface) = DynRegionBranch::get(ctx, owner)
            && let Ok(successors) = interface.successors(ctx, RegionBranchPoint::Parent)
        {
            let containing_region = |candidate: OpRef| {
                successors.as_slice().iter().find_map(|successor| {
                    let RegionSuccessor::Region(region) = successor else {
                        return None;
                    };
                    op_is_within_region(ctx, candidate, *region).then_some(*region)
                })
            };
            if let (Some(left_region), Some(right_region)) =
                (containing_region(left), containing_region(right))
                && left_region != right_region
            {
                return true;
            }
        }
        region = owner.and_then(|owner| parent_region(ctx, owner));
    }
    false
}

fn validate_resume_ownership(ctx: &IrContext, body: RegionRef, errors: &mut Vec<ValidationError>) {
    walk_region_ops(ctx, body, &mut |op| {
        if !Handler::matches(ctx, op) {
            return;
        }
        let Some(region) = ctx.op_region(op, 0) else {
            return;
        };
        let Some(&block) = ctx.region(region).blocks.first() else {
            return;
        };
        let Some(&token) = ctx.block_args(block).last() else {
            return;
        };
        if ResumeToken::matches(ctx, ctx.value_ty(token)) {
            validate_token_path(ctx, op, token, errors);
        }
    });
}

fn validate_token_placements(ctx: &IrContext, body: RegionRef, errors: &mut Vec<ValidationError>) {
    walk_region_ops(ctx, body, &mut |op| {
        for ty in ctx.op_result_types(op) {
            if ResumeToken::matches(ctx, *ty) {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "resume_token must not be produced as an operation result",
                );
            }
        }
        for region in ctx.op_regions(op) {
            for block in ctx.region(region).blocks.iter().copied() {
                for (index, arg) in ctx.block_args(block).iter().copied().enumerate() {
                    if !ResumeToken::matches(ctx, ctx.value_ty(arg)) {
                        continue;
                    }
                    let allowed =
                        Handler::matches(ctx, op) && index + 1 == ctx.block_args(block).len();
                    if !allowed {
                        push_op_error(
                            ctx,
                            op,
                            errors,
                            "resume_token block arguments are allowed only as the final handler argument",
                        );
                    }
                }
            }
        }
    });
}

/// The ability instances the module tree declares.
fn declared_instances(ctx: &IrContext, module: Module) -> HashSet<SymbolPath> {
    let mut instances = HashSet::default();
    let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
        if crate::dialect::ability::Decl::matches(ctx, op)
            && let Some(path) = qualified_name(ctx, op)
        {
            instances.insert(path);
        }
        ControlFlow::Continue(WalkAction::Advance)
    });
    instances
}

/// Check one row against the binders in scope, `None` where no binder is.
fn check_row(
    ctx: &IrContext,
    owner: OpRef,
    row: &Row,
    binders: Option<usize>,
    instances: &HashSet<SymbolPath>,
    errors: &mut Vec<ValidationError>,
) {
    if let Some(tail) = row.tail
        && binders.is_none_or(|binders| usize::try_from(tail).is_ok_and(|tail| tail >= binders))
    {
        push_op_error(
            ctx,
            owner,
            errors,
            format!("row tail {tail} names no row binder in scope"),
        );
    }
    for instance in &row.instances {
        if !instances.contains(instance) {
            push_op_error(
                ctx,
                owner,
                errors,
                format!("row instance @{instance} is not an ability.decl"),
            );
        }
    }
}

fn check_type_rows(
    ctx: &IrContext,
    owner: OpRef,
    ty: TypeRef,
    binders: Option<usize>,
    instances: &HashSet<SymbolPath>,
    errors: &mut Vec<ValidationError>,
) {
    visit_rows(ctx, ty, &mut |row| match row {
        Ok(row) => check_row(ctx, owner, &row, binders, instances, errors),
        Err(message) => push_op_error(ctx, owner, errors, message),
    });
}

/// Every row tail inside a definition names one of its binders, and every
/// row instance is declared. Module-level types have no binders.
/// Every `local` binder is quantified by exactly one lambda, and every use
/// of such a lambda is an `instantiate` whose result is the lambda's type
/// with its row arguments substituted.
fn validate_local_quantification(
    ctx: &IrContext,
    module: Module,
    errors: &mut Vec<ValidationError>,
) {
    for &func in module.ops(ctx).iter() {
        if !Func::matches(ctx, func) {
            continue;
        }
        let (Ok(binders), Some(body)) = (row_binders(ctx, func), ctx.op_region(func, 0)) else {
            continue;
        };
        let mut owners = HashSet::default();
        walk_region_ops(ctx, body, &mut |op| {
            if Lambda::matches(ctx, op) {
                let quantified = match quantified_rows(ctx, op) {
                    Ok(quantified) => quantified,
                    Err(message) => return push_op_error(ctx, op, errors, message),
                };
                for binder in &quantified {
                    let kind = usize::try_from(*binder)
                        .ok()
                        .and_then(|index| binders.get(index));
                    if kind != Some(&RowBinder::Local) {
                        push_op_error(
                            ctx,
                            op,
                            errors,
                            format!("lambda quantifies binder {binder}, which is not local"),
                        );
                    } else if !owners.insert(*binder) {
                        push_op_error(
                            ctx,
                            op,
                            errors,
                            format!("local binder {binder} is quantified by two lambdas"),
                        );
                    }
                }
                // A capture only hands the lambda to an inner body, whose
                // own uses of it are uses too.
                let result = ctx.op_results(op).first().copied();
                let direct_use = !quantified.is_empty()
                    && result.is_some_and(|result| {
                        ctx.uses(result).iter().any(|used| {
                            !Instantiate::matches(ctx, used.user)
                                && !Lambda::matches(ctx, used.user)
                        })
                    });
                if direct_use {
                    push_op_error(
                        ctx,
                        op,
                        errors,
                        "a lambda that quantifies rows is used only through instantiate",
                    );
                }
            } else if Instantiate::matches(ctx, op) {
                validate_instantiate(ctx, op, errors);
            }
        });
        for (binder, kind) in (0u32..).zip(&binders) {
            if *kind == RowBinder::Local && !owners.contains(&binder) {
                push_op_error(
                    ctx,
                    func,
                    errors,
                    format!("local binder {binder} is quantified by no lambda"),
                );
            }
        }
    }
}

fn validate_instantiate(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    let lambda = ctx
        .op_operands(op)
        .first()
        .and_then(|&operand| match ctx.value_def(operand) {
            ValueDef::OpResult(producer, _) if Lambda::matches(ctx, producer) => Some(producer),
            _ => None,
        });
    let quantified = lambda
        .and_then(|lambda| quantified_rows(ctx, lambda).ok())
        .filter(|quantified| !quantified.is_empty());
    let (Some(lambda), Some(quantified)) = (lambda, quantified) else {
        return push_op_error(
            ctx,
            op,
            errors,
            "instantiate takes a lambda that quantifies rows",
        );
    };
    let Ok(args) = row_args(ctx, op) else {
        return;
    };
    if args.len() != quantified.len() {
        return push_op_error(
            ctx,
            op,
            errors,
            format!(
                "instantiate passes {} row arguments for {} quantified binders",
                args.len(),
                quantified.len()
            ),
        );
    }
    let expected = ctx
        .op_result_types(lambda)
        .first()
        .and_then(|&ty| find_substituted_rows(ctx, ty, RowArgs::Binders(&quantified, &args)));
    if expected.is_none() || ctx.op_result_types(op).first().copied() != expected {
        push_op_error(
            ctx,
            op,
            errors,
            "instantiate result is not the lambda's type with its row arguments substituted",
        );
    }
}

fn validate_row_scopes(
    ctx: &IrContext,
    module: Module,
    instances: &HashSet<SymbolPath>,
    errors: &mut Vec<ValidationError>,
) {
    let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
        if crate::dialect::ability::Operation::matches(ctx, op) {
            let mut types = Vec::new();
            for attr in ctx.op(op).attributes.values() {
                attr.visit_types(&mut |ty| types.push(ty));
            }
            for ty in types {
                check_type_rows(ctx, op, ty, None, instances, errors);
            }
            return ControlFlow::Continue(WalkAction::Advance);
        }
        if !Func::matches(ctx, op) {
            return ControlFlow::Continue(WalkAction::Advance);
        }
        let Ok(binders) = row_binders(ctx, op) else {
            return ControlFlow::Continue(WalkAction::Skip);
        };
        let scope = Some(binders.len());
        let mut types = HashSet::default();
        if let Some(ty) = ctx.op_attrs(op).get_type("type") {
            types.insert(ty);
        }
        let mut rows = Vec::new();
        if let Some(body) = ctx.op_region(op, 0) {
            walk_region_ops(ctx, body, &mut |inner| {
                types.extend(ctx.op_result_types(inner).iter().copied());
                for attr in ctx.op(inner).attributes.values() {
                    attr.visit_types(&mut |ty| {
                        types.insert(ty);
                    });
                }
                for region in ctx.op_regions(inner) {
                    for &block in &ctx.region(region).blocks {
                        types.extend(block_arg_types(ctx, block));
                    }
                }
                if let Ok(args) = row_args(ctx, inner) {
                    rows.extend(args.into_iter().map(|row| (inner, row)));
                }
            });
            for &block in &ctx.region(body).blocks {
                types.extend(block_arg_types(ctx, block));
            }
        }
        for ty in types {
            check_type_rows(ctx, op, ty, scope, instances, errors);
        }
        for (inner, row) in rows {
            check_row(ctx, inner, &row, scope, instances, errors);
        }
        ControlFlow::Continue(WalkAction::Skip)
    });
}

/// The row tail of the callable whose body holds `op`, and the function
/// that declares the binders it names.
fn enclosing_row_tail(ctx: &IrContext, op: OpRef) -> Option<(Option<u32>, OpRef)> {
    let mut current = op;
    let mut tail = None;
    loop {
        let owner = parent_op(ctx, current)?;
        if Lambda::matches(ctx, owner) && tail.is_none() {
            let ty = ctx.op_result_types(owner).first().copied()?;
            tail = Some(FuncSig::from_type_ref(ctx, ty)?.row(ctx).tail);
        }
        if Func::matches(ctx, owner) {
            let tail = match tail {
                Some(tail) => tail,
                None => {
                    let ty = ctx.op_attrs(owner).get_type("type")?;
                    FuncSig::from_type_ref(ctx, ty)?.row(ctx).tail
                }
            };
            return Some((tail, owner));
        }
        current = owner;
    }
}

/// The sources of the union `func` declares for `tail`.
fn union_sources(ctx: &IrContext, func: OpRef, tail: Option<u32>) -> Option<Vec<u32>> {
    let tail = tail?;
    row_unions(ctx, func)
        .ok()?
        .into_iter()
        .find(|union| union.result == tail)
        .map(|union| union.sources)
}

fn check_selects(steps: &[EvidenceStep], sources: Option<&[u32]>) -> Result<(), String> {
    for step in steps {
        match step {
            EvidenceStep::Select(index) => {
                let in_range = sources.is_some_and(|sources| {
                    usize::try_from(*index).is_ok_and(|index| index < sources.len())
                });
                if !in_range {
                    return Err(format!(
                        "evidence_plan select {index} names no source of the union for the \
                         callable's row tail"
                    ));
                }
            }
            EvidenceStep::Tails(plans) => {
                for plan in plans {
                    check_selects(plan, sources)?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// `select` and `tails` count the sources of row tail unions.
fn validate_evidence_tails(
    ctx: &IrContext,
    body: RegionRef,
    symbols: &SymbolTable,
    errors: &mut Vec<ValidationError>,
) {
    walk_region_ops(ctx, body, &mut |op| {
        let Some(Attribute::List(items)) = ctx.op_attrs(op).get(EVIDENCE_PLAN_ATTR) else {
            return;
        };
        let Ok(steps) = items
            .iter()
            .map(|step| EvidenceStep::from_attribute(ctx, step))
            .collect::<Result<Vec<_>, _>>()
        else {
            return;
        };
        let Some((tail, func)) = enclosing_row_tail(ctx, op) else {
            return;
        };
        let sources = union_sources(ctx, func, tail);
        if let Err(message) = check_selects(&steps, sources.as_deref()) {
            push_op_error(ctx, op, errors, message);
        }
        let Some(EvidenceStep::Tails(plans)) = steps.last() else {
            return;
        };
        let callee = ctx
            .op_attrs(op)
            .get_symbol_ref("callee")
            .and_then(|symbol| symbols.definitions(ctx, op, symbol).first().copied())
            .filter(|&callee| Func::matches(ctx, callee));
        let Some(callee) = callee else {
            return;
        };
        let callee_tail = ctx
            .op_attrs(callee)
            .get_type("type")
            .and_then(|ty| FuncSig::from_type_ref(ctx, ty))
            .and_then(|signature| signature.row(ctx).tail);
        let expected = union_sources(ctx, callee, callee_tail).map(|sources| sources.len());
        if expected != Some(plans.len()) {
            push_op_error(
                ctx,
                op,
                errors,
                "evidence_plan tails must hold one plan for each source of the callee's row \
                 tail union",
            );
        }
    });
}

/// Validate graph-wide `tribute_control` invariants.
pub fn validate_whole_ir(
    ctx: &IrContext,
    module: Module,
    compiler_intrinsics: &[CompilerIntrinsicDeclaration],
) -> ValidationResult {
    whole_ir(
        ctx,
        module,
        &SymbolTable::collect(ctx, module),
        compiler_intrinsics,
    )
}

fn whole_ir(
    ctx: &IrContext,
    module: Module,
    symbols: &SymbolTable,
    compiler_intrinsics: &[CompilerIntrinsicDeclaration],
) -> ValidationResult {
    let mut errors = Vec::new();
    let Some(body) = module.body(ctx) else {
        return ValidationResult { errors };
    };
    let declarations = &declared_operations(ctx, module, &mut errors);
    validate_module_symbols(ctx, module, symbols, &mut errors);
    validate_row_scopes(ctx, module, &declared_instances(ctx, module), &mut errors);
    validate_local_quantification(ctx, module, &mut errors);
    validate_evidence_tails(ctx, body, symbols, &mut errors);
    validate_lambda_captures(ctx, body, &mut errors);
    let declarations = declaration_map(ctx, declarations, &mut errors);
    if conventions_assigned(ctx, module).is_none() {
        push_type_error(
            &mut errors,
            "only some source signatures have a calling convention; class specialization assigns one to all of them",
        );
    }
    let reachable_types = collect_reachable_ir_types(ctx, module.op());
    let nominal_layouts = canonical_nominal_layouts(ctx, &reachable_types, &mut errors);
    validate_managed_reference_boundaries(
        ctx,
        body,
        &reachable_types,
        &nominal_layouts,
        &mut errors,
    );
    validate_callable_conversion_casts(ctx, body, &mut errors);
    validate_callable_origins(
        ctx,
        body,
        symbols,
        compiler_intrinsics,
        &declarations,
        &nominal_layouts,
        &mut errors,
    );
    validate_return_contracts(ctx, body, &mut errors);
    validate_declaration_uses(ctx, body, &declarations, &mut errors);
    validate_resume_ownership(ctx, body, &mut errors);
    validate_token_placements(ctx, body, &mut errors);
    ValidationResult { errors }
}

/// Run the complete local and whole-IR `tribute_control` validation.
///
/// Ability operations are checked against the module's `ability.decl`
/// definitions. The [`SymbolTable`] is queried through `analyses`.
pub fn validate(
    ctx: &IrContext,
    module: Module,
    compiler_intrinsics: &[CompilerIntrinsicDeclaration],
    analyses: &mut AnalysisCache,
) -> ValidationResult {
    let symbols = analyses.require::<SymbolTable>(ctx, module.op());
    let mut local = validate_local(ctx, module);
    local
        .errors
        .extend(whole_ir(ctx, module, &symbols, compiler_intrinsics).errors);
    local
}

#[cfg(test)]
mod tests {
    use super::callable_origin::indirect_callees;
    use super::*;
    use trunk_ir::Span;
    use trunk_ir::context::{BlockArgData, BlockData, RegionData};
    use trunk_ir::dialect::core;
    use trunk_ir::op_flag::OpFlag;
    use trunk_ir::parser::parse_module;
    use trunk_ir::printer::print_module;
    use trunk_ir::types::AttributeMap;

    fn location(ctx: &mut IrContext) -> Location {
        let path = ctx.intern_path("tribute-control-test.trb");
        Location::new(path, Span::new(7, 19))
    }

    fn ability_type(_: &mut IrContext, name: &str) -> SymbolPath {
        SymbolPath::new([name])
    }

    fn block(ctx: &mut IrContext, loc: Location, args: &[TypeRef]) -> BlockRef {
        ctx.create_block(BlockData {
            location: loc,
            args: args
                .iter()
                .copied()
                .map(|ty| BlockArgData {
                    ty,
                    attrs: AttributeMap::new(),
                })
                .collect(),
            ops: Default::default(),
            parent_region: None,
        })
    }

    fn region(ctx: &mut IrContext, loc: Location, block: BlockRef) -> RegionRef {
        ctx.create_region(RegionData {
            location: loc,
            blocks: vec![block].into(),
            parent_op: None,
        })
    }

    fn module(ctx: &mut IrContext, loc: Location, ops: &[OpRef]) -> Module {
        let body_block = block(ctx, loc, &[]);
        for op in ops {
            ctx.push_op(body_block, *op);
        }
        let body = region(ctx, loc, body_block);
        let module = core::Module::operands()
            .sym_name("test")
            .regions(body)
            .build(ctx, loc);
        Module::new(ctx, module.op_ref()).expect("core.module")
    }

    fn identity_func(
        ctx: &mut IrContext,
        loc: Location,
        symbol: &str,
        ty: TypeRef,
        value_ty: TypeRef,
    ) -> Func {
        let entry = block(ctx, loc, &[value_ty]);
        let value = ctx.block_arg(entry, 0);
        let ret = Return::operands(value).build(ctx, loc);
        ctx.push_op(entry, ret.op_ref());
        let body = region(ctx, loc, entry);
        Func::operands()
            .sym_name(ctx.intern_str(symbol))
            .r#type(ty)
            .regions(body)
            .build(ctx, loc)
    }

    struct ValidFixture {
        ctx: IrContext,
        module: Module,
        handle: OpRef,
        handler: OpRef,
    }

    fn builder_fixture() -> ValidFixture {
        let mut ctx = IrContext::new();
        let loc = location(&mut ctx);
        let i32_ty = core::I32::type_ref(&mut ctx);
        let ability = ability_type(&mut ctx, "State");
        let direct = FuncSig::params([i32_ty], i32_ty)
            .convention(CallingConvention::Direct)
            .intern(&mut ctx)
            .as_type_ref();

        let id = identity_func(&mut ctx, loc, "id", direct, i32_ty);

        let entry = block(&mut ctx, loc, &[i32_ty]);
        let x = ctx.block_arg(entry, 0);

        let function_ref = FuncRef::operands()
            .func_ref(SymbolPath::new(["id"]))
            .results(direct)
            .build(&mut ctx, loc);
        ctx.push_op(entry, function_ref.op_ref());
        let function_ref_value = function_ref.result(&ctx);
        let direct_call = Call::operands([x])
            .callee(SymbolPath::new(["id"]))
            .results(i32_ty)
            .build(&mut ctx, loc);
        ctx.push_op(entry, direct_call.op_ref());
        let indirect_call = CallIndirect::operands(function_ref_value, [x]).build(&mut ctx, loc);
        ctx.push_op(entry, indirect_call.op_ref());

        let lambda_ty = FuncSig::params([], i32_ty)
            .convention(CallingConvention::Direct)
            .intern(&mut ctx)
            .as_type_ref();
        let lambda_block = block(&mut ctx, loc, &[]);
        let lambda_return = Return::operands(x).build(&mut ctx, loc);
        ctx.push_op(lambda_block, lambda_return.op_ref());
        let lambda_body = region(&mut ctx, loc, lambda_block);
        let lambda = Lambda::operands([x])
            .results(lambda_ty)
            .regions(lambda_body)
            .build(&mut ctx, loc);
        ctx.push_op(entry, lambda.op_ref());

        let body_block = block(&mut ctx, loc, &[]);
        let perform = Perform::operands([x])
            .ability(ability.clone())
            .op_name("get")
            .operation_kind("op")
            .results(i32_ty)
            .build(&mut ctx, loc);
        ctx.push_op(body_block, perform.op_ref());
        let perform_result = perform.result(&ctx);
        let body_yield = Yield::operands(perform_result).build(&mut ctx, loc);
        ctx.push_op(body_block, body_yield.op_ref());
        let handle_body = region(&mut ctx, loc, body_block);

        let completion_block = block(&mut ctx, loc, &[i32_ty]);
        let completed = ctx.block_arg(completion_block, 0);
        let completion_yield = Yield::operands(completed).build(&mut ctx, loc);
        ctx.push_op(completion_block, completion_yield.op_ref());
        let completion = region(&mut ctx, loc, completion_block);

        let token_ty = ResumeToken::params(i32_ty, i32_ty)
            .intern(&mut ctx)
            .as_type_ref();
        let handler_block = block(&mut ctx, loc, &[i32_ty, token_ty]);
        let operation_arg = ctx.block_arg(handler_block, 0);
        let token = ctx.block_arg(handler_block, 1);
        let resume_op = Resume::operands(token, operation_arg).build(&mut ctx, loc);
        ctx.push_op(handler_block, resume_op.op_ref());
        let resumed_value = resume_op.result(&ctx);
        let handler_yield = Yield::operands(resumed_value).build(&mut ctx, loc);
        ctx.push_op(handler_block, handler_yield.op_ref());
        let handler_body = region(&mut ctx, loc, handler_block);
        let handler = Handler::operands()
            .ability(ability.clone())
            .op_name("get")
            .kind("op")
            .operation_result_type(i32_ty)
            .regions(handler_body)
            .build(&mut ctx, loc);

        let handlers_block = block(&mut ctx, loc, &[]);
        ctx.push_op(handlers_block, handler.op_ref());
        let handlers = region(&mut ctx, loc, handlers_block);
        let handle = Handle::operands()
            .results(i32_ty)
            .regions(handle_body, completion, handlers)
            .build(&mut ctx, loc);
        ctx.push_op(entry, handle.op_ref());
        let handle_result = handle.result(&ctx);
        let ret = Return::operands(handle_result).build(&mut ctx, loc);
        ctx.push_op(entry, ret.op_ref());
        let control_body = region(&mut ctx, loc, entry);
        let control = Func::operands()
            .sym_name("control")
            .r#type(direct)
            .regions(control_body)
            .build(&mut ctx, loc);

        let module = module(&mut ctx, loc, &[id.op_ref(), control.op_ref()]);
        let declarations = vec![OperationDeclaration::new(
            ability,
            ctx.intern_str("get"),
            ctx.intern_str("op"),
            [i32_ty],
            i32_ty,
        )];

        declare_operations(&mut ctx, module, &declarations);

        ValidFixture {
            ctx,
            module,
            handle: handle.op_ref(),
            handler: handler.op_ref(),
        }
    }

    const VALID_CONTROL_MODULE: &str = r#"core.module @test {
  !callable = tribute_control.func_sig<(core.i32) -> core.i32, {tribute.calling_convention = 0}>

  tribute_control.func @id(%value: core.i32) -> core.i32 convention(direct) {
    tribute_control.return %value
  }
  tribute_control.func @control(%value: core.i32) -> core.i32 convention(direct) {
    %function = tribute_control.func_ref {func_ref = @id} : !callable
    %direct = tribute_control.call %value {callee = @id} : core.i32
    %indirect = tribute_control.call_indirect %function, %value : core.i32
    %closure = tribute_control.lambda() -> core.i32 convention(direct) captures [%value] {
      tribute_control.return %value
    }
    %handled = tribute_control.handle : core.i32 {
      %performed = tribute_control.perform %value {ability = @State, op_name = "get", operation_kind = "op"} : core.i32
      tribute_control.yield %performed
    } {
      ^completion(%completed: core.i32):
        tribute_control.yield %completed
    } {
      tribute_control.handler {ability = @State, kind = "op", op_name = "get", operation_result_type = core.i32} {
        ^handler(%argument: core.i32, %token: tribute_control.resume_token<core.i32, core.i32>):
          %resumed = tribute_control.resume %token, %argument : core.i32
          tribute_control.yield %resumed
      }
    }
    tribute_control.return %handled
  }
}"#;

    fn parse_fixture(input: &str) -> (IrContext, Module) {
        let mut ctx = IrContext::new();
        let root = parse_module(&mut ctx, input)
            .unwrap_or_else(|error| panic!("failed to parse fixture:\n{error}\n\n{input}"));
        let module = Module::new(&ctx, root).expect("core.module fixture");
        (ctx, module)
    }

    fn control_ops(ctx: &IrContext, module: Module, name: trunk_ir::OpName) -> Vec<OpRef> {
        let mut matches = Vec::new();
        let body = module.body(ctx).expect("module body");
        walk_region_ops(ctx, body, &mut |op| {
            if ctx.op(op).name == name {
                matches.push(op);
            }
        });
        matches
    }

    fn control_op(ctx: &IrContext, module: Module, name: trunk_ir::OpName) -> OpRef {
        let matches = control_ops(ctx, module, name);
        assert_eq!(
            matches.len(),
            1,
            "expected exactly one {}",
            name.display(ctx)
        );
        matches[0]
    }

    fn valid_fixture() -> ValidFixture {
        let (mut ctx, module) = parse_fixture(VALID_CONTROL_MODULE);
        let handle = control_op(&ctx, module, Handle::NAME);
        let handler = control_op(&ctx, module, Handler::NAME);
        let handler_body = ctx.op_region(handler, 0).unwrap();
        let handler_block = ctx.region(handler_body).blocks[0];
        let ability = ctx
            .op_attrs(handler)
            .get_symbol_ref("ability")
            .cloned()
            .expect("handler ability");
        let op_name = ctx
            .op_attrs(handler)
            .get_string_ref("op_name")
            .expect("handler operation name");
        let kind = ctx
            .op_attrs(handler)
            .get_string_ref("kind")
            .expect("handler kind");
        let result_type = ctx
            .op_attrs(handler)
            .get_type("operation_result_type")
            .expect("handler operation result");
        let parameter_types = block_arg_types(&ctx, handler_block);
        let declarations = vec![OperationDeclaration::new(
            ability,
            op_name,
            kind,
            parameter_types[..parameter_types.len() - 1].iter().copied(),
            result_type,
        )];

        declare_operations(&mut ctx, module, &declarations);

        ValidFixture {
            ctx,
            module,
            handle,
            handler,
        }
    }

    const DECLARED_PERFORM: &str = r#"core.module @test {
  tribute_control.func @run() -> core.i32 convention(cps) {
    %value = tribute_control.perform {ability = @State, op_name = "get", operation_kind = "op"} : core.i32
    tribute_control.return %value
  }
  ability.decl {sym_name = "State"} {
    ability.operation {kind = "op", op_name = "get", param_types = [], result_type = core.i32}
  }
}"#;

    #[test]
    fn ability_instances_are_symbol_references() {
        let (ctx, module) = parse_fixture(&evidence_plan_module(
            "[{mask = @State}, {dup = @Log}]",
            "[{mask = @State}]",
        ));
        let referenced = |name: trunk_ir::OpName| {
            let mut referenced = Vec::new();
            ctx.op(control_op(&ctx, module, name))
                .attributes
                .visit_symbol_refs(&mut |path| referenced.push(path.to_string()));
            referenced
        };
        // A call's callee, then the instances its evidence plan names.
        assert_eq!(referenced(Call::NAME), ["id", "State", "Log"]);
        assert_eq!(referenced(Perform::NAME), ["State"]);
        assert_eq!(referenced(Handler::NAME), ["State"]);
    }

    #[test]
    fn a_textual_module_declares_the_operations_it_performs() {
        let (ctx, module) = parse_fixture(DECLARED_PERFORM);
        let result = validate(&ctx, module, &[], &mut Default::default());
        assert!(result.is_ok(), "{result}");

        let printed = assert_round_trip(&ctx, module);
        assert!(printed.contains("ability.decl"), "{printed}");
    }

    #[test]
    fn an_operation_is_checked_against_the_declaration_of_its_instance() {
        let undeclared = DECLARED_PERFORM.replace(
            r#"op_name = "get", param_types"#,
            r#"op_name = "set", param_types"#,
        );
        let (ctx, module) = parse_fixture(&undeclared);
        let result = validate(&ctx, module, &[], &mut Default::default());
        assert!(
            messages(&result).contains("no resolved operation declaration"),
            "{result}"
        );

        let other_kind = DECLARED_PERFORM.replace(r#"{kind = "op""#, r#"{kind = "fn""#);
        let (ctx, module) = parse_fixture(&other_kind);
        let result = validate(&ctx, module, &[], &mut Default::default());
        assert!(
            !result.is_ok(),
            "a perform must agree with the declared kind"
        );

        let twice = DECLARED_PERFORM.replacen(
            "  ability.decl",
            r#"  ability.decl {sym_name = "State"} {
    ability.operation {kind = "op", op_name = "get", param_types = [], result_type = core.i32}
  }
  ability.decl"#,
            1,
        );
        let (ctx, module) = parse_fixture(&twice);
        let result = validate(&ctx, module, &[], &mut Default::default());
        assert!(
            messages(&result).contains("duplicate operation declaration"),
            "{result}"
        );
    }

    fn assert_round_trip(ctx: &IrContext, module: Module) -> String {
        let printed = print_module(ctx, module.op());
        let mut reparsed_ctx = IrContext::new();
        let reparsed = parse_module(&mut reparsed_ctx, &printed)
            .unwrap_or_else(|error| panic!("failed to reparse:\n{error}\n\n{printed}"));
        let reprinted = print_module(&reparsed_ctx, reparsed);
        assert_eq!(printed, reprinted);
        printed
    }

    fn messages(result: &ValidationResult) -> String {
        result
            .errors
            .iter()
            .map(|error| error.message.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn append_module_op(ctx: &mut IrContext, module: Module, op: OpRef) {
        let body = module.body(ctx).expect("module body");
        let body_block = ctx.region(body).blocks[0];
        ctx.push_op(body_block, op);
    }

    fn assert_diagnostics(result: &ValidationResult, expected: &[&str]) {
        let messages = messages(result);
        for expected in expected {
            assert!(
                messages.contains(expected),
                "missing diagnostic `{expected}`:\n{result}"
            );
        }
    }

    fn assert_op_diagnostics(result: &ValidationResult, op: OpRef, expected: &[&str]) {
        let messages = result
            .errors
            .iter()
            .filter(|error| error.op == Some(op))
            .map(|error| error.message.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        for expected in expected {
            assert!(
                messages.contains(expected),
                "operation {op} is missing diagnostic `{expected}`:\n{result}"
            );
        }
    }

    #[test]
    fn callable_codes_and_resume_token_accessors_are_exact() {
        let mut ctx = IrContext::new();
        let i32_ty = core::I32::type_ref(&mut ctx);

        for (convention, code) in [
            (CallingConvention::Direct, 0),
            (CallingConvention::EvidenceDirect, 1),
            (CallingConvention::Cps, 2),
        ] {
            let ty = FuncSig::params([i32_ty], i32_ty)
                .convention(convention)
                .intern(&mut ctx);
            assert_eq!(ty.result(&ctx), i32_ty);
            assert_eq!(ty.inputs(&ctx), &[i32_ty]);
            assert_eq!(
                ctx.type_attrs(ty.as_type_ref())
                    .get_i128(CALLING_CONVENTION_ATTR),
                Some(code)
            );
            assert_eq!(
                func_sig_convention(&ctx, ty.as_type_ref()),
                Some(convention)
            );
        }

        let token = ResumeToken::params(i32_ty, i32_ty).intern(&mut ctx);
        assert_eq!(token.input(&ctx), i32_ty);
        assert_eq!(token.answer(&ctx), i32_ty);
    }

    #[test]
    fn callable_wrapper_rejects_missing_result_component() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  !malformed = tribute_control.func_sig<() -> (), {tribute.calling_convention = 0}>
}"#,
        );
        let malformed = ctx
            .types()
            .iter()
            .find_map(|(ty, _)| FuncSig::matches(&ctx, ty).then_some(ty))
            .expect("malformed callable type");
        assert!(FuncSig::from_type_ref(&ctx, malformed).is_none());

        let result = validate_local(&ctx, module);
        assert!(messages(&result).contains("ResultCount(0)"));
    }

    #[test]
    fn func_sig_rejects_raw_delimiter_failures_before_conversion() {
        let (mut ctx, module) = parse_fixture("core.module @test {}");
        let i32_ty = core::I32::type_ref(&mut ctx);
        let malformed = [
            TypeDataBuilder::new("tribute_control.func_sig")
                .param(i32_ty)
                .attr(CALLING_CONVENTION_ATTR, Attribute::Int(0))
                .build(&mut ctx),
            TypeDataBuilder::new("tribute_control.func_sig")
                .param(i32_ty)
                .attr(
                    trunk_ir::dialect::func::NUM_INPUTS_ATTR,
                    ctx.string_attr("bad"),
                )
                .attr(trunk_ir::dialect::func::NUM_RESULTS_ATTR, Attribute::Int(1))
                .attr(CALLING_CONVENTION_ATTR, Attribute::Int(0))
                .build(&mut ctx),
            TypeDataBuilder::new("tribute_control.func_sig")
                .param(i32_ty)
                .attr(trunk_ir::dialect::func::NUM_INPUTS_ATTR, Attribute::Int(2))
                .attr(trunk_ir::dialect::func::NUM_RESULTS_ATTR, Attribute::Int(1))
                .attr(CALLING_CONVENTION_ATTR, Attribute::Int(0))
                .build(&mut ctx),
            TypeDataBuilder::new("tribute_control.func_sig")
                .param(i32_ty)
                .attr(
                    trunk_ir::dialect::func::NUM_INPUTS_ATTR,
                    Attribute::Int(i128::from(u32::MAX) + 1),
                )
                .attr(trunk_ir::dialect::func::NUM_RESULTS_ATTR, Attribute::Int(1))
                .attr(CALLING_CONVENTION_ATTR, Attribute::Int(0))
                .build(&mut ctx),
            TypeDataBuilder::new("tribute_control.func_sig")
                .param(i32_ty)
                .attr(trunk_ir::dialect::func::NUM_INPUTS_ATTR, Attribute::Int(0))
                .attr(trunk_ir::dialect::func::NUM_RESULTS_ATTR, Attribute::Int(2))
                .attr(CALLING_CONVENTION_ATTR, Attribute::Int(0))
                .build(&mut ctx),
        ];
        for data in malformed {
            let ty = ctx.intern_type(data);
            assert!(FuncSig::matches(&ctx, ty));
            assert!(FuncSig::from_type_ref(&ctx, ty).is_none());
        }
        let errors = validate_local(&ctx, module);
        assert!(messages(&errors).contains("malformed tribute_control.func_sig"));
    }

    #[test]
    fn all_operations_construct_with_typed_accessors_and_preserve_locations() {
        // This is intentionally builder-based: it verifies every public
        // constructor/accessor and preservation of caller-provided locations.
        let fixture = builder_fixture();
        let ctx = &fixture.ctx;
        let mut names = HashSet::default();
        let body = fixture.module.body(ctx).expect("module body");
        walk_region_ops(ctx, body, &mut |op| {
            let data = ctx.op(op);
            if data.name.dialect(ctx) == DIALECT_NAME {
                names.insert(data.name);
                assert_eq!(data.location.span, Span::new(7, 19));
            }
        });
        for expected in [
            Func::NAME,
            Lambda::NAME,
            FuncRef::NAME,
            Call::NAME,
            CallIndirect::NAME,
            Return::NAME,
            Perform::NAME,
            Handle::NAME,
            Handler::NAME,
            Resume::NAME,
            Yield::NAME,
        ] {
            assert!(names.contains(&expected), "{expected:?}");
        }

        let handler = Handler::from_op(ctx, fixture.handler).expect("typed handler accessor");
        assert_eq!(handler.kind(ctx), "op");
        let handle = Handle::from_op(ctx, fixture.handle).expect("typed handle accessor");
        assert_eq!(handle.result_ty(ctx), ctx.value_ty(handle.result(ctx)));
    }

    #[test]
    fn custom_assembly_round_trips_declarations_definitions_and_lambdas() {
        let input = r#"core.module @test {
  !inner = tribute_control.func_sig<(core.i32) -> core.i32, {tribute.calling_convention = 0}>
  !shared = tribute_control.func_sig<(core.i32, core.i32) -> core.i32, {metadata = [[!inner, @signature]], tribute.calling_convention = 0}>
  !lambda = tribute_control.func_sig<(core.i32, core.i32) -> core.i32, {metadata = [[!inner, @lambda]], tribute.calling_convention = 0}>
  !distinct = tribute_control.func_sig<(core.i32, core.i32) -> core.i32, {metadata = [[!inner, @distinct]], tribute.calling_convention = 0}>
  tribute_control.func {metadata = @declaration, sym_name = "decl", type = !shared, visibility = @private}
  tribute_control.func {metadata = @definition, sym_name = "definition", type = !shared, visibility = @private} {
    ^bb0(%left: core.i32, %right: core.i32):
      tribute_control.return %left
  }
  tribute_control.func {sym_name = "different", type = !distinct}
  tribute_control.func @identity(%value: core.i32) -> core.i32 convention(evidence_direct) {
    tribute_control.return %value
  }
  tribute_control.func @outer(%first: core.i32, %second: core.i32) -> core.i32 convention(direct) {
    %captured = tribute_control.lambda %first, %second {debug_name = "apply", inline_hint = true, metadata = @operation} : !lambda {
      ^bb0(%left: core.i32, %right: core.i32):
        tribute_control.return %first
    }
    %empty = tribute_control.lambda() -> core.i32 convention(cps) captures [] {
      %constant = arith.const {value = 1} : core.i32
      tribute_control.return %constant
    }
    tribute_control.return %first
  }
}"#;
        let (ctx, module) = parse_fixture(input);
        let printed = assert_round_trip(&ctx, module);
        assert!(
            printed.contains("!shared = tribute_control.func_sig"),
            "{printed}"
        );
        assert!(
            printed.contains("metadata = [[!inner, @signature]]"),
            "{printed}"
        );
        assert!(printed.contains("type = !shared"), "{printed}");
        assert!(printed.contains("sym_name = \"decl\""), "{printed}");
        assert!(printed.contains("sym_name = \"definition\""), "{printed}");
        assert!(printed.contains("metadata = @declaration"), "{printed}");
        assert!(printed.contains("metadata = @definition"), "{printed}");
        assert!(printed.contains(" : !lambda"), "{printed}");
        assert!(printed.contains("metadata = @operation"), "{printed}");
        assert!(
            printed.contains("%2 = tribute_control.lambda %0, %1"),
            "generic fallback must not consume an extra lambda result name: {printed}"
        );
        assert!(printed.contains("convention(evidence_direct)"));
        assert!(printed.contains("convention(cps) captures []"));
        assert!(printed.contains("debug_name = \"apply\""));
        assert!(printed.contains("inline_hint = true"));
        assert!(
            printed.contains("^bb"),
            "generic regions retain entry args: {printed}"
        );

        let funcs: Vec<_> = module
            .ops(&ctx)
            .iter()
            .copied()
            .filter(|op| Func::matches(&ctx, *op))
            .collect();
        let declaration = funcs
            .iter()
            .copied()
            .find(|op| ctx.op_attrs(*op).get_str("sym_name") == Some("decl"))
            .unwrap();
        let definition = funcs
            .iter()
            .copied()
            .find(|op| ctx.op_attrs(*op).get_str("sym_name") == Some("definition"))
            .unwrap();
        let shared = ctx.op_attrs(declaration).get_type("type").unwrap();
        assert_eq!(shared, ctx.op_attrs(definition).get_type("type").unwrap());
        let distinct = funcs
            .iter()
            .copied()
            .find(|op| ctx.op_attrs(*op).get_str("sym_name") == Some("different"))
            .and_then(|op| ctx.op_attrs(op).get_type("type"))
            .unwrap();
        assert_ne!(shared, distinct);
        assert!(matches!(
            ctx.type_attrs(shared).get("metadata"),
            Some(Attribute::List(_))
        ));
        assert_eq!(
            ctx.op_attrs(declaration).get_symbol_ref("metadata"),
            Some(&SymbolPath::new(["declaration"]))
        );

        let inline = r#"core.module @test {
  tribute_control.func {sym_name = "inline", type = tribute_control.func_sig<(core.i32) -> core.i32, {metadata = @inline, tribute.calling_convention = 0}>}
}"#;
        let (inline_ctx, inline_module) = parse_fixture(inline);
        let inline_printed = assert_round_trip(&inline_ctx, inline_module);
        assert!(
            inline_printed.contains(
                "type = tribute_control.func_sig<(core.i32) -> core.i32, {metadata = @inline, tribute.calling_convention = 0}>"
            ),
            "single-use attributed signature must stay inline: {inline_printed}"
        );
    }

    #[test]
    fn malformed_func_sig_storage_uses_generic_assembly_without_loss() {
        let input = r#"core.module @test {
  tribute_control.func {sym_name = "missing"}
  tribute_control.func {sym_name = "broken", type = tribute_control.func_sig<core.i32, {num_inputs = 2, num_results = 1, tribute.calling_convention = 0}>}
  %lambda = tribute_control.lambda : tribute_control.func_sig<core.i32, {num_inputs = 2, num_results = 1, tribute.calling_convention = 0}>
}"#;
        let (ctx, module) = parse_fixture(input);
        let printed = assert_round_trip(&ctx, module);

        assert!(
            printed.contains("tribute_control.func {sym_name = \"missing\"}"),
            "missing source signatures must use generic assembly: {printed}"
        );
        assert!(
            printed.contains(
                "!t0 = tribute_control.func_sig<core.i32, {num_inputs = 2, num_results = 1, tribute.calling_convention = 0}>"
            ),
            "malformed signature type/count storage must remain intact: {printed}"
        );
        assert!(
            printed.contains("tribute_control.func {sym_name = \"broken\", type = !t0}"),
            "malformed function signature must remain attached to the function: {printed}"
        );
        assert!(
            printed.contains("%0 = tribute_control.lambda : !t0"),
            "malformed lambda signature must remain attached to the lambda: {printed}"
        );

        let errors = validate_local(&ctx, module);
        let errors = messages(&errors);
        assert!(
            errors.contains("missing required attribute `type`"),
            "{errors}"
        );
        assert!(
            errors.contains("malformed tribute_control.func_sig"),
            "{errors}"
        );
    }

    #[test]
    fn generic_control_operations_round_trip_and_validate() {
        let fixture = valid_fixture();
        let result = validate(&fixture.ctx, fixture.module, &[], &mut Default::default());
        assert!(result.is_ok(), "{result}");

        let printed = assert_round_trip(&fixture.ctx, fixture.module);
        for expected in [
            "tribute_control.func_ref",
            "tribute_control.call ",
            "tribute_control.call_indirect",
            "tribute_control.perform",
            "tribute_control.handle",
            "tribute_control.handler",
            "tribute_control.resume",
            "tribute_control.yield",
        ] {
            assert!(printed.contains(expected), "missing {expected}\n{printed}");
        }
        assert!(printed.contains("tribute.calling_convention = 0"));
    }

    #[test]
    fn parser_rejects_bad_or_duplicate_convention() {
        for input in [
            r#"core.module @test {
  tribute_control.func @bad() -> core.i32 convention(bogus)
}"#,
            r#"core.module @test {
  tribute_control.func @bad() -> core.i32 convention(direct) attributes {tribute.calling_convention = 0}
}"#,
            r#"core.module @test {
  %0 = tribute_control.lambda() -> core.i32 convention(cps) captures [] attributes {tribute.calling_convention = 2} {
    %1 = arith.const {value = 1} : core.i32
    tribute_control.return %1
  }
}"#,
            r#"core.module @test {
  tribute_control.func @bad() -> core.i32 convention(direct) signature_attributes {metadata = @retired}
}"#,
            r#"core.module @test {
  %0 = tribute_control.lambda() -> core.i32 convention(cps) signature_attributes {metadata = @retired} captures [] {
    %1 = arith.const {value = 1} : core.i32
    tribute_control.return %1
  }
}"#,
        ] {
            let mut ctx = IrContext::new();
            assert!(parse_module(&mut ctx, input).is_err(), "{input}");
        }
    }

    #[test]
    fn parser_preserves_underscore_named_extra_attribute() {
        let input = r#"core.module @test {
  tribute_control.func @ok() -> core.i32 convention(direct) attributes {tribute_calling_convention = 7}
}"#;
        let mut ctx = IrContext::new();
        let root = parse_module(&mut ctx, input).expect("ordinary extra attribute should parse");
        let module = Module::new(&ctx, root).expect("core.module");
        let printed = assert_round_trip(&ctx, module);
        assert!(printed.contains("tribute_calling_convention = 7"));
    }

    #[test]
    fn validator_rejects_invalid_convention_and_duplicate_op_metadata() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  !bad = tribute_control.func_sig<() -> core.i32, {tribute.calling_convention = 3}>
  %call = tribute_control.call {callee = @missing, tribute.calling_convention = 0} : core.i32
}"#,
        );
        let result = validate_local(&ctx, module);
        let messages = messages(&result);
        assert!(messages.contains("malformed tribute_control.func_sig: Convention"));
        assert!(messages.contains("belongs only on tribute_control.func_sig"));
    }

    #[test]
    fn validator_rejects_wrong_required_attribute_types() {
        let (mut ctx, module) = parse_fixture(
            r#"core.module @test {
  !callable = tribute_control.func_sig<() -> core.i32, {tribute.calling_convention = 0}>
  tribute_control.func @malformed() -> core.i32 convention(direct)
  %ref = tribute_control.func_ref {func_ref = 1} : !callable
  %call = tribute_control.call {callee = 1} : core.i32
  %perform = tribute_control.perform {ability = 1, op_name = 2, operation_kind = 3} : core.i32
  tribute_control.handler {ability = 1, op_name = 2, kind = 3, operation_result_type = 4} {
    %constant = arith.const {value = 0} : core.i32
    tribute_control.yield %constant
  }
}"#,
        );

        // The custom func parser derives and hides sym_name/type, so malformed
        // values for those two required attributes are not text-representable.
        let malformed_func = control_op(&ctx, module, Func::NAME);
        ctx.op_attrs_mut(malformed_func)
            .insert("sym_name", Attribute::Int(1));
        ctx.op_attrs_mut(malformed_func)
            .insert("type", Attribute::Int(2));

        let result = validate_local(&ctx, module);
        assert_op_diagnostics(
            &result,
            malformed_func,
            &[
                "attribute `sym_name` must be a String attribute",
                "attribute `type` must be a Type attribute",
            ],
        );
        assert_op_diagnostics(
            &result,
            control_op(&ctx, module, FuncRef::NAME),
            &["attribute `func_ref` must be a SymbolRef attribute"],
        );
        assert_op_diagnostics(
            &result,
            control_op(&ctx, module, Call::NAME),
            &["attribute `callee` must be a SymbolRef attribute"],
        );
        assert_op_diagnostics(
            &result,
            control_op(&ctx, module, Perform::NAME),
            &[
                "attribute `ability` must be a SymbolRef attribute",
                "attribute `op_name` must be a String attribute",
                "attribute `operation_kind` must be a String attribute",
            ],
        );
        assert_op_diagnostics(
            &result,
            control_op(&ctx, module, Handler::NAME),
            &[
                "attribute `ability` must be a SymbolRef attribute",
                "attribute `op_name` must be a String attribute",
                "attribute `kind` must be a String attribute",
                "attribute `operation_result_type` must be a Type attribute",
            ],
        );
    }

    #[test]
    fn validator_rejects_nested_unresolved_call_and_perform_types() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  !wrapper = core.tuple<type.var>
  !never = core.never
  %value = arith.const {value = 0} : !wrapper
  %call = tribute_control.call %value {callee = @id} : !wrapper
  %perform = tribute_control.perform %value {ability = @State, op_name = "get", operation_kind = "op"} : !wrapper
}"#,
        );
        let never = ctx
            .types()
            .iter()
            .find_map(|(ty, data)| (data.name.is(&ctx, "core.never")).then_some(ty))
            .expect("core.never");
        assert!(!contains_unresolved_type(
            &ctx,
            never,
            &mut HashSet::default()
        ));

        let result = validate_local(&ctx, module);
        for op in [
            control_op(&ctx, module, Call::NAME),
            control_op(&ctx, module, Perform::NAME),
        ] {
            assert_eq!(
                result
                    .errors
                    .iter()
                    .filter(|error| {
                        error.op == Some(op)
                            && error
                                .message
                                .contains("operands and result must be resolved")
                    })
                    .count(),
                1,
                "{result}"
            );
        }
    }

    #[test]
    fn a_handle_without_handler_clauses_survives_a_round_trip() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  %handled = tribute_control.handle : core.i32 {
    %body = arith.const {value = 0} : core.i32
    tribute_control.yield %body
  } {
    ^completion(%value: core.i32):
      tribute_control.yield %value
  } {
    ^handlers:
  }
}"#,
        );
        let printed = trunk_ir::printer::print_module(&ctx, module.op());
        let (reparsed, module) = parse_fixture(&printed);
        let result = validate_local(&reparsed, module);
        assert!(result.is_ok(), "{result}\n{printed}");
    }

    #[test]
    fn validator_rejects_handlers_table_block_arguments() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  %handled = tribute_control.handle : core.i32 {
    %body = arith.const {value = 0} : core.i32
    tribute_control.yield %body
  } {
    ^completion(%value: core.i32):
      tribute_control.yield %value
  } {
    ^handlers(%unexpected: core.i32):
  }
}"#,
        );

        let result = validate_local(&ctx, module);
        assert!(messages(&result).contains("handlers table block must not have arguments"));
    }

    #[test]
    fn validator_rejects_structure_arity_type_and_terminator_errors() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  %handled = tribute_control.handle : core.i32 {
  } {
  }
  tribute_control.func @unterminated(%value: core.i32) -> core.i32 convention(direct) {
    %unused = arith.const {value = 0} : core.i32
  }
  %indirect = tribute_control.call_indirect : core.i32
}"#,
        );

        let result = validate_local(&ctx, module);
        let messages = messages(&result);
        assert!(messages.contains("expected 3 region(s), found 2"));
        assert!(messages.contains("must terminate with tribute_control.return"));
        assert!(messages.contains("expected at least 1 operand(s), found 0"));
    }

    #[test]
    fn local_validator_reports_distinct_malformed_operation_contracts() {
        let (mut ctx, module) = parse_fixture(
            r#"core.module @test {
  !direct = tribute_control.func_sig<(core.i32) -> core.i32, {tribute.calling_convention = 0}>
  !token = tribute_control.resume_token<core.i32, core.i32>
  %integer = arith.const {value = 1} : core.i32
  %boolean = arith.const {value = true} : core.i1

  tribute_control.func @bad_func() -> core.i32 convention(direct) {
    tribute_control.return %integer
  }
  %bad_lambda = tribute_control.lambda() -> core.i32 convention(direct) captures [%integer] {
    tribute_control.return %integer
  }
  %bad_ref = tribute_control.func_ref {func_ref = @bad_func, unexpected = 1} : !direct
  %bad_call0, %bad_call1 = tribute_control.call {unexpected = 1} : core.i32, core.i1 {
  }
  %non_callable = tribute_control.call_indirect %integer, %boolean : core.i32
  %callable = tribute_control.func_ref {func_ref = @bad_func} : !direct
  %mismatched = tribute_control.call_indirect %callable, %boolean : core.i1
  %bad_perform = tribute_control.perform %integer {ability = @State, op_name = "get", operation_kind = "bogus"} : core.i32
  tribute_control.handler {ability = @State, op_name = "get", kind = "bogus", operation_result_type = core.i32} {
    ^clause(%argument: core.i32):
      tribute_control.yield %argument
  }
  %bad_resume = tribute_control.resume %integer, %boolean : core.i1
  %token = test.token : !token
  %mismatched_resume = tribute_control.resume %token, %boolean : core.i1
  tribute_control.unknown
}"#,
        );
        let loc = location(&mut ctx);
        let i32_ty = ctx
            .types()
            .iter()
            .find_map(|(ty, data)| (data.name.is(&ctx, "core.i32")).then_some(ty))
            .unwrap();
        let i32_value = module
            .ops(&ctx)
            .iter()
            .copied()
            .find(|op| {
                ctx.op(*op).name.is(&ctx, "arith.const") && ctx.op_result_types(*op) == [i32_ty]
            })
            .map(|op| ctx.op_result(op, 0))
            .unwrap();

        // Custom func assembly requires its symbol and callable type, so the
        // parser cannot represent their simultaneous absence.
        let missing_func = trunk_ir::OperationDataBuilder::new(loc, "tribute_control.func")
            .operand(i32_value)
            .build(&mut ctx);
        let missing_func = ctx.create_op(missing_func);
        append_module_op(&mut ctx, module, missing_func);

        let bad_func = control_ops(&ctx, module, Func::NAME)
            .into_iter()
            .find(|op| ctx.op_attrs(*op).get_str("sym_name") == Some("bad_func"))
            .unwrap();
        // Custom assembly always derives a callable type and permits at most one
        // body. Mutate only those parser-enforced fields.
        ctx.op_attrs_mut(bad_func)
            .insert("type", Attribute::Type(i32_ty));
        let extra_block = block(&mut ctx, loc, &[]);
        let extra_region = region(&mut ctx, loc, extra_block);
        ctx.push_op_region(bad_func, extra_region);

        let bad_lambda = control_op(&ctx, module, Lambda::NAME);
        ctx.set_op_result_type(bad_lambda, 0, i32_ty);

        let bad_func_ref = control_ops(&ctx, module, FuncRef::NAME)
            .into_iter()
            .find(|op| ctx.op_attrs(*op).contains_key("unexpected"))
            .unwrap();
        let bad_call = control_op(&ctx, module, Call::NAME);
        let indirects = control_ops(&ctx, module, CallIndirect::NAME);
        let non_callable_indirect = indirects[0];
        let mismatched_indirect = indirects[1];
        let bad_perform = control_op(&ctx, module, Perform::NAME);
        let bad_handler = control_op(&ctx, module, Handler::NAME);
        let resumes = control_ops(&ctx, module, Resume::NAME);
        let bad_resume = resumes[0];
        let mismatched_resume = resumes[1];
        let unknown_name = ctx
            .lookup_op_name("tribute_control.unknown")
            .expect("the fixture uses tribute_control.unknown");
        let unknown = control_op(&ctx, module, unknown_name);

        let result = validate_local(&ctx, module);
        assert_op_diagnostics(
            &result,
            missing_func,
            &[
                "expected 0 operand(s), found 1",
                "missing required attribute `sym_name`",
                "missing required attribute `type`",
            ],
        );
        assert_op_diagnostics(&result, bad_func, &["expected 0 to 1 region(s), found 2"]);
        assert_op_diagnostics(
            &result,
            bad_lambda,
            &["result #0 `result`: expected tribute_control.func_sig, found core.i32"],
        );
        assert_op_diagnostics(
            &result,
            bad_func_ref,
            &["has unsupported attribute 'unexpected'"],
        );
        assert_op_diagnostics(
            &result,
            bad_call,
            &[
                "expected 1 result(s), found 2",
                "expected 0 region(s), found 1",
                "missing required attribute `callee`",
            ],
        );
        assert_op_diagnostics(
            &result,
            non_callable_indirect,
            &["operand #0 `callee`: expected S: tribute_control.func_sig, found core.i32"],
        );
        assert_op_diagnostics(
            &result,
            mismatched_indirect,
            &[
                "operands `args`: expected S::Inputs = (core.i32), found (core.i1)",
                "result #0 `result`: expected S::Result = core.i32, found core.i1",
            ],
        );
        assert_op_diagnostics(
            &result,
            bad_perform,
            &["operation_kind must be \"fn\" or \"op\", found \"bogus\""],
        );
        assert_op_diagnostics(
            &result,
            bad_handler,
            &["kind must be \"fn\" or \"op\", found \"bogus\""],
        );
        assert_op_diagnostics(
            &result,
            bad_resume,
            &[
                "operand #0 `resume_token`: expected T: tribute_control.resume_token, found core.i32",
            ],
        );
        assert_op_diagnostics(
            &result,
            mismatched_resume,
            &[
                "operand #1 `value`: expected T::Input = core.i32, found core.i1",
                "result #0 `result`: expected T::Answer = core.i32, found core.i1",
            ],
        );
        assert_op_diagnostics(
            &result,
            unknown,
            &["unsupported tribute_control operation 'unknown'"],
        );

        let unattached_return = Return::operands(i32_value).build(&mut ctx, loc);
        let mut return_errors = Vec::new();
        validate_local_operation(&ctx, unattached_return.op_ref(), &mut return_errors);
        let return_result = ValidationResult {
            errors: return_errors,
        };
        assert_op_diagnostics(
            &return_result,
            unattached_return.op_ref(),
            &[
                "must be attached to a block",
                "must terminate a tribute_control.func or lambda body",
            ],
        );

        let unattached_yield = Yield::operands(i32_value).build(&mut ctx, loc);
        let mut yield_errors = Vec::new();
        validate_local_operation(&ctx, unattached_yield.op_ref(), &mut yield_errors);
        let yield_result = ValidationResult {
            errors: yield_errors,
        };
        assert_op_diagnostics(
            &yield_result,
            unattached_yield.op_ref(),
            &[
                "must be attached to a block",
                "must terminate a handle body/completion or handler body",
            ],
        );
    }

    #[test]
    fn local_validator_checks_handle_and_handler_region_contracts() {
        let (ctx, handle_module) = parse_fixture(
            r#"core.module @test {
  %handled = tribute_control.handle : core.i32 {
    ^body(%unexpected: core.i32):
      %body_value = arith.const {value = 0} : core.i32
      tribute_control.yield %body_value
  } {
    ^completion(%value: core.i1, %extra: core.i32):
      tribute_control.yield %value
  } {
    %not_a_handler = arith.const {value = 0} : core.i32
    tribute_control.handler {ability = @State, kind = "op", op_name = "get", operation_result_type = core.i32} {
      ^clause(%argument: core.i32, %token: tribute_control.resume_token<core.i1, core.i1>):
        %wrong = arith.const {value = false} : core.i1
        tribute_control.yield %wrong
    }
  }
}"#,
        );
        let local = validate_local(&ctx, handle_module);
        assert_diagnostics(
            &local,
            &[
                "body block must not have arguments",
                "completion block must have exactly one argument",
                "completion argument type must match the body yield type",
                "completion yield type must match handle result type",
                "handlers block may contain only tribute_control.handler",
                "general handler yield type must match enclosing handle result type",
                "resume token input/answer types do not match operation and handle results",
                "general handler yield type must match handle result type",
            ],
        );

        let (ctx, fn_module) = parse_fixture(
            r#"core.module @test {
  %handled = tribute_control.handle : core.i32 {
    %body_value = arith.const {value = 0} : core.i32
    tribute_control.yield %body_value
  } {
    ^completion(%value: core.i32):
      tribute_control.yield %value
  } {
    tribute_control.handler {ability = @State, kind = "fn", op_name = "get", operation_result_type = core.i1} {
      ^clause(%argument: core.i32, %token: tribute_control.resume_token<core.i32, core.i32>):
        tribute_control.yield %argument
    }
  }
}"#,
        );
        let local = validate_local(&ctx, fn_module);
        assert_diagnostics(
            &local,
            &[
                "fn and op -> core.never handlers must not receive a resume token",
                "handler yield type does not match its kind-specific result",
            ],
        );

        let mut ctx = IrContext::new();
        let loc = location(&mut ctx);
        let i32_ty = core::I32::type_ref(&mut ctx);
        let module = module(&mut ctx, loc, &[]);
        // Textual regions always receive an implicit entry block, so zero-block
        // regions require direct in-memory construction.
        for malformed in 0..3 {
            let mut regions = Vec::new();
            for index in 0..3 {
                if index == malformed {
                    regions.push(ctx.create_region(RegionData {
                        location: loc,
                        blocks: Default::default(),
                        parent_op: None,
                    }));
                } else {
                    let region_block = block(&mut ctx, loc, &[]);
                    regions.push(region(&mut ctx, loc, region_block));
                }
            }
            let malformed_handle =
                trunk_ir::OperationDataBuilder::new(loc, "tribute_control.handle")
                    .result(i32_ty)
                    .region(regions[0])
                    .region(regions[1])
                    .region(regions[2])
                    .build(&mut ctx);
            let malformed_handle = ctx.create_op(malformed_handle);
            append_module_op(&mut ctx, module, malformed_handle);
        }

        let local = validate_local(&ctx, module);
        assert_diagnostics(
            &local,
            &[
                "body region expects exactly one block, found 0",
                "completion region expects exactly one block, found 0",
                "handlers region expects exactly one block, found 0",
            ],
        );
    }

    #[test]
    fn whole_ir_rejects_missing_and_excess_lambda_captures() {
        let (missing_ctx, missing_module) = parse_fixture(
            r#"core.module @test {
  tribute_control.func @outer(%external: core.i32) -> core.i32 convention(direct) {
    %closure = tribute_control.lambda() -> core.i32 convention(direct) captures [] {
      tribute_control.return %external
    }
    tribute_control.return %external
  }
}"#,
        );
        let missing = validate_whole_ir(&missing_ctx, missing_module, &[]);
        assert!(messages(&missing).contains("capture list is missing external value"));

        let (excess_ctx, excess_module) = parse_fixture(
            r#"core.module @test {
  tribute_control.func @outer(%external: core.i32) -> core.i32 convention(direct) {
    %closure = tribute_control.lambda() -> core.i32 convention(direct) captures [%external] {
      %constant = arith.const {value = 0} : core.i32
      tribute_control.return %constant
    }
    tribute_control.return %external
  }
}"#,
        );
        let excess = validate_whole_ir(&excess_ctx, excess_module, &[]);
        assert!(messages(&excess).contains("capture list contains unused external value"));
    }

    #[test]
    fn whole_ir_resolves_references_in_the_nearest_module() {
        let (ctx, module) = parse_fixture(
            r#"core.module @outer {
  core.module @integers {
    !callable = tribute_control.func_sig<(core.i32) -> core.i32, {tribute.calling_convention = 0}>
    tribute_control.func @id(%value: core.i32) -> core.i32 convention(direct) {
      tribute_control.return %value
    }
    tribute_control.func @use(%value: core.i32) -> core.i32 convention(direct) {
      %reference = tribute_control.func_ref {func_ref = @id} : !callable
      %direct = tribute_control.call %value {callee = @id} : core.i32
      %indirect = tribute_control.call_indirect %reference, %direct : core.i32
      tribute_control.return %indirect
    }
  }
  core.module @booleans {
    !callable = tribute_control.func_sig<(core.i1) -> core.i1, {tribute.calling_convention = 0}>
    tribute_control.func @id(%value: core.i1) -> core.i1 convention(direct) {
      tribute_control.return %value
    }
    tribute_control.func @use(%value: core.i1) -> core.i1 convention(direct) {
      %reference = tribute_control.func_ref {func_ref = @id} : !callable
      %direct = tribute_control.call %value {callee = @id} : core.i1
      %indirect = tribute_control.call_indirect %reference, %direct : core.i1
      tribute_control.return %indirect
    }
  }
}"#,
        );

        let result = validate_whole_ir(&ctx, module, &[]);
        assert!(result.is_ok(), "{:?}", result.errors);

        // A reference is never looked up in an enclosing module.
        let (ctx, module) = parse_fixture(
            r#"core.module @outer {
  tribute_control.func @id(%value: core.i32) -> core.i32 convention(direct) {
    tribute_control.return %value
  }
  core.module @integers {
    tribute_control.func @use(%value: core.i32) -> core.i32 convention(direct) {
      %direct = tribute_control.call %value {callee = @id} : core.i32
      tribute_control.return %direct
    }
  }
}"#,
        );
        let result = validate_whole_ir(&ctx, module, &[]);
        assert!(
            messages(&result).contains("unresolved callee @id"),
            "{:?}",
            result.errors
        );
    }

    #[test]
    fn whole_ir_reports_symbol_declaration_capture_and_token_contracts() {
        let (mut ctx, module) = parse_fixture(
            r#"core.module @test {
  !direct = tribute_control.func_sig<(core.i32) -> core.i32, {tribute.calling_convention = 0}>
  !cps = tribute_control.func_sig<(core.i32) -> core.i32, {tribute.calling_convention = 2}>
  !different = tribute_control.func_sig<() -> core.i1, {tribute.calling_convention = 0}>
  !token = tribute_control.resume_token<core.i32, core.i32>

  tribute_control.func @id(%value: core.i32) -> core.i32 convention(direct)
  tribute_control.func @id(%value: core.i32) -> core.i32 convention(direct)
  tribute_control.func @cps_target(%value: core.i32) -> core.i32 convention(cps) {
    tribute_control.return %value
  }
  %weak = tribute_control.func_ref {func_ref = @cps_target} : !direct
  %wrong_signature = tribute_control.func_ref {func_ref = @id} : !different
  %missing_ref = tribute_control.func_ref {func_ref = @missing} : !direct
  %missing_call = tribute_control.call {callee = @missing} : core.i32
  %false = arith.const {value = false} : core.i1
  %bad_call = tribute_control.call %false {callee = @id} : core.i1
  %unknown_perform = tribute_control.perform {ability = @State, op_name = "missing", operation_kind = "op"} : core.i32
  %bad_perform = tribute_control.perform %false {ability = @State, op_name = "get", operation_kind = "fn"} : core.i1
  %duplicate_capture = tribute_control.lambda() -> core.i1 convention(direct) captures [%false, %false] {
    tribute_control.return %false
  }
  %token_result = test.token_result : !token
  tribute_control.func @token_param(%token_arg: !token) -> core.i32 convention(direct) {
    %zero = arith.const {value = 0} : core.i32
    tribute_control.return %zero
  }
}"#,
        );
        let i32_ty = ctx
            .types()
            .iter()
            .find_map(|(ty, data)| (data.name.is(&ctx, "core.i32")).then_some(ty))
            .expect("core.i32");
        let declaration = OperationDeclaration {
            ability: SymbolPath::new(["State"]),
            op_name: ctx.intern_str("get"),
            kind: ctx.intern_str("op"),
            parameter_types: vec![i32_ty],
            result_type: i32_ty,
        };
        let declarations = [declaration.clone(), declaration];
        let result = {
            declare_operations(&mut ctx, module, &declarations);
            validate_whole_ir(&ctx, module, &[])
        };
        assert_diagnostics(
            &result,
            &[
                "duplicate function symbol @id",
                "func_ref result convention must be at least as strong as its target",
                "func_ref result source signature does not match its target",
                "unresolved function symbol @missing",
                "unresolved callee @missing",
                "call argument types do not match the logical signature",
                "call result types do not match the logical signature",
                "capture list contains duplicate values",
                "duplicate operation declaration",
                "no resolved operation declaration",
                "operation_kind does not match the resolved declaration",
                "operation argument types do not match the logical signature",
                "operation result type does not match the resolved declaration",
                "resume_token must not be produced as an operation result",
                "resume_token block arguments are allowed only as the final handler argument",
            ],
        );
    }

    #[test]
    fn whole_ir_rejects_duplicate_and_declaration_mismatched_handlers() {
        let (mut ctx, module) = parse_fixture(
            r#"core.module @test {
  %handled = tribute_control.handle : core.i32 {
    %input = arith.const {value = 0} : core.i32
    %performed = tribute_control.perform %input {ability = @State, op_name = "get", operation_kind = "op"} : core.i32
    tribute_control.yield %performed
  } {
    ^completion(%value: core.i32):
      tribute_control.yield %value
  } {
    tribute_control.handler {ability = @State, kind = "op", op_name = "get", operation_result_type = core.i32} {
      ^first(%argument: core.i32, %token: tribute_control.resume_token<core.i32, core.i32>):
        %resumed = tribute_control.resume %token, %argument : core.i32
        tribute_control.yield %resumed
    }
    tribute_control.handler {ability = @State, kind = "op", op_name = "get", operation_result_type = core.i32} {
      ^duplicate(%argument: core.i32, %token: tribute_control.resume_token<core.i32, core.i32>):
        %resumed = tribute_control.resume %token, %argument : core.i32
        tribute_control.yield %resumed
    }
  }
}"#,
        );
        let handler = control_ops(&ctx, module, Handler::NAME)[0];
        let fn_kind = ctx.intern_str("fn");
        let attrs = ctx.op_attrs(handler);
        let result_type = attrs.get_type("operation_result_type").unwrap();
        let declarations = [OperationDeclaration::new(
            attrs.get_symbol_ref("ability").cloned().unwrap(),
            attrs.get_string_ref("op_name").unwrap(),
            fn_kind,
            [result_type],
            result_type,
        )];
        let result = {
            declare_operations(&mut ctx, module, &declarations);
            validate(&ctx, module, &[], &mut Default::default())
        };
        let messages = messages(&result);
        assert!(messages.contains("duplicate handler clause"));
        assert!(messages.contains("kind does not match the resolved declaration"));
        assert!(messages.contains("operation_kind does not match the resolved declaration"));
    }

    #[test]
    fn whole_ir_rejects_multiple_resume_token_paths() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  %handled = tribute_control.handle : core.i32 {
    %body = arith.const {value = 0} : core.i32
    tribute_control.yield %body
  } {
    ^completion(%value: core.i32):
      tribute_control.yield %value
  } {
    tribute_control.handler {ability = @State, kind = "op", op_name = "get", operation_result_type = core.i32} {
      ^clause(%argument: core.i32, %token: tribute_control.resume_token<core.i32, core.i32>):
        %first = tribute_control.resume %token, %argument : core.i32
        %second = tribute_control.resume %token, %argument : core.i32
        tribute_control.yield %first
    }
  }
}"#,
        );
        let result = validate_whole_ir(&ctx, module, &[]);
        assert!(messages(&result).contains("more than one tribute_control.resume"));
    }

    #[test]
    fn whole_ir_rejects_branching_and_escape_at_each_affine_carrier() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  %handled = tribute_control.handle : core.i32 {
    %body = arith.const {value = 0} : core.i32
    tribute_control.yield %body
  } {
    ^completion(%value: core.i32):
      tribute_control.yield %value
  } {
    tribute_control.handler {ability = @State, kind = "op", op_name = "get", operation_result_type = core.i32} {
      ^clause(%argument: core.i32, %token: tribute_control.resume_token<core.i32, core.i32>):
        %inner = tribute_control.lambda() -> core.i32 convention(direct) captures [%token, %argument] {
          %resumed = tribute_control.resume %token, %argument : core.i32
          tribute_control.return %resumed
        }
        %wrapper = tribute_control.lambda() -> core.i32 convention(direct) captures [%inner] {
          %called = tribute_control.call_indirect %inner : core.i32
          tribute_control.return %called
        }
        %escape = tribute_control.call %wrapper {callee = @id} : core.i32
        %first_capture = tribute_control.lambda() -> core.i32 convention(direct) captures [%wrapper] {
          %called = tribute_control.call_indirect %wrapper : core.i32
          tribute_control.return %called
        }
        %second_capture = tribute_control.lambda() -> core.i32 convention(direct) captures [%wrapper] {
          %called = tribute_control.call_indirect %wrapper : core.i32
          tribute_control.return %called
        }
        %first = tribute_control.call_indirect %wrapper : core.i32
        %second = tribute_control.call_indirect %wrapper : core.i32
        tribute_control.yield %first
    }
  }
}"#,
        );
        let result = validate_whole_ir(&ctx, module, &[]);
        assert_diagnostics(
            &result,
            &[
                "resume-token carrier escapes through",
                "resume-token carrier branches into multiple lambda captures",
                "resume-token carrier call is outside its capture path",
            ],
        );
    }

    #[test]
    fn whole_ir_rejects_sibling_token_capture_paths() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  %handled = tribute_control.handle : core.i32 {
    %body = arith.const {value = 0} : core.i32
    tribute_control.yield %body
  } {
    ^completion(%value: core.i32):
      tribute_control.yield %value
  } {
    tribute_control.handler {ability = @State, kind = "op", op_name = "get", operation_result_type = core.i32} {
      ^clause(%argument: core.i32, %token: tribute_control.resume_token<core.i32, core.i32>):
        %first = tribute_control.lambda() -> core.i32 convention(direct) captures [%token, %argument] {
          %resumed = tribute_control.resume %token, %argument : core.i32
          tribute_control.return %resumed
        }
        %second = tribute_control.lambda() -> core.i32 convention(direct) captures [%token, %argument] {
          %resumed = tribute_control.resume %token, %argument : core.i32
          tribute_control.return %resumed
        }
        tribute_control.yield %argument
    }
  }
}"#,
        );
        let result = validate_whole_ir(&ctx, module, &[]);
        assert_diagnostics(
            &result,
            &["resume token is copied into multiple capture paths"],
        );
    }

    #[test]
    fn whole_ir_allows_mutually_exclusive_lambda_capture_and_terminal_paths() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  %handled = tribute_control.handle : core.i32 {
    %body = arith.const {value = 0} : core.i32
    tribute_control.yield %body
  } {
    ^completion(%value: core.i32):
      tribute_control.yield %value
  } {
    tribute_control.handler {ability = @State, kind = "op", op_name = "get", operation_result_type = core.i32} {
      ^clause(%argument: core.i32, %token: tribute_control.resume_token<core.i32, core.i32>):
        %flag = arith.const {value = true} : core.i1
        %value = scf.if %flag : core.i32 {
          %left = tribute_control.lambda() -> core.i32 convention(direct) captures [%token, %argument] {
            %resumed = tribute_control.resume %token, %argument : core.i32
            tribute_control.return %resumed
          }
          %called = tribute_control.call_indirect %left : core.i32
          scf.yield %called
        } {
          %right = tribute_control.lambda() -> core.i32 convention(direct) captures [%token, %argument] {
            %resumed = tribute_control.resume %token, %argument : core.i32
            tribute_control.return %resumed
          }
          %called = tribute_control.call_indirect %right : core.i32
          scf.yield %called
        }
        tribute_control.yield %value
    }
  }
}"#,
        );
        let result = validate_whole_ir(&ctx, module, &[]);
        let diagnostics = messages(&result);
        for forbidden in [
            "resume token is copied into multiple capture paths",
            "resume token capture does not form a single path to resume",
            "resume-token carrier branches into multiple lambda captures",
            "resume-token carrier has multiple static terminal uses",
        ] {
            assert!(!diagnostics.contains(forbidden), "{result}");
        }
    }

    #[test]
    fn whole_ir_allows_mutually_exclusive_switch_capture_paths_without_default() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  %handled = tribute_control.handle : core.i32 {
    %body = arith.const {value = 0} : core.i32
    tribute_control.yield %body
  } {
    ^completion(%value: core.i32):
      tribute_control.yield %value
  } {
    tribute_control.handler {ability = @State, kind = "op", op_name = "get", operation_result_type = core.i32} {
      ^clause(%argument: core.i32, %token: tribute_control.resume_token<core.i32, core.i32>):
        %choice = arith.const {value = 0} : core.i32
        scf.switch %choice {
          scf.case {value = 0} {
            %left = tribute_control.lambda() -> core.i32 convention(direct) captures [%token, %argument] {
              %left_resumed = tribute_control.resume %token, %argument : core.i32
              tribute_control.return %left_resumed
            }
            %left_called = tribute_control.call_indirect %left : core.i32
            scf.yield
          }
          scf.case {value = 1} {
            %right = tribute_control.lambda() -> core.i32 convention(direct) captures [%token, %argument] {
              %right_resumed = tribute_control.resume %token, %argument : core.i32
              tribute_control.return %right_resumed
            }
            %right_called = tribute_control.call_indirect %right : core.i32
            scf.yield
          }
        }
        tribute_control.yield %argument
    }
  }
}"#,
        );
        let result = validate_whole_ir(&ctx, module, &[]);
        let diagnostics = messages(&result);
        for forbidden in [
            "resume token is copied into multiple capture paths",
            "resume token capture does not form a single path to resume",
            "resume-token carrier branches into multiple lambda captures",
            "resume-token carrier has multiple static terminal uses",
        ] {
            assert!(!diagnostics.contains(forbidden), "{result}");
        }

        let mut branch_lambda = None;
        let mut switch = None;
        walk_region_ops(&ctx, module.body(&ctx).unwrap(), &mut |op| {
            let data = ctx.op(op);
            if data.name.is(&ctx, "tribute_control.lambda") {
                branch_lambda.get_or_insert(op);
            } else if data.name.is(&ctx, "scf.switch") {
                switch = Some(op);
            }
        });
        assert!(!ops_are_mutually_exclusive(
            &ctx,
            branch_lambda.unwrap(),
            switch.unwrap(),
        ));
    }

    #[test]
    fn whole_ir_rejects_same_path_lambda_carrier_terminals() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  %handled = tribute_control.handle : core.i32 {
    %body = arith.const {value = 0} : core.i32
    tribute_control.yield %body
  } {
    ^completion(%value: core.i32):
      tribute_control.yield %value
  } {
    tribute_control.handler {ability = @State, kind = "op", op_name = "get", operation_result_type = core.i32} {
      ^clause(%argument: core.i32, %token: tribute_control.resume_token<core.i32, core.i32>):
        %continuation = tribute_control.lambda() -> core.i32 convention(direct) captures [%token, %argument] {
          %resumed = tribute_control.resume %token, %argument : core.i32
          tribute_control.return %resumed
        }
        %first = tribute_control.call_indirect %continuation : core.i32
        %second = tribute_control.call_indirect %continuation : core.i32
        tribute_control.yield %first
    }
  }
}"#,
        );
        let result = validate_whole_ir(&ctx, module, &[]);
        assert!(
            messages(&result).contains("multiple static terminal uses"),
            "{result}"
        );
    }
    #[test]
    fn validators_reject_resume_token_escape_and_never_clause_capability() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  %handled = tribute_control.handle : core.i32 {
    %body = arith.const {value = 0} : core.i32
    tribute_control.yield %body
  } {
    ^completion(%value: core.i32):
      tribute_control.yield %value
  } {
    tribute_control.handler {ability = @State, kind = "op", op_name = "get", operation_result_type = core.i32} {
      ^clause(%argument: core.i32, %token: tribute_control.resume_token<core.i32, core.i32>):
        tribute_control.yield %token
    }
  }
}"#,
        );
        let local = validate_local(&ctx, module);
        assert!(messages(&local).contains("must not yield a resume token"));
        let whole = validate_whole_ir(&ctx, module, &[]);
        assert!(messages(&whole).contains("forbidden use"));

        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  %handled = tribute_control.handle : core.i32 {
    %body = arith.const {value = 0} : core.i32
    tribute_control.yield %body
  } {
    ^completion(%value: core.i32):
      tribute_control.yield %value
  } {
    tribute_control.handler {ability = @State, kind = "op", op_name = "get", operation_result_type = core.never} {
      ^clause(%argument: core.i32, %token: tribute_control.resume_token<core.i32, core.i32>):
        %resumed = tribute_control.resume %token, %argument : core.i32
        tribute_control.yield %resumed
    }
  }
}"#,
        );
        let result = validate_local(&ctx, module);
        let messages = messages(&result);
        assert!(messages.contains("must not receive a resume token"));
        assert!(messages.contains("must not contain tribute_control.resume"));
    }

    #[test]
    fn resume_token_components_are_checked_recursively() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  !physical = func.func_sig<() -> core.i32>
  !wrapper = core.tuple<!physical>
  !token = tribute_control.resume_token<!wrapper, core.i32>
}"#,
        );

        let result = validate_local(&ctx, module);
        assert!(messages(&result).contains("is not a resolved logical value type"));
    }

    #[test]
    fn validator_reports_malformed_resume_token_uses_without_panicking() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  !malformed = tribute_control.resume_token<core.i32>
  tribute_control.handler {ability = @State, kind = "op", op_name = "get", operation_result_type = core.i32} {
    ^clause(%value: core.i32, %token: !malformed):
      %resumed = tribute_control.resume %token, %value : core.i32
      tribute_control.yield %resumed
  }
}"#,
        );
        let handler = control_op(&ctx, module, Handler::NAME);
        let resume = control_op(&ctx, module, Resume::NAME);
        let result = validate_local(&ctx, module);
        assert!(
            result.errors.iter().any(|error| {
                error.op.is_none()
                    && error
                        .message
                        .contains("resume_token requires exactly input and answer types")
            }),
            "{result}"
        );
        assert!(
            result.errors.iter().any(|error| {
                error.op == Some(handler)
                    && error
                        .message
                        .contains("requires a final resume_token block argument")
            }),
            "{result}"
        );
        assert!(
            result.errors.iter().any(|error| {
                error.op == Some(resume)
                    && error
                        .message
                        .contains("expected T: tribute_control.resume_token, found")
            }),
            "{result}"
        );
    }

    #[test]
    fn whole_ir_rejects_branch_after_transitive_lambda_transfer() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  %handled = tribute_control.handle : core.i32 {
    %body = arith.const {value = 0} : core.i32
    tribute_control.yield %body
  } {
    ^completion(%value: core.i32):
      tribute_control.yield %value
  } {
    tribute_control.handler {ability = @State, kind = "op", op_name = "get", operation_result_type = core.i32} {
      ^clause(%argument: core.i32, %token: tribute_control.resume_token<core.i32, core.i32>):
        %inner = tribute_control.lambda() -> core.i32 convention(direct) captures [%token, %argument] {
          %resumed = tribute_control.resume %token, %argument : core.i32
          tribute_control.return %resumed
        }
        %wrapper = tribute_control.lambda() -> core.i32 convention(direct) captures [%inner] {
          %called = tribute_control.call_indirect %inner : core.i32
          tribute_control.return %called
        }
        %first = tribute_control.call_indirect %wrapper : core.i32
        %second = tribute_control.call_indirect %wrapper : core.i32
        tribute_control.yield %first
    }
  }
}"#,
        );
        let result = validate_whole_ir(&ctx, module, &[]);
        assert!(
            messages(&result).contains("multiple static terminal uses"),
            "{result}"
        );
    }

    #[test]
    fn whole_ir_rejects_outer_token_resumed_in_nested_handler() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  %outer = tribute_control.handle : core.i32 {
    %body = arith.const {value = 0} : core.i32
    tribute_control.yield %body
  } {
    ^completion(%value: core.i32):
      tribute_control.yield %value
  } {
    tribute_control.handler {ability = @State, kind = "op", op_name = "get", operation_result_type = core.i32} {
      ^outer_clause(%argument: core.i32, %outer_token: tribute_control.resume_token<core.i32, core.i32>):
        %nested = tribute_control.handle : core.i32 {
          tribute_control.yield %argument
        } {
          ^nested_completion(%value: core.i32):
            tribute_control.yield %value
        } {
          tribute_control.handler {ability = @State, kind = "fn", op_name = "get", operation_result_type = core.i32} {
            ^inner_clause(%inner_argument: core.i32):
              %crossed = tribute_control.resume %outer_token, %inner_argument : core.i32
              tribute_control.yield %crossed
          }
        }
        tribute_control.yield %nested
    }
  }
}"#,
        );
        let result = validate_whole_ir(&ctx, module, &[]);
        assert!(
            messages(&result).contains("crosses into a different handler"),
            "{result}"
        );
    }

    #[test]
    fn tail_call_arms_must_belong_to_a_one_result_if_in_tail_position() {
        let validate_body = |body: &str| {
            let (ctx, module) = parse_fixture(&format!(
                r#"core.module @test {{
  tribute_control.func @f(%n: core.i32) -> core.i32 convention(direct) {{
    tribute_control.return %n
  }}
  tribute_control.func @g(%n: core.i32, %flag: core.i1) -> core.i32 convention(direct) {{
{body}
  }}
}}"#
            ));
            messages(&validate_local(&ctx, module))
        };

        let tail_if = validate_body(
            r#"    %value = scf.if %flag : core.i32 {
      tribute_control.tail_call %n {callee = @f}
    } {
      scf.yield %n
    }
    tribute_control.return %value"#,
        );
        assert!(!tail_if.contains("tail position"), "{tail_if}");

        let resultless_if = validate_body(
            r#"    scf.if %flag {
      tribute_control.tail_call %n {callee = @f}
    } {
      scf.yield
    }
    tribute_control.return %n"#,
        );
        assert!(resultless_if.contains("tail position"), "{resultless_if}");

        let loop_arm = validate_body(
            r#"    %value = scf.loop %n : core.i32 {
    ^bb0(%i: core.i32):
      tribute_control.tail_call %i {callee = @f}
    }
    tribute_control.return %value"#,
        );
        assert!(loop_arm.contains("tail position"), "{loop_arm}");
    }

    #[test]
    fn local_validator_rejects_external_reference_in_isolated_func() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  %external = arith.const {value = 1} : core.i32
  tribute_control.func @invalid() -> core.i32 convention(direct) {
    tribute_control.return %external
  }
}"#,
        );

        let result = validate_local(&ctx, module);
        assert!(messages(&result).contains("references external value"));
    }

    #[test]
    fn a_registered_intrinsic_is_accepted_only_at_its_identity() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  tribute_control.func @"Nat::+"(%left: core.i32, %right: core.i32) -> core.i32 convention(direct)
    attributes {abi = "intrinsic", tribute.compiler_intrinsic = "Nat::+"}
}"#,
        );
        let exact =
            CompilerIntrinsicDeclaration::new(SymbolPath::new(["Nat::+"]), Symbol::new("Nat::+"));
        assert!(
            validate(
                &ctx,
                module,
                std::slice::from_ref(&exact),
                &mut Default::default()
            )
            .is_ok()
        );

        let another = CompilerIntrinsicDeclaration::new(exact.symbol, Symbol::new("Int::+"));
        let result = validate(&ctx, module, &[another], &mut Default::default());
        assert!(
            messages(&result).contains("identity does not match"),
            "{result}"
        );

        let result = validate(&ctx, module, &[], &mut Default::default());
        assert!(
            messages(&result).contains("unregistered declaration"),
            "{result}"
        );
    }

    #[test]
    fn registered_intrinsic_requires_direct_convention() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  tribute_control.func @read(%value: core.i32) -> core.i32 convention(evidence_direct)
    attributes {tribute.compiler_intrinsic = "read"}
}"#,
        );
        let declaration =
            CompilerIntrinsicDeclaration::new(SymbolPath::new(["read"]), Symbol::new("read"));

        let result = validate(&ctx, module, &[declaration], &mut Default::default());
        assert!(
            messages(&result).contains("must use Direct calling convention"),
            "{result}"
        );
    }

    #[test]
    fn exact_registered_intrinsic_has_indirect_callable_provenance() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  !F = tribute_control.func_sig<(core.i32, core.i32) -> core.i32, {tribute.calling_convention = 0}>
  tribute_control.func @"Nat::+"(%left: core.i32, %right: core.i32) -> core.i32 convention(direct)
    attributes {abi = "intrinsic", tribute.compiler_intrinsic = "Nat::+"}
  tribute_control.func @caller(%left: core.i32, %right: core.i32) -> core.i32 convention(direct) {
    %callee = tribute_control.func_ref {func_ref = @"Nat::+"} : !F
    %result = tribute_control.call_indirect %callee, %left, %right : core.i32
    tribute_control.return %result
  }
}"#,
        );
        let declaration =
            CompilerIntrinsicDeclaration::new(SymbolPath::new(["Nat::+"]), Symbol::new("Nat::+"));

        let result = validate(&ctx, module, &[declaration], &mut Default::default());
        assert!(result.is_ok(), "{result}");
    }

    #[test]
    fn intrinsic_registry_order_duplicates_and_defined_claims_fail_closed() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  tribute_control.func @"Int::+"(%left: core.i32, %right: core.i32) -> core.i32 convention(direct)
    attributes {abi = "intrinsic", tribute.compiler_intrinsic = "Int::+"}
  tribute_control.func @"Nat::+"(%left: core.i32, %right: core.i32) -> core.i32 convention(direct)
    attributes {abi = "intrinsic", tribute.compiler_intrinsic = "Nat::+"} {
    tribute_control.return %left
  }
}"#,
        );
        let declaration = |symbol: Symbol| {
            CompilerIntrinsicDeclaration::new(SymbolPath::new([symbol.clone()]), symbol)
        };
        let nat = declaration(Symbol::new("Nat::+"));
        let int = declaration(Symbol::new("Int::+"));

        let result = validate(
            &ctx,
            module,
            &[nat.clone(), int, nat],
            &mut Default::default(),
        );
        let diagnostics = messages(&result);
        assert!(
            diagnostics.contains("not deterministically ordered"),
            "{result}"
        );
        assert!(
            diagnostics.contains("duplicate compiler intrinsic"),
            "{result}"
        );
        assert!(
            diagnostics.contains("defined Tribute function cannot claim compiler intrinsic"),
            "{result}"
        );
    }

    #[test]
    fn managed_reference_metadata_and_return_contracts_fail_closed() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  !S = adt.struct<S()>
  !Unnamed = adt.typeref
  !Parameterized = adt.typeref<core.i32, {name = "S"}>
  !Missing = adt.typeref<{name = "Missing"}>
  tribute_control.func @unnamed(%value: !Unnamed) -> !Unnamed convention(direct) {
    tribute_control.return %value
  }
  tribute_control.func @parameterized(%value: !Parameterized) -> !Parameterized convention(direct) {
    tribute_control.return %value
  }
  tribute_control.func @missing(%value: !Missing) -> !Missing convention(direct) {
    tribute_control.return %value
  }
  tribute_control.func @wrong_return(%integer: core.i32, %flag: core.i1) -> core.i32 convention(direct) {
    tribute_control.return %flag
  }
}"#,
        );

        let result = validate(&ctx, module, &[], &mut Default::default());
        let diagnostics = messages(&result);
        assert!(
            diagnostics.contains("requires nominal name metadata"),
            "{result}"
        );
        assert!(
            diagnostics.contains("cannot have type parameters"),
            "{result}"
        );
        assert!(
            diagnostics.contains("has no verified nominal declaration"),
            "{result}"
        );
        assert!(
            diagnostics.contains("return operand type does not match containing semantic callable"),
            "{result}"
        );
    }

    #[test]
    fn managed_defined_boundaries_null_and_compatible_cast_are_valid() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  !S = adt.struct<S()>
  !R = adt.typeref<{name = "S"}>
  tribute_control.func @managed(%value: !R) -> !R convention(direct) {
    %null = adt.ref_null {type = !R} : !R
    %cast = adt.ref_cast %null {type = !R} : !R
    tribute_control.return %cast
  }
}"#,
        );

        let result = validate(&ctx, module, &[], &mut Default::default());
        assert!(result.is_ok(), "{result}");
    }

    #[test]
    fn managed_bodyless_external_read_line_shape_and_raw_pointer_cast_chain_fail_closed() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  !ReadLineResult = adt.enum<ReadLineResult { ReadLine(core.bytes), ReadEndOfFile(), ReadInvalidEncoding(), ReadSystem(core.i32, core.bytes) }>
  !ReadLineResultRef = adt.typeref<{name = "ReadLineResult"}>
  tribute_control.func @user_read_line() -> !ReadLineResultRef convention(direct)
    attributes {abi = "intrinsic"}
  tribute_control.func @masquerade(%raw: core.ptr) -> !ReadLineResultRef convention(direct) {
    %middle = core.unrealized_conversion_cast %raw : core.i64
    %managed = core.unrealized_conversion_cast %middle : !ReadLineResultRef
    tribute_control.return %managed
  }
}"#,
        );

        let result = validate(&ctx, module, &[], &mut Default::default());
        let diagnostics = messages(&result);
        assert!(diagnostics.contains("bodyless external"), "{result}");
        assert!(diagnostics.contains("core.ptr cast chain"), "{result}");
    }

    #[test]
    fn managed_nested_aggregate_in_bodyless_c_external_is_a_trusted_user_boundary() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  !S = adt.struct<S()>
  !R = adt.typeref<{name = "S"}>
  !Container = adt.struct<Container(managed: !R)>
  tribute_control.func @private_helper(%value: !Container) -> core.i32 convention(direct)
    attributes {abi = "C"}
}"#,
        );

        let result = validate(&ctx, module, &[], &mut Default::default());
        assert!(result.is_ok(), "{result}");
    }

    #[test]
    fn primitive_external_remains_a_conservative_boundary() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  tribute_control.func @primitive(%buffer: core.ptr, %length: core.i32) -> core.ptr convention(direct)
    attributes {abi = "C"}
}"#,
        );

        let result = validate(&ctx, module, &[], &mut Default::default());
        assert!(result.is_ok(), "{result}");
    }

    #[test]
    fn ref_cast_rejects_distinct_nominal_managed_references() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  !S = adt.struct<S()>
  !T = adt.struct<T()>
  !RS = adt.typeref<{name = "S"}>
  !RT = adt.typeref<{name = "T"}>
  tribute_control.func @incompatible(%value: !RS) -> !RT convention(direct) {
    %cast = adt.ref_cast %value {type = !RT} : !RT
    tribute_control.return %cast
  }
}"#,
        );

        let result = validate(&ctx, module, &[], &mut Default::default());
        assert!(
            messages(&result).contains("compatible managed nominal reference types"),
            "{result}"
        );
    }

    #[test]
    fn indirect_call_rejects_typed_callable_cast_without_semantic_provenance() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  tribute_control.func @caller(%raw: core.ptr, %value: core.i32) -> core.i32 convention(direct) {
    %callee = core.unrealized_conversion_cast %raw : tribute_control.func_sig<(core.i32) -> core.i32, {tribute.calling_convention = 0}>
    %result = tribute_control.call_indirect %callee, %value : core.i32
    tribute_control.return %result
  }
}"#,
        );

        let result = validate(&ctx, module, &[], &mut Default::default());
        assert!(
            messages(&result).contains("callable provenance"),
            "{result}"
        );
    }

    #[test]
    fn indirect_call_rejects_callable_from_uncontracted_structured_block_argument() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  !F = tribute_control.func_sig<(core.i32) -> core.i32, {tribute.calling_convention = 0}>
  tribute_control.func @caller(%condition: core.i1, %value: core.i32) -> core.i32 convention(direct) {
    %result = scf.if %condition : core.i32 {
      ^then(%callee: !F):
        %called = tribute_control.call_indirect %callee, %value : core.i32
        scf.yield %called
    } {
      ^else:
        scf.yield %value
    }
    tribute_control.return %result
  }
}"#,
        );

        let result = validate(&ctx, module, &[], &mut Default::default());
        assert!(
            messages(&result).contains("callable provenance"),
            "{result}"
        );
    }

    #[test]
    fn callable_handle_completion_argument_traces_the_body_yield() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  !F = tribute_control.func_sig<(core.i32) -> core.i32, {tribute.calling_convention = 0}>
  tribute_control.func @id(%value: core.i32) -> core.i32 convention(direct) {
    tribute_control.return %value
  }
  tribute_control.func @caller(%value: core.i32) -> !F convention(direct) {
    %callback = tribute_control.handle : !F {
      %source = tribute_control.func_ref {func_ref = @id} : !F
      tribute_control.yield %source
    } {
      ^completion(%callback: !F):
        %ignored = tribute_control.call_indirect %callback, %value : core.i32
        tribute_control.yield %callback
    } {
      ^handlers:
    }
    tribute_control.return %callback
  }
}"#,
        );

        let result = validate(&ctx, module, &[], &mut Default::default());
        assert!(result.is_ok(), "{result}");
    }

    #[test]
    fn callable_handler_parameter_requires_an_exact_operation_declaration() {
        let (mut ctx, module) = parse_fixture(
            r#"core.module @test {
  !F = tribute_control.func_sig<(core.i32) -> core.i32, {tribute.calling_convention = 0}>
  tribute_control.func @caller(%value: core.i32) -> core.i32 convention(direct) {
    %handled = tribute_control.handle : core.i32 {
      tribute_control.yield %value
    } {
      ^completion(%completed: core.i32):
        tribute_control.yield %completed
    } {
      tribute_control.handler {ability = @Callback, kind = "fn", op_name = "apply", operation_result_type = core.i32} {
        ^handler(%callback: !F):
          %called = tribute_control.call_indirect %callback, %value : core.i32
          tribute_control.yield %called
      }
    }
    tribute_control.return %handled
  }
}"#,
        );
        let handler = control_op(&ctx, module, Handler::NAME);
        let handler_region = ctx.op_region(handler, 0).unwrap();
        let handler_block = ctx.region(handler_region).blocks[0];
        let callback_type = ctx.value_ty(ctx.block_args(handler_block)[0]);
        let ability = ctx
            .op_attrs(handler)
            .get_symbol_ref("ability")
            .cloned()
            .unwrap();
        let operation_result = ctx
            .op_attrs(handler)
            .get_type("operation_result_type")
            .unwrap();
        let declaration = OperationDeclaration::new(
            ability,
            ctx.intern_str("apply"),
            ctx.intern_str("fn"),
            [callback_type],
            operation_result,
        );

        let result = {
            declare_operations(&mut ctx, module, &[declaration]);
            validate(&ctx, module, &[], &mut Default::default())
        };
        assert!(result.is_ok(), "{result}");
    }

    fn performed_callable_module(declared_result: &str) -> String {
        format!(
            r#"core.module @test {{
  !F = tribute_control.func_sig<(core.i32) -> core.i32, {{tribute.calling_convention = 0}}>
  tribute_control.func @run(%value: core.i32) -> core.i32 convention(cps) {{
    %callee = tribute_control.perform {{ability = @Reader, op_name = "read", operation_kind = "fn"}} : !F
    %result = tribute_control.call_indirect %callee, %value : core.i32
    tribute_control.return %result
  }}
  ability.decl {{sym_name = "Reader"}} {{
    ability.operation {{kind = "fn", op_name = "read", param_types = [], result_type = {declared_result}}}
  }}
}}"#
        )
    }

    #[test]
    fn declared_perform_result_has_callable_provenance() {
        let (ctx, module) = parse_fixture(&performed_callable_module("!F"));

        let result = validate(&ctx, module, &[], &mut Default::default());
        assert!(result.is_ok(), "{result}");
    }

    #[test]
    fn perform_result_needs_the_declared_callable_type() {
        let (ctx, module) = parse_fixture(&performed_callable_module("core.i32"));

        let result = validate(&ctx, module, &[], &mut Default::default());
        assert!(
            messages(&result).contains("callable provenance"),
            "{result}"
        );
    }

    fn handled_callable_module(op_yield: &str) -> String {
        format!(
            r#"core.module @test {{
  !F = tribute_control.func_sig<(core.i32) -> core.i32, {{tribute.calling_convention = 0}}>
  tribute_control.func @inc(%n: core.i32) -> core.i32 convention(direct) {{
    tribute_control.return %n
  }}
  %f = tribute_control.handle : !F {{
    %body = tribute_control.perform {{ability = @Gen, op_name = "next", operation_kind = "op"}} : core.i32
    tribute_control.yield %body
  }} {{
    ^completion(%k: core.i32):
      %done = tribute_control.func_ref {{func_ref = @inc}} : !F
      tribute_control.yield %done
  }} {{
    tribute_control.handler {{ability = @Gen, kind = "op", op_name = "next", operation_result_type = core.i32}} {{
      ^next(%token: tribute_control.resume_token<core.i32, !F>):
        %resumed_value = arith.const {{value = 40}} : core.i32
        %g = tribute_control.resume %token, %resumed_value : !F
        %called = tribute_control.call_indirect %g, %resumed_value : core.i32
        %zero = arith.const {{value = 0}} : core.i32
        %bad = core.unrealized_conversion_cast %zero : !F
        tribute_control.yield {op_yield}
    }}
    tribute_control.handler {{ability = @Ask, kind = "fn", op_name = "ask", operation_result_type = core.i32}} {{
      ^ask:
        %answer = arith.const {{value = 2}} : core.i32
        tribute_control.yield %answer
    }}
  }}
  %one = arith.const {{value = 1}} : core.i32
  %result = tribute_control.call_indirect %f, %one : core.i32
  ability.decl {{sym_name = "Gen"}} {{
    ability.operation {{kind = "op", op_name = "next", param_types = [], result_type = core.i32}}
  }}
  ability.decl {{sym_name = "Ask"}} {{
    ability.operation {{kind = "fn", op_name = "ask", param_types = [], result_type = core.i32}}
  }}
}}"#
        )
    }

    #[test]
    fn handle_and_resume_results_have_callable_provenance() {
        let (ctx, module) = parse_fixture(&handled_callable_module("%g"));

        let result = validate(&ctx, module, &[], &mut Default::default());
        assert!(result.is_ok(), "{result}");
    }

    #[test]
    fn handle_result_needs_every_yield_to_have_callable_provenance() {
        let (ctx, module) = parse_fixture(&handled_callable_module("%bad"));

        let result = validate(&ctx, module, &[], &mut Default::default());
        assert_eq!(
            messages(&result).matches("callable provenance").count(),
            2,
            "{result}"
        );
    }

    #[test]
    fn resume_result_needs_a_handler_resume_token() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  !F = tribute_control.func_sig<(core.i32) -> core.i32, {tribute.calling_convention = 0}>
  !Token = tribute_control.resume_token<core.i32, !F>
  tribute_control.func @caller(%token: !Token, %raw: core.ptr, %value: core.i32) -> core.i32 convention(direct) {
    %from_parameter = tribute_control.resume %token, %value : !F
    %first = tribute_control.call_indirect %from_parameter, %value : core.i32
    %cast_token = core.unrealized_conversion_cast %raw : !Token
    %from_cast = tribute_control.resume %cast_token, %first : !F
    %second = tribute_control.call_indirect %from_cast, %first : core.i32
    tribute_control.return %second
  }
}"#,
        );

        let result = validate(&ctx, module, &[], &mut Default::default());
        assert_eq!(
            messages(&result).matches("callable provenance").count(),
            2,
            "{result}"
        );
    }

    #[test]
    fn lambda_parameters_indirect_results_and_list_heads_have_callable_provenance() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  !F = tribute_control.func_sig<(core.i32) -> core.i32, {tribute.calling_convention = 0}>
  !G = tribute_control.func_sig<(!F, core.i32) -> core.i32, {tribute.calling_convention = 0}>
  !H = tribute_control.func_sig<(core.i32) -> !F, {tribute.calling_convention = 0}>
  tribute_control.func @caller(%list: core.ptr, %make: !H, %value: core.i32) -> core.i32 convention(direct) {
    %apply = tribute_control.lambda : !G {
      ^bb0(%callback: !F, %argument: core.i32):
        %applied = tribute_control.call_indirect %callback, %argument : core.i32
        tribute_control.return %applied
    }
    %made = tribute_control.call_indirect %make, %value : !F
    %first = tribute_control.call_indirect %made, %value : core.i32
    %head = list.head %list {element_type = !F} : !F
    %second = tribute_control.call_indirect %head, %first : core.i32
    %third = tribute_control.call_indirect %apply, %head, %second : core.i32
    tribute_control.return %third
  }
}"#,
        );

        let result = validate(&ctx, module, &[], &mut Default::default());
        assert!(result.is_ok(), "{result}");
    }

    #[test]
    fn indirect_calls_sharing_a_callee_chain_evaluate_each_value_once() {
        const CHAIN: usize = 8;
        const CALLS: usize = 5;
        let mut text = String::from("core.module @test {\n");
        text.push_str(
            "  !F0 = tribute_control.func_sig<(core.i32) -> core.i32, {tribute.calling_convention = 0}>\n",
        );
        for depth in 1..=CHAIN {
            text.push_str(&format!(
                "  !F{depth} = tribute_control.func_sig<(core.i32) -> !F{}, {{tribute.calling_convention = 0}}>\n",
                depth - 1
            ));
        }
        text.push_str(&format!(
            "  tribute_control.func @caller(%v{CHAIN}: !F{CHAIN}, %x: core.i32) -> core.i32 convention(direct) {{\n"
        ));
        for depth in (0..CHAIN).rev() {
            text.push_str(&format!(
                "    %v{depth} = tribute_control.call_indirect %v{}, %x : !F{depth}\n",
                depth + 1
            ));
        }
        for call in 0..CALLS {
            text.push_str(&format!(
                "    %r{call} = tribute_control.call_indirect %v0, %x : core.i32\n"
            ));
        }
        text.push_str("    tribute_control.return %r0\n  }\n}");
        let (ctx, module) = parse_fixture(&text);
        let body = module.body(&ctx).unwrap();
        let functions = SymbolTable::collect(&ctx, module);
        let (registered, declarations, layouts) =
            (HashMap::default(), HashMap::default(), HashMap::default());
        let mut query = CallableQuery::new(&functions, &registered, &declarations, &layouts);

        let callees = indirect_callees(&ctx, body);
        for &callee in &callees {
            query.accepts(&ctx, callee);
        }
        query.resolve(&ctx);

        assert_eq!(callees.len(), CHAIN + CALLS);
        assert!(callees.iter().all(|&callee| query.is_origin(callee)));
        assert_eq!(query.evaluations, CHAIN + 1);
    }

    #[test]
    fn a_withdrawn_origin_withdraws_the_values_that_depend_on_it() {
        let (ctx, module) = parse_fixture(&handled_callable_module("%bad"));
        let body = module.body(&ctx).unwrap();
        let functions = SymbolTable::collect(&ctx, module);
        let declarations = declared_operations(&ctx, module, &mut Vec::new());
        let declarations = declaration_map(&ctx, &declarations, &mut Vec::new());
        let (registered, layouts) = (HashMap::default(), HashMap::default());
        let mut query = CallableQuery::new(&functions, &registered, &declarations, &layouts);

        let callees = indirect_callees(&ctx, body);
        for &callee in &callees {
            query.accepts(&ctx, callee);
        }
        query.resolve(&ctx);

        assert!(callees.iter().all(|&callee| !query.is_origin(callee)));
    }

    #[test]
    fn a_value_withdrawn_twice_is_not_evaluated_again() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  !F = tribute_control.func_sig<(core.i32) -> core.i32, {tribute.calling_convention = 0}>
  %f = tribute_control.handle : !F {
    %body = tribute_control.perform {ability = @Gen, op_name = "next", operation_kind = "op"} : core.i32
    tribute_control.yield %body
  } {
    ^completion(%k: core.i32):
      %done = core.unrealized_conversion_cast %k : !F
      tribute_control.yield %done
  } {
    tribute_control.handler {ability = @Gen, kind = "op", op_name = "next", operation_result_type = core.i32} {
      ^next(%token: tribute_control.resume_token<core.i32, !F>):
        %zero = arith.const {value = 0} : core.i32
        %bad = core.unrealized_conversion_cast %zero : !F
        tribute_control.yield %bad
    }
  }
  %one = arith.const {value = 1} : core.i32
  %result = tribute_control.call_indirect %f, %one : core.i32
  ability.decl {sym_name = "Gen"} {
    ability.operation {kind = "op", op_name = "next", param_types = [], result_type = core.i32}
  }
}"#,
        );
        let body = module.body(&ctx).unwrap();
        let functions = SymbolTable::collect(&ctx, module);
        let declarations = declared_operations(&ctx, module, &mut Vec::new());
        let declarations = declaration_map(&ctx, &declarations, &mut Vec::new());
        let (registered, layouts) = (HashMap::default(), HashMap::default());
        let mut query = CallableQuery::new(&functions, &registered, &declarations, &layouts);

        let callees = indirect_callees(&ctx, body);
        assert_eq!(callees.len(), 1);
        let callee = callees[0];
        query.accepts(&ctx, callee);
        query.resolve(&ctx);

        // The handle result, both yields, then the handle result once more:
        // the second withdrawal finds it already withdrawn.
        assert!(!query.is_origin(callee));
        assert_eq!(query.evaluations, 4);
    }

    #[test]
    fn exact_adt_callable_projections_have_semantic_provenance() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  !F = tribute_control.func_sig<(core.i32) -> core.i32, {tribute.calling_convention = 0}>
  !Tuple = adt.struct<Tuple(callee: !F)>
  !TupleRef = adt.typeref<{name = "Tuple"}>
  !Choice = adt.enum<Choice { Some(!F) }>
  !ChoiceRef = adt.typeref<{name = "Choice"}>
  tribute_control.func @caller(%tuple: !TupleRef, %choice: !ChoiceRef, %value: core.i32) -> core.i32 convention(direct) {
    %tuple_callee = adt.struct_get %tuple {type = !Tuple, field = 0} : !F
    %tuple_result = tribute_control.call_indirect %tuple_callee, %value : core.i32
    %some = adt.variant_cast %choice {type = !Choice, tag = "Some"} : !Choice
    %choice_callee = adt.variant_get %some {type = !Choice, tag = "Some", field = 0} : !F
    %choice_result = tribute_control.call_indirect %choice_callee, %tuple_result : core.i32
    tribute_control.return %choice_result
  }
}"#,
        );

        let result = validate(&ctx, module, &[], &mut Default::default());
        assert!(result.is_ok(), "{result}");
    }

    #[test]
    fn adt_callable_projection_requires_matching_declared_field_type() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  !F = tribute_control.func_sig<(core.i32) -> core.i32, {tribute.calling_convention = 0}>
  !Tuple = adt.struct<Tuple(not_callable: core.i32)>
  !TupleRef = adt.typeref<{name = "Tuple"}>
  !Choice = adt.enum<Choice { Some(core.i32) }>
  !ChoiceRef = adt.typeref<{name = "Choice"}>
  tribute_control.func @caller(%tuple: !TupleRef, %choice: !ChoiceRef, %value: core.i32) -> core.i32 convention(direct) {
    %tuple_callee = adt.struct_get %tuple {type = !Tuple, field = 0} : !F
    %tuple_result = tribute_control.call_indirect %tuple_callee, %value : core.i32
    %some = adt.variant_cast %choice {type = !Choice, tag = "Some"} : !Choice
    %choice_callee = adt.variant_get %some {type = !Choice, tag = "Some", field = 0} : !F
    %choice_result = tribute_control.call_indirect %choice_callee, %tuple_result : core.i32
    tribute_control.return %choice_result
  }
}"#,
        );

        let result = validate(&ctx, module, &[], &mut Default::default());
        assert_eq!(
            messages(&result).matches("callable provenance").count(),
            2,
            "{result}"
        );
    }

    #[test]
    fn duplicate_nominal_layout_rejects_callable_projection_spoofing() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  !F = tribute_control.func_sig<(core.i32) -> core.i32, {tribute.calling_convention = 0}>
  !Canonical = adt.struct<Tuple(value: core.i32)>
  !Spoofed = adt.struct<Tuple(callee: !F)>
  !TupleRef = adt.typeref<{name = "Tuple"}>
  tribute_control.func @caller(%tuple: !TupleRef, %value: core.i32) -> core.i32 convention(direct) {
    %canonical = adt.struct_new %value {type = !Canonical} : !TupleRef
    %callee = adt.struct_get %tuple {type = !Spoofed, field = 0} : !F
    %result = tribute_control.call_indirect %callee, %value : core.i32
    tribute_control.return %result
  }
}"#,
        );

        let result = validate(&ctx, module, &[], &mut Default::default());
        let diagnostics = messages(&result);
        assert!(diagnostics.contains("nominal layout \"Tuple\" is declared more than once"));
        assert!(diagnostics.contains("callable provenance"), "{result}");
    }

    #[test]
    fn unreachable_nominal_layout_collision_does_not_affect_validation() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  !UnusedStruct = adt.struct<Unused(value: core.i32)>
  !UnusedEnum = adt.enum<Unused { Value(core.i32) }>
  tribute_control.func @caller(%value: core.i32) -> core.i32 convention(direct) {
    tribute_control.return %value
  }
}"#,
        );

        let result = validate(&ctx, module, &[], &mut Default::default());
        assert!(result.is_ok(), "{result}");
    }

    #[test]
    fn indirect_call_rejects_callable_from_unclassified_external() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  !F = tribute_control.func_sig<(core.i32) -> core.i32, {tribute.calling_convention = 0}>
  tribute_control.func @factory() -> !F convention(direct) attributes {abi = "C"}
  tribute_control.func @caller(%value: core.i32) -> core.i32 convention(direct) {
    %callee = tribute_control.call {callee = @factory} : !F
    %result = tribute_control.call_indirect %callee, %value : core.i32
    tribute_control.return %result
  }
}"#,
        );

        let result = validate(&ctx, module, &[], &mut Default::default());
        assert!(
            messages(&result).contains("callable provenance"),
            "{result}"
        );
    }

    #[test]
    fn only_func_is_isolated_and_only_safe_value_ops_are_pure() {
        let fixture = valid_fixture();
        let ctx = &fixture.ctx;
        let body = fixture.module.body(ctx).expect("module body");
        let mut func = None;
        let mut lambda = None;
        let mut func_ref = None;
        let mut call = None;
        walk_region_ops(ctx, body, &mut |op| {
            func = func.or_else(|| Func::matches(ctx, op).then_some(op));
            lambda = lambda.or_else(|| Lambda::matches(ctx, op).then_some(op));
            func_ref = func_ref.or_else(|| FuncRef::matches(ctx, op).then_some(op));
            call = call.or_else(|| Call::matches(ctx, op).then_some(op));
        });
        let func = func.expect("func");
        assert!(IsolatedFromAbove::is_set(ctx, func));
        if let Some(lambda) = lambda {
            assert!(!IsolatedFromAbove::is_set(ctx, lambda));
        }
        assert!(Pure::is_set(ctx, func_ref.expect("func_ref")));
        assert!(!Pure::is_set(ctx, call.expect("call")));
    }

    #[test]
    fn nested_self_calls_are_recognized_in_their_module() {
        let (ctx, module) = parse_fixture(
            r#"core.module @outer {
  core.module @inner {
    tribute_control.func @loop(%value: core.i32) -> core.i32 convention(direct) {
      %again = tribute_control.call %value {callee = @loop} : core.i32
      tribute_control.return %again
    }
  }
}"#,
        );
        let mut calls = Vec::new();
        let _ = walk_op::<()>(&ctx, module.op(), &mut |op| {
            if Call::matches(&ctx, op) {
                calls.push(op);
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
        assert_eq!(calls.len(), 1);
        assert!(direct_call_reenters_enclosing_func(&ctx, calls[0]));
    }

    /// `VALID_CONTROL_MODULE` with evidence selections on its call, indirect
    /// call, handle, and resume.
    fn evidence_plan_module(call: &str, handle: &str) -> String {
        VALID_CONTROL_MODULE
            .replace(
                "{callee = @id}",
                &format!("{{callee = @id, evidence_plan = {call}}}"),
            )
            .replace(
                "%function, %value : core.i32",
                "%function, %value {evidence_plan = [{dup = @State}]} : core.i32",
            )
            .replace(
                "tribute_control.handle : core.i32",
                &format!("tribute_control.handle {{evidence_plan = {handle}}} : core.i32"),
            )
            .replace(
                "%token, %argument : core.i32",
                "%token, %argument {evidence_plan = [{mask = @Log}]} : core.i32",
            )
    }

    #[test]
    fn evidence_plans_validate_and_round_trip() {
        let (ctx, module) = parse_fixture(&evidence_plan_module(
            "[{mask = @State}, {dup = @Log}]",
            "[{mask = @State}]",
        ));
        let result = validate_local(&ctx, module);
        assert!(result.is_ok(), "{result}");

        let call = Call::from_op(&ctx, control_op(&ctx, module, Call::NAME)).unwrap();
        match call.evidence_plan(&ctx).unwrap().next() {
            Some(EvidenceStep::Mask(state)) => assert_eq!(state, SymbolPath::new(["State"])),
            other => panic!("expected a leading mask, found {other:?}"),
        }
        let steps: Vec<_> = call.evidence_plan(&ctx).unwrap().collect();
        assert!(matches!(
            steps[..],
            [EvidenceStep::Mask(_), EvidenceStep::Dup(_)]
        ));

        let printed = assert_round_trip(&ctx, module);
        assert!(
            printed.contains("evidence_plan = [{mask = @State}, {dup = @Log}]"),
            "{printed}"
        );
    }

    #[test]
    fn evidence_plans_with_row_tails_validate_and_round_trip() {
        let plan = "[{select = 1}, {push = @State}, {dup = @State}, \
                    {tails = [[], [{select = 0}, {mask = @Log}]]}]";
        let (ctx, module) = parse_fixture(&evidence_plan_module(plan, "[{mask = @State}]"));
        let result = validate_local(&ctx, module);
        assert!(result.errors.is_empty(), "{result}");
        let call = Call::from_op(&ctx, control_op(&ctx, module, Call::NAME)).unwrap();
        let steps: Vec<_> = call.evidence_plan(&ctx).unwrap().collect();
        let [
            EvidenceStep::Select(1),
            EvidenceStep::Push(_),
            EvidenceStep::Dup(_),
            EvidenceStep::Tails(tails),
        ] = steps.as_slice()
        else {
            panic!("unexpected plan: {steps:?}");
        };
        assert!(tails[0].is_empty());
        assert!(matches!(
            tails[1].as_slice(),
            [EvidenceStep::Select(0), EvidenceStep::Mask(_)]
        ));
        let printed = print_module(&ctx, module.op());
        assert!(
            printed.contains("{tails = [[], [{select = 0}, {mask = @Log}]]}"),
            "{printed}"
        );
    }

    #[test]
    fn evidence_plan_verifier_rejects_malformed_selections() {
        for (call, handle, expected) in [
            ("[]", "[{mask = @State}]", "evidence_plan must not be empty"),
            (
                "[{mask = @State}, {dup = @State}]",
                "[{mask = @State}]",
                "names ability instance",
            ),
            (
                "[{keep = @State}]",
                "[{mask = @State}]",
                "must be mask, dup, outer, select, push, or tails, found keep",
            ),
            (
                "[{mask = @State}, {select = 0}]",
                "[{mask = @State}]",
                "select must be the first step",
            ),
            (
                "[{select = @State}]",
                "[{mask = @State}]",
                "select must be a row tail index",
            ),
            (
                "[{push = @State}]",
                "[{mask = @State}]",
                "push requires a selected row tail",
            ),
            (
                "[{tails = [[], []]}, {mask = @State}]",
                "[{mask = @State}]",
                "tails must be the last step",
            ),
            (
                "[{tails = [[{tails = [[]]}]]}]",
                "[{mask = @State}]",
                "a row tail may not attach tails",
            ),
            (
                "[{tails = [{mask = @State}]}]",
                "[{mask = @State}]",
                "tails must be a list of plans",
            ),
            (
                "[{outer = @State}]",
                "[{mask = @State}]",
                "may not use outer",
            ),
            (
                "[{mask = core.i32}]",
                "[{mask = @State}]",
                "evidence_plan mask must name an ability instance",
            ),
            (
                "[{mask = @State, dup = @Log}]",
                "[{mask = @State}]",
                "must have exactly one entry, found 2",
            ),
            (
                "[{mask = @State}]",
                "[{dup = @State}]",
                "a handle's evidence_plan may only mask",
            ),
        ] {
            let (ctx, module) = parse_fixture(&evidence_plan_module(call, handle));
            let result = validate_local(&ctx, module);
            let op = if handle.contains("dup") {
                control_op(&ctx, module, Handle::NAME)
            } else {
                control_op(&ctx, module, Call::NAME)
            };
            assert_op_diagnostics(&result, op, &[expected]);
            assert_eq!(result.errors.len(), 1, "{call} {handle}: {result}");
        }
    }

    #[test]
    fn evidence_plan_elements_must_be_type_dictionaries() {
        let (ctx, module) = parse_fixture(&evidence_plan_module("[1]", "[{mask = @State}]"));
        let result = validate_local(&ctx, module);
        assert_eq!(result.errors.len(), 1, "{result}");
        assert_eq!(
            result.errors[0].op,
            Some(control_op(&ctx, module, Call::NAME))
        );
    }
}
