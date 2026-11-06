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
use trunk_ir::op_flag::{IsolatedFromAbove, OpFlag, Pure};

use crate::calling_convention::{CALLING_CONVENTION_ATTR, CallingConvention};
use crate::dialect::{adt, tribute_cps};
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

mod asm;
mod callable_origin;
mod control_validation;
mod declarations;
mod evidence;
mod func_sig;
mod handler_binding;
mod interfaces;
mod local_validation;
mod row;
mod row_validation;
mod symbol_validation;
#[cfg(test)]
mod tests;
mod type_validation;
mod validation;
mod verify;

// The modules share their private helpers through these globs; the lists
// below name what the dialect exports.
use asm::*;
use control_validation::*;
use declarations::*;
use func_sig::*;
use local_validation::*;
use row_validation::*;
use symbol_validation::*;
use type_validation::*;
use validation::*;

pub(crate) use asm::{assign_lambda_names, parse_func, parse_lambda, print_func, print_lambda};
pub use declarations::{
    CompilerIntrinsicDeclaration, OperationDeclaration, declare_abilities, declare_operations,
    operation_declarations,
};
pub use evidence::{EVIDENCE_PLAN_ATTR, EvidenceStep};
pub(crate) use evidence::{EvidencePlanSite, verify_evidence_plan};
pub use func_sig::{FuncSig, FuncSigBuilder, FuncSigTypeError};
pub use handler_binding::{BindingHeader, HandleArms, HandlerBinding};
pub use interfaces::{
    DeclarationUse, DynLocalValidation, DynOperationDeclarationUse, DynSourceCallable,
    LocalValidation, OperationDeclarationUse, SourceCallable,
};
pub use local_validation::validate_local;
pub(crate) use local_validation::{
    validate_func, validate_lambda, validate_tail_call, verify_resolved_types,
    verify_row_declarations,
};
pub use type_validation::specialized;
pub use validation::{ValidationError, ValidationResult, validate, validate_whole_ir};

pub(crate) use callable_origin::is_signature_parameter;
use callable_origin::validate_callable_origins;
pub use callable_origin::{CallableArgumentOrigin, CallableQuery, CallableResultOrigin};

pub use row::{
    ABILITY_ROW_ATTR, INFERRED_ROW_ATTR, QUANTIFIED_ROWS_ATTR, ROW_ARGS_ATTR, ROW_BINDERS_ATTR,
    ROW_UNIONS_ATTR, Row, RowArgs, RowBinder, RowUnion, find_substituted_rows,
    logical_tuple_elements, logical_tuple_name, logical_tuple_type, quantified_count,
    quantified_rows, row_args, row_binders, row_unions, substitute_rows, visit_rows,
};

/// Exact semantic identity copied from the trusted compiler-intrinsic registry.
pub const COMPILER_INTRINSIC_ATTR: &str = "tribute.compiler_intrinsic";

/// Whether an ability operation, and the `tribute_control.perform` that
/// invokes it or the `tribute_control.handle` arm that handles it, is a
/// general (`op`) or a tail-resumptive (`fn`) one.
///
/// It is stored as a string attribute holding its [keyword](Self::keyword).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OperationKind {
    /// A general handler arm, which may capture the resumption.
    Op,
    /// A tail-resumptive arm, which resumes automatically.
    Fn,
}

impl OperationKind {
    /// The keyword that spells this kind, both as the stored attribute
    /// value and in the syntax of a handle arm.
    pub fn keyword(self) -> &'static str {
        match self {
            Self::Op => "op",
            Self::Fn => "fn",
        }
    }

    /// The kind a keyword spells; `None` for any other string.
    pub fn from_keyword(word: &str) -> Option<Self> {
        match word {
            "op" => Some(Self::Op),
            "fn" => Some(Self::Fn),
            _ => None,
        }
    }

    fn from_attribute(ctx: &IrContext, value: &Attribute) -> Option<Self> {
        match value {
            Attribute::String(keyword) => Self::from_keyword(ctx.str(*keyword)),
            _ => None,
        }
    }
}

/// `Attr<OperationKind>`: a keyword string, which the schema checks.
impl trunk_ir::attr_kind::AttrKind for OperationKind {
    const KIND: trunk_ir::op_schema::AttributeKind = trunk_ir::op_schema::AttributeKind::String;
    type Out<'ctx> = OperationKind;
    type In = OperationKind;
    const DOMAIN: Option<&'static str> = Some("one of \"op\" and \"fn\"");

    fn accepts(ctx: &IrContext, attr: &Attribute) -> bool {
        Self::from_attribute(ctx, attr).is_some()
    }

    fn read<'ctx>(ctx: &'ctx IrContext, attr: &'ctx Attribute) -> OperationKind {
        Self::from_attribute(ctx, attr).expect("unverified handler kind keyword")
    }

    fn write(ctx: &mut IrContext, value: OperationKind) -> Attribute {
        ctx.string_attr(value.keyword())
    }

    /// A kind is written as its keyword, `op` or `fn`.
    fn print_text(p: &mut trunk_ir::printer::AsmPrinter<'_>, attr: &Attribute) -> fmt::Result {
        fmt::Write::write_str(p, Self::read(p.ctx(), attr).keyword())
    }

    fn parse_text(
        p: &mut trunk_ir::parser::AsmParser<'_>,
    ) -> trunk_ir::parser::ParseResult<Attribute> {
        trunk_ir::asm_format::parse_keyword(p, |ctx, word| {
            let kind = Self::from_keyword(word)?;
            Some(ctx.string_attr(kind.keyword()))
        })
    }
}

/// A source named function in either dialect: `tribute_control.func`, or the
/// `tribute_cps.func` class specialization makes of one.
#[trunk_ir::op_flag]
pub struct SourceFunc;

/// A source lambda in either dialect.
#[trunk_ir::op_flag]
pub struct SourceLambda;

/// A named function reference in either dialect.
#[trunk_ir::op_flag]
pub struct SourceFuncRef;

/// An indirect call in either dialect.
#[trunk_ir::op_flag]
pub struct SourceCallIndirect;

/// An indirect proper tail call in either dialect.
#[trunk_ir::op_flag]
pub struct SourceTailCallIndirect;

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
    /// tribute_control.func @id(%x: T) -> R {
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
    ///   `ContinuationFrame<R>`. It has no convention: class specialization
    ///   replaces each instance with a `tribute_cps.func`.
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
    /// - **Symbol:** a `SymbolDefinition`. A definition with a body and a
    ///   foreign `abi` is not discardable; any other is removed once nothing
    ///   live uses it (`new-plans/ir.md`, "Core Invariants").
    /// - **Signature:** a `SourceCallable` whose signature is `type`.
    /// - **Flags:** `SourceFunc`, shared with `tribute_cps.func`.
    /// - **Location:** the whole span of the source function or extern
    ///   declaration.
    #[verify]
    #[asm_format]
    #[flags(IsolatedFromAbove, SourceFunc)]
    #[implements(
        CallableArgumentOrigin,
        LocalValidation,
        SourceCallable,
        trunk_ir::op_interface::SymbolDefinition
    )]
    fn func<S: FuncSig>(
        sym_name: Attr<String>,
        r#type: Attr<TypeOf<S>>,
        row_binders: Option<Attr<[RowBinder]>>,
        row_unions: Option<Attr<[RowUnion]>>,
        #[default(false)] inferred_row: Attr<bool>,
    ) {
        #[region(body?)]
        {}
    }

    /// A source lambda. Its result is a callable origin, and so is a
    /// parameter whose type is the one its result type declares.
    ///
    /// ```text
    /// %f = tribute_control.lambda(%x: T) -> R captures [%capture0, ...] {
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
    /// - **Signature:** a `SourceCallable` whose signature is its result type.
    /// - **Flags:** `SourceLambda`, shared with `tribute_cps.lambda`.
    /// - **Location:** the whole span of the source lambda.
    #[asm_format]
    #[flags(Pure, SourceLambda)]
    #[implements(
        CallableResultOrigin,
        CallableArgumentOrigin,
        LocalValidation,
        SourceCallable
    )]
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
    ///   substituted (`new-plans/ir.md`, "Row 인자와 치환"). The operand may be
    ///   the `tribute_cps.lambda` class specialization made. The tails and
    ///   instances of the row arguments are checked in the scope of the
    ///   enclosing definition, like any other row.
    /// - **Ownership and value flow:** the operand and result are the same
    ///   callable value; no new value is made. CPS legalization replaces the
    ///   result with the operand.
    /// - **Location:** the source span that refers to the generalized
    ///   binding.
    #[verify]
    #[flags(Pure)]
    #[implements(CallableResultOrigin)]
    #[asm_format(lambda attr_dict_bare ":" type(result))]
    fn instantiate<S: FuncSig>(row_args: Attr<[Row]>, lambda: Value<S>) -> Value<impl FuncSig> {}

    /// A named function as a value. Its result is a callable origin when the
    /// function has a body or is a registered compiler intrinsic.
    ///
    /// ```text
    /// %f = tribute_control.func_ref @id : !Callable
    /// %g = tribute_control.func_ref @map rows [{tail = 0}] : !Callable
    /// ```
    ///
    /// - **Shape:** no operands or regions, and not a terminator.
    ///   `value: Symbol` is required, and the result is one
    ///   `tribute_control.func_sig`. The concise form writes `value`, then
    ///   `row_args` after `rows` when present.
    /// - **Meaning:** makes a named function a first-class source value,
    ///   which source needs because it can use a named function as a
    ///   higher-order value. The result has the target's source signature.
    ///   Its outer row may also be a wider
    ///   row that contains the target's row: this is the subsumption that
    ///   passes a pure function where a callable with an open row is
    ///   expected.
    /// - **Verification:** the local verifier checks the attribute and the
    ///   result shape. Whole-IR verification checks symbol resolution; that
    ///   the result equals the target type with `row_args` substituted
    ///   (`new-plans/ir.md`, "Row 인자와 치환") apart from the outer row; and
    ///   that the outer row holds each instance of the substituted row and,
    ///   if that row has a tail, the same tail.
    /// - **Ownership and value flow:** the result is an ordinary SSA value.
    /// - **Flags:** `SourceFuncRef`, shared with `tribute_cps.func_ref`.
    /// - **Location:** the source reference span that uses the named function
    ///   as a value.
    #[verify]
    #[asm_format(value ("rows" row_args^)? ":" type(result))]
    #[flags(Pure, SourceFuncRef)]
    #[implements(CallableResultOrigin)]
    fn func_ref(value: Attr<SymbolRef>, row_args: Option<Attr<[Row]>>) -> Value<impl FuncSig> {}

    /// A direct call. Its result is a callable origin when the callee has a
    /// body or is a registered compiler intrinsic.
    ///
    /// ```text
    /// %result = tribute_control.call @f(%arg0, ...) : (Arg0, ...) -> Result
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
    #[asm_format(callee "(" args ")" attr_dict ":" functional_type(args, result))]
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
    /// %result = tribute_control.call_indirect %callee(%arg0, ...) : (Arg0, ...) -> Result
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
    /// - **Flags:** `SourceCallIndirect`, shared with `tribute_cps.call_indirect`.
    /// - **Location:** the source indirect-call span, including the callee
    ///   and the arguments.
    #[verify]
    #[asm_format(callee "(" args ")" attr_dict ":" signature(callee))]
    #[flags(SourceCallIndirect)]
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
    /// tribute_control.tail_call @f(%arg0, ...) : (Arg0, ...) -> ()
    /// ```
    ///
    /// - **Shape:** the source arguments in declaration order,
    ///   `callee: Symbol`, and optional `evidence_plan` (`new-plans/ir.md`,
    ///   "Evidence 선택 속성") and `row_args` (`new-plans/ir.md`, "Row 인자와
    ///   치환"). A terminator with no result or region, registered as a
    ///   `CallableExit`.
    /// - **Placement:** it may end the body of a `tribute_control.func` or
    ///   `lambda`, and in place of `scf.yield` it may end an arm of a
    ///   one-result operation that selects one region
    ///   (`DynRegionBranch::selects_one_region`), such as `scf.if` or
    ///   `scf.index_switch`, in tail position. That arm is a terminal
    ///   region, and only the arms that end in `scf.yield` produce the
    ///   operation's result. Such an operation is in tail position when its
    ///   result flows directly into the terminator of the enclosing region:
    ///   the `return` of a callable body, or the `scf.yield` of an outer arm
    ///   that is itself in tail position. It is invalid inside a
    ///   `tribute_control.handle` body or arm.
    /// - **Meaning:** a proper tail call of a named source callable. When the
    ///   call finishes, the callee's result becomes the enclosing callable's
    ///   result, and no caller frame remains.
    /// - **Verification:** the local verifier checks the attributes, the
    ///   placement rule, and that the callee result type equals the enclosing
    ///   callable's result type. Whole-IR verification, as for
    ///   `tribute_control.call`, matches the callee signature with `row_args`
    ///   substituted against the operands and the enclosing callable's
    ///   result. Inside a `tribute_cps` callable it also checks that the
    ///   callee instance's convention is not stronger than that callable's.
    ///   Typechecking owns the source diagnostics for tail position and
    ///   the handler rule; this check confirms that lowering preserved its
    ///   result.
    /// - **Ownership and value flow:** the arguments are ordinary SSA uses,
    ///   and there is no hidden operand. Shared legalization turns it into a
    ///   proper tail transfer for every combination of conventions
    ///   (`new-plans/cps-effects.md`, "Source proper tail call의 적법화").
    /// - **Location:** the source span including the `become` keyword and
    ///   the call.
    #[verify]
    #[implements(CallLike, CallableExit, LocalValidation)]
    #[asm_format(callee "(" args ")" attr_dict ":" functional_type(args))]
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
    /// tribute_control.tail_call_indirect %callee(%arg0, ...) : (Arg0, ...) -> Result
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
    ///   `evidence_plan`. Typechecking owns whether the selection matches the
    ///   ability row.
    /// - **Ownership and value flow:** as for `tribute_control.call_indirect`,
    ///   the environment, evidence, and `ContinuationFrame<R>` are not
    ///   operands.
    /// - **Flags:** `SourceTailCallIndirect`, shared with `tribute_cps.tail_call_indirect`.
    /// - **Location:** the source span including the `become` keyword and
    ///   the indirect call.
    #[verify]
    #[asm_format(callee "(" args ")" attr_dict ":" signature(callee))]
    #[flags(SourceTailCallIndirect)]
    #[implements(CallableExit, LocalValidation)]
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
    #[implements(LocalValidation)]
    #[asm_format(value attr_dict_bare)]
    fn r#return(value: Value<_>) {}

    // Direct-style control operations
    /// An ability operation. Its result is a callable origin when it claims
    /// the kind of the declaration it names and has that declaration's
    /// result type.
    ///
    /// ```text
    /// %result = tribute_control.perform op @State @get(%arg0, ...)
    ///   : (ArgType, ...) -> ResultType
    /// ```
    ///
    /// - **Operands:** zero or more already evaluated source arguments in
    ///   declaration order. The values keep their logical types; conversion
    ///   handles tuple packing and erasure.
    /// - **Results:** exactly one logical operation result. A source `Never`
    ///   result is `core.never`, and does not select a physical `Never`
    ///   control carrier.
    /// - **Attributes:** `ability: SymbolRef`, `op_name: String`, and
    ///   `operation_kind: OperationKind` are required. `operation_kind` is exactly
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
    ///   The `tribute_control.handle` verifier applies the per-kind shape of
    ///   the handler contract to each arm.
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
    #[verify]
    #[implements(OperationDeclarationUse, CallableResultOrigin)]
    #[asm_format(operation_kind ability symbol(op_name) "(" args ")" attr_dict ":" functional_type(args, result))]
    fn perform(
        ability: Attr<SymbolRef>,
        op_name: Attr<String>,
        operation_kind: Attr<OperationKind>,
        args: Variadic<_>,
    ) -> Value<_> {
    }

    /// A handled body. Its result is a callable origin when every value that
    /// becomes it is: the completion region's yield and each general (`op`)
    /// arm's yield. The completion argument is one when the body's yield is.
    /// An arm argument is one when the arm claims the kind of the
    /// declaration it names and the argument has the declaration's
    /// parameter type at its position.
    ///
    /// ```text
    /// %answer = tribute_control.handle : AnswerType {
    ///   ...
    ///   tribute_control.yield %body_value
    /// } do(%completed: BodyType) {
    ///   ...
    ///   tribute_control.yield %answer_value
    /// } op @State @get(%arg: ArgType, %k: tribute_control.resume_token<ResultType, AnswerType>) : ResultType {
    ///   ...
    ///   tribute_control.yield %answer
    /// } fn @State @peek() : ResultType {
    ///   ...
    ///   tribute_control.yield %result
    /// }
    /// ```
    ///
    /// - **Operands:** none. Values the executable regions capture use
    ///   ordinary enclosing SSA visibility.
    /// - **Results:** exactly one logical handle result.
    /// - **Attributes:** `handlers` is required: one `HandlerBinding` per
    ///   arm, in the order of `arms`, each with `ability: SymbolRef`,
    ///   `op_name: String`, `kind: OperationKind`, and
    ///   `operation_result_type: Type`. The optional `evidence_plan`
    ///   (`new-plans/ir.md`, "Evidence 선택 속성") selects the handlers to
    ///   mask in the outer evidence before the body evidence is made. Dynamic
    ///   prompt/owner tags and the physical tail-call representation are
    ///   legalization contracts, not semantic attributes of this operation.
    /// - **Regions:** `body`, `completion`, then one region in `arms` for
    ///   each `handlers` element, in the same order. The format writes the
    ///   completion after `do`, and each arm after its binding's kind,
    ///   ability, and operation name, with its entry block arguments and
    ///   `: operation_result_type`.
    /// - **Block arguments:** `body` is one block with no arguments.
    ///   `completion` is one block with exactly one argument, whose type is
    ///   the type of the value `body` yields. Each arm is one block whose
    ///   arguments are the source operation arguments, in declaration order
    ///   and with their logical types; a resumptive `op` arm has one last
    ///   argument, `resume_token<operation_result_type, handle-result-type>`.
    ///   An `op` arm whose `operation_result_type` is source `Never` has no
    ///   token, and neither does a `fn` arm.
    /// - **Terminators:** every region ends in `tribute_control.yield`.
    /// - **Meaning:** when `body` completes normally, `completion` is
    ///   evaluated exactly once and its value is returned. A general arm
    ///   that completes without resuming returns the arm value as the handle
    ///   result and skips `completion`. A tail-resumptive `fn` arm returns
    ///   the operation result supplied automatically to the suspended
    ///   computation.
    /// - **Verification:** the verifier checks that there is one arm per
    ///   binding and that no `(ability, op_name)` is bound twice. The local
    ///   verifier enforces the single-block shapes, the block-argument
    ///   counts, the terminators, and yield type equality: the result type
    ///   equals the completion yield type and the yield type of every
    ///   general arm, including an `op -> Never` arm, and a `fn` arm yields
    ///   its `operation_result_type`. It checks the token's position and
    ///   parameters, and that an `op -> Never` arm has no token and no
    ///   `tribute_control.resume` anywhere in it, nested regions included;
    ///   the converter checks the same before rewriting and reports a
    ///   violation as a conversion failure. The symbol-aware frontend
    ///   conformance check also confirms that each arm's `kind`, argument
    ///   types, and `operation_result_type` are those of the declaration it
    ///   names. No verifier reclassifies a general `op` from the shape of its
    ///   arm.
    /// - **Ownership and value flow:** this operation owns the delimited
    ///   resumption capability that arises when the body performs a
    ///   resumptive general operation. That capability is exposed only as the
    ///   `resume_token` argument of a resumptive general arm, which is
    ///   affine. It may go unused, dropping the continuation, or have one
    ///   static ownership path to a `tribute_control.resume`. When a closure
    ///   captures it, that static path moves into the closure. A copy, store,
    ///   return, yield, or any other escape is invalid. Static SSA validation
    ///   cannot prove that a capturing closure is called at most once
    ///   dynamically, so the lowered resumption must also enforce one-shot
    ///   consumption at run time and reject or trap on a second call. Values
    ///   leave an executable region only through `tribute_control.yield`.
    /// - **Location:** the whole source `handle` expression. Region and block
    ///   locations use the matching body, completion arm, and handler arm
    ///   spans.
    ///
    /// The frontend always materializes the completion region. When the
    /// source has no `do` arm, the region is the identity on the body result.
    /// This removes an optional structural case from conversion without
    /// changing source semantics.
    ///
    /// [`Handle::build_with_arms`] builds it from each binding with its arm.
    #[asm_format(attr_dict ":" type(result) region(body) "do" args(completion) region(completion) custom(HandleArms(handlers, arms)))]
    #[verify]
    #[implements(
        CallableResultOrigin,
        CallableArgumentOrigin,
        LocalValidation,
        OperationDeclarationUse
    )]
    fn handle(
        handlers: Attr<[HandlerBinding]>,
        evidence_plan: Option<Attr<[EvidenceStep]>>,
    ) -> Value<_> {
        #[region(body)]
        {}
        #[region(completion)]
        {}
        #[regions(arms)]
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
    #[asm_format(resume_token "," value attr_dict_bare ":" type(result))]
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
    /// - **Terminator:** of a `handle` body, completion region, or arm. It
    ///   is invalid anywhere else.
    /// - **Local verification:** enforces its own shape. The owning operation
    ///   checks the placement and the yield type.
    /// - **Ownership and value flow:** passes an ordinary logical value to
    ///   the owning structured operation. A `resume_token` can never be
    ///   yielded.
    /// - **Location:** the source expression that makes the region result. A
    ///   synthesized identity completion uses the owning `handle` location.
    #[implements(LocalValidation)]
    #[asm_format(value attr_dict_bare)]
    fn r#yield(value: Value<_>) {}
}

/// Read the exact input and answer types carried by a resume token.
pub fn resume_token_parts(ctx: &IrContext, ty: TypeRef) -> Option<(TypeRef, TypeRef)> {
    if !ResumeToken::matches(ctx, ty) {
        return None;
    }
    let [input, answer] = ctx.get_type(ty).params.as_array()?;
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

impl Handle {
    /// Build a handle with result type `result` from its body, its
    /// completion, and each binding with its arm, in order, so a binding
    /// cannot drift from its arm.
    pub fn build_with_arms(
        ctx: &mut IrContext,
        location: Location,
        result: TypeRef,
        body: RegionRef,
        completion: RegionRef,
        arms: impl IntoIterator<Item = (HandlerBinding, RegionRef)>,
    ) -> Self {
        let (handlers, arms): (Vec<_>, Vec<_>) = arms.into_iter().unzip();
        Self::operands()
            .handlers(handlers)
            .results(result)
            .regions(body, completion, arms)
            .build(ctx, location)
    }

    /// The binding of each arm with the arm, read fallibly for IR whose
    /// schema may not hold: `None` for a malformed binding.
    pub fn arm_bindings(
        self,
        ctx: &IrContext,
    ) -> impl Iterator<Item = (Option<HandlerBinding>, RegionRef)> + '_ {
        let handlers = match ctx.op_attrs(self.op_ref()).get("handlers") {
            Some(Attribute::List(items)) => items.as_slice(),
            _ => &[],
        };
        ctx.op_regions(self.op_ref())
            .skip(2)
            .enumerate()
            .map(move |(index, arm)| {
                let binding = handlers
                    .get(index)
                    .and_then(|item| HandlerBinding::from_attribute(ctx, item).ok());
                (binding, arm)
            })
    }

    /// The binding of `arm`, when it is one of this handle's arms with a
    /// well-formed binding.
    pub fn binding_of(self, ctx: &IrContext, arm: RegionRef) -> Option<HandlerBinding> {
        self.arm_bindings(ctx)
            .find(|&(_, region)| region == arm)
            .and_then(|(binding, _)| binding)
    }
}

/// The handle `region` is an arm of, with the arm's binding.
pub fn enclosing_arm(ctx: &IrContext, region: RegionRef) -> Option<(Handle, HandlerBinding)> {
    let handle = Handle::from_op(ctx, ctx.region(region).parent_op?).ok()?;
    Some((handle, handle.binding_of(ctx, region)?))
}
