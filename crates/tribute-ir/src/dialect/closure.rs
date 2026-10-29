//! Closure dialect — closures and captures.

use crate::calling_convention::CallingConvention;
use crate::dialect::tribute_rtti::ManagedRefType;
use trunk_ir::attr_kind::{Dict, SymbolRef, Type};
use trunk_ir::dialect::func::FuncSig;
use trunk_ir::op_flag::Pure;
use trunk_ir::op_interface::CallableOwner;

#[trunk_ir::dialect]
mod closure {
    // Types
    /// A closure whose callable identity is `FuncType`.
    ///
    /// `FuncType` is not an entry ABI: a source callable value's is its
    /// converted source signature, with no evidence, continuation frame, or
    /// environment, and a generated continuation's is its `Cps` entry
    /// without the environment (`new-plans/cps-effects.md`, "Closure type의
    /// callable"). The type records no calling convention and no environment
    /// index. A closure is entered through the convention of each transfer,
    /// whose exact signature carries the entry ABI, and that convention alone
    /// places the environment in the entry.
    ///
    /// ```text
    /// closure.closure<func.func_sig<(core.i32) -> core.i32>>
    /// ```
    #[implements(ManagedRefType)]
    struct Closure<FuncType>;

    /// A closure whose convention table holds the entry functions `entries`,
    /// holding `captures` in order.
    ///
    /// - **Shape:** exactly three entries, one per calling convention in the
    ///   order of `CallingConvention::slot` (`Direct`, `EvidenceDirect`,
    ///   `Cps`), and any number of captures.
    /// - **Meaning:** the entry at a convention's slot is the physical
    ///   callable a transfer of that convention enters: a body, an adapter,
    ///   or a trap entry (`new-plans/ir.md`, "Convention entry"). Every entry
    ///   function receives the closure itself as its environment, at the
    ///   environment index of its slot's convention, and reads the captures
    ///   with `unpack`.
    /// - **Lowering:** closure lowering makes it one allocation of the
    ///   convention table reference (`tribute_rt.convention_table`) and the
    ///   captures; see the closure storage layout in `new-plans/ir.md`.
    ///
    /// ```text
    /// %c = closure.new %x, %y {entries = [@f, @trap_e, @trap_c]} : closure.closure<S>
    /// ```
    #[flags(Pure)]
    fn new(entries: Attr<[SymbolRef]>, captures: Variadic<_>) -> Value<_> {}

    /// The entry function at `index` of the convention table of `closure`,
    /// as an indirect callee: the slot of the convention of the transfer
    /// that calls it. Closure lowering introduces it for each closure call
    /// and lowers it in the same step to a read of the closure's convention
    /// table field and `tribute_rt.convention_entry`.
    ///
    /// ```text
    /// %f = closure.entry %c {index = 2} : core.i32
    /// ```
    #[flags(Pure)]
    fn entry(index: Attr<u32>, closure: Value<_>) -> Value<_> {}

    /// The captures of `closure`, read by its entry functions from their
    /// environment parameter. Its results are the closure's capture types.
    #[flags(Pure)]
    fn unpack(closure: Value<_>) -> Variadic<_> {}

    /// A closure with one body per convention entry it implements,
    /// holding `captures`.
    ///
    /// ```text
    /// %f = closure.lambda [%y] direct(%x: core.i32) -> core.i32 { .. }
    /// %g = closure.lambda {adapters = {evidence_direct = @a0}} [%y] direct(%x: core.i32) -> core.i32 {
    ///   ..
    /// } cps(%ev: E, %k: K, %x: core.i32) -> core.i32 {
    ///   ..
    /// } : closure.closure<S>
    /// ```
    ///
    /// - **Shape:** `conventions` lists the conventions of the bodies,
    ///   strictly increasing, and `signatures` the entry signature of each
    ///   body in the same order: its parameters, the hidden ones of its
    ///   convention included, without the environment, and its result. There
    ///   is one body region per convention, whose entry block takes its
    ///   signature's parameters. Each body is written under its
    ///   convention's keyword with the parameters and result of its
    ///   signature; the closure type is omitted when the lambda has one body
    ///   and the type is the closure of its signature.
    /// - **Meaning:** a body is the closure's entry for its convention. The
    ///   closure type's callable is the closure's identity and may differ
    ///   from every signature. `adapters` names, by convention keyword, the
    ///   adapter entry functions of a source lambda's stronger convention
    ///   table slots without a body, and a trap entry fills each other slot
    ///   (`new-plans/ir.md`, "Convention entry"). A generated closure, whose
    ///   closure type's callable is its own entry, has one body and no
    ///   adapters.
    /// - **Verification:** the verifier checks that the conventions are
    ///   strictly increasing, that each signature is a `func.func_sig`, and
    ///   that there is a signature and a body for each convention.
    /// - **Ownership and value flow:** the captures are values of the
    ///   enclosing scope (the lambda is not isolated from above), shared by
    ///   every body. Lambda lifting extracts each body into a top-level
    ///   `func.func` of its convention, which reads the captures from its
    ///   environment parameter, and replaces the lambda with `closure.new`.
    /// - **Location:** the lambda's span.
    #[verify]
    #[asm_format]
    #[implements(CallableOwner)]
    #[flags(Pure)]
    fn lambda(
        signatures: Attr<[Type]>,
        conventions: Attr<[CallingConvention]>,
        adapters: Option<Attr<Dict<SymbolRef>>>,
        captures: Variadic<_>,
    ) -> Value<impl Closure> {
        #[regions(bodies)]
        {}
    }
}

// A closure value points to a reference-counted closure object.
impl ManagedRefType for Closure {}

/// One body of a `closure.lambda`: its convention, entry signature, and
/// region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LambdaEntry {
    pub convention: CallingConvention,
    pub signature: FuncSig,
    pub body: trunk_ir::refs::RegionRef,
}

/// The lists of a `closure.lambda` do not describe its bodies.
#[derive(Clone, Copy, Debug, derive_more::Display, derive_more::Error)]
#[display(
    "closure.lambda needs strictly increasing conventions, and a func.func_sig and a body for each"
)]
pub struct MalformedEntries;

impl Lambda {
    /// The bodies in order, with their conventions and signatures; `None`
    /// when the lists do not describe them.
    pub fn entries(&self, ctx: &trunk_ir::IrContext) -> Option<Vec<LambdaEntry>> {
        use trunk_ir::ops::DialectType;

        let op = self.op_ref();
        let attrs = ctx.op_attrs(op);
        let (
            Some(trunk_ir::Attribute::List(conventions)),
            Some(trunk_ir::Attribute::List(signatures)),
        ) = (attrs.get("conventions"), attrs.get("signatures"))
        else {
            return None;
        };
        if conventions.is_empty()
            || conventions.len() != signatures.len()
            || conventions.len() != ctx.op_region_count(op)
        {
            return None;
        }
        let entries = conventions
            .iter()
            .zip(signatures)
            .zip(ctx.op_regions(op))
            .map(|((convention, signature), body)| {
                let convention = CallingConvention::from_attribute(ctx, convention)?;
                let trunk_ir::Attribute::Type(signature) = signature else {
                    return None;
                };
                let signature = FuncSig::from_type_ref(ctx, *signature)?;
                Some(LambdaEntry {
                    convention,
                    signature,
                    body,
                })
            })
            .collect::<Option<Vec<_>>>()?;
        entries
            .is_sorted_by(|a, b| a.convention < b.convention)
            .then_some(entries)
    }
}

impl trunk_ir::ops::Verify for Lambda {
    fn verify(&self, ctx: &trunk_ir::IrContext) -> Result<(), trunk_ir::ops::VerificationError> {
        match self.entries(ctx) {
            Some(_) => Ok(()),
            None => Err(MalformedEntries.into()),
        }
    }
}

// === Custom assembly format for closure.lambda ===

/// Whether `signature` is spelled by the parameters of `body` and an
/// optional result.
fn spells_signature(
    ctx: &trunk_ir::IrContext,
    signature: FuncSig,
    body: trunk_ir::refs::RegionRef,
) -> bool {
    let Some(&entry) = ctx.region(body).blocks.first() else {
        return false;
    };
    signature.non_reserved_attrs(ctx).next().is_none()
        && signature
            .inputs_with_attrs(ctx)
            .all(|(_, attrs)| attrs.is_empty())
        && signature
            .results_with_attrs(ctx)
            .all(|(_, attrs)| attrs.is_empty())
        && signature.results(ctx).len() <= 1
        && ctx
            .block_args(entry)
            .iter()
            .map(|&arg| ctx.value_ty(arg))
            .eq(signature.inputs(ctx).iter().copied())
}

/// Print closure.lambda with each body under its convention:
/// `%r = closure.lambda {attrs} [%cap0, %cap1] <convention>(%param: type, ...) -> ret { body } ... [: closure]`
///
/// Each body's parameters and result spell its signature, so a signature
/// with attributes of its own, or whose inputs are not its body's
/// parameters, prints generically.
fn print_closure_lambda(
    h: &mut trunk_ir::printer::OpPrintHelper<'_, '_>,
    op: trunk_ir::OpRef,
) -> std::fmt::Result {
    use std::fmt::Write;
    use trunk_ir::ops::{DialectOp, DialectType};

    let ctx = h.ctx();
    let Some(entries) = Lambda::from_op(ctx, op)
        .ok()
        .and_then(|lambda| lambda.entries(ctx))
        .filter(|entries| {
            entries
                .iter()
                .all(|entry| spells_signature(ctx, entry.signature, entry.body))
        })
    else {
        return h.print_generic(op);
    };
    let result_ty = ctx.op_result_types(op)[0];
    let result_data = ctx.get_type(result_ty);
    let closure_of_signature = matches!(entries[..], [entry]
        if result_data.name == Closure::NAME
            && result_data.params.as_slice() == [entry.signature.as_type_ref()]
            && result_data.attrs.is_empty());

    let results = h.ctx().op_results(op);
    let name = h.assign_value_name(results[0]);
    write!(h, "{name} = closure.lambda")?;

    // " {key = val, ...}" — attributes other than the entry lists (if any)
    let attrs: Vec<_> = h
        .ctx()
        .op_attrs(op)
        .iter()
        .filter(|(key, _)| *key != "signatures" && *key != "conventions")
        .map(|(key, val)| (key.to_owned(), val.clone()))
        .collect();
    if !attrs.is_empty() {
        write!(h, " {{")?;
        for (i, (key, val)) in attrs.iter().enumerate() {
            if i > 0 {
                write!(h, ", ")?;
            }
            write!(h, "{key} = ")?;
            h.write_attribute(val)?;
        }
        write!(h, "}}")?;
    }

    // " [%x, %y]" — captures (operands)
    let capture_names: Vec<String> = h
        .ctx()
        .op_operands(op)
        .iter()
        .map(|&v| h.get_value_name(v).to_owned())
        .collect();
    if !capture_names.is_empty() {
        write!(h, " [")?;
        for (i, name) in capture_names.iter().enumerate() {
            if i > 0 {
                write!(h, ", ")?;
            }
            write!(h, "{name}")?;
        }
        write!(h, "]")?;
    }

    // " <convention>(%param: type, ...) -> ret { body }" — entry label elided
    for entry in entries {
        let entry_args = {
            let block = h.ctx().region(entry.body).blocks[0];
            h.ctx().block_args(block).to_vec()
        };
        write!(h, " {}(", entry.convention.keyword())?;
        for (i, &arg) in entry_args.iter().enumerate() {
            if i > 0 {
                write!(h, ", ")?;
            }
            let n = h.assign_value_name(arg);
            let ty = h.ctx().value_ty(arg);
            write!(h, "{n}: ")?;
            h.write_type(ty)?;
        }
        write!(h, ")")?;
        if let Some(return_ty) = entry.signature.single_result(h.ctx()) {
            write!(h, " -> ")?;
            h.write_type(return_ty)?;
        }
        h.print_body(entry.body)?;
    }

    if !closure_of_signature {
        write!(h, " : ")?;
        h.write_type(result_ty)?;
    }
    Ok(())
}

/// Lookahead: does the input start with an attribute dict `{key = ...}`?
///
/// Distinguishes `{key = val}` (attr dict) from `{ ops... }` (body region).
/// Checks for `{` + optional whitespace + identifier + whitespace + `=`.
fn starts_attr_dict(input: &str) -> bool {
    let rest = input.strip_prefix('{').unwrap_or("");
    let rest = rest.trim_start();
    // Must start with an ident char (letter or _)
    let after_ident =
        rest.trim_start_matches(|c: char| c.is_ascii_alphanumeric() || c == '_' || c == '.');
    // Must have consumed at least one ident char, then whitespace + '='
    if after_ident.len() == rest.len() {
        return false; // no ident consumed
    }
    let after_ws = after_ident.trim_start();
    after_ws.starts_with('=')
}

/// Parse closure.lambda custom format back into a RawOperation.
fn parse_closure_lambda<'a>(
    input: &mut &'a str,
    results: Vec<&'a str>,
    sym_name: Option<String>,
) -> winnow::ModalResult<trunk_ir::parser::raw::RawOperation<'a>> {
    use trunk_ir::parser::raw::*;
    use winnow::combinator::{delimited, opt, repeat, separated};
    use winnow::prelude::*;

    // " {key = val, ...}" (optional attributes)
    ws.parse_next(input)?;
    let mut attributes = if starts_attr_dict(input) {
        raw_attr_dict.parse_next(input)?
    } else {
        vec![]
    };
    if attributes
        .iter()
        .any(|(key, _)| key == "signatures" || key == "conventions")
    {
        return Err(winnow::error::ErrMode::Backtrack(
            winnow::error::ContextError::new(),
        ));
    }

    // " [%x, %y]" (optional captures)
    ws.parse_next(input)?;
    let captures: Vec<&'a str> = if input.starts_with('[') {
        delimited(
            ('[', ws),
            separated(0.., (ws, value_ref, ws).map(|(_, v, _)| v), ','),
            (ws, ']'),
        )
        .parse_next(input)?
    } else {
        vec![]
    };

    // " <convention>(%param: type, ...) -> ret { body }", one or more
    let entries: Vec<_> = repeat(
        1..,
        (
            ws,
            ident.try_map(str::parse::<CallingConvention>),
            func_params,
            opt(return_type),
            ws,
            raw_region,
        )
            .map(|(_, convention, params, ret_ty, _, region)| (convention, params, ret_ty, region)),
    )
    .parse_next(input)?;

    // ": closure_type" (optional; the closure of the one signature when omitted)
    ws.parse_next(input)?;
    let closure_ty = opt((':', ws, raw_type).map(|(_, _, ty)| ty)).parse_next(input)?;

    let mut conventions = Vec::new();
    let mut signatures = Vec::new();
    let mut regions = Vec::new();
    for (convention, params, ret_ty, mut region) in entries {
        // Inject formal params into entry block args
        if !params.is_empty() && !region.blocks.is_empty() {
            let mut merged = params.clone();
            merged.append(&mut region.blocks[0].args);
            region.blocks[0].args = merged;
        }
        conventions.push(RawAttribute::String(convention.keyword().to_owned()));
        // The signature is func.func_sig<(param_types...) -> return_ty>.
        signatures.push(trunk_ir::dialect::func::signature_raw_type(
            "func.func_sig",
            params.iter().map(|(_, ty)| ty.clone().into()).collect(),
            ret_ty.into_iter().map(Into::into).collect(),
            vec![],
        ));
        regions.push(region);
    }
    let closure_raw_ty = match (closure_ty, &signatures[..]) {
        (Some(ty), _) => ty,
        (None, [signature]) => RawType::Concrete {
            name: "closure.closure",
            params: vec![signature.clone().into()],
            attrs: vec![],
        },
        (None, _) => {
            return Err(winnow::error::ErrMode::Cut(
                winnow::error::ContextError::new(),
            ));
        }
    };
    attributes.push(("conventions".into(), RawAttribute::List(conventions)));
    attributes.push((
        "signatures".into(),
        RawAttribute::List(signatures.into_iter().map(RawAttribute::Type).collect()),
    ));

    Ok(RawOperation {
        results,
        name: "closure.lambda",
        sym_name,
        has_func_signature: false,
        func_params: vec![],
        return_type: None,
        operands: captures,
        attributes,
        result_types: vec![closure_raw_ty],
        regions,
        successors: vec![],
    })
}

impl trunk_ir::asm_format::OpAsmFormat for Lambda {
    fn print(&self, h: &mut trunk_ir::printer::OpPrintHelper<'_, '_>) -> std::fmt::Result {
        print_closure_lambda(h, self.op_ref())
    }

    fn parse<'a>(
        input: &mut &'a str,
        results: Vec<&'a str>,
        sym_name: Option<String>,
    ) -> winnow::ModalResult<trunk_ir::parser::raw::RawOperation<'a>> {
        parse_closure_lambda(input, results, sym_name)
    }
}

impl trunk_ir::op_interface::CallableOwner for Lambda {
    fn callable_signature(
        &self,
        ctx: &trunk_ir::IrContext,
        body: trunk_ir::refs::RegionRef,
    ) -> Option<FuncSig> {
        self.entries(ctx)?
            .into_iter()
            .find(|entry| entry.body == body)
            .map(|entry| entry.signature)
    }
}
#[cfg(test)]
mod tests {
    use crate::calling_convention::CallingConvention;
    use trunk_ir::Span;
    use trunk_ir::SymbolPath;
    use trunk_ir::op_interface::DynCallableOwner;
    use trunk_ir::ops::DialectOp;
    use trunk_ir::refs::PathRef;
    use trunk_ir::types::Location;
    use trunk_ir::{Attribute, IrContext, TypeDataBuilder};

    fn dummy_location() -> Location {
        Location::new(PathRef::from_u32(0), Span::default())
    }

    fn make_i32_type(ctx: &mut IrContext) -> trunk_ir::TypeRef {
        TypeDataBuilder::new("core.i32").intern(ctx)
    }

    fn make_closure_type(ctx: &mut IrContext) -> trunk_ir::TypeRef {
        TypeDataBuilder::new("closure.closure").intern(ctx)
    }

    #[test]
    fn test_closure_new_round_trip() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let i32_ty = make_i32_type(&mut ctx);
        let closure_ty = make_closure_type(&mut ctx);

        // Create an env value via arith.const
        let env_op = trunk_ir::dialect::arith::Const::operands()
            .value(Attribute::Int(0))
            .results(i32_ty)
            .build(&mut ctx, loc);
        let env_val = env_op.result(&ctx);

        let op = super::New::operands([env_val])
            .entries([SymbolPath::new(["my_func"])])
            .results(closure_ty)
            .build(&mut ctx, loc);

        // Verify from_op round-trip
        let op2 = super::New::from_op(&ctx, op.op_ref()).expect("should match closure.new");
        assert_eq!(op.op_ref(), op2.op_ref());

        itertools::assert_equal(op.entries(&ctx), [&SymbolPath::new(["my_func"])]);

        // Verify result type
        let result = op.result(&ctx);
        assert_eq!(ctx.value_ty(result), closure_ty);
    }

    #[test]
    fn test_closure_entry_round_trip() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let i32_ty = make_i32_type(&mut ctx);
        let closure_ty = make_closure_type(&mut ctx);

        // Create a closure value
        let env_op = trunk_ir::dialect::arith::Const::operands()
            .value(Attribute::Int(0))
            .results(i32_ty)
            .build(&mut ctx, loc);
        let env_val = env_op.result(&ctx);
        let closure_op = super::New::operands([env_val])
            .entries([SymbolPath::new(["f"])])
            .results(closure_ty)
            .build(&mut ctx, loc);
        let closure_val = closure_op.result(&ctx);

        let func_op = super::Entry::operands(closure_val)
            .index(0)
            .results(i32_ty)
            .build(&mut ctx, loc);

        // Verify from_op round-trip
        let func_op2 =
            super::Entry::from_op(&ctx, func_op.op_ref()).expect("should match closure.entry");
        assert_eq!(func_op.op_ref(), func_op2.op_ref());

        // Verify operand
        assert_eq!(func_op.closure(&ctx), closure_val);

        // Verify result type
        let result = func_op.result(&ctx);
        assert_eq!(ctx.value_ty(result), i32_ty);
    }

    #[test]
    fn test_closure_unpack_round_trip() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let i32_ty = make_i32_type(&mut ctx);
        let closure_ty = make_closure_type(&mut ctx);

        let capture_op = trunk_ir::dialect::arith::Const::operands()
            .value(Attribute::Int(0))
            .results(i32_ty)
            .build(&mut ctx, loc);
        let capture = capture_op.result(&ctx);
        let closure_op = super::New::operands([capture, capture])
            .entries([SymbolPath::new(["f"])])
            .results(closure_ty)
            .build(&mut ctx, loc);
        assert_eq!(closure_op.captures(&ctx), [capture, capture]);
        let closure_val = closure_op.result(&ctx);

        let unpack_op = super::Unpack::operands(closure_val)
            .results([i32_ty, i32_ty])
            .build(&mut ctx, loc);

        let unpack_op2 =
            super::Unpack::from_op(&ctx, unpack_op.op_ref()).expect("should match closure.unpack");
        assert_eq!(unpack_op.op_ref(), unpack_op2.op_ref());
        assert_eq!(unpack_op.closure(&ctx), closure_val);
        let results = unpack_op.results(&ctx);
        assert_eq!(results.len(), 2);
        assert_eq!(ctx.value_ty(results[0]), i32_ty);
    }

    #[test]
    fn test_closure_lambda_round_trip() {
        use trunk_ir::context::{BlockArgData, BlockData, RegionData};
        use trunk_ir::dialect::func;

        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let i32_ty = make_i32_type(&mut ctx);
        let closure_ty = make_closure_type(&mut ctx);

        // Create a capture value
        let cap_op = trunk_ir::dialect::arith::Const::operands()
            .value(Attribute::Int(7))
            .results(i32_ty)
            .build(&mut ctx, loc);
        let cap_val = cap_op.result(&ctx);

        // Build body region: ^bb0(%x: i32): func.return %x
        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![BlockArgData {
                ty: i32_ty,
                attrs: Default::default(),
            }],
            ops: Default::default(),
            parent_region: None,
        });
        let x_val = ctx.block_arg(entry, 0);
        let ret_op = func::Return::operands([x_val]).build(&mut ctx, loc);
        ctx.push_op(entry, ret_op.op_ref());

        let body_region = ctx.create_region(RegionData {
            location: loc,
            blocks: trunk_ir::smallvec::smallvec![entry],
            parent_op: None,
        });

        // Create closure.lambda [%cap] { body } -> closure_ty
        let signature = func::func_sig(&mut ctx, [i32_ty], [i32_ty]).as_type_ref();
        let lambda_op = super::Lambda::operands(vec![cap_val])
            .signatures([signature])
            .conventions([CallingConvention::Direct])
            .results(closure_ty)
            .regions([body_region])
            .build(&mut ctx, loc);

        // Verify from_op round-trip
        let lambda_op2 =
            super::Lambda::from_op(&ctx, lambda_op.op_ref()).expect("should match closure.lambda");
        assert_eq!(lambda_op.op_ref(), lambda_op2.op_ref());

        // Verify matches
        assert!(super::Lambda::matches(&ctx, lambda_op.op_ref()));
        assert!(!super::New::matches(&ctx, lambda_op.op_ref()));

        // Verify result type
        let result = lambda_op.result(&ctx);
        assert_eq!(ctx.value_ty(result), closure_ty);

        // Verify body region exists
        let mut bodies = lambda_op.bodies(&ctx);
        let body = bodies.next().expect("the lambda has a body");
        assert_eq!(bodies.next(), None);
        let body_blocks = &ctx.region(body).blocks;
        assert_eq!(body_blocks.len(), 1);
        let body_entry = body_blocks[0];
        assert_eq!(ctx.block(body_entry).args.len(), 1);
        assert_eq!(ctx.block(body_entry).args[0].ty, i32_ty);

        // Verify captures (operands)
        let operands = ctx.op_operands(lambda_op.op_ref());
        assert_eq!(operands.len(), 1);
        assert_eq!(operands[0], cap_val);
    }

    #[test]
    fn test_closure_from_op_wrong_dialect() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let i32_ty = make_i32_type(&mut ctx);

        // Create an arith.const — should not match closure ops
        let c = trunk_ir::dialect::arith::Const::operands()
            .value(Attribute::Int(1))
            .results(i32_ty)
            .build(&mut ctx, loc);
        assert!(super::New::from_op(&ctx, c.op_ref()).is_err());
        assert!(super::Entry::from_op(&ctx, c.op_ref()).is_err());
        assert!(super::Unpack::from_op(&ctx, c.op_ref()).is_err());
    }

    #[test]
    fn test_closure_matches() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let i32_ty = make_i32_type(&mut ctx);
        let closure_ty = make_closure_type(&mut ctx);

        let env_op = trunk_ir::dialect::arith::Const::operands()
            .value(Attribute::Int(0))
            .results(i32_ty)
            .build(&mut ctx, loc);
        let env_val = env_op.result(&ctx);

        let closure_new = super::New::operands([env_val])
            .entries([SymbolPath::new(["f"])])
            .results(closure_ty)
            .build(&mut ctx, loc);

        assert!(super::New::matches(&ctx, closure_new.op_ref()));
        assert!(!super::Entry::matches(&ctx, closure_new.op_ref()));
        assert!(!super::Unpack::matches(&ctx, closure_new.op_ref()));
    }

    // =================================================================
    // Print → Parse → Print round-trip tests for closure.lambda
    // =================================================================

    fn assert_roundtrip(input: &str) {
        let mut ctx = IrContext::new();
        let op = trunk_ir::parser::parse_module(&mut ctx, input).unwrap_or_else(|e| {
            panic!(
                "Parse failed at offset {}: {}\n\nInput:\n{}",
                e.offset, e.message, input
            );
        });
        let printed = trunk_ir::printer::print_module(&ctx, op);

        let mut ctx2 = IrContext::new();
        let op2 = trunk_ir::parser::parse_module(&mut ctx2, &printed).unwrap_or_else(|e| {
            panic!(
                "Round-trip parse failed at offset {}: {}\n\nPrinted:\n{}",
                e.offset, e.message, printed
            );
        });
        let reprinted = trunk_ir::printer::print_module(&ctx2, op2);
        assert_eq!(printed, reprinted, "Round-trip mismatch:\n{printed}");
    }

    #[test]
    fn review_lambda_result_cardinality_roundtrip() {
        for result in [None, Some("core.nil"), Some("core.never"), Some("core.i32")] {
            let arrow = result.map(|ty| format!(" -> {ty}")).unwrap_or_default();
            let text = format!(
                "core.module @test {{ %f = closure.lambda direct(){arrow} {{ func.unreachable }} }}"
            );
            let mut ctx = IrContext::new();
            let module = trunk_ir::parser::parse_test_module(&mut ctx, &text);
            let lambda = module.ops(&ctx)[0];
            let body = ctx.op_region(lambda, 0).expect("lambda body");
            let signature = DynCallableOwner::get(&ctx, lambda)
                .and_then(|owner| owner.callable_signature(&ctx, body))
                .expect("lambda signature");
            assert_eq!(signature.results(&ctx).len(), usize::from(result.is_some()));
            if let Some(name) = result {
                assert_eq!(
                    ctx.get_type(signature.results(&ctx)[0])
                        .name
                        .display(&ctx)
                        .to_string(),
                    name
                );
            }
            let printed = trunk_ir::printer::print_module(&ctx, module.op());
            assert_roundtrip(&printed);
            assert_eq!(printed.contains(" -> "), result.is_some());
        }
    }

    #[test]
    fn a_lambda_whose_closure_is_not_its_signature_prints_the_closure() {
        let text = r#"core.module @test {
  %f = closure.lambda direct(%ev: core.i64, %x: core.i32) -> core.i32 {
    func.return %x
  } : closure.closure<func.func_sig<(core.i32) -> core.i32>>
}"#;
        let mut ctx = IrContext::new();
        let module = trunk_ir::parser::parse_test_module(&mut ctx, text);
        let lambda = super::Lambda::from_op(&ctx, module.ops(&ctx)[0]).unwrap();
        let [entry] = lambda.entries(&ctx).expect("the lists describe the body")[..] else {
            panic!("the lambda has one body");
        };
        let signature = entry.signature;
        assert_eq!(signature.inputs(&ctx).len(), 2);
        let printed = trunk_ir::printer::print_module(&ctx, module.op());
        assert!(printed.contains(" : closure.closure<"), "{printed}");
        assert_roundtrip(&printed);
    }

    #[test]
    fn a_lambda_result_must_be_a_closure() {
        let text = r#"core.module @test {
  %f = closure.lambda direct(%x: core.i32) -> core.i32 {
    func.return %x
  } : core.i32
}"#;
        let mut ctx = IrContext::new();
        let module = trunk_ir::parser::parse_test_module(&mut ctx, text);
        let result = trunk_ir::validation::validate_op_schemas(&ctx, module.op());
        assert!(!result.is_ok(), "{result}");
    }

    #[test]
    fn test_lambda_roundtrip_with_params_and_captures() {
        assert_roundtrip(
            r#"core.module @test {
  func.func @main() -> core.i32 {
    %0 = arith.const {value = 10} : core.i32
    %1 = closure.lambda [%0] direct(%2: core.i32) -> core.i32 {
        %3 = arith.addi %2, %0 : core.i32
        func.return %3
    }
    func.return %1
  }
}"#,
        );
    }

    #[test]
    fn test_lambda_roundtrip_no_captures() {
        assert_roundtrip(
            r#"core.module @test {
  func.func @main() -> core.i32 {
    %0 = closure.lambda direct(%1: core.i32) -> core.i32 {
        func.return %1
    }
    func.return %0
  }
}"#,
        );
    }

    #[test]
    fn test_lambda_roundtrip_no_params() {
        assert_roundtrip(
            r#"core.module @test {
  func.func @main() -> core.i32 {
    %0 = closure.lambda direct() -> core.i32 {
        %1 = arith.const {value = 42} : core.i32
        func.return %1
    }
    func.return %0
  }
}"#,
        );
    }

    #[test]
    fn test_lambda_roundtrip_with_attributes() {
        assert_roundtrip(
            r#"core.module @test {
  func.func @main() -> core.i32 {
    %0 = closure.lambda {tag = 42} direct(%1: core.i32) -> core.i32 {
        func.return %1
    }
    func.return %0
  }
}"#,
        );
    }

    #[test]
    fn a_lambda_writes_each_body_under_its_convention() {
        let text = r#"core.module @test {
  func.func @main(%y: core.i32) {
    %f = closure.lambda {adapters = {evidence_direct = @a0}} [%y] direct(%x: core.i32) -> core.i32 {
      func.return %y
    } cps(%ev: core.i64, %k: core.i64, %x2: core.i32) -> core.i32 {
      func.return %x2
    } : closure.closure<func.func_sig<(core.i32) -> core.i32>>
    func.return
  }
}"#;
        let mut ctx = IrContext::new();
        let module = trunk_ir::parser::parse_test_module(&mut ctx, text);
        let main = trunk_ir::dialect::func::Func::from_op(&ctx, module.ops(&ctx)[0]).unwrap();
        let op = ctx.block(ctx.region(main.body(&ctx)).blocks[0]).ops[0];
        let lambda = super::Lambda::from_op(&ctx, op).unwrap();
        let entries = lambda.entries(&ctx).expect("the lists describe the bodies");
        let conventions: Vec<_> = entries.iter().map(|entry| entry.convention).collect();
        assert_eq!(
            conventions,
            [CallingConvention::Direct, CallingConvention::Cps]
        );
        for entry in &entries {
            let owner = DynCallableOwner::get(&ctx, op).unwrap();
            assert_eq!(
                owner.callable_signature(&ctx, entry.body),
                Some(entry.signature)
            );
        }
        assert!(trunk_ir::validation::validate_all(&ctx, module, &mut Default::default()).is_ok());
        assert_roundtrip(text);
    }

    #[test]
    fn a_lambda_needs_a_signature_per_body_and_increasing_conventions() {
        for (conventions, signatures) in [
            (r#"["cps", "direct"]"#, 2),
            (r#"["direct", "direct"]"#, 2),
            (r#"["direct", "cps"]"#, 1),
            ("[]", 0),
        ] {
            let signature = "func.func_sig<() -> core.i32>";
            let signatures = vec![signature; signatures].join(", ");
            let text = format!(
                r#"core.module @test {{
  %f = closure.lambda direct() -> core.i32 {{
    %0 = arith.const {{value = 0}} : core.i32
    func.return %0
  }} cps() -> core.i32 {{
    %1 = arith.const {{value = 1}} : core.i32
    func.return %1
  }} : closure.closure<{signature}>
}}"#
            );
            let mut ctx = IrContext::new();
            let module = trunk_ir::parser::parse_test_module(&mut ctx, &text);
            let op = module.ops(&ctx)[0];
            let conventions = trunk_ir::parser::parse_test_module(
                &mut ctx,
                &format!(
                    "core.module @a {{ test.op {{conventions = {conventions}, signatures = [{signatures}]}} }}"
                ),
            );
            let source = conventions.ops(&ctx)[0];
            for key in ["conventions", "signatures"] {
                let value = ctx.op_attrs(source).get(key).unwrap().clone();
                ctx.op_attrs_mut(op).insert(key, value);
            }
            let lambda = super::Lambda::from_op(&ctx, op).unwrap();
            assert_eq!(lambda.entries(&ctx), None);
            let result = trunk_ir::validation::validate_operation_verifiers(&ctx, module);
            assert!(!result.is_ok(), "{result}");
        }
    }

    #[test]
    fn test_lambda_roundtrip_multiple_params_and_captures() {
        assert_roundtrip(
            r#"core.module @test {
  func.func @main() -> core.i32 {
    %0 = arith.const {value = 1} : core.i32
    %1 = arith.const {value = 2} : core.i32
    %2 = closure.lambda [%0, %1] direct(%3: core.i32, %4: core.i32) -> core.i32 {
        %5 = arith.addi %3, %4 : core.i32
        %6 = arith.addi %5, %0 : core.i32
        %7 = arith.addi %6, %1 : core.i32
        func.return %7
    }
    func.return %2
  }
}"#,
        );
    }
}

#[cfg(test)]
mod malformed_owner_tests {
    use trunk_ir::{
        dialect::{core, func},
        op_interface::DynCallableOwner,
        ops::DialectOp,
    };
    #[test]
    fn malformed_lambda_owner_never_falls_back_to_outer_function() {
        for transfer in [
            "func.tail_call_indirect %k {signature = func.func_sig<() -> core.never>}",
            "func.tail_call {callee = @runtime}",
        ] {
            let mut ctx = trunk_ir::IrContext::new();
            let input = "core.module @m {
          func.func @host() -> core.never {
            %lambda = closure.lambda direct(%k: func.func_sig<() -> core.never>) -> core.never {
              func.tail_call_indirect %k
            }
            func.unreachable
          }
        }";
            let input = input.replace("func.tail_call_indirect %k", transfer);
            let module = trunk_ir::parser::parse_test_module(&mut ctx, &input);
            assert!(
                trunk_ir::validation::validate_all(&ctx, module, &mut Default::default()).is_ok()
            );
            let host = func::Func::from_op(&ctx, module.ops(&ctx)[0]).unwrap();
            let lambda = ctx.block(ctx.region(host.body(&ctx)).blocks[0]).ops[0];
            assert!(super::Lambda::matches(&ctx, lambda));
            let nil = core::Nil::params().intern(&mut ctx).as_type_ref();
            ctx.op_attrs_mut(lambda).insert(
                "signatures",
                trunk_ir::Attribute::List(vec![trunk_ir::Attribute::Type(nil)]),
            );
            let body = ctx.op_region(lambda, 0).expect("lambda body");
            assert_eq!(
                DynCallableOwner::get(&ctx, lambda)
                    .map(|owner| owner.callable_signature(&ctx, body)),
                Some(None)
            );
            let result = trunk_ir::validation::validate_all(&ctx, module, &mut Default::default());
            assert!(
                result
                    .to_string()
                    .contains("requires registered enclosing callable signature"),
                "{result}"
            );
        }
    }
}

#[cfg(test)]
mod callable_owner_regressions {
    #[test]
    fn normal_validation_checks_nearest_lambda_tail_results() {
        for (callee_result, succeeds) in [("core.never", true), ("core.nil", false)] {
            let mut ctx = trunk_ir::IrContext::new();
            let input = format!(
                "core.module @m {{
              func.func @host() {{
                %outer = closure.lambda direct() -> core.i32 {{
                  %inner = closure.lambda direct(%k: func.func_sig<() -> {callee_result}>) -> core.never {{
                    func.tail_call_indirect %k {{signature = func.func_sig<() -> {callee_result}>}}
                  }}
                  %v = arith.const {{value = 0}} : core.i32
                  func.return %v
                }}
                func.return
              }}
            }}"
            );
            let module = trunk_ir::parser::parse_test_module(&mut ctx, &input);
            let result = trunk_ir::validation::validate_all(&ctx, module, &mut Default::default());
            if succeeds {
                assert!(result.is_ok(), "{result}");
            } else {
                assert!(
                    result
                        .to_string()
                        .contains("tail caller/callee result lists differ"),
                    "{result}"
                );
            }
        }
    }
}
