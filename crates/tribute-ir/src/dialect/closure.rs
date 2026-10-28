//! Closure dialect — closures and captures.

use crate::calling_convention::ConventionBearing;
use crate::dialect::tribute_rtti::ManagedRefType;
use trunk_ir::attr_kind::SymbolRef;
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

    /// High-level lambda: captures + body region → closure value.
    ///
    /// `signature` is the entry signature without the environment: the
    /// body's parameters, hidden ones of its convention included, and its
    /// result. The closure type's callable is the closure's identity and may
    /// differ from it. The body region receives block arguments for the
    /// signature's parameters.
    /// Captured values are referenced from the parent scope (NOT isolated from above).
    /// A downstream `lower_closure_lambda` pass extracts the body into a top-level
    /// `func.func` and replaces this op with `closure.new`. Its calling
    /// convention is recorded through `ConventionBearing`.
    ///
    /// ```text
    /// %f = closure.lambda(%ev: E, %x: core.i32) -> core.i32 : closure.closure<S> [%y] { .. }
    /// ```
    ///
    /// The parameters and result spell `signature`. The closure type is
    /// omitted when it is the closure of `signature`.
    #[asm_format]
    #[implements(CallableOwner, ConventionBearing)]
    #[flags(Pure)]
    fn lambda<S: FuncSig>(
        signature: Attr<TypeOf<S>>,
        captures: Variadic<_>,
    ) -> Value<impl Closure> {
        #[region(body)]
        {}
    }
}

// A closure value points to a reference-counted closure object.
impl ManagedRefType for Closure {}

// === Custom assembly format for closure.lambda ===

/// Print closure.lambda with decomposed signature:
/// `%r = closure.lambda(%param: type, ...) -> ret [: closure] [%cap0, %cap1] { body }`
///
/// The parameters and result spell `signature`, so a signature with
/// attributes of its own, or whose inputs are not the body's parameters,
/// prints generically.
fn print_closure_lambda(
    h: &mut trunk_ir::printer::OpPrintHelper<'_, '_>,
    op: trunk_ir::OpRef,
) -> std::fmt::Result {
    use std::fmt::Write;
    use trunk_ir::ops::DialectType;

    let Some(region) = h.ctx().op_region(op, 0) else {
        return h.print_generic(op);
    };
    let entry_args: Vec<_> = {
        let entry = h.ctx().region(region).blocks[0];
        h.ctx().block_args(entry).to_vec()
    };
    let Some(signature) = h
        .ctx()
        .op_attrs(op)
        .get_type("signature")
        .and_then(|ty| FuncSig::from_type_ref(h.ctx(), ty))
    else {
        return h.print_generic(op);
    };
    let ctx = h.ctx();
    let spelled = signature.non_reserved_attrs(ctx).next().is_none()
        && signature
            .inputs_with_attrs(ctx)
            .all(|(_, attrs)| attrs.is_empty())
        && signature
            .results_with_attrs(ctx)
            .all(|(_, attrs)| attrs.is_empty())
        && signature.results(ctx).len() <= 1
        && entry_args
            .iter()
            .map(|&arg| ctx.value_ty(arg))
            .eq(signature.inputs(ctx).iter().copied());
    if !spelled {
        return h.print_generic(op);
    }
    let result_ty = ctx.op_result_types(op)[0];
    let result_data = ctx.get_type(result_ty);
    let closure_of_signature = result_data.name == Closure::NAME
        && result_data.params.as_slice() == [signature.as_type_ref()]
        && result_data.attrs.is_empty();
    let return_ty = signature.single_result(ctx);

    let results = h.ctx().op_results(op);
    let name = h.assign_value_name(results[0]);
    write!(h, "{name} = closure.lambda")?;

    // "(%param: type, ...)" — entry block args (formal parameters)
    write!(h, "(")?;
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

    if let Some(return_ty) = return_ty {
        write!(h, " -> ")?;
        h.write_type(return_ty)?;
    }
    if !closure_of_signature {
        write!(h, " : ")?;
        h.write_type(result_ty)?;
    }

    // " {key = val, ...}" — attributes other than the signature (if any)
    let attrs: Vec<_> = h
        .ctx()
        .op_attrs(op)
        .iter()
        .filter(|(key, _)| *key != "signature")
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

    // " { body }" — entry label elided
    h.print_body(region)
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
    use winnow::combinator::{delimited, opt, separated};
    use winnow::prelude::*;

    // "(%param: type, ...)" (optional, may be just "()")
    ws.parse_next(input)?;
    let params = if input.starts_with('(') {
        func_params.parse_next(input)?
    } else {
        vec![]
    };

    // "-> return_type" (optional)
    let ret_ty = opt(return_type).parse_next(input)?;

    // ": closure_type" (optional; the closure of the signature when omitted)
    ws.parse_next(input)?;
    let closure_ty = opt((':', ws, raw_type).map(|(_, _, ty)| ty)).parse_next(input)?;

    // " {key = val, ...}" (optional attributes — distinguished from body by lookahead)
    ws.parse_next(input)?;
    let attributes = if starts_attr_dict(input) {
        raw_attr_dict.parse_next(input)?
    } else {
        vec![]
    };

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

    // "{ body }"
    ws.parse_next(input)?;
    let mut regions = Vec::new();
    if input.starts_with('{') {
        let mut region = raw_region.parse_next(input)?;

        // Inject formal params into entry block args
        if !params.is_empty() && !region.blocks.is_empty() {
            let mut merged = params.clone();
            merged.append(&mut region.blocks[0].args);
            region.blocks[0].args = merged;
        }

        regions.push(region);
    }

    // The signature is func.func_sig<(param_types...) -> return_ty>.
    let signature = trunk_ir::dialect::func::signature_raw_type(
        "func.func_sig",
        params.iter().map(|(_, ty)| ty.clone().into()).collect(),
        ret_ty.into_iter().map(Into::into).collect(),
        vec![],
    );
    let closure_raw_ty = closure_ty.unwrap_or_else(|| RawType::Concrete {
        name: "closure.closure",
        params: vec![signature.clone().into()],
        attrs: vec![],
    });
    let mut attributes = attributes;
    attributes.push(("signature".into(), RawAttribute::Type(signature)));

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
    ) -> Option<trunk_ir::dialect::func::FuncSig> {
        use trunk_ir::ops::DialectType;
        let signature = ctx.op_attrs(self.op_ref()).get_type("signature")?;
        FuncSig::from_type_ref(ctx, signature)
    }
}
#[cfg(test)]
mod tests {
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
            .signature(signature)
            .results(closure_ty)
            .regions(body_region)
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
        let body = lambda_op.body(&ctx);
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
                "core.module @test {{ %f = closure.lambda(){arrow} {{ func.unreachable }} }}"
            );
            let mut ctx = IrContext::new();
            let module = trunk_ir::parser::parse_test_module(&mut ctx, &text);
            let lambda = module.ops(&ctx)[0];
            let signature = DynCallableOwner::get(&ctx, lambda)
                .and_then(|owner| owner.callable_signature(&ctx))
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
        use trunk_ir::ops::DialectType;
        let text = r#"core.module @test {
  %f = closure.lambda(%ev: core.i64, %x: core.i32) -> core.i32 : closure.closure<func.func_sig<(core.i32) -> core.i32>> {
    func.return %x
  }
}"#;
        let mut ctx = IrContext::new();
        let module = trunk_ir::parser::parse_test_module(&mut ctx, text);
        let lambda = super::Lambda::from_op(&ctx, module.ops(&ctx)[0]).unwrap();
        let signature =
            trunk_ir::dialect::func::FuncSig::from_type_ref(&ctx, lambda.signature(&ctx))
                .expect("the parameters and result spell the signature");
        assert_eq!(signature.inputs(&ctx).len(), 2);
        let printed = trunk_ir::printer::print_module(&ctx, module.op());
        assert!(printed.contains(" : closure.closure<"), "{printed}");
        assert_roundtrip(&printed);
    }

    #[test]
    fn a_lambda_result_must_be_a_closure() {
        let text = r#"core.module @test {
  %f = closure.lambda(%x: core.i32) -> core.i32 : core.i32 {
    func.return %x
  }
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
    %1 = closure.lambda(%2: core.i32) -> core.i32 [%0] {
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
    %0 = closure.lambda(%1: core.i32) -> core.i32 {
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
    %0 = closure.lambda() -> core.i32 {
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
    %0 = closure.lambda(%1: core.i32) -> core.i32 {tag = 42} {
        func.return %1
    }
    func.return %0
  }
}"#,
        );
    }

    #[test]
    fn test_lambda_roundtrip_multiple_params_and_captures() {
        assert_roundtrip(
            r#"core.module @test {
  func.func @main() -> core.i32 {
    %0 = arith.const {value = 1} : core.i32
    %1 = arith.const {value = 2} : core.i32
    %2 = closure.lambda(%3: core.i32, %4: core.i32) -> core.i32 [%0, %1] {
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
            %lambda = closure.lambda(%k: func.func_sig<() -> core.never>) -> core.never [] {
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
            ctx.op_attrs_mut(lambda)
                .insert("signature", trunk_ir::Attribute::Type(nil));
            assert_eq!(
                DynCallableOwner::get(&ctx, lambda).map(|owner| owner.callable_signature(&ctx)),
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
                %outer = closure.lambda() -> core.i32 [] {{
                  %inner = closure.lambda(%k: func.func_sig<() -> {callee_result}>) -> core.never [] {{
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
