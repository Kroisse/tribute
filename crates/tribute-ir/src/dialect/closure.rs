//! Closure dialect — closures and captures.

#[trunk_ir::dialect]
mod closure {
    // Types
    struct Closure<FuncType>;

    #[attr(func_ref: Symbol)]
    fn new(env: ()) -> result {}

    fn func(closure: ()) -> result {}

    fn env(closure: ()) -> result {}

    /// High-level lambda: captures + body region → closure value.
    ///
    /// The body region receives block arguments for the lambda's formal parameters.
    /// Captured values are referenced from the parent scope (NOT isolated from above).
    /// A downstream `lower_closure_lambda` pass extracts the body into a top-level
    /// `func.func` and replaces this op with `closure.new`.
    fn lambda(#[rest] captures: ()) -> result {
        #[region(body)]
        {}
    }
}

// === Pure operation registrations ===
// All closure operations are pure

inventory::submit! { trunk_ir::op_interface::PureOps::register("closure", "new") }
inventory::submit! { trunk_ir::op_interface::PureOps::register("closure", "func") }
inventory::submit! { trunk_ir::op_interface::PureOps::register("closure", "env") }
inventory::submit! { trunk_ir::op_interface::PureOps::register("closure", "lambda") }

// === Custom assembly format for closure.lambda ===

/// Print closure.lambda with decomposed signature:
/// `%r = closure.lambda(%param: type, ...) -> ret effects eff [%cap0, %cap1] { body }`
fn print_closure_lambda(
    h: &mut trunk_ir::printer::OpPrintHelper<'_, '_>,
    op: trunk_ir::OpRef,
    indent: usize,
) -> std::fmt::Result {
    use std::fmt::Write;

    let indent_str = " ".repeat(indent);

    // Assign result name
    let results = h.ctx().op_results(op);
    let name = h.assign_value_name(results[0]);
    write!(h, "{indent_str}{name} = closure.lambda")?;

    // Extract region ref
    let region = {
        let data = h.ctx().op(op);
        data.regions[0]
    };

    // "(%param: type, ...)" — entry block args (formal parameters)
    let entry_args: Vec<_> = {
        let entry = h.ctx().region(region).blocks[0];
        h.ctx().block_args(entry).to_vec()
    };

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

    // "-> return_type" and "effects effect_type"
    // Decompose: result type = closure.closure<core.func<ret, params...> {effect = eff}>
    let type_info = {
        let result_ty = h.ctx().op_result_types(op)[0];
        let closure_ty_data = h.ctx().types.get(result_ty);
        if !closure_ty_data.params.is_empty() {
            let func_ty = closure_ty_data.params[0];
            let func_ty_data = h.ctx().types.get(func_ty);
            let return_ty = func_ty_data.params.first().copied();
            let effect_ty = func_ty_data
                .attrs
                .get(&trunk_ir::Symbol::new("effect"))
                .and_then(|a| match a {
                    trunk_ir::Attribute::Type(t) => Some(*t),
                    _ => None,
                });
            (return_ty, effect_ty)
        } else {
            (None, None)
        }
    };

    if let Some(return_ty) = type_info.0 {
        write!(h, " -> ")?;
        h.write_type(return_ty)?;
    }
    if let Some(eff_ty) = type_info.1 {
        write!(h, " effects ")?;
        h.write_type(eff_ty)?;
    }

    // " {key = val, ...}" — attributes (if any)
    let attrs: Vec<(trunk_ir::Symbol, trunk_ir::Attribute)> = h
        .ctx()
        .op(op)
        .attributes
        .iter()
        .map(|(k, v)| (*k, v.clone()))
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
    writeln!(h, " {{")?;
    h.print_region_eliding_entry(region, indent + 2)?;
    writeln!(h, "{indent_str}}}")
}

/// Lookahead: does the input start with an attribute dict `{key = ...}`?
///
/// Distinguishes `{key = val}` (attr dict) from `{ ops... }` (body region).
/// Checks for `{` + optional whitespace + identifier + whitespace + `=`.
fn starts_attr_dict(input: &str) -> bool {
    let rest = input.strip_prefix('{').unwrap_or("");
    let rest = rest.trim_start();
    // Must start with an ident char (letter or _)
    let after_ident = rest.trim_start_matches(|c: char| c.is_ascii_alphanumeric() || c == '_');
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
    use winnow::combinator::{delimited, opt, preceded, separated};
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

    // "effects effect_type" (optional)
    let eff_ty = opt(preceded((ws, "effects", ws), raw_type)).parse_next(input)?;

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

    // Reconstruct closure.closure<core.func<return_ty, param_types...>> type
    let return_raw = ret_ty.unwrap_or(RawType::Concrete {
        dialect: "core",
        name: "nil",
        params: vec![],
        attrs: vec![],
    });

    let param_raw_types: Vec<RawType<'a>> = params.iter().map(|(_, ty)| ty.clone()).collect();

    // core.func<return_ty, param_types...> (+ optional effect attr)
    let mut func_params_list = vec![return_raw];
    func_params_list.extend(param_raw_types);
    let mut func_attrs = vec![];
    if let Some(eff) = eff_ty {
        func_attrs.push(("effect", RawAttribute::Type(eff)));
    }
    let func_raw_ty = RawType::Concrete {
        dialect: "core",
        name: "func",
        params: func_params_list,
        attrs: func_attrs,
    };

    // closure.closure<core.func<...>>
    let closure_raw_ty = RawType::Concrete {
        dialect: "closure",
        name: "closure",
        params: vec![func_raw_ty],
        attrs: vec![],
    };

    Ok(RawOperation {
        results,
        dialect: "closure",
        op_name: "lambda",
        sym_name,
        func_params: vec![],
        return_type: None,
        effect_type: None,
        operands: captures,
        attributes,
        result_types: vec![closure_raw_ty],
        regions,
        successors: vec![],
    })
}

inventory::submit! {
    trunk_ir::op_interface::OpAsmFormat {
        dialect: "closure",
        op_name: "lambda",
        print_fn: print_closure_lambda,
        parse_fn: parse_closure_lambda,
    }
}

#[cfg(test)]
mod tests {
    use trunk_ir::Span;
    use trunk_ir::Symbol;
    use trunk_ir::ops::DialectOp;
    use trunk_ir::refs::PathRef;
    use trunk_ir::types::Location;
    use trunk_ir::{Attribute, IrContext, TypeDataBuilder, TypeInterner};

    fn dummy_location() -> Location {
        Location::new(PathRef::from_u32(0), Span::default())
    }

    fn make_i32_type(types: &mut TypeInterner) -> trunk_ir::TypeRef {
        types.intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build())
    }

    fn make_closure_type(types: &mut TypeInterner) -> trunk_ir::TypeRef {
        types.intern(TypeDataBuilder::new(Symbol::new("closure"), Symbol::new("closure")).build())
    }

    #[test]
    fn test_closure_new_round_trip() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let i32_ty = make_i32_type(&mut ctx.types);
        let closure_ty = make_closure_type(&mut ctx.types);

        // Create an env value via arith.const
        let env_op = trunk_ir::dialect::arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(0));
        let env_val = env_op.result(&ctx);

        // Create closure.new with func_ref attribute
        let op = super::new(&mut ctx, loc, env_val, closure_ty, Symbol::new("my_func"));

        // Verify from_op round-trip
        let op2 = super::New::from_op(&ctx, op.op_ref()).expect("should match closure.new");
        assert_eq!(op.op_ref(), op2.op_ref());

        // Verify func_ref attribute
        assert_eq!(op.func_ref(&ctx), Symbol::new("my_func"));

        // Verify result type
        let result = op.result(&ctx);
        assert_eq!(ctx.value_ty(result), closure_ty);

        // Verify dialect and op name constants
        assert_eq!(super::New::DIALECT_NAME, "closure");
        assert_eq!(super::New::OP_NAME, "new");
    }

    #[test]
    fn test_closure_func_round_trip() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let i32_ty = make_i32_type(&mut ctx.types);
        let closure_ty = make_closure_type(&mut ctx.types);

        // Create a closure value
        let env_op = trunk_ir::dialect::arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(0));
        let env_val = env_op.result(&ctx);
        let closure_op = super::new(&mut ctx, loc, env_val, closure_ty, Symbol::new("f"));
        let closure_val = closure_op.result(&ctx);

        // Create closure.func
        let func_op = super::func(&mut ctx, loc, closure_val, i32_ty);

        // Verify from_op round-trip
        let func_op2 =
            super::Func::from_op(&ctx, func_op.op_ref()).expect("should match closure.func");
        assert_eq!(func_op.op_ref(), func_op2.op_ref());

        // Verify operand
        assert_eq!(func_op.closure(&ctx), closure_val);

        // Verify result type
        let result = func_op.result(&ctx);
        assert_eq!(ctx.value_ty(result), i32_ty);

        // Verify dialect and op name constants
        assert_eq!(super::Func::DIALECT_NAME, "closure");
        assert_eq!(super::Func::OP_NAME, "func");
    }

    #[test]
    fn test_closure_env_round_trip() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let i32_ty = make_i32_type(&mut ctx.types);
        let closure_ty = make_closure_type(&mut ctx.types);

        // Create a closure value
        let env_op = trunk_ir::dialect::arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(0));
        let env_val = env_op.result(&ctx);
        let closure_op = super::new(&mut ctx, loc, env_val, closure_ty, Symbol::new("f"));
        let closure_val = closure_op.result(&ctx);

        // Create closure.env
        let env_result_op = super::env(&mut ctx, loc, closure_val, i32_ty);

        // Verify from_op round-trip
        let env_result_op2 =
            super::Env::from_op(&ctx, env_result_op.op_ref()).expect("should match closure.env");
        assert_eq!(env_result_op.op_ref(), env_result_op2.op_ref());

        // Verify operand
        assert_eq!(env_result_op.closure(&ctx), closure_val);

        // Verify result type
        let result = env_result_op.result(&ctx);
        assert_eq!(ctx.value_ty(result), i32_ty);

        // Verify dialect and op name constants
        assert_eq!(super::Env::DIALECT_NAME, "closure");
        assert_eq!(super::Env::OP_NAME, "env");
    }

    #[test]
    fn test_closure_lambda_round_trip() {
        use trunk_ir::context::{BlockArgData, BlockData, RegionData};
        use trunk_ir::dialect::func as arena_func;

        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let i32_ty = make_i32_type(&mut ctx.types);
        let closure_ty = make_closure_type(&mut ctx.types);

        // Create a capture value
        let cap_op = trunk_ir::dialect::arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(7));
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
        let ret_op = arena_func::r#return(&mut ctx, loc, [x_val]);
        ctx.push_op(entry, ret_op.op_ref());

        let body_region = ctx.create_region(RegionData {
            location: loc,
            blocks: trunk_ir::smallvec::smallvec![entry],
            parent_op: None,
        });

        // Create closure.lambda [%cap] { body } -> closure_ty
        let lambda_op = super::lambda(&mut ctx, loc, vec![cap_val], closure_ty, body_region);

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

        // Verify dialect and op name constants
        assert_eq!(super::Lambda::DIALECT_NAME, "closure");
        assert_eq!(super::Lambda::OP_NAME, "lambda");
    }

    #[test]
    fn test_closure_from_op_wrong_dialect() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let i32_ty = make_i32_type(&mut ctx.types);

        // Create an arith.const — should not match closure ops
        let c = trunk_ir::dialect::arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(1));
        assert!(super::New::from_op(&ctx, c.op_ref()).is_err());
        assert!(super::Func::from_op(&ctx, c.op_ref()).is_err());
        assert!(super::Env::from_op(&ctx, c.op_ref()).is_err());
    }

    #[test]
    fn test_closure_matches() {
        let mut ctx = IrContext::new();
        let loc = dummy_location();
        let i32_ty = make_i32_type(&mut ctx.types);
        let closure_ty = make_closure_type(&mut ctx.types);

        let env_op = trunk_ir::dialect::arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(0));
        let env_val = env_op.result(&ctx);

        let closure_new = super::new(&mut ctx, loc, env_val, closure_ty, Symbol::new("f"));

        assert!(super::New::matches(&ctx, closure_new.op_ref()));
        assert!(!super::Func::matches(&ctx, closure_new.op_ref()));
        assert!(!super::Env::matches(&ctx, closure_new.op_ref()));
    }

    #[test]
    fn test_dialect_name_function() {
        assert_eq!(super::DIALECT_NAME(), Symbol::new("closure"));
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
    fn test_lambda_roundtrip_with_params_and_captures() {
        assert_roundtrip(
            r#"core.module @test {
  func.func @main() -> core.i32 {
    %0 = arith.const {value = 10} : core.i32
    %1 = closure.lambda(%2: core.i32) -> core.i32 effects core.effect_row() {tail_var_id = 42} [%0] {
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
