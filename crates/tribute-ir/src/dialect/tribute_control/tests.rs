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

    let lambda_ty = FuncSig::params([], i32_ty).intern(&mut ctx).as_type_ref();
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
  !callable = tribute_control.func_sig<(core.i32) -> core.i32>

  tribute_control.func @id(%value: core.i32) -> core.i32 {
    tribute_control.return %value
  }
  tribute_control.func @control(%value: core.i32) -> core.i32 {
    %function = "tribute_control.func_ref"() {func_ref = @id} : !callable
    %direct = tribute_control.call %value {callee = @id} : core.i32
    %indirect = "tribute_control.call_indirect"(%function, %value) : core.i32
    %closure = tribute_control.lambda() -> core.i32 captures [%value] {
      tribute_control.return %value
    }
    %handled = "tribute_control.handle"() ({
      %performed = tribute_control.perform %value {ability = @State, op_name = "get", operation_kind = "op"} : core.i32
      tribute_control.yield %performed
    }, {
      ^completion(%completed: core.i32):
        tribute_control.yield %completed
    }, {
      "tribute_control.handler"() ({
        ^handler(%argument: core.i32, %token: tribute_control.resume_token<core.i32, core.i32>):
          %resumed = tribute_control.resume %token, %argument : core.i32
          tribute_control.yield %resumed
      }) {ability = @State, kind = "op", op_name = "get", operation_result_type = core.i32}
    }) : core.i32
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
  tribute_control.func @run() -> core.i32 {
    %value = tribute_control.perform {ability = @State, op_name = "get", operation_kind = "op"} : core.i32
    tribute_control.return %value
  }
  "ability.decl"() ({
    ability.operation {kind = "op", op_name = "get", param_types = [], result_type = core.i32}
  }) {sym_name = "State"}
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
        "  \"ability.decl\"",
        r#"  "ability.decl"() ({
    ability.operation {kind = "op", op_name = "get", param_types = [], result_type = core.i32}
  }) {sym_name = "State"}
  "ability.decl""#,
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
fn callable_and_resume_token_accessors_are_exact() {
    let mut ctx = IrContext::new();
    let i32_ty = core::I32::type_ref(&mut ctx);

    let ty = FuncSig::params([i32_ty], i32_ty).intern(&mut ctx);
    assert_eq!(ty.result(&ctx), i32_ty);
    assert_eq!(ty.inputs(&ctx), &[i32_ty]);

    let token = ResumeToken::params(i32_ty, i32_ty).intern(&mut ctx);
    assert_eq!(token.input(&ctx), i32_ty);
    assert_eq!(token.answer(&ctx), i32_ty);
}

#[test]
fn callable_wrapper_rejects_missing_result_component() {
    let (ctx, module) = parse_fixture(
        r#"core.module @test {
  !malformed = tribute_control.func_sig<() -> ()>
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
            .build(&mut ctx),
        TypeDataBuilder::new("tribute_control.func_sig")
            .param(i32_ty)
            .attr(
                trunk_ir::dialect::func::NUM_INPUTS_ATTR,
                ctx.string_attr("bad"),
            )
            .attr(trunk_ir::dialect::func::NUM_RESULTS_ATTR, Attribute::Int(1))
            .build(&mut ctx),
        TypeDataBuilder::new("tribute_control.func_sig")
            .param(i32_ty)
            .attr(trunk_ir::dialect::func::NUM_INPUTS_ATTR, Attribute::Int(2))
            .attr(trunk_ir::dialect::func::NUM_RESULTS_ATTR, Attribute::Int(1))
            .build(&mut ctx),
        TypeDataBuilder::new("tribute_control.func_sig")
            .param(i32_ty)
            .attr(
                trunk_ir::dialect::func::NUM_INPUTS_ATTR,
                Attribute::Int(i128::from(u32::MAX) + 1),
            )
            .attr(trunk_ir::dialect::func::NUM_RESULTS_ATTR, Attribute::Int(1))
            .build(&mut ctx),
        TypeDataBuilder::new("tribute_control.func_sig")
            .param(i32_ty)
            .attr(trunk_ir::dialect::func::NUM_INPUTS_ATTR, Attribute::Int(0))
            .attr(trunk_ir::dialect::func::NUM_RESULTS_ATTR, Attribute::Int(2))
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
  !inner = tribute_control.func_sig<(core.i32) -> core.i32>
  !shared = tribute_control.func_sig<(core.i32, core.i32) -> core.i32, {metadata = [[!inner, @signature]]}>
  !lambda = tribute_control.func_sig<(core.i32, core.i32) -> core.i32, {metadata = [[!inner, @lambda]]}>
  !distinct = tribute_control.func_sig<(core.i32, core.i32) -> core.i32, {metadata = [[!inner, @distinct]]}>
  "tribute_control.func"() {metadata = @declaration, sym_name = "decl", type = !shared, visibility = @private}
  "tribute_control.func"() ({
    ^bb0(%left: core.i32, %right: core.i32):
      tribute_control.return %left
  }) {metadata = @definition, sym_name = "definition", type = !shared, visibility = @private}
  "tribute_control.func"() {sym_name = "different", type = !distinct}
  tribute_control.func @identity(%value: core.i32) -> core.i32 {
    tribute_control.return %value
  }
  tribute_control.func @outer(%first: core.i32, %second: core.i32) -> core.i32 {
    %captured = "tribute_control.lambda"(%first, %second) ({
      ^bb0(%left: core.i32, %right: core.i32):
        tribute_control.return %first
    }) {debug_name = "apply", inline_hint = true, metadata = @operation} : !lambda
    %empty = tribute_control.lambda() -> core.i32 captures [] {
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
        printed.contains("%2 = \"tribute_control.lambda\"(%0, %1)"),
        "generic fallback must not consume an extra lambda result name: {printed}"
    );
    assert!(printed.contains("-> core.i32 captures []"), "{printed}");
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
  "tribute_control.func"() {sym_name = "inline", type = tribute_control.func_sig<(core.i32) -> core.i32, {metadata = @inline}>}
}"#;
    let (inline_ctx, inline_module) = parse_fixture(inline);
    let inline_printed = assert_round_trip(&inline_ctx, inline_module);
    assert!(
        inline_printed.contains(
            "type = tribute_control.func_sig<(core.i32) -> core.i32, {metadata = @inline}>"
        ),
        "single-use attributed signature must stay inline: {inline_printed}"
    );
}

#[test]
fn malformed_func_sig_storage_uses_generic_assembly_without_loss() {
    let input = r#"core.module @test {
  "tribute_control.func"() {sym_name = "missing"}
  "tribute_control.func"() {sym_name = "broken", type = tribute_control.func_sig<core.i32, {num_inputs = 2, num_results = 1}>}
  %lambda = "tribute_control.lambda"() : tribute_control.func_sig<core.i32, {num_inputs = 2, num_results = 1}>
}"#;
    let (ctx, module) = parse_fixture(input);
    let printed = assert_round_trip(&ctx, module);

    assert!(
        printed.contains("\"tribute_control.func\"() {sym_name = \"missing\"}"),
        "missing source signatures must use generic assembly: {printed}"
    );
    assert!(
        printed.contains(
            "!t0 = tribute_control.func_sig<core.i32, {num_inputs = 2, num_results = 1}>"
        ),
        "malformed signature type/count storage must remain intact: {printed}"
    );
    assert!(
        printed.contains("\"tribute_control.func\"() {sym_name = \"broken\", type = !t0}"),
        "malformed function signature must remain attached to the function: {printed}"
    );
    assert!(
        printed.contains("%0 = \"tribute_control.lambda\"() : !t0"),
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
}

#[test]
fn parser_rejects_conventions_on_control_callables() {
    for input in [
        r#"core.module @test {
  tribute_control.func @bad() -> core.i32 convention(direct)
}"#,
        r#"core.module @test {
  %0 = tribute_control.lambda() -> core.i32 convention(cps) captures [] {
    %1 = arith.const {value = 1} : core.i32
    tribute_control.return %1
  }
}"#,
        r#"core.module @test {
  tribute_cps.func @bad() -> core.i32 convention(fast)
}"#,
        r#"core.module @test {
  tribute_cps.func @bad() -> core.i32 convention(direct) convention(cps)
}"#,
        r#"core.module @test {
  tribute_control.func @bad() -> core.i32 attributes {tribute.calling_convention = "direct"}
}"#,
        r#"core.module @test {
  %0 = tribute_control.lambda() -> core.i32 captures [] attributes {tribute.calling_convention = "cps"} {
    %1 = arith.const {value = 1} : core.i32
    tribute_control.return %1
  }
}"#,
        r#"core.module @test {
  tribute_control.func @bad() -> core.i32 signature_attributes {metadata = @retired}
}"#,
        r#"core.module @test {
  %0 = tribute_control.lambda() -> core.i32 signature_attributes {metadata = @retired} captures [] {
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
  tribute_control.func @ok() -> core.i32 attributes {tribute_calling_convention = 7}
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
  %call = tribute_control.call {callee = @missing, tribute.calling_convention = "direct"} : core.i32
}"#,
    );
    let result = validate_local(&ctx, module);
    let messages = messages(&result);
    assert!(messages.contains("malformed tribute_control.func_sig: Convention"));
    assert!(messages.contains("belongs only on physical operations"));
}

#[test]
fn validator_rejects_wrong_required_attribute_types() {
    let (mut ctx, module) = parse_fixture(
        r#"core.module @test {
  !callable = tribute_control.func_sig<() -> core.i32>
  tribute_control.func @malformed() -> core.i32
  %ref = "tribute_control.func_ref"() {func_ref = 1} : !callable
  %call = tribute_control.call {callee = 1} : core.i32
  %perform = tribute_control.perform {ability = 1, op_name = 2, operation_kind = 3} : core.i32
  "tribute_control.handler"() ({
    %constant = arith.const {value = 0} : core.i32
    tribute_control.yield %constant
  }) {ability = 1, op_name = 2, kind = 3, operation_result_type = 4}
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
                            .contains("operand and result types must be resolved")
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
  %handled = "tribute_control.handle"() ({
    %body = arith.const {value = 0} : core.i32
    tribute_control.yield %body
  }, {
    ^completion(%value: core.i32):
      tribute_control.yield %value
  }, {
    ^handlers:
  }) : core.i32
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
  %handled = "tribute_control.handle"() ({
    %body = arith.const {value = 0} : core.i32
    tribute_control.yield %body
  }, {
    ^completion(%value: core.i32):
      tribute_control.yield %value
  }, {
    ^handlers(%unexpected: core.i32):
  }) : core.i32
}"#,
    );

    let result = validate_local(&ctx, module);
    assert!(messages(&result).contains("handlers table block must not have arguments"));
}

#[test]
fn validator_rejects_structure_arity_type_and_terminator_errors() {
    let (ctx, module) = parse_fixture(
        r#"core.module @test {
  %handled = "tribute_control.handle"() ({
  }, {
  }) : core.i32
  tribute_control.func @unterminated(%value: core.i32) -> core.i32 {
    %unused = arith.const {value = 0} : core.i32
  }
  %indirect = "tribute_control.call_indirect"() : core.i32
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
  !direct = tribute_control.func_sig<(core.i32) -> core.i32>
  !token = tribute_control.resume_token<core.i32, core.i32>
  %integer = arith.const {value = 1} : core.i32
  %boolean = arith.const {value = true} : core.i1

  tribute_control.func @bad_func() -> core.i32 {
    tribute_control.return %integer
  }
  %bad_lambda = tribute_control.lambda() -> core.i32 captures [%integer] {
    tribute_control.return %integer
  }
  %bad_ref = "tribute_control.func_ref"() {func_ref = @bad_func, unexpected = 1} : !direct
  %bad_call0, %bad_call1 = "tribute_control.call"() ({
  }) {unexpected = 1} : core.i32, core.i1
  %non_callable = "tribute_control.call_indirect"(%integer, %boolean) : core.i32
  %callable = "tribute_control.func_ref"() {func_ref = @bad_func} : !direct
  %mismatched = "tribute_control.call_indirect"(%callable, %boolean) : core.i1
  %bad_perform = tribute_control.perform %integer {ability = @State, op_name = "get", operation_kind = "bogus"} : core.i32
  "tribute_control.handler"() ({
    ^clause(%argument: core.i32):
      tribute_control.yield %argument
  }) {ability = @State, op_name = "get", kind = "bogus", operation_result_type = core.i32}
  %bad_resume = tribute_control.resume %integer, %boolean : core.i1
  %token = "test.token"() : !token
  %mismatched_resume = tribute_control.resume %token, %boolean : core.i1
  "tribute_control.unknown"()
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
        .find(|op| ctx.op(*op).name.is(&ctx, "arith.const") && ctx.op_result_types(*op) == [i32_ty])
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
        &["operand #0 `resume_token`: expected T: tribute_control.resume_token, found core.i32"],
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
  %handled = "tribute_control.handle"() ({
    ^body(%unexpected: core.i32):
      %body_value = arith.const {value = 0} : core.i32
      tribute_control.yield %body_value
  }, {
    ^completion(%value: core.i1, %extra: core.i32):
      tribute_control.yield %value
  }, {
    %not_a_handler = arith.const {value = 0} : core.i32
    "tribute_control.handler"() ({
      ^clause(%argument: core.i32, %token: tribute_control.resume_token<core.i1, core.i1>):
        %wrong = arith.const {value = false} : core.i1
        tribute_control.yield %wrong
    }) {ability = @State, kind = "op", op_name = "get", operation_result_type = core.i32}
  }) : core.i32
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
  %handled = "tribute_control.handle"() ({
    %body_value = arith.const {value = 0} : core.i32
    tribute_control.yield %body_value
  }, {
    ^completion(%value: core.i32):
      tribute_control.yield %value
  }, {
    "tribute_control.handler"() ({
      ^clause(%argument: core.i32, %token: tribute_control.resume_token<core.i32, core.i32>):
        tribute_control.yield %argument
    }) {ability = @State, kind = "fn", op_name = "get", operation_result_type = core.i1}
  }) : core.i32
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
        let malformed_handle = trunk_ir::OperationDataBuilder::new(loc, "tribute_control.handle")
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
  tribute_control.func @outer(%external: core.i32) -> core.i32 {
    %closure = tribute_control.lambda() -> core.i32 captures [] {
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
  tribute_control.func @outer(%external: core.i32) -> core.i32 {
    %closure = tribute_control.lambda() -> core.i32 captures [%external] {
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
    !callable = tribute_control.func_sig<(core.i32) -> core.i32>
    tribute_control.func @id(%value: core.i32) -> core.i32 {
      tribute_control.return %value
    }
    tribute_control.func @use(%value: core.i32) -> core.i32 {
      %reference = "tribute_control.func_ref"() {func_ref = @id} : !callable
      %direct = tribute_control.call %value {callee = @id} : core.i32
      %indirect = "tribute_control.call_indirect"(%reference, %direct) : core.i32
      tribute_control.return %indirect
    }
  }
  core.module @booleans {
    !callable = tribute_control.func_sig<(core.i1) -> core.i1>
    tribute_control.func @id(%value: core.i1) -> core.i1 {
      tribute_control.return %value
    }
    tribute_control.func @use(%value: core.i1) -> core.i1 {
      %reference = "tribute_control.func_ref"() {func_ref = @id} : !callable
      %direct = tribute_control.call %value {callee = @id} : core.i1
      %indirect = "tribute_control.call_indirect"(%reference, %direct) : core.i1
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
  tribute_control.func @id(%value: core.i32) -> core.i32 {
    tribute_control.return %value
  }
  core.module @integers {
    tribute_control.func @use(%value: core.i32) -> core.i32 {
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
  !direct = tribute_control.func_sig<(core.i32) -> core.i32>
  !cps = tribute_control.func_sig<(core.i32) -> core.i32>
  !different = tribute_control.func_sig<() -> core.i1>
  !token = tribute_control.resume_token<core.i32, core.i32>

  tribute_control.func @id(%value: core.i32) -> core.i32
  tribute_control.func @id(%value: core.i32) -> core.i32
  tribute_control.func @cps_target(%value: core.i32) -> core.i32 {
    tribute_control.return %value
  }
  %weak = "tribute_control.func_ref"() {func_ref = @cps_target} : !direct
  %wrong_signature = "tribute_control.func_ref"() {func_ref = @id} : !different
  %missing_ref = "tribute_control.func_ref"() {func_ref = @missing} : !direct
  %missing_call = tribute_control.call {callee = @missing} : core.i32
  %false = arith.const {value = false} : core.i1
  %bad_call = tribute_control.call %false {callee = @id} : core.i1
  %unknown_perform = tribute_control.perform {ability = @State, op_name = "missing", operation_kind = "op"} : core.i32
  %bad_perform = tribute_control.perform %false {ability = @State, op_name = "get", operation_kind = "fn"} : core.i1
  %duplicate_capture = tribute_control.lambda() -> core.i1 captures [%false, %false] {
    tribute_control.return %false
  }
  %token_result = "test.token_result"() : !token
  tribute_control.func @token_param(%token_arg: !token) -> core.i32 {
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
  %handled = "tribute_control.handle"() ({
    %input = arith.const {value = 0} : core.i32
    %performed = tribute_control.perform %input {ability = @State, op_name = "get", operation_kind = "op"} : core.i32
    tribute_control.yield %performed
  }, {
    ^completion(%value: core.i32):
      tribute_control.yield %value
  }, {
    "tribute_control.handler"() ({
      ^first(%argument: core.i32, %token: tribute_control.resume_token<core.i32, core.i32>):
        %resumed = tribute_control.resume %token, %argument : core.i32
        tribute_control.yield %resumed
    }) {ability = @State, kind = "op", op_name = "get", operation_result_type = core.i32}
    "tribute_control.handler"() ({
      ^duplicate(%argument: core.i32, %token: tribute_control.resume_token<core.i32, core.i32>):
        %resumed = tribute_control.resume %token, %argument : core.i32
        tribute_control.yield %resumed
    }) {ability = @State, kind = "op", op_name = "get", operation_result_type = core.i32}
  }) : core.i32
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
  %handled = "tribute_control.handle"() ({
    %body = arith.const {value = 0} : core.i32
    tribute_control.yield %body
  }, {
    ^completion(%value: core.i32):
      tribute_control.yield %value
  }, {
    "tribute_control.handler"() ({
      ^clause(%argument: core.i32, %token: tribute_control.resume_token<core.i32, core.i32>):
        %first = tribute_control.resume %token, %argument : core.i32
        %second = tribute_control.resume %token, %argument : core.i32
        tribute_control.yield %first
    }) {ability = @State, kind = "op", op_name = "get", operation_result_type = core.i32}
  }) : core.i32
}"#,
    );
    let result = validate_whole_ir(&ctx, module, &[]);
    assert!(messages(&result).contains("more than one tribute_control.resume"));
}

#[test]
fn whole_ir_rejects_branching_and_escape_at_each_affine_carrier() {
    let (ctx, module) = parse_fixture(
        r#"core.module @test {
  %handled = "tribute_control.handle"() ({
    %body = arith.const {value = 0} : core.i32
    tribute_control.yield %body
  }, {
    ^completion(%value: core.i32):
      tribute_control.yield %value
  }, {
    "tribute_control.handler"() ({
      ^clause(%argument: core.i32, %token: tribute_control.resume_token<core.i32, core.i32>):
        %inner = tribute_control.lambda() -> core.i32 captures [%token, %argument] {
          %resumed = tribute_control.resume %token, %argument : core.i32
          tribute_control.return %resumed
        }
        %wrapper = tribute_control.lambda() -> core.i32 captures [%inner] {
          %called = "tribute_control.call_indirect"(%inner) : core.i32
          tribute_control.return %called
        }
        %escape = tribute_control.call %wrapper {callee = @id} : core.i32
        %first_capture = tribute_control.lambda() -> core.i32 captures [%wrapper] {
          %called = "tribute_control.call_indirect"(%wrapper) : core.i32
          tribute_control.return %called
        }
        %second_capture = tribute_control.lambda() -> core.i32 captures [%wrapper] {
          %called = "tribute_control.call_indirect"(%wrapper) : core.i32
          tribute_control.return %called
        }
        %first = "tribute_control.call_indirect"(%wrapper) : core.i32
        %second = "tribute_control.call_indirect"(%wrapper) : core.i32
        tribute_control.yield %first
    }) {ability = @State, kind = "op", op_name = "get", operation_result_type = core.i32}
  }) : core.i32
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
  %handled = "tribute_control.handle"() ({
    %body = arith.const {value = 0} : core.i32
    tribute_control.yield %body
  }, {
    ^completion(%value: core.i32):
      tribute_control.yield %value
  }, {
    "tribute_control.handler"() ({
      ^clause(%argument: core.i32, %token: tribute_control.resume_token<core.i32, core.i32>):
        %first = tribute_control.lambda() -> core.i32 captures [%token, %argument] {
          %resumed = tribute_control.resume %token, %argument : core.i32
          tribute_control.return %resumed
        }
        %second = tribute_control.lambda() -> core.i32 captures [%token, %argument] {
          %resumed = tribute_control.resume %token, %argument : core.i32
          tribute_control.return %resumed
        }
        tribute_control.yield %argument
    }) {ability = @State, kind = "op", op_name = "get", operation_result_type = core.i32}
  }) : core.i32
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
  %handled = "tribute_control.handle"() ({
    %body = arith.const {value = 0} : core.i32
    tribute_control.yield %body
  }, {
    ^completion(%value: core.i32):
      tribute_control.yield %value
  }, {
    "tribute_control.handler"() ({
      ^clause(%argument: core.i32, %token: tribute_control.resume_token<core.i32, core.i32>):
        %flag = arith.const {value = true} : core.i1
        %value = scf.if %flag -> (core.i32) {
          %left = tribute_control.lambda() -> core.i32 captures [%token, %argument] {
            %resumed = tribute_control.resume %token, %argument : core.i32
            tribute_control.return %resumed
          }
          %called = "tribute_control.call_indirect"(%left) : core.i32
          scf.yield %called
        } else {
          %right = tribute_control.lambda() -> core.i32 captures [%token, %argument] {
            %resumed = tribute_control.resume %token, %argument : core.i32
            tribute_control.return %resumed
          }
          %called = "tribute_control.call_indirect"(%right) : core.i32
          scf.yield %called
        }
        tribute_control.yield %value
    }) {ability = @State, kind = "op", op_name = "get", operation_result_type = core.i32}
  }) : core.i32
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
  %handled = "tribute_control.handle"() ({
    %body = arith.const {value = 0} : core.i32
    tribute_control.yield %body
  }, {
    ^completion(%value: core.i32):
      tribute_control.yield %value
  }, {
    "tribute_control.handler"() ({
      ^clause(%argument: core.i32, %token: tribute_control.resume_token<core.i32, core.i32>):
        %choice = arith.const {value = 0} : core.i32
        "scf.switch"(%choice) ({
          "scf.case"() ({
            %left = tribute_control.lambda() -> core.i32 captures [%token, %argument] {
              %left_resumed = tribute_control.resume %token, %argument : core.i32
              tribute_control.return %left_resumed
            }
            %left_called = "tribute_control.call_indirect"(%left) : core.i32
            scf.yield
          }) {value = 0}
          "scf.case"() ({
            %right = tribute_control.lambda() -> core.i32 captures [%token, %argument] {
              %right_resumed = tribute_control.resume %token, %argument : core.i32
              tribute_control.return %right_resumed
            }
            %right_called = "tribute_control.call_indirect"(%right) : core.i32
            scf.yield
          }) {value = 1}
        })
        tribute_control.yield %argument
    }) {ability = @State, kind = "op", op_name = "get", operation_result_type = core.i32}
  }) : core.i32
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
  %handled = "tribute_control.handle"() ({
    %body = arith.const {value = 0} : core.i32
    tribute_control.yield %body
  }, {
    ^completion(%value: core.i32):
      tribute_control.yield %value
  }, {
    "tribute_control.handler"() ({
      ^clause(%argument: core.i32, %token: tribute_control.resume_token<core.i32, core.i32>):
        %continuation = tribute_control.lambda() -> core.i32 captures [%token, %argument] {
          %resumed = tribute_control.resume %token, %argument : core.i32
          tribute_control.return %resumed
        }
        %first = "tribute_control.call_indirect"(%continuation) : core.i32
        %second = "tribute_control.call_indirect"(%continuation) : core.i32
        tribute_control.yield %first
    }) {ability = @State, kind = "op", op_name = "get", operation_result_type = core.i32}
  }) : core.i32
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
  %handled = "tribute_control.handle"() ({
    %body = arith.const {value = 0} : core.i32
    tribute_control.yield %body
  }, {
    ^completion(%value: core.i32):
      tribute_control.yield %value
  }, {
    "tribute_control.handler"() ({
      ^clause(%argument: core.i32, %token: tribute_control.resume_token<core.i32, core.i32>):
        tribute_control.yield %token
    }) {ability = @State, kind = "op", op_name = "get", operation_result_type = core.i32}
  }) : core.i32
}"#,
    );
    let local = validate_local(&ctx, module);
    assert!(messages(&local).contains("must not yield a resume token"));
    let whole = validate_whole_ir(&ctx, module, &[]);
    assert!(messages(&whole).contains("forbidden use"));

    let (ctx, module) = parse_fixture(
        r#"core.module @test {
  %handled = "tribute_control.handle"() ({
    %body = arith.const {value = 0} : core.i32
    tribute_control.yield %body
  }, {
    ^completion(%value: core.i32):
      tribute_control.yield %value
  }, {
    "tribute_control.handler"() ({
      ^clause(%argument: core.i32, %token: tribute_control.resume_token<core.i32, core.i32>):
        %resumed = tribute_control.resume %token, %argument : core.i32
        tribute_control.yield %resumed
    }) {ability = @State, kind = "op", op_name = "get", operation_result_type = core.never}
  }) : core.i32
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
  "tribute_control.handler"() ({
    ^clause(%value: core.i32, %token: !malformed):
      %resumed = tribute_control.resume %token, %value : core.i32
      tribute_control.yield %resumed
  }) {ability = @State, kind = "op", op_name = "get", operation_result_type = core.i32}
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
  %handled = "tribute_control.handle"() ({
    %body = arith.const {value = 0} : core.i32
    tribute_control.yield %body
  }, {
    ^completion(%value: core.i32):
      tribute_control.yield %value
  }, {
    "tribute_control.handler"() ({
      ^clause(%argument: core.i32, %token: tribute_control.resume_token<core.i32, core.i32>):
        %inner = tribute_control.lambda() -> core.i32 captures [%token, %argument] {
          %resumed = tribute_control.resume %token, %argument : core.i32
          tribute_control.return %resumed
        }
        %wrapper = tribute_control.lambda() -> core.i32 captures [%inner] {
          %called = "tribute_control.call_indirect"(%inner) : core.i32
          tribute_control.return %called
        }
        %first = "tribute_control.call_indirect"(%wrapper) : core.i32
        %second = "tribute_control.call_indirect"(%wrapper) : core.i32
        tribute_control.yield %first
    }) {ability = @State, kind = "op", op_name = "get", operation_result_type = core.i32}
  }) : core.i32
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
  %outer = "tribute_control.handle"() ({
    %body = arith.const {value = 0} : core.i32
    tribute_control.yield %body
  }, {
    ^completion(%value: core.i32):
      tribute_control.yield %value
  }, {
    "tribute_control.handler"() ({
      ^outer_clause(%argument: core.i32, %outer_token: tribute_control.resume_token<core.i32, core.i32>):
        %nested = "tribute_control.handle"() ({
          tribute_control.yield %argument
        }, {
          ^nested_completion(%value: core.i32):
            tribute_control.yield %value
        }, {
          "tribute_control.handler"() ({
            ^inner_clause(%inner_argument: core.i32):
              %crossed = tribute_control.resume %outer_token, %inner_argument : core.i32
              tribute_control.yield %crossed
          }) {ability = @State, kind = "fn", op_name = "get", operation_result_type = core.i32}
        }) : core.i32
        tribute_control.yield %nested
    }) {ability = @State, kind = "op", op_name = "get", operation_result_type = core.i32}
  }) : core.i32
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
  tribute_control.func @f(%n: core.i32) -> core.i32 {{
    tribute_control.return %n
  }}
  tribute_control.func @g(%n: core.i32, %flag: core.i1) -> core.i32 {{
{body}
  }}
}}"#
        ));
        messages(&validate_local(&ctx, module))
    };

    let tail_if = validate_body(
        r#"    %value = scf.if %flag -> (core.i32) {
      tribute_control.tail_call %n {callee = @f}
    } else {
      scf.yield %n
    }
    tribute_control.return %value"#,
    );
    assert!(!tail_if.contains("tail position"), "{tail_if}");

    let resultless_if = validate_body(
        r#"    scf.if %flag {
      tribute_control.tail_call %n {callee = @f}
    } else {
      scf.yield
    }
    tribute_control.return %n"#,
    );
    assert!(resultless_if.contains("tail position"), "{resultless_if}");

    let loop_arm = validate_body(
        r#"    %value = "scf.loop"(%n) ({
    ^bb0(%i: core.i32):
      tribute_control.tail_call %i {callee = @f}
    }) : core.i32
    tribute_control.return %value"#,
    );
    assert!(loop_arm.contains("tail position"), "{loop_arm}");
}

#[test]
fn local_validator_rejects_external_reference_in_isolated_func() {
    let (ctx, module) = parse_fixture(
        r#"core.module @test {
  %external = arith.const {value = 1} : core.i32
  tribute_control.func @invalid() -> core.i32 {
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
  tribute_control.func @"Nat::+"(%left: core.i32, %right: core.i32) -> core.i32
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
  tribute_cps.func @read(%value: core.i32) -> core.i32 convention(cps)
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
  !F = tribute_control.func_sig<(core.i32, core.i32) -> core.i32>
  tribute_control.func @"Nat::+"(%left: core.i32, %right: core.i32) -> core.i32
    attributes {abi = "intrinsic", tribute.compiler_intrinsic = "Nat::+"}
  tribute_control.func @caller(%left: core.i32, %right: core.i32) -> core.i32 {
    %callee = "tribute_control.func_ref"() {func_ref = @"Nat::+"} : !F
    %result = "tribute_control.call_indirect"(%callee, %left, %right) : core.i32
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
  tribute_control.func @"Int::+"(%left: core.i32, %right: core.i32) -> core.i32
    attributes {abi = "intrinsic", tribute.compiler_intrinsic = "Int::+"}
  tribute_control.func @"Nat::+"(%left: core.i32, %right: core.i32) -> core.i32
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
  tribute_control.func @unnamed(%value: !Unnamed) -> !Unnamed {
    tribute_control.return %value
  }
  tribute_control.func @parameterized(%value: !Parameterized) -> !Parameterized {
    tribute_control.return %value
  }
  tribute_control.func @missing(%value: !Missing) -> !Missing {
    tribute_control.return %value
  }
  tribute_control.func @wrong_return(%integer: core.i32, %flag: core.i1) -> core.i32 {
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
  tribute_control.func @managed(%value: !R) -> !R {
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
  tribute_control.func @user_read_line() -> !ReadLineResultRef
    attributes {abi = "intrinsic"}
  tribute_control.func @masquerade(%raw: core.ptr) -> !ReadLineResultRef {
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
  tribute_control.func @private_helper(%value: !Container) -> core.i32
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
  tribute_control.func @primitive(%buffer: core.ptr, %length: core.i32) -> core.ptr
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
  tribute_control.func @incompatible(%value: !RS) -> !RT {
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
  tribute_control.func @caller(%raw: core.ptr, %value: core.i32) -> core.i32 {
    %callee = core.unrealized_conversion_cast %raw : tribute_control.func_sig<(core.i32) -> core.i32>
    %result = "tribute_control.call_indirect"(%callee, %value) : core.i32
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
  !F = tribute_control.func_sig<(core.i32) -> core.i32>
  tribute_control.func @caller(%condition: core.i1, %value: core.i32) -> core.i32 {
    %result = scf.if %condition -> (core.i32) {
      ^then(%callee: !F):
        %called = "tribute_control.call_indirect"(%callee, %value) : core.i32
        scf.yield %called
    } else {
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
  !F = tribute_control.func_sig<(core.i32) -> core.i32>
  tribute_control.func @id(%value: core.i32) -> core.i32 {
    tribute_control.return %value
  }
  tribute_control.func @caller(%value: core.i32) -> !F {
    %callback = "tribute_control.handle"() ({
      %source = "tribute_control.func_ref"() {func_ref = @id} : !F
      tribute_control.yield %source
    }, {
      ^completion(%callback: !F):
        %ignored = "tribute_control.call_indirect"(%callback, %value) : core.i32
        tribute_control.yield %callback
    }, {
      ^handlers:
    }) : !F
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
  !F = tribute_control.func_sig<(core.i32) -> core.i32>
  tribute_control.func @caller(%value: core.i32) -> core.i32 {
    %handled = "tribute_control.handle"() ({
      tribute_control.yield %value
    }, {
      ^completion(%completed: core.i32):
        tribute_control.yield %completed
    }, {
      "tribute_control.handler"() ({
        ^handler(%callback: !F):
          %called = "tribute_control.call_indirect"(%callback, %value) : core.i32
          tribute_control.yield %called
      }) {ability = @Callback, kind = "fn", op_name = "apply", operation_result_type = core.i32}
    }) : core.i32
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
  !F = tribute_control.func_sig<(core.i32) -> core.i32>
  tribute_control.func @run(%value: core.i32) -> core.i32 {{
    %callee = tribute_control.perform {{ability = @Reader, op_name = "read", operation_kind = "fn"}} : !F
    %result = "tribute_control.call_indirect"(%callee, %value) : core.i32
    tribute_control.return %result
  }}
  "ability.decl"() ({{
    ability.operation {{kind = "fn", op_name = "read", param_types = [], result_type = {declared_result}}}
  }}) {{sym_name = "Reader"}}
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
  !F = tribute_control.func_sig<(core.i32) -> core.i32>
  tribute_control.func @inc(%n: core.i32) -> core.i32 {{
    tribute_control.return %n
  }}
  %f = "tribute_control.handle"() ({{
    %body = tribute_control.perform {{ability = @Gen, op_name = "next", operation_kind = "op"}} : core.i32
    tribute_control.yield %body
  }}, {{
    ^completion(%k: core.i32):
      %done = "tribute_control.func_ref"() {{func_ref = @inc}} : !F
      tribute_control.yield %done
  }}, {{
    "tribute_control.handler"() ({{
      ^next(%token: tribute_control.resume_token<core.i32, !F>):
        %resumed_value = arith.const {{value = 40}} : core.i32
        %g = tribute_control.resume %token, %resumed_value : !F
        %called = "tribute_control.call_indirect"(%g, %resumed_value) : core.i32
        %zero = arith.const {{value = 0}} : core.i32
        %bad = core.unrealized_conversion_cast %zero : !F
        tribute_control.yield {op_yield}
    }}) {{ability = @Gen, kind = "op", op_name = "next", operation_result_type = core.i32}}
    "tribute_control.handler"() ({{
      ^ask:
        %answer = arith.const {{value = 2}} : core.i32
        tribute_control.yield %answer
    }}) {{ability = @Ask, kind = "fn", op_name = "ask", operation_result_type = core.i32}}
  }}) : !F
  %one = arith.const {{value = 1}} : core.i32
  %result = "tribute_control.call_indirect"(%f, %one) : core.i32
  "ability.decl"() ({{
    ability.operation {{kind = "op", op_name = "next", param_types = [], result_type = core.i32}}
  }}) {{sym_name = "Gen"}}
  "ability.decl"() ({{
    ability.operation {{kind = "fn", op_name = "ask", param_types = [], result_type = core.i32}}
  }}) {{sym_name = "Ask"}}
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
  !F = tribute_control.func_sig<(core.i32) -> core.i32>
  !Token = tribute_control.resume_token<core.i32, !F>
  tribute_control.func @caller(%token: !Token, %raw: core.ptr, %value: core.i32) -> core.i32 {
    %from_parameter = tribute_control.resume %token, %value : !F
    %first = "tribute_control.call_indirect"(%from_parameter, %value) : core.i32
    %cast_token = core.unrealized_conversion_cast %raw : !Token
    %from_cast = tribute_control.resume %cast_token, %first : !F
    %second = "tribute_control.call_indirect"(%from_cast, %first) : core.i32
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
  !F = tribute_control.func_sig<(core.i32) -> core.i32>
  !G = tribute_control.func_sig<(!F, core.i32) -> core.i32>
  !H = tribute_control.func_sig<(core.i32) -> !F>
  tribute_control.func @caller(%list: core.ptr, %make: !H, %value: core.i32) -> core.i32 {
    %apply = "tribute_control.lambda"() ({
      ^bb0(%callback: !F, %argument: core.i32):
        %applied = "tribute_control.call_indirect"(%callback, %argument) : core.i32
        tribute_control.return %applied
    }) : !G
    %made = "tribute_control.call_indirect"(%make, %value) : !F
    %first = "tribute_control.call_indirect"(%made, %value) : core.i32
    %head = list.head %list {element_type = !F} : !F
    %second = "tribute_control.call_indirect"(%head, %first) : core.i32
    %third = "tribute_control.call_indirect"(%apply, %head, %second) : core.i32
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
    text.push_str("  !F0 = tribute_control.func_sig<(core.i32) -> core.i32>\n");
    for depth in 1..=CHAIN {
        text.push_str(&format!(
            "  !F{depth} = tribute_control.func_sig<(core.i32) -> !F{}>\n",
            depth - 1
        ));
    }
    text.push_str(&format!(
        "  tribute_control.func @caller(%v{CHAIN}: !F{CHAIN}, %x: core.i32) -> core.i32 {{\n"
    ));
    for depth in (0..CHAIN).rev() {
        text.push_str(&format!(
            "    %v{depth} = \"tribute_control.call_indirect\"(%v{}, %x) : !F{depth}\n",
            depth + 1
        ));
    }
    for call in 0..CALLS {
        text.push_str(&format!(
            "    %r{call} = \"tribute_control.call_indirect\"(%v0, %x) : core.i32\n"
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
  !F = tribute_control.func_sig<(core.i32) -> core.i32>
  %f = "tribute_control.handle"() ({
    %body = tribute_control.perform {ability = @Gen, op_name = "next", operation_kind = "op"} : core.i32
    tribute_control.yield %body
  }, {
    ^completion(%k: core.i32):
      %done = core.unrealized_conversion_cast %k : !F
      tribute_control.yield %done
  }, {
    "tribute_control.handler"() ({
      ^next(%token: tribute_control.resume_token<core.i32, !F>):
        %zero = arith.const {value = 0} : core.i32
        %bad = core.unrealized_conversion_cast %zero : !F
        tribute_control.yield %bad
    }) {ability = @Gen, kind = "op", op_name = "next", operation_result_type = core.i32}
  }) : !F
  %one = arith.const {value = 1} : core.i32
  %result = "tribute_control.call_indirect"(%f, %one) : core.i32
  "ability.decl"() ({
    ability.operation {kind = "op", op_name = "next", param_types = [], result_type = core.i32}
  }) {sym_name = "Gen"}
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
  !F = tribute_control.func_sig<(core.i32) -> core.i32>
  !Tuple = adt.struct<Tuple(callee: !F)>
  !TupleRef = adt.typeref<{name = "Tuple"}>
  !Choice = adt.enum<Choice { Some(!F) }>
  !ChoiceRef = adt.typeref<{name = "Choice"}>
  tribute_control.func @caller(%tuple: !TupleRef, %choice: !ChoiceRef, %value: core.i32) -> core.i32 {
    %tuple_callee = adt.struct_get %tuple {type = !Tuple, field = 0} : !F
    %tuple_result = "tribute_control.call_indirect"(%tuple_callee, %value) : core.i32
    %some = adt.variant_cast %choice {type = !Choice, tag = "Some"} : !Choice
    %choice_callee = adt.variant_get %some {type = !Choice, tag = "Some", field = 0} : !F
    %choice_result = "tribute_control.call_indirect"(%choice_callee, %tuple_result) : core.i32
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
  !F = tribute_control.func_sig<(core.i32) -> core.i32>
  !Tuple = adt.struct<Tuple(not_callable: core.i32)>
  !TupleRef = adt.typeref<{name = "Tuple"}>
  !Choice = adt.enum<Choice { Some(core.i32) }>
  !ChoiceRef = adt.typeref<{name = "Choice"}>
  tribute_control.func @caller(%tuple: !TupleRef, %choice: !ChoiceRef, %value: core.i32) -> core.i32 {
    %tuple_callee = adt.struct_get %tuple {type = !Tuple, field = 0} : !F
    %tuple_result = "tribute_control.call_indirect"(%tuple_callee, %value) : core.i32
    %some = adt.variant_cast %choice {type = !Choice, tag = "Some"} : !Choice
    %choice_callee = adt.variant_get %some {type = !Choice, tag = "Some", field = 0} : !F
    %choice_result = "tribute_control.call_indirect"(%choice_callee, %tuple_result) : core.i32
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
  !F = tribute_control.func_sig<(core.i32) -> core.i32>
  !Canonical = adt.struct<Tuple(value: core.i32)>
  !Spoofed = adt.struct<Tuple(callee: !F)>
  !TupleRef = adt.typeref<{name = "Tuple"}>
  tribute_control.func @caller(%tuple: !TupleRef, %value: core.i32) -> core.i32 {
    %canonical = adt.struct_new %value {type = !Canonical} : !TupleRef
    %callee = adt.struct_get %tuple {type = !Spoofed, field = 0} : !F
    %result = "tribute_control.call_indirect"(%callee, %value) : core.i32
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
  tribute_control.func @caller(%value: core.i32) -> core.i32 {
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
  !F = tribute_control.func_sig<(core.i32) -> core.i32>
  tribute_control.func @factory() -> !F attributes {abi = "C"}
  tribute_control.func @caller(%value: core.i32) -> core.i32 {
    %callee = tribute_control.call {callee = @factory} : !F
    %result = "tribute_control.call_indirect"(%callee, %value) : core.i32
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
    tribute_control.func @loop(%value: core.i32) -> core.i32 {
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
            "(%function, %value) : core.i32",
            "(%function, %value) {evidence_plan = [{dup = @State}]} : core.i32",
        )
        .replace(
            "    }) : core.i32\n    tribute_control.return %handled",
            &format!(
                "    }}) {{evidence_plan = {handle}}} : core.i32\n    tribute_control.return %handled"
            ),
        )
        .replace(
            "(%token, %argument) : core.i32",
            "(%token, %argument) {evidence_plan = [{mask = @Log}]} : core.i32",
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
