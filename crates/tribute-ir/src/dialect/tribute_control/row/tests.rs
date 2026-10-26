use trunk_ir::ops::DialectType;
use trunk_ir::parser::parse_module;
use trunk_ir::printer::print_module;
use trunk_ir::rewrite::Module;
use trunk_ir::types::TypeDataBuilder;
use trunk_ir::{IrContext, SymbolPath};

use super::super::{
    CallingConvention, FuncSig, ValidationResult, conventions_assigned, func_sig_convention,
    validate_local, validate_whole_ir,
};
use super::*;

const MODULE: &str = r#"core.module @test {
  !open = tribute_control.func_sig<(core.i32) -> core.i32, {ability_row = {tail = 0}, tribute.calling_convention = "cps"}>
  !closed = tribute_control.func_sig<(core.i32) -> core.i32, {ability_row = {instances = [@A, @B]}, tribute.calling_convention = "cps"}>
  !applied = tribute_control.func_sig<(!closed, core.i32) -> core.i32, {ability_row = {instances = [@A, @B]}, tribute.calling_convention = "cps"}>

  ability.decl {sym_name = "A"} {
    ability.operation {kind = "fn", op_name = "get", param_types = [], result_type = core.i32}
  }
  ability.decl {sym_name = "B"} {
    ability.operation {kind = "op", op_name = "put", param_types = [core.i32], result_type = core.i32}
  }
  tribute_control.func @apply['a0](%f: !open, %x: core.i32) -> core.i32 row(| 'a0) convention(cps) {
    %r = tribute_control.call_indirect %f, %x : core.i32
    tribute_control.return %r
  }
  tribute_control.func @main(%x: core.i32) -> core.i32 row(@A, @B) convention(cps) {
    %f = tribute_control.lambda(%y: core.i32) -> core.i32 row(@A, @B) convention(cps) captures [] {
      tribute_control.return %y
    }
    %g = tribute_control.func_ref {func_ref = @apply, row_args = [{instances = [@A, @B]}]} : !applied
    %r = tribute_control.call %f, %x {callee = @apply, row_args = [{instances = [@A, @B]}]} : core.i32
    tribute_control.return %r
  }
}"#;

fn parse(input: &str) -> (IrContext, Module) {
    let mut ctx = IrContext::new();
    let root = parse_module(&mut ctx, input)
        .unwrap_or_else(|error| panic!("failed to parse:\n{error}\n\n{input}"));
    let module = Module::new(&ctx, root).expect("core.module");
    (ctx, module)
}

fn validate(ctx: &IrContext, module: Module) -> ValidationResult {
    let mut result = validate_local(ctx, module);
    result
        .errors
        .extend(validate_whole_ir(ctx, module, &[]).errors);
    result
}

fn assert_rejects(input: &str, expected: &str) {
    let (ctx, module) = parse(input);
    let result = validate(&ctx, module);
    assert!(
        result
            .errors
            .iter()
            .any(|error| error.message.contains(expected)),
        "expected an error containing {expected:?}, found:\n{result}"
    );
}

#[test]
fn rows_validate_and_round_trip_in_concise_syntax() {
    let (ctx, module) = parse(MODULE);
    let result = validate(&ctx, module);
    assert!(result.is_ok(), "{result}");

    let printed = print_module(&ctx, module.op());
    assert!(
        printed.contains(
            "@apply['a0](%0: !open, %1: core.i32) -> core.i32 row(| 'a0) convention(cps) {"
        ),
        "{printed}"
    );
    assert!(
        printed.contains("-> core.i32 row(@A, @B) convention(cps)"),
        "{printed}"
    );
    let mut reparsed = IrContext::new();
    let root = parse_module(&mut reparsed, &printed)
        .unwrap_or_else(|error| panic!("failed to reparse:\n{error}\n\n{printed}"));
    assert_eq!(print_module(&reparsed, root), printed);
}

#[test]
fn concise_rows_cover_instances_tails_and_both() {
    for (written, row) in [
        ("row(@A)", Row::new([SymbolPath::new(["A"])], None)),
        ("row(| 'a0)", Row::new([], Some(0))),
        (
            "row(@A, @B | 'a0)",
            Row::new([SymbolPath::new(["A"]), SymbolPath::new(["B"])], Some(0)),
        ),
    ] {
        let input = format!(
            "core.module @test {{\n  tribute_control.func @f() -> core.i32 {written} convention(cps)\n}}"
        );
        let (ctx, module) = parse(&input);
        let func = module.ops(&ctx)[0];
        let ty = ctx.op_attrs(func).get_type("type").unwrap();
        assert_eq!(FuncSig::from_type_ref(&ctx, ty).unwrap().row(&ctx), row);
        assert!(print_module(&ctx, module.op()).contains(written));
    }
}

#[test]
fn concise_syntax_does_not_write_the_closed_empty_row() {
    let mut ctx = IrContext::new();
    assert!(
        parse_module(
            &mut ctx,
            "core.module @test {\n  tribute_control.func @f() -> core.i32 row() convention(cps)\n}",
        )
        .is_err()
    );
}

#[test]
fn a_row_argument_that_changes_the_callee_signature_is_rejected() {
    assert_rejects(
        &MODULE.replace(
            "{callee = @apply, row_args = [{instances = [@A, @B]}]}",
            "{callee = @apply, row_args = [{instances = [@A]}]}",
        ),
        "call argument types do not match the logical signature",
    );
    assert_rejects(
        &MODULE.replace(
            "{func_ref = @apply, row_args = [{instances = [@A, @B]}]}",
            "{func_ref = @apply, row_args = [{instances = [@B]}]}",
        ),
        "func_ref result source signature does not match its target",
    );
}

#[test]
fn a_reference_passes_one_row_for_each_quantified_binder() {
    assert_rejects(
        &MODULE.replace(
            ", row_args = [{instances = [@A, @B]}]} : core.i32",
            "} : core.i32",
        ),
        "passes 0 row arguments, but its target quantifies 1 row binders",
    );
}

#[test]
fn rows_must_be_sorted_and_not_written_when_closed_and_empty() {
    assert_rejects(
        &MODULE.replace(
            "row(@A, @B) convention(cps) {",
            "row(@B, @A) convention(cps) {",
        ),
        "malformed tribute_control.func_sig: Row",
    );
    assert_rejects(
        &MODULE.replace(
            "!closed = tribute_control.func_sig<(core.i32) -> core.i32, {ability_row = {instances = [@A, @B]}",
            "!closed = tribute_control.func_sig<(core.i32) -> core.i32, {ability_row = {}",
        ),
        "malformed tribute_control.func_sig: Row",
    );
}

#[test]
fn a_tail_names_a_binder_of_the_enclosing_definition() {
    assert_rejects(
        &MODULE.replace(
            "-> core.i32 row(@A, @B) convention(cps) {\n    %f",
            "-> core.i32 row(@A, @B | 'a0) convention(cps) {\n    %f",
        ),
        "row tail 0 names no row binder in scope",
    );
    assert_rejects(
        &MODULE.replace(
            "param_types = [], result_type = core.i32}",
            "param_types = [!open], result_type = core.i32}",
        ),
        "row tail 0 names no row binder in scope",
    );
}

#[test]
fn row_instances_name_ability_declarations() {
    assert_rejects(
        &MODULE.replace(
            "row(@A, @B) convention(cps) {\n    %f",
            "row(@A, @C) convention(cps) {\n    %f",
        ),
        "row instance @C is not an ability.decl",
    );
}

#[test]
fn the_row_belongs_to_the_signature_type() {
    let mut ctx = IrContext::new();
    let input = MODULE.replace(
        "row(| 'a0) convention(cps) {\n    %r",
        "row(| 'a0) convention(cps) attributes {ability_row = {tail = 0}} {\n    %r",
    );
    assert!(parse_module(&mut ctx, &input).is_err());
}

#[test]
fn row_binders_put_quantified_binders_first() {
    assert_rejects(
        &MODULE.replace("@apply['a0]", "@apply[body 'a0, 'a1]"),
        "quantified row binders must come before the others",
    );
}

const UNIONS: &str = r#"core.module @test {
  ability.decl {sym_name = "A"} {
    ability.operation {kind = "op", op_name = "get", param_types = [], result_type = core.i32}
  }
  tribute_control.func @join['a0 = 'a1 | 'a2, 'a1, 'a2](%x: core.i32) -> core.i32 row(| 'a0) convention(cps) {
    tribute_control.return %x
  }
  tribute_control.func @caller['a0 = 'a1 | 'a2, 'a1, 'a2](%x: core.i32) -> core.i32 row(| 'a0) convention(cps) {
    %r = tribute_control.call %x {callee = @join, evidence_plan = PLAN, row_args = [{tail = 0}, {tail = 1}, {tail = 2}]} : core.i32
    tribute_control.return %r
  }
}"#;

#[test]
fn evidence_selections_count_the_sources_of_a_row_union() {
    let (ctx, module) =
        parse(&UNIONS.replace("PLAN", "[{select = 1}, {tails = [[], [{select = 0}]]}]"));
    let result = validate(&ctx, module);
    assert!(result.is_ok(), "{result}");

    assert_rejects(
        &UNIONS.replace("PLAN", "[{select = 2}]"),
        "evidence_plan select 2 names no source",
    );
    assert_rejects(
        &UNIONS.replace("PLAN", "[{tails = [[]]}]"),
        "evidence_plan tails must hold one plan for each source",
    );
}

#[test]
fn a_union_names_binders_and_excludes_its_result() {
    assert_rejects(
        &UNIONS
            .replace("PLAN", "[{select = 0}]")
            .replace("@caller['a0 = 'a1 | 'a2,", "@caller['a0 = 'a0 | 'a2,"),
        "row union lists its result among its sources",
    );
    assert_rejects(
        &UNIONS
            .replace("PLAN", "[{select = 0}]")
            .replace("@caller['a0 = 'a1 | 'a2,", "@caller['a0 = 'a1 | 'a3,"),
        "row union names a tail that is not a binder",
    );
}

#[test]
fn substitution_rebuilds_callables_and_tuple_layouts() {
    let mut ctx = IrContext::new();
    let i32_ty = TypeDataBuilder::new("core.i32").intern(&mut ctx);
    let a = SymbolPath::new(["A"]);
    let b = SymbolPath::new(["B"]);
    let open = FuncSig::params([i32_ty], i32_ty)
        .row(Row::new([b.clone()], Some(0)))
        .convention(CallingConvention::Cps)
        .intern(&mut ctx)
        .as_type_ref();
    let tuple = logical_tuple_type(&mut ctx, &[open, i32_ty]);
    let args = [Row::new([a.clone()], Some(3))];

    assert_eq!(find_substituted_rows(&ctx, tuple, &args), None);
    let substituted = substitute_rows(&mut ctx, tuple, &args).unwrap();
    assert_eq!(find_substituted_rows(&ctx, tuple, &args), Some(substituted));

    let elements = logical_tuple_elements(&ctx, substituted).unwrap();
    assert_eq!(elements[1], i32_ty);
    let callable = FuncSig::from_type_ref(&ctx, elements[0]).unwrap();
    assert_eq!(callable.row(&ctx), Row::new([b, a], Some(3)));
    assert_eq!(callable.convention(&ctx), CallingConvention::Cps);

    assert_eq!(substitute_rows(&mut ctx, tuple, &[]), None);
    assert_eq!(substitute_rows(&mut ctx, i32_ty, &[]), Some(i32_ty));
}

#[test]
fn finding_a_substituted_row_needs_only_the_keys_it_holds() {
    let mut ctx = IrContext::new();
    let i32_ty = TypeDataBuilder::new("core.i32").intern(&mut ctx);
    let open = FuncSig::params([i32_ty], i32_ty)
        .row(Row::new([], Some(0)))
        .convention(CallingConvention::Cps)
        .intern(&mut ctx)
        .as_type_ref();
    let args = [Row::new([], Some(1))];
    let substituted = substitute_rows(&mut ctx, open, &args).unwrap();
    assert_eq!(ctx.lookup_str("instances"), None);
    assert_eq!(find_substituted_rows(&ctx, open, &args), Some(substituted));
}

#[test]
fn tuple_names_tell_callables_apart_by_their_attributes() {
    let mut ctx = IrContext::new();
    let i32_ty = TypeDataBuilder::new("core.i32").intern(&mut ctx);
    let noted = |ctx: &mut IrContext, value| {
        let attrs = trunk_ir::types::AttributeMap::from_entries(ctx, [("note", value)]);
        FuncSig::params([i32_ty], i32_ty)
            .convention(CallingConvention::Direct)
            .attrs(attrs)
            .intern(ctx)
            .as_type_ref()
    };
    let plain = FuncSig::params([i32_ty], i32_ty)
        .convention(CallingConvention::Direct)
        .intern(&mut ctx)
        .as_type_ref();
    let int = noted(&mut ctx, trunk_ir::types::Attribute::Int(1));
    let one = ctx.string_attr("1");
    let string = noted(&mut ctx, one);
    let names = [plain, int, string].map(|ty| logical_tuple_name(&ctx, &[ty]));
    assert_ne!(names[0], names[1]);
    assert_ne!(names[1], names[2]);
}

#[test]
fn tuple_names_keep_attribute_names_and_text_apart() {
    let mut ctx = IrContext::new();
    let i32_ty = TypeDataBuilder::new("core.i32").intern(&mut ctx);
    let with = |ctx: &mut IrContext, name: &'static str, value| {
        let attrs = trunk_ir::types::AttributeMap::from_entries(ctx, [(name, value)]);
        FuncSig::params([i32_ty], i32_ty)
            .convention(CallingConvention::Direct)
            .attrs(attrs)
            .intern(ctx)
            .as_type_ref()
    };
    use trunk_ir::types::Attribute;
    let list = with(&mut ctx, "x", Attribute::List(vec![Attribute::Unit]));
    let unit = with(&mut ctx, "x_1_13_list", Attribute::Unit);
    assert_ne!(
        logical_tuple_name(&ctx, &[list]),
        logical_tuple_name(&ctx, &[unit])
    );
    let spaced = ctx.string_attr("hello world");
    let spaced = with(&mut ctx, "note", spaced);
    let name = logical_tuple_name(&ctx, &[spaced]);
    assert!(
        name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'),
        "{name}"
    );
}

#[test]
fn tuple_names_keep_symbol_path_components_apart() {
    let mut ctx = IrContext::new();
    let i32_ty = TypeDataBuilder::new("core.i32").intern(&mut ctx);
    let names = [SymbolPath::new(["a::b"]), SymbolPath::new(["a", "b"])].map(|instance| {
        let callable = FuncSig::params([i32_ty], i32_ty)
            .row(Row::new([instance], None))
            .convention(CallingConvention::Cps)
            .intern(&mut ctx)
            .as_type_ref();
        logical_tuple_name(&ctx, &[callable])
    });
    assert_ne!(names[0], names[1]);
}

#[test]
fn binder_headers_name_kinds_unions_and_an_inferred_row() {
    let input = r#"core.module @test {
  tribute_control.func @f['a0 = 'a1 | 'a2, ability_arg 'a1, local 'a2, body 'a3](%x: core.i32) -> core.i32 row(| 'a0) inferred convention(cps) {
    tribute_control.return %x
  }
}"#;
    let (ctx, module) = parse(input);
    let func = module.ops(&ctx)[0];
    assert_eq!(
        row_binders(&ctx, func).unwrap(),
        [
            RowBinder::Param,
            RowBinder::AbilityArg,
            RowBinder::Local,
            RowBinder::Body
        ]
    );
    assert_eq!(
        row_unions(&ctx, func).unwrap(),
        [RowUnion {
            result: 0,
            sources: vec![1, 2]
        }]
    );
    assert_eq!(
        ctx.op_attrs(func).get(INFERRED_ROW_ATTR),
        Some(&trunk_ir::types::Attribute::Bool(true))
    );
    let printed = print_module(&ctx, module.op());
    assert!(
        printed.contains(
            "@f['a0 = 'a1 | 'a2, ability_arg 'a1, local 'a2, body 'a3](%0: core.i32) -> core.i32 row(| 'a0) inferred convention(cps) {"
        ),
        "{printed}"
    );
}

#[test]
fn a_binder_is_named_by_its_position() {
    let mut ctx = IrContext::new();
    assert!(
        parse_module(
            &mut ctx,
            "core.module @test {\n  tribute_control.func @f['a1]() -> core.i32 convention(cps)\n}",
        )
        .is_err()
    );
}

#[test]
fn a_header_may_declare_only_unquantified_binders() {
    let input = r#"core.module @test {
  tribute_control.func @run[local 'a0, body 'a1]() -> core.i32 convention(direct) {
    %f = tribute_control.lambda['a0]() -> core.i32 row(| 'a0) convention(cps) captures [] {
      %x = arith.const {value = 1} : core.i32
      tribute_control.return %x
    }
    %y = arith.const {value = 2} : core.i32
    tribute_control.return %y
  }
}"#;
    let (ctx, module) = parse(input);
    let result = validate(&ctx, module);
    assert!(result.is_ok(), "{result}");
    let printed = print_module(&ctx, module.op());
    assert!(
        printed.contains("@run[local 'a0, body 'a1]() -> core.i32 convention(direct) {"),
        "{printed}"
    );
    assert!(
        printed.contains("lambda['a0]() -> core.i32 row(| 'a0) convention(cps) captures []"),
        "{printed}"
    );
}

#[test]
fn a_function_reference_may_widen_the_row_of_its_target() {
    let input = MODULE.replace(
        "{func_ref = @apply, row_args = [{instances = [@A, @B]}]}",
        "{func_ref = @apply, row_args = [{instances = [@A]}]}",
    );
    let narrowed = input.replace(
        "!applied = tribute_control.func_sig<(!closed, core.i32) -> core.i32, {ability_row = {instances = [@A, @B]}",
        "!narrow = tribute_control.func_sig<(core.i32) -> core.i32, {ability_row = {instances = [@A]}, tribute.calling_convention = \"cps\"}>\n  !applied = tribute_control.func_sig<(!narrow, core.i32) -> core.i32, {ability_row = {instances = [@A, @B]}",
    );
    let (ctx, module) = parse(&narrowed);
    let result = validate(&ctx, module);
    assert!(result.is_ok(), "{result}");

    assert_rejects(
        &narrowed.replace(
            "-> core.i32, {ability_row = {instances = [@A, @B]}, tribute.calling_convention = \"cps\"}>\n\n",
            "-> core.i32, {ability_row = {instances = [@B]}, tribute.calling_convention = \"cps\"}>\n\n",
        ),
        "func_ref result source signature does not match its target",
    );
}

#[test]
fn row_inclusion_counts_instances_and_keeps_the_tail() {
    let a = SymbolPath::new(["A"]);
    let b = SymbolPath::new(["B"]);
    let wide = Row::new([a.clone(), a.clone(), b.clone()], Some(1));
    assert!(wide.includes(&Row::new([a.clone(), b.clone()], None)));
    assert!(wide.includes(&Row::new([a.clone()], Some(1))));
    assert!(!wide.includes(&Row::new([a.clone()], Some(0))));
    assert!(!Row::new([a.clone()], None).includes(&Row::new([a.clone(), a], None)));
    assert!(!Row::new([b], None).includes(&Row::new([], Some(1))));
}

const LOCALS: &str = r#"core.module @test {
  !step = tribute_control.func_sig<(core.i32) -> core.i32, {ability_row = {tail = 1}, tribute.calling_convention = "direct"}>
  !step0 = tribute_control.func_sig<(core.i32) -> core.i32, {ability_row = {tail = 0}, tribute.calling_convention = "direct"}>
  ability.decl {sym_name = "A"} {
  }
  tribute_control.func @f['a0, local 'a1](%x: core.i32) -> core.i32 row(| 'a0) convention(direct) {
    %g = tribute_control.lambda['a1](%y: core.i32) -> core.i32 row(| 'a1) convention(direct) captures [] {
      tribute_control.return %y
    }
    %s = tribute_control.instantiate %g {row_args = [ARGS]} : RESULT
    %r = tribute_control.call_indirect %s, %x : core.i32
    tribute_control.return %r
  }
}"#;

fn locals(args: &str, result: &str) -> String {
    LOCALS.replace("ARGS", args).replace("RESULT", result)
}

#[test]
fn a_quantifying_lambda_round_trips_and_its_uses_instantiate_it() {
    let input = locals("{tail = 0}", "!step0");
    let (ctx, module) = parse(&input);
    let result = validate(&ctx, module);
    assert!(result.is_ok(), "{result}");
    let printed = print_module(&ctx, module.op());
    assert!(
        printed.contains(
            "%1 = tribute_control.lambda['a1](%2: core.i32) -> core.i32 row(| 'a1) convention(direct) captures [] {"
        ),
        "{printed}"
    );
    assert!(
        printed.contains("%3 = tribute_control.instantiate %1 {row_args = [{tail = 0}]} : !step0"),
        "{printed}"
    );
    let (reparsed, module) = parse(&printed);
    assert_eq!(print_module(&reparsed, module.op()), printed);
}

#[test]
fn instantiate_substitutes_only_the_quantified_binders() {
    // The lambda's own binder becomes `@A`; the enclosing `'a0` stays.
    let input = locals("{instances = [@A]}", "!closed").replace(
        "  ability.decl",
        "  !closed = tribute_control.func_sig<(core.i32) -> core.i32, {ability_row = {instances = [@A]}, tribute.calling_convention = \"direct\"}>\n  ability.decl",
    );
    let (ctx, module) = parse(&input);
    let result = validate(&ctx, module);
    assert!(result.is_ok(), "{result}");
}

#[test]
fn instantiate_checks_its_lambda_arguments_and_result() {
    assert_rejects(
        &locals("{tail = 0}", "!step"),
        "instantiate result is not the lambda's type with its row arguments substituted",
    );
    assert_rejects(
        &locals("{tail = 0}, {}", "!step0"),
        "instantiate passes 2 row arguments for 1 quantified binders",
    );
    assert_rejects(
        &locals("{tail = 2}", "!step0"),
        "row tail 2 names no row binder in scope",
    );
    assert_rejects(
        &locals("{tail = 0}", "!step0").replace("lambda['a1]", "lambda"),
        "instantiate takes a lambda that quantifies rows",
    );
    assert_rejects(
        &locals("{tail = 0}", "!step0").replace(
            "convention(direct) captures [] {",
            "convention(evidence_direct) captures [] {",
        ),
        "instantiate result is not the lambda's type with its row arguments substituted",
    );
}

#[test]
fn a_lambda_quantifies_local_binders_it_alone_owns() {
    assert_rejects(
        &locals("{tail = 0}", "!step0")
            .replace("@f['a0, local 'a1]", "@f['a0, local 'a1, local 'a2]"),
        "local binder 2 is quantified by no lambda",
    );
    assert_rejects(
        &locals("{tail = 0}", "!step0").replace("lambda['a1]", "lambda['a0]"),
        "lambda quantifies binder 0, which is not local",
    );
    assert_rejects(
        &locals("{tail = 0}", "!step0").replace(
            "    %s = tribute_control.instantiate",
            "    %h = tribute_control.lambda['a1](%z: core.i32) -> core.i32 row(| 'a1) convention(direct) captures [] {\n      tribute_control.return %z\n    }\n    %s = tribute_control.instantiate",
        ),
        "local binder 1 is quantified by two lambdas",
    );
    assert_rejects(
        &locals("{tail = 0}", "!step0").replace(
            "tribute_control.call_indirect %s, %x",
            "tribute_control.call_indirect %g, %x",
        ),
        "a lambda that quantifies rows is used only through instantiate",
    );
}

#[test]
fn a_lambda_captured_by_another_is_instantiated_in_its_body() {
    let nested = |use_: &str| {
        locals("{tail = 0}", "!step0").replace(
            "    %s = tribute_control.instantiate %g {row_args = [{tail = 0}]} : !step0\n    %r = tribute_control.call_indirect %s, %x : core.i32\n",
            &format!(
                "    %h = tribute_control.lambda(%z: core.i32) -> core.i32 row(| 'a0) convention(direct) captures [%g] {{\n{use_}      tribute_control.return %r\n    }}\n    %r = tribute_control.call_indirect %h, %x : core.i32\n"
            ),
        )
    };
    let (ctx, module) = parse(&nested(
        "      %s = tribute_control.instantiate %g {row_args = [{tail = 0}]} : !step0\n      %r = tribute_control.call_indirect %s, %z : core.i32\n",
    ));
    let result = validate(&ctx, module);
    assert!(result.is_ok(), "{result}");
    assert_rejects(
        &nested("      %r = tribute_control.call_indirect %g, %z : core.i32\n"),
        "a lambda that quantifies rows is used only through instantiate",
    );
}

/// Frontend output before class specialization: rows without conventions.
const UNASSIGNED: &str = r#"core.module @test {
  !cb = tribute_control.func_sig<(core.i32) -> core.i32, {ability_row = {tail = 0}}>
  !pure = tribute_control.func_sig<(core.i32) -> core.i32>
  !applied = tribute_control.func_sig<(!pure, core.i32) -> core.i32>
  tribute_control.func @apply['a0](%f: !cb, %x: core.i32) -> core.i32 row(| 'a0) {
    %r = tribute_control.call_indirect %f, %x : core.i32
    tribute_control.return %r
  }
  tribute_control.func @forward['a0](%f: !cb, %x: core.i32) -> core.i32 row(| 'a0) {
    tribute_control.tail_call_indirect %f, %x
  }
  tribute_control.func @main[local 'a0](%x: core.i32) -> core.i32 inferred {
    %g = tribute_control.lambda(%y: core.i32) -> core.i32 LAMBDA captures [] {
      tribute_control.return %y
    }
    %h = tribute_control.lambda['a0](%z: core.i32) -> core.i32 row(| 'a0) captures [] {
      tribute_control.return %z
    }
    %s = tribute_control.instantiate %h {row_args = [{}]} : !pure
    %a = tribute_control.func_ref {func_ref = @apply, row_args = [{}]} : !applied
    %v = tribute_control.call_indirect %s, %x : core.i32
    %r = tribute_control.call_indirect %a, %g, %v : core.i32
    tribute_control.return %r
  }
}"#;

#[test]
fn signatures_may_wait_for_class_specialization_to_assign_conventions() {
    let input = UNASSIGNED.replace(" LAMBDA", "");
    let (ctx, module) = parse(&input);
    let result = validate(&ctx, module);
    assert!(result.is_ok(), "{result}");
    assert_eq!(conventions_assigned(&ctx, module), Some(false));
    let printed = print_module(&ctx, module.op());
    assert!(
        printed.contains("@apply['a0](%0: !cb, %1: core.i32) -> core.i32 row(| 'a0) {")
            && printed.contains("@main[local 'a0](%0: core.i32) -> core.i32 inferred {")
            && printed.contains("tribute_control.lambda['a0](%")
            && printed.contains("-> core.i32 captures [] {"),
        "{printed}"
    );
    let (reparsed, module) = parse(&printed);
    assert_eq!(print_module(&reparsed, module.op()), printed);
}

#[test]
fn a_module_assigns_conventions_to_every_signature_or_to_none() {
    let input = UNASSIGNED.replace(" LAMBDA", " convention(direct)");
    let (ctx, module) = parse(&input);
    assert_eq!(conventions_assigned(&ctx, module), None);
    assert_rejects(
        &input,
        "only some source signatures have a calling convention",
    );
}

#[test]
fn a_tuple_layout_names_a_convention_only_once_assigned() {
    let mut ctx = IrContext::new();
    let i32_ty = TypeDataBuilder::new("core.i32").intern(&mut ctx);
    let unassigned = TypeDataBuilder::new("tribute_control.func_sig")
        .params([i32_ty, i32_ty])
        .attr(
            trunk_ir::dialect::func::NUM_INPUTS_ATTR,
            trunk_ir::types::Attribute::from(1_u32),
        )
        .attr(
            trunk_ir::dialect::func::NUM_RESULTS_ATTR,
            trunk_ir::types::Attribute::from(1_u32),
        )
        .intern(&mut ctx);
    assert!(FuncSig::from_type_ref(&ctx, unassigned).is_some());
    let assigned = FuncSig::params([i32_ty], i32_ty)
        .convention(CallingConvention::Direct)
        .intern(&mut ctx)
        .as_type_ref();
    let names = [unassigned, assigned].map(|callable| logical_tuple_name(&ctx, &[callable]));
    assert_ne!(names[0], names[1]);
    assert!(!names[0].contains("direct"), "{}", names[0]);
    assert!(names[1].contains("direct"), "{}", names[1]);
}

/// A source signature without a convention, as the frontend builds it.
fn unassigned(ctx: &mut IrContext, inputs: &[TypeRef], result: TypeRef, row: &Row) -> TypeRef {
    let mut builder = TypeDataBuilder::new("tribute_control.func_sig")
        .params(inputs.iter().copied())
        .param(result)
        .attr(
            trunk_ir::dialect::func::NUM_INPUTS_ATTR,
            trunk_ir::types::Attribute::from(inputs.len() as u32),
        )
        .attr(
            trunk_ir::dialect::func::NUM_RESULTS_ATTR,
            trunk_ir::types::Attribute::from(1_u32),
        );
    if !row.is_empty() {
        let row = row.to_attribute(ctx);
        builder = builder.attr(ABILITY_ROW_ATTR, row);
    }
    builder.intern(ctx)
}

#[test]
fn assigning_conventions_reaches_every_callable_and_layout() {
    let mut ctx = IrContext::new();
    let i32_ty = TypeDataBuilder::new("core.i32").intern(&mut ctx);
    let inner = unassigned(&mut ctx, &[i32_ty], i32_ty, &Row::default());
    let pair = logical_tuple_type(&mut ctx, &[inner, i32_ty]);
    let outer = unassigned(&mut ctx, &[pair], i32_ty, &Row::new([], Some(0)));
    let class = |row: &Row| {
        if row.tail.is_some() {
            CallingConvention::Cps
        } else {
            CallingConvention::Direct
        }
    };

    let assigned = assign_conventions(&mut ctx, outer, class);
    assert_eq!(
        func_sig_convention(&ctx, assigned),
        Some(CallingConvention::Cps)
    );
    let signature = FuncSig::from_type_ref(&ctx, assigned).unwrap();
    let elements = logical_tuple_elements(&ctx, signature.inputs(&ctx)[0])
        .expect("the input is still a tuple layout")
        .to_vec();
    assert_eq!(
        func_sig_convention(&ctx, elements[0]),
        Some(CallingConvention::Direct)
    );
    assert_eq!(elements[1], i32_ty);
    assert_eq!(signature.row(&ctx), Row::new([], Some(0)));
    assert_eq!(assign_conventions(&mut ctx, assigned, class), assigned);
}

#[test]
fn setting_a_signature_convention_leaves_inner_callables_alone() {
    let mut ctx = IrContext::new();
    let i32_ty = TypeDataBuilder::new("core.i32").intern(&mut ctx);
    let inner = unassigned(&mut ctx, &[i32_ty], i32_ty, &Row::default());
    let outer = unassigned(&mut ctx, &[inner], i32_ty, &Row::default());
    let set = FuncSig::from_type_ref(&ctx, outer)
        .unwrap()
        .with_convention(&mut ctx, CallingConvention::EvidenceDirect)
        .as_type_ref();
    assert_eq!(
        func_sig_convention(&ctx, set),
        Some(CallingConvention::EvidenceDirect)
    );
    let signature = FuncSig::from_type_ref(&ctx, set).unwrap();
    assert_eq!(signature.inputs(&ctx), [inner]);
    assert_eq!(func_sig_convention(&ctx, inner), None);
}
