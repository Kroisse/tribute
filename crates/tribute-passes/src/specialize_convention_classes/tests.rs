use trunk_ir::parser::parse_test_module;
use trunk_ir::printer::print_module;

use super::*;

/// The module `input` becomes, after checking the three properties every
/// specialized module has: each signature has a convention, the module is
/// valid, and the classes agree with its rows.
fn specialized(input: &str) -> String {
    let mut ctx = IrContext::new();
    let module = parse_test_module(&mut ctx, input);
    specialize(&mut ctx, module).expect("specializes");
    let printed = print_module(&ctx, module.op());
    assert_eq!(
        tribute_control::conventions_assigned(&ctx, module),
        Some(true),
        "{printed}"
    );
    let validation = tribute_control::validate(&ctx, module, &[], &mut Default::default());
    assert!(validation.is_ok(), "{validation}\n{printed}");
    let disagreements = crate::check_convention_classes::check(&ctx, module);
    assert!(disagreements.is_empty(), "{disagreements:?}\n{printed}");
    printed
}

/// A pure function passed for an open callback selects the `Direct`
/// instance of the function that calls it.
#[test]
fn a_pure_callback_selects_the_direct_instance() {
    let printed = specialized(
        r#"core.module @m {
  !cb = tribute_control.func_sig<(core.i32) -> core.i32, {ability_row = {tail = 0}}>
  !pure = tribute_control.func_sig<(core.i32) -> core.i32>
  tribute_control.func @app['a0](%x: core.i32, %f: !cb) -> core.i32 row(| 'a0) {
    %r = tribute_control.call_indirect %f, %x : core.i32
    tribute_control.return %r
  }
  tribute_control.func @bang(%x: core.i32) -> core.i32 {
    tribute_control.return %x
  }
  tribute_control.func @main(%x: core.i32) -> core.i32 {
    %f = tribute_control.func_ref {func_ref = @bang} : !pure
    %r = tribute_control.call %x, %f {callee = @app, row_args = [{}]} : core.i32
    tribute_control.return %r
  }
}"#,
    );
    for expected in [
        "tribute_control.func @app['a0](%0: core.i32, %1: tribute_control.func_sig<(core.i32) -> core.i32, {ability_row = {tail = 0}, tribute.calling_convention = 2}>) -> core.i32 row(| 'a0) convention(cps) {",
        "tribute_control.func @\"app$9D\"['a0](%0: core.i32, %1: tribute_control.func_sig<(core.i32) -> core.i32, {ability_row = {tail = 0}, tribute.calling_convention = 0}>) -> core.i32 row(| 'a0) convention(direct) {",
        "tribute_control.func @bang(%0: core.i32) -> core.i32 convention(direct) {",
        "%2 = tribute_control.call %0, %1 {callee = @\"app$9D\", row_args = [{}]} : core.i32",
    ] {
        assert!(printed.contains(expected), "{expected}\n{printed}");
    }
}

fn assert_contains(printed: &str, expected: &[&str]) {
    for expected in expected {
        assert!(printed.contains(expected), "{expected}\n{printed}");
    }
}

/// A class variable that only a reference's rows reach is selected through
/// the callee, and a function reference points at the instance its rows
/// select.
#[test]
fn a_class_variable_reached_through_a_callee_is_selected() {
    let printed = specialized(
        r#"core.module @m {
  !cb = tribute_control.func_sig<(core.i32) -> core.i32, {ability_row = {tail = 0}}>
  !pure = tribute_control.func_sig<(core.i32) -> core.i32>
  tribute_control.func @app['a0](%x: core.i32, %f: !cb) -> core.i32 row(| 'a0) {
    %r = tribute_control.call_indirect %f, %x : core.i32
    tribute_control.return %r
  }
  tribute_control.func @ident['a0](%x: core.i32) -> core.i32 row(| 'a0) inferred {
    tribute_control.return %x
  }
  tribute_control.func @bang['a0](%x: core.i32) -> core.i32 row(| 'a0) inferred {
    %f = tribute_control.func_ref {func_ref = @ident, row_args = [{tail = 0}]} : !cb
    %r = tribute_control.call %x, %f {callee = @app, row_args = [{tail = 0}]} : core.i32
    tribute_control.return %r
  }
  tribute_control.func @twice['a0](%x: core.i32, %g: !cb) -> core.i32 row(| 'a0) {
    %r = tribute_control.call_indirect %g, %x : core.i32
    tribute_control.return %r
  }
  tribute_control.func @main(%x: core.i32) -> core.i32 {
    %g = tribute_control.func_ref {func_ref = @bang, row_args = [{}]} : !pure
    %r = tribute_control.call %x, %g {callee = @twice, row_args = [{}]} : core.i32
    tribute_control.return %r
  }
}"#,
    );
    assert_contains(
        &printed,
        &[
            "tribute_control.func @bang['a0](%0: core.i32) -> core.i32 row(| 'a0) inferred convention(cps) {",
            "tribute_control.func @\"bang$9D\"['a0](%0: core.i32) -> core.i32 row(| 'a0) inferred convention(direct) {",
            "%2 = tribute_control.call %0, %1 {callee = @\"app$9D\", row_args = [{tail = 0}]} : core.i32",
            "%1 = tribute_control.func_ref {func_ref = @\"bang$9D\", row_args = [{}]}",
            "%2 = tribute_control.call %0, %1 {callee = @\"twice$9D\", row_args = [{}]} : core.i32",
            "tribute_control.func @ident['a0](%0: core.i32) -> core.i32 row(| 'a0) inferred convention(direct) {",
        ],
    );
}

/// An instance tail-calls itself, and a lambda in its body takes the
/// instance's class.
#[test]
fn a_recursive_instance_calls_itself() {
    let printed = specialized(
        r#"core.module @m {
  !cb = tribute_control.func_sig<(core.i32) -> core.i32, {ability_row = {tail = 0}}>
  tribute_control.func @loop['a0](%n: core.i32, %f: !cb) -> core.i32 row(| 'a0) {
    %g = tribute_control.lambda(%y: core.i32) -> core.i32 row(| 'a0) captures [%f] {
      %v = tribute_control.call_indirect %f, %y : core.i32
      tribute_control.return %v
    }
    tribute_control.tail_call %n, %g {callee = @loop, row_args = [{tail = 0}]}
  }
  tribute_control.func @main(%x: core.i32) -> core.i32 {
    %h = tribute_control.lambda(%y: core.i32) -> core.i32 captures [] {
      tribute_control.return %y
    }
    %r = tribute_control.call %x, %h {callee = @loop, row_args = [{}]} : core.i32
    tribute_control.return %r
  }
}"#,
    );
    assert_contains(
        &printed,
        &[
            "tribute_control.func @loop['a0](%0: core.i32, %1: !t1) -> core.i32 row(| 'a0) convention(cps) {",
            "%2 = tribute_control.lambda(%3: core.i32) -> core.i32 row(| 'a0) convention(cps) captures [%1] {",
            "tribute_control.func @\"loop$9D\"['a0](%0: core.i32, %1: !t0) -> core.i32 row(| 'a0) convention(direct) {",
            "%2 = tribute_control.lambda(%3: core.i32) -> core.i32 row(| 'a0) convention(direct) captures [%1] {",
            "tribute_control.tail_call %0, %2 {callee = @\"loop$9D\", row_args = [{tail = 0}]}",
        ],
    );
}

/// A struct layout whose field is a callable is rebuilt with the field's
/// convention.
#[test]
fn a_layout_takes_the_convention_of_its_callable_field() {
    let printed = specialized(
        r#"core.module @m {
  !cb = tribute_control.func_sig<(core.i32) -> core.i32>
  !Holder = adt.struct<Holder(callback: !cb)>
  !HolderRef = adt.typeref<{name = "Holder"}>
  tribute_control.func @"Holder::callback"(%h: !HolderRef) -> !cb {
    %f = adt.struct_get %h {field = 0, type = !Holder} : !cb
    tribute_control.return %f
  }
  tribute_control.func @main(%x: core.i32) -> core.i32 {
    %l = tribute_control.lambda(%y: core.i32) -> core.i32 captures [] {
      tribute_control.return %y
    }
    %h = adt.struct_new %l {type = !Holder} : !HolderRef
    %f = tribute_control.call %h {callee = @"Holder::callback"} : !cb
    %r = tribute_control.call_indirect %f, %x : core.i32
    tribute_control.return %r
  }
}"#,
    );
    assert_contains(
        &printed,
        &[
            "!Holder_1 = adt.struct<Holder(callback: !t0)>",
            "!t0 = tribute_control.func_sig<(core.i32) -> core.i32, {tribute.calling_convention = 0}>",
            "%1 = adt.struct_get %0 {field = 0, type = !Holder_1} : !t0",
            "%3 = adt.struct_new %1 {type = !Holder_1} : !HolderRef",
        ],
    );
}

/// A definition with an inferred row whose tail is not a class variable
/// takes its convention from its abilities alone.
#[test]
fn an_inferred_row_without_a_class_variable_reads_its_abilities() {
    let printed = specialized(
        r#"core.module @m {
  ability.decl {sym_name = "Ctl"} {
    ability.operation {kind = "op", op_name = "ctl", param_types = [], result_type = core.i32}
  }
  tribute_control.func @each['a0](%x: core.i32) -> core.i32 row(| 'a0) inferred {
    tribute_control.return %x
  }
  tribute_control.func @ask['a0](%x: core.i32) -> core.i32 row(@Ctl | 'a0) inferred {
    tribute_control.return %x
  }
  tribute_control.func @main(%x: core.i32) -> core.i32 row(@Ctl) {
    %a = tribute_control.call %x {callee = @each, row_args = [{instances = [@Ctl]}]} : core.i32
    %b = tribute_control.call %a {callee = @ask, row_args = [{}]} : core.i32
    tribute_control.return %b
  }
}"#,
    );
    assert_contains(
        &printed,
        &[
            "tribute_control.func @each['a0](%0: core.i32) -> core.i32 row(| 'a0) inferred convention(direct) {",
            "tribute_control.func @ask['a0](%0: core.i32) -> core.i32 row(@Ctl | 'a0) inferred convention(cps) {",
        ],
    );
}

#[test]
fn a_module_with_conventions_is_rejected() {
    let mut ctx = IrContext::new();
    let module = parse_test_module(
        &mut ctx,
        r#"core.module @m {
  tribute_control.func @main(%x: core.i32) -> core.i32 convention(direct) {
    tribute_control.return %x
  }
}"#,
    );
    let error = specialize(&mut ctx, module).unwrap_err();
    assert!(
        matches!(error, ClassSpecializationError::ConventionsAssigned),
        "{error}"
    );
}

#[test]
fn a_quantifying_lambda_is_rejected() {
    let mut ctx = IrContext::new();
    let module = parse_test_module(
        &mut ctx,
        r#"core.module @m {
  !l0 = tribute_control.func_sig<(core.i32) -> core.i32>
  tribute_control.func @main[local 'a0](%x: core.i32) -> core.i32 {
    %g = tribute_control.lambda['a0](%y: core.i32) -> core.i32 row(| 'a0) captures [] {
      tribute_control.return %y
    }
    %a = tribute_control.instantiate %g {row_args = [{}]} : !l0
    %r = tribute_control.call_indirect %a, %x : core.i32
    tribute_control.return %r
  }
}"#,
    );
    let error = specialize(&mut ctx, module).unwrap_err();
    assert!(
        matches!(error, ClassSpecializationError::QuantifyingLambda { .. }),
        "{error}"
    );
}
