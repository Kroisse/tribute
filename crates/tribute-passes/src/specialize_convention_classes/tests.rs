use trunk_ir::parser::parse_test_module;
use trunk_ir::printer::print_module;

use super::*;

/// The module `input` becomes, after checking the three properties every
/// specialized module has: its callables are all `tribute_cps` operations,
/// the module is valid, and the classes agree with its rows.
fn specialized(input: &str) -> String {
    let mut ctx = IrContext::new();
    let module = parse_test_module(&mut ctx, input);
    specialize(&mut ctx, module).expect("specializes");
    let printed = print_module(&ctx, module.op());
    assert_eq!(
        tribute_control::specialized(&ctx, module),
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
        "tribute_cps.func @app['a0](%0: core.i32, %1: !cb) -> core.i32 row(| 'a0) convention(cps) {",
        "%2 = tribute_cps.call_indirect %1, %0 {convention = \"cps\"} : core.i32",
        "tribute_cps.func @\"app$9D\"['a0](%0: core.i32, %1: !cb) -> core.i32 row(| 'a0) convention(direct) {",
        "%2 = tribute_cps.call_indirect %1, %0 {convention = \"direct\"} : core.i32",
        "tribute_cps.func @bang(%0: core.i32) -> core.i32 convention(direct) {",
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
            "tribute_cps.func @bang['a0](%0: core.i32) -> core.i32 row(| 'a0) inferred convention(cps) {",
            "tribute_cps.func @\"bang$9D\"['a0](%0: core.i32) -> core.i32 row(| 'a0) inferred convention(direct) {",
            "%2 = tribute_control.call %0, %1 {callee = @\"app$9D\", row_args = [{tail = 0}]} : core.i32",
            "%1 = tribute_cps.func_ref {func_ref = @\"bang$9D$8\", row_args = [{}]}",
            "tribute_cps.func_table {conventions = [\"direct\"], sym_name = \"bang$9D$8\", targets = [@\"bang$9D\"]}",
            "%2 = tribute_control.call %0, %1 {callee = @\"twice$9D\", row_args = [{}]} : core.i32",
            "tribute_cps.func @ident['a0](%0: core.i32) -> core.i32 row(| 'a0) inferred convention(direct) {",
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
            "tribute_cps.func @loop['a0](%0: core.i32, %1: !cb) -> core.i32 row(| 'a0) convention(cps) {",
            "%2 = tribute_cps.lambda(%3: core.i32) -> core.i32 row(| 'a0) captures [%1] entries {\n        cps {",
            "%4 = tribute_cps.call_indirect %1, %3 {convention = \"cps\"} : core.i32",
            "tribute_cps.func @\"loop$9D\"['a0](%0: core.i32, %1: !cb) -> core.i32 row(| 'a0) convention(direct) {",
            "%2 = tribute_cps.lambda(%3: core.i32) -> core.i32 row(| 'a0) captures [%1] entries {\n        direct {",
            "%4 = tribute_cps.call_indirect %1, %3 {convention = \"direct\"} : core.i32",
            "tribute_control.tail_call %0, %2 {callee = @\"loop$9D\", row_args = [{tail = 0}]}",
        ],
    );
}

/// A struct layout whose field is a callable keeps the field's type, and a
/// call of the field takes the class of its row.
#[test]
fn a_layout_keeps_its_callable_field() {
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
            "!Holder = adt.struct<Holder(callback: !cb)>",
            "%1 = adt.struct_get %0 {field = 0, type = !Holder} : !cb",
            "%3 = adt.struct_new %1 {type = !Holder} : !HolderRef",
            "%5 = tribute_cps.call_indirect %4, %0 {convention = \"direct\"} : core.i32",
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
            "tribute_cps.func @each['a0](%0: core.i32) -> core.i32 row(| 'a0) inferred convention(direct) {",
            "tribute_cps.func @ask['a0](%0: core.i32) -> core.i32 row(@Ctl | 'a0) inferred convention(cps) {",
        ],
    );
}

#[test]
fn a_specialized_module_is_rejected() {
    let mut ctx = IrContext::new();
    let module = parse_test_module(
        &mut ctx,
        r#"core.module @m {
  tribute_cps.func @main(%x: core.i32) -> core.i32 convention(direct) {
    tribute_control.return %x
  }
}"#,
    );
    let module =
        trunk_ir::dialect::core::Module::from_op(&ctx, module.op()).expect("a core.module");
    let mut pm = trunk_ir::pass::PassManager::new();
    pm.add_pass(SpecializeConventionClasses);
    let error = pm
        .run(
            &mut ctx,
            module,
            &mut trunk_ir::analysis::AnalysisCache::new(),
        )
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("specialize-convention-classes")
            && error.contains("takes a module without tribute_cps operations"),
        "{error}"
    );
}

/// A lambda has a body for each class its instantiations select, in one
/// value that every instantiation uses. A lambda's class can depend on a
/// lambda defined after it.
#[test]
fn a_lambda_has_a_body_per_class_of_its_instantiations() {
    let printed = specialized(
        r#"core.module @m {
  !l0 = tribute_control.func_sig<(core.i32) -> core.i32>
  !lc = tribute_control.func_sig<(core.i32) -> core.i32, {ability_row = {instances = [@Ctl]}}>
  !l1 = tribute_control.func_sig<(core.i32) -> core.i32, {ability_row = {tail = 1}}>
  !h0 = tribute_control.func_sig<() -> core.i32>
  ability.decl {sym_name = "Ctl"} {
    ability.operation {kind = "op", op_name = "ctl", param_types = [], result_type = core.i32}
  }
  tribute_control.func @f[local 'a0, local 'a1](%x: core.i32) -> core.i32 row(@Ctl) {
    %g = tribute_control.lambda['a0](%y: core.i32) -> core.i32 row(| 'a0) captures [] {
      tribute_control.return %y
    }
    %h = tribute_control.lambda['a1]() -> core.i32 row(| 'a1) captures [%g] {
      %s = tribute_control.instantiate %g {row_args = [{tail = 1}]} : !l1
      %c = arith.const {value = 1} : core.i32
      %v = tribute_control.call_indirect %s, %c : core.i32
      tribute_control.return %v
    }
    %a = tribute_control.instantiate %g {row_args = [{}]} : !l0
    %i = tribute_control.instantiate %h {row_args = [{}]} : !h0
    %r = tribute_control.call_indirect %i : core.i32
    %b = tribute_control.instantiate %g {row_args = [{instances = [@Ctl]}]} : !lc
    %t = tribute_control.call_indirect %b, %r : core.i32
    tribute_control.return %t
  }
}"#,
    );
    assert_contains(
        &printed,
        &[
            "tribute_cps.func @f[local 'a0, local 'a1](%0: core.i32) -> core.i32 row(@Ctl) convention(cps) {",
            "%1 = tribute_cps.lambda['a0](%2: core.i32) -> core.i32 row(| 'a0) captures [] entries {\n        direct {\n            tribute_control.return %2\n        }\n        cps {",
            "%3 = tribute_cps.lambda['a1]() -> core.i32 row(| 'a1) captures [%1] entries {\n        direct {",
            "%4 = tribute_control.instantiate %1 {row_args = [{tail = 1}]}",
            "%7 = tribute_control.instantiate %1 {row_args = [{}]}",
            "%10 = tribute_control.instantiate %1 {row_args = [{instances = [@Ctl]}]} : !lc",
            "%11 = tribute_cps.call_indirect %10, %9 {convention = \"cps\"} : core.i32",
        ],
    );
}

/// Each instance gives its own lambdas their bodies: a lambda's body of a
/// class copies the lambdas inside it with their own binders, and an
/// instance whose instantiations agree keeps one body.
#[test]
fn each_instance_copies_the_lambdas_inside_each_body() {
    let printed = specialized(
        r#"core.module @m {
  !cb = tribute_control.func_sig<(core.i32) -> core.i32, {ability_row = {tail = 0}}>
  !in1 = tribute_control.func_sig<(core.i32) -> core.i32, {ability_row = {tail = 1}}>
  !pure = tribute_control.func_sig<(core.i32) -> core.i32>
  tribute_control.func @app['a0](%x: core.i32, %f: !cb) -> core.i32 row(| 'a0) {
    %r = tribute_control.call_indirect %f, %x : core.i32
    tribute_control.return %r
  }
  tribute_control.func @run['a0, local 'a1, local 'a2](%x: core.i32, %k: !cb) -> core.i32 row(| 'a0) {
    %o = tribute_control.lambda['a1](%y: core.i32) -> core.i32 row(| 'a1) captures [] {
      %n = tribute_control.lambda['a2](%z: core.i32) -> core.i32 row(| 'a2) captures [] {
        tribute_control.return %z
      }
      %m = tribute_control.instantiate %n {row_args = [{tail = 1}]} : !in1
      %v = tribute_control.call_indirect %m, %y : core.i32
      tribute_control.return %v
    }
    %p = tribute_control.instantiate %o {row_args = [{}]} : !pure
    %q = tribute_control.instantiate %o {row_args = [{tail = 0}]} : !cb
    %r1 = tribute_control.call %x, %p {callee = @app, row_args = [{}]} : core.i32
    %r2 = tribute_control.call %r1, %q {callee = @app, row_args = [{tail = 0}]} : core.i32
    %r3 = tribute_control.call_indirect %k, %r2 : core.i32
    tribute_control.return %r3
  }
  tribute_control.func @main(%x: core.i32) -> core.i32 {
    %l = tribute_control.lambda(%y: core.i32) -> core.i32 captures [] {
      tribute_control.return %y
    }
    %r = tribute_control.call %x, %l {callee = @run, row_args = [{}]} : core.i32
    tribute_control.return %r
  }
}"#,
    );
    assert_contains(
        &printed,
        &[
            "tribute_cps.func @run['a0, local 'a1, local 'a2, local 'a3](%0: core.i32, %1: !cb) -> core.i32 row(| 'a0) convention(cps) {",
            "%4 = tribute_cps.lambda['a2](%5: core.i32) -> core.i32 row(| 'a2) captures [] entries {\n              direct {",
            "%8 = tribute_cps.lambda['a3](%9: core.i32) -> core.i32 row(| 'a3) captures [] entries {\n              cps {",
            "%10 = tribute_control.instantiate %8 {row_args = [{tail = 1}]} : !in1",
            "%13 = tribute_control.instantiate %2 {row_args = [{tail = 0}]} : !cb",
            "tribute_cps.func @\"run$9D\"['a0, local 'a1, local 'a2](%0: core.i32, %1: !cb) -> core.i32 row(| 'a0) convention(direct) {",
            "%9 = tribute_control.instantiate %2 {row_args = [{tail = 0}]} : !cb",
        ],
    );
}

/// The instantiations of a lambda inside other lambdas use the one value
/// that has a body per class, which those lambdas capture once.
#[test]
fn a_lambda_using_a_lambda_of_two_classes_captures_it_once() {
    let printed = specialized(
        r#"core.module @m {
  !l0 = tribute_control.func_sig<(core.i32) -> core.i32>
  !lc = tribute_control.func_sig<(core.i32) -> core.i32, {ability_row = {instances = [@Ctl]}}>
  !hc = tribute_control.func_sig<() -> core.i32, {ability_row = {instances = [@Ctl]}}>
  ability.decl {sym_name = "Ctl"} {
    ability.operation {kind = "op", op_name = "ctl", param_types = [], result_type = core.i32}
  }
  tribute_control.func @f[local 'a0](%x: core.i32) -> core.i32 row(@Ctl) {
    %g = tribute_control.lambda['a0](%y: core.i32) -> core.i32 row(| 'a0) captures [] {
      tribute_control.return %y
    }
    %a = tribute_control.instantiate %g {row_args = [{}]} : !l0
    %r = tribute_control.call_indirect %a, %x : core.i32
    %h = tribute_control.lambda() -> core.i32 row(@Ctl) captures [%g, %r] {
      %s = tribute_control.instantiate %g {row_args = [{instances = [@Ctl]}]} : !lc
      %v = tribute_control.call_indirect %s, %r : core.i32
      tribute_control.return %v
    }
    %k = tribute_control.lambda() -> core.i32 row(@Ctl) captures [%g, %r] {
      %p = tribute_control.instantiate %g {row_args = [{}]} : !l0
      %q = tribute_control.instantiate %g {row_args = [{instances = [@Ctl]}]} : !lc
      %u = tribute_control.call_indirect %p, %r : core.i32
      %w = tribute_control.call_indirect %q, %u : core.i32
      tribute_control.return %w
    }
    %b = tribute_control.call_indirect %h : core.i32
    %c = tribute_control.call_indirect %k : core.i32
    tribute_control.return %c
  }
}"#,
    );
    assert_contains(
        &printed,
        &[
            "tribute_cps.func @f[local 'a0](%0: core.i32) -> core.i32 row(@Ctl) convention(cps) {",
            "%1 = tribute_cps.lambda['a0](%2: core.i32) -> core.i32 row(| 'a0) captures [] entries {\n        direct {",
            "%5 = tribute_cps.lambda() -> core.i32 row(@Ctl) captures [%1, %4] entries {\n        cps {",
            "%8 = tribute_cps.lambda() -> core.i32 row(@Ctl) captures [%1, %4] entries {\n        cps {",
        ],
    );
}

/// A lambda with no row binder of its own is one value: passed both to a
/// pure consumer and to an open one at the closed row, it stays a single
/// `Direct` lambda, and the open consumer is its `Direct` instance.
#[test]
fn a_lambda_without_binders_is_shared_by_its_consumers() {
    let printed = specialized(
        r#"core.module @m {
  !cb = tribute_control.func_sig<(core.i32) -> core.i32, {ability_row = {tail = 0}}>
  !pure = tribute_control.func_sig<(core.i32) -> core.i32>
  tribute_control.func @pure(%f: !pure, %x: core.i32) -> core.i32 {
    %r = tribute_control.call_indirect %f, %x : core.i32
    tribute_control.return %r
  }
  tribute_control.func @open['a0](%f: !cb, %x: core.i32) -> core.i32 row(| 'a0) {
    %r = tribute_control.call_indirect %f, %x : core.i32
    tribute_control.return %r
  }
  tribute_control.func @main(%x: core.i32) -> core.i32 {
    %l = tribute_control.lambda(%y: core.i32) -> core.i32 captures [] {
      tribute_control.return %y
    }
    %a = tribute_control.call %l, %x {callee = @pure} : core.i32
    %b = tribute_control.call %l, %a {callee = @open, row_args = [{}]} : core.i32
    tribute_control.return %b
  }
}"#,
    );
    let main = &printed[printed
        .find("tribute_cps.func @main")
        .expect("main is printed")..];
    assert_eq!(main.matches("tribute_cps.lambda").count(), 1, "{printed}");
    assert_contains(
        main,
        &[
            "%1 = tribute_cps.lambda(%2: core.i32) -> core.i32 captures [] entries {\n        direct {",
            "%3 = tribute_control.call %1, %0 {callee = @pure} : core.i32",
            "%4 = tribute_control.call %1, %3 {callee = @\"open$9D\", row_args = [{}]} : core.i32",
        ],
    );
}

/// Types outside definitions keep their shape, a bodyless definition has no
/// class variables, a reference to root `main` stays on it, and an instance
/// of the last definition follows it.
#[test]
fn module_level_types_and_references_to_main_keep_their_shape() {
    let printed = specialized(
        r#"core.module @m {
  !cb = tribute_control.func_sig<(core.i32) -> core.i32, {ability_row = {tail = 0}}>
  !pure = tribute_control.func_sig<(core.i32) -> core.i32>
  ability.decl {sym_name = "Map"} {
    ability.operation {kind = "fn", op_name = "map", param_types = [!pure], result_type = core.i32}
  }
  tribute_control.func @ext(%x: core.i32) -> core.i32 attributes {abi = "C"}
  tribute_control.func @main(%x: core.i32) -> core.i32 {
    %m = tribute_control.func_ref {func_ref = @main} : !pure
    %f = tribute_control.func_ref {func_ref = @ext} : !pure
    %r = tribute_control.call %x, %f {callee = @app, row_args = [{}]} : core.i32
    tribute_control.return %r
  }
  tribute_control.func @app['a0](%x: core.i32, %f: !cb) -> core.i32 row(| 'a0) {
    %r = tribute_control.call_indirect %f, %x : core.i32
    tribute_control.return %r
  }
}"#,
    );
    assert_contains(
        &printed,
        &[
            "ability.operation {kind = \"fn\", op_name = \"map\", param_types = [!pure], result_type = core.i32}",
            "tribute_cps.func @ext(%arg0: core.i32) -> core.i32 convention(direct) attributes {abi = \"C\"}",
            "%1 = tribute_cps.func_ref {func_ref = @\"main$8\"} : !pure",
            "tribute_cps.func_table {conventions = [\"direct\"], sym_name = \"main$8\", targets = [@main]}",
            "callee = @\"app$9D\"",
        ],
    );
    let base = printed
        .find("tribute_cps.func @app[")
        .expect("the base stays");
    let instance = printed
        .find("tribute_cps.func @\"app$9D\"")
        .expect("the instance");
    assert!(base < instance, "{printed}");
}

/// An instantiation's class can come from a class variable: the base
/// instance gives the lambda a second body, which the `Direct` instance
/// does not.
#[test]
fn an_instantiation_reads_the_class_variables_of_its_instance() {
    let printed = specialized(
        r#"core.module @m {
  !cb = tribute_control.func_sig<(core.i32) -> core.i32, {ability_row = {tail = 0}}>
  !pure = tribute_control.func_sig<(core.i32) -> core.i32>
  tribute_control.func @f['a0, local 'a1](%x: core.i32) -> core.i32 row(| 'a0) {
    %g = tribute_control.lambda['a1](%y: core.i32) -> core.i32 row(| 'a1) captures [] {
      tribute_control.return %y
    }
    %k = tribute_control.lambda(%y: core.i32) -> core.i32 row(| 'a0) captures [%g] {
      %s = tribute_control.instantiate %g {row_args = [{tail = 0}]} : !cb
      %v = tribute_control.call_indirect %s, %y : core.i32
      tribute_control.return %v
    }
    %a = tribute_control.instantiate %g {row_args = [{}]} : !pure
    %r = tribute_control.call_indirect %a, %x : core.i32
    %t = tribute_control.call_indirect %k, %r : core.i32
    tribute_control.return %t
  }
  tribute_control.func @main(%x: core.i32) -> core.i32 {
    %r = tribute_control.call %x {callee = @f, row_args = [{}]} : core.i32
    tribute_control.return %r
  }
}"#,
    );
    assert_contains(
        &printed,
        &[
            "tribute_cps.func @f['a0, local 'a1](%0: core.i32) -> core.i32 row(| 'a0) convention(cps) {",
            "%1 = tribute_cps.lambda['a1](%2: core.i32) -> core.i32 row(| 'a1) captures [] entries {\n        direct {\n            tribute_control.return %2\n        }\n        cps {",
            "%7 = tribute_control.instantiate %1 {row_args = [{}]}",
            "tribute_cps.func @\"f$9D\"['a0, local 'a1](%0: core.i32) -> core.i32 row(| 'a0) convention(direct) {\n      %1 = tribute_cps.lambda['a1](%2: core.i32) -> core.i32 row(| 'a1) captures [] entries {\n        direct {\n            tribute_control.return %2\n        }\n      }",
        ],
    );
}

/// A reference inside a lambda selects its callee's instance with the
/// lambda's settled class, here `EvidenceDirect`.
#[test]
fn a_reference_in_a_lambda_selects_by_the_local_class() {
    let printed = specialized(
        r#"core.module @m {
  !cb = tribute_control.func_sig<(core.i32) -> core.i32, {ability_row = {tail = 0}}>
  !ask = tribute_control.func_sig<(core.i32) -> core.i32, {ability_row = {instances = [@Ask]}}>
  !ga = tribute_control.func_sig<(core.i32, !ask) -> core.i32, {ability_row = {instances = [@Ask]}}>
  ability.decl {sym_name = "Ask"} {
    ability.operation {kind = "fn", op_name = "ask", param_types = [], result_type = core.i32}
  }
  tribute_control.func @app['a0](%x: core.i32, %f: !cb) -> core.i32 row(| 'a0) {
    %r = tribute_control.call_indirect %f, %x : core.i32
    tribute_control.return %r
  }
  tribute_control.func @main[local 'a0](%x: core.i32) -> core.i32 row(@Ask) {
    %g = tribute_control.lambda['a0](%y: core.i32, %f: !cb) -> core.i32 row(| 'a0) captures [] {
      %v = tribute_control.call %y, %f {callee = @app, row_args = [{tail = 0}]} : core.i32
      tribute_control.return %v
    }
    %cb = tribute_control.lambda(%z: core.i32) -> core.i32 row(@Ask) captures [] {
      tribute_control.return %z
    }
    %a = tribute_control.instantiate %g {row_args = [{instances = [@Ask]}]} : !ga
    %r = tribute_control.call_indirect %a, %x, %cb : core.i32
    tribute_control.return %r
  }
}"#,
    );
    assert_contains(
        &printed,
        &[
            "tribute_cps.func @\"app$9E\"['a0]",
            "%4 = tribute_control.call %2, %3 {callee = @\"app$9E\", row_args = [{tail = 0}]} : core.i32",
        ],
    );
}

/// A lambda whose classes depend on an enclosing lambda's binder reads that
/// binder in each body of the enclosing lambda, so each body holds its own
/// lambda with the bodies its classes select there.
#[test]
fn a_lambda_reads_an_enclosing_binder_in_each_body() {
    let printed = specialized(
        r#"core.module @m {
  !pure = tribute_control.func_sig<(core.i32) -> core.i32>
  !in0 = tribute_control.func_sig<(core.i32) -> core.i32, {ability_row = {tail = 0}}>
  !ctl = tribute_control.func_sig<(core.i32) -> core.i32, {ability_row = {instances = [@Ctl]}}>
  ability.decl {sym_name = "Ctl"} {
    ability.operation {kind = "op", op_name = "ctl", param_types = [], result_type = core.i32}
  }
  tribute_control.func @f[local 'a0, local 'a1](%x: core.i32) -> core.i32 row(@Ctl) {
    %o = tribute_control.lambda['a0](%y: core.i32) -> core.i32 row(| 'a0) captures [] {
      %n = tribute_control.lambda['a1](%z: core.i32) -> core.i32 row(| 'a1) captures [] {
        tribute_control.return %z
      }
      %d = tribute_control.instantiate %n {row_args = [{}]} : !pure
      %u = tribute_control.instantiate %n {row_args = [{tail = 0}]} : !in0
      %v = tribute_control.call_indirect %d, %y : core.i32
      %w = tribute_control.call_indirect %u, %v : core.i32
      tribute_control.return %w
    }
    %p = tribute_control.instantiate %o {row_args = [{}]} : !pure
    %q = tribute_control.instantiate %o {row_args = [{instances = [@Ctl]}]} : !ctl
    %r1 = tribute_control.call_indirect %p, %x : core.i32
    %r2 = tribute_control.call_indirect %q, %r1 : core.i32
    tribute_control.return %r2
  }
}"#,
    );
    assert_contains(
        &printed,
        &[
            "tribute_cps.func @f[local 'a0, local 'a1, local 'a2](%0: core.i32) -> core.i32 row(@Ctl) convention(cps) {",
            "%3 = tribute_cps.lambda['a1](%4: core.i32) -> core.i32 row(| 'a1) captures [] entries {\n              direct {\n                  tribute_control.return %4\n              }\n            }",
            "%8 = tribute_cps.call_indirect %6, %7 {convention = \"direct\"} : core.i32",
            "%9 = tribute_cps.lambda['a2](%10: core.i32) -> core.i32 row(| 'a2) captures [] entries {\n              direct {\n                  tribute_control.return %10\n              }\n              cps {",
            "%14 = tribute_cps.call_indirect %12, %13 {convention = \"cps\"} : core.i32",
        ],
    );
}

/// A lambda split before an enclosing lambda is copied stays one value in
/// each copy: the inner lambda splits first, since the outer one waits for
/// the lambda its use reads, and the copy of the outer lambda folds its own
/// copy of the inner lambda.
#[test]
fn a_lambda_split_before_its_enclosing_lambda_is_one_value_in_each_copy() {
    let printed = specialized(
        r#"core.module @m {
  !pure = tribute_control.func_sig<(core.i32) -> core.i32>
  !ctl = tribute_control.func_sig<(core.i32) -> core.i32, {ability_row = {instances = [@Ctl]}}>
  !in0 = tribute_control.func_sig<(core.i32) -> core.i32, {ability_row = {tail = 0}}>
  !h0 = tribute_control.func_sig<() -> core.i32>
  !hc = tribute_control.func_sig<() -> core.i32, {ability_row = {instances = [@Ctl]}}>
  ability.decl {sym_name = "Ctl"} {
    ability.operation {kind = "op", op_name = "ctl", param_types = [], result_type = core.i32}
  }
  tribute_control.func @f[local 'a0, local 'a1, local 'a2](%x: core.i32) -> core.i32 row(@Ctl) {
    %o = tribute_control.lambda['a1](%y: core.i32) -> core.i32 row(| 'a1) captures [] {
      %n = tribute_control.lambda['a2](%z: core.i32) -> core.i32 row(| 'a2) captures [] {
        tribute_control.return %z
      }
      %d = tribute_control.instantiate %n {row_args = [{}]} : !pure
      %u = tribute_control.instantiate %n {row_args = [{instances = [@Ctl]}]} : !ctl
      %v = tribute_control.call_indirect %d, %y : core.i32
      tribute_control.return %v
    }
    %w = tribute_control.lambda['a0]() -> core.i32 row(| 'a0) captures [%o] {
      %s = tribute_control.instantiate %o {row_args = [{tail = 0}]} : !in0
      %c = arith.const {value = 1} : core.i32
      %v = tribute_control.call_indirect %s, %c : core.i32
      tribute_control.return %v
    }
    %a = tribute_control.instantiate %w {row_args = [{}]} : !h0
    %b = tribute_control.instantiate %w {row_args = [{instances = [@Ctl]}]} : !hc
    %r = tribute_control.call_indirect %a : core.i32
    %t = tribute_control.call_indirect %b : core.i32
    tribute_control.return %t
  }
}"#,
    );
    // `w`, `o`, and one `n` in each body of `o`.
    assert_eq!(
        printed.matches("tribute_cps.lambda").count(),
        4,
        "{printed}"
    );
}

/// Dropping the binders of folded copies keeps the definition's row unions
/// on the binders they name.
#[test]
fn folding_keeps_the_row_unions_of_the_definition() {
    let printed = specialized(
        r#"core.module @m {
  !l0 = tribute_control.func_sig<(core.i32) -> core.i32>
  !lc = tribute_control.func_sig<(core.i32) -> core.i32, {ability_row = {instances = [@Ctl]}}>
  ability.decl {sym_name = "Ctl"} {
    ability.operation {kind = "op", op_name = "ctl", param_types = [], result_type = core.i32}
  }
  tribute_control.func @f['a0 = 'a1 | 'a2, 'a1, 'a2, local 'a3](%x: core.i32) -> core.i32 row(@Ctl | 'a0) {
    %g = tribute_control.lambda['a3](%y: core.i32) -> core.i32 row(| 'a3) captures [] {
      tribute_control.return %y
    }
    %a = tribute_control.instantiate %g {row_args = [{}]} : !l0
    %r = tribute_control.call_indirect %a, %x : core.i32
    %b = tribute_control.instantiate %g {row_args = [{instances = [@Ctl]}]} : !lc
    %t = tribute_control.call_indirect %b, %r : core.i32
    tribute_control.return %t
  }
}"#,
    );
    assert_contains(
        &printed,
        &[
            "tribute_cps.func @f['a0 = 'a1 | 'a2, 'a1, 'a2, local 'a3](%0: core.i32) -> core.i32 row(@Ctl | 'a0) convention(cps) {",
            "%1 = tribute_cps.lambda['a3](%2: core.i32) -> core.i32 row(| 'a3) captures [] entries {\n        direct {",
        ],
    );
}

/// A lambda's parameter types name its own binders in every body.
#[test]
fn every_body_takes_parameters_over_the_lambdas_own_binders() {
    let printed = specialized(
        r#"core.module @m {
  !l0 = tribute_control.func_sig<(core.i32) -> core.i32>
  !lc = tribute_control.func_sig<(core.i32) -> core.i32, {ability_row = {instances = [@Ctl]}}>
  !k0 = tribute_control.func_sig<(core.i32) -> core.i32, {ability_row = {tail = 0}}>
  !g0 = tribute_control.func_sig<(core.i32, !l0) -> core.i32>
  !gc = tribute_control.func_sig<(core.i32, !lc) -> core.i32, {ability_row = {instances = [@Ctl]}}>
  ability.decl {sym_name = "Ctl"} {
    ability.operation {kind = "op", op_name = "ctl", param_types = [], result_type = core.i32}
  }
  tribute_control.func @f[local 'a0](%x: core.i32) -> core.i32 row(@Ctl) {
    %g = tribute_control.lambda['a0](%y: core.i32, %k: !k0) -> core.i32 row(| 'a0) captures [] {
      %v = tribute_control.call_indirect %k, %y : core.i32
      tribute_control.return %v
    }
    %p = tribute_control.lambda(%n: core.i32) -> core.i32 captures [] {
      tribute_control.return %n
    }
    %q = tribute_control.lambda(%n: core.i32) -> core.i32 row(@Ctl) captures [] {
      tribute_control.return %n
    }
    %a = tribute_control.instantiate %g {row_args = [{}]} : !g0
    %b = tribute_control.instantiate %g {row_args = [{instances = [@Ctl]}]} : !gc
    %r = tribute_control.call_indirect %a, %x, %p : core.i32
    %t = tribute_control.call_indirect %b, %r, %q : core.i32
    tribute_control.return %t
  }
}"#,
    );
    assert_contains(
        &printed,
        &[
            "tribute_cps.func @f[local 'a0](%0: core.i32) -> core.i32 row(@Ctl) convention(cps) {",
            "%1 = tribute_cps.lambda['a0](%2: core.i32, %3: !k0) -> core.i32 row(| 'a0) captures [] entries {\n        direct {",
            "%5 = tribute_cps.call_indirect %3, %2 {convention = \"cps\"} : core.i32",
        ],
    );
}

/// Instantiations that select one class but pass other classes for the
/// lambda's binder still split the value: a lambda whose own row always
/// requires `Cps` is copied with a new binder for the second combination.
#[test]
fn instantiations_of_one_class_with_other_binder_classes_split_the_value() {
    let printed = specialized(
        r#"core.module @m {
  !pure = tribute_control.func_sig<(core.i32) -> core.i32>
  !l0 = tribute_control.func_sig<(core.i32) -> core.i32, {ability_row = {instances = [@Ctl]}}>
  !la = tribute_control.func_sig<(core.i32) -> core.i32, {ability_row = {instances = [@Ask, @Ctl]}}>
  !in0 = tribute_control.func_sig<(core.i32) -> core.i32, {ability_row = {tail = 0}}>
  ability.decl {sym_name = "Ask"} {
    ability.operation {kind = "fn", op_name = "ask", param_types = [], result_type = core.i32}
  }
  ability.decl {sym_name = "Ctl"} {
    ability.operation {kind = "op", op_name = "ctl", param_types = [], result_type = core.i32}
  }
  tribute_control.func @f[local 'a0, local 'a1](%x: core.i32) -> core.i32 row(@Ctl) {
    %g = tribute_control.lambda['a0](%y: core.i32) -> core.i32 row(@Ctl | 'a0) captures [] {
      %n = tribute_control.lambda['a1](%z: core.i32) -> core.i32 row(| 'a1) captures [] {
        tribute_control.return %z
      }
      %u = tribute_control.instantiate %n {row_args = [{tail = 0}]} : !in0
      %v = tribute_control.call_indirect %u, %y : core.i32
      tribute_control.return %v
    }
    %a = tribute_control.instantiate %g {row_args = [{}]} : !l0
    %b = tribute_control.instantiate %g {row_args = [{instances = [@Ask]}]} : !la
    %r = tribute_control.call_indirect %a, %x : core.i32
    %t = tribute_control.call_indirect %b, %r : core.i32
    tribute_control.return %t
  }
}"#,
    );
    assert_contains(
        &printed,
        &[
            "tribute_cps.func @f[local 'a0, local 'a1, local 'a2, local 'a3](%0: core.i32) -> core.i32 row(@Ctl) convention(cps) {",
            "%1 = tribute_cps.lambda['a0](%2: core.i32) -> core.i32 row(@Ctl | 'a0) captures [] entries {\n        cps {",
            "convention = \"direct\"",
            "%7 = tribute_cps.lambda['a2](%8: core.i32) -> core.i32 row(@Ctl | 'a2) captures [] entries {\n        cps {",
            "%11 = tribute_control.instantiate %9 {row_args = [{tail = 2}]}",
            "%12 = tribute_cps.call_indirect %11, %8 {convention = \"evidence_direct\"} : core.i32",
            "%14 = tribute_control.instantiate %7 {row_args = [{instances = [@Ask]}]} : !la",
        ],
    );
}

/// The layouts the context publishes keep their callable fields.
#[test]
fn a_published_layout_keeps_its_callable_fields() {
    let mut ctx = IrContext::new();
    let module = parse_test_module(
        &mut ctx,
        r#"core.module @m {
  !cb = tribute_control.func_sig<(core.i32) -> core.i32>
  !Holder = adt.struct<Holder(callback: !cb)>
  tribute_control.func @main(%x: core.i32) -> core.i32 {
    tribute_control.return %x
  }
}"#,
    );
    let name = trunk_ir::Symbol::new("Holder");
    let before = ctx.type_alias_by_name(&name).expect("published layout");
    specialize(&mut ctx, module).expect("specializes");
    assert_eq!(ctx.type_alias_by_name(&name), Some(before));
}
