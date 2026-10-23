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
        "tribute_control.func @app['a0](%0: core.i32, %1: tribute_control.func_sig<(core.i32) -> core.i32, {ability_row = {tail = 0}, tribute.calling_convention = \"cps\"}>) -> core.i32 row(| 'a0) convention(cps) {",
        "tribute_control.func @\"app$9D\"['a0](%0: core.i32, %1: tribute_control.func_sig<(core.i32) -> core.i32, {ability_row = {tail = 0}, tribute.calling_convention = \"direct\"}>) -> core.i32 row(| 'a0) convention(direct) {",
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
            "!Holder = adt.struct<Holder(callback: tribute_control.func_sig<(core.i32) -> core.i32, {tribute.calling_convention = \"direct\"}>)>",
            "!t0 = tribute_control.func_sig<(core.i32) -> core.i32, {tribute.calling_convention = \"direct\"}>",
            "%1 = adt.struct_get %0 {field = 0, type = !Holder} : !t0",
            "%3 = adt.struct_new %1 {type = !Holder} : !HolderRef",
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
            && error.contains("takes source signatures without calling conventions"),
        "{error}"
    );
}

/// A lambda is copied for each class its instantiations select; the first
/// in body order keeps the original, and the copy follows it with a new local
/// binder. A lambda's class can depend on a lambda defined after it.
#[test]
fn a_lambda_is_copied_per_class_of_its_instantiations() {
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
            "tribute_control.func @f[local 'a0, local 'a1, local 'a2](%0: core.i32) -> core.i32 row(@Ctl) convention(cps) {",
            "%1 = tribute_control.lambda['a0](%2: core.i32) -> core.i32 row(| 'a0) convention(direct) captures [] {",
            "%3 = tribute_control.lambda['a2](%4: core.i32) -> core.i32 row(| 'a2) convention(cps) captures [] {",
            "%5 = tribute_control.lambda['a1]() -> core.i32 row(| 'a1) convention(direct) captures [%1] {",
            "%6 = tribute_control.instantiate %1 {row_args = [{tail = 1}]}",
            "%9 = tribute_control.instantiate %1 {row_args = [{}]}",
            "%12 = tribute_control.instantiate %3 {row_args = [{instances = [@Ctl]}]} : !t0",
        ],
    );
}

/// Each instance splits its own lambdas: the copy of a lambda copies the
/// lambdas inside it with their own binders, and an instance whose
/// instantiations agree keeps one lambda.
#[test]
fn each_instance_copies_its_lambdas_with_the_lambdas_inside() {
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
            "tribute_control.func @run['a0, local 'a1, local 'a2, local 'a3, local 'a4](%0: core.i32, %1: !t3) -> core.i32 row(| 'a0) convention(cps) {",
            "%8 = tribute_control.lambda['a3](%9: core.i32) -> core.i32 row(| 'a3) convention(cps) captures [] {",
            "%10 = tribute_control.lambda['a4](%11: core.i32) -> core.i32 row(| 'a4) convention(cps) captures [] {",
            "%12 = tribute_control.instantiate %10 {row_args = [{tail = 3}]} : !t7",
            "%15 = tribute_control.instantiate %8 {row_args = [{tail = 0}]} : !t3",
            "tribute_control.func @\"run$9D\"['a0, local 'a1, local 'a2](%0: core.i32, %1: !t2) -> core.i32 row(| 'a0) convention(direct) {",
            "%9 = tribute_control.instantiate %2 {row_args = [{tail = 0}]} : !t2",
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
        .find("tribute_control.func @main")
        .expect("main is printed")..];
    assert_eq!(
        main.matches("tribute_control.lambda").count(),
        1,
        "{printed}"
    );
    assert_contains(
        main,
        &[
            "%1 = tribute_control.lambda(%2: core.i32) -> core.i32 convention(direct) captures [] {",
            "%3 = tribute_control.call %1, %0 {callee = @pure} : core.i32",
            "%4 = tribute_control.call %1, %3 {callee = @\"open$9D\", row_args = [{}]} : core.i32",
        ],
    );
}

/// Types outside definitions take the conventions of their abilities, a
/// bodyless definition has no class variables, a reference to root `main`
/// stays on it, and an instance of the last definition follows it.
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
            "ability.operation {kind = \"fn\", op_name = \"map\", param_types = [!t0], result_type = core.i32}",
            "!t0 = tribute_control.func_sig<(core.i32) -> core.i32, {tribute.calling_convention = \"direct\"}>",
            "tribute_control.func @ext(%arg0: core.i32) -> core.i32 convention(direct) attributes {abi = \"C\"}",
            "%1 = tribute_control.func_ref {func_ref = @main} : !t0",
            "callee = @\"app$9D\"",
        ],
    );
    let base = printed
        .find("tribute_control.func @app[")
        .expect("the base stays");
    let instance = printed
        .find("tribute_control.func @\"app$9D\"")
        .expect("the instance");
    assert!(base < instance, "{printed}");
}

/// An instantiation's class can come from a class variable: the base
/// instance splits the lambda that the `Direct` instance keeps whole.
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
            "tribute_control.func @f['a0, local 'a1, local 'a2](%0: core.i32) -> core.i32 row(| 'a0) convention(cps) {",
            "%1 = tribute_control.lambda['a1](%2: core.i32) -> core.i32 row(| 'a1) convention(cps) captures [] {",
            "%3 = tribute_control.lambda['a2](%4: core.i32) -> core.i32 row(| 'a2) convention(direct) captures [] {",
            "%9 = tribute_control.instantiate %3 {row_args = [{}]}",
            "tribute_control.func @\"f$9D\"['a0, local 'a1](%0: core.i32) -> core.i32 row(| 'a0) convention(direct) {",
            "%7 = tribute_control.instantiate %1 {row_args = [{}]}",
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
            "tribute_control.func @\"app$9E\"['a0]",
            "%4 = tribute_control.call %2, %3 {callee = @\"app$9E\", row_args = [{tail = 0}]} : core.i32",
        ],
    );
}

/// A lambda whose classes depend on an enclosing lambda's binder waits for
/// that lambda to be copied, so each copy of the outer lambda holds one inner
/// lambda per class it uses.
#[test]
fn a_lambda_is_copied_after_the_lambdas_its_classes_depend_on() {
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
            "tribute_control.func @f[local 'a0, local 'a1, local 'a2, local 'a3, local 'a4](%0: core.i32) -> core.i32 row(@Ctl) convention(cps) {",
            "%3 = tribute_control.lambda['a1](%4: core.i32) -> core.i32 row(| 'a1) convention(direct) captures [] {",
            "%6 = tribute_control.instantiate %3 {row_args = [{tail = 0}]}",
            "%11 = tribute_control.lambda['a3](%12: core.i32) -> core.i32 row(| 'a3) convention(direct) captures [] {",
            "%13 = tribute_control.lambda['a4](%14: core.i32) -> core.i32 row(| 'a4) convention(cps) captures [] {",
            "%16 = tribute_control.instantiate %13 {row_args = [{tail = 2}]}",
        ],
    );
}

/// A layout published as a type alias takes the conventions of its callable
/// fields even where no operation names it.
#[test]
fn a_published_layout_takes_the_conventions_of_its_fields() {
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
    specialize(&mut ctx, module).expect("specializes");
    let holder = ctx
        .type_alias_by_name(&trunk_ir::Symbol::new("Holder"))
        .expect("published layout");
    let field = ctx.get_type(holder).params[0];
    assert_eq!(
        tribute_control::func_sig_convention(&ctx, field),
        Some(CallingConvention::Direct)
    );
}
