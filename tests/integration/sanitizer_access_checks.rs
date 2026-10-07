//! Native execution of the address sanitizer's access checks.
//!
//! Tribute source cannot express an invalid access, so these programs are
//! written in the `clif` dialect and instrumented by the same passes a
//! sanitized build runs.

use std::process::Output;

use tribute_passes::native::sanitize_access::{DeclareAccessChecks, InstrumentMemoryAccesses};
use trunk_ir::context::IrContext;
use trunk_ir::dialect::{clif, core};
use trunk_ir::ops::DialectOp;
use trunk_ir::parser::parse_test_module;
use trunk_ir::pass::PassManager;
use trunk_ir_cranelift_backend::emit_module_to_native;

use crate::common::NativeTestBinary;

/// A `main` that allocates a 16-byte object, runs `body` on it and returns 0.
fn program(body: &str) -> String {
    format!(
        r#"core.module @test {{
  clif.func {{abi = "C", sym_name = "__asan_init", type = clif.func_sig<() -> core.nil>}}
  clif.func {{sym_name = "main", type = clif.func_sig<() -> core.i32>}} {{
    ^entry:
      %init = clif.call {{callee = @__asan_init}} : core.nil
      %size = clif.iconst {{value = 16}} : core.i64
      %object = clif.call %size {{callee = @__tribute_alloc}} : core.ptr
      %value = clif.iconst {{value = 7}} : core.i64
{body}
      %code = clif.iconst {{value = 0}} : core.i32
      clif.return %code
  }}
}}"#
    )
}

fn run(body: &str, sanitize: bool) -> Output {
    let mut ctx = IrContext::new();
    let module = parse_test_module(&mut ctx, &program(body));
    if sanitize {
        let mut pm = PassManager::new();
        pm.add_pass(DeclareAccessChecks);
        pm.nest::<clif::Func>().add_pass(InstrumentMemoryAccesses);
        let target = core::Module::from_op(&ctx, module.op()).expect("core.module");
        pm.run(&mut ctx, target, &mut Default::default())
            .expect("module is instrumented");
    }
    let object = emit_module_to_native(&ctx, module).expect("module emits");
    NativeTestBinary::from_object_bytes(&object).run_with_stdin(&[])
}

fn assert_reports(body: &str, report: &str) {
    // Without the checks the access goes through: the allocator's red zones
    // and quarantine keep the memory mapped.
    assert_eq!(run(body, false).status.code(), Some(0));

    let output = run(body, true);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "stderr: {stderr}");
    assert!(stderr.contains(report), "stderr: {stderr}");
}

const FREE: &str =
    "      %freed = clif.call %object, %size {callee = @__tribute_dealloc} : core.nil";

#[test]
fn accesses_inside_a_live_allocation_pass() {
    let output = run(
        &format!(
            r#"      clif.store %value, %object {{offset = 0}}
      clif.store %value, %object {{offset = 8}}
      %read = clif.load %object {{offset = 8}} : core.i64
{FREE}"#
        ),
        true,
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(0), "stderr: {stderr}");
}

#[test]
fn load_past_the_end_of_an_allocation_reports_at_the_access() {
    assert_reports(
        "      %read = clif.load %object {offset = 16} : core.i64",
        "heap-buffer-overflow",
    );
    assert_reports(
        "      %read = clif.load %object {offset = 12} : core.i64",
        "READ of size 8",
    );
}

#[test]
fn store_before_the_start_of_an_allocation_reports_at_the_access() {
    assert_reports(
        "      clif.store %value, %object {offset = -8}",
        "heap-buffer-overflow",
    );
    assert_reports(
        "      clif.store %value, %object {offset = -8}",
        "WRITE of size 8",
    );
}

#[test]
fn access_after_free_reports_at_the_access() {
    assert_reports(
        &format!("{FREE}\n      %read = clif.load %object {{offset = 0}} : core.i64"),
        "heap-use-after-free",
    );
    assert_reports(
        &format!(
            r#"{FREE}
      %one = clif.iconst {{value = 1}} : core.i32
      %old = clif.atomic_rmw %object, %one {{bin_op = "add", offset = 0}} : core.i32"#
        ),
        "WRITE of size 4",
    );
}
