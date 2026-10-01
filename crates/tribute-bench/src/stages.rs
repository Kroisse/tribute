//! Compilation stages shared by the boundary benchmark and report.
//!
//! Each stage starts from a fresh database, so the frontend stage includes
//! the prelude work of a cold compile. Unlike production compilation, the
//! stages skip the call-arity diagnostic check between the shared middle-end
//! and the target pipeline, since it reports through Salsa accumulators.

use std::path::Path;
use std::process::{Command, Output, Stdio};

use itertools::Itertools;
use salsa::Database;
use tribute::database::parse_with_thread_local;
use tribute::pipeline::{
    FrontendCompilation, compile_frontend_for_shared_route, compile_with_diagnostics,
    emit_from_boundary_exit, run_shared_middle_end, run_target_to_boundary_exit,
};
use tribute::{Rope, SourceCst, TributeDatabaseImpl};
use tribute_passes::abi_boundary::TargetKind;
use trunk_ir::{IrContext, Module};

use crate::programs::Program;

pub const TARGETS: [TargetKind; 2] = [TargetKind::Native, TargetKind::Wasm];

pub fn target_name(target: TargetKind) -> &'static str {
    match target {
        TargetKind::Native => "native",
        TargetKind::Wasm => "wasm",
    }
}

/// Run the frontend on a fresh database.
pub fn frontend(program: &Program) -> Result<FrontendCompilation, String> {
    let rope = Rope::from_str(program.source);
    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&rope, None);
        let source = SourceCst::from_path(db, program.name, rope.clone(), tree);
        compile_frontend_for_shared_route(db, source).ok_or_else(|| {
            let diagnostics = compile_with_diagnostics(db, source).diagnostics;
            format!(
                "frontend failed: {}",
                diagnostics
                    .iter()
                    .format_with("; ", |diagnostic, f| f(&diagnostic.inner.message))
            )
        })
    })
}

pub fn shared_middle_end(frontend: FrontendCompilation) -> Result<(IrContext, Module), String> {
    run_shared_middle_end(frontend).map_err(|error| format!("shared middle-end: {error}"))
}

pub fn to_boundary_exit(
    ctx: &mut IrContext,
    module: Module,
    target: TargetKind,
) -> Result<(), String> {
    run_target_to_boundary_exit(ctx, module, target)
        .map_err(|error| format!("{} boundary: {error}", target_name(target)))
}

pub fn after_boundary_exit(
    ctx: &mut IrContext,
    module: Module,
    target: TargetKind,
) -> Result<Vec<u8>, String> {
    emit_from_boundary_exit(ctx, module, target).map_err(|error| error.to_string())
}

/// The shared middle-end output for `program`.
pub fn through_shared(program: &Program) -> Result<(IrContext, Module), String> {
    shared_middle_end(frontend(program)?)
}

/// `program` at `target`'s boundary exit.
pub fn through_boundary_exit(
    program: &Program,
    target: TargetKind,
) -> Result<(IrContext, Module), String> {
    let (mut ctx, module) = through_shared(program)?;
    to_boundary_exit(&mut ctx, module, target)?;
    Ok((ctx, module))
}

/// `program`'s emitted native object or Wasm module.
pub fn emitted(program: &Program, target: TargetKind) -> Result<Vec<u8>, String> {
    let (mut ctx, module) = through_boundary_exit(program, target)?;
    after_boundary_exit(&mut ctx, module, target)
}

/// Link a native object into `output`.
pub fn link(object: &[u8], output: &Path) -> Result<(), String> {
    tribute::link_native_binary(object, output, None).map_err(|error| error.to_string())
}

/// Run an executable with `stdin` and wait for it.
pub fn run(command: &mut Command, stdin: &[u8]) -> std::io::Result<Output> {
    use std::io::Write;

    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let mut child_stdin = child.stdin.take().expect("piped stdin");
    let input = stdin.to_vec();
    let writer = std::thread::spawn(move || child_stdin.write_all(&input));
    let output = child.wait_with_output()?;
    writer.join().expect("stdin writer panicked")?;
    Ok(output)
}
