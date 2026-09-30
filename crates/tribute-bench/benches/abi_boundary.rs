//! Compile-time and runtime benchmarks around the representation/ABI boundary.
//!
//! ```text
//! cargo bench -p tribute-bench --bench abi_boundary -- --save-baseline <name>
//! cargo bench -p tribute-bench --bench abi_boundary -- --baseline <name>
//! ```
//!
//! Compile stages are timed separately: frontend, shared middle-end, target
//! pipeline to the boundary exit, and lowering and emission after it. The
//! frontend and shared middle-end are timed for representative programs only
//! (`FRONT_STAGE_PROGRAMS`). Each stage's input is prepared outside the timed
//! region. Native runtime is timed on a linked executable, which needs the
//! development sysroot (`cargo xtask runtime`). Allocations and code sizes need no statistics and
//! are reported by the `abi_boundary_report` binary of this crate.
//!
//! CI runs the compile stages under CodSpeed (`.github/workflows/codspeed.yml`),
//! which counts simulated CPU work instead of wall time.

use std::process::Command;
use std::time::Duration;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use tribute_passes::abi_boundary::TargetKind;

use tribute_bench::programs::{PROGRAMS, Program};
use tribute_bench::stages::{self, TARGETS, target_name};

/// Programs whose frontend and shared middle-end are timed. Those stages cost
/// nearly the same for every program here, so one pure and one effectful
/// program represent them.
const FRONT_STAGE_PROGRAMS: [&str; 2] = ["fibonacci", "state_handler"];

fn compile_stages(c: &mut Criterion) {
    for program in PROGRAMS {
        if let Err(error) = stages::through_shared(program) {
            eprintln!("skipping {}: {error}", program.name);
            continue;
        }
        let mut group = c.benchmark_group(format!("compile/{}", program.name));
        if FRONT_STAGE_PROGRAMS.contains(&program.name) {
            group.bench_function("frontend", |b| {
                b.iter_batched(
                    || (),
                    |()| stages::frontend(program).expect("frontend"),
                    BatchSize::PerIteration,
                )
            });
            group.bench_function("shared_middle_end", |b| {
                b.iter_batched(
                    || stages::frontend(program).expect("frontend"),
                    |frontend| stages::shared_middle_end(frontend).expect("shared middle-end"),
                    BatchSize::PerIteration,
                )
            });
        }
        for target in TARGETS {
            bench_target(&mut group, program, target);
        }
        group.finish();
    }
}

fn bench_target(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    program: &Program,
    target: TargetKind,
) {
    let name = target_name(target);
    if let Err(error) = stages::emitted(program, target) {
        eprintln!("skipping {}/{name}: {error}", program.name);
        return;
    }
    group.bench_function(format!("{name}/to_boundary_exit"), |b| {
        b.iter_batched(
            || stages::through_shared(program).expect("shared middle-end"),
            |(mut ctx, module)| {
                stages::to_boundary_exit(&mut ctx, module, target).expect("boundary exit");
                ctx
            },
            BatchSize::PerIteration,
        )
    });
    group.bench_function(format!("{name}/after_boundary_exit"), |b| {
        b.iter_batched(
            || stages::through_boundary_exit(program, target).expect("boundary exit"),
            |(mut ctx, module)| {
                stages::after_boundary_exit(&mut ctx, module, target).expect("emission");
                ctx
            },
            BatchSize::PerIteration,
        )
    });
}

fn native_runtime(c: &mut Criterion) {
    // CodSpeed instruments this process only, so it cannot see the time spent
    // in a child executable.
    if cfg!(codspeed) {
        return;
    }
    let dir = tempfile::tempdir().expect("temporary directory");
    let mut group = c.benchmark_group("run/native");
    for program in PROGRAMS {
        let executable = dir.path().join(program.name);
        let linked = stages::emitted(program, TargetKind::Native)
            .and_then(|object| stages::link(&object, &executable));
        if let Err(error) = linked {
            eprintln!("skipping native run of {}: {error}", program.name);
            continue;
        }
        group.bench_function(program.name, |b| {
            b.iter(|| {
                let output = stages::run(&mut Command::new(&executable), program.stdin)
                    .expect("run native executable");
                assert!(output.status.success(), "{} failed", program.name);
            })
        });
    }
    group.finish();
}

fn config() -> Criterion {
    Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
}

criterion_group! {
    name = benches;
    config = config();
    targets = compile_stages, native_runtime
}
criterion_main!(benches);
