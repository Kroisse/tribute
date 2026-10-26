//! Language server edit latency: the work one edit costs a database that
//! keeps what it computed for the revision before.
//!
//! ```text
//! cargo bench -p tribute-bench --bench editing -- --save-baseline <name>
//! cargo bench -p tribute-bench --bench editing -- --baseline <name>
//! ```
//!
//! Each iteration switches the source between the two texts of an edit
//! (`tribute_bench::editing`) outside the timed region, including the
//! tree-sitter parse, then times one stage:
//!
//! - `editing/<program>/<edit>/diagnostics`: `compile_with_diagnostics`, what
//!   the language server runs after each edit. An edit that introduces an
//!   error is timed in each direction, `diagnostics/introduce` and
//!   `diagnostics/fix`, since only fixing it reaches lowering.
//! - For the `body` edit, its parts:
//!   - `frontend`: parsing to the specialization result, which Salsa caches;
//!   - `lowering`: lowering the cached frontend result to source-logical IR;
//!   - `ir_checks`: the stages after lowering that diagnostics run, class
//!     specialization and the `fn` arm call check, over every function.

use std::cell::RefCell;
use std::time::Duration;

use criterion::measurement::WallTime;
use criterion::{BatchSize, BenchmarkGroup, Criterion, criterion_group, criterion_main};
use salsa::Database;
use tribute::compile_with_diagnostics;
use tribute::parse_and_lower_ast;
use tribute::pipeline::{
    compile_frontend_for_shared_route, prepare_frontend_for_lowering, run_diagnostic_stages,
};
use tribute_bench::editing::{Edit, EditSession, workloads};

/// The edit whose stages are timed separately.
const STAGED_EDIT: &str = "body";

/// Run the frontend queries on the session's current text.
fn frontend(session: &EditSession) {
    session.db().attach(|db| {
        let source = session.source();
        let typed = parse_and_lower_ast(db, source).expect("the program type checks");
        prepare_frontend_for_lowering(db, typed, source).expect("the program specializes");
    });
}

fn bench_edit(group: &mut BenchmarkGroup<'_, WallTime>, workload: &str, edit: &Edit) {
    let name = |stage: &str| format!("{workload}/{}/{stage}", edit.name);
    let session = RefCell::new(EditSession::new(edit));
    let toggle = || session.borrow_mut().toggle();
    let diagnostics = || {
        let session = session.borrow();
        session
            .db()
            .attach(|db| compile_with_diagnostics(db, session.source()))
    };

    if edit.breaks {
        // Introducing an error stops before lowering and fixing it runs the
        // whole pipeline, so the two directions are timed apart: each
        // iteration starts from the other text, already compiled.
        for (direction, edited) in [("introduce", true), ("fix", false)] {
            group.bench_function(name(&format!("diagnostics/{direction}")), |b| {
                b.iter_batched(
                    || {
                        session.borrow_mut().set_edited(!edited);
                        diagnostics();
                        session.borrow_mut().set_edited(edited);
                    },
                    |()| diagnostics(),
                    BatchSize::PerIteration,
                )
            });
        }
    } else {
        group.bench_function(name("diagnostics"), |b| {
            b.iter_batched(toggle, |()| diagnostics(), BatchSize::PerIteration)
        });
    }

    if edit.name == STAGED_EDIT {
        group.bench_function(name("frontend"), |b| {
            b.iter_batched(
                toggle,
                |()| frontend(&session.borrow()),
                BatchSize::PerIteration,
            )
        });
        group.bench_function(name("lowering"), |b| {
            b.iter_batched(
                || {
                    toggle();
                    frontend(&session.borrow());
                },
                |()| {
                    let session = session.borrow();
                    session
                        .db()
                        .attach(|db| compile_frontend_for_shared_route(db, session.source()))
                        .expect("the program lowers")
                },
                BatchSize::PerIteration,
            )
        });
        group.bench_function(name("ir_checks"), |b| {
            b.iter_batched(
                || {
                    toggle();
                    let session = session.borrow();
                    session
                        .db()
                        .attach(|db| compile_frontend_for_shared_route(db, session.source()))
                        .expect("the program lowers")
                },
                |frontend| run_diagnostic_stages(frontend).expect("the checks pass"),
                BatchSize::PerIteration,
            )
        });
    }
}

fn editing(c: &mut Criterion) {
    let mut group = c.benchmark_group("editing");
    for workload in workloads() {
        for edit in &workload.edits {
            bench_edit(&mut group, workload.name, edit);
        }
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
    targets = editing
}
criterion_main!(benches);
