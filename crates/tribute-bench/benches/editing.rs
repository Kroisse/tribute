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
//!   the language server runs after each edit.
//! - For the `body` edit, its parts:
//!   - `frontend`: parsing to the specialization result, which Salsa caches;
//!   - `lowering`: lowering the cached frontend result to source-logical IR;
//!   - `middle_end`: the shared middle-end over every function, as
//!     diagnostics run it.

use std::cell::RefCell;
use std::time::Duration;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use salsa::Database;
use tribute::compile_with_diagnostics;
use tribute::parse_and_lower_ast;
use tribute::pipeline::{
    check_shared_middle_end, compile_frontend_for_shared_route, prepare_frontend_for_lowering,
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

fn bench_edit(c: &mut Criterion, workload: &str, edit: &Edit) {
    let mut group = c.benchmark_group(format!("editing/{workload}/{}", edit.name));
    let session = RefCell::new(EditSession::new(edit));
    let toggle = || session.borrow_mut().toggle();

    group.bench_function("diagnostics", |b| {
        b.iter_batched(
            toggle,
            |()| {
                let session = session.borrow();
                session
                    .db()
                    .attach(|db| compile_with_diagnostics(db, session.source()))
            },
            BatchSize::PerIteration,
        )
    });

    if edit.name == STAGED_EDIT {
        group.bench_function("frontend", |b| {
            b.iter_batched(
                toggle,
                |()| frontend(&session.borrow()),
                BatchSize::PerIteration,
            )
        });
        group.bench_function("lowering", |b| {
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
        group.bench_function("middle_end", |b| {
            b.iter_batched(
                || {
                    toggle();
                    let session = session.borrow();
                    session
                        .db()
                        .attach(|db| compile_frontend_for_shared_route(db, session.source()))
                        .expect("the program lowers")
                },
                |frontend| check_shared_middle_end(frontend).expect("the middle-end succeeds"),
                BatchSize::PerIteration,
            )
        });
    }
    group.finish();
}

fn editing(c: &mut Criterion) {
    for workload in workloads() {
        for edit in &workload.edits {
            bench_edit(c, workload.name, edit);
        }
    }
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
