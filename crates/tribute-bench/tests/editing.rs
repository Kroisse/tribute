//! Each edit of the editing benchmark changes a valid program into the
//! program it means to, so the benchmark measures the path it names.

use salsa::Database;
use tribute::compile_with_diagnostics;
use tribute::{DiagnosticSeverity, TributeDatabaseImpl};
use tribute_bench::editing::{EditSession, workloads};

fn has_errors(db: &TributeDatabaseImpl, session: &EditSession) -> bool {
    db.attach(|db| {
        compile_with_diagnostics(db, session.source())
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.inner.severity == DiagnosticSeverity::Error)
    })
}

#[test]
fn every_edit_switches_between_the_programs_it_names() {
    for workload in workloads() {
        for edit in &workload.edits {
            let label = format!("{}/{}", workload.name, edit.name);
            let mut session = EditSession::new(edit);
            assert!(
                !has_errors(session.db(), &session),
                "{label}: before has errors"
            );
            session.toggle();
            assert!(session.is_edited());
            assert_eq!(
                has_errors(session.db(), &session),
                edit.breaks,
                "{label}: after"
            );
            session.toggle();
            assert!(
                !has_errors(session.db(), &session),
                "{label}: reverting has errors"
            );
        }
    }
}
