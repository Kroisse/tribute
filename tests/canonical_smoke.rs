//! Product-level assertions for the canonical M0 examples.

use salsa_test_macros::salsa_test;
use tribute::pipeline::compile_with_diagnostics;
use tribute::{DiagnosticSeverity, SourceCst};
use tribute_passes::CompilationPhase;

const INVALID_SOURCE: &str = include_str!("../lang-examples/invalid_unresolved_name.trb");

#[salsa_test]
fn canonical_invalid_source_has_stable_unresolved_name_diagnostic(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "lang-examples/invalid_unresolved_name.trb",
        INVALID_SOURCE,
    );
    let result = compile_with_diagnostics(db, source);

    assert!(
        result.module.is_none(),
        "the canonical invalid source must not produce a module"
    );

    let matching: Vec<_> = result
        .diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.inner.message == "unresolved name `missing_value`")
        .collect();
    assert_eq!(
        matching.len(),
        1,
        "expected one canonical unresolved-name diagnostic, got {:?}",
        result.diagnostics
    );

    let diagnostic = matching[0];
    assert_eq!(diagnostic.phase, CompilationPhase::NameResolution);
    assert_eq!(diagnostic.inner.severity, DiagnosticSeverity::Error);

    let span = diagnostic.inner.span;
    let primary_source = INVALID_SOURCE
        .get(span.start..span.end)
        .expect("diagnostic span must be a valid UTF-8 source range");
    assert_eq!(primary_source, "missing_value");
}
