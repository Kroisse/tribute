//! Diagnostic formatting utilities for the Tribute compiler CLI.

use ariadne::{Color, Label, Report, ReportKind, sources};
use tribute_core::diagnostic::{CompilationPhase, Diagnostic};
use trunk_ir::diagnostic::DiagnosticSeverity;

use tribute::SourceCst;
use tribute::pipeline::compile_with_diagnostics;
use tribute_front::ast::source_hash;
use tribute_front::source_tree::source_tree_of;
use tribute_front::uri_to_path;

/// Get the display color for a compilation phase.
pub fn phase_color(phase: &CompilationPhase) -> Color {
    match phase {
        CompilationPhase::Parsing => Color::Red,
        CompilationPhase::AstGeneration => Color::Red,
        CompilationPhase::TirGeneration => Color::Yellow,
        CompilationPhase::NameResolution => Color::Yellow,
        CompilationPhase::TypeChecking => Color::Magenta,
        CompilationPhase::Lowering => Color::Cyan,
        CompilationPhase::Optimization => Color::Blue,
    }
}

/// Normalize a span to ensure end > start (required by ariadne).
pub fn normalize_span(start: usize, end: usize) -> (usize, usize) {
    (start, end.max(start + 1))
}

/// The name and text of the file whose spans carry the source id `source`:
/// the root file `root`, shown as `root_path`, or one of its package's file
/// modules.
fn span_file(
    db: &dyn salsa::Database,
    root: SourceCst,
    root_path: &str,
    source: u64,
) -> (String, String) {
    let file = source_tree_of(db, root)
        .and_then(|tree| tree.file(source))
        .map(|file| file.source);
    match file {
        Some(file) if file != root => (
            uri_to_path(file.uri(db)).map_or_else(
                || file.uri(db).as_str().to_owned(),
                |path| path.display().to_string(),
            ),
            file.text(db).to_string(),
        ),
        _ => (root_path.to_owned(), root.text(db).to_string()),
    }
}

/// Print a diagnostic of the package whose root file is `root` using
/// ariadne, against the file each of its spans is in.
pub fn print_diagnostic(
    db: &dyn salsa::Database,
    root: SourceCst,
    root_path: &str,
    diag: &Diagnostic,
) {
    let span = diag.inner.span;
    let (file, text) = span_file(db, root, root_path, span.source);
    let (start, end) = normalize_span(span.start as usize, span.end as usize);
    let color = phase_color(&diag.phase);

    let report_kind = match diag.inner.severity {
        DiagnosticSeverity::Error => ReportKind::Error,
        DiagnosticSeverity::Warning => ReportKind::Warning,
        DiagnosticSeverity::Info => ReportKind::Advice,
    };

    let mut report = Report::build(report_kind, (file.clone(), start..end))
        .with_code(format!("{:?}", diag.phase))
        .with_message(&diag.inner.message)
        .with_label(
            Label::new((file.clone(), start..end))
                .with_message(&diag.inner.message)
                .with_color(color),
        );
    let mut files = vec![(file, text)];

    // Add secondary labels
    for label in diag.inner.labels.iter() {
        let (label_file, label_text) = span_file(db, root, root_path, label.span.source);
        let (ls, le) = normalize_span(label.span.start as usize, label.span.end as usize);
        report = report.with_label(
            Label::new((label_file.clone(), ls..le))
                .with_message(label.message.as_ref())
                .with_color(Color::Blue),
        );
        if !files.iter().any(|(name, _)| *name == label_file) {
            files.push((label_file, label_text));
        }
    }

    // Add note
    if let Some(note) = &diag.inner.note {
        report = report.with_note(note.as_ref());
    }

    report.finish().eprint(sources(files)).ok();
}

/// Print accumulated diagnostics in source order.
pub fn print_diagnostics(
    db: &dyn salsa::Database,
    source: SourceCst,
    file_path: &str,
    diags: &mut [&Diagnostic],
) {
    diags.sort_by(|left, right| tribute_core::diagnostic::compare_diagnostics(left, right));
    // Keep each file's diagnostics together, the root file's first.
    let root = source_hash(source.uri(db).as_str());
    diags.sort_by_key(|diag| {
        let source = diag.inner.span.source;
        // A span in no file is shown against the root, with its diagnostics.
        if source == root || source == 0 {
            None
        } else {
            Some(source)
        }
    });
    for diag in diags.iter() {
        print_diagnostic(db, source, file_path, diag);
    }
}

/// Collect and print diagnostics from a failed compilation.
///
/// First tries target-specific accumulated diagnostics; if empty,
/// falls back to `compile_with_diagnostics` for frontend diagnostics.
pub fn report_diagnostics(
    db: &dyn salsa::Database,
    source: SourceCst,
    file_path: &str,
    mut accumulated_diags: Vec<&Diagnostic>,
) {
    if !accumulated_diags.is_empty() {
        print_diagnostics(db, source, file_path, &mut accumulated_diags);
    } else {
        let result = compile_with_diagnostics(db, source);
        if !result.diagnostics.is_empty() {
            for diag in &result.diagnostics {
                print_diagnostic(db, source, file_path, diag);
            }
        } else {
            eprintln!("Compilation failed (unknown error)");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_color_parsing() {
        assert_eq!(phase_color(&CompilationPhase::Parsing), Color::Red);
    }

    #[test]
    fn test_phase_color_tir_generation() {
        assert_eq!(phase_color(&CompilationPhase::TirGeneration), Color::Yellow);
    }

    #[test]
    fn test_phase_color_name_resolution() {
        assert_eq!(
            phase_color(&CompilationPhase::NameResolution),
            Color::Yellow
        );
    }

    #[test]
    fn test_phase_color_type_checking() {
        assert_eq!(phase_color(&CompilationPhase::TypeChecking), Color::Magenta);
    }

    #[test]
    fn test_phase_color_lowering() {
        assert_eq!(phase_color(&CompilationPhase::Lowering), Color::Cyan);
    }

    #[test]
    fn test_phase_color_optimization() {
        assert_eq!(phase_color(&CompilationPhase::Optimization), Color::Blue);
    }

    #[test]
    fn test_normalize_span_valid() {
        assert_eq!(normalize_span(0, 10), (0, 10));
        assert_eq!(normalize_span(5, 15), (5, 15));
    }

    #[test]
    fn test_normalize_span_zero_length() {
        // Zero-length span should become length 1
        assert_eq!(normalize_span(5, 5), (5, 6));
    }

    #[test]
    fn test_normalize_span_start_equals_end() {
        assert_eq!(normalize_span(0, 0), (0, 1));
        assert_eq!(normalize_span(100, 100), (100, 101));
    }
}
