//! Diagnostic formatting utilities for the Tribute compiler CLI.

use ariadne::{Color, Label, Report, ReportKind, Source};
use ropey::Rope;
use tribute_passes::diagnostic::{CompilationPhase, Diagnostic};

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

/// Print a diagnostic using ariadne for pretty output.
pub fn print_diagnostic(diag: &Diagnostic, source: &Rope, file_path: &str) {
    let (start, end) = normalize_span(diag.span.start, diag.span.end);
    let color = phase_color(&diag.phase);
    let source_text: String = source.to_string();

    Report::build(ReportKind::Error, (file_path, start..end))
        .with_code(format!("{:?}", diag.phase))
        .with_message(&diag.message)
        .with_label(
            Label::new((file_path, start..end))
                .with_message(&diag.message)
                .with_color(color),
        )
        .finish()
        .eprint((file_path, Source::from(source_text)))
        .ok();
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
