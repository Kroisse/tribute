//! Diagnostic formatting utilities for the Tribute compiler CLI.

use std::fmt;
use std::ops::Range;

use ariadne::{Cache, Color, Config, IndexType, Label, Report, ReportKind, Source};
use rustc_hash::FxHashMap as HashMap;
use tribute_core::diagnostic::{CompilationPhase, Diagnostic};
use trunk_ir::diagnostic::DiagnosticSeverity;

use tribute::SourceCst;
use tribute::pipeline::compile_with_diagnostics;
use tribute_front::ast::source_hash;
use tribute_front::source_file::SourceFiles;
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

/// The files a diagnostic's spans are in. Each holds only the lines its
/// spans cover, after an empty line for each line before them, so that
/// ariadne numbers them as in the file. (Ariadne's `display_line_offset`
/// moves the location in a report's header, but not its line numbers.)
struct SpanFiles<'a> {
    db: &'a dyn salsa::Database,
    root: SourceCst,
    /// How the root file is shown.
    root_path: &'a str,
    excerpts: HashMap<SourceCst, Excerpt>,
}

/// The lines of a file that a diagnostic's spans cover.
struct Excerpt {
    /// The byte offset in the file of the excerpt's start.
    offset: usize,
    source: Source<String>,
}

impl<'a> SpanFiles<'a> {
    fn new(db: &'a dyn salsa::Database, root: SourceCst, root_path: &'a str) -> Self {
        Self {
            db,
            root,
            root_path,
            excerpts: HashMap::default(),
        }
    }

    /// The file whose spans carry the source id `source`: another file the
    /// build read, such as a file module, a manifest, or a dependency's
    /// file, or else the root.
    fn file(&self, source: u64) -> SourceCst {
        SourceFiles::try_get(self.db)
            .and_then(|files| files.provider(self.db).find(self.db, source))
            .unwrap_or(self.root)
    }

    /// Read the lines that `spans`, byte ranges in their files, cover.
    fn read(&mut self, spans: &[(SourceCst, Range<usize>)]) {
        let mut covered: HashMap<SourceCst, Range<usize>> = HashMap::default();
        for (file, span) in spans {
            covered
                .entry(*file)
                .and_modify(|range| {
                    range.start = range.start.min(span.start);
                    range.end = range.end.max(span.end);
                })
                .or_insert_with(|| span.clone());
        }
        for (file, range) in covered {
            let text = file.text(self.db);
            let len = text.len_bytes();
            let first = text.byte_to_line(range.start.min(len));
            let last = text.byte_to_line(range.end.min(len));
            let start = text.line_to_byte(first);
            let end = text.line_to_byte((last + 1).min(text.len_lines()));
            let mut excerpt = "\n".repeat(first);
            excerpt.extend(text.byte_slice(start..end).chunks());
            self.excerpts.insert(
                file,
                Excerpt {
                    offset: start - first,
                    source: Source::from(excerpt),
                },
            );
        }
    }

    /// `span` of `file` as a span of its excerpt.
    fn span(&self, file: SourceCst, span: &Range<usize>) -> (SourceCst, Range<usize>) {
        let offset = self.excerpts.get(&file).map_or(0, |excerpt| excerpt.offset);
        (file, span.start - offset..span.end - offset)
    }
}

impl Cache<SourceCst> for SpanFiles<'_> {
    type Storage = String;

    fn fetch(&mut self, file: &SourceCst) -> Result<&Source<String>, impl fmt::Debug> {
        self.excerpts
            .get(file)
            .map(|excerpt| &excerpt.source)
            .ok_or("a file no span of the diagnostic is in")
    }

    fn display<'b>(&self, file: &'b SourceCst) -> Option<impl fmt::Display + 'b> {
        if *file == self.root {
            return Some(self.root_path.to_owned());
        }
        let uri = file.uri(self.db);
        Some(uri_to_path(uri).map_or_else(
            || uri.as_str().to_owned(),
            |path| path.display().to_string(),
        ))
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
    let mut files = SpanFiles::new(db, root, root_path);
    let span_of = |span: &trunk_ir::Span| {
        let (start, end) = normalize_span(span.start as usize, span.end as usize);
        (files.file(span.source), start..end)
    };
    let main = span_of(&diag.inner.span);
    let labels: Vec<_> = diag
        .inner
        .labels
        .iter()
        .map(|label| (span_of(&label.span), label.message.as_ref()))
        .collect();
    let spans: Vec<_> = std::iter::once(main.clone())
        .chain(labels.iter().map(|(span, _)| span.clone()))
        .collect();
    files.read(&spans);
    let main = files.span(main.0, &main.1);
    let color = phase_color(&diag.phase);

    let report_kind = match diag.inner.severity {
        DiagnosticSeverity::Error => ReportKind::Error,
        DiagnosticSeverity::Warning => ReportKind::Warning,
        DiagnosticSeverity::Info => ReportKind::Advice,
    };

    let mut report = Report::build(report_kind, main.clone())
        .with_config(Config::default().with_index_type(IndexType::Byte))
        .with_code(format!("{:?}", diag.phase))
        .with_message(&diag.inner.message)
        .with_label(
            Label::new(main)
                .with_message(&diag.inner.message)
                .with_color(color),
        );

    for ((file, span), message) in &labels {
        report = report.with_label(
            Label::new(files.span(*file, span))
                .with_message(message)
                .with_color(Color::Blue),
        );
    }

    // Add note
    if let Some(note) = &diag.inner.note {
        report = report.with_note(note.as_ref());
    }

    report.finish().eprint(files).ok();
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
