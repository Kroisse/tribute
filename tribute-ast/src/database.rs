use crate::{ast::SimpleSpan, parser::TributeParser, Item, Program};
use derive_builder::Builder;
use salsa::Accumulator;
use std::path::{Path, PathBuf};

#[derive(Default, Clone)]
#[salsa::db]
pub struct TributeDatabaseImpl {
    storage: salsa::Storage<Self>,
}

#[salsa::db]
impl salsa::Database for TributeDatabaseImpl {}

#[salsa::input(debug)]
pub struct SourceFile {
    #[return_ref]
    pub path: PathBuf,
    #[return_ref]
    pub text: String,
}

#[derive(Builder, Clone, Debug, PartialEq, Eq, Hash)]
#[salsa::accumulator]
pub struct Diagnostic {
    #[builder(setter(into))]
    pub message: String,
    #[builder(default = "SimpleSpan::new(0, 0)")]
    pub span: SimpleSpan,
    pub severity: DiagnosticSeverity,
    pub phase: CompilationPhase,
}

impl Diagnostic {
    pub fn error() -> DiagnosticBuilder {
        let mut builder = DiagnosticBuilder::default();
        builder.severity(DiagnosticSeverity::Error);
        builder
    }

    pub fn warning() -> DiagnosticBuilder {
        let mut builder = DiagnosticBuilder::default();
        builder.severity(DiagnosticSeverity::Warning);
        builder
    }

    pub fn info() -> DiagnosticBuilder {
        let mut builder = DiagnosticBuilder::default();
        builder.severity(DiagnosticSeverity::Info);
        builder
    }

    pub fn debug() -> DiagnosticBuilder {
        let mut builder = DiagnosticBuilder::default();
        builder.severity(DiagnosticSeverity::Debug);
        builder
    }
}

impl DiagnosticBuilder {
    pub fn accumulate(&mut self, db: &dyn crate::Db) {
        self.build()
            .expect("Insufficient fields to build Diagnostic")
            .accumulate(db);
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DiagnosticSeverity {
    Error,
    Warning,
    Info,
    Debug,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum CompilationPhase {
    Parsing,
    HirLowering,
    TypeChecking,
    Optimization,
}

impl std::fmt::Display for DiagnosticSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DiagnosticSeverity::Error => write!(f, "ERROR"),
            DiagnosticSeverity::Warning => write!(f, "WARNING"),
            DiagnosticSeverity::Info => write!(f, "INFO"),
            DiagnosticSeverity::Debug => write!(f, "DEBUG"),
        }
    }
}

#[salsa::tracked]
pub fn parse_source_file<'db>(db: &'db dyn salsa::Database, source: SourceFile) -> Program<'db> {
    let mut parser = match TributeParser::new() {
        Ok(parser) => parser,
        Err(e) => {
            Diagnostic {
                message: format!("Failed to create parser: {}", e),
                span: SimpleSpan::new(0, 0),
                severity: DiagnosticSeverity::Error,
                phase: CompilationPhase::Parsing,
            }
            .accumulate(db);

            return Program::new(db, Vec::new());
        }
    };

    let expressions = match parser.parse(&source.text(db)) {
        Ok(exprs) => exprs
            .into_iter()
            .map(|(expr, span)| Item::new(db, (expr, span)))
            .collect(),
        Err(e) => {
            Diagnostic {
                message: format!("Parse error: {}", e),
                span: SimpleSpan::new(0, source.text(db).len()),
                severity: DiagnosticSeverity::Error,
                phase: CompilationPhase::Parsing,
            }
            .accumulate(db);

            Vec::new()
        }
    };

    Program::new(db, expressions)
}

#[salsa::tracked]
pub fn diagnostics<'db>(db: &'db dyn salsa::Database, source: SourceFile) -> Vec<Diagnostic> {
    let _ = parse_source_file(db, source);
    parse_source_file::accumulated::<Diagnostic>(db, source)
        .into_iter()
        .cloned()
        .collect()
}

pub fn parse_with_database(
    path: &Path,
    text: &str,
) -> (Vec<(crate::ast::Expr, SimpleSpan)>, Vec<Diagnostic>) {
    let db = TributeDatabaseImpl::default();
    let source = SourceFile::new(&db, path.to_path_buf(), text.to_string());

    let program = parse_source_file(&db, source);
    let diagnostics = diagnostics(&db, source);

    let expressions = program
        .items(&db)
        .iter()
        .map(|item| item.expr(&db).clone())
        .collect();

    (expressions, diagnostics)
}

// Note: This function cannot return Salsa tracked structs directly
// Use TributeDatabaseImpl::default().attach(|db| { ... }) pattern instead
pub fn parse_with_database_attachment<T>(
    path: &Path,
    text: &str,
    f: impl for<'db> FnOnce(&'db TributeDatabaseImpl, Program<'db>, Vec<Diagnostic>) -> T,
) -> T {
    let db = TributeDatabaseImpl::default();
    let source = SourceFile::new(&db, path.to_path_buf(), text.to_string());
    let program = parse_source_file(&db, source);
    let diagnostics = diagnostics(&db, source);
    f(&db, program, diagnostics)
}
