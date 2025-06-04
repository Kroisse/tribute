use crate::{
    ast::{Expr, SimpleSpan},
    parser::TributeParser,
};
use salsa::Accumulator;
use std::path::{Path, PathBuf};

#[derive(Default, Clone)]
#[salsa::db]
pub struct TributeDatabaseImpl {
    storage: salsa::Storage<Self>,
}

#[salsa::db]
impl salsa::Database for TributeDatabaseImpl {}

#[salsa::input]
pub struct SourceFile {
    #[return_ref]
    pub path: PathBuf,
    #[return_ref]
    pub text: String,
}

#[salsa::tracked]
pub struct Program<'db> {
    #[return_ref]
    pub expressions: Vec<TrackedExpression<'db>>,
}

#[salsa::tracked]
pub struct TrackedExpression<'db> {
    pub expr: Expr,
    pub span: SimpleSpan,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[salsa::accumulator]
pub struct Diagnostic {
    pub message: String,
    pub span: SimpleSpan,
    pub severity: DiagnosticSeverity,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DiagnosticSeverity {
    Error,
    Warning,
    Info,
}

impl std::fmt::Display for DiagnosticSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DiagnosticSeverity::Error => write!(f, "ERROR"),
            DiagnosticSeverity::Warning => write!(f, "WARNING"),
            DiagnosticSeverity::Info => write!(f, "INFO"),
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
            }
            .accumulate(db);

            return Program::new(db, Vec::new());
        }
    };

    let expressions = match parser.parse(&source.text(db)) {
        Ok(exprs) => exprs
            .into_iter()
            .map(|(expr, span)| TrackedExpression::new(db, expr, span))
            .collect(),
        Err(e) => {
            Diagnostic {
                message: format!("Parse error: {}", e),
                span: SimpleSpan::new(0, source.text(db).len()),
                severity: DiagnosticSeverity::Error,
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

pub fn parse_with_database(path: &Path, text: &str) -> (Vec<(Expr, SimpleSpan)>, Vec<Diagnostic>) {
    let db = TributeDatabaseImpl::default();
    let source = SourceFile::new(&db, path.to_path_buf(), text.to_string());

    let program = parse_source_file(&db, source);
    let diagnostics = diagnostics(&db, source);

    let expressions = program
        .expressions(&db)
        .iter()
        .map(|tracked| (tracked.expr(&db).clone(), tracked.span(&db)))
        .collect();

    (expressions, diagnostics)
}
