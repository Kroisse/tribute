use crate::{ast::SimpleSpan, parser::TributeParser, Item, Program};
use salsa::Accumulator;
use std::path::PathBuf;

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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[salsa::accumulator]
pub struct Diagnostic {
    pub message: String,
    pub span: SimpleSpan,
    pub severity: DiagnosticSeverity,
    pub phase: CompilationPhase,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DiagnosticSeverity {
    Error,
    Warning,
    Info,
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
