use dashmap::{DashMap, Entry};

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub const fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }
}

pub type Spanned<T> = (T, Span);

#[salsa::input(debug)]
pub struct SourceFile {
    #[returns(ref)]
    pub path: PathBuf,
    #[returns(ref)]
    pub text: String,
}

#[salsa::db]
pub trait Db: salsa::Database {
    fn input(&self, path: PathBuf) -> Result<SourceFile, Box<dyn std::error::Error + Send + Sync>>;
}

#[derive(Default, Clone)]
#[salsa::db]
pub struct TributeDatabaseImpl {
    storage: salsa::Storage<Self>,
    files: DashMap<PathBuf, SourceFile>,
}

#[salsa::db]
impl salsa::Database for TributeDatabaseImpl {}

#[salsa::db]
impl Db for TributeDatabaseImpl {
    fn input(&self, path: PathBuf) -> Result<SourceFile, Box<dyn std::error::Error + Send + Sync>> {
        let path = path.canonicalize()?;
        match self.files.entry(path.clone()) {
            Entry::Occupied(entry) => Ok(*entry.get()),
            Entry::Vacant(entry) => {
                let contents = std::fs::read_to_string(&path)?;
                let source_file = SourceFile::new(self, path, contents);
                Ok(*entry.insert(source_file))
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[salsa::accumulator]
pub struct Diagnostic {
    pub message: String,
    pub span: Span,
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
