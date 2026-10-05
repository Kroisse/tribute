//! Tribute language-specific IR dialects.
//!
//! This crate provides dialects specific to the Tribute programming language,
//! built on top of the trunk-ir infrastructure.

pub mod continuation_frame;
pub mod dialect;
pub mod metadata;
pub mod runtime_layout;

// Re-export common trunk-ir types for convenience
pub use trunk_ir::{BlockId, ConversionError, IdVec, Span, Symbol, idvec};

// Re-export trunk_ir::register_pure_op for convenience
pub use trunk_ir::register_pure_op;

/// Tribute 모듈 경로 조작을 위한 Symbol extension trait
/// "::"는 Tribute의 네임스페이스 구분자
pub trait ModulePathExt {
    /// "std::io::Reader" → "Reader"
    fn last_segment(&self) -> Symbol;

    /// "std::io::Reader" → Some("std::io")
    fn parent_path(&self) -> Option<Symbol>;

    /// "std::io" + "Reader" → "std::io::Reader"
    fn join_path(&self, name: &Symbol) -> Symbol;

    /// "::"를 포함하지 않으면 true
    fn is_simple(&self) -> bool;
}

impl ModulePathExt for Symbol {
    fn last_segment(&self) -> Symbol {
        let text = self.as_str();
        Symbol::new(text.rsplit("::").next().unwrap_or(text))
    }

    fn parent_path(&self) -> Option<Symbol> {
        let (parent, _) = self.as_str().rsplit_once("::")?;
        Some(Symbol::new(parent))
    }

    fn join_path(&self, name: &Symbol) -> Symbol {
        Symbol::new(&format!("{self}::{name}"))
    }

    fn is_simple(&self) -> bool {
        self.with_str(|s| !s.contains("::"))
    }
}
