//! Tribute language-specific IR dialects.
//!
//! This crate provides dialects specific to the Tribute programming language,
//! built on top of the trunk-ir infrastructure.

pub mod dialect;

// Re-export common trunk-ir types for convenience
pub use trunk_ir::{
    Attribute, Attrs, Block, BlockBuilder, BlockId, ConversionError, DialectOp, DialectType, IdVec,
    Location, Operation, PathId, Region, Span, Symbol, Type, Value, ValueDef, idvec,
};

// Re-export trunk_ir::register_pure_op for convenience
// Users can use:
//   use tribute_ir::register_pure_op;
//   register_pure_op!(crate::dialect::src::Var<'_>);
pub use trunk_ir::register_pure_op;

/// Tribute 모듈 경로 조작을 위한 Symbol extension trait
/// "::"는 Tribute의 네임스페이스 구분자
pub trait ModulePathExt {
    /// "std::io::Reader" → "Reader"
    fn last_segment(self) -> Symbol;

    /// "std::io::Reader" → Some("std::io")
    fn parent_path(self) -> Option<Symbol>;

    /// "std::io" + "Reader" → "std::io::Reader"
    fn join_path(self, name: Symbol) -> Symbol;

    /// "::"를 포함하지 않으면 true
    fn is_simple(&self) -> bool;
}

impl ModulePathExt for Symbol {
    fn last_segment(self) -> Symbol {
        let s = self.with_str(|s| s.rsplit("::").next().unwrap_or(s).to_owned());
        Symbol::from_dynamic(&s)
    }

    fn parent_path(self) -> Option<Symbol> {
        let s = self.with_str(|s| s.rsplit_once("::").map(|(p, _)| p.to_owned()));
        s.as_deref().map(Symbol::from_dynamic)
    }

    fn join_path(self, name: Symbol) -> Symbol {
        Symbol::from_dynamic(&format!("{}::{}", self, name))
    }

    fn is_simple(&self) -> bool {
        self.with_str(|s| !s.contains("::"))
    }
}
