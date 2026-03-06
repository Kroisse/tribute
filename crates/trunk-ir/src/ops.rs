//! Dialect operation utilities.

/// Convert an identifier token into a clean string, handling raw identifiers.
///
/// Rust requires the `r#` prefix for reserved keywords like `type` or `yield`
/// when used as identifiers. This macro strips that prefix at expansion time
/// so the IR stores clean names like "type" instead of "r#type".
#[doc(hidden)]
#[macro_export]
macro_rules! raw_ident_str {
    (r#type) => {
        "type"
    };
    (r#const) => {
        "const"
    };
    (r#use) => {
        "use"
    };
    (r#yield) => {
        "yield"
    };
    (r#return) => {
        "return"
    };
    (r#if) => {
        "if"
    };
    (r#else) => {
        "else"
    };
    (r#loop) => {
        "loop"
    };
    (r#case) => {
        "case"
    };
    (r#struct) => {
        "struct"
    };
    (r#enum) => {
        "enum"
    };
    (r#break) => {
        "break"
    };
    (r#continue) => {
        "continue"
    };
    (r#ref) => {
        "ref"
    };
    ($ident:ident) => {
        stringify!($ident)
    };
}

/// Error when converting an Operation to a dialect-specific type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConversionError {
    /// Operation name doesn't match expected dialect.operation.
    WrongOperation {
        expected: &'static str,
        actual: String,
    },
    /// Missing required attribute.
    MissingAttribute(&'static str),
    /// Attribute has wrong type.
    WrongAttributeType(&'static str),
    /// Missing result type.
    MissingResult,
    /// Missing region.
    MissingRegion,
    /// Missing or insufficient successors.
    MissingSuccessor,
    /// Wrong number of operands.
    WrongOperandCount { expected: usize, actual: usize },
}
