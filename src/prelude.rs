//! The prelude source: the root module of package `std`.

use ropey::Rope;
use tree_sitter::Parser;
use tribute_front::source_file::parse_with_rope;

/// The prelude source code, embedded at compile time.
const PRELUDE_SOURCE: &str = include_str!("../lib/std/prelude.trb");

/// The URI the prelude source is registered under.
pub(crate) const PRELUDE_URI: &str = "prelude:///std/prelude";

/// Source appended to the prelude and compiled as part of package `std`.
///
/// Tests use it to exercise package-internal items, such as the
/// compiler-owned runtime under `std::runtime`, that user packages cannot
/// name. Create it before the first query on a database: an absent input
/// records no dependency, so creating it later recompiles nothing. Its
/// declarations are prelude declarations and can shadow the prelude's own.
#[salsa::input(singleton)]
pub struct PreludeExtension {
    pub text: String,
}

/// The prelude source, with any [`PreludeExtension`] appended.
pub(crate) fn prelude_source(db: &dyn salsa::Database) -> Option<crate::SourceCst> {
    let uri = fluent_uri::Uri::parse_from(PRELUDE_URI.to_owned()).expect("valid prelude URI");
    let text: Rope = match PreludeExtension::try_get(db) {
        Some(extension) => format!("{PRELUDE_SOURCE}\n{}", extension.text(db)).into(),
        None => PRELUDE_SOURCE.into(),
    };
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_tribute::LANGUAGE.into())
        .expect("Failed to set language");
    let tree = parse_with_rope(&mut parser, &text, None)?;
    Some(crate::SourceCst::new(db, uri, text, Some(tree)))
}
