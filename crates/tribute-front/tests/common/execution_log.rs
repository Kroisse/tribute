//! A database that records which queries Salsa executes, so tests can assert
//! that an edit reuses the results it should.

use std::sync::{Arc, Mutex};

use ropey::Rope;
use salsa::{EventKind, Setter};
use tribute_front::{SourceCst, path_to_uri};

/// A database that records the name of every query it executes.
#[salsa::db]
#[derive(Clone)]
pub struct LoggingDb {
    storage: salsa::Storage<Self>,
    executed: Arc<Mutex<Vec<String>>>,
}

#[salsa::db]
impl salsa::Database for LoggingDb {}

impl Default for LoggingDb {
    fn default() -> Self {
        let executed: Arc<Mutex<Vec<String>>> = Arc::default();
        let log = Arc::clone(&executed);
        let storage = salsa::Storage::new(Some(Box::new(move |event: salsa::Event| {
            if let EventKind::WillExecute { database_key } = event.kind {
                let key = format!("{database_key:?}");
                let name = key.split_once('(').map_or(key.as_str(), |(name, _)| name);
                log.lock().unwrap().push(name.to_owned());
            }
        })));
        Self { storage, executed }
    }
}

impl LoggingDb {
    /// A source at `path` with `text`, parsed with tree-sitter.
    pub fn source(&self, path: &str, text: &str) -> SourceCst {
        SourceCst::new(
            self,
            path_to_uri(path.as_ref()),
            Rope::from_str(text),
            parse(text),
        )
    }

    /// Replace the text of `source`, and its tree with a fresh parse of it.
    pub fn edit(&mut self, source: SourceCst, text: &str) {
        source.set_text(self).to(Rope::from_str(text));
        source.set_tree(self).to(parse(text));
    }

    /// The queries executed since the last call, in execution order.
    pub fn take_executed(&self) -> Vec<String> {
        std::mem::take(&mut *self.executed.lock().unwrap())
    }

    /// Assert that none of `queries` ran since the last call to
    /// [`LoggingDb::take_executed`], and clear the record.
    pub fn assert_not_executed(&self, queries: &[&str]) {
        let executed = self.take_executed();
        for query in queries {
            assert!(
                !executed.iter().any(|name| name == query),
                "`{query}` should not have run; executed: {executed:?}"
            );
        }
    }

    /// Assert that each of `queries` ran since the last call to
    /// [`LoggingDb::take_executed`], and clear the record.
    pub fn assert_executed(&self, queries: &[&str]) {
        let executed = self.take_executed();
        for query in queries {
            assert!(
                executed.iter().any(|name| name == query),
                "`{query}` should have run; executed: {executed:?}"
            );
        }
    }
}

fn parse(text: &str) -> Option<tree_sitter::Tree> {
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&tree_sitter_tribute::LANGUAGE.into())
        .expect("tree-sitter language");
    parser.parse(text, None)
}
