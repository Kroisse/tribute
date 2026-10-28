//! A database that records which queries Salsa executes, so tests can assert
//! that an edit reuses the results it should.

use std::sync::{Arc, Mutex};

use ropey::Rope;
use salsa::{EventKind, Setter};

use crate::source_file::{FileProvider, SourceFiles, apply_edit, new_parser, parse_rope};
use crate::{SourceCst, path_to_uri};

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
        let db = Self { storage, executed };
        SourceFiles::new(&db, Arc::new(FileProvider::in_memory()));
        db
    }
}

impl LoggingDb {
    /// A source at `path` with `text`, parsed with tree-sitter.
    pub fn source(&self, path: &str, text: &str) -> SourceCst {
        let rope = Rope::from_str(text);
        let tree = parse_rope(&rope, None);
        SourceCst::new(self, path_to_uri(path.as_ref()), rope, tree)
    }

    /// Make `text` the file at `path`, which a package's source tree finds
    /// by its URI; a file already there keeps its source input.
    pub fn file(&mut self, path: &str, text: &str) -> SourceCst {
        let provider = SourceFiles::get(self).provider(self).clone();
        provider
            .set(
                self,
                &path_to_uri(path.as_ref()),
                Some(Rope::from_str(text)),
            )
            .expect("a file with text")
    }

    /// Remove the file at `path`.
    pub fn remove_file(&mut self, path: &str) {
        let provider = SourceFiles::get(self).provider(self).clone();
        provider.set(self, &path_to_uri(path.as_ref()), None);
    }

    /// Replace the whole text of `source`, and its tree with a fresh parse.
    pub fn edit(&mut self, source: SourceCst, text: &str) {
        let rope = Rope::from_str(text);
        let tree = parse_rope(&rope, None);
        source.set_text(self).to(rope);
        source.set_tree(self).to(tree);
    }

    /// Replace the one occurrence of `old` in `source` with `new`, reparsing
    /// incrementally as the language server does.
    pub fn replace(&mut self, source: SourceCst, old: &str, new: &str) {
        let text = source.text(self).to_string();
        let mut matches = text.match_indices(old).map(|(start, _)| start);
        let start = matches
            .next()
            .unwrap_or_else(|| panic!("{old:?} does not occur in the source"));
        assert!(
            matches.next().is_none(),
            "{old:?} occurs more than once in the source"
        );
        apply_edit(
            self,
            &mut new_parser(),
            source,
            start..start + old.len(),
            new,
        );
    }

    /// Run `f`, recording the queries it executes.
    pub fn record(&self, f: impl FnOnce(&Self)) -> Executed {
        self.take_executed();
        f(self);
        Executed(self.take_executed())
    }

    fn take_executed(&self) -> Vec<String> {
        std::mem::take(&mut *self.executed.lock().unwrap())
    }
}

/// The queries one [`LoggingDb::record`] call executed, in execution order.
#[derive(Debug)]
pub struct Executed(Vec<String>);

impl Executed {
    /// Whether `query` executed.
    pub fn contains(&self, query: &str) -> bool {
        self.0.iter().any(|name| name == query)
    }

    /// How many times `query` executed, once per input it ran for.
    pub fn count(&self, query: &str) -> usize {
        self.0.iter().filter(|name| *name == query).count()
    }

    /// Assert that each of `reran` executed and none of `reused` did.
    ///
    /// Each of `reused` must have executed in `baseline`, the record of the
    /// initial computation, so that a misspelled or renamed query fails
    /// instead of passing.
    #[track_caller]
    pub fn assert_reuse(&self, baseline: &Executed, reran: &[&str], reused: &[&str]) {
        for query in reran {
            assert!(
                self.contains(query),
                "`{query}` should have rerun; executed: {self:?}"
            );
        }
        for query in reused {
            assert!(
                baseline.contains(query),
                "`{query}` never executed in the baseline; baseline: {baseline:?}"
            );
            assert!(
                !self.contains(query),
                "`{query}` should have been reused; executed: {self:?}"
            );
        }
    }
}
