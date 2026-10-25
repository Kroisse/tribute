//! Source edits for the editing benchmark, and a session that applies them
//! to one database the way the language server does.
//!
//! Each edit is a pair of sources. A session switches between them, so every
//! revision changes the input and the database keeps what it cached from the
//! revision before.

use tribute::database::parse_with_thread_local;
use tribute::{Rope, SourceCst, TributeDatabaseImpl};

use salsa::Setter;

use crate::programs::PROGRAMS;

/// One kind of edit to a program.
pub struct Edit {
    pub name: &'static str,
    pub before: String,
    pub after: String,
    /// Whether `after` has a frontend error, so compilation stops before
    /// lowering.
    pub breaks: bool,
}

/// A program and the edits the benchmark applies to it.
pub struct Workload {
    pub name: &'static str,
    pub edits: Vec<Edit>,
}

/// The number of functions in the generated workload.
const GENERATED_FUNCTIONS: usize = 100;

/// The workloads: `native_calculator` and a generated program of 100
/// functions.
pub fn workloads() -> Vec<Workload> {
    vec![calculator(), generated()]
}

fn calculator() -> Workload {
    let source = PROGRAMS
        .iter()
        .find(|program| program.name == "native_calculator")
        .expect("native_calculator is a benchmark program")
        .source;
    Workload {
        name: "native_calculator",
        edits: vec![
            edit(
                "comment",
                source,
                &[(
                    "    case operator == \"add\" {",
                    "    // add, sub, or mul\n    case operator == \"add\" {",
                )],
                false,
            ),
            edit(
                "body",
                source,
                &[("False -> byte == 9", "False -> byte == 11")],
                false,
            ),
            edit(
                "signature",
                source,
                &[
                    (
                        "fn tokenize_from(bytes: Bytes, index: Nat)",
                        "fn tokenize_from(bytes: Bytes, index: Nat, count: Nat)",
                    ),
                    (
                        "tokenize_from(bytes, end)",
                        "tokenize_from(bytes, end, count + 1)",
                    ),
                    (
                        "tokenize_from(String::to_bytes(line), 0)",
                        "tokenize_from(String::to_bytes(line), 0, 0)",
                    ),
                ],
                false,
            ),
            edit(
                "type_error",
                source,
                &[(
                    "    case index == bytes.len() {\n        True -> index",
                    "    case index == bytes.len() {\n        True -> True",
                )],
                true,
            ),
        ],
    }
}

/// A chain of functions that each call `helper` and the function before.
fn generated_source(helper_params: &str, helper_call: &str) -> String {
    let mut source = format!("fn helper({helper_params}) -> Nat {{ x + 1 }}\n\n");
    source.push_str(&format!(
        "fn step_0(x: Nat) -> Nat {{\n    {}\n}}\n",
        helper_call.replace("ARG", "x")
    ));
    for index in 1..GENERATED_FUNCTIONS {
        source.push_str(&format!(
            "\nfn step_{index}(x: Nat) -> Nat {{\n    let y = {}\n    step_{}(y)\n}}\n",
            helper_call.replace("ARG", "x"),
            index - 1
        ));
    }
    source.push_str(&format!(
        "\nfn main() -> Nil {{\n    let _ = step_{}(1)\n}}\n",
        GENERATED_FUNCTIONS - 1
    ));
    source
}

fn generated() -> Workload {
    let source = generated_source("x: Nat", "helper(ARG)");
    let middle = format!("fn step_{}(x: Nat) -> Nat {{\n", GENERATED_FUNCTIONS / 2);
    Workload {
        name: "generated",
        edits: vec![
            edit(
                "comment",
                &source,
                &[(&middle, &format!("{middle}    // the middle step\n"))],
                false,
            ),
            edit(
                "body",
                &source,
                &[(
                    &format!("{middle}    let y = helper(x)"),
                    &format!("{middle}    let y = helper(x + 1)"),
                )],
                false,
            ),
            Edit {
                name: "signature",
                before: source.clone(),
                after: generated_source("x: Nat, y: Nat", "helper(ARG, 1)"),
                breaks: false,
            },
            edit(
                "type_error",
                &source,
                &[(
                    &format!("{middle}    let y = helper(x)"),
                    &format!("{middle}    let y = helper(True)"),
                )],
                true,
            ),
        ],
    }
}

/// An edit that replaces every occurrence of each `from` in `source` with its
/// `to`.
fn edit(name: &'static str, source: &str, replacements: &[(&str, &str)], breaks: bool) -> Edit {
    let after = replacements
        .iter()
        .fold(source.to_owned(), |text, (from, to)| {
            assert!(
                text.contains(from),
                "edit `{name}`: `{from}` is not in the source"
            );
            text.replace(from, to)
        });
    Edit {
        name,
        before: source.to_owned(),
        after,
        breaks,
    }
}

/// One source in a database that lives across edits.
pub struct EditSession {
    db: TributeDatabaseImpl,
    source: SourceCst,
    texts: [Rope; 2],
    /// The index in `texts` of the source's current text.
    current: usize,
}

impl EditSession {
    /// A session whose source holds `edit.before`.
    pub fn new(edit: &Edit) -> Self {
        let texts = [Rope::from_str(&edit.before), Rope::from_str(&edit.after)];
        let db = TributeDatabaseImpl::default();
        let tree = parse_with_thread_local(&texts[0], None);
        let source = SourceCst::from_path(&db, edit.name, texts[0].clone(), tree);
        Self {
            db,
            source,
            texts,
            current: 0,
        }
    }

    /// Switch the source to the other text of the edit, as one revision.
    pub fn toggle(&mut self) {
        self.current = 1 - self.current;
        let text = self.texts[self.current].clone();
        let tree = parse_with_thread_local(&text, None);
        self.source.set_text(&mut self.db).to(text);
        self.source.set_tree(&mut self.db).to(tree);
    }

    /// Whether the source holds the edit's `after` text.
    pub fn is_edited(&self) -> bool {
        self.current == 1
    }

    pub fn db(&self) -> &TributeDatabaseImpl {
        &self.db
    }

    pub fn source(&self) -> SourceCst {
        self.source
    }
}
