//! Every single-file example under `lang-examples/` passes the frontend,
//! except the ones listed below with the reason they are expected to fail.
//! `modules_file/` is a multi-file package layout and is not checked here.

use std::path::{Path, PathBuf};

use ropey::Rope;
use salsa::Database;
use tribute::TributeDatabaseImpl;
use tribute::database::parse_with_thread_local;
use tribute_core::diagnostic::Diagnostic;
use tribute_front::SourceCst;

/// Examples that must fail, each with a message its diagnostics contain.
const EXPECTED_FAILURES: &[(&str, &str)] = &[
    // The canonical invalid example.
    (
        "invalid_unresolved_name.trb",
        "unresolved name `missing_value`",
    ),
    // String interpolation is not implemented yet (#92).
    (
        "string_interpolation.trb",
        "interpolation is not implemented yet",
    ),
    (
        "strings/string_interpolation.trb",
        "interpolation is not implemented yet",
    ),
    // Qualified UFCS (`x.a::b()`) is designed but not implemented (#1210).
    ("ufcs-qualified.trb", "unresolved method 'math::double'"),
];

fn examples(dir: &Path, root: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("lang-examples directory") {
        let path = entry.expect("directory entry").path();
        if path.is_dir() {
            if path.file_name().is_some_and(|name| name != "modules_file") {
                examples(&path, root, out);
            }
        } else if path.extension().is_some_and(|extension| extension == "trb") {
            out.push(path.strip_prefix(root).unwrap().to_path_buf());
        }
    }
}

fn frontend_messages(path: &Path, text: &str) -> Vec<String> {
    let source_code = Rope::from_str(text);
    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source = SourceCst::from_path(db, path, source_code.clone(), tree);
        tribute::pipeline::parse_and_lower_ast::accumulated::<Diagnostic>(db, source)
            .into_iter()
            .map(|diagnostic| diagnostic.inner.message.clone())
            .collect()
    })
}

#[test]
fn lang_examples_pass_the_frontend() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("lang-examples");
    let mut files = Vec::new();
    examples(&root, &root, &mut files);
    files.sort();
    assert!(!files.is_empty());

    let mut problems = Vec::new();
    for file in &files {
        let name = file.to_string_lossy().replace('\\', "/");
        let text = std::fs::read_to_string(root.join(file)).expect("example source");
        let messages = frontend_messages(file, &text);
        match EXPECTED_FAILURES
            .iter()
            .find(|(expected, _)| *expected == name)
        {
            Some((_, expected)) if !messages.iter().any(|message| message.contains(expected)) => {
                problems.push(format!("{name}: expected `{expected}`, got {messages:?}"));
            }
            Some(_) => {}
            None if !messages.is_empty() => problems.push(format!("{name}: {messages:?}")),
            None => {}
        }
    }
    assert!(problems.is_empty(), "{problems:#?}");
}
