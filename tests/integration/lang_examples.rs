//! Every example under `lang-examples/` passes the frontend, except the ones
//! listed below with the reason they are expected to fail.
//! `modules_file/` is one package whose root is `main.trb`; its other files
//! are its file modules.

use std::path::{Path, PathBuf};

use salsa::Database;
use tribute::TributeDatabaseImpl;

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
];

fn examples(dir: &Path, root: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("lang-examples directory") {
        let path = entry.expect("directory entry").path();
        if path.is_dir() {
            if path.file_name().is_some_and(|name| name == "modules_file") {
                out.push(
                    path.join("main.trb")
                        .strip_prefix(root)
                        .unwrap()
                        .to_path_buf(),
                );
            } else {
                examples(&path, root, out);
            }
        } else if path.extension().is_some_and(|extension| extension == "trb") {
            out.push(path.strip_prefix(root).unwrap().to_path_buf());
        }
    }
}

fn frontend_messages(path: &Path) -> Vec<String> {
    let db = TributeDatabaseImpl::default();
    let source = db.input(path.to_path_buf()).expect("example source");
    db.attach(|db| {
        tribute::pipeline::frontend_diagnostics(db, source)
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
        let messages = frontend_messages(&root.join(file));
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
