//! Generates the static atom set backing `Symbol`.
//!
//! A static atom is an index into one set, so the set lists trunk-ir's own
//! names: those declared by a `#[dialect]` module or `symbols!`, and the
//! literals passed to `Symbol::new`. A name missing from the set is still a
//! valid symbol; it is interned in the dynamic set instead.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Shorter names are stored inline in the atom and never reach the set.
const MAX_INLINE_LEN: usize = 7;

fn main() {
    let src = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("src");
    println!("cargo::rerun-if-changed={}", src.display());

    let mut atoms = BTreeSet::new();
    scan_dir(&src, &mut atoms);
    atoms.retain(|name| name.len() > MAX_INLINE_LEN);

    let out = PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("symbol_atom.rs");
    string_cache_codegen::AtomType::new("symbol::SymbolAtom", "symbol_atom!")
        .atoms(atoms)
        .write_to_file(&out)
        .unwrap();
}

fn scan_dir(dir: &Path, atoms: &mut BTreeSet<String>) {
    for entry in std::fs::read_dir(dir).unwrap().flatten() {
        let path = entry.path();
        if path.is_dir() {
            scan_dir(&path, atoms);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            let text = std::fs::read_to_string(&path).unwrap();
            scan_symbol_literals(&text, atoms);
            scan_dialects(&text, atoms);
        }
    }
}

/// Collect `Symbol::new("..")` arguments, `symbols!` entries, and `&str`
/// constants, which name attributes passed to `Symbol::new`.
fn scan_symbol_literals(text: &str, atoms: &mut BTreeSet<String>) {
    for marker in [
        "Symbol::new(\"",
        "=> \"",
        ": &str = \"",
        ": &'static str = \"",
    ] {
        for (start, _) in text.match_indices(marker) {
            let rest = &text[start + marker.len()..];
            if let Some(end) = rest.find('"') {
                insert_name(&rest[..end], atoms);
            }
        }
    }
}

/// Collect the names a `#[dialect]` module declares: the dialect, its
/// operations, its types, and their attributes.
fn scan_dialects(text: &str, atoms: &mut BTreeSet<String>) {
    for (start, _) in text.match_indices("dialect]") {
        let Some(rest) = text[start + "dialect]".len()..]
            .trim_start()
            .strip_prefix("mod ")
        else {
            continue;
        };
        let Some(open) = rest.find('{') else { continue };
        let body = &rest[open..open + matching_brace(&rest[open..])];
        insert_name(rest[..open].trim(), atoms);
        for (marker, snake) in [("fn ", false), ("struct ", true)] {
            for (at, _) in body.match_indices(marker) {
                let name = identifier(&body[at + marker.len()..]);
                if snake {
                    insert_name(&snake_case(name), atoms);
                } else {
                    insert_name(name, atoms);
                }
            }
        }
        for marker in [": Attr<", ": Option<Attr<"] {
            for (at, _) in body.match_indices(marker) {
                let before = &body[..at];
                let begin = before
                    .rfind(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '#'))
                    .map_or(0, |index| index + 1);
                insert_name(&before[begin..], atoms);
            }
        }
        for (at, _) in body.match_indices("#[attr(") {
            insert_name(identifier(&body[at + "#[attr(".len()..]), atoms);
        }
    }
}

/// The length of the text from an opening brace through its matching close.
fn matching_brace(text: &str) -> usize {
    let mut depth = 0usize;
    for (index, c) in text.char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return index + 1;
                }
            }
            _ => {}
        }
    }
    text.len()
}

fn identifier(text: &str) -> &str {
    let end = text
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '#'))
        .unwrap_or(text.len());
    &text[..end]
}

fn insert_name(name: &str, atoms: &mut BTreeSet<String>) {
    let name = name.strip_prefix("r#").unwrap_or(name);
    if !name.is_empty() {
        atoms.insert(name.to_owned());
    }
}

fn snake_case(word: &str) -> String {
    let mut out = String::new();
    for (index, c) in word.chars().enumerate() {
        if c.is_ascii_uppercase() {
            if index != 0 {
                out.push('_');
            }
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}
