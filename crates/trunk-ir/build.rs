//! Generates the static atom set backing `Symbol`.
//!
//! EXPERIMENT: a static atom is an index into one concrete set, so the set
//! must list every name up front. This script scans the workspace sources for
//! names passed to `Symbol::new`, declared through `symbols!`, or declared by
//! a `#[dialect]` module, including the crates downstream of trunk-ir.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Shorter names are stored inline in the atom and never reach the set.
const MAX_INLINE_LEN: usize = 7;

fn main() {
    let manifest_dir = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let mut roots = vec![manifest_dir.join("src")];
    if std::env::var_os("TRUNK_IR_SYMBOL_ATOMS_LOCAL_ONLY").is_none() {
        let workspace = manifest_dir.join("../..");
        roots.extend(
            [
                "crates/tribute-ir/src",
                "crates/tribute-core/src",
                "crates/tribute-front/src",
                "crates/tribute-passes/src",
                "crates/trunk-ir-cranelift-backend/src",
                "crates/trunk-ir-wasm-backend/src",
                "src",
            ]
            .map(|path| workspace.join(path)),
        );
    }
    println!("cargo::rerun-if-env-changed=TRUNK_IR_SYMBOL_ATOMS_LOCAL_ONLY");

    let mut atoms = BTreeSet::new();
    for root in &roots {
        println!("cargo::rerun-if-changed={}", root.display());
        scan_dir(root, &mut atoms);
    }

    let out = PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("symbol_atom.rs");
    string_cache_codegen::AtomType::new("symbol::SymbolAtom", "symbol_atom!")
        .atoms(atoms)
        .write_to_file(&out)
        .unwrap();
}

fn scan_dir(dir: &Path, atoms: &mut BTreeSet<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            scan_dir(&path, atoms);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            scan_literals(&text, atoms);
            if path.components().any(|part| part.as_os_str() == "dialect") {
                scan_identifiers(&text, atoms);
            }
        }
    }
}

/// Collect identifier-like string literals: every `Symbol::new` and
/// `symbols!` argument is one.
fn scan_literals(text: &str, atoms: &mut BTreeSet<String>) {
    for literal in text.split('"').skip(1).step_by(2) {
        if literal.len() > MAX_INLINE_LEN
            && literal.len() <= 64
            && literal
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b':' | b'$'))
        {
            atoms.insert(literal.to_owned());
        }
    }
}

/// Collect identifiers of dialect sources: dialect, operation, type and
/// attribute names are declared as Rust identifiers.
fn scan_identifiers(text: &str, atoms: &mut BTreeSet<String>) {
    for word in text.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')) {
        if word.len() > MAX_INLINE_LEN {
            atoms.insert(word.to_owned());
            atoms.insert(snake_case(word));
        }
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
