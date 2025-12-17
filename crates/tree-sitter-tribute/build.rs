use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Run tree-sitter generate to ensure parser.c is up to date
    println!("cargo:rerun-if-changed=grammar.js");
    println!("cargo:rerun-if-changed=src/parser.c");
    println!("cargo:rerun-if-changed=src/scanner.c");
    println!("cargo:rerun-if-changed=build.rs");

    // Check if tree-sitter CLI is available and run generate
    let output = Command::new("tree-sitter").arg("generate").output();

    match output {
        Ok(output) => {
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                println!("cargo:warning=tree-sitter generate failed: {}", stderr);
            } else {
                println!("cargo:warning=Successfully generated parser from grammar.js");
            }
        }
        Err(e) => {
            println!(
                "cargo:warning=tree-sitter command not found or failed to execute: {}",
                e
            );
            println!(
                "cargo:warning=Make sure tree-sitter CLI is installed: npm install -g tree-sitter-cli"
            );
        }
    }

    let dir: PathBuf = ["src"].iter().collect();

    let mut build = cc::Build::new();
    build.include(&dir);
    build.file(dir.join("parser.c"));

    // Include external scanner if it exists
    let scanner_path = dir.join("scanner.c");
    if scanner_path.exists() {
        build.file(&scanner_path);
    }

    build.compile("tree-sitter-tribute");
}
