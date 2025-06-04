use std::path::PathBuf;

fn main() {
    let dir: PathBuf = ["src"].iter().collect();
    
    println!("cargo:rerun-if-changed=src/parser.c");
    println!("cargo:rerun-if-changed=build.rs");

    cc::Build::new()
        .include(&dir)
        .file(dir.join("parser.c"))
        .compile("tree-sitter-tribute");
}