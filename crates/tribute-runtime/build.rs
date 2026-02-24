use std::env;
use std::path::PathBuf;

fn main() {
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    let mut build = cc::Build::new();
    println!("cargo:rerun-if-changed=libmprompt/src");
    println!("cargo:rerun-if-changed=libmprompt/include");
    build
        .file("libmprompt/src/mprompt/main.c")
        .include("libmprompt/include")
        .define("MP_STATIC_LIB", None)
        .warnings(false);

    // Workaround for LLVM setjmp/longjmp miscompilation on ARM64:
    // LLVM incorrectly clobbers local variables across custom setjmp/longjmp
    // calls despite the returns_twice attribute on mp_setjmp. At -O1 and above,
    // the register allocator may place variables live across setjmp into registers
    // that are not preserved by the custom longjmp, causing corruption.
    // Compile libmprompt C code at -O0 to ensure correctness.
    // The performance-critical fiber-switching code is in hand-written assembly
    // (longjmp_arm64.S) and is unaffected by this setting.
    // See: https://github.com/llvm/llvm-project/issues/21557
    if target_arch == "aarch64" {
        build.opt_level(0);
    }

    // Select the platform-specific assembly file for longjmp/setjmp
    match (target_arch.as_str(), target_os.as_str()) {
        ("x86_64", "windows") => {
            // Windows x86_64 uses MASM
            build.file("libmprompt/src/mprompt/asm/longjmp_amd64_win.asm");
        }
        ("x86_64", _) => {
            build.file("libmprompt/src/mprompt/asm/longjmp_amd64.S");
        }
        ("aarch64", _) => {
            build.file("libmprompt/src/mprompt/asm/longjmp_arm64.S");
        }
        _ => {
            panic!("unsupported target architecture for libmprompt: {target_arch}-{target_os}");
        }
    }

    build.compile("mprompt");

    // Emit metadata for consuming crates via the `links` mechanism.
    // Crates that depend on tribute-runtime can read:
    //   DEP_TRIBUTE_RUNTIME_LIB_DIR  — directory containing libmprompt.a
    //   DEP_TRIBUTE_RUNTIME_STATIC_LIB_DIR — directory containing libtribute_runtime.a
    let out_dir = env::var("OUT_DIR").unwrap();
    println!("cargo:lib_dir={out_dir}");

    // The staticlib output goes to the cargo target directory.
    // Walk up from OUT_DIR to find it: OUT_DIR is typically
    // target/<profile>/build/<crate>-<hash>/out
    let static_lib_dir = PathBuf::from(&out_dir)
        .ancestors()
        .nth(3) // up to target/<profile>/
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from(&out_dir));
    println!("cargo:static_lib_dir={}", static_lib_dir.display());

    // Link pthread on non-Windows platforms
    if target_os != "windows" {
        println!("cargo:rustc-link-lib=pthread");
    }
}
