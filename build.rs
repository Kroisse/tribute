fn main() {
    // Read the library directories from tribute-runtime's `links` metadata.
    // - DEP_TRIBUTE_RUNTIME_LIB_DIR: directory containing libmprompt.a (cc output)
    // - DEP_TRIBUTE_RUNTIME_STATIC_LIB_DIR: directory containing libtribute_runtime.a
    //
    // These are forwarded as compile-time environment variables so that
    // `link_native_binary` can locate the runtime libraries at build time.
    if let Ok(lib_dir) = std::env::var("DEP_TRIBUTE_RUNTIME_LIB_DIR") {
        println!("cargo:rustc-env=TRIBUTE_RUNTIME_LIB_DIR={lib_dir}");
    }
    if let Ok(static_lib_dir) = std::env::var("DEP_TRIBUTE_RUNTIME_STATIC_LIB_DIR") {
        println!("cargo:rustc-env=TRIBUTE_RUNTIME_STATIC_LIB_DIR={static_lib_dir}");
    }
}
