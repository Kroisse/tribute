//! Layout of a Tribute sysroot.
//!
//! The compiler reads the native runtime from a sysroot, and development
//! tooling populates one; both use this crate so they agree on the layout.
//! See `new-plans/linking.md` for the lookup contract.

use std::path::{Path, PathBuf};

/// Environment variable that selects the Tribute sysroot.
pub const SYSROOT_ENV: &str = "TRIBUTE_SYSROOT";

/// File name of the native runtime staticlib for the host target, following
/// rustc's staticlib naming (`tribute_runtime.lib` on MSVC).
pub fn runtime_library_name() -> &'static str {
    if target_lexicon::HOST.environment == target_lexicon::Environment::Msvc {
        "tribute_runtime.lib"
    } else {
        "libtribute_runtime.a"
    }
}

/// Path of the native runtime staticlib for the host target inside `sysroot`.
pub fn runtime_library_path(sysroot: &Path) -> PathBuf {
    sysroot
        .join("lib")
        .join("tribute")
        .join(target_lexicon::HOST.to_string())
        .join(runtime_library_name())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_library_path_uses_target_layout() {
        let path = runtime_library_path(Path::new("/opt/tribute"));
        let expected = Path::new("/opt/tribute/lib/tribute")
            .join(target_lexicon::HOST.to_string())
            .join(runtime_library_name());
        assert_eq!(path, expected);
    }
}
