//! Native executable linking against the Tribute runtime.
//!
//! The runtime staticlib is found in a sysroot chosen at link time; see
//! `new-plans/linking.md` for the lookup contract.

use std::path::{Path, PathBuf};

/// Errors that can occur during native binary linking.
#[derive(Debug, derive_more::Display)]
pub enum LinkError {
    #[display("failed to create temporary file: {_0}")]
    TempFile(std::io::Error),
    #[display("failed to invoke linker (cc): {_0}")]
    LinkerNotFound(std::io::Error),
    #[display("linker failed with exit code {_0}")]
    LinkerFailed(i32),
    #[display("failed to locate the compiler executable for the default sysroot: {_0}")]
    SysrootNotFound(std::io::Error),
    #[display(
        "tribute runtime library not found at {}; pass --sysroot, set {SYSROOT_ENV}, \
         or run `cargo xtask runtime` in a development checkout",
        _0.display()
    )]
    RuntimeNotFound(PathBuf),
}

impl std::error::Error for LinkError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LinkError::TempFile(e)
            | LinkError::LinkerNotFound(e)
            | LinkError::SysrootNotFound(e) => Some(e),
            LinkError::LinkerFailed(_) | LinkError::RuntimeNotFound(_) => None,
        }
    }
}

/// Environment variable that selects the Tribute sysroot.
pub const SYSROOT_ENV: &str = "TRIBUTE_SYSROOT";

/// File name of the native runtime staticlib for the host target, following
/// rustc's staticlib naming (`tribute_runtime.lib` on MSVC).
fn runtime_library_name() -> &'static str {
    if target_lexicon::HOST.environment == target_lexicon::Environment::Msvc {
        "tribute_runtime.lib"
    } else {
        "libtribute_runtime.a"
    }
}

/// Choose the sysroot for native linking.
///
/// An explicit path wins over [`SYSROOT_ENV`]; without either, the sysroot is
/// the parent of the directory holding the compiler executable, so an
/// installed `<prefix>/bin/tribute` uses `<prefix>`.
pub fn resolve_sysroot(explicit: Option<&Path>) -> Result<PathBuf, LinkError> {
    if let Some(sysroot) = explicit {
        return Ok(sysroot.to_path_buf());
    }
    if let Some(sysroot) = std::env::var_os(SYSROOT_ENV).filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(sysroot));
    }
    let exe = std::env::current_exe().map_err(LinkError::SysrootNotFound)?;
    exe.ancestors()
        .nth(2)
        .map(Path::to_path_buf)
        .ok_or_else(|| {
            LinkError::SysrootNotFound(std::io::Error::other(format!(
                "{} has no parent directory",
                exe.display()
            )))
        })
}

/// Path of the native runtime staticlib for the host target inside `sysroot`.
pub fn runtime_library_path(sysroot: &Path) -> PathBuf {
    sysroot
        .join("lib")
        .join("tribute")
        .join(target_lexicon::HOST.to_string())
        .join(runtime_library_name())
}

/// Link native object bytes into an executable.
///
/// Writes object bytes to a temp file and invokes the system linker (`cc`),
/// linking against the tribute runtime library from the sysroot chosen by
/// [`resolve_sysroot`].
pub fn link_native_binary(
    object_bytes: &[u8],
    output: &Path,
    sysroot: Option<&Path>,
) -> Result<(), LinkError> {
    let runtime_lib = runtime_library_path(&resolve_sysroot(sysroot)?);
    if !runtime_lib.is_file() {
        return Err(LinkError::RuntimeNotFound(runtime_lib));
    }

    let obj_file = tempfile::Builder::new()
        .suffix(".o")
        .tempfile()
        .map_err(LinkError::TempFile)?;
    std::fs::write(obj_file.path(), object_bytes).map_err(LinkError::TempFile)?;

    let mut cmd = std::process::Command::new("cc");
    cmd.arg(obj_file.path());
    cmd.arg(&runtime_lib);
    if !cfg!(windows) {
        cmd.arg("-lpthread");
    }

    cmd.arg("-o").arg(output);

    let status = cmd.status().map_err(LinkError::LinkerNotFound)?;

    if !status.success() {
        return Err(LinkError::LinkerFailed(status.code().unwrap_or(-1)));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_link_error_display_temp_file() {
        let err = LinkError::TempFile(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "permission denied",
        ));
        let msg = err.to_string();
        assert!(msg.contains("failed to create temporary file"));
        assert!(msg.contains("permission denied"));
    }

    #[test]
    fn test_link_error_display_linker_not_found() {
        let err = LinkError::LinkerNotFound(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "not found",
        ));
        let msg = err.to_string();
        assert!(msg.contains("failed to invoke linker (cc)"));
        assert!(msg.contains("not found"));
    }

    #[test]
    fn test_link_error_display_linker_failed() {
        let err = LinkError::LinkerFailed(1);
        assert_eq!(err.to_string(), "linker failed with exit code 1");
    }

    #[test]
    fn test_link_error_source() {
        use std::error::Error;

        let io_err = LinkError::TempFile(std::io::Error::other("test"));
        assert!(io_err.source().is_some());

        let linker_err = LinkError::LinkerFailed(1);
        assert!(linker_err.source().is_none());
    }

    #[test]
    fn test_link_native_binary_invalid_object() {
        // Passing garbage bytes should cause the linker to fail
        let output = std::env::temp_dir().join("tribute_test_invalid_link");
        let result = link_native_binary(b"not valid object code", &output, None);
        assert!(result.is_err());
        match result.unwrap_err() {
            LinkError::LinkerNotFound(_) => {
                eprintln!("warning: system linker (cc) not found, skipping test");
                return;
            }
            LinkError::LinkerFailed(code) => {
                assert_ne!(code, 0, "linker should fail with non-zero exit code");
            }
            other => panic!("expected LinkerFailed, got: {other}"),
        }
        // Clean up in case it somehow succeeded
        let _ = std::fs::remove_file(&output);
    }

    #[test]
    fn test_explicit_sysroot_takes_precedence() {
        let sysroot = Path::new("/opt/tribute");
        assert_eq!(resolve_sysroot(Some(sysroot)).unwrap(), sysroot);
    }

    #[test]
    fn test_runtime_library_path_uses_target_layout() {
        let path = runtime_library_path(Path::new("/opt/tribute"));
        let expected = Path::new("/opt/tribute/lib/tribute")
            .join(target_lexicon::HOST.to_string())
            .join(runtime_library_name());
        assert_eq!(path, expected);
    }

    #[test]
    fn test_link_native_binary_requires_runtime_library() {
        let sysroot = tempfile::tempdir().unwrap();
        let output = sysroot.path().join("never-linked");
        match link_native_binary(b"not valid object code", &output, Some(sysroot.path())) {
            Err(LinkError::RuntimeNotFound(path)) => {
                assert_eq!(path, runtime_library_path(sysroot.path()));
            }
            other => panic!("expected RuntimeNotFound, got: {other:?}"),
        }
        assert!(!output.exists());
    }
}
