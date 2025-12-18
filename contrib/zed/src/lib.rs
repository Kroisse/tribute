use zed_extension_api::{self as zed, LanguageServerId, Result};

struct TributeExtension;

impl zed::Extension for TributeExtension {
    fn new() -> Self {
        TributeExtension
    }

    fn language_server_command(
        &mut self,
        _language_server_id: &LanguageServerId,
        worktree: &zed::Worktree,
    ) -> Result<zed::Command> {
        // For development: use target/debug/tribute from worktree root
        // For production: fall back to PATH lookup
        let root = worktree.root_path();
        let path = format!("{root}/target/debug/tribute");

        Ok(zed::Command {
            command: path,
            args: vec!["lsp".to_string()],
            env: Default::default(),
        })
    }
}

zed::register_extension!(TributeExtension);
