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
        // Try PATH first (for installed binary), then fall back to dev build
        let command = worktree
            .which("tribute")
            .unwrap_or_else(|| format!("{}/target/debug/tribute", worktree.root_path()));

        Ok(zed::Command {
            command,
            args: vec!["lsp".to_string()],
            env: Default::default(),
        })
    }
}

zed::register_extension!(TributeExtension);
