use zed_extension_api::{self as zed, settings::LspSettings, LanguageServerId, Result};

struct TributeExtension;

impl zed::Extension for TributeExtension {
    fn new() -> Self {
        TributeExtension
    }

    fn language_server_command(
        &mut self,
        language_server_id: &LanguageServerId,
        worktree: &zed::Worktree,
    ) -> Result<zed::Command> {
        // Try PATH first (for installed binary), then fall back to dev build
        let command = worktree
            .which("tribute")
            .unwrap_or_else(|| format!("{}/target/debug/tribute", worktree.root_path()));

        // Read log level from LSP settings, default to "warn"
        let log_level = LspSettings::for_worktree(language_server_id.as_ref(), worktree)?
            .settings
            .and_then(|settings| settings.get("log_level").and_then(|v| v.as_str().map(String::from)))
            .unwrap_or_else(|| "warn".to_string());

        Ok(zed::Command {
            command,
            args: vec!["--log".to_string(), log_level, "lsp".to_string()],
            env: Default::default(),
        })
    }
}

zed::register_extension!(TributeExtension);
