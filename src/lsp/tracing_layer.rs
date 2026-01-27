//! Custom tracing layer that sends log events as LSP `window/logMessage` notifications.

use crossbeam_channel::Sender;
use lsp_server::{Message, Notification};
use lsp_types::notification::Notification as _;
use lsp_types::{LogMessageParams, MessageType};
use tracing::field::{Field, Visit};
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::layer::Context;
use tracing_subscriber::Layer;

/// A [`tracing_subscriber::Layer`] that forwards tracing events
/// to an LSP client via `window/logMessage` notifications.
pub struct LspLayer {
    sender: Sender<Message>,
}

impl LspLayer {
    pub fn new(sender: Sender<Message>) -> Self {
        Self { sender }
    }
}

impl<S: Subscriber> Layer<S> for LspLayer {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let level = *event.metadata().level();
        let typ = match level {
            Level::ERROR => MessageType::ERROR,
            Level::WARN => MessageType::WARNING,
            Level::INFO => MessageType::INFO,
            _ => MessageType::LOG,
        };

        let mut visitor = MessageVisitor::default();
        event.record(&mut visitor);

        let message = if visitor.fields.is_empty() {
            visitor.message
        } else {
            format!("{} {}", visitor.message, visitor.fields.join(", "))
        };

        let params = LogMessageParams { typ, message };
        let notif = Notification::new(
            lsp_types::notification::LogMessage::METHOD.to_string(),
            params,
        );
        // Best-effort: ignore send errors (e.g. connection closed).
        let _ = self.sender.send(Message::Notification(notif));
    }
}

#[derive(Default)]
struct MessageVisitor {
    message: String,
    fields: Vec<String>,
}

impl Visit for MessageVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.message = format!("{value:?}");
        } else {
            self.fields.push(format!("{}={:?}", field.name(), value));
        }
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.message = value.to_string();
        } else {
            self.fields.push(format!("{}={}", field.name(), value));
        }
    }
}
