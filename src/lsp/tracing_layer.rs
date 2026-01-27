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

        let message = match (visitor.message.is_empty(), visitor.fields.is_empty()) {
            (true, true) => String::new(),
            (true, false) => visitor.fields.join(", "),
            (false, true) => visitor.message,
            (false, false) => format!("{} {}", visitor.message, visitor.fields.join(", ")),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crossbeam_channel::unbounded;
    use tracing_subscriber::layer::SubscriberExt;

    /// Set up a scoped subscriber with LspLayer and return the receiver.
    /// Uses `tracing::subscriber::with_default` to avoid global state conflicts.
    fn with_lsp_layer(f: impl FnOnce()) -> crossbeam_channel::Receiver<Message> {
        let (tx, rx) = unbounded();
        let subscriber = tracing_subscriber::registry().with(LspLayer::new(tx));
        tracing::subscriber::with_default(subscriber, f);
        rx
    }

    fn recv_log_params(rx: &crossbeam_channel::Receiver<Message>) -> LogMessageParams {
        match rx.try_recv().unwrap() {
            Message::Notification(notif) => {
                assert_eq!(notif.method, "window/logMessage");
                serde_json::from_value(notif.params).unwrap()
            }
            other => panic!("expected Notification, got {other:?}"),
        }
    }

    #[test]
    fn test_simple_message() {
        let rx = with_lsp_layer(|| {
            tracing::info!("hello world");
        });
        let params = recv_log_params(&rx);
        assert_eq!(params.typ, MessageType::INFO);
        assert_eq!(params.message, "hello world");
    }

    #[test]
    fn test_message_with_fields() {
        let rx = with_lsp_layer(|| {
            tracing::debug!(count = 42, "items found");
        });
        let params = recv_log_params(&rx);
        assert_eq!(params.typ, MessageType::LOG);
        assert_eq!(params.message, "items found count=42");
    }

    #[test]
    fn test_fields_only() {
        let rx = with_lsp_layer(|| {
            tracing::warn!(uri = "file:///test.trb");
        });
        let params = recv_log_params(&rx);
        assert_eq!(params.typ, MessageType::WARNING);
        assert_eq!(params.message, "uri=file:///test.trb");
    }

    #[test]
    fn test_level_mapping() {
        let rx = with_lsp_layer(|| {
            tracing::error!("e");
            tracing::warn!("w");
            tracing::info!("i");
            tracing::debug!("d");
            tracing::trace!("t");
        });
        assert_eq!(recv_log_params(&rx).typ, MessageType::ERROR);
        assert_eq!(recv_log_params(&rx).typ, MessageType::WARNING);
        assert_eq!(recv_log_params(&rx).typ, MessageType::INFO);
        assert_eq!(recv_log_params(&rx).typ, MessageType::LOG);
        assert_eq!(recv_log_params(&rx).typ, MessageType::LOG);
    }

    #[test]
    fn test_debug_formatted_field() {
        let rx = with_lsp_layer(|| {
            let v = vec![1, 2, 3];
            tracing::info!(?v, "data");
        });
        let params = recv_log_params(&rx);
        assert_eq!(params.message, "data v=[1, 2, 3]");
    }

    #[test]
    fn test_closed_channel_does_not_panic() {
        let (tx, rx) = unbounded();
        drop(rx);
        let subscriber = tracing_subscriber::registry().with(LspLayer::new(tx));
        tracing::subscriber::with_default(subscriber, || {
            tracing::info!("should not panic");
        });
    }
}
