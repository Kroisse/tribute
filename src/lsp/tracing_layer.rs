//! Tracing layer that forwards events to LSP window/logMessage.

use crossbeam_channel::Sender;
use lsp_server::{Connection, Message, Notification};
use lsp_types::notification::{LogMessage, Notification as _};
use lsp_types::{LogMessageParams, MessageType};
use tracing::{Level, Metadata, Subscriber};
use tracing_subscriber::Layer;

/// A tracing layer that sends log messages to the LSP client.
pub struct LspLayer {
    sender: Sender<Message>,
}

impl LspLayer {
    /// Create a new LSP tracing layer.
    pub fn new(connection: &Connection) -> Self {
        Self {
            sender: connection.sender.clone(),
        }
    }

    fn level_to_message_type(level: &Level) -> MessageType {
        match *level {
            Level::ERROR => MessageType::ERROR,
            Level::WARN => MessageType::WARNING,
            Level::INFO => MessageType::INFO,
            Level::DEBUG | Level::TRACE => MessageType::LOG,
        }
    }
}

impl<S: Subscriber> Layer<S> for LspLayer {
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let metadata = event.metadata();
        let level = Self::level_to_message_type(metadata.level());

        // Extract the message from the event
        struct MessageVisitor(String);

        impl tracing::field::Visit for MessageVisitor {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                if field.name() == "message" {
                    self.0 = format!("{:?}", value);
                }
            }

            fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                if field.name() == "message" {
                    self.0 = value.to_string();
                }
            }
        }

        let mut visitor = MessageVisitor(String::new());
        event.record(&mut visitor);

        let message = if visitor.0.is_empty() {
            metadata.target().to_string()
        } else {
            visitor.0
        };

        let params = LogMessageParams {
            typ: level,
            message,
        };

        let notif = Notification::new(LogMessage::METHOD.to_string(), params);
        let _ = self.sender.send(Message::Notification(notif));
    }

    fn enabled(
        &self,
        _metadata: &Metadata<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) -> bool {
        true
    }
}
