//! Tracing layer that forwards events to LSP window/logMessage.
//!
//! Currently disabled due to interference with LSP stdio communication.
//! TODO: Investigate using a separate thread for log forwarding.

#![allow(dead_code)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crossbeam_channel::Sender;
use lsp_server::{Connection, Message, Notification};
use lsp_types::notification::{LogMessage, Notification as _};
use lsp_types::{LogMessageParams, MessageType};
use tracing::{Level, Metadata, Subscriber};
use tracing_subscriber::Layer;

/// A tracing layer that sends log messages to the LSP client.
///
/// Before `mark_initialized()` is called, messages are dropped.
/// After initialization, messages are sent immediately.
pub struct LspLayer {
    sender: Sender<Message>,
    initialized: Arc<AtomicBool>,
}

/// Handle to mark the LspLayer as initialized.
#[derive(Clone)]
pub struct LspLayerHandle {
    initialized: Arc<AtomicBool>,
}

impl LspLayerHandle {
    /// Mark the layer as initialized. Messages will now be sent.
    pub fn mark_initialized(&self) {
        self.initialized.store(true, Ordering::SeqCst);
    }
}

impl LspLayer {
    /// Create a new LSP tracing layer and a handle to control it.
    pub fn new(connection: &Connection) -> (Self, LspLayerHandle) {
        let initialized = Arc::new(AtomicBool::new(false));

        let layer = Self {
            sender: connection.sender.clone(),
            initialized: Arc::clone(&initialized),
        };

        let handle = LspLayerHandle { initialized };

        (layer, handle)
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

        if self.initialized.load(Ordering::SeqCst) {
            let _ = self.sender.send(Message::Notification(notif));
        }
        // Drop messages before initialization
    }

    fn enabled(
        &self,
        _metadata: &Metadata<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) -> bool {
        true
    }
}
