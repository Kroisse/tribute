//! Tribute MLIR Dialect Definition

use melior::{
    dialect::DialectRegistry,
    ir::{Location, Module},
    Context,
};

/// The Tribute MLIR dialect
pub struct TributeDialect<'c> {
    context: &'c Context,
}

impl<'c> TributeDialect<'c> {
    /// Create a new Tribute dialect with the given MLIR context
    pub fn new(context: &'c Context) -> Self {
        Self { context }
    }

    /// Register the Tribute dialect with the given registry
    pub fn register(_registry: &DialectRegistry) -> Result<(), String> {
        // For now, we'll use melior's built-in dialect registration
        // In the future, we might need custom C++ dialect registration
        Ok(())
    }

    /// Get the MLIR context
    pub fn context(&self) -> &'c Context {
        self.context
    }

    /// Create a new module for Tribute operations
    pub fn create_module(&self, location: Location<'c>) -> Module<'c> {
        Module::new(location)
    }
}

/// Helper functions for working with the Tribute dialect
impl<'c> TributeDialect<'c> {
    /// Create an unknown location (useful for generated code)
    pub fn unknown_location(&self) -> Location<'c> {
        Location::unknown(self.context)
    }

    /// Create a file location with line/column information
    pub fn file_location(&self, _filename: &str, _line: u32, _column: u32) -> Location<'c> {
        // For now, return unknown location - we'll implement proper source mapping later
        Location::unknown(self.context)
    }
}