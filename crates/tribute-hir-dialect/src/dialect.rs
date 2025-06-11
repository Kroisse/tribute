//! Tribute MLIR Dialect Definition

use melior::{
    dialect::DialectRegistry,
    ir::{Location, Module},
    utility::register_all_dialects,
    Context,
};

/// The Tribute MLIR dialect
pub struct TributeDialect<'c> {
    context: &'c Context,
}

/// Operation names in the Tribute dialect
pub mod ops {
    /// Function definition operation
    pub const FUNC: &str = "tribute.func";
    
    /// Function call operation
    pub const CALL: &str = "tribute.call";
    
    /// Return operation
    pub const RETURN: &str = "tribute.return";
    
    /// Constant value operation
    pub const CONSTANT: &str = "tribute.constant";
    
    /// Runtime type conversion
    pub const TO_RUNTIME: &str = "tribute.to_runtime";
    
    /// Arithmetic operations
    pub const ADD: &str = "tribute.add";
    pub const SUB: &str = "tribute.sub";
    pub const MUL: &str = "tribute.mul";
    pub const DIV: &str = "tribute.div";
    
    /// String operations
    pub const STRING_CONCAT: &str = "tribute.string_concat";
    pub const STRING_INTERPOLATION: &str = "tribute.string_interpolation";
}

/// Tribute dialect namespace and operation definitions
impl TributeDialect<'_> {
    /// The namespace for Tribute dialect operations
    pub const NAMESPACE: &'static str = "tribute";
}

impl<'c> TributeDialect<'c> {
    /// Create a new Tribute dialect with the given MLIR context
    pub fn new(context: &'c Context) -> Self {
        // Initialize the context with necessary dialects
        Self::initialize_context(context);
        
        Self { context }
    }
    
    /// Initialize the MLIR context with required dialects
    fn initialize_context(context: &'c Context) {
        // Register standard dialects that we depend on
        let registry = DialectRegistry::new();
        
        // Register the dialects we use
        // TODO: When melior supports custom dialect registration from Rust,
        // we should properly register the Tribute dialect here.
        // For now, we use the standard dialects and allow unregistered operations
        // for our custom "tribute.*" operations.
        
        // Register standard dialects we use (arith, func, scf, etc.)
        register_all_dialects(&registry);
        
        // Apply the registry to the context
        context.append_dialect_registry(&registry);
        
        // As a temporary measure, allow unregistered dialects for tribute operations
        // This should be removed once we have proper C++ dialect implementation
        context.set_allow_unregistered_dialects(true);
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
    pub fn file_location(&self, filename: &str, line: usize, column: usize) -> Location<'c> {
        // For now, return unknown location - we'll implement proper source mapping later
        Location::new(self.context, filename, line, column)
    }
}
