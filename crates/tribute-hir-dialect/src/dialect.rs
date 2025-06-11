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
/// 
/// These are now available through the generated_ops module when TableGen is available,
/// or through fallback constants when TableGen is not available.
pub mod ops {
    // Re-export generated or fallback operation constants
    pub use crate::generated_ops::*;
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
        
        // Register standard dialects we use (arith, func, scf, etc.)
        register_all_dialects(&registry);
        
        // Apply the registry to the context
        context.append_dialect_registry(&registry);
        
        // Load all available dialects
        context.load_all_available_dialects();
        
        // Allow unregistered dialects for tribute operations
        // This is necessary because Tribute dialect operations (tribute.*)
        // are not registered as a proper MLIR dialect yet.
        // 
        // In the future, this could be replaced by:
        // 1. Creating a proper C++ MLIR dialect implementation
        // 2. Using MLIR's dynamic dialect registration
        // 3. Building the dialect from our TableGen definitions
        context.set_allow_unregistered_dialects(true);
        
        // Initialize Tribute dialect based on TableGen definitions
        {
            // Try to initialize using generated dialect information
            if let Err(e) = crate::initialization::initialize_tribute_dialect() {
                eprintln!("Warning: Failed to initialize Tribute dialect: {}", e);
            }
        }
        
        // Verify that basic operations can be created
        let registered_count = context.registered_dialect_count();
        let loaded_count = context.loaded_dialect_count();
        
        // Log initialization status (only in debug builds)
        #[cfg(debug_assertions)]
        {
            eprintln!("Tribute dialect context initialized:");
            eprintln!("  - Registered dialects: {}", registered_count);
            eprintln!("  - Loaded dialects: {}", loaded_count);
            eprintln!("  - Unregistered dialects allowed: {}", context.allow_unregistered_dialects());
            
            {
                use crate::dialect_info;
                eprintln!("  - Tribute dialect info:");
                eprintln!("    - Name: {}", dialect_info::NAME);
                eprintln!("    - Namespace: {}", dialect_info::NAMESPACE);
                eprintln!("    - Summary: {}", dialect_info::SUMMARY);
            }
            
            // Test if some key dialects are available
            let key_dialects = [
                "arith", "builtin", "func", "scf", "cf", "llvm", 
                "memref", "tensor", "linalg", "gpu", "async"
            ];
            
            eprintln!("  - Available key dialects:");
            for dialect_name in &key_dialects {
                match std::panic::catch_unwind(|| {
                    let dialect = context.get_or_load_dialect(dialect_name);
                    dialect.namespace().map(|s| s.to_string()).unwrap_or_else(|_| "unknown".to_string())
                }) {
                    Ok(namespace) => eprintln!("    ✓ {} (namespace: {})", dialect_name, namespace),
                    Err(_) => eprintln!("    ✗ {} (failed to load)", dialect_name),
                }
            }
        }
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
