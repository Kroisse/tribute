//! Type system for Tribute MLIR dialect

use melior::{
    ir::{
        r#type::IntegerType,
        Type,
    },
    Context,
};

/// Tribute type system using MLIR types
pub struct TributeTypes<'c> {
    context: &'c Context,
}

impl<'c> TributeTypes<'c> {
    /// Create a new type system with the given context
    pub fn new(context: &'c Context) -> Self {
        Self { context }
    }

    /// Get the dynamic `!tribute.value` type
    /// For now, we'll use an opaque pointer type to represent dynamic values
    pub fn value_type(&self) -> Type<'c> {
        // Using i64 as a placeholder for dynamic value representation
        // In a full implementation, this would be a custom opaque type
        IntegerType::new(self.context, 64).into()
    }

    /// Get the string type
    pub fn string_type(&self) -> Type<'c> {
        // Using LLVM string representation (pointer to i8)
        IntegerType::new(self.context, 8).into()
    }

    /// Get the f64 type for numeric constants
    pub fn f64_type(&self) -> Type<'c> {
        // For now, use i64 as a placeholder until we find the correct FloatType API
        IntegerType::new(self.context, 64).into()
    }

    /// Get the i64 type for integer operations
    pub fn i64_type(&self) -> Type<'c> {
        IntegerType::new(self.context, 64).into()
    }

    /// Get the boolean type
    pub fn bool_type(&self) -> Type<'c> {
        IntegerType::new(self.context, 1).into()
    }

    /// Get the void type for functions that don't return values
    pub fn void_type(&self) -> Type<'c> {
        // MLIR doesn't have a true void type, so we use an empty tuple
        // In practice, functions without return values just don't have results
        IntegerType::new(self.context, 0).into()
    }

    /// Convert a static type to a dynamic tribute.value type
    /// This represents the `tribute.to_runtime` operation at the type level
    pub fn to_runtime_type(&self, _static_type: Type<'c>) -> Type<'c> {
        self.value_type()
    }

    /// Get the function type for Tribute functions
    /// All Tribute functions take and return dynamic values
    pub fn function_type(&self, num_args: usize) -> Type<'c> {
        let value_type = self.value_type();
        let _arg_types = vec![value_type; num_args];
        let _result_types = [value_type];
        
        // For now, we'll represent function types as the value type
        // In a full implementation, this would create proper function types
        value_type
    }
}