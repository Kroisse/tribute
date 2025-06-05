//! MLIR operation and type definitions for Tribute programs.

/// Tracked MLIR module representation
#[salsa::tracked]
pub struct MlirModule<'db> {
    #[return_ref]
    pub functions: Vec<(tribute_ast::Identifier, MlirFunction<'db>)>,
}

/// Tracked MLIR function representation
#[salsa::tracked]
pub struct MlirFunction<'db> {
    pub name: tribute_ast::Identifier,
    #[return_ref]
    pub params: Vec<tribute_ast::Identifier>,
    #[return_ref]
    pub body: Vec<MlirOperation>,
    pub span: tribute_ast::SimpleSpan,
}

/// MLIR operation representation
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MlirOperation {
    /// Create a boxed number value
    BoxNumber { value: i64 },
    /// Create a boxed string value  
    BoxString { value: String },
    /// Unbox a value to get its content
    Unbox { boxed_value: String, expected_type: BoxedType },
    /// Function call (func.call) - all functions work with boxed values
    Call { func: String, args: Vec<String> },
    /// Variable reference - always returns a boxed value
    Variable { name: String },
    /// Return operation - always returns a boxed value
    Return { value: Option<String> },
    /// GC operations
    GcRetain { boxed_value: String },
    GcRelease { boxed_value: String },
    GcCollect,
    /// List operations
    ListOp { operation: MlirListOperation },
    /// Placeholder for unimplemented operations
    Placeholder { description: String },
}

/// Types that can be stored in boxed values
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum BoxedType {
    Number,
    String,
    Boolean,
    Function,
    List,
    Nil,
}

/// List operation representation
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MlirListOperation {
    /// Create empty list: tribute_box_list_empty(capacity)
    CreateEmpty { capacity: usize },
    /// Create from array: tribute_box_list_from_array(elements, count) 
    CreateFromArray { elements: Vec<String> },
    /// Get element: tribute_list_get(list, index) - O(1)
    Get { list: String, index: String },
    /// Set element: tribute_list_set(list, index, value) - O(1)
    Set { list: String, index: String, value: String },
    /// Push element: tribute_list_push(list, value) - Amortized O(1)
    Push { list: String, value: String },
    /// Pop element: tribute_list_pop(list) - O(1)
    Pop { list: String },
    /// Get length: tribute_list_length(list) - O(1)
    Length { list: String },
}

/// Tracked MLIR expression result
#[salsa::tracked]
pub struct MlirExpressionResult<'db> {
    #[return_ref]
    pub operations: Vec<MlirOperation>,
    pub result_value: Option<String>,
}

/// Helper function to get string representation of BoxedType
pub fn expected_type_name(boxed_type: &BoxedType) -> &'static str {
    match boxed_type {
        BoxedType::Number => "number",
        BoxedType::String => "string", 
        BoxedType::Boolean => "boolean",
        BoxedType::Function => "function",
        BoxedType::List => "list",
        BoxedType::Nil => "nil",
    }
}