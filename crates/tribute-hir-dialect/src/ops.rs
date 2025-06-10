//! MLIR operations for Tribute dialect

use crate::{errors::LoweringError, types::TributeTypes};
use melior::{
    ir::{
        attribute::{FloatAttribute, StringAttribute},
        operation::OperationBuilder,
        Identifier, Location, Operation, Region, Type, Value,
    },
    Context,
};

/// Builder for Tribute MLIR operations
pub struct TributeOps<'c> {
    context: &'c Context,
    types: TributeTypes<'c>,
}

impl<'c> TributeOps<'c> {
    /// Create a new operation builder
    pub fn new(context: &'c Context) -> Self {
        let types = TributeTypes::new(context);
        Self { context, types }
    }

    /// Create a constant operation for f64 values
    pub fn constant_f64(&self, value: f64, location: Location<'c>) -> Result<Operation<'c>, LoweringError> {
        let f64_type = self.types.f64_type();
        let attr = FloatAttribute::new(self.context, f64_type, value).into();
        
        OperationBuilder::new("arith.constant", location)
            .add_attributes(&[(Identifier::new(self.context, "value"), attr)])
            .add_results(&[f64_type])
            .build()
            .map_err(|e| LoweringError::OperationCreationFailed(format!("constant_f64: {}", e)))
    }

    /// Create a constant operation for string values
    pub fn constant_string(&self, value: &str, location: Location<'c>) -> Result<Operation<'c>, LoweringError> {
        let string_type = self.types.string_type();
        let attr = StringAttribute::new(self.context, value).into();
        
        OperationBuilder::new("tribute.constant", location)
            .add_attributes(&[(Identifier::new(self.context, "value"), attr)])
            .add_results(&[string_type])
            .build()
            .map_err(|e| LoweringError::OperationCreationFailed(format!("constant_string: {}", e)))
    }

    /// Create a runtime conversion operation (static type -> dynamic type)
    pub fn to_runtime(&self, value: Value<'c, '_>, location: Location<'c>) -> Result<Operation<'c>, LoweringError> {
        let runtime_type = self.types.value_type();
        
        OperationBuilder::new("tribute.to_runtime", location)
            .add_operands(&[value])
            .add_results(&[runtime_type])
            .build()
            .map_err(|e| LoweringError::OperationCreationFailed(format!("to_runtime: {}", e)))
    }

    /// Create an addition operation
    pub fn add(&self, lhs: Value<'c, '_>, rhs: Value<'c, '_>, location: Location<'c>) -> Result<Operation<'c>, LoweringError> {
        let result_type = self.types.value_type();
        
        OperationBuilder::new("tribute.add", location)
            .add_operands(&[lhs, rhs])
            .add_results(&[result_type])
            .build()
            .map_err(|e| LoweringError::OperationCreationFailed(format!("add: {}", e)))
    }

    /// Create a subtraction operation
    pub fn sub(&self, lhs: Value<'c, '_>, rhs: Value<'c, '_>, location: Location<'c>) -> Result<Operation<'c>, LoweringError> {
        let result_type = self.types.value_type();
        
        OperationBuilder::new("tribute.sub", location)
            .add_operands(&[lhs, rhs])
            .add_results(&[result_type])
            .build()
            .map_err(|e| LoweringError::OperationCreationFailed(format!("sub: {}", e)))
    }

    /// Create a multiplication operation
    pub fn mul(&self, lhs: Value<'c, '_>, rhs: Value<'c, '_>, location: Location<'c>) -> Result<Operation<'c>, LoweringError> {
        let result_type = self.types.value_type();
        
        OperationBuilder::new("tribute.mul", location)
            .add_operands(&[lhs, rhs])
            .add_results(&[result_type])
            .build()
            .map_err(|e| LoweringError::OperationCreationFailed(format!("mul: {}", e)))
    }

    /// Create a division operation
    pub fn div(&self, lhs: Value<'c, '_>, rhs: Value<'c, '_>, location: Location<'c>) -> Result<Operation<'c>, LoweringError> {
        let result_type = self.types.value_type();
        
        OperationBuilder::new("tribute.div", location)
            .add_operands(&[lhs, rhs])
            .add_results(&[result_type])
            .build()
            .map_err(|e| LoweringError::OperationCreationFailed(format!("div: {}", e)))
    }

    /// Create a function operation
    pub fn func(
        &self,
        name: &str,
        _args: &[Type<'c>],
        _results: &[Type<'c>],
        location: Location<'c>,
    ) -> Result<Operation<'c>, LoweringError> {
        let name_attr = StringAttribute::new(self.context, name).into();
        
        OperationBuilder::new("tribute.func", location)
            .add_attributes(&[(Identifier::new(self.context, "sym_name"), name_attr)])
            .add_regions([Region::new()])
            .build()
            .map_err(|e| LoweringError::OperationCreationFailed(format!("func: {}", e)))
    }

    /// Create a function call operation
    pub fn call(
        &self,
        function_name: &str,
        args: &[Value<'c, '_>],
        location: Location<'c>,
    ) -> Result<Operation<'c>, LoweringError> {
        let name_attr = StringAttribute::new(self.context, function_name).into();
        let result_type = self.types.value_type();
        
        OperationBuilder::new("tribute.call", location)
            .add_attributes(&[(Identifier::new(self.context, "callee"), name_attr)])
            .add_operands(args)
            .add_results(&[result_type])
            .build()
            .map_err(|e| LoweringError::OperationCreationFailed(format!("call: {}", e)))
    }

    /// Create a return operation
    pub fn return_op(&self, value: Option<Value<'c, '_>>, location: Location<'c>) -> Result<Operation<'c>, LoweringError> {
        let mut builder = OperationBuilder::new("tribute.return", location);
        
        if let Some(val) = value {
            builder = builder.add_operands(&[val]);
        }
        
        builder
            .build()
            .map_err(|e| LoweringError::OperationCreationFailed(format!("return: {}", e)))
    }

    /// Get the types helper
    pub fn types(&self) -> &TributeTypes<'c> {
        &self.types
    }
}