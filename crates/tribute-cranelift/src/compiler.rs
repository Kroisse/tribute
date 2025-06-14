//! Main compiler interface
//!
//! This module provides the high-level interface for compiling Tribute
//! programs to native object files using Cranelift.

use cranelift_codegen::settings::{self, Configurable};
use cranelift_object::{ObjectBuilder, ObjectModule};
use target_lexicon::Triple;

use salsa::Database;
use tribute_hir::hir::HirProgram;

use crate::codegen::CodeGenerator;
use crate::errors::{BoxError, CompilationError, CompilationResult};
use crate::runtime::RuntimeFunctions;

/// Tribute compiler using Cranelift
pub struct TributeCompiler {
    module: ObjectModule,
    runtime: RuntimeFunctions,
}

impl TributeCompiler {
    /// Create a new compiler for the given target
    pub fn new(target: Option<Triple>) -> CompilationResult<Self> {
        // Use native target if not specified
        let target = target.unwrap_or_else(Triple::host);

        // Configure Cranelift settings
        let mut flag_builder = settings::builder();
        flag_builder
            .set("use_colocated_libcalls", "false")
            .map_err(|e| CompilationError::CraneliftError(e.to_string()))?;
        flag_builder
            .set("is_pic", "false")
            .map_err(|e| CompilationError::CraneliftError(e.to_string()))?;

        let isa_builder = cranelift_codegen::isa::lookup(target.clone())
            .map_err(|e| CompilationError::InvalidTarget(e.to_string()))?;

        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| CompilationError::CraneliftError(e.to_string()))?;

        // Create object module
        let object_builder = ObjectBuilder::new(
            isa,
            format!("tribute_{}", std::process::id()),
            cranelift_module::default_libcall_names(),
        )
        .box_err()?;

        let mut module = ObjectModule::new(object_builder);

        // Declare runtime functions
        let runtime = RuntimeFunctions::declare_all(&mut module)?;

        Ok(TributeCompiler { module, runtime })
    }

    /// Compile a HIR program to an object file
    pub fn compile_program<'db>(
        mut self,
        db: &'db dyn Database,
        program: HirProgram<'db>,
    ) -> CompilationResult<Vec<u8>> {
        // Create code generator
        let mut codegen = CodeGenerator::new(&mut self.module, &self.runtime);

        // Generate code for the program
        codegen.compile_program(db, program)?;

        // Finalize the module and get the object file
        let object = self.module.finish();
        let bytes = object.emit().box_err()?;

        Ok(bytes)
    }
}
