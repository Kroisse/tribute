//! Main compiler interface
//!
//! This module provides the high-level interface for compiling Tribute
//! programs to native object files using Cranelift.

use cranelift_codegen::settings::{self, Configurable};
use cranelift_object::{ObjectBuilder, ObjectModule};
use target_lexicon::Triple;

use tribute_hir::hir::HirProgram;

use crate::codegen::CodeGenerator;
use crate::errors::CompilationResult;
use crate::runtime::RuntimeFunctions;
use tribute_core::{Db, TargetInfo};

/// Tribute compiler using Cranelift
pub struct TributeCompiler {
    module: ObjectModule,
    runtime: RuntimeFunctions,
    target: TargetInfo,
}

impl TributeCompiler {
    /// Create a new compiler for the given target
    pub fn new(db: &dyn Db, target: Option<Triple>) -> CompilationResult<Self> {
        // Use native target if not specified
        let target_triple = target.unwrap_or_else(Triple::host);

        // Create target info using the provided database
        let target_info = TargetInfo::from_triple(db, target_triple.clone());

        // Configure Cranelift settings
        let mut flag_builder = settings::builder();
        flag_builder.set("use_colocated_libcalls", "false")?;
        // Enable PIC for better linking compatibility
        flag_builder.set("is_pic", "true")?;

        let isa_builder = cranelift_codegen::isa::lookup(target_triple)?;

        let isa = isa_builder.finish(settings::Flags::new(flag_builder))?;

        // Create object module
        let object_builder = ObjectBuilder::new(
            isa,
            format!("tribute_{}", std::process::id()),
            cranelift_module::default_libcall_names(),
        )?;

        let mut module = ObjectModule::new(object_builder);

        // Declare runtime functions
        let runtime = RuntimeFunctions::declare_all(&mut module)?;

        Ok(TributeCompiler {
            module,
            runtime,
            target: target_info,
        })
    }

    /// Compile a HIR program to an object file
    pub fn compile_program<'db>(
        mut self,
        db: &'db dyn Db,
        program: HirProgram<'db>,
    ) -> CompilationResult<Vec<u8>> {
        // Create code generator
        let mut codegen = CodeGenerator::new(&mut self.module, &self.runtime, &self.target);

        // Generate code for the program
        codegen.compile_program(db, program)?;

        // Finalize the module and get the object file
        let object = self.module.finish();
        let bytes = object.emit()?;

        Ok(bytes)
    }
}
