//! Build script for tribute-hir-dialect
//!
//! This script processes TableGen files using CMake and generates Rust code for MLIR operations.

use std::{env, fs, io::Write, path::Path, path::PathBuf};

type Error = Box<dyn std::error::Error>;

fn main() -> Result<(), Error> {
    // Tell Cargo to rerun if TableGen files change
    println!("cargo:rerun-if-changed=tablegen/TributeOps.td");
    println!("cargo:rerun-if-changed=CMakeLists.txt");
    println!("cargo:rerun-if-changed=build.rs");

    // Try CMake-based TableGen generation first
    if try_cmake_tablegen_generation().is_ok() {
        println!("cargo:rustc-cfg=feature=\"tablegen\"");
        println!("CMake TableGen generation successful");
    } else {
        // Fallback to tblgen crate (requires LLVM installation)
        try_tablegen_generation()?;
        println!("cargo:rustc-cfg=feature=\"tablegen\"");
        println!("Fallback TableGen generation successful");
    }

    Ok(())
}

fn try_cmake_tablegen_generation() -> Result<(), Error> {
    use std::process::Command;
    
    println!("cargo:warning=Attempting CMake-based TableGen generation...");
    
    // Get the output directory for build artifacts
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let build_dir = out_dir.join("cmake-build");
    
    // Create build directory
    fs::create_dir_all(&build_dir)?;
    
    // Get the crate root directory where CMakeLists.txt is located
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    
    // Configure step
    let status = Command::new("cmake")
        .current_dir(&build_dir)
        .arg(&manifest_dir)  // Point to the crate directory where CMakeLists.txt is
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .status()
        .map_err(|e| format!("Failed to run cmake configure: {}", e))?;
    
    if !status.success() {
        return Err("CMake configuration failed".into());
    }
    
    // Build step - just build the tribute-dialect target
    let status = Command::new("cmake")
        .current_dir(&build_dir)
        .arg("--build")
        .arg(".")
        .arg("--target")
        .arg("tribute-dialect")
        .status()
        .map_err(|e| format!("Failed to run cmake build: {}", e))?;
    
    if !status.success() {
        return Err("CMake build failed".into());
    }
    
    // Check if the files were actually generated
    let expected_files = [
        "TributeOps.h.inc",
        "TributeOps.cpp.inc", 
        "TributeDialect.h.inc",
        "TributeDialect.cpp.inc",
        "TributeTypes.h.inc",
        "TributeTypes.cpp.inc",
    ];
    
    for file in &expected_files {
        let file_path = build_dir.join(file);
        if !file_path.exists() {
            return Err(format!("Expected generated file not found: {}", file_path.display()).into());
        }
    }
    
    // Tell cargo where to find the generated headers
    println!("cargo:include={}", build_dir.display());
    
    // Set environment variable for use in the crate
    println!("cargo:rustc-env=TRIBUTE_TABLEGEN_INCLUDE_DIR={}", build_dir.display());
    
    println!("cargo:warning=CMake TableGen generation completed successfully");
    Ok(())
}

fn try_tablegen_generation() -> Result<(), Error> {
    use tblgen::{RecordKeeper, TableGenParser};

    // Try to find LLVM installation
    let llvm_prefix = find_llvm_installation()?;

    // Parse the TableGen file
    let keeper: RecordKeeper = TableGenParser::new()
        .add_source_file("tablegen/TributeOps.td")
        .add_include_directory(&format!("{}/include", llvm_prefix))
        .parse()?;

    // Generate Rust code from TableGen definitions
    let out_dir = env::var("OUT_DIR")?;
    let dest_path = Path::new(&out_dir).join("tablegen_ops.rs");
    let mut dest = fs::File::create(&dest_path).map_err(|e| {
        format!(
            "Failed to create output file {}: {}",
            dest_path.display(),
            e
        )
    })?;
    generate_rust_code_from_tablegen(&mut dest, &keeper)?;

    Ok(())
}

fn find_llvm_installation() -> Result<String, Error> {
    // Try different environment variables for LLVM
    for version in ["200", "190", "180", "170", "160"] {
        let env_var = format!("TABLEGEN_{}_PREFIX", version);
        if let Ok(prefix) = env::var(&env_var) {
            return Ok(prefix);
        }
    }

    // Try common installation paths
    let common_paths = [
        "/usr/local/llvm",
        "/opt/homebrew/opt/llvm",
        "/usr/lib/llvm-17",
        "/usr/lib/llvm-16",
    ];

    for path in &common_paths {
        if Path::new(path).join("include/mlir").exists() {
            return Ok(path.to_string());
        }
    }

    Err("LLVM installation not found. Please set TABLEGEN_*_PREFIX environment variable.".into())
}

fn generate_rust_code_from_tablegen(
    dest: &mut dyn Write,
    keeper: &tblgen::RecordKeeper,
) -> Result<(), Error> {
    writeln!(dest, "// Generated code from TableGen")?;
    writeln!(
        dest,
        "// DO NOT EDIT - this file is automatically generated\n"
    )?;

    // Extract actual dialect information from TableGen
    let DialectInfo {
        name,
        namespace,
        summary,
        description,
    } = extract_dialect_info(keeper)?;
    let operation_info = extract_operation_info(keeper)?;
    let type_info = extract_type_info(keeper)?;

    // Generate dialect metadata
    writeln!(dest, "/// Dialect information extracted from TableGen")?;
    writeln!(dest, "pub mod dialect_info {{")?;
    writeln!(dest, "    pub const NAME: &str = \"{name}\";")?;
    writeln!(dest, "    pub const NAMESPACE: &str = \"{namespace}\";")?;
    writeln!(dest, "    pub const SUMMARY: &str = r#\"{summary}\"#;")?;
    writeln!(
        dest,
        "    pub const DESCRIPTION: &str = r#\"{description}\"#;"
    )?;
    writeln!(dest, "}}")?;

    // Generate operation names
    writeln!(dest, "/// Operation names generated from TableGen")?;
    writeln!(dest, "pub mod generated_ops {{")?;

    // Use operations extracted from TableGen
    for op in &operation_info {
        let const_name = op.mnemonic.to_uppercase();
        writeln!(
            dest,
            "    pub const {const_name}: &str = \"tribute.{}\";",
            op.mnemonic
        )?;
    }
    let operations_count = operation_info.len();

    writeln!(dest, "}}")?;

    // Generate operation metadata
    writeln!(dest, "/// Operation metadata extracted from TableGen")?;
    writeln!(dest, "pub mod operation_metadata {{")?;
    writeln!(dest, "    use super::generated_ops;")?;
    writeln!(dest)?;
    
    writeln!(dest, "    pub struct OperationInfo {{")?;
    writeln!(dest, "        pub name: &'static str,")?;
    writeln!(dest, "        pub mnemonic: &'static str,")?;
    writeln!(dest, "        pub summary: &'static str,")?;
    writeln!(dest, "        pub description: &'static str,")?;
    writeln!(dest, "        pub traits: &'static [&'static str],")?;
    writeln!(dest, "    }}")?;
    writeln!(dest)?;
    
    writeln!(dest, "    pub const OPERATIONS: &[OperationInfo] = &[")?;
    for op in &operation_info {
        let const_name = op.mnemonic.to_uppercase();
        writeln!(dest, "        OperationInfo {{")?;
        writeln!(dest, "            name: generated_ops::{const_name},")?;
        writeln!(dest, "            mnemonic: \"{}\",", op.mnemonic)?;
        writeln!(dest, "            summary: r#\"{}\"#,", op.summary)?;
        writeln!(dest, "            description: r#\"{}\"#,", op.description)?;
        writeln!(dest, "            traits: &[")?;
        for trait_name in &op.traits {
            writeln!(dest, "                \"{trait_name}\",")?;
        }
        writeln!(dest, "            ],")?;
        writeln!(dest, "        }},")?;
    }
    writeln!(dest, "    ];")?;
    writeln!(dest, "    pub const OPERATION_COUNT: usize = OPERATIONS.len();")?;
    writeln!(dest, "}}")?;

    // Generate type information if available
    if !type_info.is_empty() {
        writeln!(dest, "/// Type information extracted from TableGen")?;
        writeln!(dest, "pub mod type_info {{")?;
        writeln!(dest, "    pub struct TypeInfo {{")?;
        writeln!(dest, "        pub name: &'static str,")?;
        writeln!(dest, "        pub mnemonic: &'static str,")?;
        writeln!(dest, "        pub summary: &'static str,")?;
        writeln!(dest, "    }}")?;
        writeln!(dest)?;
        
        writeln!(dest, "    pub const TYPES: &[TypeInfo] = &[")?;
        for typ in &type_info {
            writeln!(dest, "        TypeInfo {{")?;
            writeln!(dest, "            name: \"{}\",", typ.name)?;
            writeln!(dest, "            mnemonic: \"{}\",", typ.mnemonic)?;
            writeln!(dest, "            summary: r#\"{}\"#,", typ.summary)?;
            writeln!(dest, "        }},")?;
        }
        writeln!(dest, "    ];")?;
        writeln!(dest, "}}")?;
    }

    // Generate initialization function based on toy dialect pattern
    writeln!(dest, "/// Dialect initialization functions")?;
    writeln!(dest, "pub mod initialization {{")?;
    writeln!(dest, "    use super::*;")?;
    writeln!(dest)?;

    writeln!(dest, "    /// Initialize the Tribute dialect")?;
    writeln!(
        dest,
        "    /// This mimics the pattern from ToyDialect::initialize() in C++"
    )?;
    writeln!(
        dest,
        "    pub fn initialize_tribute_dialect() -> Result<(), String> {{"
    )?;
    writeln!(dest, "        // In C++, this would call:")?;
    writeln!(dest, "        // addOperations<GET_OP_LIST>();")?;
    writeln!(dest, "        // addTypes<GET_TYPEDEF_LIST>();")?;
    writeln!(dest, "        // addInterfaces<TributeInlinerInterface>();")?;
    writeln!(dest)?;
    writeln!(
        dest,
        "        // For now, we verify that our TableGen parsing worked"
    )?;
    writeln!(
        dest,
        "        println!(\"Tribute Dialect: {{}}\", dialect_info::NAME);"
    )?;
    writeln!(
        dest,
        "        println!(\"Namespace: {{}}\", dialect_info::NAMESPACE);"
    )?;
    writeln!(
        dest,
        "        println!(\"Operations defined: {{}}\", operation_metadata::OPERATION_COUNT);"
    )?;
    
    writeln!(dest, "        println!(\"Detailed operations:\");")?;
    writeln!(dest, "        for op in operation_metadata::OPERATIONS {{")?;
    writeln!(dest, "            println!(\"  - {{}} ({{}})\", op.name, op.summary);")?;
    writeln!(dest, "        }}")?;
    
    if !type_info.is_empty() {
        writeln!(dest, "        println!(\"Types defined:\");")?;
        writeln!(dest, "        for typ in type_info::TYPES {{")?;
        writeln!(dest, "            println!(\"  - {{}} ({{}})\", typ.name, typ.summary);")?;
        writeln!(dest, "        }}")?;
    }

    writeln!(dest, "        Ok(())")?;
    writeln!(dest, "    }}")?;
    writeln!(dest, "}}")?;

    writeln!(dest, "// TableGen processing completed successfully")?;
    writeln!(
        dest,
        "// Found {} operations and {} types",
        operations_count,
        type_info.len()
    )?;

    Ok(())
}

#[derive(Debug)]
struct DialectInfo {
    name: String,
    namespace: String,
    summary: String,
    description: String,
}

#[derive(Debug)]
struct OperationInfo {
    #[allow(dead_code)]
    name: String,
    mnemonic: String,
    summary: String,
    description: String,
    traits: Vec<String>,
}

#[derive(Debug)]
struct TypeInfo {
    name: String,
    mnemonic: String,
    summary: String,
}

fn extract_dialect_info(keeper: &tblgen::RecordKeeper) -> Result<DialectInfo, Error> {
    // Try to extract dialect information from TableGen records
    // This is a simplified version - real implementation would parse the actual records

    // Check if we can find dialect records
    let dialect_records: Vec<_> = keeper.all_derived_definitions("Dialect").collect();

    if let Some(dialect_record) = dialect_records.into_iter().next() {
        let name = dialect_record
            .string_value("name")
            .unwrap_or_else(|_| "tribute".to_string());
        let summary = dialect_record
            .string_value("summary")
            .unwrap_or_else(|_| "The Tribute programming language dialect".to_string());
        let description = dialect_record
            .string_value("description")
            .unwrap_or_else(|_| "Tribute dialect for MLIR".to_string());
        let namespace = dialect_record
            .string_value("cppNamespace")
            .unwrap_or_else(|_| "::mlir::tribute".to_string())
            .trim_start_matches("::")
            .to_string();

        Ok(DialectInfo {
            name,
            namespace,
            summary,
            description,
        })
    } else {
        // Fallback to hardcoded values
        Ok(DialectInfo {
            name: "tribute".to_string(),
            namespace: "mlir::tribute".to_string(),
            summary: "The Tribute programming language dialect".to_string(),
            description: "Tribute dialect for MLIR representing Tribute language constructs"
                .to_string(),
        })
    }
}

fn extract_operation_info(keeper: &tblgen::RecordKeeper) -> Result<Vec<OperationInfo>, Error> {
    let mut operations = Vec::new();
    let mut seen_operations = std::collections::HashSet::new();

    // Only look for Tribute_Op classes to avoid duplicates
    let op_records: Vec<_> = keeper.all_derived_definitions("Tribute_Op").collect();

    println!("cargo:warning=Found {} Tribute operation records", op_records.len());

    for record in op_records {
        if let Ok(name) = record.name() {
            // Skip if we've already seen this operation
            if !seen_operations.insert(name.to_string()) {
                continue;
            }
            
            // Skip non-tribute operations
            if !name.starts_with("Tribute_") {
                continue;
            }
            
            // Extract the actual mnemonic (remove "Tribute_" prefix and "Op" suffix)
            let clean_mnemonic = if name.starts_with("Tribute_") && name.ends_with("Op") {
                let clean = name.strip_prefix("Tribute_").unwrap_or(name);
                let clean = clean.strip_suffix("Op").unwrap_or(clean);
                
                // Map some operation names to match expected names
                match clean {
                    "StringConcat" => "string_concat".to_string(),
                    "StringInterpolation" => "string_interpolation".to_string(),
                    "ToRuntime" => "to_runtime".to_string(),
                    other => other.to_lowercase(),
                }
            } else {
                name.to_lowercase()
            };
            
            println!("cargo:warning=Found operation: {} -> {}", name, clean_mnemonic);

            let full_name = format!("tribute.{}", clean_mnemonic);
            let summary = record
                .string_value("summary")
                .unwrap_or_else(|_| format!("Tribute {} operation", clean_mnemonic));
            let description = record
                .string_value("description")
                .unwrap_or_else(|_| format!("The {} operation in Tribute dialect", clean_mnemonic));

            // Extract traits (simplified)
            let traits = vec!["TributeOp".to_string()]; // Placeholder

            operations.push(OperationInfo {
                name: full_name,
                mnemonic: clean_mnemonic,
                summary,
                description,
                traits,
            });
        }
    }

    println!("cargo:warning=Extracted {} unique operations from TableGen", operations.len());
    Ok(operations)
}

fn extract_type_info(keeper: &tblgen::RecordKeeper) -> Result<Vec<TypeInfo>, Error> {
    let mut types = Vec::new();

    // Try to find type records
    let type_records: Vec<_> = keeper.all_derived_definitions("TypeDef").collect();

    for record in type_records {
        if let Ok(mnemonic) = record.string_value("mnemonic") {
            let name = format!("!tribute.{}", mnemonic);
            let summary = record
                .string_value("summary")
                .unwrap_or_else(|_| format!("Tribute {} type", mnemonic));

            types.push(TypeInfo {
                name,
                mnemonic,
                summary,
            });
        }
    }

    Ok(types)
}
