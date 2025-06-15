//! Tribute compiler and interpreter
//!
//! This is a command-line tool that can both interpret and compile Tribute programs.
//!
//! # Usage
//!
//! ## Interpreter mode (default)
//! ```bash
//! trbc program.trb
//! ```
//!
//! ## Compiler mode
//! ```bash
//! trbc --compile program.trb -o output_binary
//! ```

use clap::{Arg, ArgAction, Command};
use std::path::PathBuf;
use tribute::{TributeDatabaseImpl, eval_str, parse_str};
use tribute_cranelift::TributeCompiler;
use tribute_hir::queries::lower_program_to_hir;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("trbc")
        .version("0.1.0")
        .about("Tribute compiler and interpreter")
        .arg(
            Arg::new("input")
                .help("Input Tribute source file")
                .required_unless_present("test")
                .value_name("FILE")
                .index(1),
        )
        .arg(
            Arg::new("compile")
                .long("compile")
                .short('c')
                .help("Compile to native binary instead of interpreting")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("output")
                .long("output")
                .short('o')
                .help("Output file path (required when compiling)")
                .value_name("OUTPUT")
                .requires("compile"),
        )
        .arg(
            Arg::new("test")
                .long("test")
                .help("Run compilation tests on all examples in lang-examples/")
                .action(ArgAction::SetTrue)
                .conflicts_with_all(["input", "compile"]),
        )
        .get_matches();

    let test_mode = matches.get_flag("test");
    let compile_mode = matches.get_flag("compile");

    if test_mode {
        run_compilation_tests()?;
    } else if compile_mode {
        let input_path = PathBuf::from(
            matches
                .get_one::<String>("input")
                .ok_or("Input file is required when compiling")?,
        );
        let output_path = matches
            .get_one::<String>("output")
            .ok_or("Output path is required when compiling")?;
        compile_program(&input_path, output_path)?;
    } else {
        // Interpreter mode (default)
        let input_path = PathBuf::from(
            matches
                .get_one::<String>("input")
                .ok_or("Input file is required for interpretation")?,
        );
        interpret_program(&input_path)?;
    }

    Ok(())
}

/// Interprets a Tribute program
fn interpret_program(path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let source = std::fs::read_to_string(path)?;
    let db = TributeDatabaseImpl::default();

    match eval_str(&db, path, &source) {
        Ok(result) => {
            // Only print non-unit results
            match result {
                tribute::Value::Unit => {} // Don't print unit values
                _ => println!("{}", result),
            }
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}

/// Runs compilation tests on example files
fn run_compilation_tests() -> Result<(), Box<dyn std::error::Error>> {
    println!("Running compilation tests...");

    // Find lang-examples directory
    let examples_dir = std::env::current_dir()?.join("lang-examples");
    if !examples_dir.exists() {
        eprintln!(
            "Warning: lang-examples directory not found at {}",
            examples_dir.display()
        );
        return Ok(());
    }

    // Test cases that should compile successfully
    let should_compile = &[
        "basic.trb",
        "functions.trb",
        "pattern_matching.trb",
        "pattern_advanced.trb",
        "hello.trb",
        "let_simple.trb",
        "let_bindings.trb",
        "calc.trb",
        "empty_program.trb",
        "simple_function.trb",
    ];

    // Test cases that should fail to compile
    let should_fail = &[
        "string_interpolation.trb", // Complex interpolation not implemented
    ];

    let mut success_count = 0;
    let mut failure_count = 0;
    let mut total_size = 0;

    println!("\n=== Testing successful compilation cases ===");
    for &example_name in should_compile {
        let example_path = examples_dir.join(example_name);
        if example_path.exists() {
            print!("Testing {}... ", example_name);
            match test_compile_file(&example_path) {
                Ok(size) => {
                    println!("✓ ({} bytes)", size);
                    success_count += 1;
                    total_size += size;
                }
                Err(e) => {
                    println!("✗ Failed: {}", e);
                    failure_count += 1;
                }
            }
        } else {
            println!("Skipping {} (file not found)", example_name);
        }
    }

    println!("\n=== Testing expected failure cases ===");
    for &example_name in should_fail {
        let example_path = examples_dir.join(example_name);
        if example_path.exists() {
            print!("Testing {} (should fail)... ", example_name);
            match test_compile_file(&example_path) {
                Ok(_) => {
                    println!("✗ Unexpectedly succeeded");
                    failure_count += 1;
                }
                Err(_) => {
                    println!("✓ Failed as expected");
                    success_count += 1;
                }
            }
        } else {
            println!("Skipping {} (file not found)", example_name);
        }
    }

    println!("\n=== Test Results ===");
    println!("✓ Successful tests: {}", success_count);
    println!("✗ Failed tests: {}", failure_count);
    if success_count > 0 {
        println!(
            "📊 Average object size: {} bytes",
            total_size / success_count
        );
        println!("📏 Total object size: {} bytes", total_size);
    }

    if failure_count > 0 {
        println!("\nSome tests failed. Check the output above for details.");
        std::process::exit(1);
    } else {
        println!("\n🎉 All compilation tests passed!");
    }

    Ok(())
}

/// Test compiling a single file and return the object size
fn test_compile_file(path: &PathBuf) -> Result<usize, Box<dyn std::error::Error>> {
    let source = std::fs::read_to_string(path)?;
    let db = TributeDatabaseImpl::default();

    // Parse to AST
    let (program, diagnostics) = parse_str(&db, path, &source);

    // Check for parsing errors
    if !diagnostics.is_empty() {
        return Err(format!("Parsing errors: {:?}", diagnostics).into());
    }

    // Lower to HIR
    let hir_program = lower_program_to_hir(&db, program).ok_or("Failed to lower program to HIR")?;

    // Create Cranelift compiler
    let compiler = TributeCompiler::new(&db, None)?; // Use native target

    // Compile to object code
    let object_bytes = compiler.compile_program(&db, hir_program)?;

    Ok(object_bytes.len())
}

/// Compiles a Tribute program to native binary
fn compile_program(
    input_path: &PathBuf,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Compiling {} to {}...", input_path.display(), output_path);

    // Read source code
    let source = std::fs::read_to_string(input_path)?;
    let db = TributeDatabaseImpl::default();

    // Parse to AST
    let (program, diagnostics) = parse_str(&db, input_path, &source);

    // Check for parsing errors
    if !diagnostics.is_empty() {
        eprintln!("Compilation errors:");
        for diagnostic in diagnostics {
            eprintln!("  {:?}", diagnostic);
        }
        std::process::exit(1);
    }

    // Lower to HIR
    let hir_program = lower_program_to_hir(&db, program).ok_or("Failed to lower program to HIR")?;

    // Create Cranelift compiler
    let compiler = TributeCompiler::new(&db, None)?; // Use native target

    // Compile to object code
    let object_bytes = compiler.compile_program(&db, hir_program)?;

    // Save intermediate object file for debugging
    let object_path = format!("{}.o", output_path);
    std::fs::write(&object_path, &object_bytes)?;
    println!(
        "Generated object file: {} ({} bytes)",
        object_path,
        object_bytes.len()
    );

    // Link to create executable
    link_executable(&object_path, output_path)?;

    // Clean up intermediate object file
    std::fs::remove_file(&object_path)?;

    println!("✅ Successfully compiled to executable: {}", output_path);

    Ok(())
}

/// Links object file with system libraries to create an executable
fn link_executable(object_path: &str, output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    use std::process::Command;

    println!("Linking executable...");

    // Determine the appropriate linker and flags for the current platform
    let (linker, args) = if cfg!(target_os = "macos") {
        // macOS linking with ld
        let arch = match std::env::consts::ARCH {
            "aarch64" => "arm64",
            "x86_64" => "x86_64",
            other => other,
        };

        let mut args = vec![
            "-o".to_string(),
            output_path.to_string(),
            object_path.to_string(),
            "-lSystem".to_string(), // Link with system library
            "-arch".to_string(),
            arch.to_string(),
        ];

        // Add platform-specific flags
        args.extend([
            "-platform_version".to_string(),
            "macos".to_string(),
            "10.15".to_string(), // Minimum macOS version
            "15.0".to_string(),  // SDK version
        ]);

        ("ld", args)
    } else if cfg!(target_os = "linux") {
        // Linux linking with ld
        let args = vec![
            "-o".to_string(),
            output_path.to_string(),
            object_path.to_string(),
            "-lc".to_string(), // Link with libc
            "--dynamic-linker".to_string(),
            "/lib64/ld-linux-x86-64.so.2".to_string(), // Default dynamic linker path
        ];
        ("ld", args)
    } else if cfg!(target_os = "windows") {
        // Windows linking with link.exe (MSVC)
        let args = vec![
            format!("/OUT:{}", output_path),
            object_path.to_string(),
            "kernel32.lib".to_string(),
            "msvcrt.lib".to_string(),
        ];
        ("link", args)
    } else {
        return Err("Unsupported target platform for linking".into());
    };

    // Try to use system linker first
    let mut cmd = Command::new(linker);
    cmd.args(&args);

    println!("Running linker: {} {}", linker, args.join(" "));

    let output = cmd.output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);

        // If system linker fails, try using rustc as a fallback linker
        println!("System linker failed, trying rustc as fallback...");
        println!("Linker stderr: {}", stderr);
        println!("Linker stdout: {}", stdout);

        return link_with_rustc(object_path, output_path);
    }

    println!("Linking completed successfully");
    Ok(())
}

/// Fallback linking using rustc (which handles cross-platform linking better)
fn link_with_rustc(object_path: &str, output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    use std::process::Command;

    println!("Using rustc for linking...");

    // Create a minimal main.rs file that includes our object
    let temp_dir = std::env::temp_dir();
    let main_rs_path = temp_dir.join("tribute_main.rs");
    let main_rs_content = r#"
// Minimal Rust wrapper to link with our Tribute object file
#[link(name = "tribute_object", kind = "static")]
extern "C" {
    fn tribute_main_entry() -> i32;
}

fn main() {
    unsafe {
        std::process::exit(tribute_main_entry() as i32);
    }
}
"#;

    std::fs::write(&main_rs_path, main_rs_content)?;

    // Get the path to the tribute-runtime static library
    let target_dir = std::env::current_dir()?.join("target").join("debug");
    let runtime_lib_path = target_dir.join("libtribute_runtime.a");

    // Build tribute-runtime if it doesn't exist
    if !runtime_lib_path.exists() {
        println!("Building tribute-runtime...");
        let mut build_cmd = Command::new("cargo");
        build_cmd.args(["build", "-p", "tribute-runtime"]);
        let build_output = build_cmd.output()?;
        if !build_output.status.success() {
            return Err(format!(
                "Failed to build tribute-runtime: {}",
                String::from_utf8_lossy(&build_output.stderr)
            )
            .into());
        }
    }

    // Copy runtime library to temp directory
    let temp_runtime_path = temp_dir.join("libtribute_runtime.a");
    std::fs::copy(&runtime_lib_path, &temp_runtime_path)?;

    // Copy object file to temp directory with expected name
    let temp_object_path = temp_dir.join("libtribute_object.a");
    std::fs::copy(object_path, &temp_object_path)?;

    // Use rustc to compile and link
    let mut cmd = Command::new("rustc");
    cmd.args([
        main_rs_path.to_str().unwrap(),
        "-o",
        output_path,
        "-L",
        &temp_dir.to_string_lossy(),
        "-l",
        "static=tribute_object",
        "-l",
        "static=tribute_runtime",
    ]);

    println!("Running rustc linker: {:?}", cmd);

    let output = cmd.output()?;

    // Clean up temporary files
    let _ = std::fs::remove_file(&main_rs_path);
    let _ = std::fs::remove_file(&temp_object_path);
    let _ = std::fs::remove_file(&temp_runtime_path);

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(format!(
            "Rustc linking failed:\nstdout: {}\nstderr: {}",
            stdout, stderr
        )
        .into());
    }

    println!("Rustc linking completed successfully");
    Ok(())
}
