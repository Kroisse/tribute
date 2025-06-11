# CMakeLists.txt for Tribute HIR Dialect
cmake_minimum_required(VERSION 3.20)
project(tribute-hir-dialect LANGUAGES CXX C)

# Find MLIR and LLVM
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Include directories
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

add_definitions(${LLVM_DEFINITIONS})

#-------------------------------------------------------------------------------
# TableGen for TributeOps.td
#-------------------------------------------------------------------------------

set(LLVM_TARGET_DEFINITIONS tablegen/TributeOps.td)

# Generate operation declarations and definitions
mlir_tablegen(TributeOps.h.inc -gen-op-decls)
mlir_tablegen(TributeOps.cpp.inc -gen-op-defs)

# Generate dialect declarations and definitions
mlir_tablegen(TributeDialect.h.inc -gen-dialect-decls -dialect=tribute)
mlir_tablegen(TributeDialect.cpp.inc -gen-dialect-defs -dialect=tribute)

# Generate type declarations and definitions
mlir_tablegen(TributeTypes.h.inc -gen-typedef-decls)
mlir_tablegen(TributeTypes.cpp.inc -gen-typedef-defs)

# Generate attribute declarations and definitions (if needed)
mlir_tablegen(TributeAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(TributeAttributes.cpp.inc -gen-attrdef-defs)

# Create a target for the generated files
add_public_tablegen_target(TributeOpsIncGen)

#-------------------------------------------------------------------------------
# Documentation generation (optional)
#-------------------------------------------------------------------------------

add_mlir_doc(TributeOps TributeOps . -gen-op-doc)
add_mlir_doc(TributeOps TributeDialect . -gen-dialect-doc)

#-------------------------------------------------------------------------------
# Library target
#-------------------------------------------------------------------------------

# For now, we just ensure the TableGen files are generated
# The actual C++ implementation will be handled by Rust through bindings
add_custom_target(tribute-dialect ALL
  DEPENDS TributeOpsIncGen
  COMMENT "Generating Tribute dialect TableGen files"
)

# Files are generated in CMAKE_CURRENT_BINARY_DIR and cmake-rs will find them there
# No need to install - build.rs will locate them directly