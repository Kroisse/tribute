# Compiler Implementation Plan

**Status**: 📋 **PLANNED** - Not yet started  
**Prerequisites**: ✅ Modern Syntax (Plan 01), ✅ String Interpolation (Plan 01.01), 📋 HIR to MLIR (Plan 02.01), 📋 MLIR Interpreter (Plan 02.02)  
**Estimated Timeline**: 3-4 months (after 02.02)  
**Complexity**: High

## Overview

This plan outlines the implementation of a native Tribute compiler using MLIR (Multi-Level Intermediate Representation). Instead of maintaining a separate HIR in Rust, we define Tribute operations directly as an MLIR dialect, leveraging MLIR's infrastructure for transformations and optimizations.

### Strategic Goals

1. **Performance**: Native compilation for production workloads
2. **Optimization**: Leverage MLIR's advanced optimization passes
3. **Interoperability**: Enable integration with other MLIR-based languages
4. **Debugging**: Utilize MLIR's debugging and profiling tools
5. **Future-proofing**: Build foundation for advanced language features

### Implementation Approach

**Dynamic-First with Gradual Optimization**
- Start with fully dynamic typing using `!tribute.value` (mirrors current HIR)
- All operations use runtime type checking initially
- Later add type inference and specialized code paths for typed expressions
- Gradual migration allows leveraging existing HIR evaluation logic

**Why This Approach**:
- Gradual typing requires both dynamic and static systems anyway
- Current HIR logic can be directly translated to MLIR operations
- Type system can be added incrementally as optimization layer
- Lower implementation risk with proven dynamic foundation

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Defining the Tribute Dialect](#defining-the-tribute-dialect)
3. [Tribute Types in TableGen](#tribute-types-in-tablegen)
4. [Tribute Operations](#tribute-operations)
5. [C++ Implementation](#c-implementation)
6. [Rust Integration with melior](#rust-integration-with-melior)
7. [AST to Tribute Dialect Lowering](#ast-to-tribute-dialect-lowering)
8. [Progressive Lowering Pipeline](#progressive-lowering-pipeline)
9. [Runtime Integration](#runtime-integration)
10. [Testing Strategy](#testing-strategy)

## Architecture Overview

### Traditional Approach vs MLIR Dialect Approach

**Traditional Approach:**
```
AST (Rust) → HIR (Rust) → MLIR → LLVM IR
```

**MLIR Dialect Approach:**
```
AST (Rust) → Tribute Dialect (MLIR) → Standard Dialects → LLVM IR
```

### Benefits of the MLIR Dialect Approach

1. **Unified Infrastructure**: Leverage MLIR's passes, transformations, and tools
2. **Declarative Definitions**: Use TableGen for concise operation definitions
3. **Automatic Code Generation**: Parsers, printers, and verifiers are generated
4. **Progressive Lowering**: Natural transformation pipeline through MLIR dialects
5. **Better Debugging**: MLIR's built-in debugging and visualization tools

### Project Structure

```
tribute-codegen/
├── mlir/
│   ├── include/
│   │   └── Tribute/
│   │       ├── TributeDialect.h
│   │       ├── TributeDialect.td     # Dialect definition
│   │       ├── TributeOps.h
│   │       ├── TributeOps.td         # Operations definition
│   │       ├── TributeTypes.h
│   │       └── TributeTypes.td       # Types definition
│   ├── lib/
│   │   └── Tribute/
│   │       ├── TributeDialect.cpp
│   │       ├── TributeOps.cpp
│   │       └── TributeTypes.cpp
│   └── CMakeLists.txt
├── src/
│   ├── lib.rs
│   ├── ast_lowering.rs              # AST to Tribute dialect
│   ├── dialect_lowering.rs          # Tribute to standard dialects
│   ├── mlir_builder.rs              # MLIR construction helpers
│   └── codegen.rs                   # Main codegen orchestration
└── Cargo.toml
```

## Defining the Tribute Dialect

### TributeDialect.td

```tablegen
//===----------------------------------------------------------------------===//
// Tribute Dialect Definition
//===----------------------------------------------------------------------===//

#ifndef TRIBUTE_DIALECT
#define TRIBUTE_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Tribute Dialect
//===----------------------------------------------------------------------===//

def Tribute_Dialect : Dialect {
  let name = "tribute";
  let summary = "High-level dialect for the Tribute programming language";
  let description = [{
    The Tribute dialect represents high-level constructs of the Tribute
    programming language. It includes:
    - Dynamic typing with runtime type information
    - First-class functions and closures
    - Pattern matching
    - String interpolation
    - List and array operations
    
    This dialect serves as the initial lowering target from the Tribute AST
    and progressively lowers to standard MLIR dialects.
  }];
  
  let cppNamespace = "::mlir::tribute";
  
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  
  // Extra class declarations
  let extraClassDeclaration = [{
    /// Register all Tribute types
    void registerTypes();
    
    /// Register all Tribute attributes
    void registerAttributes();
    
    /// Dialect initialization
    static void initialize();
  }];
  
  // Dialect dependencies
  let dependentDialects = [
    "arith::ArithDialect",
    "cf::ControlFlowDialect",
    "func::FuncDialect",
    "scf::SCFDialect",
    "llvm::LLVMDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Base Classes
//===----------------------------------------------------------------------===//

// Base class for all Tribute operations
class Tribute_Op<string mnemonic, list<Trait> traits = []> :
    Op<Tribute_Dialect, mnemonic, traits>;

// Type constraints
def Tribute_Type : DialectType<Tribute_Dialect,
    CPred<"$_self.isa<::mlir::tribute::TributeType>()">,
    "Tribute type">;

def Tribute_ValueType : DialectType<Tribute_Dialect,
    CPred<"$_self.isa<::mlir::tribute::ValueType>()">,
    "Tribute value type">;

def Tribute_StringType : DialectType<Tribute_Dialect,
    CPred<"$_self.isa<::mlir::tribute::StringType>()">,
    "Tribute string type">;

def Tribute_ListType : DialectType<Tribute_Dialect,
    CPred<"$_self.isa<::mlir::tribute::ListType>()">,
    "Tribute list type">;

def Tribute_FunctionType : DialectType<Tribute_Dialect,
    CPred<"$_self.isa<::mlir::tribute::FunctionType>()">,
    "Tribute function type">;

#endif // TRIBUTE_DIALECT
```

## Tribute Types in TableGen

### TributeTypes.td

```tablegen
//===----------------------------------------------------------------------===//
// Tribute Types
//===----------------------------------------------------------------------===//

#ifndef TRIBUTE_TYPES
#define TRIBUTE_TYPES

include "mlir/IR/AttrTypeBase.td"
include "TributeDialect.td"

//===----------------------------------------------------------------------===//
// Value Type - Dynamic typed value
//===----------------------------------------------------------------------===//

def Tribute_ValueType : TypeDef<Tribute_Dialect, "Value"> {
  let mnemonic = "value";
  
  let summary = "Dynamically typed Tribute value";
  let description = [{
    Represents a dynamically typed value in Tribute. At runtime, this can be:
    - Number (f64)
    - String
    - Boolean
    - List
    - Function
    - Nil
    
    This type is used when static type information is not available at compile time.
    Operations on !tribute.value require runtime type checking and may fail at runtime.
    
    For performance-critical code, prefer using specific types (f64, !tribute.string, etc.)
    when the type is statically known.
    
    Example:
      !tribute.value  // Fully dynamic
      f64             // Statically known number
      !tribute.string // Statically known string
  }];
  
  let extraClassDeclaration = [{
    /// Check if this type can be statically determined
    bool isStaticallyKnown() const { return false; }
    
    /// Get the runtime representation size
    unsigned getRuntimeSize() const { return 64; /* Handle size */ }
  }];
}

//===----------------------------------------------------------------------===//
// String Type
//===----------------------------------------------------------------------===//

def Tribute_StringType : TypeDef<Tribute_Dialect, "String"> {
  let mnemonic = "string";
  
  let summary = "Tribute string type";
  let description = [{
    Represents an immutable UTF-8 string in Tribute.
    Strings support interpolation at the language level.
    
    Example:
      !tribute.string
  }];
  
  let extraClassDeclaration = [{
    /// Get the runtime representation type
    Type getRuntimeType() const;
  }];
}

//===----------------------------------------------------------------------===//
// List Type
//===----------------------------------------------------------------------===//

def Tribute_ListType : TypeDef<Tribute_Dialect, "List"> {
  let mnemonic = "list";
  
  let summary = "Tribute list type";
  let description = [{
    Represents a dynamically-sized list of values.
    Lists are heterogeneous and can contain any Tribute value.
    
    Example:
      !tribute.list<!tribute.value>    // List of dynamic values
      !tribute.list<!tribute.string>   // List of strings
  }];
  
  let parameters = (ins
    "Type":$elementType
  );
  
  let builders = [
    TypeBuilderWithInferredContext<(ins
      "Type":$elementType
    ), [{
      return $_get(elementType.getContext(), elementType);
    }]>
  ];
  
  let extraClassDeclaration = [{
    /// Check if the list has a homogeneous element type
    bool isHomogeneous() const {
      return !getElementType().isa<ValueType>();
    }
  }];
  
  let assemblyFormat = "`<` $elementType `>`";
}

//===----------------------------------------------------------------------===//
// Function Type
//===----------------------------------------------------------------------===//

def Tribute_FunctionType : TypeDef<Tribute_Dialect, "Function"> {
  let mnemonic = "function";
  
  let summary = "Tribute function type";
  let description = [{
    Represents a function type with parameter and return types.
    Functions are first-class values and support closures.
    
    Example:
      !tribute.function<(!tribute.value, !tribute.value) -> !tribute.value>
  }];
  
  let parameters = (ins
    ArrayRefParameter<"Type">:$inputs,
    ArrayRefParameter<"Type">:$results
  );
  
  let builders = [
    TypeBuilder<(ins
      "ArrayRef<Type>":$inputs,
      "ArrayRef<Type>":$results
    ), [{
      return $_get($_ctxt, inputs, results);
    }]>
  ];
  
  let extraClassDeclaration = [{
    /// Get the number of inputs
    unsigned getNumInputs() const { return getInputs().size(); }
    
    /// Get the number of results
    unsigned getNumResults() const { return getResults().size(); }
    
    /// Check if this is a void function
    bool isVoid() const { return getNumResults() == 0; }
  }];
  
  let assemblyFormat = "`<` `(` $inputs `)` `->` `(` $results `)` `>`";
}

//===----------------------------------------------------------------------===//
// Array Type (for internal use)
//===----------------------------------------------------------------------===//

def Tribute_ArrayType : TypeDef<Tribute_Dialect, "Array"> {
  let mnemonic = "array";
  
  let summary = "Fixed-size array type";
  let description = [{
    Represents a fixed-size array, primarily used for internal
    optimizations when list sizes are statically known.
    
    Example:
      !tribute.array<10xf64>
  }];
  
  let parameters = (ins
    "int64_t":$size,
    "Type":$elementType
  );
  
  let assemblyFormat = "`<` $size `x` $elementType `>`";
}

//===----------------------------------------------------------------------===//
// Comparison Predicate Attribute
//===----------------------------------------------------------------------===//

def Tribute_CmpPredicateAttr : EnumAttr<Tribute_Dialect, Tribute_CmpPredicate, "cmp_predicate"> {
  let summary = "Comparison predicate for tribute.cmp operation";
  let description = [{
    Specifies the type of comparison to perform in tribute.cmp operations.
    
    Available predicates:
    - eq: equal
    - ne: not equal  
    - lt: less than
    - le: less than or equal
    - gt: greater than
    - ge: greater than or equal
  }];
}

def Tribute_CmpPredicate : I32EnumAttr<"CmpPredicate", "Tribute comparison predicate", [
  I32EnumAttrCase<"eq", 0, "eq">,
  I32EnumAttrCase<"ne", 1, "ne">,
  I32EnumAttrCase<"lt", 2, "lt">,
  I32EnumAttrCase<"le", 3, "le">,
  I32EnumAttrCase<"gt", 4, "gt">,
  I32EnumAttrCase<"ge", 5, "ge">
]> {
  let cppNamespace = "::mlir::tribute";
}

#endif // TRIBUTE_TYPES
```

## Tribute Operations

### TributeOps.td

```tablegen
//===----------------------------------------------------------------------===//
// Tribute Operations
//===----------------------------------------------------------------------===//

#ifndef TRIBUTE_OPS
#define TRIBUTE_OPS

include "TributeDialect.td"
include "TributeTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Constants and Literals
//===----------------------------------------------------------------------===//

def Tribute_ConstantOp : Tribute_Op<"constant", [
    ConstantLike, Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Create a constant value";
  let description = [{
    Creates a constant value of any Tribute type.
    
    Examples:
      %0 = tribute.constant 42.0 : f64
      %1 = tribute.constant "hello" : !tribute.string
      %2 = tribute.constant true : i1
  }];
  
  let arguments = (ins AnyAttr:$value);
  let results = (outs Tribute_Type:$result);
  
  let builders = [
    OpBuilder<(ins "Attribute":$value, "Type":$type), [{
      $_state.addAttribute("value", value);
      $_state.addTypes(type);
    }]>
  ];
  
  let hasFolder = 1;
  let hasVerifier = 1;
  
  let assemblyFormat = "attr-dict $value `:` type($result)";
}

def Tribute_StringInterpolationOp : Tribute_Op<"string_interpolation", [Pure]> {
  let summary = "String interpolation operation";
  let description = [{
    Constructs a string by interpolating values into a template.
    
    Example:
      // Tribute source: "Hello, \{name}! You have \{count} messages."
      // Lowered to template with positional arguments:
      %result = tribute.string_interpolation "Hello, {0}! You have {1} messages." 
                (%name, %count) : (!tribute.value, !tribute.value) -> !tribute.string
  }];
  
  let arguments = (ins 
    StrAttr:$template,
    Variadic<Tribute_Type>:$args
  );
  let results = (outs Tribute_StringType:$result);
  
  let assemblyFormat = [{
    $template `(` $args `)` attr-dict `:` 
    functional-type($args, $result)
  }];
}

//===----------------------------------------------------------------------===//
// Variable Operations
//===----------------------------------------------------------------------===//

def Tribute_AllocaOp : Tribute_Op<"alloca", []> {
  let summary = "Allocate a mutable variable";
  let description = [{
    Allocates storage for a mutable variable on the stack.
    
    Example:
      %ptr = tribute.alloca : !tribute.ref<!tribute.value>
  }];
  
  let results = (outs AnyType:$result);
  
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Tribute_LoadOp : Tribute_Op<"load", []> {
  let summary = "Load value from variable";
  let description = [{
    Loads a value from a variable reference.
    
    Example:
      %val = tribute.load %ptr : !tribute.ref<!tribute.value>
  }];
  
  let arguments = (ins AnyType:$ptr);
  let results = (outs Tribute_Type:$result);
  
  let assemblyFormat = "$ptr attr-dict `:` type($ptr)";
}

def Tribute_StoreOp : Tribute_Op<"store", []> {
  let summary = "Store value to variable";
  let description = [{
    Stores a value to a variable reference.
    
    Example:
      tribute.store %val, %ptr : !tribute.value, !tribute.ref<!tribute.value>
  }];
  
  let arguments = (ins Tribute_Type:$value, AnyType:$ptr);
  
  let assemblyFormat = "$value `,` $ptr attr-dict `:` type($value) `,` type($ptr)";
}

//===----------------------------------------------------------------------===//
// Arithmetic Operations
//===----------------------------------------------------------------------===//

class Tribute_BinaryOp<string mnemonic, list<Trait> traits = []> :
    Tribute_Op<mnemonic, !listconcat(traits, [Pure])> {
  let arguments = (ins Tribute_Type:$lhs, Tribute_Type:$rhs);
  let results = (outs Tribute_Type:$result);
  
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def Tribute_AddOp : Tribute_BinaryOp<"add"> {
  let summary = "Addition operation";
  let description = [{
    Performs addition on two values. For numbers, this is arithmetic addition.
    For strings, this is concatenation.
    
    Example:
      %sum = tribute.add %a, %b : !tribute.value
  }];
  
  let hasFolder = 1;
}

def Tribute_SubOp : Tribute_BinaryOp<"sub"> {
  let summary = "Subtraction operation";
  let hasFolder = 1;
}

def Tribute_MulOp : Tribute_BinaryOp<"mul"> {
  let summary = "Multiplication operation";
  let hasFolder = 1;
}

def Tribute_DivOp : Tribute_BinaryOp<"div"> {
  let summary = "Division operation";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Comparison Operations
//===----------------------------------------------------------------------===//

def Tribute_CmpOp : Tribute_Op<"cmp", [Pure]> {
  let summary = "Compare two values";
  let description = [{
    Compares two values according to the specified predicate.
    
    Example:
      %result = tribute.cmp eq %a, %b : !tribute.value
  }];
  
  let arguments = (ins 
    Tribute_CmpPredicateAttr:$predicate,
    Tribute_Type:$lhs, 
    Tribute_Type:$rhs
  );
  let results = (outs I1:$result);
  
  let assemblyFormat = [{
    $predicate $lhs `,` $rhs attr-dict `:` type($lhs)
  }];
}

//===----------------------------------------------------------------------===//
// List Operations
//===----------------------------------------------------------------------===//

def Tribute_ListCreateOp : Tribute_Op<"list_create", [Pure]> {
  let summary = "Create a list from elements";
  let description = [{
    Creates a new list containing the provided elements.
    
    Example:
      %list = tribute.list_create [%a, %b, %c] : !tribute.list<!tribute.value>
  }];
  
  let arguments = (ins Variadic<Tribute_Type>:$elements);
  let results = (outs Tribute_ListType:$result);
  
  let assemblyFormat = [{
    `[` $elements `]` attr-dict `:` type($result)
  }];
}

def Tribute_ListGetOp : Tribute_Op<"list_get", [Pure]> {
  let summary = "Get element from list";
  let description = [{
    Retrieves an element from a list at the specified index.
    
    Example:
      %elem = tribute.list_get %list[%index] : !tribute.list<!tribute.value>
  }];
  
  let arguments = (ins 
    Tribute_ListType:$list,
    Index:$index
  );
  let results = (outs Tribute_Type:$result);
  
  let assemblyFormat = [{
    $list `[` $index `]` attr-dict `:` type($list)
  }];
}

def Tribute_ListLengthOp : Tribute_Op<"list_length", [Pure]> {
  let summary = "Get list length";
  let description = [{
    Returns the number of elements in a list.
    
    Example:
      %len = tribute.list_length %list : !tribute.list<!tribute.value>
  }];
  
  let arguments = (ins Tribute_ListType:$list);
  let results = (outs Index:$length);
  
  let assemblyFormat = "$list attr-dict `:` type($list)";
}

//===----------------------------------------------------------------------===//
// Function Operations
//===----------------------------------------------------------------------===//

def Tribute_FuncOp : Tribute_Op<"func", [
    FunctionOpInterface, IsolatedFromAbove, Symbol
  ]> {
  let summary = "Function definition";
  let description = [{
    Defines a Tribute function with a name, parameters, and body.
    
    Example:
      tribute.func @add(%a: !tribute.value, %b: !tribute.value) -> !tribute.value {
        %sum = tribute.add %a, %b : !tribute.value
        tribute.return %sum : !tribute.value
      }
  }];
  
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<Tribute_FunctionType>:$function_type,
    OptionalAttr<StrAttr>:$visibility
  );
  let regions = (region AnyRegion:$body);
  
  let builders = [
    OpBuilder<(ins
      "StringRef":$name,
      "FunctionType":$type,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs
    )>
  ];
  
  let extraClassDeclaration = [{
    /// Returns the function type
    FunctionType getFunctionType() {
      return getFunctionTypeAttr().getValue();
    }
    
    /// Returns the number of arguments
    unsigned getNumArguments() {
      return getFunctionType().getNumInputs();
    }
    
    /// Returns the number of results
    unsigned getNumResults() {
      return getFunctionType().getNumResults();
    }
  }];
  
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Tribute_CallOp : Tribute_Op<"call", [
    CallOpInterface,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
  ]> {
  let summary = "Call a function";
  let description = [{
    Calls a Tribute function with the given arguments.
    
    Example:
      %result = tribute.call @add(%x, %y) : (!tribute.value, !tribute.value) -> !tribute.value
  }];
  
  let arguments = (ins 
    FlatSymbolRefAttr:$callee,
    Variadic<Tribute_Type>:$operands
  );
  let results = (outs Variadic<Tribute_Type>:$results);
  
  let builders = [
    OpBuilder<(ins
      "StringRef":$callee,
      "ArrayRef<Value>":$operands
    )>
  ];
  
  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, $results)
  }];
}

def Tribute_LambdaOp : Tribute_Op<"lambda", [Pure]> {
  let summary = "Create an anonymous function";
  let description = [{
    Creates a closure capturing the specified values.
    
    Example:
      %closure = tribute.lambda [%x, %y] (%a) -> (%r) {
        %sum = tribute.add %a, %x : !tribute.value
        tribute.yield %sum : !tribute.value
      } : !tribute.function<(!tribute.value) -> (!tribute.value)>
  }];
  
  let arguments = (ins Variadic<Tribute_Type>:$captures);
  let results = (outs Tribute_FunctionType:$result);
  let regions = (region SizedRegion<1>:$body);
  
  let assemblyFormat = [{
    `[` $captures `]` $body attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Pattern Matching
//===----------------------------------------------------------------------===//

def Tribute_MatchOp : Tribute_Op<"match", [
    DeclareOpInterfaceMethods<RegionBranchOpInterface>,
    SingleBlock, NoTerminator
  ]> {
  let summary = "Pattern matching operation";
  let description = [{
    Matches a value against multiple patterns.
    
    Example:
      %result = tribute.match %value : !tribute.value -> !tribute.value {
      ^bb0(%arg: !tribute.value):
        tribute.case number(%n: f64) {
          tribute.yield %n : f64
        }
        tribute.case string(%s: !tribute.string) {
          %len = tribute.string_length %s : !tribute.string
          tribute.yield %len : index
        }
        tribute.default {
          %nil = tribute.constant nil : !tribute.value
          tribute.yield %nil : !tribute.value
        }
      }
  }];
  
  let arguments = (ins Tribute_Type:$value);
  let results = (outs Variadic<Tribute_Type>:$results);
  let regions = (region SizedRegion<1>:$body);
  
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Control Flow
//===----------------------------------------------------------------------===//

def Tribute_IfOp : Tribute_Op<"if", [
    DeclareOpInterfaceMethods<RegionBranchOpInterface>
  ]> {
  let summary = "Conditional execution";
  let description = [{
    Executes one of two regions based on a condition.
    
    Example:
      %result = tribute.if %cond -> !tribute.value {
        %true_val = tribute.constant 1.0 : f64
        tribute.yield %true_val : f64
      } else {
        %false_val = tribute.constant 0.0 : f64
        tribute.yield %false_val : f64
      }
  }];
  
  let arguments = (ins I1:$condition);
  let results = (outs Variadic<Tribute_Type>:$results);
  let regions = (region SizedRegion<1>:$then, AnyRegion:$else);
  
  let assemblyFormat = [{
    $condition `->` type($results) $then `else` $else attr-dict
  }];
}

def Tribute_WhileOp : Tribute_Op<"while", [
    DeclareOpInterfaceMethods<LoopLikeOpInterface>
  ]> {
  let summary = "While loop";
  let description = [{
    Executes a loop while a condition is true.
    
    Example:
      %result = tribute.while %init : !tribute.value -> !tribute.value {
      ^bb0(%arg: !tribute.value):
        %cond = tribute.cmp lt %arg, %limit : !tribute.value
        tribute.condition %cond : i1
      } do {
      ^bb0(%arg: !tribute.value):
        %next = tribute.add %arg, %step : !tribute.value
        tribute.yield %next : !tribute.value
      }
  }];
  
  let arguments = (ins Variadic<Tribute_Type>:$inits);
  let results = (outs Variadic<Tribute_Type>:$results);
  let regions = (region SizedRegion<1>:$before, SizedRegion<1>:$after);
  
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Terminator Operations
//===----------------------------------------------------------------------===//

def Tribute_ReturnOp : Tribute_Op<"return", [
    Pure, HasParent<"FuncOp">, Terminator
  ]> {
  let summary = "Return from function";
  let description = [{
    Returns values from a Tribute function.
    
    Example:
      tribute.return %result : !tribute.value
  }];
  
  let arguments = (ins Variadic<Tribute_Type>:$operands);
  
  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";
}

def Tribute_YieldOp : Tribute_Op<"yield", [Pure, Terminator]> {
  let summary = "Yield values from a region";
  let description = [{
    Yields values from a region to its parent operation.
    
    Example:
      tribute.yield %value : !tribute.value
  }];
  
  let arguments = (ins Variadic<Tribute_Type>:$operands);
  
  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";
}

//===----------------------------------------------------------------------===//
// Runtime Interface Operations
//===----------------------------------------------------------------------===//

def Tribute_PrintOp : Tribute_Op<"print"> {
  let summary = "Print a value";
  let description = [{
    Prints a value to standard output.
    
    Example:
      tribute.print %value : !tribute.value
  }];
  
  let arguments = (ins Tribute_Type:$value);
  
  let assemblyFormat = "$value attr-dict `:` type($value)";
}

def Tribute_ToRuntimeOp : Tribute_Op<"to_runtime", [Pure]> {
  let summary = "Convert to runtime representation";
  let description = [{
    Converts a statically typed value to the dynamic runtime representation.
    This operation is always safe and never fails.
    
    Use this when:
    - Calling functions that expect !tribute.value
    - Storing values in heterogeneous containers
    - Interfacing with dynamic evaluation
    
    Example:
      %runtime_val = tribute.to_runtime %typed_val : f64 -> !tribute.value
  }];
  
  let arguments = (ins AnyType:$input);
  let results = (outs Tribute_ValueType:$output);
  
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

def Tribute_FromRuntimeOp : Tribute_Op<"from_runtime", [Pure]> {
  let summary = "Convert from runtime representation";
  let description = [{
    Converts a dynamic runtime value to a specific typed representation.
    This operation may fail at runtime if the value is not of the expected type.
    
    Use this when:
    - You know the expected type from context
    - Optimizing hot paths with known types
    - Interfacing with typed operations
    
    Runtime behavior:
    - Success: Returns the typed value
    - Failure: Throws runtime type error
    
    Example:
      %typed_val = tribute.from_runtime %runtime_val : !tribute.value -> f64
  }];
  
  let arguments = (ins Tribute_ValueType:$input);
  let results = (outs AnyType:$output);
  
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

#endif // TRIBUTE_OPS
```

## C++ Implementation

### TributeDialect.cpp

```cpp
//===----------------------------------------------------------------------===//
// TributeDialect.cpp - Tribute Dialect Implementation
//===----------------------------------------------------------------------===//

#include "Tribute/TributeDialect.h"
#include "Tribute/TributeOps.h"
#include "Tribute/TributeTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::tribute;

//===----------------------------------------------------------------------===//
// TableGen generated definitions
//===----------------------------------------------------------------------===//

#include "Tribute/TributeDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Tribute Dialect
//===----------------------------------------------------------------------===//

void TributeDialect::initialize() {
  // Register operations
  addOperations<
#define GET_OP_LIST
#include "Tribute/TributeOps.cpp.inc"
  >();
  
  // Register types
  addTypes<
#define GET_TYPEDEF_LIST
#include "Tribute/TributeTypes.cpp.inc"
  >();
  
  // Register attributes
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Tribute/TributeAttrs.cpp.inc"
  >();
  
  // Register interfaces
  addInterfaces<TributeInlinerInterface>();
}

//===----------------------------------------------------------------------===//
// Type Parsing and Printing
//===----------------------------------------------------------------------===//

Type TributeDialect::parseType(DialectAsmParser &parser) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();
  
  Type result;
  OptionalParseResult parseResult = 
    generatedTypeParser(parser, keyword, result);
  if (parseResult.has_value())
    return result;
  
  parser.emitError(parser.getNameLoc(), "unknown tribute type: ") << keyword;
  return Type();
}

void TributeDialect::printType(Type type, DialectAsmPrinter &os) const {
  if (failed(generatedTypePrinter(type, os))) {
    llvm_unreachable("unexpected 'tribute' type");
  }
}

//===----------------------------------------------------------------------===//
// Attribute Parsing and Printing
//===----------------------------------------------------------------------===//

Attribute TributeDialect::parseAttribute(DialectAsmParser &parser,
                                         Type type) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Attribute();
  
  Attribute result;
  OptionalParseResult parseResult = 
    generatedAttributeParser(parser, keyword, type, result);
  if (parseResult.has_value())
    return result;
  
  parser.emitError(parser.getNameLoc(), "unknown tribute attribute: ") 
    << keyword;
  return Attribute();
}

void TributeDialect::printAttribute(Attribute attr,
                                   DialectAsmPrinter &os) const {
  if (failed(generatedAttributePrinter(attr, os))) {
    llvm_unreachable("unexpected 'tribute' attribute");
  }
}

//===----------------------------------------------------------------------===//
// Tribute Inliner Interface
//===----------------------------------------------------------------------===//

struct TributeInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  
  /// All Tribute operations can be inlined
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }
  
  /// All Tribute operations can be inlined into regions
  bool isLegalToInline(Operation *, Region *, bool,
                       BlockAndValueMapping &) const final {
    return true;
  }
  
  /// Handle terminators during inlining
  void handleTerminator(Operation *op,
                       ArrayRef<Value> valuesToRepl) const final {
    // Handle tribute.return
    if (auto returnOp = dyn_cast<tribute::ReturnOp>(op)) {
      for (const auto &it : llvm::enumerate(returnOp.getOperands()))
        valuesToRepl[it.index()].replaceAllUsesWith(it.value());
      return;
    }
    
    // Handle tribute.yield
    if (auto yieldOp = dyn_cast<tribute::YieldOp>(op)) {
      for (const auto &it : llvm::enumerate(yieldOp.getOperands()))
        valuesToRepl[it.index()].replaceAllUsesWith(it.value());
      return;
    }
  }
};
```

## Rust Integration with melior

### Cargo.toml

```toml
[dependencies]
melior = { version = "0.19", features = ["ods-dialects"] }
tribute-ast = { path = "../tribute-ast" }
tribute-runtime = { path = "../tribute-runtime" }
thiserror = "1.0"
inkwell = "0.5"  # For LLVM integration

[build-dependencies]
bindgen = "0.69"
cmake = "0.1"
```

### Custom Dialect Integration with FFI

Since melior doesn't directly support custom dialects, we need to create FFI bindings to our C++ Tribute dialect implementation:

```rust
// build.rs
use std::env;
use std::path::PathBuf;

fn main() {
    // Build the C++ Tribute dialect library
    let dst = cmake::Config::new("mlir")
        .define("CMAKE_BUILD_TYPE", "Release")
        .build();
    
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=TributeDialect");
    println!("cargo:rustc-link-lib=MLIR-C");
    
    // Generate bindings for our custom dialect
    let bindings = bindgen::Builder::default()
        .header("mlir/include/Tribute/TributeDialect-c.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("tribute_dialect_bindings.rs"))
        .expect("Couldn't write bindings!");
}
```

### MLIR Builder Module

```rust
// src/mlir_builder.rs

use melior::{
    Context, Module, Block, Region, Location, Operation,
    dialect::{func, arith, cf, scf, llvm},
    ir::{Type, Value, Attribute, TypeLike},
};
use std::collections::HashMap;

// Include generated FFI bindings
include!(concat!(env!("OUT_DIR"), "/tribute_dialect_bindings.rs"));

/// Helper struct for building MLIR with Tribute dialect
pub struct TributeMLIRBuilder<'c> {
    context: &'c Context,
    module: Module<'c>,
    /// Symbol table for resolving names
    symbol_table: HashMap<String, Value<'c, '_>>,
    /// Type cache
    type_cache: HashMap<String, Type<'c>>,
}

impl<'c> TributeMLIRBuilder<'c> {
    pub fn new(context: &'c Context) -> Self {
        // Register standard dialects
        context.append_dialect_registry(&{
            let registry = melior::dialect::DialectRegistry::new();
            registry.insert_dialect::<func::FuncDialect>();
            registry.insert_dialect::<arith::ArithDialect>();
            registry.insert_dialect::<cf::CfDialect>();
            registry.insert_dialect::<scf::ScfDialect>();
            registry.insert_dialect::<llvm::LLVMDialect>();
            registry
        });
        
        // Register our custom Tribute dialect via FFI
        unsafe {
            tribute_dialect_register(context.to_raw());
        }
        
        context.load_all_available_dialects();
        
        let module = Module::new(Location::unknown(context));
        
        Self {
            context,
            module,
            symbol_table: HashMap::new(),
            type_cache: HashMap::new(),
        }
    }
    
    /// Get Tribute value type
    pub fn get_value_type(&self) -> Type<'c> {
        unsafe {
            let raw_type = tribute_value_type_get(self.context.to_raw());
            Type::from_raw(raw_type)
        }
    }
    
    /// Get Tribute string type
    pub fn get_string_type(&self) -> Type<'c> {
        unsafe {
            let raw_type = tribute_string_type_get(self.context.to_raw());
            Type::from_raw(raw_type)
        }
    }
    
    /// Get Tribute list type
    pub fn get_list_type(&self, element_type: Type<'c>) -> Type<'c> {
        unsafe {
            let raw_type = tribute_list_type_get(
                self.context.to_raw(), 
                element_type.to_raw()
            );
            Type::from_raw(raw_type)
        }
    }
    
    /// Create a Tribute constant
    pub fn create_constant(
        &self,
        value: TributeConstant,
        location: Location<'c>,
    ) -> Operation<'c> {
        unsafe {
            match value {
                TributeConstant::Number(n) => {
                    let raw_op = tribute_constant_op_create_number(
                        self.context.to_raw(),
                        n,
                        location.to_raw()
                    );
                    Operation::from_raw(raw_op)
                }
                TributeConstant::String(s) => {
                    let c_string = std::ffi::CString::new(s).unwrap();
                    let raw_op = tribute_constant_op_create_string(
                        self.context.to_raw(),
                        c_string.as_ptr(),
                        location.to_raw()
                    );
                    Operation::from_raw(raw_op)
                }
                TributeConstant::Boolean(b) => {
                    let raw_op = tribute_constant_op_create_bool(
                        self.context.to_raw(),
                        b,
                        location.to_raw()
                    );
                    Operation::from_raw(raw_op)
                }
                TributeConstant::Nil => {
                    let raw_op = tribute_constant_op_create_nil(
                        self.context.to_raw(),
                        location.to_raw()
                    );
                    Operation::from_raw(raw_op)
                }
            }
        }
    }
    
    /// Create a Tribute function
    pub fn create_function(
        &mut self,
        name: &str,
        params: Vec<(&str, Type<'c>)>,
        return_type: Option<Type<'c>>,
        body_builder: impl FnOnce(&mut Self, &Block<'c>),
    ) -> Operation<'c> {
        unsafe {
            let c_name = std::ffi::CString::new(name).unwrap();
            
            // Convert parameters
            let param_types: Vec<_> = params.iter()
                .map(|(_, ty)| ty.to_raw())
                .collect();
            
            let return_types = match return_type {
                Some(ty) => vec![ty.to_raw()],
                None => vec![],
            };
            
            let raw_op = tribute_func_op_create(
                self.context.to_raw(),
                c_name.as_ptr(),
                param_types.as_ptr(),
                param_types.len(),
                return_types.as_ptr(),
                return_types.len()
            );
            
            let operation = Operation::from_raw(raw_op);
            
            // Build function body
            let entry_block = operation.region(0).unwrap().block(0).unwrap();
            body_builder(self, &entry_block);
            
            operation
        }
    }
    
    /// Create binary operation
    pub fn create_binary_op(
        &self,
        op_type: TributeBinaryOpType,
        lhs: Value<'c, '_>,
        rhs: Value<'c, '_>,
        location: Location<'c>,
    ) -> Operation<'c> {
        unsafe {
            let raw_op = match op_type {
                TributeBinaryOpType::Add => tribute_add_op_create(
                    self.context.to_raw(),
                    lhs.to_raw(),
                    rhs.to_raw(),
                    location.to_raw()
                ),
                TributeBinaryOpType::Sub => tribute_sub_op_create(
                    self.context.to_raw(),
                    lhs.to_raw(),
                    rhs.to_raw(),
                    location.to_raw()
                ),
                TributeBinaryOpType::Mul => tribute_mul_op_create(
                    self.context.to_raw(),
                    lhs.to_raw(),
                    rhs.to_raw(),
                    location.to_raw()
                ),
                TributeBinaryOpType::Div => tribute_div_op_create(
                    self.context.to_raw(),
                    lhs.to_raw(),
                    rhs.to_raw(),
                    location.to_raw()
                ),
            };
            Operation::from_raw(raw_op)
        }
    }
}

/// Tribute constant values
pub enum TributeConstant {
    Number(f64),
    String(String),
    Boolean(bool),
    Nil,
}

/// Tribute binary operation types
pub enum TributeBinaryOpType {
    Add,
    Sub,
    Mul,
    Div,
}
```

### C API Header for Tribute Dialect

We need to create a C API header for our dialect:

```c
// mlir/include/Tribute/TributeDialect-c.h

#ifndef TRIBUTE_DIALECT_C_H
#define TRIBUTE_DIALECT_C_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Register the Tribute dialect with the given context
void tribute_dialect_register(MlirContext context);

/// Type creation functions
MlirType tribute_value_type_get(MlirContext context);
MlirType tribute_string_type_get(MlirContext context);
MlirType tribute_list_type_get(MlirContext context, MlirType elementType);
MlirType tribute_function_type_get(MlirContext context, 
                                   MlirType *inputTypes, size_t numInputs,
                                   MlirType *resultTypes, size_t numResults);

/// Operation creation functions
MlirOperation tribute_constant_op_create_number(MlirContext context, 
                                                double value, 
                                                MlirLocation location);
MlirOperation tribute_constant_op_create_string(MlirContext context, 
                                                const char* value, 
                                                MlirLocation location);
MlirOperation tribute_constant_op_create_bool(MlirContext context, 
                                              bool value, 
                                              MlirLocation location);
MlirOperation tribute_constant_op_create_nil(MlirContext context, 
                                             MlirLocation location);

MlirOperation tribute_func_op_create(MlirContext context,
                                     const char* name,
                                     MlirType *paramTypes, size_t numParams,
                                     MlirType *resultTypes, size_t numResults);

MlirOperation tribute_add_op_create(MlirContext context,
                                    MlirValue lhs, MlirValue rhs,
                                    MlirLocation location);
MlirOperation tribute_sub_op_create(MlirContext context,
                                    MlirValue lhs, MlirValue rhs,
                                    MlirLocation location);
MlirOperation tribute_mul_op_create(MlirContext context,
                                    MlirValue lhs, MlirValue rhs,
                                    MlirLocation location);
MlirOperation tribute_div_op_create(MlirContext context,
                                    MlirValue lhs, MlirValue rhs,
                                    MlirLocation location);

#ifdef __cplusplus
}
#endif

#endif // TRIBUTE_DIALECT_C_H
```

### AST to Tribute Dialect Lowering

```rust
// src/ast_lowering.rs

use tribute_ast::{Expr, Item, Program};
use crate::mlir_builder::{TributeMLIRBuilder, TributeConstant};
use melior::{Context, Location, ir::Block};

pub struct ASTLowerer<'c> {
    builder: TributeMLIRBuilder<'c>,
}

impl<'c> ASTLowerer<'c> {
    pub fn new(context: &'c Context) -> Self {
        Self {
            builder: TributeMLIRBuilder::new(context),
        }
    }
    
    pub fn lower_program(&mut self, program: &Program) -> Result<(), LoweringError> {
        for item in &program.items {
            self.lower_item(item)?;
        }
        Ok(())
    }
    
    fn lower_item(&mut self, item: &Item) -> Result<(), LoweringError> {
        match item {
            Item::Function { name, params, body, .. } => {
                // Get parameter types (all dynamic for now)
                let param_types: Vec<_> = params.iter()
                    .map(|p| (p.as_str(), self.builder.get_value_type()))
                    .collect();
                
                // Create function
                self.builder.create_function(
                    name,
                    param_types,
                    Some(self.builder.get_value_type()),
                    |builder, entry_block| {
                        // Lower function body
                        self.lower_expr_in_block(body, builder, entry_block);
                    },
                );
            }
            Item::Let { pattern, value, .. } => {
                // Global variable initialization
                todo!("Global variables")
            }
        }
        Ok(())
    }
    
    fn lower_expr_in_block(
        &mut self,
        expr: &Expr,
        builder: &mut TributeMLIRBuilder<'c>,
        block: &Block<'c>,
    ) -> melior::ir::Value<'c, '_> {
        let loc = Location::unknown(builder.context);
        
        match expr {
            Expr::Number(n) => {
                let const_op = builder.create_constant(
                    TributeConstant::Number(*n),
                    loc,
                );
                block.append_operation(const_op);
                block.last_operation().unwrap().result(0).unwrap()
            }
            Expr::String(s) => {
                let const_op = builder.create_constant(
                    TributeConstant::String(s.clone()),
                    loc,
                );
                block.append_operation(const_op);
                block.last_operation().unwrap().result(0).unwrap()
            }
            Expr::Identifier(name) => {
                // Variable lookup
                builder.symbol_table.get(name).cloned()
                    .expect("Undefined variable")
            }
            Expr::Binary { op, left, right } => {
                let lhs = self.lower_expr_in_block(left, builder, block);
                let rhs = self.lower_expr_in_block(right, builder, block);
                
                // Create binary operation
                let op_name = match op.as_str() {
                    "+" => "add",
                    "-" => "sub",
                    "*" => "mul",
                    "/" => "div",
                    _ => panic!("Unknown operator"),
                };
                
                // Create tribute.add, tribute.sub, etc.
                todo!("FFI to tribute binary ops")
            }
            Expr::Call { func, args } => {
                // Lower arguments
                let arg_values: Vec<_> = args.iter()
                    .map(|arg| self.lower_expr_in_block(arg, builder, block))
                    .collect();
                
                // Create tribute.call
                todo!("FFI to tribute::CallOp")
            }
            Expr::If { condition, then_expr, else_expr } => {
                // Create tribute.if
                todo!("FFI to tribute::IfOp")
            }
            Expr::Let { bindings, body } => {
                // Create local bindings
                for (pattern, value) in bindings {
                    let val = self.lower_expr_in_block(value, builder, block);
                    // Pattern matching would go here
                    if let Some(name) = pattern.as_identifier() {
                        builder.symbol_table.insert(name.to_string(), val);
                    }
                }
                
                // Lower body
                self.lower_expr_in_block(body, builder, block)
            }
            _ => todo!("Lower expression: {:?}", expr),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LoweringError {
    #[error("Undefined variable: {0}")]
    UndefinedVariable(String),
    
    #[error("Type mismatch")]
    TypeMismatch,
    
    #[error("Unsupported feature: {0}")]
    UnsupportedFeature(String),
}
```

## Progressive Lowering Pipeline

### Dialect Lowering

```rust
// src/dialect_lowering.rs

use melior::{
    Context, Module, Operation,
    pass::{PassManager, transform},
};

/// Progressive lowering from Tribute dialect to LLVM
pub struct TributeDialectLowering {
    pass_manager: PassManager<'static>,
}

impl TributeDialectLowering {
    pub fn new(context: &Context) -> Self {
        let mut pm = PassManager::new(context);
        
        // Phase 1: Tribute to standard dialects
        pm.add_pass(create_tribute_to_standard_pass());
        pm.add_pass(transform::create_canonicalizer());
        
        // Phase 2: Standard dialects optimization
        pm.add_pass(transform::create_cse());
        pm.add_pass(transform::create_loop_invariant_code_motion());
        
        // Phase 3: Lower to LLVM
        pm.add_pass(create_standard_to_llvm_pass());
        pm.add_pass(create_func_to_llvm_pass());
        
        Self { pass_manager: pm }
    }
    
    pub fn run(&mut self, module: &mut Module) -> Result<(), String> {
        self.pass_manager.run(module)
            .map_err(|e| format!("Pass manager failed: {:?}", e))
    }
}

/// Custom pass to lower Tribute dialect to standard dialects
fn create_tribute_to_standard_pass() -> Pass {
    // This would be implemented in C++ and exposed via FFI
    todo!("FFI to TributeToStandardPass")
}

/// Example lowering patterns (would be in C++):
/// 
/// tribute.constant -> arith.constant (for numbers)
/// tribute.constant -> llvm.mlir.constant (for strings)
/// tribute.add -> arith.addf (after type checking)
/// tribute.string_interpolation -> series of llvm calls
/// tribute.list_create -> llvm.alloca + llvm.store
/// tribute.func -> func.func
/// tribute.call -> func.call
```

### C++ Lowering Pass Example

```cpp
//===----------------------------------------------------------------------===//
// TributeToStandard.cpp - Lower Tribute dialect to standard dialects
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace {

/// Convert tribute.constant to appropriate standard constant
struct TributeConstantLowering : public OpConversionPattern<tribute::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;
  
  LogicalResult matchAndRewrite(
      tribute::ConstantOp op,
      OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    auto attr = op.getValue();
    auto resultType = op.getResult().getType();
    
    // Handle different constant types
    if (auto floatAttr = attr.dyn_cast<FloatAttr>()) {
      // Lower to arith.constant
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(
          op, resultType, floatAttr);
      return success();
    }
    
    if (auto stringAttr = attr.dyn_cast<StringAttr>()) {
      // Lower to LLVM global string
      auto llvmType = typeConverter->convertType(resultType);
      // Create global string and return pointer
      // ...
      return success();
    }
    
    return failure();
  }
};

/// Convert tribute.add to appropriate arithmetic operation
struct TributeAddLowering : public OpConversionPattern<tribute::AddOp> {
  using OpConversionPattern::OpConversionPattern;
  
  LogicalResult matchAndRewrite(
      tribute::AddOp op,
      OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    // For now, assume numeric addition
    // In full implementation, would check runtime types
    rewriter.replaceOpWithNewOp<arith::AddFOp>(
        op, adaptor.getLhs(), adaptor.getRhs());
    
    return success();
  }
};

/// Main conversion pass
struct TributeToStandardPass
    : public PassWrapper<TributeToStandardPass, OperationPass<ModuleOp>> {
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<LLVM::LLVMDialect>();
  }
  
  void runOnOperation() override {
    ConversionTarget target(getContext());
    
    // Tribute dialect is illegal
    target.addIllegalDialect<tribute::TributeDialect>();
    
    // Standard dialects are legal
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<LLVM::LLVMDialect>();
    
    // Set up type converter
    TributeTypeConverter typeConverter;
    
    // Populate patterns
    RewritePatternSet patterns(&getContext());
    patterns.add<TributeConstantLowering>(typeConverter, &getContext());
    patterns.add<TributeAddLowering>(typeConverter, &getContext());
    // ... more patterns
    
    // Apply conversion
    if (failed(applyPartialConversion(getOperation(), target,
                                     std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createTributeToStandardPass() {
  return std::make_unique<TributeToStandardPass>();
}
```

## Runtime Integration

### Runtime Bindings

```rust
// src/runtime_integration.rs

use melior::{Context, Module};
use tribute_runtime::*;

/// Declare runtime functions in MLIR module
pub fn declare_runtime_functions(module: &Module) {
    // These would be LLVM function declarations that link to tribute-runtime
    
    // Value creation
    declare_external_func(module, "tribute_value_from_f64", 
        "(f64) -> i64");  // Returns handle
    declare_external_func(module, "tribute_value_from_string",
        "(!llvm.ptr, i64) -> i64");  // ptr, len -> handle
    
    // Value operations
    declare_external_func(module, "tribute_value_add",
        "(i64, i64) -> i64");  // handle, handle -> handle
    
    // String operations
    declare_external_func(module, "tribute_string_create",
        "(!llvm.ptr, i64) -> i64");  // ptr, len -> handle
    declare_external_func(module, "tribute_string_concat",
        "(i64, i64) -> i64");  // handle, handle -> handle
    
    // List operations
    declare_external_func(module, "tribute_list_create",
        "(i64) -> i64");  // capacity -> handle
    declare_external_func(module, "tribute_list_push",
        "(i64, i64) -> ()");  // list_handle, value_handle
    
    // I/O operations
    declare_external_func(module, "tribute_print",
        "(i64) -> ()");  // handle
}

fn declare_external_func(module: &Module, name: &str, signature: &str) {
    // Parse signature and create LLVM function declaration
    todo!("Create LLVM function declaration")
}
```

## Memory Management and Safety

### Handle-Based Memory Model

The Tribute MLIR dialect uses a handle-based approach for memory management, similar to the runtime library design:

```rust
// src/memory_management.rs

/// Memory management strategy for Tribute MLIR integration
pub struct TributeMemoryManager {
    /// Global handle table for runtime values
    handle_table: HandleTable,
    /// Garbage collector integration
    gc: GarbageCollector,
}

impl TributeMemoryManager {
    pub fn new() -> Self {
        Self {
            handle_table: HandleTable::new(),
            gc: GarbageCollector::new(),
        }
    }
    
    /// Register a new runtime value and return its handle
    pub fn register_value(&mut self, value: RuntimeValue) -> Handle {
        let handle = self.handle_table.allocate();
        self.handle_table.store(handle, value);
        self.gc.track(handle);
        handle
    }
    
    /// Get a value by handle (may trigger GC)
    pub fn get_value(&mut self, handle: Handle) -> Option<&RuntimeValue> {
        self.gc.maybe_collect();
        self.handle_table.get(handle)
    }
}
```

### FFI Safety Considerations

**1. Handle Lifecycle Management**
```rust
/// RAII wrapper for handle cleanup
pub struct SafeHandle {
    handle: Handle,
    manager: Arc<Mutex<TributeMemoryManager>>,
}

impl Drop for SafeHandle {
    fn drop(&mut self) {
        // Automatically decrement reference count
        if let Ok(mut manager) = self.manager.lock() {
            manager.handle_table.decref(self.handle);
        }
    }
}
```

**2. Cross-FFI Boundary Safety**
```rust
/// Safe wrapper for FFI calls
#[no_mangle]
pub unsafe extern "C" fn tribute_value_add_safe(
    lhs: Handle, 
    rhs: Handle
) -> Handle {
    // Validate handles before use
    if !is_valid_handle(lhs) || !is_valid_handle(rhs) {
        return INVALID_HANDLE;
    }
    
    // Perform operation with exception safety
    match std::panic::catch_unwind(|| {
        tribute_value_add_impl(lhs, rhs)
    }) {
        Ok(result) => result,
        Err(_) => INVALID_HANDLE,
    }
}
```

**3. Garbage Collection Integration**
```rust
/// GC-aware MLIR lowering
impl TributeDialectLowering {
    fn emit_gc_safepoint(&mut self, builder: &OpBuilder) {
        // Insert GC safepoint calls at function boundaries
        let safepoint = builder.create::<llvm::CallOp>(
            location,
            "tribute_gc_safepoint",
            vec![],
            vec![]
        );
        safepoint.set_attribute("gc.safepoint", true);
    }
    
    fn emit_handle_cleanup(&mut self, builder: &OpBuilder, handles: Vec<Handle>) {
        // Emit cleanup code for local handles
        for handle in handles {
            builder.create::<llvm::CallOp>(
                location,
                "tribute_handle_decref",
                vec![handle],
                vec![]
            );
        }
    }
}
```

### Memory Layout Considerations

**Runtime Value Representation:**
```c
// C representation for LLVM integration
typedef struct {
    uint32_t type_tag;    // Value type discriminator
    uint32_t flags;       // GC and mutation flags
    union {
        double number;
        struct {
            char* data;
            size_t len;
        } string;
        struct {
            Handle* elements;
            size_t len;
            size_t capacity;
        } list;
        Handle function;
    } data;
} TributeValue;
```

**Handle Table Structure:**
```rust
/// Efficient handle table implementation
pub struct HandleTable {
    values: Vec<Option<RuntimeValue>>,
    free_list: Vec<Handle>,
    generation: Vec<u32>,  // ABA prevention
}

impl HandleTable {
    pub fn is_valid(&self, handle: Handle) -> bool {
        let index = handle.index() as usize;
        index < self.values.len() 
            && self.values[index].is_some()
            && self.generation[index] == handle.generation()
    }
}
```

### Error Handling in Memory Operations

**1. Out-of-Memory Handling**
```rust
impl TributeMemoryManager {
    pub fn try_allocate(&mut self, size: usize) -> Result<Handle, MemoryError> {
        // Check available memory before allocation
        if self.available_memory() < size {
            self.gc.force_collect()?;
            if self.available_memory() < size {
                return Err(MemoryError::OutOfMemory);
            }
        }
        
        self.allocate_impl(size)
    }
}
```

**2. Memory Leak Detection**
```rust
#[cfg(debug_assertions)]
impl TributeMemoryManager {
    pub fn check_leaks(&self) -> Vec<Handle> {
        self.handle_table.values.iter()
            .enumerate()
            .filter_map(|(i, v)| {
                if v.is_some() && self.gc.is_unreachable(i as Handle) {
                    Some(i as Handle)
                } else {
                    None
                }
            })
            .collect()
    }
}
```

## Error Handling and Diagnostics

### MLIR Diagnostic Integration

MLIR provides a comprehensive diagnostic system that integrates well with compiler pipelines:

```rust
// src/error_handling.rs

use melior::{Context, Operation, diagnostic::{Diagnostic, DiagnosticEngine}};
use std::sync::Arc;

/// Tribute-specific error types for MLIR compilation
#[derive(Debug, thiserror::Error)]
pub enum TributeMLIRError {
    #[error("Type conversion failed: {message}")]
    TypeConversion { message: String, location: Option<String> },
    
    #[error("Runtime call failed: {function_name}")]
    RuntimeCall { function_name: String },
    
    #[error("MLIR verification failed: {details}")]
    Verification { details: String },
    
    #[error("Memory allocation failed: {size} bytes")]
    Memory { size: usize },
    
    #[error("Handle validation failed: {handle:?}")]
    InvalidHandle { handle: u64 },
}

/// Diagnostic handler that converts MLIR diagnostics to Tribute errors
pub struct TributeDiagnosticHandler {
    errors: Vec<TributeMLIRError>,
    warnings: Vec<String>,
}

impl TributeDiagnosticHandler {
    pub fn new() -> Self {
        Self {
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }
    
    /// Register this handler with MLIR context
    pub fn register_with_context(&mut self, context: &Context) {
        context.set_diagnostic_handler(Box::new(move |diagnostic| {
            self.handle_diagnostic(diagnostic);
        }));
    }
    
    fn handle_diagnostic(&mut self, diagnostic: &Diagnostic) {
        match diagnostic.severity() {
            melior::diagnostic::Severity::Error => {
                self.errors.push(TributeMLIRError::Verification {
                    details: diagnostic.message().to_string()
                });
            }
            melior::diagnostic::Severity::Warning => {
                self.warnings.push(diagnostic.message().to_string());
            }
            _ => {} // Note, info, remark
        }
    }
    
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }
    
    pub fn into_result(self) -> Result<Vec<String>, Vec<TributeMLIRError>> {
        if self.has_errors() {
            Err(self.errors)
        } else {
            Ok(self.warnings)
        }
    }
}
```

### Compilation Error Recovery

When MLIR compilation fails, provide fallback strategies:

```rust
// src/compilation_fallback.rs

pub struct TributeCompiler {
    mlir_enabled: bool,
    fallback_interpreter: Arc<HirEvaluator>,
}

impl TributeCompiler {
    pub fn compile_with_fallback(
        &self, 
        program: &HirProgram
    ) -> Result<CompiledProgram, CompilationError> {
        if self.mlir_enabled {
            match self.try_mlir_compilation(program) {
                Ok(compiled) => {
                    log::info!("MLIR compilation successful");
                    return Ok(compiled);
                }
                Err(mlir_error) => {
                    log::warn!("MLIR compilation failed: {}, falling back to interpreter", mlir_error);
                    // Continue to fallback
                }
            }
        }
        
        // Fallback to interpreter mode
        self.compile_for_interpreter(program)
    }
    
    fn try_mlir_compilation(&self, program: &HirProgram) -> Result<CompiledProgram, TributeMLIRError> {
        let context = Context::new();
        let mut diagnostic_handler = TributeDiagnosticHandler::new();
        diagnostic_handler.register_with_context(&context);
        
        // Attempt MLIR lowering
        let mut builder = TributeMLIRBuilder::new(&context);
        let lowerer = HirToMLIRLowerer::new(&mut builder);
        
        let module = lowerer.lower_program(program)?;
        
        // Verify the module
        if !module.verify() {
            return Err(TributeMLIRError::Verification {
                details: "Module verification failed".to_string()
            });
        }
        
        // Check for compilation errors
        diagnostic_handler.into_result()
            .map_err(|errors| errors.into_iter().next().unwrap())?;
        
        // Apply optimization passes
        self.apply_optimization_passes(&module)?;
        
        // Generate executable code
        self.generate_executable(&module)
    }
}
```

### Runtime Error Propagation

Handle runtime errors that can occur during execution:

```rust
// src/runtime_errors.rs

/// Runtime errors that can occur during MLIR-compiled code execution
#[derive(Debug, thiserror::Error)]
pub enum TributeRuntimeError {
    #[error("Type mismatch: expected {expected}, got {actual}")]
    TypeMismatch { expected: String, actual: String },
    
    #[error("Division by zero")]
    DivisionByZero,
    
    #[error("Index out of bounds: {index} not in 0..{length}")]
    IndexOutOfBounds { index: i64, length: usize },
    
    #[error("Function not found: {name}")]
    FunctionNotFound { name: String },
    
    #[error("Stack overflow")]
    StackOverflow,
    
    #[error("Out of memory")]
    OutOfMemory,
}

/// Convert runtime errors to MLIR exception handling
impl TributeRuntimeError {
    pub fn to_mlir_exception(&self) -> MLIRException {
        match self {
            Self::TypeMismatch { .. } => MLIRException::new("type_error", &self.to_string()),
            Self::DivisionByZero => MLIRException::new("arithmetic_error", &self.to_string()),
            Self::IndexOutOfBounds { .. } => MLIRException::new("index_error", &self.to_string()),
            Self::FunctionNotFound { .. } => MLIRException::new("name_error", &self.to_string()),
            Self::StackOverflow => MLIRException::new("stack_error", &self.to_string()),
            Self::OutOfMemory => MLIRException::new("memory_error", &self.to_string()),
        }
    }
}

/// Runtime error handling in generated MLIR code
pub fn emit_error_handling_wrapper(
    builder: &OpBuilder,
    operation: Operation,
    error_handler: &str,
) -> Operation {
    // Wrap operations in try-catch equivalent
    // Uses MLIR exception handling or error codes
    
    // For now, use simple error code approach
    let error_code_type = IntegerType::new(builder.context(), 32);
    
    // Check operation result
    let check_op = builder.create_operation(
        "tribute.check_error",
        &[operation.result(0)],
        &[error_code_type],
        &[]
    );
    
    // Branch on error code
    let cond = builder.create_comparison(
        ComparisonPredicate::Ne,
        check_op.result(0),
        builder.create_constant(0, error_code_type)
    );
    
    builder.create_conditional_branch(
        cond,
        error_handler,  // Error block
        "continue"      // Success block
    )
}
```

### Integration with Tribute's Diagnostic System

Connect MLIR diagnostics with Tribute's existing error reporting:

```rust
// src/diagnostic_integration.rs

use tribute_ast::diagnostic::{Diagnostic as TributeDiagnostic, DiagnosticLevel};

/// Convert MLIR diagnostics to Tribute's diagnostic format
pub fn convert_mlir_diagnostic(
    mlir_diag: &melior::diagnostic::Diagnostic
) -> TributeDiagnostic {
    let level = match mlir_diag.severity() {
        melior::diagnostic::Severity::Error => DiagnosticLevel::Error,
        melior::diagnostic::Severity::Warning => DiagnosticLevel::Warning,
        melior::diagnostic::Severity::Note => DiagnosticLevel::Note,
        _ => DiagnosticLevel::Info,
    };
    
    TributeDiagnostic::new(
        level,
        mlir_diag.message(),
        mlir_diag.location().map(|loc| convert_mlir_location(loc))
    )
}

fn convert_mlir_location(mlir_loc: MLIRLocation) -> tribute_ast::SourceLocation {
    // Extract file, line, column from MLIR location
    tribute_ast::SourceLocation {
        file: mlir_loc.file_name().unwrap_or("unknown"),
        line: mlir_loc.line().unwrap_or(0),
        column: mlir_loc.column().unwrap_or(0),
    }
}

/// Aggregate all compilation diagnostics
pub struct CompilationDiagnostics {
    pub parse_diagnostics: Vec<TributeDiagnostic>,
    pub hir_diagnostics: Vec<TributeDiagnostic>,
    pub mlir_diagnostics: Vec<TributeDiagnostic>,
}

impl CompilationDiagnostics {
    pub fn has_errors(&self) -> bool {
        self.parse_diagnostics.iter().any(|d| d.is_error()) ||
        self.hir_diagnostics.iter().any(|d| d.is_error()) ||
        self.mlir_diagnostics.iter().any(|d| d.is_error())
    }
    
    pub fn display_all(&self) {
        for diag in &self.parse_diagnostics {
            println!("{}", diag);
        }
        for diag in &self.hir_diagnostics {
            println!("{}", diag);
        }
        for diag in &self.mlir_diagnostics {
            println!("{}", diag);
        }
    }
}
```

## Debugging and Profiling

### MLIR Debugging Tools Integration

MLIR provides powerful debugging and profiling capabilities that can be leveraged for Tribute development:

```rust
// src/debugging.rs

use melior::{Context, Module, Operation, pass::{PassManager, PassInstrumentation}};

/// Debugging configuration for Tribute MLIR compilation
pub struct TributeDebugConfig {
    pub dump_ir: bool,
    pub verify_each_pass: bool,
    pub time_passes: bool,
    pub debug_counter: Option<String>,
}

impl Default for TributeDebugConfig {
    fn default() -> Self {
        Self {
            dump_ir: cfg!(debug_assertions),
            verify_each_pass: cfg!(debug_assertions),
            time_passes: false,
            debug_counter: None,
        }
    }
}

/// Set up debugging infrastructure for MLIR compilation
pub fn setup_mlir_debugging(
    context: &Context,
    config: &TributeDebugConfig,
) -> PassManager {
    let mut pm = PassManager::new(context);
    
    if config.dump_ir {
        // Enable IR dumping before and after each pass
        pm.enable_ir_printing(|pass, operation| {
            eprintln!("=== IR before pass '{}' ===", pass.name());
            eprintln!("{}", operation.to_string());
        });
    }
    
    if config.verify_each_pass {
        // Enable verification after each pass
        pm.enable_verifier(true);
    }
    
    if config.time_passes {
        // Enable pass timing
        pm.enable_timing();
    }
    
    if let Some(counter) = &config.debug_counter {
        // Enable debug counter for selective debugging
        pm.enable_debug_counter(counter);
    }
    
    pm
}

/// Debug information generation for Tribute operations
pub fn emit_debug_info(
    builder: &TributeMLIRBuilder,
    operation: &Operation,
    source_info: &tribute_ast::SourceLocation,
) {
    // Create debug location from source information
    let debug_loc = builder.context.create_file_line_col_location(
        &source_info.file,
        source_info.line as u32,
        source_info.column as u32
    );
    
    // Attach debug location to operation
    operation.set_location(debug_loc);
    
    // Add debug attributes for runtime inspection
    operation.set_attribute(
        "tribute.source_file", 
        StringAttribute::new(builder.context, &source_info.file)
    );
    operation.set_attribute(
        "tribute.source_line",
        IntegerAttribute::new(builder.context, source_info.line as i64)
    );
}
```

### Runtime Profiling Integration

```rust
// src/profiling.rs

use std::time::{Duration, Instant};
use std::collections::HashMap;

/// Performance profiler for Tribute runtime operations
pub struct TributeProfiler {
    enabled: bool,
    operation_times: HashMap<String, Vec<Duration>>,
    memory_usage: HashMap<String, usize>,
    current_operation: Option<(String, Instant)>,
}

impl TributeProfiler {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            operation_times: HashMap::new(),
            memory_usage: HashMap::new(),
            current_operation: None,
        }
    }
    
    /// Start profiling an operation
    pub fn start_operation(&mut self, operation_name: &str) {
        if !self.enabled {
            return;
        }
        
        self.current_operation = Some((operation_name.to_string(), Instant::now()));
    }
    
    /// End profiling current operation
    pub fn end_operation(&mut self) {
        if !self.enabled {
            return;
        }
        
        if let Some((name, start_time)) = self.current_operation.take() {
            let duration = start_time.elapsed();
            self.operation_times.entry(name).or_default().push(duration);
        }
    }
    
    /// Record memory usage for an operation
    pub fn record_memory_usage(&mut self, operation_name: &str, bytes: usize) {
        if !self.enabled {
            return;
        }
        
        self.memory_usage.insert(operation_name.to_string(), bytes);
    }
    
    /// Generate profiling report
    pub fn generate_report(&self) -> ProfilingReport {
        ProfilingReport {
            operation_stats: self.operation_times.iter()
                .map(|(name, times)| {
                    let total: Duration = times.iter().sum();
                    let avg = total / times.len() as u32;
                    (name.clone(), OperationStats {
                        total_time: total,
                        average_time: avg,
                        call_count: times.len(),
                    })
                })
                .collect(),
            memory_stats: self.memory_usage.clone(),
        }
    }
}

#[derive(Debug)]
pub struct ProfilingReport {
    pub operation_stats: HashMap<String, OperationStats>,
    pub memory_stats: HashMap<String, usize>,
}

#[derive(Debug)]
pub struct OperationStats {
    pub total_time: Duration,
    pub average_time: Duration,
    pub call_count: usize,
}

impl ProfilingReport {
    pub fn print_summary(&self) {
        println!("=== Tribute Runtime Profiling Report ===");
        println!();
        
        println!("Operation Performance:");
        let mut ops: Vec<_> = self.operation_stats.iter().collect();
        ops.sort_by_key(|(_, stats)| std::cmp::Reverse(stats.total_time));
        
        for (name, stats) in ops {
            println!("  {}: {} calls, {:?} total, {:?} avg", 
                name, stats.call_count, stats.total_time, stats.average_time);
        }
        
        println!();
        println!("Memory Usage:");
        for (name, bytes) in &self.memory_stats {
            println!("  {}: {} bytes", name, bytes);
        }
    }
}
```

### MLIR Pass Debugging

```rust
// src/pass_debugging.rs

/// Debug utilities for MLIR pass development
pub struct PassDebugUtils;

impl PassDebugUtils {
    /// Dump module IR to file for inspection
    pub fn dump_module_to_file(module: &Module, filename: &str) {
        use std::fs::File;
        use std::io::Write;
        
        let mut file = File::create(filename)
            .expect("Failed to create debug file");
        writeln!(file, "{}", module.as_operation().to_string())
            .expect("Failed to write to debug file");
    }
    
    /// Verify module and print detailed errors
    pub fn verify_module_detailed(module: &Module) -> bool {
        if module.verify() {
            println!("✅ Module verification passed");
            true
        } else {
            println!("❌ Module verification failed");
            // MLIR will have already printed diagnostic information
            false
        }
    }
    
    /// Count operations by type in module
    pub fn count_operations(module: &Module) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        
        module.body().walk_operations(|op| {
            let op_name = op.name().to_string();
            *counts.entry(op_name).or_insert(0) += 1;
        });
        
        counts
    }
    
    /// Print operation statistics
    pub fn print_operation_stats(module: &Module) {
        let counts = Self::count_operations(module);
        
        println!("=== Operation Statistics ===");
        for (op_name, count) in counts {
            println!("  {}: {} instances", op_name, count);
        }
    }
}
```

### Integration with IDE Debugging

```rust
// src/ide_integration.rs

/// Generate debug information compatible with IDE debuggers
pub struct TributeDebugInfo {
    source_map: HashMap<String, tribute_ast::SourceLocation>,
    mlir_locations: HashMap<String, String>,
}

impl TributeDebugInfo {
    pub fn new() -> Self {
        Self {
            source_map: HashMap::new(),
            mlir_locations: HashMap::new(),
        }
    }
    
    /// Map MLIR operation to source location
    pub fn add_source_mapping(
        &mut self,
        mlir_op_id: String,
        source_loc: tribute_ast::SourceLocation,
    ) {
        self.source_map.insert(mlir_op_id.clone(), source_loc);
    }
    
    /// Generate source map file for debugging
    pub fn generate_source_map(&self) -> serde_json::Value {
        serde_json::json!({
            "version": 3,
            "sources": self.source_map.values()
                .map(|loc| &loc.file)
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect::<Vec<_>>(),
            "mappings": self.generate_mappings(),
            "names": []
        })
    }
    
    fn generate_mappings(&self) -> String {
        // Simplified source map generation
        // In a real implementation, this would generate proper VLQ-encoded mappings
        self.source_map.iter()
            .map(|(mlir_id, source_loc)| {
                format!("{}:{}:{}", mlir_id, source_loc.line, source_loc.column)
            })
            .collect::<Vec<_>>()
            .join(",")
    }
    
    /// Export debug info for external tools
    pub fn export_debug_info(&self, filename: &str) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::Write;
        
        let debug_info = serde_json::json!({
            "tribute_debug_info": {
                "version": "1.0",
                "source_map": self.source_map,
                "mlir_locations": self.mlir_locations
            }
        });
        
        let mut file = File::create(filename)?;
        writeln!(file, "{}", serde_json::to_string_pretty(&debug_info)?)?;
        
        Ok(())
    }
}
```

### Performance Benchmarking

```rust
// src/benchmarking.rs

use criterion::{Criterion, BenchmarkId};
use std::time::Duration;

/// Benchmark Tribute MLIR compilation pipeline
pub fn benchmark_compilation_pipeline() {
    let mut criterion = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(100);
    
    let test_programs = load_test_programs();
    
    for (name, program) in test_programs {
        criterion.bench_with_input(
            BenchmarkId::new("mlir_compilation", &name),
            &program,
            |b, program| {
                b.iter(|| {
                    let context = Context::new();
                    let mut builder = TributeMLIRBuilder::new(&context);
                    let lowerer = HirToMLIRLowerer::new(&mut builder);
                    lowerer.lower_program(program).unwrap()
                });
            }
        );
        
        criterion.bench_with_input(
            BenchmarkId::new("hir_evaluation", &name),
            &program,
            |b, program| {
                b.iter(|| {
                    let db = TributeDatabaseImpl::default();
                    tribute::eval_str(&db, "benchmark.trb", &program.source)
                });
            }
        );
    }
}

fn load_test_programs() -> Vec<(String, HirProgram)> {
    // Load various test programs of different complexity
    vec![
        ("simple_arithmetic", parse_program("1 + 2 * 3")),
        ("function_call", parse_program("fn add(a, b) { a + b } add(1, 2)")),
        ("string_interpolation", parse_program(r#"let name = "world"; "Hello, \{name}!""#)),
        // Add more test cases...
    ]
}
```

## Testing Strategy

### MLIR Test Files

```mlir
// tests/tribute_ops.mlir

// RUN: tribute-opt %s | FileCheck %s

// CHECK-LABEL: @test_constant
tribute.func @test_constant() -> !tribute.value {
  // CHECK: tribute.constant 42.0
  %0 = tribute.constant 42.0 : f64
  %1 = tribute.to_runtime %0 : f64 -> !tribute.value
  tribute.return %1 : !tribute.value
}

// CHECK-LABEL: @test_arithmetic
tribute.func @test_arithmetic(%arg0: !tribute.value, %arg1: !tribute.value) 
    -> !tribute.value {
  // CHECK: tribute.add
  %sum = tribute.add %arg0, %arg1 : !tribute.value
  // CHECK: tribute.mul
  %product = tribute.mul %sum, %arg0 : !tribute.value
  tribute.return %product : !tribute.value
}

// CHECK-LABEL: @test_string_interpolation
tribute.func @test_string_interpolation(%name: !tribute.value, %count: !tribute.value) 
    -> !tribute.string {
  // CHECK: tribute.string_interpolation
  // Tribute source: "Hello, \{name}! You have \{count} messages."
  %result = tribute.string_interpolation "Hello, {0}! You have {1} messages." 
            (%name, %count) : (!tribute.value, !tribute.value) -> !tribute.string
  tribute.return %result : !tribute.string
}

// CHECK-LABEL: @test_list_operations
tribute.func @test_list_operations() -> !tribute.value {
  %0 = tribute.constant 1.0 : f64
  %1 = tribute.constant 2.0 : f64
  %2 = tribute.constant 3.0 : f64
  
  // CHECK: tribute.list_create
  %list = tribute.list_create [%0, %1, %2] : !tribute.list<f64>
  
  %idx = arith.constant 1 : index
  // CHECK: tribute.list_get
  %elem = tribute.list_get %list[%idx] : !tribute.list<f64>
  
  %val = tribute.to_runtime %elem : f64 -> !tribute.value
  tribute.return %val : !tribute.value
}

// CHECK-LABEL: @test_control_flow
tribute.func @test_control_flow(%cond: i1) -> !tribute.value {
  // CHECK: tribute.if
  %result = tribute.if %cond -> !tribute.value {
    %true_val = tribute.constant 1.0 : f64
    %true_result = tribute.to_runtime %true_val : f64 -> !tribute.value
    tribute.yield %true_result : !tribute.value
  } else {
    %false_val = tribute.constant 0.0 : f64
    %false_result = tribute.to_runtime %false_val : f64 -> !tribute.value
    tribute.yield %false_result : !tribute.value
  }
  tribute.return %result : !tribute.value
}
```

### Integration Tests

```rust
// tests/integration_test.rs

#[test]
fn test_simple_function_lowering() {
    let context = Context::new();
    let mut lowerer = ASTLowerer::new(&context);
    
    // Parse Tribute code
    let ast = parse_tribute_code(r#"
        fn add(a, b) {
            a + b
        }
    "#);
    
    // Lower to Tribute dialect
    lowerer.lower_program(&ast).unwrap();
    
    // Get MLIR module
    let module = lowerer.builder.module;
    
    // Verify it produces expected MLIR
    let mlir_text = module.as_operation().to_string();
    assert!(mlir_text.contains("tribute.func @add"));
    assert!(mlir_text.contains("tribute.add"));
}

#[test]
fn test_lowering_pipeline() {
    let context = Context::new();
    let mut lowerer = ASTLowerer::new(&context);
    
    // Create simple program
    let ast = parse_tribute_code(r#"
        fn main() {
            print_line("Hello, World!")
        }
    "#);
    
    // Lower to Tribute dialect
    lowerer.lower_program(&ast).unwrap();
    
    // Apply lowering passes
    let mut pipeline = TributeDialectLowering::new(&context);
    pipeline.run(&mut lowerer.builder.module).unwrap();
    
    // Verify it lowered to LLVM
    let mlir_text = lowerer.builder.module.as_operation().to_string();
    assert!(mlir_text.contains("llvm.func"));
}
```

## Performance Considerations

### Compilation Performance

MLIR compilation introduces overhead compared to direct interpretation, but provides significant runtime benefits:

```rust
// src/performance_analysis.rs

/// Performance metrics for different execution modes
pub struct PerformanceMetrics {
    pub compilation_time: Duration,
    pub execution_time: Duration,
    pub memory_usage: usize,
    pub binary_size: usize,
}

/// Compare performance between interpreter and MLIR compilation
pub fn benchmark_execution_modes(program: &HirProgram) -> ComparisonReport {
    let interpreter_metrics = benchmark_interpreter(program);
    let mlir_metrics = benchmark_mlir_compilation(program);
    
    ComparisonReport {
        interpreter: interpreter_metrics,
        mlir_compiled: mlir_metrics,
        speedup_factor: mlir_metrics.execution_time.as_nanos() as f64 /
                       interpreter_metrics.execution_time.as_nanos() as f64,
    }
}

/// Performance optimizations for Tribute dialect operations
pub struct TributeOptimizations;

impl TributeOptimizations {
    /// Optimize dynamic type checks by caching type information
    pub fn optimize_type_checks(module: &Module) {
        // Convert runtime type checks to compile-time when possible
        // Use MLIR's constant folding and dead code elimination
    }
    
    /// Optimize string interpolation by pre-computing static parts
    pub fn optimize_string_interpolation(module: &Module) {
        // Identify constant string parts that can be pre-computed
        // Use MLIR's string constant optimization passes
    }
    
    /// Optimize function calls by inlining small functions
    pub fn optimize_function_calls(module: &Module) {
        // Apply MLIR's function inlining pass
        // Specialize functions for common argument patterns
    }
}
```

### Runtime Performance Trade-offs

**Dynamic vs Static Optimization:**

1. **Fully Dynamic Approach**: Simple but slower
   - All operations go through runtime type checking
   - Flexibility for duck typing and dynamic features
   - Higher memory overhead due to handle-based values

2. **Hybrid Approach**: Balanced performance and flexibility
   - Static types where possible, dynamic fallback
   - Type specialization for hot code paths
   - Gradual optimization based on profiling data

3. **Static Optimization**: Fastest but limited
   - Requires type annotations or inference
   - May not support all dynamic features
   - Best for performance-critical applications

### Memory Performance

```rust
/// Memory optimization strategies for Tribute runtime
pub struct MemoryOptimizer {
    gc_threshold: usize,
    handle_pool_size: usize,
}

impl MemoryOptimizer {
    /// Optimize handle allocation patterns
    pub fn optimize_handle_allocation(&self, module: &Module) {
        // Pre-allocate handle pools for common operations
        // Use object pooling for frequently created/destroyed values
        // Implement handle recycling to reduce allocation pressure
    }
    
    /// Optimize string storage and sharing
    pub fn optimize_string_storage(&self, module: &Module) {
        // Use string interning for literal strings
        // Implement copy-on-write for string modifications
        // Share immutable string data between values
    }
    
    /// Optimize list and array operations
    pub fn optimize_collections(&self, module: &Module) {
        // Use specialized storage for homogeneous lists
        // Implement growth strategies to minimize reallocations
        // Cache array bounds checks for safe indexing
    }
}
```

### Compilation Time Optimization

```rust
/// Strategies to reduce MLIR compilation time
pub struct CompilationOptimizer;

impl CompilationOptimizer {
    /// Cache compiled modules to avoid recompilation
    pub fn enable_module_caching(cache_dir: &Path) -> Result<ModuleCache, io::Error> {
        ModuleCache::new(cache_dir)
    }
    
    /// Parallel compilation for independent modules
    pub fn compile_modules_parallel(modules: Vec<Module>) -> Vec<CompiledModule> {
        modules.into_par_iter()
            .map(|module| Self::compile_single_module(module))
            .collect()
    }
    
    /// Incremental compilation for development
    pub fn enable_incremental_compilation(
        &self,
        previous_compilation: &CompilationResult
    ) -> IncrementalCompiler {
        IncrementalCompiler::new(previous_compilation)
    }
}
```

## Migration Path from Current HIR

### Phase 1: Parallel Implementation

The migration should be gradual to minimize risk and maintain development velocity:

```rust
// src/migration_strategy.rs

/// Multi-phase migration from HIR to MLIR
pub struct MigrationStrategy {
    current_phase: MigrationPhase,
    fallback_enabled: bool,
}

#[derive(Debug)]
pub enum MigrationPhase {
    /// Phase 1: MLIR backend alongside existing HIR
    ParallelImplementation,
    /// Phase 2: MLIR as primary, HIR as fallback
    MLIRPrimary,
    /// Phase 3: HIR removal, MLIR only
    MLIROnly,
}

impl MigrationStrategy {
    /// Start migration with parallel implementation
    pub fn phase_1_parallel_implementation() -> Self {
        // Implement MLIR backend without removing HIR
        // Add compilation flag to choose backend
        // Comprehensive testing of both backends
        Self {
            current_phase: MigrationPhase::ParallelImplementation,
            fallback_enabled: true,
        }
    }
    
    /// Move to MLIR-primary phase
    pub fn phase_2_mlir_primary(self) -> Self {
        // Make MLIR the default compilation target
        // Keep HIR for complex cases and fallback
        // Gradually expand MLIR coverage
        Self {
            current_phase: MigrationPhase::MLIRPrimary,
            fallback_enabled: true,
        }
    }
    
    /// Final phase: MLIR only
    pub fn phase_3_mlir_only(self) -> Self {
        // Remove HIR evaluation completely
        // MLIR handles all compilation
        // Simplify codebase
        Self {
            current_phase: MigrationPhase::MLIROnly,
            fallback_enabled: false,
        }
    }
}
```

### Implementation Roadmap

**Phase 1: Foundation (1-2 months)**
```rust
/// Tasks for Phase 1
pub struct Phase1Tasks {
    // Infrastructure setup
    setup_mlir_build_system: bool,
    implement_basic_dialect: bool,
    create_ffi_bindings: bool,
    
    // Basic operations
    implement_arithmetic_ops: bool,
    implement_function_calls: bool,
    implement_constants: bool,
    
    // Testing
    create_mlir_test_suite: bool,
    benchmark_against_hir: bool,
}
```

**Phase 2: Feature Parity (2-3 months)**
```rust
/// Tasks for Phase 2
pub struct Phase2Tasks {
    // Advanced features
    implement_string_interpolation: bool,
    implement_pattern_matching: bool,
    implement_control_flow: bool,
    
    // Optimization
    implement_basic_passes: bool,
    optimize_type_conversions: bool,
    implement_inlining: bool,
    
    // Integration
    integrate_with_runtime: bool,
    handle_error_propagation: bool,
    support_debugging: bool,
}
```

**Phase 3: Production Ready (1-2 months)**
```rust
/// Tasks for Phase 3
pub struct Phase3Tasks {
    // Performance
    optimize_compilation_speed: bool,
    implement_caching: bool,
    profile_and_optimize: bool,
    
    // Stability
    extensive_testing: bool,
    handle_edge_cases: bool,
    documentation_complete: bool,
    
    // Migration
    remove_hir_evaluation: bool,
    simplify_codebase: bool,
    update_tooling: bool,
}
```

### Compatibility Strategy

```rust
/// Ensure compatibility during migration
pub struct CompatibilityManager {
    version: String,
    supported_features: HashSet<TributeFeature>,
}

impl CompatibilityManager {
    /// Check if a program can be compiled with MLIR
    pub fn can_compile_with_mlir(&self, program: &HirProgram) -> bool {
        let required_features = self.analyze_required_features(program);
        required_features.iter().all(|feature| {
            self.supported_features.contains(feature)
        })
    }
    
    /// Provide migration recommendations
    pub fn migration_recommendations(&self, program: &HirProgram) -> Vec<MigrationRecommendation> {
        let unsupported = self.find_unsupported_features(program);
        unsupported.into_iter()
            .map(|feature| self.recommend_alternative(feature))
            .collect()
    }
}

#[derive(Debug)]
pub enum MigrationRecommendation {
    /// Feature can be refactored to supported equivalent
    Refactor { from: TributeFeature, to: TributeFeature },
    /// Feature requires waiting for MLIR implementation
    WaitForImplementation { feature: TributeFeature, eta: String },
    /// Feature may need redesign
    Redesign { feature: TributeFeature, reason: String },
}
```

### Testing Strategy for Migration

```rust
/// Comprehensive testing during migration
pub struct MigrationTesting {
    compatibility_tests: Vec<CompatibilityTest>,
    performance_benchmarks: Vec<PerformanceBenchmark>,
    regression_tests: Vec<RegressionTest>,
}

impl MigrationTesting {
    /// Test that MLIR and HIR produce identical results
    pub fn test_result_equivalence(&self, test_cases: &[TestCase]) -> TestResults {
        test_cases.iter()
            .map(|test| {
                let hir_result = self.run_with_hir(test);
                let mlir_result = self.run_with_mlir(test);
                TestResult {
                    test_name: test.name.clone(),
                    hir_result,
                    mlir_result,
                    equivalent: hir_result == mlir_result,
                }
            })
            .collect()
    }
    
    /// Ensure no performance regressions
    pub fn benchmark_performance_impact(&self) -> PerformanceReport {
        // Compare compilation and execution times
        // Measure memory usage changes
        // Assess binary size impact
        PerformanceReport::new()
    }
}
```

## Summary

This architecture leverages MLIR's infrastructure by implementing Tribute as a proper MLIR dialect rather than maintaining a separate HIR. The benefits include:

1. **Unified representation**: No translation between Rust HIR and MLIR
2. **Reusable infrastructure**: MLIR's passes, analyses, and tools work out of the box
3. **Progressive lowering**: Natural path from high-level Tribute operations to LLVM
4. **Better debugging**: MLIR's visualization and debugging tools
5. **Extensibility**: Easy to add new operations and transformations

The implementation requires:
- TableGen definitions for the Tribute dialect
- C++ implementation of dialect operations
- FFI bindings to use the dialect from Rust (melior)
- Lowering passes from Tribute dialect to standard dialects
- Runtime library integration for dynamic features

This approach aligns with MLIR's design philosophy and provides a solid foundation for implementing advanced compiler optimizations and transformations.