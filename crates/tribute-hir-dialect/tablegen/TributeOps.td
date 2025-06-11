//===- TributeOps.td - Tribute dialect operations ------*- tablegen -*-===//
//
// This file defines the operations for the Tribute MLIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TRIBUTE_OPS
#define TRIBUTE_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

//===----------------------------------------------------------------------===//
// Tribute dialect definition
//===----------------------------------------------------------------------===//

def Tribute_Dialect : Dialect {
  let name = "tribute";
  let summary = "The Tribute programming language dialect";
  let description = [{
    This dialect contains operations for the Tribute programming language,
    a dynamic Lisp-like language with modern syntax.
  }];
  let cppNamespace = "::mlir::tribute";
}

//===----------------------------------------------------------------------===//
// Tribute types
//===----------------------------------------------------------------------===//

// Base class for Tribute types
class Tribute_Type<string name, string typeMnemonic> : TypeDef<Tribute_Dialect, name> {
  let mnemonic = typeMnemonic;
}

// The dynamic value type - represents any Tribute value at runtime
def Tribute_ValueType : Tribute_Type<"Value", "value"> {
  let summary = "Tribute dynamic value type";
  let description = [{
    The value type represents any Tribute value at runtime.
    This includes numbers, strings, functions, and other dynamic values.
  }];
}

//===----------------------------------------------------------------------===//
// Base operation classes
//===----------------------------------------------------------------------===//

// Base class for all Tribute operations
class Tribute_Op<string mnemonic, list<Trait> traits = []> :
  Op<Tribute_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Constant operations
//===----------------------------------------------------------------------===//

def Tribute_ConstantOp : Tribute_Op<"constant", [ConstantLike, Pure]> {
  let summary = "Tribute constant operation";
  let description = [{
    The constant operation produces a constant value of a given type.
    This is used for literal values in the source code.
  }];

  let arguments = (ins AnyAttr:$value);
  let results = (outs Tribute_ValueType:$result);

  let assemblyFormat = "attr-dict $value `:` type($result)";

  let hasFolder = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Type conversion operations
//===----------------------------------------------------------------------===//

def Tribute_ToRuntimeOp : Tribute_Op<"to_runtime", [Pure]> {
  let summary = "Convert static type to runtime type";
  let description = [{
    Converts a statically-typed value to the dynamic runtime value type.
    This is used when interfacing between statically-typed operations
    (like MLIR arithmetic) and the dynamic Tribute runtime.
  }];

  let arguments = (ins AnyType:$input);
  let results = (outs Tribute_ValueType:$result);

  let assemblyFormat = "attr-dict $input `:` type($input) `to` type($result)";

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Arithmetic operations
//===----------------------------------------------------------------------===//

def Tribute_AddOp : Tribute_Op<"add", [Pure, Commutative]> {
  let summary = "Tribute addition operation";
  let description = [{
    Performs addition on two Tribute values.
    This handles both numeric addition and string concatenation.
  }];

  let arguments = (ins Tribute_ValueType:$lhs, Tribute_ValueType:$rhs);
  let results = (outs Tribute_ValueType:$result);

  let assemblyFormat = "attr-dict $lhs `,` $rhs `:` type($result)";

  let hasVerifier = 1;
}

def Tribute_SubOp : Tribute_Op<"sub", [Pure]> {
  let summary = "Tribute subtraction operation";
  let description = [{
    Performs subtraction on two Tribute numeric values.
  }];

  let arguments = (ins Tribute_ValueType:$lhs, Tribute_ValueType:$rhs);
  let results = (outs Tribute_ValueType:$result);

  let assemblyFormat = "attr-dict $lhs `,` $rhs `:` type($result)";

  let hasVerifier = 1;
}

def Tribute_MulOp : Tribute_Op<"mul", [Pure, Commutative]> {
  let summary = "Tribute multiplication operation";
  let description = [{
    Performs multiplication on two Tribute numeric values.
  }];

  let arguments = (ins Tribute_ValueType:$lhs, Tribute_ValueType:$rhs);
  let results = (outs Tribute_ValueType:$result);

  let assemblyFormat = "attr-dict $lhs `,` $rhs `:` type($result)";

  let hasVerifier = 1;
}

def Tribute_DivOp : Tribute_Op<"div", [Pure]> {
  let summary = "Tribute division operation";
  let description = [{
    Performs division on two Tribute numeric values.
  }];

  let arguments = (ins Tribute_ValueType:$lhs, Tribute_ValueType:$rhs);
  let results = (outs Tribute_ValueType:$result);

  let assemblyFormat = "attr-dict $lhs `,` $rhs `:` type($result)";

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Function operations
//===----------------------------------------------------------------------===//

def Tribute_FuncOp : Tribute_Op<"func", [
  FunctionOpInterface, IsolatedFromAbove, Symbol
]> {
  let summary = "Tribute function operation";
  let description = [{
    Defines a function in the Tribute language.
    Functions take dynamic values as arguments and return a dynamic value.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<StrAttr>:$sym_visibility);
  let regions = (region AnyRegion:$body);

  let assemblyFormat = "attr-dict $sym_name $function_type $body";

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    /// Returns the function type of this function.
    FunctionType getFunctionType() {
      return getAttr("function_type").cast<TypeAttr>().getValue()
          .cast<FunctionType>();
    }
  }];

  let hasVerifier = 1;
}

def Tribute_CallOp : Tribute_Op<"call", [CallOpInterface]> {
  let summary = "Tribute function call operation";
  let description = [{
    Calls a function with the given arguments and returns the result.
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee,
                       Variadic<Tribute_ValueType>:$operands);
  let results = (outs Tribute_ValueType:$result);

  let assemblyFormat = "attr-dict $callee `(` $operands `)` `:` type($result)";

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // CallOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Get the argument operands to the called function.
    operand_range getArgOperands() { return getOperands(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }
  }];

  let hasVerifier = 1;
}

def Tribute_ReturnOp : Tribute_Op<"return", [Pure, HasParent<"Tribute_FuncOp">,
                                             ReturnLike, Terminator]> {
  let summary = "Tribute return operation";
  let description = [{
    Returns a value from a function.
  }];

  let arguments = (ins Optional<Tribute_ValueType>:$operand);

  let assemblyFormat = "attr-dict ($operand^ `:` type($operand))?";

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// String operations
//===----------------------------------------------------------------------===//

def Tribute_StringConcatOp : Tribute_Op<"string_concat", [Pure]> {
  let summary = "String concatenation operation";
  let description = [{
    Concatenates two string values together.
  }];

  let arguments = (ins Tribute_ValueType:$lhs, Tribute_ValueType:$rhs);
  let results = (outs Tribute_ValueType:$result);

  let assemblyFormat = "attr-dict $lhs `,` $rhs `:` type($result)";

  let hasVerifier = 1;
}

def Tribute_StringInterpolationOp : Tribute_Op<"string_interpolation", [Pure]> {
  let summary = "String interpolation operation";
  let description = [{
    Performs string interpolation with embedded expressions.
  }];

  let arguments = (ins Variadic<Tribute_ValueType>:$parts);
  let results = (outs Tribute_ValueType:$result);

  let assemblyFormat = "attr-dict `(` $parts `)` `:` type($result)";

  let hasVerifier = 1;
}

#endif // TRIBUTE_OPS