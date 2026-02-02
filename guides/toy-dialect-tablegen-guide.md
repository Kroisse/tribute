# TableGen을 사용한 Toy Dialect 정의 가이드

## 목차

1. [개요](#개요)
2. [프로젝트 구조](#프로젝트-구조)
3. [Dialect 정의](#dialect-정의)
4. [타입 정의](#타입-정의)
5. [Operation 정의](#operation-정의)
6. [C++ 통합](#c-통합)
7. [빌드 설정](#빌드-설정)
8. [사용 예제](#사용-예제)

## 개요

이 문서는 MLIR의 TableGen을 사용하여 간단한 수치 계산 언어인 "Toy"
dialect를 정의하는 방법을 설명합니다. Toy 언어는 텐서 기반 연산, 함수
정의, 그리고 간단한 제어 흐름을 지원합니다.

### Toy 언어 예제

```toy
# 사용자 정의 함수
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  # 2x3 행렬 정의
  var a = [[1, 2, 3], [4, 5, 6]];
  var b = [[1, 2, 3], [4, 5, 6]];
  
  # 함수 호출
  var c = multiply_transpose(a, b);
  
  # 결과 출력
  print(c);
}
```

## 프로젝트 구조

```text
toy-dialect/
├── include/
│   └── Toy/
│       ├── ToyDialect.h
│       ├── ToyDialect.td      # Dialect 정의
│       ├── ToyOps.h
│       ├── ToyOps.td          # Operations 정의
│       ├── ToyTypes.h
│       └── ToyTypes.td        # Types 정의
├── lib/
│   └── Toy/
│       ├── ToyDialect.cpp
│       ├── ToyOps.cpp
│       └── ToyTypes.cpp
└── CMakeLists.txt
```

## Dialect 정의

### ToyDialect.td

```tablegen
//===----------------------------------------------------------------------===//
// Toy Dialect 정의
//===----------------------------------------------------------------------===//

#ifndef TOY_DIALECT
#define TOY_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Toy Dialect
//===----------------------------------------------------------------------===//

def Toy_Dialect : Dialect {
  let name = "toy";
  let summary = "고수준 텐서 연산을 위한 Toy MLIR dialect";
  let description = [{
    Toy dialect는 텐서 기반 수치 계산을 표현하기 위한 고수준 dialect입니다.
    이 dialect는 다음을 포함합니다:
    - 텐서 타입과 연산
    - 함수 정의와 호출
    - 기본적인 수학 연산
    - 출력을 위한 print 연산
    
    이 dialect는 교육 목적으로 설계되었으며, 실제 ML 프레임워크의
    단순화된 버전을 제공합니다.
  }];
  
  let cppNamespace = "::mlir::toy";
  
  // Dialect가 사용하는 타입을 정의할 수 있음을 명시
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  
  // 추가 verification 로직을 위한 C++ 코드
  let extraClassDeclaration = [{
    /// RegisterTypes 후크 - 커스텀 타입 등록
    void registerTypes();
    
    /// Dialect 초기화 시 호출되는 함수
    static void initialize();
  }];
  
  // Dialect가 의존하는 다른 dialect들
  let dependentDialects = [
    "arith::ArithDialect",
    "func::FuncDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Toy Dialect Base Classes
//===----------------------------------------------------------------------===//

// Toy dialect의 모든 op들이 상속받을 base class
class Toy_Op<string mnemonic, list<Trait> traits = []> :
    Op<Toy_Dialect, mnemonic, traits>;

// Type constraint을 위한 helper
def Toy_Type : DialectType<Toy_Dialect,
    CPred<"$_self.isa<::mlir::toy::ToyType>()">,
    "Toy dialect 타입">;

// Tensor 타입을 위한 특별한 constraint
def Toy_TensorType : DialectType<Toy_Dialect,
    CPred<"$_self.isa<::mlir::toy::TensorType>()">,
    "Toy tensor 타입">;

#endif // TOY_DIALECT
```

## 타입 정의

### ToyTypes.td

```tablegen
//===----------------------------------------------------------------------===//
// Toy Types
//===----------------------------------------------------------------------===//

#ifndef TOY_TYPES
#define TOY_TYPES

include "mlir/IR/AttrTypeBase.td"
include "ToyDialect.td"

//===----------------------------------------------------------------------===//
// Tensor Type
//===----------------------------------------------------------------------===//

def Toy_TensorType : TypeDef<Toy_Dialect, "Tensor"> {
  let mnemonic = "tensor";
  
  let summary = "Toy tensor 타입";
  let description = [{
    다차원 배열을 나타내는 텐서 타입입니다.
    shape와 element type을 가집니다.
    
    예제:
      !toy.tensor<2x3xf64>    // 2x3 double 텐서
      !toy.tensor<*xf64>      // 동적 shape의 텐서
      !toy.tensor<10x?xf32>   // 부분적으로 동적인 shape
  }];
  
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType
  );
  
  let builders = [
    TypeBuilderWithInferredContext<(ins
      "ArrayRef<int64_t>":$shape,
      "Type":$elementType
    ), [{
      return $_get(elementType.getContext(), shape, elementType);
    }]>,
    TypeBuilder<(ins
      "ArrayRef<int64_t>":$shape,
      "Type":$elementType
    ), [{
      return $_get($_ctxt, shape, elementType);
    }]>
  ];
  
  let extraClassDeclaration = [{
    /// shape의 rank(차원 수) 반환
    unsigned getRank() const { return getShape().size(); }
    
    /// 특정 차원의 크기 반환
    int64_t getDimSize(unsigned index) const {
      assert(index < getRank() && "invalid index");
      return getShape()[index];
    }
    
    /// 모든 차원이 정적으로 알려져 있는지 확인
    bool hasStaticShape() const {
      return llvm::none_of(getShape(), ShapedType::isDynamic);
    }
    
    /// 전체 원소 개수 반환 (정적 shape인 경우)
    int64_t getNumElements() const;
    
    /// 동적 차원이 있는지 확인
    bool isDynamicDim(unsigned index) const {
      return ShapedType::isDynamic(getShape()[index]);
    }
  }];
  
  let genVerifyDecl = 1;
  
  // 커스텀 assembly format
  let assemblyFormat = "`<` custom<Shape>($shape, $elementType) `>`";
}

//===----------------------------------------------------------------------===//
// Struct Type
//===----------------------------------------------------------------------===//

def Toy_StructType : TypeDef<Toy_Dialect, "Struct"> {
  let mnemonic = "struct";
  
  let summary = "Toy struct 타입";
  let description = [{
    여러 필드를 가진 구조체 타입입니다.
    
    예제:
      !toy.struct<{x: f64, y: f64, label: !toy.tensor<*xf32>}>
  }];
  
  let parameters = (ins
    ArrayRefParameter<"Type">:$elementTypes,
    ArrayRefParameter<"StringRef">:$elementNames
  );
  
  let extraClassDeclaration = [{
    /// 필드 개수 반환
    size_t getNumFields() const { return getElementTypes().size(); }
    
    /// 이름으로 필드 타입 찾기
    Type getFieldType(StringRef name) const;
    
    /// 인덱스로 필드 이름 가져오기
    StringRef getFieldName(unsigned index) const {
      assert(index < getNumFields());
      return getElementNames()[index];
    }
  }];
  
  let genVerifyDecl = 1;
}

#endif // TOY_TYPES
```

## Operation 정의

### ToyOps.td

```tablegen
//===----------------------------------------------------------------------===//
// Toy Operations
//===----------------------------------------------------------------------===//

#ifndef TOY_OPS
#define TOY_OPS

include "ToyDialect.td"
include "ToyTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Constant Operation
//===----------------------------------------------------------------------===//

def Toy_ConstantOp : Toy_Op<"constant",
    [ConstantLike, Pure,
     DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "상수 텐서 생성";
  let description = [{
    상수 텐서를 생성하는 operation입니다.
    
    예제:
      %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> 
           : !toy.tensor<2x3xf64>
  }];
  
  let arguments = (ins F64ElementsAttr:$value);
  let results = (outs Toy_TensorType:$output);
  
  let builders = [
    OpBuilder<(ins "DenseElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,
    OpBuilder<(ins "DenseElementsAttr":$value, "Type":$type), [{
      $_state.addTypes(type);
      $_state.addAttribute("value", value);
    }]>
  ];
  
  let hasFolder = 1;
  let hasVerifier = 1;
  
  let extraClassDeclaration = [{
    /// 상수 값이 splat인지 확인
    bool isSplat() const { return getValue().isSplat(); }
    
    /// Splat 값 가져오기
    APFloat getSplatValue() const {
      assert(isSplat());
      return getValue().getSplatValue<APFloat>();
    }
  }];
  
  let assemblyFormat = "attr-dict $value `:` type($output)";
}

//===----------------------------------------------------------------------===//
// Binary Operations
//===----------------------------------------------------------------------===//

// Binary operation들을 위한 base class
class Toy_BinaryOp<string mnemonic, list<Trait> traits = []> :
    Toy_Op<mnemonic, !listconcat(traits, [Pure,
           DeclareOpInterfaceMethods<InferTypeOpInterface>])> {
  let arguments = (ins Toy_TensorType:$lhs, Toy_TensorType:$rhs);
  let results = (outs Toy_TensorType:$output);
  
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($output)";
}

def Toy_AddOp : Toy_BinaryOp<"add"> {
  let summary = "원소별 텐서 덧셈";
  let description = [{
    두 텐서의 원소별 덧셈을 수행합니다.
    두 텐서는 같은 shape이어야 합니다.
    
    예제:
      %2 = toy.add %0, %1 : !toy.tensor<2x3xf64>
  }];
  
  let hasFolder = 1;
}

def Toy_MulOp : Toy_BinaryOp<"mul"> {
  let summary = "원소별 텐서 곱셈";
  let description = [{
    두 텐서의 원소별 곱셈을 수행합니다.
    
    예제:
      %2 = toy.mul %0, %1 : !toy.tensor<2x3xf64>
  }];
  
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Matrix Operations
//===----------------------------------------------------------------------===//

def Toy_MatMulOp : Toy_Op<"matmul", [Pure]> {
  let summary = "행렬 곱셈";
  let description = [{
    두 2D 텐서의 행렬 곱셈을 수행합니다.
    lhs의 열 수와 rhs의 행 수가 같아야 합니다.
    
    예제:
      %2 = toy.matmul %0, %1 : (!toy.tensor<2x3xf64>, !toy.tensor<3x4xf64>) 
                               -> !toy.tensor<2x4xf64>
  }];
  
  let arguments = (ins 
    Toy_TensorType:$lhs,
    Toy_TensorType:$rhs
  );
  let results = (outs Toy_TensorType:$output);
  
  let hasVerifier = 1;
  
  let extraClassDeclaration = [{
    /// 결과 shape 계산
    SmallVector<int64_t, 2> inferResultShape();
  }];
  
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` 
    `(` type($lhs) `,` type($rhs) `)` `->` type($output)
  }];
}

//===----------------------------------------------------------------------===//
// Reshape Operations
//===----------------------------------------------------------------------===//

def Toy_TransposeOp : Toy_Op<"transpose", [Pure]> {
  let summary = "2D 텐서 전치";
  let description = [{
    2D 텐서의 전치를 수행합니다.
    
    예제:
      %1 = toy.transpose %0 : !toy.tensor<2x3xf64> -> !toy.tensor<3x2xf64>
  }];
  
  let arguments = (ins Toy_TensorType:$input);
  let results = (outs Toy_TensorType:$output);
  
  let hasFolder = 1;
  let hasVerifier = 1;
  
  let extraClassDeclaration = [{
    /// transpose(transpose(x)) = x 인지 확인
    bool isInvolution();
  }];
  
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

def Toy_ReshapeOp : Toy_Op<"reshape", [Pure]> {
  let summary = "텐서 reshape";
  let description = [{
    텐서를 새로운 shape으로 변환합니다.
    전체 원소 개수는 유지되어야 합니다.
    
    예제:
      %1 = toy.reshape %0 : !toy.tensor<2x3xf64> -> !toy.tensor<6xf64>
  }];
  
  let arguments = (ins Toy_TensorType:$input);
  let results = (outs Toy_TensorType:$output);
  
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
  
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

//===----------------------------------------------------------------------===//
// Function Call Operation
//===----------------------------------------------------------------------===//

def Toy_CallOp : Toy_Op<"call",
    [DeclareOpInterfaceMethods<CallOpInterface>]> {
  let summary = "함수 호출";
  let description = [{
    Toy 함수를 호출합니다.
    
    예제:
      %0 = toy.call @multiply_transpose(%a, %b) 
           : (!toy.tensor<2x3xf64>, !toy.tensor<2x3xf64>) -> !toy.tensor<3x2xf64>
  }];
  
  let arguments = (ins 
    FlatSymbolRefAttr:$callee,
    Variadic<Toy_Type>:$inputs
  );
  let results = (outs Variadic<Toy_Type>:$outputs);
  
  let builders = [
    OpBuilder<(ins "StringRef":$callee, "ArrayRef<Value>":$arguments)>
  ];
  
  let extraClassDeclaration = [{
    /// 호출되는 함수의 이름 반환
    StringRef getCallee() { return getCalleeAttr().getValue(); }
    
    /// CallOpInterface 구현
    CallInterfaceCallable getCallableForCallee() {
      return getCalleeAttr();
    }
    
    /// 피연산자 범위 반환
    operand_range getArgOperands() {
      return getInputs();
    }
  }];
  
  let assemblyFormat = [{
    $callee `(` $inputs `)` attr-dict `:` 
    functional-type($inputs, $outputs)
  }];
}

//===----------------------------------------------------------------------===//
// Print Operation
//===----------------------------------------------------------------------===//

def Toy_PrintOp : Toy_Op<"print"> {
  let summary = "텐서 출력";
  let description = [{
    텐서를 표준 출력으로 출력합니다.
    이 operation은 side effect를 가집니다.
    
    예제:
      toy.print %0 : !toy.tensor<2x3xf64>
  }];
  
  let arguments = (ins Toy_TensorType:$input);
  
  let extraClassDeclaration = [{
    /// MemoryEffectOpInterface 구현
    void getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
      effects.emplace_back(MemoryEffects::Write::get(), 
                          &getOperation()->getOpOperandUse(0));
    }
  }];
  
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

//===----------------------------------------------------------------------===//
// Return Operation
//===----------------------------------------------------------------------===//

def Toy_ReturnOp : Toy_Op<"return", 
    [Pure, HasParent<"FuncOp">, Terminator]> {
  let summary = "함수에서 값 반환";
  let description = [{
    Toy 함수에서 값을 반환합니다.
    
    예제:
      toy.return %0 : !toy.tensor<2x3xf64>
  }];
  
  let arguments = (ins Variadic<Toy_Type>:$operands);
  
  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, llvm::None);
    }]>
  ];
  
  let hasVerifier = 1;
  
  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";
}

#endif // TOY_OPS
```

## C++ 통합

### ToyDialect.cpp

```cpp
//===----------------------------------------------------------------------===//
// ToyDialect.cpp - Toy Dialect 구현
//===----------------------------------------------------------------------===//

#include "Toy/ToyDialect.h"
#include "Toy/ToyOps.h"
#include "Toy/ToyTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::toy;

//===----------------------------------------------------------------------===//
// TableGen에서 생성된 정의 포함
//===----------------------------------------------------------------------===//

#include "Toy/ToyDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Toy Dialect
//===----------------------------------------------------------------------===//

void ToyDialect::initialize() {
  // Operations 등록
  addOperations<
#define GET_OP_LIST
#include "Toy/ToyOps.cpp.inc"
  >();
  
  // Types 등록
  addTypes<
#define GET_TYPEDEF_LIST
#include "Toy/ToyTypes.cpp.inc"
  >();
  
  // Interfaces 등록
  addInterfaces<ToyInlinerInterface>();
}

//===----------------------------------------------------------------------===//
// Type Parsing and Printing
//===----------------------------------------------------------------------===//

/// Parse a type registered to this dialect.
Type ToyDialect::parseType(DialectAsmParser &parser) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();
  
  Type result;
  OptionalParseResult parseResult = 
    generatedTypeParser(parser, keyword, result);
  if (parseResult.has_value())
    return result;
  
  parser.emitError(parser.getNameLoc(), "unknown toy type: ") << keyword;
  return Type();
}

/// Print a type registered to this dialect.
void ToyDialect::printType(Type type, DialectAsmPrinter &os) const {
  if (failed(generatedTypePrinter(type, os))) {
    llvm_unreachable("unexpected 'toy' type");
  }
}

//===----------------------------------------------------------------------===//
// Toy Inliner Interface
//===----------------------------------------------------------------------===//

struct ToyInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  
  /// Toy operation들은 모두 inlining 가능
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }
  
  /// Toy operation들은 모두 region으로 inlining 가능
  bool isLegalToInline(Operation *, Region *, bool,
                       BlockAndValueMapping &) const final {
    return true;
  }
  
  /// toy.return을 처리
  void handleTerminator(Operation *op,
                       ArrayRef<Value> valuesToRepl) const final {
    auto returnOp = dyn_cast<toy::ReturnOp>(op);
    if (!returnOp)
      return;
    
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }
};
```

### ToyOps.cpp

```cpp
//===----------------------------------------------------------------------===//
// ToyOps.cpp - Toy Operations 구현
//===----------------------------------------------------------------------===//

#include "Toy/ToyOps.h"
#include "Toy/ToyDialect.h"
#include "Toy/ToyTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::toy;

//===----------------------------------------------------------------------===//
// TableGen에서 생성된 정의 포함
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Toy/ToyOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Toy ConstantOp
//===----------------------------------------------------------------------===//

LogicalResult ConstantOp::verify() {
  // value attribute의 타입과 result 타입이 일치하는지 확인
  auto attrType = getValue().getType().cast<TensorType>();
  auto resultType = getResult().getType().cast<TensorType>();
  
  if (attrType.getShape() != resultType.getShape()) {
    return emitOpError("shape mismatch between attribute and result type");
  }
  
  if (attrType.getElementType() != resultType.getElementType()) {
    return emitOpError("element type mismatch");
  }
  
  return success();
}

OpFoldResult ConstantOp::fold(ArrayRef<Attribute> operands) {
  // 상수는 항상 자신의 값으로 fold
  return getValue();
}

//===----------------------------------------------------------------------===//
// Toy AddOp
//===----------------------------------------------------------------------===//

LogicalResult AddOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    ValueRange operands, DictionaryAttr attributes,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  
  auto lhsType = operands[0].getType().cast<TensorType>();
  auto rhsType = operands[1].getType().cast<TensorType>();
  
  // Broadcasting 규칙 적용 (간단한 버전)
  if (lhsType != rhsType) {
    return emitOptionalError(location, 
                           "operands must have the same type");
  }
  
  inferredReturnTypes.push_back(lhsType);
  return success();
}

OpFoldResult AddOp::fold(ArrayRef<Attribute> operands) {
  // 상수 folding: 두 피연산자가 모두 상수인 경우
  if (operands.size() != 2 || !operands[0] || !operands[1])
    return {};
  
  auto lhs = operands[0].cast<DenseElementsAttr>();
  auto rhs = operands[1].cast<DenseElementsAttr>();
  
  // 원소별 덧셈 수행
  return constFoldBinaryOp<FloatAttr>(
      ArrayRef<Attribute>{lhs, rhs},
      [](const APFloat &a, const APFloat &b) {
        APFloat result(a);
        result.add(b, APFloat::rmNearestTiesToEven);
        return result;
      });
}

//===----------------------------------------------------------------------===//
// Toy MulOp
//===----------------------------------------------------------------------===//

void MulOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                       MLIRContext *context) {
  // 패턴 추가: mul(x, 1) -> x
  results.add<MulOpIdentityPattern>(context);
  
  // 패턴 추가: mul(x, 0) -> 0
  results.add<MulOpZeroPattern>(context);
}

//===----------------------------------------------------------------------===//
// Canonicalization Patterns
//===----------------------------------------------------------------------===//

namespace {
/// mul(x, 1) -> x
struct MulOpIdentityPattern : public OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;
  
  LogicalResult matchAndRewrite(MulOp op,
                               PatternRewriter &rewriter) const override {
    // RHS가 1인지 확인
    DenseElementsAttr constValue;
    if (!matchPattern(op.getRhs(), m_Constant(&constValue)))
      return failure();
    
    if (!constValue.isSplat() || 
        !constValue.getSplatValue<APFloat>().isExactlyValue(1.0))
      return failure();
    
    rewriter.replaceOp(op, op.getLhs());
    return success();
  }
};

/// mul(x, 0) -> 0
struct MulOpZeroPattern : public OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;
  
  LogicalResult matchAndRewrite(MulOp op,
                               PatternRewriter &rewriter) const override {
    // RHS가 0인지 확인
    DenseElementsAttr constValue;
    if (!matchPattern(op.getRhs(), m_Constant(&constValue)))
      return failure();
    
    if (!constValue.isSplat() || 
        !constValue.getSplatValue<APFloat>().isExactlyValue(0.0))
      return failure();
    
    rewriter.replaceOpWithNewOp<ConstantOp>(op, constValue);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Toy TransposeOp
//===----------------------------------------------------------------------===//

LogicalResult TransposeOp::verify() {
  auto inputType = getInput().getType().cast<TensorType>();
  auto resultType = getResult().getType().cast<TensorType>();
  
  // 2D 텐서인지 확인
  if (inputType.getRank() != 2) {
    return emitOpError("expected 2D tensor");
  }
  
  // Shape이 올바르게 전치되었는지 확인
  auto inputShape = inputType.getShape();
  auto resultShape = resultType.getShape();
  
  if (inputShape[0] != resultShape[1] || inputShape[1] != resultShape[0]) {
    return emitOpError("invalid transpose dimensions");
  }
  
  return success();
}

OpFoldResult TransposeOp::fold(ArrayRef<Attribute> operands) {
  // transpose(transpose(x)) -> x
  if (auto transposeInput = getInput().getDefiningOp<TransposeOp>()) {
    return transposeInput.getInput();
  }
  
  return {};
}

//===----------------------------------------------------------------------===//
// Toy MatMulOp
//===----------------------------------------------------------------------===//

LogicalResult MatMulOp::verify() {
  auto lhsType = getLhs().getType().cast<TensorType>();
  auto rhsType = getRhs().getType().cast<TensorType>();
  auto resultType = getResult().getType().cast<TensorType>();
  
  // 2D 텐서인지 확인
  if (lhsType.getRank() != 2 || rhsType.getRank() != 2) {
    return emitOpError("expected 2D tensors");
  }
  
  // 행렬 곱셈 차원 확인
  if (lhsType.getDimSize(1) != rhsType.getDimSize(0)) {
    return emitOpError("incompatible matrix dimensions for multiplication");
  }
  
  // 결과 shape 확인
  if (resultType.getDimSize(0) != lhsType.getDimSize(0) ||
      resultType.getDimSize(1) != rhsType.getDimSize(1)) {
    return emitOpError("incorrect result dimensions");
  }
  
  return success();
}

SmallVector<int64_t, 2> MatMulOp::inferResultShape() {
  auto lhsType = getLhs().getType().cast<TensorType>();
  auto rhsType = getRhs().getType().cast<TensorType>();
  
  return {lhsType.getDimSize(0), rhsType.getDimSize(1)};
}

//===----------------------------------------------------------------------===//
// Toy ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  // parent function의 반환 타입과 일치하는지 확인
  auto function = dyn_cast<func::FuncOp>((*this)->getParentOp());
  if (!function)
    return emitOpError("expected to be within a function");
  
  // 반환 타입 검증
  auto functionType = function.getFunctionType();
  if (functionType.getNumResults() != getNumOperands()) {
    return emitOpError("number of return values does not match function");
  }
  
  for (unsigned i = 0, e = getNumOperands(); i < e; ++i) {
    if (getOperand(i).getType() != functionType.getResult(i)) {
      return emitOpError("type of return value ")
             << i << " does not match function result type";
    }
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// Toy CallOp
//===----------------------------------------------------------------------===//

void CallOp::build(OpBuilder &builder, OperationState &state,
                   StringRef callee, ArrayRef<Value> arguments) {
  state.addOperands(arguments);
  state.addAttribute("callee", builder.getSymbolRefAttr(callee));
  
  // 함수 타입에서 결과 타입 추론
  auto function = 
      builder.getInsertionBlock()->getParent()->getParentOfType<ModuleOp>()
          .lookupSymbol<func::FuncOp>(callee);
  if (function) {
    state.addTypes(function.getFunctionType().getResults());
  }
}
```

## 빌드 설정

### CMakeLists.txt

```cmake
# TableGen 파일들을 처리
set(LLVM_TARGET_DEFINITIONS ToyOps.td)
mlir_tablegen(ToyOps.h.inc -gen-op-decls)
mlir_tablegen(ToyOps.cpp.inc -gen-op-defs)
mlir_tablegen(ToyOpsDialect.h.inc -gen-dialect-decls)
mlir_tablegen(ToyOpsDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(ToyOpsIncGen)

set(LLVM_TARGET_DEFINITIONS ToyTypes.td)
mlir_tablegen(ToyTypes.h.inc -gen-typedef-decls)
mlir_tablegen(ToyTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(ToyTypesIncGen)

# Documentation 생성
add_mlir_doc(ToyDialect ToyDialect Toy/ -gen-dialect-doc)
add_mlir_doc(ToyOps ToyOps Toy/ -gen-op-doc)

# 라이브러리 빌드
add_mlir_dialect_library(MLIRToy
  ToyDialect.cpp
  ToyOps.cpp
  ToyTypes.cpp
  
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Toy
  
  DEPENDS
  ToyOpsIncGen
  ToyTypesIncGen
  
  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRIR
  MLIRFuncDialect
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)
```

## 사용 예제

### Toy 프로그램을 MLIR로 변환

```mlir
// example.mlir - Toy dialect를 사용한 MLIR 표현
module {
  // 함수 정의
  func.func @multiply_transpose(%arg0: !toy.tensor<2x3xf64>, 
                               %arg1: !toy.tensor<2x3xf64>) 
                               -> !toy.tensor<3x2xf64> {
    // 전치 연산
    %0 = toy.transpose %arg0 : !toy.tensor<2x3xf64> -> !toy.tensor<3x2xf64>
    %1 = toy.transpose %arg1 : !toy.tensor<2x3xf64> -> !toy.tensor<3x2xf64>
    
    // 행렬 곱셈
    %2 = toy.matmul %0, %1 : (!toy.tensor<3x2xf64>, !toy.tensor<3x2xf64>) 
                            -> !toy.tensor<3x2xf64>
    
    toy.return %2 : !toy.tensor<3x2xf64>
  }
  
  func.func @main() {
    // 상수 텐서 생성
    %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> 
         : !toy.tensor<2x3xf64>
    %1 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> 
         : !toy.tensor<2x3xf64>
    
    // 함수 호출
    %2 = toy.call @multiply_transpose(%0, %1) 
         : (!toy.tensor<2x3xf64>, !toy.tensor<2x3xf64>) -> !toy.tensor<3x2xf64>
    
    // 결과 출력
    toy.print %2 : !toy.tensor<3x2xf64>
    
    toy.return
  }
}
```

### MLIR 도구 사용

```bash
# Toy dialect 파싱 및 검증
mlir-opt example.mlir

# Canonicalization 적용
mlir-opt --canonicalize example.mlir

# Affine dialect로 lowering
mlir-opt --convert-toy-to-affine example.mlir

# 전체 lowering 파이프라인
mlir-opt --convert-toy-to-affine \
         --lower-affine \
         --convert-scf-to-cf \
         --convert-cf-to-llvm \
         --convert-func-to-llvm \
         example.mlir
```

## 정리

TableGen을 사용한 MLIR dialect 정의는 다음과 같은 장점을 제공합니다:

1. **선언적 정의**: Operation, 타입, 속성을 간결하게 정의
2. **자동 코드 생성**: Parser, printer, verifier 등 자동 생성
3. **문서화**: 자동으로 문서 생성
4. **타입 안전성**: C++ 타입 시스템과 통합
5. **재사용성**: 공통 패턴과 trait 재사용

이 가이드를 통해 완전한 기능을 갖춘 MLIR dialect를 정의하고 사용할 수 있습니다.
