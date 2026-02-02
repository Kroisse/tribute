# Tribute Syntax Specification

> 이 문서는 Tribute 언어의 전체 문법을 정의한다.
> 다른 설계 문서들(design.md, types.md, abilities.md, modules.md)의 문법을 통합 정리한 것이다.

## Notation

```ebnf
A | B       택일 (A 또는 B)
A*          0개 이상 반복
A+          1개 이상 반복
A?          선택적 (0개 또는 1개)
'literal'   리터럴 토큰
(A B)       그룹화
```

---

## Lexical Elements

### Keywords

```text
fn let const struct enum ability mod pub use case handle if as
True False Nil
```

**Note:** `if`는 guard 문법에서만 사용 (독립적인 if expression 없음)

### Reserved (향후 사용)

```text
type where in do
```

**Note:** 대부분의 제어 흐름은 algebraic effect로 처리하므로 예약어를 최소화함

### Operators

```text
// 산술
+  -  *  /  %

// 비교
==  !=  <  >  <=  >=

// 논리
&&  ||

// 기타
=  ->  ::  .  ,  ;  :  |  <>

// 괄호류
(  )  {  }  [  ]
```

### Literals

```ebnf
// 숫자 리터럴
NatLiteral   ::= Digit+                        // 0, 1, 42 → Nat
               | '0b' BinDigit+                // 0b1010 → Nat (binary)
               | '0o' OctDigit+                // 0o777 → Nat (octal)
               | '0x' HexDigit+                // 0xc0ffee → Nat (hexadecimal)
IntLiteral   ::= ('+' | '-') Digit+            // +1, -1 → Int
               | ('+' | '-') '0b' BinDigit+    // +0b1010, -0b1010 → Int
               | ('+' | '-') '0o' OctDigit+    // +0o777, -0o777 → Int
               | ('+' | '-') '0x' HexDigit+    // +0xc0ffee, -0xc0ffee → Int
FloatLiteral ::= Digit+ '.' Digit+             // 1.0, 3.14 → Float
               | ('+' | '-') Digit+ '.' Digit+ // +1.0, -3.14 → Float

BinDigit   ::= '0' | '1'
OctDigit   ::= '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7'
HexDigit   ::= Digit | 'a'..'f' | 'A'..'F'

Number     ::= NatLiteral | IntLiteral | FloatLiteral

// String literals
String       ::= 's'? '"' StringContent* '"'              // "..." 또는 s"..."
               | 's'? '#'+ '"' StringContent* '"' '#'+    // s#"..."# (multiline)
RawString    ::= 'r' '"' RawStringContent* '"'            // r"..."
               | 'r' '#'+ '"' RawStringContent* '"' '#'+  // r#"..."# (multiline)
               | 'r' 's' '"' RawStringContent* '"'        // rs"..."
               | 'r' 's' '#'+ '"' RawStringContent* '"' '#'+ // rs#"..."#
               | 's' 'r' '"' RawStringContent* '"'        // sr"..."
               | 's' 'r' '#'+ '"' RawStringContent* '"' '#'+ // sr#"..."#

// Bytes literals
Bytes        ::= 'b' '"' BytesContent* '"'                // b"..."
               | 'b' '#'+ '"' BytesContent* '"' '#'+      // b#"..."# (multiline)
RawBytes     ::= 'r' 'b' '"' RawBytesContent* '"'         // rb"..."
               | 'r' 'b' '#'+ '"' RawBytesContent* '"' '#'+ // rb#"..."# (multiline)
               | 'b' 'r' '"' RawBytesContent* '"'         // br"..."
               | 'b' 'r' '#'+ '"' RawBytesContent* '"' '#'+ // br#"..."#

StringContent     ::= TextChar | EscapeSeq | StringInterpolation
BytesContent      ::= ByteChar | EscapeSeq | BytesInterpolation
RawStringContent  ::= RawChar | StringInterpolation       // escape 처리 안 함
RawBytesContent   ::= RawChar | BytesInterpolation        // escape 처리 안 함
RawChar           ::= Any

EscapeSeq    ::= '\' ('n' | 'r' | 't' | '0' | '"' | '\' | 'x' HexDigit{2} | 'u' HexDigit{4})
StringInterpolation ::= '\{' Expression '}'   // Expression은 Text 타입
BytesInterpolation  ::= '\{' Expression '}'   // Expression은 Bytes 타입

// Rune (Unicode codepoint)
Rune       ::= '?' (PrintableChar | RuneEscape)
RuneEscape ::= '\' ('n' | 'r' | 't' | '0' | '\' | 'x' HexDigit{2} | 'u' HexDigit{4})

// Bool / Nil (키워드)
Bool       ::= 'True' | 'False'
Unit       ::= 'Nil'

// Literal (Text와 Bytes는 raw 변형 포함)
StringLit  ::= String | RawString
BytesLit   ::= Bytes | RawBytes
Literal    ::= Number | StringLit | BytesLit | Rune | Bool | Unit

Identifier ::= (Letter | '_') (Letter | Digit | '_')*
TypeId     ::= UpperLetter (Letter | Digit | '_')*   // 타입명은 대문자 시작
```

**Note:** `#` 개수는 양쪽이 일치해야 함. 내부에 `"#`이 포함된 경우 `##`로 감싸면 됨.

**String 리터럴 예시:**

```rust
"hello"                     // 기본
"Hello, \{name}!"           // interpolation
s#"
    multiline
    string
"#                          // multiline
r"\d+\.\d+"                 // raw (escape 없음)
r#"she said "hello""#       // raw + 따옴표 포함
r##"contains "#"##          // ## 로 감싸기
```

**Bytes 리터럴 예시:**

```rust
b"hello"                    // Bytes
b"\x00\x01\x02"             // escape
b"data: \{chunk}"           // interpolation (Bytes 타입)
rb"\x00"                    // raw bytes (문자 그대로 \x00)
```

**Rune 리터럴 예시:**

```rust
?a       // 'a'
?Z       // 'Z'
?\n      // newline
?\t      // tab
?\x41    // 'A' (hex)
?\u0041  // 'A' (unicode)
?\u3042  // 'あ' (unicode)
```

**숫자 리터럴 예시:**

```rust
// Nat (0과 양수)
0
1
42
1000
0b1010       // binary: 10
0o777        // octal: 511
0xc0ffee     // hexadecimal: 12648430

// Int (부호 명시)
+1
-1
+42
-1000
+0b1010      // binary: +10
-0b1010      // binary: -10
+0o777       // octal: +511
-0xc0ffee    // hexadecimal: -12648430

// Float (소수점 + 소수부 필수)
1.0
3.14
+1.0
-3.14
0.5

// UFCS와 구분
1.abs        // Nat(1).abs() - UFCS 호출
1.0.abs      // Float(1.0).abs() - UFCS 호출
```

### Comments

```ebnf
LineComment  ::= '//' .* '\n'
BlockComment ::= '/*' .* '*/'

// Doc comments
DocComment   ::= '///' .* '\n'              // 한 줄 문서화
DocBlock     ::= '/**' .* '*/'              // 블록 문서화
```

**Doc comment 예시:**

```rust
/// 두 숫자를 더한다.
///
/// ## Examples
/// ```
/// add(1, 2)  // 3
/// ```
fn add(x: Nat, y: Nat) -> Nat {
    x + y
}

/**
 * 사용자 정보를 담는 구조체.
 *
 * name: 사용자 이름
 * age: 사용자 나이
 */
struct User {
    name: Text
    age: Nat
}
```

---

## Program Structure

```ebnf
Program ::= Item*

Item ::= UseDecl
       | ModDecl
       | ConstDecl
       | StructDecl
       | EnumDecl
       | AbilityDecl
       | FunctionDef
       | Expression
```

---

## Module System

### Use Declaration

```ebnf
UseDecl ::= 'use' UsePath

UsePath ::= PathSegment ('::' PathSegment)* UseTree?

UseTree ::= '::' '{' UseItem (',' UseItem)* ','? '}'
          | 'as' Identifier

UseItem ::= Identifier ('as' Identifier)?
```

**예시:**

```rust
use std::collections::List
use std::collections::{List, Option, Result}
use std::io::Console as IO
```

### Module Declaration

```ebnf
ModDecl ::= 'pub'? 'mod' TypeId '{' Item* '}'
```

**예시:**

```rust
pub mod List {
    pub fn empty() -> List(a) { Empty }
    pub fn map(xs: List(a), f: fn(a) ->{g} b) ->{g} List(b) { ... }
}
```

### Path Expression

```ebnf
Path ::= PathSegment ('::' PathSegment)*
PathSegment ::= Identifier | TypeId
```

**예시:**

```rust
List::empty()
Option::Some(42)
std::io::Console::println
```

---

## Constant Declaration

```ebnf
ConstDecl ::= 'pub'? 'const' Identifier (':' Type)? '=' ConstExpr

ConstExpr ::= Literal                        // -1은 Int 리터럴로 처리
            | Path                           // 다른 상수 참조
            | ConstExpr BinOp ConstExpr      // 상수 폴딩
            | '{' ConstExpr '}'              // 그룹화
```

**Note:** 함수 호출은 const 표현식에서 허용되지 않음 (const fn 없음)

**예시:**

```rust
const MAX_SIZE = 1000
const PI = 3.14159
const GREETING = "Hello, Tribute!"
const DOUBLE_MAX = MAX_SIZE * 2
const NEWLINE = ?\n

pub const VERSION = "0.1.0"
```

---

## Type Declarations

### Struct (Product Type)

```ebnf
StructDecl ::= 'pub'? 'struct' TypeId TypeParams? StructBody

TypeParams ::= '(' TypeParam (',' TypeParam)* ','? ')'
TypeParam  ::= LowerIdentifier

StructBody ::= '{' StructFields '}'
StructFields ::= StructField (FieldSep StructField)* FieldSep?
StructField ::= Identifier ':' Type
FieldSep ::= ',' | '\n'
```

**예시:**

```rust
struct User {
    name: Text
    age: Nat
}

struct Box(a) {
    value: a
}

// 한 줄이면 쉼표 필수
struct Point { x: Int, y: Int }
```

### Enum (Sum Type)

```ebnf
EnumDecl ::= 'pub'? 'enum' TypeId TypeParams? EnumBody

EnumBody ::= '{' EnumVariants '}'
EnumVariants ::= EnumVariant (FieldSep EnumVariant)* FieldSep?
EnumVariant ::= TypeId VariantFields?

VariantFields ::= '(' Type (',' Type)* ','? ')'     // positional
                | '{' StructFields '}'          // named
```

**예시:**

```rust
enum Option(a) {
    None
    Some(a)
}

enum Result(a, e) {
    Ok { value: a }
    Error { error: e }
}

// 혼합
enum Expr {
    Lit(Int)
    Var(Text)
    BinOp { op: Text, lhs: Expr, rhs: Expr }
}
```

---

## Type Syntax

```ebnf
Type ::= TypePath TypeArgs?
       | FunctionType
       | TupleType

TupleType ::= '#(' TypeList? ')'              // #(Int, Text, Float)

TypePath ::= (PathSegment '::')* TypeId
TypeArgs ::= '(' Type (',' Type)* ','? ')'

FunctionType ::= 'fn' '(' TypeList? ')' ReturnType
TypeList ::= Type (',' Type)* ','?

ReturnType ::= '->' Type                      // 암묵적 effect polymorphic
             | '->' '{' EffectRow? '}' Type   // 명시적 effect

EffectRow ::= EffectItem (',' EffectItem)* EffectTail? ','?
EffectItem ::= TypeId TypeArgs?
EffectTail ::= ',' LowerIdentifier            // row variable
```

**예시:**

```rust
Nat                           // 0, 양수
Int                           // 정수 (부호 있음)
Float                         // 부동소수점
Text
List(Int)
Option(Text)
Result(Int, Text)

#(Int, Text)                  // 2-tuple (pair)
#(Int, Text, Float)           // 3-tuple
Nil                           // unit type (#() 대신 사용)

fn(Int, Int) -> Int           // 암묵적 effect polymorphic
fn(Int) ->{} Int              // 순수 함수
fn(Text) ->{Http} Response    // Http effect
fn() ->{State(Int), e} Int    // State + row variable e
```

---

## Ability (Effect) System

### Ability Declaration

```ebnf
AbilityDecl ::= 'ability' TypeId TypeParams? '{' AbilityOp* '}'

AbilityOp ::= 'fn' Identifier '(' ParamList? ')' '->' Type
```

**예시:**

```rust
ability Console {
    fn print(msg: Text) -> Nil
    fn read() -> Text
}

ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

ability Http {
    fn get(url: Text) -> Response
    fn post(url: Text, body: Text) -> Response
}
```

### Handle Expression

```ebnf
HandleExpr ::= 'handle' Expression '{' HandlerArm+ '}'

HandlerArm ::= HandlerPattern '->' Expression
             | HandlerPattern GuardedBranch+

HandlerPattern ::= '{' Identifier '}'                              // completion: { result }
                 | '{' Path '(' PatternList? ')' '->' Identifier '}'  // suspend: { Op(args) -> k }
```

`handle expr { arms }`는 computation을 실행하고 handler arm으로 결과를 처리한다.

**Handler pattern 종류:**

| 패턴 | 의미 |
|------|------|
| `{ value }` | Computation이 완료됨, 결과값을 `value`에 바인딩 |
| `{ Op(args) -> k }` | Ability operation에서 suspend됨, continuation을 `k`에 바인딩 |

**Continuation:**

Continuation `k`는 computation을 재개하는 함수다:

```rust
{ State::get() -> k } -> k(current_state)  // resume with value
{ State::set(v) -> k } -> k(Nil)           // resume with unit
```

Continuation은 **linear type**이다. 반드시 한 번 사용하거나 명시적으로 버려야 한다:

```rust
// 사용: 함수로 호출
{ State::get() -> k } -> k(state)

// 버림: drop 함수 사용
{ Fail::fail(msg) -> k } -> {
    drop(k)
    None
}
```

**예시:**

```rust
fn run_state(comp: fn() ->{e, State(s)} a, state: s) ->{e} a {
    handle comp() {
        { result } -> result
        { State::get() -> k } -> run_state(fn() k(state), state)
        { State::set(v) -> k } -> run_state(fn() k(Nil), v)
    }
}
```

---

## Functions

### Function Definition

```ebnf
FunctionDef ::= 'pub'? 'fn' Identifier '(' ParamList? ')' ReturnType? Block

ParamList ::= Param (',' Param)* ','?
Param ::= Identifier (':' Type)?

// ReturnType은 Type Syntax 섹션에 정의됨
```

**예시:**

```rust
fn add(x: Int, y: Int) -> Int {
    x + y
}

fn fetch(url: Text) ->{Http} Response {
    Http::get(url)
}

fn example() ->{State(Int), Console} Nil {
    let n = State::get()
    Console::print(Int::to_string(n))
}

// 순수 함수 명시
fn pure_add(x: Int, y: Int) ->{} Int {
    x + y
}
```

### Lambda Expression

```ebnf
Lambda ::= 'fn' '(' ParamList? ')' Expression
```

**예시:**

```rust
fn(x) x + 1
fn(x, y) x + y
fn(x) {
    let y = x + 1
    y * 2
}
```

---

## Expressions

### Primary Expressions

```ebnf
PrimaryExpr ::= Literal
              | Identifier
              | Path
              | Block                             // { expr } 로 그룹화
              | ListExpr
              | TupleExpr
              | RecordExpr
              | OperatorFn
              | Lambda
              | CaseExpr
              | HandleExpr

ListExpr ::= '[' ExprList? ']'
TupleExpr ::= '#(' ExprList? ')'          // #(1, "hello", 3.14)
OperatorFn ::= '(' Operator ')'           // (+), (<>)
             | '(' QualifiedOp ')'        // (Int::+), (Text::<>)
```

### Block Expression

```ebnf
Block ::= '{' Statement* Expression? '}'

Statement ::= LetStatement
            | Expression ';'?

LetStatement ::= 'let' Pattern (':' Type)? '=' Expression
```

**예시:**

```rust
// 여러 문장
{
    let x = 1
    let y = 2
    x + y
}

// 연산 우선순위 조정 (괄호 대신 블록 사용)
{ a + b } * c           // (a + b) * c 와 동일
x * { y + z }           // x * (y + z) 와 동일
```

**Note:** `(expr)` 형태의 괄호 표현식은 지원하지 않음. 우선순위 조정에는 `{ expr }` 사용. `(...)` 는 연산자-함수 `(+)`, `(<>)` 전용.

### Record Expression

```ebnf
RecordExpr ::= TypeId '{' RecordFields? '}'
RecordFields ::= RecordField (',' RecordField)* ','?
RecordField ::= '..' Expression                    // spread
              | Identifier ':' Expression          // field: value
              | Identifier                         // shorthand
```

**예시:**

```rust
User { name: "Alice", age: 30 }
User { name, age }              // shorthand
User { ..user, age: 31 }        // spread
Point { x: 10, y: 20 }
```

### Variant Construction

```ebnf
VariantExpr ::= TypeId '(' ExprList? ')'     // positional
              | TypeId '{' RecordFields '}'   // named
              | TypeId                         // unit variant
```

**예시:**

```rust
Some(42)
None
Cons(1, Cons(2, Empty))
Ok { value: 42 }
Error { error: "failed" }
```

### Call and UFCS

```ebnf
CallExpr ::= Expression '(' ExprList? ')'
UFCSExpr ::= Expression '.' Path CallArgs?
CallArgs ::= '(' ExprList? ')'
           | /* empty - 인자 없으면 괄호 생략 가능 */

ExprList ::= Expression (',' Expression)* ','?
```

**UFCS 규칙:**

- `x.f(y)` → `f(x, y)` 또는 `T::f(x, y)` (타입 T에서 해소)
- `x.f` → `f(x)` (인자가 없으면 괄호 생략 가능)
- `x.a::b(y)` → `a::b(x, y)` (qualified path도 가능)

**예시:**

```rust
// 일반 호출
add(1, 2)
List::map(xs, fn(x) x + 1)

// UFCS - 단순 식별자
xs.map(fn(x) x + 1)     // List::map(xs, ...)
xs.len                   // List::len(xs) - 괄호 생략
user.name                // User::name(user) - 필드 접근도 UFCS

// UFCS - qualified path
user.name::set("Jane")   // User::name::set(user, "Jane")
user.age::modify(fn(n) n + 1)

// 체이닝
data
    .filter(fn(x) x > 0)
    .map(fn(x) x * 2)
    .fold(0, fn(a, b) a + b)
```

### Binary Operators

```ebnf
BinaryExpr ::= Expression BinOp Expression
             | Expression QualifiedOp Expression

QualifiedOp ::= Path '::' Operator        // List::<>, Int::+

// 연산자를 함수로 사용
OperatorFn ::= '(' Operator ')'           // (+), (<>)
             | '(' QualifiedOp ')'        // (Int::+), (Text::<>)

// 우선순위 (높은 것부터)
// 1. * / %
// 2. + - <>
// 3. == != < > <= >=
// 4. &&
// 5. ||
```

**연결 연산자 `<>`**: Text, List 등에 사용 (type-directed resolution)

```rust
"Hello, " <> name <> "!"        // Text::<>
[1, 2] <> [3, 4]                // List::<>

// 명시적으로 연산자 지정
xs List::<> ys                  // List::<> 명시
a Int::+ b                      // Int::+ 명시
```

**연산자를 함수로 사용:**

```rust
(+)(a, b)                       // a + b 와 동일
(Text::<>)("a", "b")            // "a" <> "b" 와 동일

// 고차 함수에 전달
xs.fold(0, (+))                 // 합계
xs.fold("", (Text::<>))         // 문자열 연결
numbers.reduce((Int::*))        // 곱셈
```

**단항 연산 (UFCS로 처리):**

단항 연산자 없음. 모든 단항 연산은 UFCS 메서드로 처리:

```rust
x.negate      // -x (숫자 부정)
flag.not      // !flag (논리 부정)
bits.bit_not  // ~bits (비트 부정)
```

### Case Expression

```ebnf
CaseExpr ::= 'case' Expression '{' CaseArm+ '}'

CaseArm ::= Pattern '->' Expression           // guard 없음
          | Pattern GuardedBranch+            // guard 하나 이상

GuardedBranch ::= 'if' Expression '->' Expression
```

**예시:**

```rust
case opt {
    Some(x) -> x
    None -> 0
}

// 단일 guard
case value {
    n if n > 0 -> "positive"
    _ -> "non-positive"
}

// 다중 guard (같은 패턴에 여러 조건)
case value {
    n if n > 0 -> "positive"
      if n < 0 -> "negative"
    _ -> "zero"
}

case result {
    Ok { value } -> value
    Error { error } -> panic(error)
}
```

## Patterns

```ebnf
Pattern ::= LiteralPattern
          | WildcardPattern
          | IdentifierPattern
          | VariantPattern
          | RecordPattern
          | ListPattern
          | TuplePattern
          | AsPattern

AsPattern ::= Pattern 'as' Identifier        // 전체를 바인딩

LiteralPattern ::= Number | StringLit | Rune | 'True' | 'False' | 'Nil'
WildcardPattern ::= '_'
IdentifierPattern ::= Identifier
VariantPattern ::= TypeId ('(' PatternList ')' | '{' RecordPatternFields '}')?
RecordPattern ::= TypeId '{' RecordPatternFields '}'
ListPattern ::= '[' PatternList? ']'
              | '[' PatternList ',' '..' Identifier? ']'    // [head, ..tail] or [head, ..]
TuplePattern ::= '#(' PatternList? ')'

PatternList ::= Pattern (',' Pattern)* ','?
RecordPatternFields ::= RecordPatternField (',' RecordPatternField)* ','? '..'?
RecordPatternField ::= Identifier (':' Pattern)?
```

**Note:** Handler pattern (`{ result }`, `{ Op(args) -> k }`)은 `handle` 표현식 내에서만 사용된다. 일반 `case` 표현식에서는 사용할 수 없다.

**예시:**

```rust
// Literal
0
"hello"

// Wildcard
_

// Identifier (바인딩)
x
name

// Variant
Some(x)
None
Cons(head, tail)
Ok { value }
Error { error: e }

// Record destructuring
User { name, age }
User { name, .. }           // 나머지 무시
Point { x, y: y_coord }     // 이름 변경

// List pattern
[]                          // 빈 리스트
[x]                         // 단일 원소
[a, b, c]                   // 정확히 3개
[head, ..tail]              // head + 나머지
[first, second, ..]         // 처음 두 개만

// Tuple pattern
#(a, b)                     // pair
#(x, _, z)                  // 일부만 바인딩

// As pattern (전체 바인딩)
Some(x) as opt              // x에 내부값, opt에 전체
[head, ..tail] as list      // head, tail, list 모두 바인딩
User { name, .. } as user   // name과 전체 user 바인딩
```

---

## Field Access and Update

### Getter (자동 생성)

```rust
// struct 필드는 자동으로 getter 생성
struct User { name: Text, age: Nat }

// 생성되는 함수:
// User::name : fn(User) -> Text
// User::age  : fn(User) -> Nat

user.name    // User::name(user)
user.age     // User::age(user)
```

### Setter and Modifier

Struct 필드에 대해 자동 생성되는 함수들 (별도 문법 없음, UFCS로 호출):

```rust
// 자동 생성되는 함수:
// User::name::set    : fn(User, Text) -> User
// User::name::modify : fn(User, fn(Text) -> Text) -> User

user.name::set("Jane")              // UFCS: User::name::set(user, "Jane")
user.age::modify(fn(n) n + 1)       // UFCS: User::age::modify(user, ...)

// 체이닝
user
    .name::set("Jane")
    .age::modify(fn(n) n + 1)
```

---

## Visibility

```ebnf
Visibility ::= 'pub'?
```

`pub` 키워드가 붙으면 모듈 외부에서 접근 가능.

**적용 대상:**

- `pub struct`
- `pub enum`
- `pub ability`
- `pub fn`
- `pub mod`
- `pub use` (reexport)

---

## Whitespace and Separators

### 개행으로 구분

다음 위치에서는 개행이 구분자 역할을 한다:

- struct/enum 필드 사이
- 블록 내 문장 사이
- case arm 사이

**예시:**

```rust
// 개행으로 구분
struct User {
    name: Text
    age: Nat
}

// 한 줄이면 쉼표 필수
struct User { name: Text, age: Nat }
```

### 세미콜론

세미콜론은 선택적이지만, 한 줄에 여러 문장을 쓸 때 필요:

```rust
let x = 1; let y = 2; x + y
```

---

## Complete Example

```rust
use std::collections::{List, Option}
use std::io::Console

struct User {
    name: Text
    age: Nat
}

enum Status {
    Active
    Inactive { reason: Text }
}

ability Logger {
    fn log(msg: Text) -> Nil
}

pub fn greet(user: User) ->{Console} Nil {
    let greeting = "Hello, " <> user.name <> "!"
    Console::print(greeting)
}

fn process(users: List(User)) ->{Logger} List(Text) {
    users
        .filter(fn(u) u.age >= 18)
        .map(fn(u) {
            Logger::log("Processing: " <> u.name)
            u.name
        })
}

fn run_logger(comp: fn() ->{e, Logger} a) ->{e, Console} a {
    handle comp() {
        { result } -> result
        { Logger::log(msg) -> k } -> {
            Console::print("[LOG] " <> msg)
            run_logger(fn() k(Nil))
        }
    }
}

fn main() ->{Console} Nil {
    let users = [
        User { name: "Alice", age: 30 },
        User { name: "Bob", age: 17 },
        User { name: "Charlie", age: 25 },
    ]

    run_logger(fn() {
        let names = process(users)
        names.each(fn(name) {
            Console::print("Name: " <> name)
        })
    })
}
```

---

## Grammar Summary

### Top-level

| 구문                       | 설명          |
| -------------------------- | ------------- |
| `use path`                 | 모듈 가져오기 |
| `pub? mod Name { }`        | 모듈 선언     |
| `pub? const NAME = expr`   | 상수 선언     |
| `pub? struct Name(a) { }`  | Product type  |
| `pub? enum Name(a) { }`    | Sum type      |
| `pub? ability Name(a) { }` | Effect 선언   |
| `pub? fn name() { }`       | 함수 정의     |

### Types

| 구문                     | 의미                           |
| ------------------------ | ------------------------------ |
| `Nat`, `Int`, `Float`, `Text`, ... | 기본 타입               |
| `List(a)`, `Option(Int)` | 제네릭 타입                    |
| `#(Int, Text)`           | Tuple 타입                     |
| `fn(a) -> b`             | 함수 타입 (암묵적 polymorphic) |
| `fn(a) ->{} b`           | 순수 함수 타입                 |
| `fn(a) ->{E} b`          | Effect E를 수행하는 함수       |
| `fn(a) ->{E, e} b`       | E + row variable e             |

### Expressions

| 구문                    | 의미                   |
| ----------------------- | ---------------------- |
| `{ stmts; expr }`       | Block / 그룹화         |
| `fn(x) expr`            | Lambda                 |
| `case e { pat -> e }`   | Pattern matching       |
| `handle e { arms }`     | Effect handling        |
| `x.f`                   | UFCS (괄호 생략)       |
| `x.f(y)`                | UFCS                   |
| `T::f(x)`               | Qualified call         |
| `T { f: v }`            | Record construction    |
| `T { ..x, f: v }`       | Record update (spread) |
| `[a, b, c]`             | List literal           |
| `#(a, b, c)`            | Tuple literal          |
| `a <> b`                | Concatenation          |
| `a T::<> b`             | Qualified operator     |
| `(+)`, `(T::<>)`        | Operator as function   |

### Patterns

| 패턴               | 의미                           |
| ------------------ | ------------------------------ |
| `42`, `"hi"`, `?a` | Literal (Number, Text, Rune)   |
| `_`                | Wildcard                       |
| `x`                | Binding                        |
| `Some(x)`          | Variant (positional)           |
| `Ok { value }`     | Variant (named)                |
| `T { f, .. }`      | Record (나머지 무시)           |
| `[a, b, c]`        | List (정확히 일치)             |
| `#(a, b, c)`       | Tuple                          |
| `[h, ..t]`         | List (head + tail)             |
| `pat as x`         | As (전체 바인딩)               |

### Handler Patterns (handle 전용)

| 패턴               | 의미                           |
| ------------------ | ------------------------------ |
| `{ result }`       | Computation 완료               |
| `{ Op(x) -> k }`   | Effect suspend + continuation  |
