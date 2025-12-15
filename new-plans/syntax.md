# Tribute Syntax Specification

> 이 문서는 Tribute 언어의 전체 문법을 정의한다.
> 다른 설계 문서들(design.md, types.md, abilities.md, modules.md)의 문법을 통합 정리한 것이다.

## Notation

```
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

```
fn let const struct enum ability mod pub use case handle if as
```

**Note:** `if`는 guard 문법에서만 사용 (독립적인 if expression 없음)

### Reserved (향후 사용)

```
type where in do
```

**Note:** 대부분의 제어 흐름은 algebraic effect로 처리하므로 예약어를 최소화함

### Operators

```
// 산술
+  -  *  /  %

// 비교
==  !=  <  >  <=  >=

// 논리
&&  ||  !

// 기타
=  ->  ::  .  ,  ;  :  |  <>

// 괄호류
(  )  {  }  [  ]
```

### Literals

```
Number     ::= '-'? Digit+ ('.' Digit+)?

// String literals
String       ::= 's'? '"' StringContent* '"'              // "..." 또는 s"..."
               | 's'? '#'+ '"' StringContent* '"' '#'+    // s#"..."# (multiline)
RawString    ::= 'r' 's'? '"' RawContent* '"'             // r"..." 또는 rs"..."
               | 'r' 's'? '#'+ '"' RawContent* '"' '#'+   // r#"..."# (multiline)
               | 's' 'r' '"' RawContent* '"'              // sr"..."
               | 's' 'r' '#'+ '"' RawContent* '"' '#'+    // sr#"..."#

// Bytes literals
Bytes        ::= 'b' '"' BytesContent* '"'                // b"..."
               | 'b' '#'+ '"' BytesContent* '"' '#'+      // b#"..."# (multiline)
RawBytes     ::= 'r' 'b' '"' RawContent* '"'              // rb"..."
               | 'r' 'b' '#'+ '"' RawContent* '"' '#'+    // rb#"..."# (multiline)
               | 'b' 'r' '"' RawContent* '"'              // br"..."
               | 'b' 'r' '#'+ '"' RawContent* '"' '#'+    // br#"..."#

StringContent ::= TextChar | EscapeSeq | StringInterpolation
BytesContent  ::= ByteChar | EscapeSeq | BytesInterpolation
RawContent    ::= Any*                                    // escape 처리 안 함

EscapeSeq    ::= '\' ('n' | 'r' | 't' | '0' | '"' | '\' | 'x' HexDigit{2} | 'u' HexDigit{4})
StringInterpolation ::= '\{' Expression '}'   // Expression은 String 타입
BytesInterpolation  ::= '\{' Expression '}'   // Expression은 Bytes 타입

// Rune (Unicode codepoint)
Rune       ::= '?' (PrintableChar | RuneEscape)
RuneEscape ::= '\' ('n' | 'r' | 't' | '0' | '\' | 'x' HexDigit{2} | 'u' HexDigit{4})

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

### Comments

```
LineComment  ::= '//' .* '\n'
BlockComment ::= '/*' .* '*/'
```

---

## Program Structure

```
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

```
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

```
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

```
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

```
ConstDecl ::= 'pub'? 'const' Identifier (':' Type)? '=' ConstExpr

ConstExpr ::= Literal
            | Path                           // 다른 상수 참조
            | ConstExpr BinOp ConstExpr      // 상수 폴딩
            | '-' ConstExpr                  // 단항 마이너스
            | '(' ConstExpr ')'
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

```
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
    name: String
    age: Int
}

struct Box(a) {
    value: a
}

// 한 줄이면 쉼표 필수
struct Point { x: Int, y: Int }
```

### Enum (Sum Type)

```
EnumDecl ::= 'pub'? 'enum' TypeId TypeParams? EnumBody

EnumBody ::= '{' EnumVariants '}'
EnumVariants ::= EnumVariant (FieldSep EnumVariant)* FieldSep?
EnumVariant ::= TypeId VariantFields?

VariantFields ::= '(' Type (',' Type)* ','? ')'     // positional
                | '{' StructFields '}'          // named
```

**예시:**

```rust
enum Bool { True, False }

enum Option(a) {
    None
    Some(a)
}

enum Result(a, e) {
    Ok { value: a }
    Err { error: e }
}

// 혼합
enum Expr {
    Lit(Int)
    Var(String)
    BinOp { op: String, lhs: Expr, rhs: Expr }
}
```

---

## Type Syntax

```
Type ::= TypePath TypeArgs?
       | FunctionType
       | '(' Type ')'

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
Int
String
List(Int)
Option(String)
Result(Int, String)

fn(Int, Int) -> Int           // 암묵적 effect polymorphic
fn(Int) ->{} Int              // 순수 함수
fn(String) ->{Http} Response  // Http effect
fn() ->{State(Int), e} Int    // State + row variable e
```

---

## Ability (Effect) System

### Ability Declaration

```
AbilityDecl ::= 'ability' TypeId TypeParams? '{' AbilityOp* '}'

AbilityOp ::= 'fn' Identifier '(' ParamList? ')' '->' Type
```

**예시:**

```rust
ability Console {
    fn print(msg: String) -> Nil
    fn read() -> String
}

ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

ability Http {
    fn get(url: String) -> Response
    fn post(url: String, body: String) -> Response
}
```

### Handle Expression

```
HandleExpr ::= 'handle' Expression
```

`handle expr`은 `Request` 타입의 값을 반환한다. 이 값은 일반적인 `case` 표현식에서 handler pattern으로 매칭할 수 있다.

`Request(e, a)`는 **builtin 타입**이다 (Unison의 `Request {A} T`와 동일한 역할):

- `e`: ability 타입
- `a`: computation의 결과 타입

개념적으로 다음과 같은 구조이지만, 사용자가 직접 정의하거나 생성할 수 없다:

```
Request(e, a) ≈
  | Done(a)                    -- computation 완료
  | Suspend(op: e, k: ...)     -- effect 발생 + continuation
```

**예시:**

```rust
// handle은 Request 값을 반환
let req = handle comp()

// case로 Request를 패턴 매칭
case req {
    { result } -> result
    { State::get() -> k } -> run_state(fn() k(state), state)
    { State::set(v) -> k } -> run_state(fn() k(Nil), v)
}

// 보통은 한 번에 작성
case handle comp() {
    { result } -> result
    { State::get() -> k } -> run_state(fn() k(state), state)
    { State::set(v) -> k } -> run_state(fn() k(Nil), v)
}
```

---

## Functions

### Function Definition

```
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

fn fetch(url: String) ->{Http} Response {
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

```
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

```
PrimaryExpr ::= Literal
              | Identifier
              | Path
              | '(' Expression ')'
              | Block
              | ListExpr
              | RecordExpr
              | OperatorFn
              | Lambda
              | CaseExpr
              | HandleExpr

ListExpr ::= '[' ExprList? ']'
OperatorFn ::= '(' Operator ')'           // (+), (<>)
             | '(' QualifiedOp ')'        // (Int::+), (String::<>)
```

### Block Expression

```
Block ::= '{' Statement* Expression? '}'

Statement ::= LetStatement
            | Expression ';'?

LetStatement ::= 'let' Pattern (':' Type)? '=' Expression
```

**예시:**

```rust
{
    let x = 1
    let y = 2
    x + y
}
```

### Record Expression

```
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

```
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
Err { error: "failed" }
```

### Call and UFCS

```
CallExpr ::= Expression '(' ExprList? ')'
UFCSExpr ::= Expression '.' Identifier CallArgs?
CallArgs ::= '(' ExprList? ')'
           | /* empty - 인자 없으면 괄호 생략 가능 */

ExprList ::= Expression (',' Expression)* ','?
```

**UFCS 규칙:**

- `x.f(y)` → `f(x, y)` 또는 `T::f(x, y)` (타입 T에서 해소)
- `x.f` → `f(x)` (인자가 없으면 괄호 생략 가능)

**예시:**

```rust
// 일반 호출
add(1, 2)
List::map(xs, fn(x) x + 1)

// UFCS
xs.map(fn(x) x + 1)     // List::map(xs, ...)
xs.len                   // List::len(xs) - 괄호 생략
user.name                // User::name(user) - 필드 접근도 UFCS

// 체이닝
data
    .filter(fn(x) x > 0)
    .map(fn(x) x * 2)
    .fold(0, fn(a, b) a + b)
```

### Binary Operators

```
BinaryExpr ::= Expression BinOp Expression
             | Expression QualifiedOp Expression

QualifiedOp ::= Path '::' Operator        // List::<>, Int::+

// 연산자를 함수로 사용
OperatorFn ::= '(' Operator ')'           // (+), (<>)
             | '(' QualifiedOp ')'        // (Int::+), (String::<>)

// 우선순위 (높은 것부터)
// 1. * / %
// 2. + - <>
// 3. == != < > <= >=
// 4. &&
// 5. ||
```

**연결 연산자 `<>`**: String, List 등에 사용 (type-directed resolution)

```rust
"Hello, " <> name <> "!"        // String::<>
[1, 2] <> [3, 4]                // List::<>

// 명시적으로 연산자 지정
xs List::<> ys                  // List::<> 명시
a Int::+ b                      // Int::+ 명시
```

**연산자를 함수로 사용:**

```rust
(+)(a, b)                       // a + b 와 동일
(String::<>)("a", "b")          // "a" <> "b" 와 동일

// 고차 함수에 전달
xs.fold(0, (+))                 // 합계
xs.fold("", (String::<>))       // 문자열 연결
numbers.reduce((Int::*))        // 곱셈
```

### Case Expression

```
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
    Err { error } -> panic(error)
}
```

## Patterns

```
Pattern ::= LiteralPattern
          | WildcardPattern
          | IdentifierPattern
          | VariantPattern
          | RecordPattern
          | ListPattern
          | HandlerPattern
          | ParenPattern

LiteralPattern ::= Number | String | Rune
WildcardPattern ::= '_'
IdentifierPattern ::= Identifier
VariantPattern ::= TypeId ('(' PatternList ')' | '{' RecordPatternFields '}')?
RecordPattern ::= TypeId '{' RecordPatternFields '}'
ListPattern ::= '[' PatternList? ']'
              | '[' PatternList ',' '..' Identifier? ']'    // [head, ..tail] or [head, ..]
HandlerPattern ::= '{' HandlerCase '}'
HandlerCase ::= Identifier                                  // completion: { result }
              | Path '(' PatternList? ')' '->' Identifier   // suspend: { Op(args) -> k }
ParenPattern ::= '(' Pattern ')'

PatternList ::= Pattern (',' Pattern)* ','?
RecordPatternFields ::= RecordPatternField (',' RecordPatternField)* ','? '..'?
RecordPatternField ::= Identifier (':' Pattern)?
```

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
Err { error: e }

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

// Handler pattern (Request 타입 매칭)
{ result }                   // Done(result) 매칭
{ State::get() -> k }        // Suspend 매칭: effect 연산 + continuation
{ Logger::log(msg) -> k }    // Suspend 매칭: 인자와 continuation 바인딩
```

---

## Field Access and Update

### Getter (자동 생성)

```rust
// struct 필드는 자동으로 getter 생성
struct User { name: String, age: Int }

// 생성되는 함수:
// User::name : fn(User) -> String
// User::age  : fn(User) -> Int

user.name    // User::name(user)
user.age     // User::age(user)
```

### Setter and Modifier

```
SetterExpr ::= Expression '.' Identifier '::set' '(' Expression ')'
ModifierExpr ::= Expression '.' Identifier '::modify' '(' Expression ')'
```

**예시:**

```rust
// 생성되는 함수:
// User::name::set    : fn(User, String) -> User
// User::name::modify : fn(User, fn(String) -> String) -> User

user.name::set("Jane")
user.age::modify(fn(n) n + 1)

// 체이닝
user
    .name::set("Jane")
    .age::modify(fn(n) n + 1)
```

---

## Visibility

```
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
    name: String
    age: Int
}

// 한 줄이면 쉼표 필수
struct User { name: String, age: Int }
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
    name: String
    age: Int
}

enum Status {
    Active
    Inactive { reason: String }
}

ability Logger {
    fn log(msg: String) -> Nil
}

pub fn greet(user: User) ->{Console} Nil {
    let greeting = "Hello, " <> user.name <> "!"
    Console::print(greeting)
}

fn process(users: List(User)) ->{Logger} List(String) {
    users
        .filter(fn(u) u.age >= 18)
        .map(fn(u) {
            Logger::log("Processing: " <> u.name)
            u.name
        })
}

fn run_logger(comp: fn() ->{e, Logger} a) ->{e, Console} a {
    case handle comp() {
        { result } -> result
        { Logger::log(msg) -> k } -> {
            Console::print("[LOG] " <> msg)
            run_logger(fn() k(Nil))
        }
    }
}

fn main() ->{Console} Nil {
    let users = List::of(
        User { name: "Alice", age: 30 },
        User { name: "Bob", age: 17 },
        User { name: "Charlie", age: 25 }
    )

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
| `Int`, `String`, ...     | 기본 타입                      |
| `List(a)`, `Option(Int)` | 제네릭 타입                    |
| `fn(a) -> b`             | 함수 타입 (암묵적 polymorphic) |
| `fn(a) ->{} b`           | 순수 함수 타입                 |
| `fn(a) ->{E} b`          | Effect E를 수행하는 함수       |
| `fn(a) ->{E, e} b`       | E + row variable e             |

### Expressions

| 구문                  | 의미                   |
| --------------------- | ---------------------- |
| `{ stmts; expr }`     | Block                  |
| `fn(x) expr`          | Lambda                 |
| `case e { pat -> e }` | Pattern matching       |
| `handle e`            | Effect handling        |
| `x.f`                 | UFCS (괄호 생략)       |
| `x.f(y)`              | UFCS                   |
| `T::f(x)`             | Qualified call         |
| `T { f: v }`          | Record construction    |
| `T { ..x, f: v }`     | Record update (spread) |
| `[a, b, c]`           | List literal           |
| `a <> b`              | Concatenation          |
| `a T::<> b`           | Qualified operator     |
| `(+)`, `(T::<>)`      | Operator as function   |

### Patterns

| 패턴               | 의미                           |
| ------------------ | ------------------------------ |
| `42`, `"hi"`, `?a` | Literal (Number, String, Rune) |
| `_`                | Wildcard                       |
| `x`                | Binding                        |
| `Some(x)`          | Variant (positional)           |
| `Ok { value }`     | Variant (named)                |
| `T { f, .. }`      | Record (나머지 무시)           |
| `[a, b, c]`        | List (정확히 일치)             |
| `[h, ..t]`         | List (head + tail)             |
| `{ result }`       | Handler: Request::Done 매칭    |
| `{ Op(x) -> k }`   | Handler: Request::Suspend 매칭 |
