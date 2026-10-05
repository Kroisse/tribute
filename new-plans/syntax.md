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
fn op do let const struct enum ability mod pub use extern case handle resume if as
True False Nil
pkg super self
```

**Note:** `if`는 guard 문법에서만 사용 (독립적인 if expression 없음). `pkg`,
`super`, `self`는 경로의 첫 segment로 쓰는 경로 키워드다.

### Reserved (향후 사용)

```text
type where in
```

**Note:** 대부분의 제어 흐름은 algebraic effect로 처리하므로 예약어를 최소화함

### 키워드 규칙

- 위의 키워드와 예약어는 모두 **strict**하다. 문맥과 관계없이 식별자로 쓸 수
  없다. `let op = 1`, `struct T { type: Int }`, `x.as`는 모두 오류다.
- **Raw identifier** `r#name`은 소문자로 시작하는 이름을 키워드 여부와 관계없이
  식별자로 쓴다. `r#type`은 이름 `type`이고, 키워드가 아닌 이름에도 쓸 수
  있다(`r#x`와 `x`는 같은 이름). 바인딩, 함수, 필드, 경로 segment, `use` 항목
  등 식별자가 오는 모든 자리에 쓸 수 있다. Raw string은 `r#"`로 시작하므로 `#`
  다음 문자로 둘을 구분한다.
- 경로 키워드(`pkg`, `super`, `self`)는 raw identifier가 될 수 없다. 대문자로
  시작하는 키워드(`True`, `False`, `Nil`)도 raw 형식이 없다.
- 새 구문에 필요한 키워드는 **contextual keyword**로 추가한다. 그 구문의 특정
  위치에서만 키워드로 읽고 다른 곳에서는 식별자로 남기므로, 키워드를 추가해도
  기존 코드가 깨지지 않는다. 예약어도 도입할 때 strict로 둘지 contextual로
  바꿀지 정한다.
- 새 strict 키워드가 필요해지면 그때 manifest 단위의 edition으로 도입한다.
  Edition 이전의 코드는 그 단어를 raw identifier로 옮겨 쓸 수 있다.

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
NatLiteral   ::= Magnitude                             // 42, 0xFF, 1_000, 1e10 → Nat
IntLiteral   ::= ('+' | '-') Magnitude                 // +1, -0b1010, -1e3 → Int
FloatLiteral ::= ('+' | '-')? DecDigits '.' DecDigits Exponent? NumSuffix?
                                                       // 1.0, -3.14, 1.5e-3 → Float
Magnitude    ::= DecDigits Exponent? NumSuffix?        // 42, 1_000, 1e10, 42i
               | '0b' BinDigits                        // 0b1010 (binary)
               | '0o' OctDigits                        // 0o777 (octal)
               | '0x' HexDigits                        // 0xc0ffee (hexadecimal)
Exponent     ::= ('e' | 'E') ('+' | '-')? '_'* DecDigits // 10진 리터럴에만
NumSuffix    ::= 'n' | 'i' | 'f'                       // Nat / Int / Float

// '_'는 첫 문자가 아닌 곳이면 어디든 올 수 있다. 숫자는 최소 하나 있어야 한다.
DecDigits  ::= Digit (Digit | '_')*
BinDigits  ::= '_'* BinDigit (BinDigit | '_')*
OctDigits  ::= '_'* OctDigit (OctDigit | '_')*
HexDigits  ::= '_'* HexDigit (HexDigit | '_')*

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

StringContent     ::= TextChar | StringEscape | StringInterpolation
BytesContent      ::= ByteChar | BytesEscape | BytesInterpolation
RawStringContent  ::= RawChar | StringInterpolation       // escape 처리 안 함
RawBytesContent   ::= RawChar | BytesInterpolation        // escape 처리 안 함
RawChar           ::= Any

StringEscape  ::= '\' ('n' | 'r' | 't' | '0' | '"' | '\' | 'x' AsciiHex) | UnicodeEscape
BytesEscape   ::= '\' ('n' | 'r' | 't' | '0' | '"' | '\' | 'x' HexDigit{2})
AsciiHex      ::= '0'..'7' HexDigit                 // \x00 ~ \x7F
UnicodeEscape ::= '\' 'u' '{' HexDigit{1,6} '}'    // Unicode scalar value 하나
StringInterpolation ::= '\{' Expression '}'   // Expression은 String 타입
BytesInterpolation  ::= '\{' Expression '}'   // Expression은 Bytes 타입

// Rune (Unicode codepoint)
Rune       ::= '?' (PrintableChar | RuneEscape)
RuneEscape ::= '\' ('n' | 'r' | 't' | '0' | '\' | 'x' HexDigit{2}) | UnicodeEscape

// Bool / Nil (키워드)
Bool       ::= 'True' | 'False'
Unit       ::= 'Nil'

// Literal (Text와 Bytes는 raw 변형 포함)
StringLit  ::= String | RawString
BytesLit   ::= Bytes | RawBytes
Literal    ::= Number | StringLit | BytesLit | Rune | Bool | Unit

Identifier ::= IdentStart (Letter | Digit | '_')*    // 키워드와 예약어 제외
             | 'r#' IdentStart (Letter | Digit | '_')* // raw identifier
IdentStart ::= LowerLetter | '_'
TypeId     ::= UpperLetter (Letter | Digit | '_')*   // 타입명은 대문자 시작
```

**Note:** `#` 개수는 양쪽이 일치해야 함. 내부에 `"#`이 포함된 경우 `##`로 감싸면 됨.

**숫자 리터럴 규칙:**

- 리터럴의 모양이 기본 타입을 정한다. 소수점이 있으면 Float, 부호(`+`/`-`)가
  있으면 Int, 둘 다 없으면 Nat이다. 지수만으로는 Float가 되지 않는다: `1e10`은
  Nat, `-1e3`은 Int, `1.0e10`은 Float이다.
- 소수점이 없는 리터럴의 음수 지수(`1e-3`)는 정수로 표현할 수 없으므로 오류다.
  Float가 필요하면 `1.0e-3`이나 `1e-3f`로 쓴다.
- 지수는 10진 리터럴에만 붙는다. 16진 리터럴에서 `e`는 숫자다(`0x1e`는 30).
- 접미사는 10진 리터럴에만 붙으며 기본 타입을 바꾼다. 2진·8진·16진 리터럴은
  진법 접두사 뒤가 전부 숫자이고 접미사를 받지 않는다. 그래서 `0xff`의 끝
  `f`는 언제나 숫자이고, 새 접미사가 16진 숫자(`a`–`f`)와 부딪히지 않는다.
  Int가 필요하면 부호를 붙인다(`+0xFF`).
  - `n`(Nat): 부호도 소수점도 없는 리터럴에만 붙는다. `-1n`, `1.5n`은 오류다.
  - `i`(Int): 정수 리터럴에 붙는다. `42i`, `1e3i`는 Int이고 `1.5i`는 오류다.
  - `f`(Float): `42f`, `1e-3f`는 Float이다.
- 리터럴 바로 뒤에 붙은 식별자 문자(`[0-9A-Za-z_]`)는 lex 단계에서 모두
  리터럴의 일부다. 정의되지 않은 접미사(`42u8`)나 진법에 맞지 않는
  숫자(`0b102`)는 lexical error다. 따라서 접미사를 새로 추가해도 기존 코드의
  의미가 바뀌지 않는다.

**Escape 규칙:**

- `\u{…}`는 1~6자리 16진수로 Unicode scalar value 하나를 나타낸다. Surrogate
  (`D800`–`DFFF`)나 `10FFFF`를 넘는 값은 lexical error다. 고정 폭 4자리 형식은
  없다.
- String의 `\xHH`는 ASCII 범위(`00`–`7F`)만 허용한다. 그 밖의 문자는
  `\u{…}`로 쓴다. Rune의 `\xHH`는 `U+0000`–`U+00FF`를 나타낸다.
- Bytes는 `\u{…}`를 받지 않는다. 임의의 byte는 `\xHH`(`00`–`FF`)로 쓰고,
  UTF-8 문자열이 필요하면 byte를 직접 나열한다.
- 위에 정의되지 않은 escape는 lexical error다. `\` 바로 뒤의 줄바꿈도 오류이며,
  줄 이어 붙이기 용도로 남겨 둔다.

**줄바꿈 정규화:**

String과 Bytes 리터럴 소스의 CRLF와 단독 CR은 값에서 LF 하나가 된다. 따라서
리터럴의 값은 소스 파일의 줄바꿈 방식에 따라 달라지지 않는다. CR이 필요하면
`\r` escape를 쓴다(raw 리터럴에서는 쓸 수 없다).

**블록 리터럴과 들여쓰기 제거:**

`#`로 감싼 String·Bytes 리터럴(`#"`, `s#"`, `r#"`, `rs#"`, `b#"`, `rb#"` 등)은
여는 구분자 바로 뒤에 공백·탭만 있고 줄이 바뀌면 **블록 리터럴**이다. 블록
리터럴은 코드의 들여쓰기가 값에 섞이지 않도록 다음 규칙으로 해석한다. 블록
리터럴이 아닌 리터럴의 내용은 소스 그대로다.

- 여는 구분자 뒤의 공백과 첫 줄바꿈은 값에 들어가지 않는다.
- 닫는 구분자는 자기 줄에 있어야 한다. 그 줄에서 닫는 구분자 앞에는 공백과
  탭만 올 수 있고, 이 공백이 리터럴의 **들여쓰기 접두사**다. 닫는 구분자의 줄과
  그 앞의 줄바꿈은 값에 들어가지 않는다. 값이 줄바꿈으로 끝나야 하면 닫는
  구분자 앞에 빈 줄을 둔다.
- 그 사이의 각 줄은 들여쓰기 접두사로 시작해야 하며, 접두사를 뺀 나머지가
  값이 된다. 접두사는 문자 단위로 같아야 하므로 탭과 공백을 다르게 섞은 줄도
  오류다. 공백과 탭만 있는 줄은 예외로, 접두사로 시작하지 않으면 빈 줄이 된다.
- 판정은 escape와 interpolation을 처리하기 전의 소스 텍스트로 한다. `\t`나
  `\u{20}` 같은 escape는 들여쓰기가 아니라 내용이다. Interpolation `\{…}` 안의
  소스는 여러 줄에 걸쳐 있어도 줄 판정에 참여하지 않고, interpolation 값에 든
  줄바꿈도 들여쓰기에 영향을 주지 않는다.
- 줄 끝의 공백은 그대로 값에 남는다.

닫는 구분자의 위치로 남길 들여쓰기를 정할 수 있다:

```rust
fn query() -> String {
    s#"
        SELECT *
          FROM t
        "#          // "SELECT *\n  FROM t"
}

fn indented() -> String {
    s#"
        SELECT *
      "#            // "  SELECT *"
}
```

**String 리터럴 예시:**

```rust
"hello"                     // 기본
"Hello, \{name}!"           // interpolation
s#"
    multiline
    string
    "#                      // 블록 리터럴: "multiline\nstring"
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
?a          // 'a'
?Z          // 'Z'
?\n         // newline
?\t         // tab
?\x41       // 'A' (hex)
?\u{41}     // 'A' (unicode)
?\u{3042}   // 'あ' (unicode)
?\u{1F600}  // '😀' (unicode, BMP 밖)
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
1.5e-3       // 0.0015
6.02E23

// 숫자 구분자
1_000_000
0xFF_FF
0b1010_1010

// 지수 (소수점이 없으면 정수)
1e10         // Nat: 10000000000
2E+3         // Nat: 2000
-1e3         // Int: -1000

// 접미사
42i          // Int
42f          // Float
1e-3f        // Float: 0.001
+0xFF        // Int: 255 (진법 리터럴은 접미사 대신 부호)

// 오류
1e-3         // 소수점 없는 음수 지수 → 1.0e-3 또는 1e-3f
42u8         // 정의되지 않은 접미사
0b102        // 2진 리터럴에 맞지 않는 숫자
1.5i         // Float 리터럴에 Int 접미사
0xFFi        // 진법 리터럴에는 접미사 없음 → +0xFF

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
    name: String
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

UsePath ::= PathStart ('::' PathSegment)* UseTree?

UseTree ::= '::' '{' UseItem (',' UseItem)* ','? '}'
          | 'as' Identifier

UseItem ::= Identifier ('as' Identifier)?
```

**예시:**

```rust
use std::collections::List
use std::collections::{List, Option, Result}
use std::io::Error as IoError
```

### Module Declaration

```ebnf
ModDecl ::= 'pub'? 'mod' TypeId '{' Item* '}'
```

**예시:**

```rust
pub mod Option {
    pub fn map(opt: Option(a), f: fn(a) ->{g} b) ->{g} Option(b) { ... }
}
```

### Item Paths

```ebnf
Path ::= PathStart ('::' PathSegment)*
PathStart ::= PathKeyword | PathSegment        // 경로 키워드는 첫 segment에만
PathKeyword ::= 'pkg' | 'super' | 'self'
PathSegment ::= Identifier | TypeId
PathPrefix ::= PathStart '::' (PathSegment '::')*
ValuePath ::= PathPrefix? Identifier
TypePath ::= PathPrefix? TypeId
```

선언 이름은 단일 `Identifier` 또는 `TypeId`지만, 참조 위치의 CST는 단일
이름도 각각 `value_path` 또는 `type_path`로 감싼다. 따라서 `print`와
`std::io::print`, `String`과 `std::io::Error`는 세그먼트 수와 무관하게 같은
종류의 path node로 처리된다.

**예시:**

```rust
List::prepend(1, [])
Option::Some(42)
std::io::print_line("hello")
```

---

## Constant Declaration

```ebnf
ConstDecl ::= 'pub'? 'const' Identifier (':' Type)? '=' ConstExpr

ConstExpr ::= Literal                        // -1은 Int 리터럴로 처리
            | ValuePath                      // 다른 상수 참조
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
    name: String
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
    Var(String)
    BinOp { op: String, lhs: Expr, rhs: Expr }
}
```

---

## Type Syntax

```ebnf
Type ::= TypePath TypeArgs?
       | FunctionType
       | TupleType

TupleType ::= '#(' TypeList ')'               // #(Int, String, Float)

TypePath ::= PathPrefix? TypeId
TypeArgs ::= '(' Type (',' Type)* ','? ')'

FunctionType ::= 'fn' '(' TypeList? ')' ReturnType
TypeList ::= Type (',' Type)* ','?

ReturnType ::= '->' Type                      // 암묵적 effect polymorphic
             | '->' '{' EffectRow? '}' Type   // 명시적 effect

EffectRow ::= EffectItem (',' EffectItem)* EffectTail? ','?
EffectItem ::= TypePath TypeArgs?
EffectTail ::= ',' LowerIdentifier            // row variable
```

**예시:**

```rust
Nat                           // 0, 양수
Int                           // 정수 (부호 있음)
Float                         // 부동소수점
String
List(Int)
Option(String)
Result(Int, String)

#(Int, String)                  // 2-tuple (pair)
#(Int, String, Float)           // 3-tuple
Nil                           // unit type (빈 튜플 `#()`은 없다)

fn(Int, Int) -> Int           // 암묵적 effect polymorphic
fn(Int) ->{} Int              // 순수 함수
fn(String) ->{Http} Response    // Http effect
fn() ->{State(Int), e} Int    // State + row variable e
```

---

## Ability (Effect) System

### Ability Declaration

```ebnf
AbilityDecl ::= 'ability' TypeId TypeParams? '{' AbilityOp* '}'

AbilityOp ::= 'fn' Identifier '(' ParamList? ')' '->' Type
            | 'op' Identifier '(' ParamList? ')' '->' Type
```

Ability operation은 두 종류로 선언한다:

- **`fn`**: Tail-resumptive. Handler에서 반환값이 자동으로 resume 값이 된다.
  Continuation을 캡처하지 않는다.
- **`op`**: General. Handler에서 `resume` 키워드를 사용하여 명시적으로
  continuation을 호출한다. `-> Never`를 반환하면 절대 resume하지 않는
  abort 패턴을 표현한다.

**예시:**

```rust
ability Logger {
    fn log(msg: String) -> Nil
}

ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}

ability Http {
    fn get(url: String) -> Response
    fn post(url: String, body: String) -> Response
}

ability Fail {
    op fail(msg: String) -> Never
}
```

### Handle Expression

```ebnf
HandleExpr ::= 'handle' Expression '{' HandlerArm+ '}'

HandlerArm ::= CompletionArm | FnHandlerArm | OpHandlerArm

CompletionArm  ::= 'do' Identifier Block                        // do result { body }
FnHandlerArm   ::= 'fn' ValuePath '(' PatternList? ')' Block   // fn Op(args) { body }
OpHandlerArm   ::= 'op' ValuePath '(' PatternList? ')' Block   // op Op(args) { body }
```

`handle expr { arms }`는 computation을 실행하고 handler arm으로 결과를 처리한다.

Handler arm은 함수 정의와 대칭적인 구조를 가진다. Ability 선언에서 `fn`/`op`으로
operation을 정의하듯, handler에서도 같은 키워드로 각 operation의 구현을 작성한다:

| arm                    | 대상           | 의미                                               |
| ---------------------- | -------------- | -------------------------------------------------- |
| `do value { expr }`    | completion     | Computation 완료, 결과값 바인딩 (생략 시 identity) |
| `fn Op(args) { body }` | `fn` operation | Tail-resumptive: body의 반환값이 resume 값         |
| `op Op(args) { body }` | `op` operation | body에서 `resume` 키워드로 명시적 resume           |

**`fn` handler arm:**

Body의 반환값이 곧 resume 값. `resume` 사용 불가:

```rust
fn Logger::log(msg) { print_line(msg) } // Nil 반환 → resume Nil
```

**`op` handler arm과 `resume`:**

`resume`은 키워드로, computation을 재개하는 함수처럼 호출한다.
`op -> T` handler body에서 `resume`은 최대 1회 호출할 수 있다 (affine).
호출하지 않으면 continuation은 암묵적으로 drop된다:

```rust
op State::get() { resume current_state }
op State::set(v) { resume Nil }
```

항상 resume하지 않는 operation은 `-> Never`로 선언하면
continuation 캡처 자체를 생략하는 최적화가 가능하다.

**`op -> Never` (abort 패턴):**

`-> Never`를 반환하는 operation의 handler body에서는 `resume`을 사용할 수 없다:

```rust
op Fail::fail(msg) { None }
```

**예시:**

```rust
fn run_state(comp: fn() ->{e, State(s)} a, state: s) ->{e} a {
    handle comp() {
        do result { result }
        op State::get() { run_state(fn() resume state, state) }
        op State::set(v) { run_state(fn() resume Nil, v) }
    }
}

fn run_logger(comp: fn() ->{e, Logger} a) ->{e, Io} a {
    handle comp() {
        do result { result }
        fn Logger::log(msg) { print_line(msg) }
    }
}

fn run_maybe(comp: fn() ->{e, Fail} a) ->{e} Option(a) {
    handle comp() {
        do result { Some(result) }
        op Fail::fail(msg) { None }
    }
}
```

---

## Functions

### Function Definition

```ebnf
FunctionDef ::= 'pub'? 'fn' Identifier '(' TypedParamList? ')' ReturnType Block

TypedParamList ::= TypedParam (',' TypedParam)* ','?
TypedParam ::= Identifier ':' Type

ParamList ::= Param (',' Param)* ','?
Param ::= Identifier (':' Type)?

// ReturnType은 Type Syntax 섹션에 정의됨
```

함수 선언은 파라미터 타입과 반환 타입을 모두 적는다. `main`도 `fn main() -> Nil`로
쓴다. Effect row는 생략할 수 있으며, 생략하면 fresh row 변수이다. 람다의
파라미터 타입은 생략할 수 있다([Lambda Expression](#lambda-expression)).

**예시:**

```rust
fn add(x: Int, y: Int) -> Int {
    x + y
}

fn fetch(url: String) ->{Http} Response {
    Http::get(url)
}

fn example() ->{State(Int), Io} Nil {
    let n = State::get()
    print_line(Int::to_string(n))
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
              | ValuePath
              | Block                             // { expr } 로 그룹화
              | ListExpr
              | TupleExpr
              | RecordExpr
              | OperatorFn
              | Lambda
              | CaseExpr
              | HandleExpr
              | ResumeExpr

ListExpr ::= '[' ExprList? ']'
TupleExpr ::= '#(' ExprList ')'           // #(1, "hello", 3.14)
OperatorFn ::= '(' Operator ')'           // (+), (<>)
             | '(' QualifiedOp ')'        // (Int::+), (String::<>)
ResumeExpr ::= 'resume' Expression?            // op handler body 전용 (affine, 생략 시 Nil)
```

List literal은 canonical opaque `List(a)`를 만든다. 각 element expression은 source
순서대로 왼쪽에서 오른쪽으로 정확히 한 번 평가되며, 내부 persistent representation을
구성하기 위한 reverse fold가 이 evaluation order를 바꾸거나 재평가해서는 안 된다.
Runtime value를 기존 tail 앞에 붙이는 최소 public API는
`List::prepend(value, tail)`이다. 이 함수는 새 canonical List를 반환하며 tail을
변경하지 않는다. `Empty`와 `Cons`는 List syntax의 constructor가 아니다.

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

**Note:** `(expr)` 형태의 괄호 표현식은 지원하지 않음. 우선순위 조정에는
`{ expr }` 사용. `(...)` 는 연산자-함수 `(+)`, `(<>)` 전용.

### Record Expression

```ebnf
RecordExpr ::= TypePath '{' RecordFields? '}'
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
VariantExpr ::= TypePath '(' ExprList? ')'     // positional
              | TypePath '{' RecordFields '}'   // named
              | TypePath                         // unit variant
```

**예시:**

```rust
Some(42)
None
Branch(Leaf(1), Leaf(2))
Ok { value: 42 }
Error { error: "failed" }
```

### Call and UFCS

```ebnf
CallExpr ::= Expression '(' ExprList? ')'
UFCSExpr ::= Expression '.' ValuePath CallArgs?
CallArgs ::= '(' ExprList? ')'
           | /* empty - 인자 없으면 괄호 생략 가능 */

ExprList ::= Expression (',' Expression)* ','?
```

**UFCS 규칙:**

- `x.f(y)` → `f(x, y)` 또는 `T::f(x, y)` (타입 T에서 해소)
- `x.f` → `f(x)` (인자가 없으면 괄호 생략 가능)
- `x.a::b(y)` → `a::b(x, y)` (qualified path도 가능)

Qualified UFCS `x.a::b(y)`도 비한정 UFCS처럼 receiver 타입으로 후보를 고른다
([modules.md](modules.md#type-directed-resolution)). 타입에는 namespace가 없고
`User::name::set`의 `User`는 타입과 동명인 모듈이므로, 후보는 호출 위치 스코프의
경로로만 찾는다:

- 스코프에서 그대로 해소되는 `a::b`
- 스코프 안의 모듈 `m`마다 `m::a::b`

이 중 첫 매개변수가 receiver 타입과 맞는 후보가 하나면 그것을 쓰고, 여럿이면
모호성 오류, 없으면 경로 `a::b`에 대한 미해소 진단을 원래 위치에 낸다. 스코프
밖의 모듈은 receiver 타입을 정의한 모듈이라도 찾지 않으므로, 다른 모듈의 타입에
쓰려면 그 동명 모듈을 `use`한다(`use m::User`).

**예시:**

```rust
// 일반 호출
add(1, 2)
Option::map(opt, fn(x) x + 1)

// UFCS - 단순 식별자
opt.map(fn(x) x + 1)     // Option::map(opt, ...)
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
(String::<>)("a", "b")            // "a" <> "b" 와 동일

// 고차 함수에 전달
xs.fold(0, (+))                 // 합계
xs.fold("", (String::<>))         // 문자열 연결
numbers.reduce((Int::*))        // 곱셈
```

`(Int::+)`는 `Int`의 `+` 선언을 가리키는 함수 값이다. 지역 변수에 바인딩하고
별칭을 통해 고차 함수에 전달할 수도 있다.

```rust
fn apply(f: fn(Int, Int) ->{} Int, x: Int, y: Int) ->{} Int { f(x, y) }

fn main() -> Nil {
    let add = (Int::+)
    let alias = add
    let sum = apply(alias, +1, +2)    // +3
}
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

`case`는 망라적이어야 한다. 망라성 검사와 도달 불가 arm 규칙은
[types.md의 망라성과 도달 불가 arm](types.md#망라성과-도달-불가-arm)을 따른다.

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

LiteralPattern ::= Number | StringLit | BytesLit | Rune | 'True' | 'False' | 'Nil'
WildcardPattern ::= '_'
IdentifierPattern ::= Identifier
VariantPattern ::= TypePath ('(' PatternList? ')' | '{' RecordPatternFields? '}')?
RecordPattern ::= VariantPattern                // 중괄호 형식; 필드를 이름으로 매칭
ListPattern ::= '[' ListPatternItems? ']'
ListPatternItems ::= Pattern (',' Pattern)* (',' RestPattern)? ','?  // [head, ..tail] or [head, ..]
                   | RestPattern ','?                                // [..tail]
RestPattern ::= '..' Identifier?
TuplePattern ::= '#(' PatternList ')'

PatternList ::= Pattern (',' Pattern)* ','?
RecordPatternFields ::= RecordPatternField (',' RecordPatternField)* (',' '..')? ','?
                      | '..' ','?                   // Name { .. }
RecordPatternField ::= Identifier (':' Pattern)?
```

중괄호 필드의 이름 매칭, `..`, 필드 수 규칙은
[types.md의 패턴 필드 검증](types.md#패턴-필드-검증)을 따른다.

**Note:** Handler arm (`do`, `fn`, `op`)은 `handle`
표현식 내에서만 사용된다. 일반 `case` 표현식에서는 사용할 수 없다.

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
Branch(left, right)
Ok { value }
Error { error: e }
std::io::Error::EndOfFile
std::io::ReadLineResult::Line(bytes)

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

List patterns are sequence views over the opaque canonical `List(a)`:

- `[]` matches only length zero.
- `[p1, ..., pn]` matches exactly length `n`.
- `[p1, ..., pn, ..tail]` matches length at least `n` and binds `tail` to
  the remaining `List(a)` without mutation or element reordering.
- `[p1, ..., pn, ..]` has the same prefix requirement and ignores the remainder.

Pattern lowering must use representation-independent shared list observation
operations. It must not synthesize public variant patterns or expose target
field offsets in frontend/shared IR.

---

## Field Access and Update

### Getter (자동 생성)

```rust
// struct 필드는 자동으로 getter 생성
struct User { name: String, age: Nat }

// 생성되는 함수:
// User::name : fn(User) -> String
// User::age  : fn(User) -> Nat

user.name    // User::name(user)
user.age     // User::age(user)
```

### Setter and Modifier

Struct 필드에 대해 자동 생성되는 함수들 (별도 문법 없음, UFCS로 호출):

```rust
// 자동 생성되는 함수:
// User::name::set    : fn(User, String) -> User
// User::name::modify : fn(User, fn(String) -> String) -> User

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
    name: String
    age: Nat
}

// 한 줄이면 쉼표 필수
struct User { name: String, age: Nat }
```

### 세미콜론

세미콜론은 선택적이지만, 한 줄에 여러 문장을 쓸 때 필요:

```rust
let x = 1; let y = 2; x + y
```

---

## Complete Example

이 예제의 `List::filter`, `List::map`, `List::each`는 complete-program 구성을
보여 주기 위한 illustrative API이며 별도 public API 절에서 확정되지 않았다.

```rust
use std::collections::{List, Option}
use std::io::{Io, print_line}

struct User {
    name: String
    age: Nat
}

enum Status {
    Active
    Inactive { reason: String }
}

ability Logger {
    fn log(msg: String) -> Nil
}

pub fn greet(user: User) ->{Io} Nil {
    let greeting = "Hello, " <> user.name <> "!"
    print_line(greeting)
}

fn process(users: List(User)) ->{Logger} List(String) {
    users
        .filter(fn(u) u.age >= 18)
        .map(fn(u) {
            Logger::log("Processing: " <> u.name)
            u.name
        })
}

fn run_logger(comp: fn() ->{e, Logger} a) ->{e, Io} a {
    handle comp() {
        do result { result }
        fn Logger::log(msg) { print_line("[LOG] " <> msg) }
    }
}

fn main() ->{Io} Nil {
    let users = [
        User { name: "Alice", age: 30 },
        User { name: "Bob", age: 17 },
        User { name: "Charlie", age: 25 },
    ]

    run_logger(fn() {
        let names = process(users)
        names.each(fn(name) {
            print_line("Name: " <> name)
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

| 구문                                 | 의미                           |
| ------------------------------------ | ------------------------------ |
| `Nat`, `Int`, `Float`, `String`, ... | 기본 타입                      |
| `List(a)`, `Option(Int)`             | 제네릭 타입                    |
| `#(Int, String)`                     | Tuple 타입                     |
| `fn(a) -> b`                         | 함수 타입 (암묵적 polymorphic) |
| `fn(a) ->{} b`                       | 명시적 빈 effect 함수 타입     |
| `fn(a) ->{E} b`                      | Effect E를 수행하는 함수       |
| `fn(a) ->{E, e} b`                   | E + row variable e             |

### Expressions

| 구문                  | 의미                   |
| --------------------- | ---------------------- |
| `{ stmts; expr }`     | Block / 그룹화         |
| `fn(x) expr`          | Lambda                 |
| `case e { pat -> e }` | Pattern matching       |
| `handle e { arms }`   | Effect handling        |
| `x.f`                 | UFCS (괄호 생략)       |
| `x.f(y)`              | UFCS                   |
| `T::f(x)`             | Qualified call         |
| `T { f: v }`          | Record construction    |
| `T { ..x, f: v }`     | Record update (spread) |
| `[a, b, c]`           | List literal           |
| `#(a, b, c)`          | Tuple literal          |
| `a <> b`              | Concatenation          |
| `a T::<> b`           | Qualified operator     |
| `(+)`, `(T::<>)`      | Operator as function   |
| `resume expr`         | Continuation 재개      |

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
| `#(a, b, c)`       | Tuple                          |
| `[h, ..t]`         | List (head + tail)             |
| `pat as x`         | As (전체 바인딩)               |

### Handler Arms (handle 전용)

| arm                  | 의미                               |
| -------------------- | ---------------------------------- |
| `do result { expr }` | Completion (생략 시 identity)      |
| `fn Op(x) { body }`  | `fn` operation (tail-resumptive)   |
| `op Op(x) { body }`  | `op` operation (explicit `resume`) |
