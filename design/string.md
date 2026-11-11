# String and Bytes

Tribute는 텍스트와 바이트 데이터를 명확히 구분한다.

## 개요

| 타입 | 설명 | 리터럴 |
| ---- | ---- | ------ |
| `Bytes` | raw bytes, 인코딩 가정 없음 | `b"..."` |
| `String` | Unicode 텍스트 (UTF-8 rope) | `"..."` |
| `Rune` | 단일 Unicode codepoint | `?a`, `?\n` |

**설계 원칙:**

- `Bytes`는 byte 배열의 한 범위를 보는 primitive 타입
- `String`은 UTF-8 인코딩된 텍스트 (내부적으로 rope 구조)
- 인코딩 변환은 명시적으로 수행

---

## Bytes

### 내부 표현

`Bytes`는 `Int`, `Nat`처럼 언어가 정한 primitive 타입이다. Field가 없으므로 레코드
생성, spread, 레코드 패턴의 대상이 아니며, 값은 literal과 prelude 함수로만 만든다.
표현은 [primitive 타입의 표현](types.md#primitive-타입의-표현) 규칙에 따라 prelude
source의 `Bytes::impl`이 정한다.

```rust
pub mod Bytes {
    pub(pkg) mod impl {
        struct Repr { data: ByteArray, start: Nat, len: Nat }

        extern "intrinsic" fn repr(bytes: Bytes) -> Repr
        extern "intrinsic" fn from_repr(repr: Repr) -> Bytes

        pub(pkg) fn data(bytes: Bytes) -> ByteArray { repr(bytes).data }
        pub(pkg) fn start(bytes: Bytes) -> Nat { repr(bytes).start }
        pub(pkg) fn len(bytes: Bytes) -> Nat { repr(bytes).len }
        pub(pkg) fn new(data: ByteArray, start: Nat, len: Nat) -> Bytes { .. }
    }
}
```

`std` 안의 다른 코드는 `bytes.impl::len()`처럼 `Bytes::impl`의 함수로 표현을 읽고
만든다. `Repr`과 두 intrinsic은 `Bytes::impl` 밖에서 보이지 않는다.

Byte는 compiler 소유 불변 배열 [`std::runtime::ByteArray`](types.md#bytearray)(IR에서
`adt.array<core.i8>`)에 담기고, 값은 그중 `[start, start + len)`을 본다.
불변(immutable)이다.

- 잘라내기는 같은 `data`를 담은 새 `Bytes`를 만들며 byte를 복사하지 않는다.
  소유권과 해제는 일반 struct와 배열의 규칙을 따른다([rc.md](rc.md#bytes)).
- `start + len ≤ len(data)`는 `Bytes` 값의 불변식이다. `Bytes::impl`은 `std` 밖에서
  보이지 않으므로 값을 만들고 범위를 정하는 코드는 모두 `std` 안에 있고, 그 코드가
  불변식을 지킨다. 범위가 배열 밖인 값이 생기더라도 byte 읽기는
  `ByteArray::get`의 범위 검사가 trap으로 끝낸다.

### 리터럴

```rust
b"hello"                    // 기본
b#"contains "quotes""#      // raw
b"\x00\xFF"                 // escape sequences
```

Bytes 리터럴은 `\u{…}`를 받지 않는다. 모든 byte 값은 `\xHH`(`00`–`FF`)로
쓴다.

### 주요 연산

```rust
// 생성
Bytes::empty() -> Bytes
Bytes::from_array(arr: Array(U8)) -> Bytes

// 접근
bytes.len() -> Nat
bytes.get(index: Nat) -> Option(U8)
bytes.slice(start: Nat, end: Nat) -> Bytes

// 연결
bytes1 <> bytes2 -> Bytes   // 새 Bytes 생성

// 비교
bytes1 == bytes2 -> Bool    // byte 내용 비교
bytes1 != bytes2 -> Bool

// 변환
bytes.to_array() -> Array(U8)
bytes.to_string_utf8() -> Result(String, DecodeError)
bytes.to_string_lossy() -> String  // invalid → U+FFFD
```

---

## String

### 내부 표현

UTF-8 rope 구조:

```rust
enum String {
    Leaf(Bytes)                      // 짧은 텍스트 (UTF-8 validated)
    Branch(String, String, Nat)      // left, right, total_len
}
```

**Rope의 장점:**

- O(log n) concatenation, insert, delete
- 큰 텍스트도 효율적으로 처리
- 부분 문자열 공유 가능

`String`은 prelude에 정의된 library enum이다. 컴파일러는 문자열 리터럴과
interpolation을 위해 canonical prelude identity를 well-known type으로 인식하지만,
별도의 compiler-owned `String` 타입이나 target 전용 source-visible 타입을 만들지
않는다.

```tribute
enum String {
    Leaf(Bytes)
    Branch(String, String, Nat)
}
```

Backend는 이 ADT의 source variant와 field 순서를 보존해야 한다. 특히 문자열
리터럴을 `core.string`으로 위장한 integer pointer 같은 별도 표현으로 lowering하면
안 된다.

### 리터럴

```rust
"hello"                     // 기본
"Hello, \{name}!"           // interpolation → String
s#"
    multiline
    text
    "#                      // 블록 리터럴: "multiline\ntext"
r"\d+\.\d+"                 // raw (escape 없음)
"caf\u{E9} \u{1F600}"       // unicode escape (1~6자리 16진수)
```

`\u{…}`는 Unicode scalar value 하나이며 surrogate와 `10FFFF` 초과 값은 lexical
error다. `\xHH`는 ASCII(`00`–`7F`)만 허용한다.

블록 리터럴은 닫는 구분자 앞의 공백을 들여쓰기 접두사로 삼아 모든 줄에서
제거하고, 소스의 줄바꿈은 LF로 정규화한다. 자세한 규칙은
[syntax.md](syntax.md)의 "블록 리터럴과 들여쓰기 제거"를 따른다.

### 주요 연산

```rust
// 생성
String::empty() -> String
String::from_rune(r: Rune) -> String

// 길이
s.len() -> Nat           // byte 길이
s.rune_count() -> Nat    // codepoint 수

// 연결
s1 <> s2 -> String

// 순회 (Stream ability 사용)
s.runes() ->{Stream(Rune)} Nil
s.chars() ->{Stream(Rune)} Nil  // alias

// Slice (codepoint 단위)
s.slice(start: Nat, end: Nat) -> String
s.take(n: Nat) -> String
s.drop(n: Nat) -> String

// 검색
s.contains(pattern: String) -> Bool
s.starts_with(prefix: String) -> Bool
s.ends_with(suffix: String) -> Bool
s.find(pattern: String) -> Option(Nat)

// 변환
s.to_bytes() -> Bytes    // UTF-8 bytes
s.to_upper() -> String
s.to_lower() -> String
s.trim() -> String
s.split(sep: String) -> List(String)
s.lines() -> List(String)

// 비교
s1 == s2 -> Bool      // logical UTF-8 bytes의 equality (rope shape과 무관)
s1 != s2 -> Bool      // ==의 logical complement
String::compare(a: String, b: String) -> Ordering
```

String equality는 cached rope metadata, variant identity, 또는 pointer identity가
아니라 전체 logical UTF-8 byte sequence를 비교한다. 따라서 `String::Leaf(b"ab")`와
`String::Branch(String::Leaf(b"a"), String::Leaf(b"b"), 2)`는 같다.
Flattening, traversal, allocation 전략은 일반적으로 library implementation
detail이지만, canonical prelude 구현은 equality를 위해 전체 rope를 flatten하지
않는다. 양쪽 rope의 leaf를 순서대로 lockstep 순회하면서 현재 leaf의 남은 길이
중 더 짧은 연속 byte span을 비교하고, 소진된 쪽 cursor를 다음 non-empty leaf로
전진시킨다. 연속 span 비교는 target primitive로 lowering되어 Tribute-level
byte별 함수 호출을 만들지 않는다.

먼저 root의 cached byte length를 비교하여 길이가 다르면 O(1)에 실패할 수 있다.
길이가 같고 양쪽 root가 모두 `Leaf`이면 전체 leaf를 하나의 연속 span으로 정확히 한
번 비교한다. 이 흔한 경로는 cursor를 만들지 않고 auxiliary allocation도 하지 않는다.

한쪽이라도 `Branch`이면 cursor fallback을 사용한다. 처음 다른 byte에서 early
exit하며, 모두 같을 때 시간 복잡도는 O(n + r1 + r2)이다. 여기서 `n`은 logical byte
length이고 `r1`, `r2`는 방문한 rope node 수이다. Equality는 어느 경로에서도 byte
length에 비례하는 payload allocation을 하지 않는다. Fallback cursor node의 총
allocation은 방문한 rope node 수에 대해 선형이고, 동시에 살아 있는 cursor heap
state는 각 rope 깊이에 비례한다. 현재 source-level recursive comparison driver를
일반 tail call로 변환하지 않는 target에서는 비교한 leaf span 수에 비례하는 control
stack도 사용할 수 있다.

---

## Rune

단일 Unicode codepoint (U+0000 ~ U+10FFFF, surrogate 제외).

### 리터럴

```rust
?a                          // 'a' (U+0061)
?\n                         // newline (U+000A)
?\t                         // tab (U+0009)
?\\                         // backslash
?\x41                       // hex (U+0041 = 'A')
?\u{41}                     // unicode (U+0041 = 'A')
?\u{1F600}                  // unicode (U+1F600, BMP 밖)
```

### 내부 표현

```rust
Rune = u32  // 실제로는 21비트면 충분하지만 정렬상 32비트
```

### 주요 연산

```rust
rune.to_string() -> String
rune.to_bytes() -> Bytes    // UTF-8 인코딩
rune.codepoint() -> Nat     // U+XXXX 값

// 분류
rune.is_alphabetic() -> Bool
rune.is_numeric() -> Bool
rune.is_whitespace() -> Bool
rune.is_uppercase() -> Bool
rune.is_lowercase() -> Bool

// 변환
rune.to_upper() -> Rune
rune.to_lower() -> Rune
```

---

## Stream Ability

순회는 `Stream` ability로 표현한다. Producer가 `emit`을 호출하고,
handler가 소비 방식을 결정한다:

```rust
ability Stream(a) {
    op emit(value: a) -> Nil
}
```

Handler가 `resume` 여부를 선택할 수 있으므로 early termination도 자연스럽다.

### String::runes

`String`에서 `Rune` 시퀀스를 lazy하게 순회:

```rust
pub fn runes(s: String) ->{Stream(Rune)} Nil {
    // 각 leaf의 UTF-8 bytes를 디코딩하며 emit
    case s {
        .Leaf(bytes) -> emit_runes_from_bytes(bytes)
        .Branch(left, right, _) -> {
            left.runes()
            right.runes()
        }
    }
}
```

### 사용 예시

```rust
// 모든 rune 순회
fn count_vowels(s: String) -> Nat {
    let count = 0
    handle s.runes() {
        op Stream::emit(r) {
            case "aeiouAEIOU".contains(r.to_string()) {
                True -> resume({ count = count + 1; Nil })
                False -> resume(Nil)
            }
        }
    }
    count
}

// Early termination: 첫 공백 전까지만
fn first_word(s: String) -> String {
    let result = String::empty()
    handle s.runes() {
        op Stream::emit(r) {
            case r.is_whitespace() {
                True -> result       // resume 안 함 → 중단
                False -> {
                    result = result <> String::from_rune(r)
                    resume(Nil)
                }
            }
        }
    }
    result
}
```

---

## 인코딩 변환

Bytes ↔ String 변환은 명시적으로 인코딩을 지정:

```rust
// Bytes → String
bytes.to_string_utf8() -> Result(String, DecodeError)
bytes.to_string_utf16_le() -> Result(String, DecodeError)
bytes.to_string_utf16_be() -> Result(String, DecodeError)
bytes.to_string_latin1() -> String  // 항상 성공

// String → Bytes
s.to_bytes() -> Bytes        // UTF-8 (기본)
s.to_bytes_utf16_le() -> Bytes
s.to_bytes_utf16_be() -> Bytes

// Lossy 변환
bytes.to_string_lossy() -> String   // invalid bytes → U+FFFD
```

### DecodeError

```rust
struct DecodeError {
    offset: Nat
    message: String
}
```

---

## 문자열 Interpolation

`"\{expr}"` 구문은 `expr`이 `String` 타입일 것을 요구:

```rust
let name = "Alice"
let age = 30

// 직접 String인 경우
"Hello, \{name}!"  // OK

// String이 아닌 경우 → 명시적 변환 필요
"Age: \{Nat::to_string(age)}"  // OK
"Age: \{age}"                  // 타입 에러
```

### Display Ability (미래 고려)

```rust
ability Display {
    fn display(self) -> String
}

// Display가 있으면 자동 변환 가능하도록 할 수도 있음
// 하지만 현재는 명시적 변환 유지
```

---

## WasmGC 구현

### Bytes

```wasm
;; ByteArray는 원소가 i8인 일반 GC 배열이다. 예약 GC 타입 index를 받지 않는다.
(type $byte_array (array (mut i8)))

;; Bytes는 표현 struct Bytes::impl::Repr의 GC 타입이며, 다른 사용자 struct와
;; 같은 structural GC 타입이다.
(type $bytes (sub $described (struct
  (field $descriptor i32)
  (field $data (ref $byte_array))
  (field $start i32)
  (field $len i32))))
```

원소가 scalar인 배열은 erased `arrayref`가 되지 않으므로
([GC 타입 표현](wasm-backend.md#structural-gc-타입과-descriptor-필드)) byte를
읽는 곳은 cast 없이 `array.get_u`로 읽는다.

### String (Rope)

```wasm
;; The ordinary WasmGC representation of the prelude String enum.
;; Concrete indices are assigned with the other ADT types; they are not
;; reserved builtin indices.
(type $string (sub (struct)))

;; Leaf: UTF-8 bytes
(type $string_leaf (sub $string (struct
  (field $bytes (ref $bytes)))))

;; Branch: two children + cached length
(type $string_branch (sub $string (struct
  (field $left (ref $string))
  (field $right (ref $string))
  (field $len i32))))
```

### Constant lowering

`adt.bytes_const`의 결과는 byte 배열 `adt.array<core.i8>`이다. `b"..."` 리터럴은
그 배열 전체를 보는 `Bytes` 표현이고, `"..."` 리터럴은 그것을 다시 `String::Leaf`로
감싼다. `Bytes`의 IR 타입은 `Repr`의 IR 타입이므로 리터럴은 `Repr`의 field를 이름으로
찾아 만든다.

```text
b"..."  ->  Repr { data: adt.bytes_const(bytes), start: 0, len }
"..."   ->  String::Leaf(Repr { data: adt.bytes_const(utf8), start: 0, len })
```

Target은 byte 배열을 다음처럼 만든다. 표현은 의미에 드러나지 않으므로
([runtime-types.md](runtime-types.md#표현은-의미에-드러나지-않는다)) 두 target의
선택은 서로 달라도 된다.

- Native는 literal마다 [정적 객체](rc.md#정적-객체) 하나를 data에 두고, 평가할
  때마다 그 객체를 retain한다. 할당과 byte 복사가 없다.
- Wasm은 평가할 때마다 passive data segment에서 `array.new_data`로 새 배열을
  만든다. `array.new_data`는 상수식이 아니므로 global 초기화식으로 쓸 수 없다.

`Bytes` struct는 두 target 모두 평가마다 일반 할당이다. Wasm에서
`adt.bytes_const`와 `adt.string_const`는 같은 passive data-segment index 공간을
사용하며, 동일한 payload는 `String`과 `Bytes` 리터럴 사이에서도 data segment를
재사용할 수 있다. Constant lowering은 자신이 참조하는 data segment를 같은 단계에서 모듈에
선언한다([모듈 수준 자원의 선언][wasm-resources]). `String::Leaf` 생성은 target의
일반 enum/variant lowering보다 먼저 일어나고, 그 이후에는 source에서 작성한
`.Leaf(bytes)`와 똑같은 ADT 경로를 따른다.

[wasm-resources]: wasm-backend.md#모듈-수준-자원의-선언

Native와 Wasm 모두 이 의미적 경계를 공유한다. Public I/O wrapper도 canonical
`String`을 받아 명시적으로 `Bytes`로 변환한 뒤 target I/O operation에 전달한다.

### Rune

```wasm
;; Rune = i32 (unboxed)
;; 함수 인자/반환은 그냥 i32로 전달
```

---

## 설계 결정 사항

### 결정됨

| 항목 | 결정 |
| ---- | ---- |
| 타입 이름 | `String` (보편적인 이름 원칙) |
| String 인코딩 | UTF-8 |
| String 구조 | Rope |
| Bytes 표현 | primitive 타입, 표현은 byte 배열 위의 `Bytes::impl::Repr { data, start, len }` |
| Bytes 리터럴 | native는 정적 byte 배열 객체, Wasm은 평가마다 `array.new_data` |
| Rune 표현 | u32 |
| 인코딩 변환 | 명시적 |
| `"..."` 리터럴 타입 | String |
| `b"..."` 리터럴 타입 | Bytes |

### 미결정

| 항목 | 선택지 |
| ---- | ------ |
| String byte 인덱싱 | 허용 vs 금지 |
| Display ability | 도입 vs 명시적 변환만 |

### 결정 완료 (이전 미결정)

| 항목 | 결정 |
| ---- | ---- |
| String 정의 위치 | Library type (prelude에 enum으로 정의) |
| String backend 표현 | 일반 ADT layout 사용 (target별 별도 builtin identity 없음) |
| String literal lowering | UTF-8 `Bytes`를 담은 `String::Leaf` |

---

## 참고

- Rust: `&str`/`String` (UTF-8), `&[u8]`/`Vec<u8>`
- Python 3: `str` (Unicode), `bytes`
- Elixir: String (UTF-8 binary), binary
- Go: `string` (UTF-8 bytes), `[]byte`
- Swift: `String` (grapheme clusters), `Data`
