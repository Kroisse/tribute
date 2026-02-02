# String and Bytes

Tribute는 텍스트와 바이트 데이터를 명확히 구분한다.

## 개요

| 타입 | 설명 | 리터럴 |
| ---- | ---- | ------ |
| `Bytes` | raw bytes, 인코딩 가정 없음 | `b"..."` |
| `String` | Unicode 텍스트 (UTF-8 rope) | `"..."` |
| `Rune` | 단일 Unicode codepoint | `?a`, `?\n` |

**설계 원칙:**

- `Bytes`는 primitive type (ptr, len)
- `String`은 UTF-8 인코딩된 텍스트 (내부적으로 rope 구조)
- 인코딩 변환은 명시적으로 수행

---

## Bytes

### 내부 표현

```text
Bytes = (ptr: *const u8, len: usize)
```

연속된 메모리 영역을 가리키는 fat pointer. 불변(immutable).

### 리터럴

```rust
b"hello"                    // 기본
b#"contains "quotes""#      // raw
b"\x00\xFF"                 // escape sequences
```

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

**Library vs Builtin:**

| | Library Type | Builtin Type |
| --- | --- | --- |
| 정의 위치 | 표준 라이브러리 | 컴파일러 내장 |
| Bytes 의존 | String이 Bytes 사용 | 독립적 primitive |
| 확장성 | 사용자 정의 가능 | 컴파일러만 정의 |

어느 쪽이든 컴파일러는 `String`을 "well-known type"으로 인식해야 한다 (리터럴, interpolation 지원).

### 리터럴

```rust
"hello"                     // 기본
"Hello, \{name}!"           // interpolation → String
s#"
    multiline
    text
"#                          // multiline
r"\d+\.\d+"                 // raw (escape 없음)
```

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

// 접근 (byte 단위는 제공하지 않음)
s.runes() -> Iterator(Rune)
s.chars() -> Iterator(Rune)  // alias

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
s1 == s2 -> Bool      // byte-wise equality
String::compare(a: String, b: String) -> Ordering
```

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
?\u0041                     // unicode (U+0041 = 'A')
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

## String::runes - Lazy Rune Stream

`String`에서 `Rune` 시퀀스를 lazy하게 순회:

```rust
fn count_vowels(s: String) -> Nat {
    s.runes()
        .filter(fn(r) "aeiouAEIOU".contains(r.to_string()))
        .count()
}

fn first_word(s: String) -> String {
    s.runes()
        .take_while(fn(r) !r.is_whitespace())
        .collect()  // Iterator(Rune) -> String
}
```

### Iterator 타입

```rust
// String::runes()가 반환하는 타입
struct RuneIterator {
    s: String
    byte_offset: Nat
}

impl Iterator for RuneIterator {
    type Item = Rune

    fn next(self) ->{} Option((Rune, RuneIterator))
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
;; Bytes = (ref $bytes)
(type $bytes (array (mut i8)))

;; 또는 struct로 감싸서 추가 정보 저장
(type $bytes (struct
  (field $data (ref (array i8)))
  (field $offset i32)
  (field $len i32)))
```

### String (Rope)

```wasm
;; Rope node (abstract base)
(type $string (sub (struct)))

;; Leaf: UTF-8 bytes
(type $string_leaf (sub $string (struct
  (field $bytes (ref $bytes)))))

;; Branch: two children + cached length
(type $string_branch (sub $string (struct
  (field $left (ref $string))
  (field $right (ref $string))
  (field $len i32)
  (field $depth i32))))  ;; balancing을 위한 깊이
```

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
| Bytes 표현 | (ptr, len) |
| Rune 표현 | u32 |
| 인코딩 변환 | 명시적 |
| `"..."` 리터럴 타입 | String |
| `b"..."` 리터럴 타입 | Bytes |

### 미결정

| 항목 | 선택지 |
| ---- | ------ |
| String 정의 위치 | Library type vs Builtin type |
| String byte 인덱싱 | 허용 vs 금지 |
| Display ability | 도입 vs 명시적 변환만 |

---

## 참고

- Rust: `&str`/`String` (UTF-8), `&[u8]`/`Vec<u8>`
- Python 3: `str` (Unicode), `bytes`
- Elixir: String (UTF-8 binary), binary
- Go: `string` (UTF-8 bytes), `[]byte`
- Swift: `String` (grapheme clusters), `Data`
