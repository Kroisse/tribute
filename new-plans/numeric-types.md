# Tribute Numeric Types

> 이 문서는 Tribute의 수치 타입과 WasmGC 표현을 정의한다.

## Design Decisions

### 결정 사항 요약

| 항목                | 선택                     | 대안 (채택하지 않음)  |
| ------------------- | ------------------------ | --------------------- |
| Int 표현            | Fixnum/Bignum 하이브리드 | 고정 i64, 순수 BigInt |
| Float 표현          | f64 고정                 | f32, 임의 정밀도      |
| 다형적 컨텍스트 Int | i31ref + BigInt          | BoxedI64              |
| 오버플로우 처리     | 자동 승격 (무한 정밀도)  | 랩어라운드, 에러      |

---

## Int Type

### Fixnum/Bignum 하이브리드

Tribute의 `Int`는 **임의 정밀도 정수**이다:

```
Int = i31ref (fixnum) | BigInt (bignum)
     ├─ -2³⁰ ≤ n < 2³⁰ → i31ref (인라인, 0바이트 오버헤드)
     └─ 그 외 → BigInt (힙 할당, GMP 스타일)
```

### 설계 근거

- **오버플로우 없음**: 수학적으로 순수한 정수 의미론
- **작은 수 최적화**: 대부분의 정수는 31비트 내에서 처리
- **Uniform representation 호환**: i31ref와 BigInt 모두 anyref 서브타입

### WasmGC 표현

**Fixnum (i31ref):**

```wasm
;; 31비트 범위 내 정수는 힙 할당 없이 직접 저장
(ref.i31 (i32.const 42))

;; 타입 검사
(ref.test i31ref (local.get $int_value))
```

**Bignum (BigInt):**

```wasm
(type $BigInt (struct
  (field $size i32)              ;; 부호 포함 크기 (음수면 음의 정수)
  (field $digits (ref $i32_array))))  ;; limbs (little-endian)

;; 예: -12345678901234567890
(struct.new $BigInt
  (i32.const -2)                 ;; 음수, 2개 limb
  (array.new_fixed $i32_array 2
    (i32.const 0xEB1F0AD2)       ;; 하위 32비트
    (i32.const 0xAB54A98C)))     ;; 상위 32비트
```

### 산술 연산

모든 산술 연산은 **자동 승격**을 포함:

```wasm
(func $int_add (param $a anyref) (param $b anyref) (result anyref)
  ;; Fast path: 둘 다 fixnum인 경우
  (if (result anyref)
    (i32.and
      (ref.test i31ref (local.get $a))
      (ref.test i31ref (local.get $b)))
    (then
      ;; i32로 더하고 오버플로우 체크
      (local.set $sum
        (i32.add
          (i31.get_s (ref.cast i31ref (local.get $a)))
          (i31.get_s (ref.cast i31ref (local.get $b)))))

      ;; 오버플로우 체크 (31비트 범위 초과?)
      (if (result anyref)
        (i32.and
          (i32.ge_s (local.get $sum) (i32.const -1073741824))  ;; >= -2^30
          (i32.lt_s (local.get $sum) (i32.const 1073741824))) ;; < 2^30
        (then
          ;; 범위 내: i31ref 반환
          (ref.i31 (local.get $sum)))
        (else
          ;; 오버플로우: BigInt로 승격
          (call $promote_to_bigint_i32 (local.get $sum)))))
    (else
      ;; Slow path: BigInt 연산
      (call $bigint_add (local.get $a) (local.get $b)))))
```

### 비교 연산

```wasm
(func $int_eq (param $a anyref) (param $b anyref) (result i32)
  ;; 둘 다 i31ref면 직접 비교
  (if (result i32)
    (i32.and
      (ref.test i31ref (local.get $a))
      (ref.test i31ref (local.get $b)))
    (then
      (i32.eq
        (i31.get_s (ref.cast i31ref (local.get $a)))
        (i31.get_s (ref.cast i31ref (local.get $b)))))
    (else
      ;; BigInt 비교
      (call $bigint_eq (local.get $a) (local.get $b)))))
```

---

## Float Type

### f64 고정

```rust
Float = f64  // IEEE 754 64비트 부동소수점
```

### WasmGC 표현

**일반 컨텍스트:**

```wasm
;; 직접 f64 값
(local.get $float_value)  ;; f64 타입
```

**다형적 컨텍스트:**

```wasm
;; Boxing 필요
(type $BoxedF64 (struct (field $value f64)))

(struct.new $BoxedF64 (f64.const 3.14159))
```

### Boxing/Unboxing

```wasm
;; Boxing
(func $box_f64 (param $v f64) (result (ref $BoxedF64))
  (struct.new $BoxedF64 (local.get $v)))

;; Unboxing
(func $unbox_f64 (param $box (ref $BoxedF64)) (result f64)
  (struct.get $BoxedF64 $value (local.get $box)))
```

---

## 다형적 컨텍스트에서의 Boxing

### 타입 계층

```
any (anyref)
 ├─ i31       ← Int (fixnum)
 └─ struct
     ├─ $BigInt      ← Int (bignum)
     ├─ $BoxedF64    ← Float
     └─ 사용자 타입   ← struct/enum
```

### Boxing 전략

| 타입         | WasmGC 표현       | 힙 할당           |
| ------------ | ----------------- | ----------------- |
| Int (fixnum) | `i31ref`          | 없음              |
| Int (bignum) | `(ref $BigInt)`   | 자동 승격         |
| Float        | `(ref $BoxedF64)` | 필요              |
| struct/enum  | 기존 참조         | 없음 (업캐스트만) |

### Int는 추가 Boxing 불필요

Int는 이미 anyref 서브타입이므로, 다형적 함수에서 추가 boxing이 필요 없다:

```rust
fn identity(a)(x: a) -> a { x }

// Int로 호출
identity(42)
// → i31ref 또는 BigInt 그대로 전달
// → 추가 wrapping 없음!
```

---

## 타입별 연산 성능

### Int

| 연산   | Fixnum (i31)           | Bignum                |
| ------ | ---------------------- | --------------------- |
| 덧셈   | O(1) + 오버플로우 체크 | O(n)                  |
| 곱셈   | O(1) + 오버플로우 체크 | O(n²) 또는 O(n log n) |
| 비교   | O(1)                   | O(n)                  |
| 메모리 | 0바이트                | 8 + 4n 바이트         |

### Float

| 연산 | 일반          | 다형적 컨텍스트        |
| ---- | ------------- | ---------------------- |
| 산술 | O(1), 0바이트 | O(1), 8바이트 (boxing) |

---

## 리터럴

### 정수 리터럴

```rust
42        // Int (i31ref)
0xFF      // Int (i31ref, 255)
0b1010    // Int (i31ref, 10)
999999999999999999999999  // Int (BigInt)
```

### 실수 리터럴

```rust
3.14      // Float (f64)
1.0e10    // Float (f64, 과학적 표기법)
0.0       // Float (f64)
```

---

## Cranelift Backend

### Int 표현

**Fixnum:**

- Tagged pointer: 최하위 비트 1로 fixnum 표시
- 값: (n << 1) | 1

**Bignum:**

- 힙 할당된 구조체 포인터
- 최하위 비트 0

```c
// C 의사 코드
typedef union {
    intptr_t fixnum;        // 최하위 비트 = 1
    struct BigInt* bignum;  // 최하위 비트 = 0
} Int;

#define IS_FIXNUM(x) ((x) & 1)
#define FIXNUM_VALUE(x) ((x) >> 1)
#define MAKE_FIXNUM(n) (((n) << 1) | 1)
```

### Float 표현

- 일반 컨텍스트: f64 직접 사용
- 다형적 컨텍스트: 힙 할당 박스

---

## 향후 확장

### 추가 수치 타입

```rust
// 향후 고려
Int32   // 32비트 고정 (FFI용)
Int64   // 64비트 고정 (FFI용)
Float32 // 32비트 부동소수점
Rational // 유리수
Complex  // 복소수
```

### FFI 호환성

FFI에서 고정 크기 정수가 필요한 경우:

```rust
// C 함수 호출
extern fn c_function(n: Int32) -> Int32

// Tribute 코드
let n: Int = 42
c_function(Int32::from(n))  // 명시적 변환
```

---

## References

- [i31ref in WasmGC](https://github.com/WebAssembly/gc/issues/100)
- [OCaml Integer Representation](https://ocaml.org/manual/5.1/intfc.html)
- [GMP - GNU Multiple Precision Arithmetic](https://gmplib.org/)
