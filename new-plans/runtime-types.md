# Runtime Type Descriptors

Tribute 값의 저장 모양과 runtime 타입 정보를 나눈다. 저장 모양(layout)은
target lowering이 다루는 structural한 사실이고, 이름·필드 이름·필드 종류 같은
nominal 정보는 runtime 타입 descriptor가 소유한다. 값은 자기 descriptor를
가리키며, 메모리 해제·값 출력·variant 판별이 이 descriptor를 읽는다.

## 개요

| 개념 | 소유하는 사실 | 성격 |
| ---- | ---- | ---- |
| layout | field offset, 크기, 필드의 물리 표현 | structural. 모양이 같으면 같은 layout |
| descriptor | 이름, 필드 이름, 필드 종류, 해제 방법, variant 소속 | nominal. 소스 타입마다 하나 |

**설계 원칙:**

- 이름은 IR 타입에서 지워져도 runtime에서 지워지지 않는다. Nominal layout을
  해석하는 단계가 끝나면 IR의 struct 타입은 이름 없는
  [structural struct](ir.md#nominal-수준과-structural-수준)가 되고, 그 아래에서
  이름이 필요한 일은 모두 descriptor를 거친다.
- 저수준 layout 타입이 갖는 identity는 runtime 동작이 달라지는 경우로 한정한다.
  Layout이 같고 runtime 동작이 같은 두 소스 타입은 layout을 공유하며 descriptor로만
  구별된다.
- Descriptor는 runtime이 값만 보고 그 모양을 알아낼 수 있게 한다. 제네릭으로
  지워진 `anyref` 값도 가리키는 대상의 descriptor로 모양을 안다.

이 문서는 값 출력 기능(예: 값을 그대로 보여 주는 `echo`/`inspect`류 기능)이 기대는
runtime 정보를 정한다. 그런 기능의 문법과 출력 형식은 이 문서의 범위가 아니다.

---

## Descriptor 단위

Descriptor는 runtime에 할당되는 값의 종류마다 하나다.

- 소스 struct의 인스턴스마다 하나. 특수화된 제네릭 struct는 특수화 인자마다
  별개의 descriptor를 가진다.
- 소스 enum의 각 variant마다 하나. 같은 enum의 variant descriptor는 같은 enum
  descriptor를 가리킨다.
- Compiler가 할당하는 값마다 하나. Boxing된 `Bool`, `Nat`, `Int`, `Float`, runtime이
  할당하는 `Bytes`, compiler 소유 layout(closure, evidence marker 등)이 여기에
  속한다.

이름만 다르고 모양이 같은 두 struct는 같은 layout을 쓰고 서로 다른 descriptor를
가진다. 같은 소스 타입을 서로 다른 layout으로 표현하는 일은 없다.

## Descriptor 내용

| 항목 | 뜻 |
| ---- | ---- |
| kind | struct, variant, enum, compiler 소유 값 |
| 이름 | 정규화된 nominal 이름. 특수화 인자를 포함한다 |
| 소속 | variant이면 소속 enum descriptor와 variant tag |
| 필드 | 선언 순서의 필드 목록. 필드마다 이름과 필드 종류 |
| 해제 정보 | 값이 해제될 때 함께 release할 필드. 필드 종류에서 유도한다 |

필드 종류는 runtime이 그 필드를 해석하는 방법이다.

| 필드 종류 | 해석 | 해제 |
| ---- | ---- | ---- |
| scalar | 폭과 부호, 정수·부동소수·bool 구분 | 하지 않음 |
| managed 참조 | 가리키는 값의 descriptor를 따라간다 | release |
| unmanaged 포인터 | 따라가지 않는다 | 하지 않음 |
| 동적 값 | 값에서 descriptor를 찾는다. Boxing된 scalar도 여기에 든다 | release |

필드 이름이 없는 compiler 소유 값은 필드 이름 없이 필드 종류만 가진다.

## IR에서의 표현

Descriptor 참조는 nominal 정보가 남아 있는 동안 정해진다.

- 할당 operation(`adt.struct_new`, `adt.variant_new`)은 nominal layout을 해석하는
  단계에서 자기 descriptor 참조를 얻는다. 참조는 모듈 수준 descriptor 선언을
  가리키는 operation 속성이며, layout 타입과 별개다.
- 이후 layout 타입이 이름 없는 structural struct로 바뀌어도 descriptor 참조는
  그대로 남는다. 저수준 pass는 descriptor를 layout 타입에서 다시 찾지 않는다.
- Descriptor 선언은 descriptor 내용을 모두 담는다. Target은 선언을 자기 runtime
  표현(table, data segment)으로 내리고, 할당 operation의 참조를 그 표현의
  번호로 바꾼다.
- 해제 정보는 ownership 계획이 정한 managed 판정을 그대로 쓴다. 저수준에서 필드의
  물리 타입만 보고 다시 판정하지 않는다.

## Native 배치

Native RC 객체의 header가 descriptor를 가리킨다.

```text
[-8] refcount: u32
[-4] descriptor index: u32
[ 0] payload...
```

- Header의 index 칸은 [RC header](rc.md#object-header)의 RTTI index다. 이 번호는
  descriptor table의 index이며, table 항목은 해제 함수와 출력에 필요한 descriptor
  레코드를 함께 담는다.
- Structural layout(`mem.struct`)의 필드 타입은 managed 참조와 unmanaged 포인터를
  구분한다. Managed 참조는 `tribute_rt.anyref`, unmanaged 포인터는 `core.ptr`로
  둔다. 둘의 크기와 정렬은 같지만 해제 동작이 다르기 때문이다.

## Wasm 배치

WasmGC 객체에는 header가 없으므로 객체가 descriptor를 필드로 가진다.

- 사용자 struct와 variant의 GC 타입은 첫 필드에 descriptor를 둔다. 표현은 `i32`
  descriptor index다. Descriptor 레코드는 module이 소유한 table(data segment나
  global 배열)에 둔다.
- Descriptor를 GC struct 참조로 두는 방법도 가능하다. 참조는 table 접근 없이
  레코드를 읽을 수 있지만 필드마다 참조 크기를 쓰고, `i32` index는 native 번호
  체계와 같은 table 모양을 쓴다. 기본은 `i32` index다.
- GC 타입은 structural로 합친다. 필드 표현이 같은 struct와 variant는 같은 GC
  타입이며, runtime identity는 descriptor 필드가 맡는다. 그래서 variant마다 별개의
  GC 타입 index를 둘 필요가 없다.
- Builtin layout(bytes, closure, marker, evidence, boxing된 scalar)은 지금처럼
  예약 GC 타입을 쓰며, descriptor 필드를 두지 않는다. 이 값들의 descriptor는 예약
  번호로 정해진다.

## Variant 판별

어느 variant인지는 값의 descriptor로 판별한다.

- Wasm은 descriptor 필드를 읽어 비교한다. 서로 다른 GC 타입 index에 대한
  `ref.test`로 variant를 판별하지 않는다.
- Native tagged union은 payload에 tag 필드를 가진다. Descriptor가 variant를 이미
  알려 주므로 tag 필드를 유지할지, descriptor로 대체할지는 enum 표현을 정할 때
  함께 정한다.

## 불변 조건

- Runtime에 할당되는 모든 값은 descriptor를 가진다.
- Descriptor의 필드 종류와 layout 필드의 물리 표현은 일치한다. Managed 참조
  필드는 managed 참조 표현을, scalar 필드는 해당 폭의 scalar 표현을 쓴다.
- Descriptor 번호는 전체 프로그램 컴파일을 전제로 한 프로그램 내부 번호다. 따로
  컴파일한 단위 사이에서 객체가 오가게 되면 번호 대신 descriptor가 스스로를
  설명하는 방식이 필요하다([RTTI table](rc.md#rtti-table)과 같은 전제).
- Layout은 descriptor를 대신하지 않는다. Layout이 같다는 사실로 두 값을 같은
  소스 타입으로 간주하지 않는다.
