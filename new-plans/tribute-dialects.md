# Tribute 고수준 dialect

TrunkIR core 위에서 Tribute가 정의하는 고수준 dialect의 여러 operation에 걸친
규칙이다. TrunkIR 자체의 설계는 [ir.md](ir.md)를 따른다.

`tribute.*` represents source-level constructs that should disappear after
resolution, type checking (including TDNR), and AST-to-IR lowering.

<!-- markdownlint-disable-next-line MD033 -->
<a id="direct-style-control"></a>

## 직접형 호출 대상과 제어

`tribute_control.*`은 typed frontend lowering과 실행 전략 사이의 target-independent
직접형 경계다. CPS 전략은 이 경계 뒤에서 convention에 따라 뜻이 달라지는 callable
operation을 [`tribute_cps`](#convention과-class-특수화)로 바꾼다. Dialect identifier는 정확히
`tribute_control`이다. TrunkIR은 qualified operation을 하나의
`<dialect>.<operation>` 쌍으로 parse하므로 dialect identifier 안에 점을 넣는
등 separator를 하나 더 사용하는 표기는 invalid이다.

이 dialect는 CPS legalization 전의 Tribute 고유 callable과 effect-control
의미를 함께 소유한다. ANF는 input invariant이지 dialect의 정체성이 아니다.
산술, ADT/list/tuple/record 구성과 structured selection은 기존 dialect에 남지만,
`func.*`, `closure.*`, `func.func_sig`는 physical 표현이므로 이 경계에 나타나지
않는다.

논리 callable type은 `tribute_control.func_sig<(Params...) -> Result>`이며
calling convention을 갖지 않는다([convention entry](#convention-entry)). 두 dialect가 이
type을 함께 쓰며, convention은 `tribute_cps`의 정의, 람다의 convention entry, 간접
이전 operation에 둔다([convention과 class 특수화](#convention과-class-특수화)).
It uses flat `[inputs..., results...]` storage with mandatory `num_inputs` and
`num_results` u32 delimiters; its source-owned validated API requires exactly
one result. `Result`와 `Params`는 source-logical type이다. Convention은 keyword 문자열
`"direct"`, `"evidence_direct"`, `"cps"` 가운데 하나이며, 이 문자열이 convention의
유일한 저장 형식이다. 정수나 다른 값은 잘못된 convention이다. Convention은
[CPS 전략](cps-effects.md#row와-호출-규약)이 [ability row](#ability-row)에서 계산한
값이며 body에서 추론하지 않는다. Legalization은 이를 기존
`CallableAbi`와 physical `func.func_sig`, `closure.closure` 및
`tribute.calling_convention` operation attribute로 바꾼다. `tribute_cps` operation의
convention은 그 operation이 선언한 속성이고, physical operation의 convention은
[`ConventionBearing` interface](cps-effects.md#pre-cps-callable-shape)로만 읽고
쓴다. `closure.lambda`는 본문마다 convention을 자기 `conventions` 속성으로 선언한다.
Type verifier는
result 하나와 parameter 0개 이상, 모든 component type의 resolution을 검사하며,
convention 속성과 physical `func.func_sig`나 `closure.closure` component를
허용하지 않는다. 전체 규칙은
[cps-effects.md](cps-effects.md#pre-cps-callable-shape)에 있다.

최소 operation 집합은 다음과 같다:

| Operation | 용도 |
| ---- | ---- |
| `tribute_control.func` | named source callable의 선언 또는 정의 |
| `tribute_control.lambda` | capture를 가진 source lambda와 callable value 생성 |
| `tribute_control.func_ref` | named function을 first-class callable value로 참조 |
| `tribute_control.call` | named source callable 직접 호출 |
| `tribute_control.call_indirect` | source callable value 간접 호출 |
| `tribute_control.tail_call` | source `become`의 named callable 직접 proper tail call |
| `tribute_control.tail_call_indirect` | source `become`의 callable value 간접 proper tail call |
| `tribute_control.return` | `func` 또는 `lambda` body의 logical result 반환 |
| `tribute_control.perform` | source `fn` 또는 general `op` 하나를 semantic kind를 보존한 직접형으로 호출 |
| `tribute_control.handle` | 직접형 computation, completion arm, handler arm의 경계를 설정. 각 arm은 처리하는 operation의 binding과 짝을 이루는 region이다 |
| `tribute_control.resume` | resumptive general handler arm에 바인딩된 affine resumption을 소비 |
| `tribute_control.yield` | 실행 가능한 `tribute_control` region을 logical value로 종료 |

Class 특수화가 만드는 `tribute_cps` operation은 다음과 같다. `func_table`을 뺀
각각은 같은 이름의 `tribute_control` operation에 convention을 더한 것이고,
`func_table`은 `func_ref`가 가리키는 convention table 정의다.

| Operation | 용도 |
| ---- | ---- |
| `tribute_cps.func` | convention을 가진 named callable 인스턴스 |
| `tribute_cps.lambda` | convention entry 목록을 가진 source lambda |
| `tribute_cps.func_ref` | `func_table`을 가리키는 named function 참조 |
| `tribute_cps.func_table` | convention entry마다 대상 인스턴스를 적은 named function의 convention table |
| `tribute_cps.call_indirect` | convention을 가진 callable value 간접 호출 |
| `tribute_cps.tail_call_indirect` | convention을 가진 callable value 간접 proper tail call |

Operation 하나의 형상, 의미, 검증, 소유권, 위치 계약은 그 operation의
`#[dialect]` 선언 doc comment가 정한다
([`tribute_control`](../crates/tribute-ir/src/dialect/tribute_control.rs),
[`tribute_cps`](../crates/tribute-ir/src/dialect/tribute_cps.rs),
[`ability`](../crates/tribute-ir/src/dialect/ability.rs)). 이 문서는 여러
operation에 걸친 규칙만 다룬다.

`func.constant`, `func.unreachable`의 logical 복제는 없다. Named function value는
`func_ref`가 표현한다. Source [`become`](syntax.md#tail-call-become)은
`tribute_control.tail_call`/`tail_call_indirect`로 나타나며, 이 두 operation이
`func.tail_call`/`func.tail_call_indirect`의 source-logical 짝이다. 표시 없는 호출을
tail transfer로 바꾸는 것은 보장이 없는 최적화다
([cps-effects.md](cps-effects.md#source-proper-tail-call의-적법화)).
Legalization은 알려진 target에 `func.tail_call`, closure, continuation과 `done_k`
target에 새 `func.tail_call_indirect`를 만들 수 있다.
`func.constant`는 후속 physical closure lowering이 만들며 `func.unreachable`은
reject adapter 같은 compiler helper 안에서만 legalization 뒤에 사용한다.

이 dialect는 opaque type `tribute_control.resume_token<input, answer>`도
소유한다. `input`은 중단된 operation continuation이 받는 값이고, `answer`는
그 continuation을 enclosing handle까지 실행한 logical result다. 이 type은 source
type, callable ABI, backend carrier가 아니며 continuation representation을
검사할 권한도 아니다. Logical result가 `Never`인 source general operation은
canonical `core.never` TypeRef를 사용하고 resumption을 만들지 않으며
`resume_token`도 노출하지 않는다. Verifier가 erased type을 추측하지 않고
non-resumptive case를 식별해야 하므로 `Never`용 `anyref` placeholder는 이
경계에서 valid하지 않다.

## Ability row

`tribute_control.func_sig`는 callable의 ability row를 `ability_row` type 속성으로
가진다. 함수 형태에서는 결과 뒤의 `row(..)` 절로 쓴다. `|` 앞은 `instances`이고,
뒤의 `'aN`은 `tail = N`이다. 아래 타입의 저장 표현은
`{ability_row = {instances = [@"<instance A>", @"<instance B>"], tail = 1}}`이다.

```text
tribute_control.func_sig<(T) -> R row(@"<instance A>", @"<instance B>" | 'a1)>
```

- **값:** `instances`는
  [`ability.decl`](../crates/tribute-ir/src/dialect/ability.rs)에 대한 symbol
  참조 목록이고
  `tail`은 둘러싼 정의의 [row binder](#row-binder) index다. 둘 다 없으면 닫힌 빈
  row이며 속성을 생략해 쓴다. `instances`는 symbol 경로 순서로 정렬하며 같은
  instance를 한 번만 둔다. Row는 [instance의 집합](type-inference.md#기본-규칙)이다.
  명시 label과 tail이 같은 instance를 담는 호출에서 어느 위치가 어느 handler를 받는지는
  type이 아니라 typechecking이 인스턴스화에서 계산한
  [evidence 선택](#evidence-선택-속성)이 담는다.
- **정체성:** signature의 정체성은 입력, 결과, row의 조합이다. Convention은
  signature에 들어가지 않는다. 같은 row도 인스턴스가 고정한 class에 따라 다른
  convention으로 불릴 수 있다([cps-effects.md](cps-effects.md#row와-호출-규약)).
- **범위:** row는 source-logical 경계 안에만 있다. CPS legalization은
  `func.func_sig`를 만들 때 row를 버리고 [`instantiate`](../crates/tribute-ir/src/dialect/tribute_control.rs)를
  피연산자로 대체하며, representation/ABI 경계의 출구는 row와 아래 속성들을 거부한다.

### Row binder

두 dialect의 `func`는 자기 row binder를 `row_binders` 속성으로 선언한다. 원소는
binder의 종류를 나타내는 문자열이다.

| 종류 | 뜻 |
| ---- | ---- |
| `param` | 정의의 scheme이 양화한 row 변수 |
| `ability_arg` | scheme이 양화했고 signature나 본문의 ability 인자 안에 나타나는 row 변수 |
| `local` | 값이 람다인 `let`의 지역 scheme이 양화한 row 변수. 그 람다의 type이 적는 변수이며 람다가 [소유](#지역-람다의-양화와-인스턴스화)한다 |
| `body` | 어느 scheme도 양화하지 않은 row 변수, 또는 지역 scheme이 양화했지만 [양화하는 람다가 없는](#지역-람다의-양화와-인스턴스화) row 변수 |

양화된 binder(`param`, `ability_arg`)가 scheme의 binder 순서로 먼저 오며, 그 위치가
row 인자의 위치다. 나머지는 정의 안에 처음 나타난 순서로 뒤에 온다. Class 특수화가
람다의 entry 본문을 만들 때 그 본문 안에서 row를 양화하는 람다는 새 `local` binder를
목록 끝에 받으므로, 한 정의의 인스턴스들은 양화된 binder만 공유하고 나머지 binder
목록은 서로 다를 수 있다. 속성이 없으면
binder가 없다. `ability_arg`는 ability instance의 symbol이 그 인자를 드러내지
않으므로 따로 표시한다.

값이 람다가 아닌 binding은 한 번 계산되므로 그 지역 scheme의 변수를 인스턴스화하는
사용이 없고, 그 변수는 `body`다. Binder 종류는 row의 구조만 기록하며 실행 전략을 담지
않는다. CPS 전략의 class 특수화는 tail의 class를 binder 종류에 따라 읽는다. `param`은
그 인스턴스가 고정한 class, `ability_arg`는 `Cps`다. `body`는 정의 안의 어느 callable
type의 매개변수 타입에도 나타나지 않으면 요구 없음이고, 나타나면 `Cps`다
([tail 규칙](cps-effects.md#row와-호출-규약)). 매개변수 타입 안의 `adt.typeref` row
인자에 나타난 것도 매개변수 타입에 나타난 것이다. 이 판정은 binder마다 정의 전체에서 한
번 하므로 그 binder를 읽는 모든 자리가 같은 class를 읽는다. `local`의 class는 그것을
소유한 람다의 본문마다 다르며, [convention entry](#convention-entry)의 class 읽기가
정한다.

정의 안의 모든 tail은 이 목록의 index다. Signature, 중첩된 callable type, block
인자, operation 결과, tuple layout의 원소, [`adt.typeref`의 row 인자](#nominal-row-인자)가
모두 같다. Type은 binder를 도입하지 않는다. 예외는 row 매개변수를 가진 nominal
layout 하나이며, 그 binder는 layout 안의 type에만 미친다. Module 수준의 다른 type에는
둘러싼 정의가 없으므로 [`ability.operation`](../crates/tribute-ir/src/dialect/ability.rs)의
type은 tail을 가질 수 없다.

### Nominal row 인자

[Row 매개변수](types.md#row-매개변수)를 가진 struct·enum의 layout(`adt.struct`,
`adt.enum`)은 자기 row binder를 type 속성 `row_params`(개수)로 선언한다. Layout
안의 필드 type, 곧 필드의 callable type과 중첩된 `adt.typeref`의 row 인자가 쓰는
tail은 이 binder의 index다. 이 binder는 layout 밖으로 나가지 않는다. Binder는 모두
선언이 양화한 row 매개변수이므로 `func`의 `row_binders`와 달리 종류 없이 개수만 둔다.
[Tuple layout](#compiler-generated-tuple-layout)은 binder를 소유할 선언이 없으므로
row를 layout 이름에 담고 둘러싼 정의의 binder를 쓰지만, nominal layout은 선언이
binder를 소유하므로 row 인자와 상관없이 하나다.

```text
adt.struct<Callback['a0](run: tribute_control.func_sig<(core.i64) -> core.i64 row(| 'a0)>)>
adt.typeref<Callback[row(@"<State(Int)>")]>
adt.typeref<Callback[row(| 'a2)]>
adt.typeref<Callback[row()]>
```

Textual form은 layout의 binder를 이름 바로 뒤에 [정의 header의 binder 목록](#row-binder)과
같은 `['a0, 'a1, ...]`로 쓰고, typeref의 row 인자를 이름 바로 뒤에 `[row(...), ...]`로
쓴다. 필드 안의 tail이 그 앞에 선언된 binder를 가리키도록 binder 목록은 필드보다 먼저
온다. Binder는 종류 keyword와 합집합(`=`)을 쓰지 않으며 `'a0`부터 차례로 번호를 붙인다.
Row 인자 하나는 `func_sig`의 row와 같은 `row(...)`이지만, row 인자는 위치마다 하나씩
있어야 하므로 빈 row도 생략하지 않고 `row()`로 쓴다. Row 매개변수가 없으면 `[...]`를
쓰지 않는다. 이 문법으로 쓸 수 없는 속성 값은 일반 형태로 출력한다.

- 그 선언의 값 type인 `adt.typeref`는 row 매개변수마다 row 하나를 type 속성
  `row_args`에 선언 순서대로 가진다. 각 row는 `ability_row`와 같은 형식이며, tail은
  그 type을 쓴 정의의 binder index다. Layout 안에서는 layout의 binder index다. Row
  매개변수가 없는 선언의 typeref는 속성을 생략한다.
- `row_args`는 typeref의 정체성에 들어간다. 같은 layout을 가리키며 row 인자만 다른
  두 typeref는 다른 type이다. Layout의 이름과 게시된 alias는 row 인자를 담지 않으므로
  그런 typeref들은 layout 하나를 공유한다.
- `adt` 생성·접근 operation이 보는 필드 type은 layout의 필드 type에서 layout binder를
  그 operation이 다루는 값의 typeref `row_args`로 [치환](#row-인자와-치환)한 것이다.
  Frontend는 `struct_get`, `variant_get` 결과와 `struct_new`, `variant_new` 피연산자의
  type을 이 치환으로 쓴다.
- Row 인자는 다른 row와 같이 source-logical 경계 안에만 있다. CPS legalization은
  `row_args`와 `row_params`를 버리고 tuple layout을
  [새 이름](#compiler-generated-tuple-layout)으로 다시 만들며, representation/ABI
  경계의 출구는 `row_args`와 `row_params`를 거부한다.

### Tail의 합집합

Row tail이 여러 tail의 합집합으로 선언된 정의는 `row_unions` 속성으로 그 합집합을
적는다. 원소는 `{result = i, sources = [j, k, ...]}`이며 `sources`는 열린 source의
tail을 선언 순서로 담는다. `result`와 같은 tail은 `sources`에 넣지 않는다.

이 순서가 evidence tail 칸의 번호다. 정의 안의 [`{select = i}`](#evidence-선택-속성)는
자기 row tail을 결과로 하는 합집합의 `i`번째 source를 가리키고, 호출의
`{tails = [...]}`는 callee 합집합의 source마다 선택 하나를 같은 순서로 담는다.

### 지역 람다의 양화와 인스턴스화

값이 람다인 `let`의 지역 scheme이 양화한 row 변수는 그 람다가 소유한다.
두 dialect의 `lambda`는 `quantified_rows` 속성에 자기가 양화하는 `local` binder의
index를 인스턴스화 순서로 적는다. 이 binder는 둘러싼 `func`의 binder이며 람다가 새
binder를 만들지는 않는다. 모든 `local` binder는 정확히 한 람다가 소유하고,
`quantified_rows`의 원소는 모두 `local` binder다. 람다의 entry 본문들은 같은
binder를 공유하며, entry마다 그 binder에 class를 하나씩 고정한다.

이런 람다의 모든 사용은 [`tribute_control.instantiate`](../crates/tribute-ir/src/dialect/tribute_control.rs)를
거친다. 호출뿐 아니라 인자로 넘기거나 다른 값에 담는 사용도 같다. 다른 지역 람다의
본문 안에 있는 사용도 그 자리의 `instantiate`이며, row 인자의 tail은 바깥 람다가
소유한 binder를 가리킬 수 있다. Binder 번호는 정의 하나에서 평평하므로 scope가
중첩되지 않는다.

`instantiate`는 type의 row만 치환하며 convention이나 entry를 고르지 않는다. 한 람다
식은 `tribute_control.lambda` 하나이고, 사용처마다 따로 만들지 않는다. Class
특수화도 람다를 값 하나로 두며 [본문](#convention-entry)만 class마다 늘린다. 쓰이지 않는
람다는 양화할 사용이 없으므로 그 변수는 `body`다. 다른 `let`의 값으로 람다를 다시
묶는 별칭은 람다가 아닌 값이다. 별칭 자리에서 람다를 한 번 인스턴스화한다.
람다나 함수 참조가 아닌 `let`은 row 변수를 양화하지 않으므로
([일반화](type-inference.md#generalization)), 그런 binding의 모든 사용은 값을 만든
쪽과 같은 row를 본다. 사용이 고정한 row는 값의 type에도 그대로 들어가고, 아무 사용도
고정하지 않은 row 변수만 `body` binder로 남는다. 그래서 별칭이나 계산 결과를 통한
간접 이전과 별칭 자리의 인스턴스화는 그 binder를 같은
[tail 규칙](cps-effects.md#row와-호출-규약)으로 읽어 같은 class를 고르며, 값은 그
entry를 가진다. 그 binder가 정의 안의 어느 매개변수 타입에도 나타나지 않으면 둘 다
요구 없음으로 읽는다. 값은 binder를 추상 변수로 두고 만들어졌고, binder가 정체를 잃는
매개변수 자리가 없으므로 그 binder를 통해 어떤 operation에도 닿지 못한다. 나타나면
둘 다 `Cps`로 읽는다. 그런 값의 매개변수 row tail이 그 `body` binder이면, 그 자리에
넘기는 람다는 매개변수의 callable type으로 만들고, 함수 참조는 그 type의
`tribute_control.func_ref`로 다시 만든다. Conversion cast는 callable의 ABI를 바꿀 수
없기 때문이다.

람다의 별칭과 달리, 이름 붙은 함수를 그대로 묶은 binding과 그런 binding을 다시
묶은 별칭은 값을 묶지 않는다. 함수 참조는 할당도 효과도 없으므로, 그 binding의
사용은 묶인 값을 읽지 않고 함수를 다시 참조한다. 그 `tribute_control.func_ref`는
타입 검사가 그 사용에 고른 인스턴스의 row를 넘긴다. 별칭마다 scheme을 인스턴스화한
row를 사용 쪽에서부터 거슬러 대입해 함수 참조 자리의 row 인자를 얻는다. 그래서
순수한 사용은 `body` binder 대신 그 사용의 row를 보고, 닫힌 row의 람다를 그대로
넘길 수 있다.

### Row 인자와 치환

Row-polymorphic 정의를 참조하는 `call`, `tail_call`, `func_ref`는 `row_args`로 callee의
양화된 binder마다 row 하나를 넘긴다. 각 row는 `ability_row`와 같은 형식이며 tail은
참조하는 쪽 정의의 binder index다. Callee에 양화된 binder가 없으면 속성을 생략한다.

참조의 callee symbol은 class 특수화 전에는 정의를, 뒤에는 class 인스턴스 하나를
가리키며, 그 convention은 인스턴스 정의의 것이다. Row 인자의 class가 인스턴스의 class와 맞는지는
[convention class](cps-effects.md#row-변수의-convention-class) 규칙이 정한다.
참조가 보는 callee signature는 callee의 type에 row 인자를 치환한 것이다. 치환은
type 안의 모든 `func_sig`(자기 자신 포함)와 `adt.typeref`의 row 인자에서 tail이
양화된 binder `t`인 row `(I, t)`를 `(I ∪ args[t].instances, args[t].tail)`로 바꾸고
정렬한다. 다른 속성은 바꾸지 않는다. Layout 안의 type은 layout의 binder를 쓰므로
치환하지 않는다. Tuple layout은 원소를 치환한 뒤 같은 이름 규칙으로 다시
만든다([layout 이름](#compiler-generated-tuple-layout)). `call_indirect`와
`tail_call_indirect`의 callee type은 이미 caller의 binder로 쓰여 있으므로 row 인자가
없다.

`instantiate`의 `row_args`는 대상 람다의 `quantified_rows` 순서를 따른다. `i`번째
row 인자가 `quantified_rows`의 `i`번째 binder를 대체하며, 치환 규칙은 위와 같다.

### Convention과 class 특수화

Convention은 `tribute_cps` dialect의 operation만 가진다. Frontend는 `tribute_control`만
내보내며, 그 IR에는 convention이 없다. CPS 전략의
[class 특수화 pass](cps-effects.md#class-특수화-pass)는 class 인스턴스를 만들면서
convention에 따라 뜻이 달라지는 operation 다섯을 같은 이름의 `tribute_cps`
operation으로 바꾼다.

- `tribute_cps.func`는 자기 convention을 가진다. 인스턴스마다 하나다.
- `tribute_cps.lambda`는 class마다의 본문과,
  [convention entry](#convention-entry)마다 그 entry를 채우는 본문을 가진다.
- `tribute_cps.call_indirect`와 `tribute_cps.tail_call_indirect`는 callee를 부르는
  convention을 가진다.
- `tribute_cps.func_ref`는 `tribute_cps.func_table` 하나를 가리키고, 그 table이
  convention entry마다 그 entry로 들어갈 대상 인스턴스를 가진다.

나머지 `tribute_control` operation(`call`, `tail_call`, `return`, `instantiate`,
`perform`, `handle`, `handler`, `resume`, `yield`)은 그대로 남아 `tribute_cps`
callable의 본문 안에서 쓰인다. 이 변환은 full conversion이다. Class 특수화 뒤에는
위 다섯 operation의 `tribute_control` 형태가 남지 않으며, legalization의 입력 검증은
남은 것을 거부한다. 두 dialect의 callable 값은 모두 `tribute_control.func_sig`를
type으로 쓰며, 이 type은 convention을 갖지 않는다. `tribute_cps`도 row와 logical
type을 가진 source-logical IR이며 hidden parameter는 아직 없다.

Convention에 관한 규칙은 다음과 같다.

- `tribute_control.tail_call`의 callee 인스턴스와 `tribute_cps.tail_call_indirect`의
  convention은 감싼 callable(정의, 또는 그 이전을 담은 entry)보다 강하지 않다.
- 정의의 convention과 간접 이전의 convention은 [CPS 전략](cps-effects.md#row와-호출-규약)이
  그 row와 인스턴스의 class에서 계산한 값이다.

Type의 일치를 요구하는 규칙은 두 dialect에서 같고, convention을 비교하지 않는다.

### Convention entry

Callable 값과 그 type은 convention을 갖지 않는다. 한 값이 여러 class의 자리에 놓이고
자리마다 다른 convention으로 불릴 수 있기 때문이다. 호출 지점은 둘러싼 인스턴스의
class로 정해지는 자기 convention 하나만 정적으로 알며, callee 값이 어느 class에서
만들어졌는지 묻지 않는다. 그 차이는 값이 가진 convention entry가 흡수한다.

**Convention entry.** Callable 값은 convention마다 convention entry 하나를 가진다.
Convention entry는 그 convention으로 들어왔을 때 실행할 물리 callable이며, 본문,
adapter, trap entry 가운데 하나다. 물리 ABI(hidden parameter)는 그 convention이
정하고, convention entry를 구현하는 물리 함수를 entry 함수라 한다. 간접 이전은 자기
convention의 convention entry로 들어가며, 실행 중에 entry를 고르지 않는다. 그래서
값마다 convention entry는 `Direct`, `EvidenceDirect`, `Cps` 셋을 넘지 않는다.

**Class 읽기.** Class 특수화는 row의 class를 정확한 class나 상한으로 읽는다. 상한
`c`로 읽은 row는 실행 중에 `c`보다 약하거나 같은 어느 class든 요구할 수 있다는
뜻이다. 상한 `Direct`는 정확한 `Direct`와 같다. Row의 **하한**은 row에 적힌 ability
instance들의 requirement를 합친 class이며, row가 실행 중에 요구하는 class는 하한보다
약하지 않다.

- 둘러싼 인스턴스의 class 변수는 그 인스턴스가 고정한 정확한 class나 상한으로 읽는다.
- 람다가 양화한 `local` binder는 아래 본문 `c` 안에서, class `c`를 고르는 사용들이 그
  binder에 넘기는 row의 class로 읽는다. 그 사용들이 모두 같은 정확한 class를 넘기면
  그 class이고, 그렇지 않으면 넘기는 class들을 합친 상한이다. 람다 자신의 row에
  나타나는지와 관계없다.
- Row의 class는 하한과 tail의 class를 합친 것이다. Tail이 상한이면 row의 class도
  상한이다. 다만 하한이 그 상한과 같으면 정확한 class다.

**본문.** 람다는 다음 class마다 본문 하나를 가지며, 다른 class의 본문은 없다.

- `instantiate`로만 쓰이는 `let` 람다는 사용이 고르는 class마다 본문을 둔다. 사용의
  class는 그 사용의 `instantiate`가 람다 자신의 row에 넘기는 row의 class다.
- 그 밖의 람다는 자기 row의 class가 정확하면 그 class의 본문 하나를 둔다.
- 사용의 class나 자기 row의 class가 상한이면, 그 row의 하한부터 상한까지의 class마다
  본문을 둔다.

Class `c`의 본문은 위 읽기를 따르되, 자기 row의 tail이 상한이면 그 상한을 `c`로
좁혀 상한으로 읽는다. 상한 이하의 class로 부르면 그 class의 본문으로 들어오므로,
본문 `c`에 들어온 값의 실제 class는 `c`보다 강하지 않다는 것만 알 수 있다. 그래서
하한이 `c`보다 약해도 tail을 정확한 `c`로 읽지 않으며, 본문 `c`가 그 tail로 만드는
callable도 하한부터 `c`까지 본문을 가진다.

**값은 나누지 않는다.** 같은 class를 고르는 사용들이 양화한 binder에 서로 다른 class를
넘겨도 람다는 값 하나다. 그 binder로 읽는 row가 상한이 되고, 본문이 그 row로 만드는
callable은 하한부터 상한까지 본문을 가지므로 어느 사용 쪽에서 불러도 맞는 본문이
있다. 예를 들어 `let wrap = fn(k) { fn() { k() } }`를 순수한 `k`와 `EvidenceDirect`인
`k`로 쓰면, 두 사용은 모두 `Direct`를 고르고 `wrap`은 `Direct` 본문 하나를 가진다. 그
본문이 만드는 `fn() { k() }`는 상한 `EvidenceDirect`의 row를 가지므로 `Direct`와
`EvidenceDirect` 본문을 함께 가지며, 각 본문은 `k`를 자기 convention으로 부른다.

이 규칙은 다음 두 사실에 기댄다.

- 상한으로 읽은 row로 만든 값의 실제 class는 그 row의 하한과 상한 사이에 있고, 값이
  놓이는 자리의 class는 실제 class보다 약하지 않다. 상한 이하로 부르면 그 class의
  본문이 있고, 상한보다 강하게 부르면 상한 본문을 부르는 adapter가 있다.
- 간접 이전은 상한으로 부른다. 본문이 어떤 tail의 callable을 부르면 그 tail은 본문을
  담은 callable의 row에도 있다. `handle`은 ability instance를 처리할 뿐 tail을 지우지
  않기 때문이다. 그래서 그 상한은 본문의 convention보다 강하지 않다.

대가는 정확도다. 상한으로 읽은 tail의 callable을 부르면 실제 callee보다 강한 entry로
들어가 adapter를 거칠 수 있고, 상한으로 읽은 row의 callable은 쓰이지 않는 class의
본문을 가질 수 있다. 결과는 바뀌지 않는다.

**Capture.** 람다의 capture 목록은 하나이며, 모든 본문이 실제 사용하는 외부 SSA
값과 일치한다. Capture는 closure 하나에 한 번만 담긴다.

**채우기.** Class `c`의 convention entry는 다음과 같다.

- 람다에 class `c`의 본문이 있으면 그 본문이다.
- 없고 `c`보다 약한 본문이 있으면, 그 가운데 가장 강한 본문을 부르는 adapter다.
- 약한 본문도 없으면 trap entry다.

**Adapter.** `EvidenceDirect` entry의 adapter는 evidence를 받고 쓰지 않으며, `Cps`
entry의 adapter는 본문의 결과를 `Done<R>`로 넘긴다. Adapter는 본문을 한 번 부르고
끝나며 CPS 제어를 만들지 않는다. Adapter는 본문을 이름으로 부르지 않는다.
Environment 입력으로 받은 closure 자신의 convention table에서 본문 class의 entry를
`closure.entry`로 읽고, 같은 closure를 넘겨 부른다. 그래서 adapter는 람다나 capture에
매이지 않고 자기 entry signature와 본문의 class로만 정해지며, target은 둘이 같은
convention entry들이 adapter 하나를 공유하게 할 수 있다.

**Trap entry.** Trap entry는 그 convention entry의 signature를 가지고 들어오자마자
trap하는 함수이며, target은 signature가 같은 trap entry들이 하나를 공유하게 한다.
Trap entry는 실행되지 않는다. 값이 놓이는 자리의 class는 그 값의 실제 class보다
약하지 않고, 값은 실제 class부터 자기가 읽힌 상한까지의 본문을 모두 갖기 때문이다.
Class는 row의 tail에서 계산하고, 값이 tail의 정체를 잃을 수 있는 자리(`ability_arg`
tail, 매개변수 타입에 나타나는 `body` tail)는 `Cps`로만 읽는다. Class 특수화는 확인할 수 있는 것을 확인한다. 모든 간접
이전의 convention은 그 이전을 담은 인스턴스와 본문의 class 읽기로 callee type을 읽은
class(상한이면 상한)와 같아야 하며, 다르면 컴파일 오류다. Trap entry는 값의 흐름을 따라가야만 드러나는
위반을 실행 중에 멈추게 하는 안전망이다.

**함수 참조.** 함수 참조의 convention entry 대상은 module 수준의 `func_table` 정의
하나가 모아 가지며, `func_ref`는 그 table을 가리킨다. Table은 정의와 참조의 class
인자마다 하나이므로, 같은 정의를 같은 class 인자로 참조하는 `func_ref`들은 table을
공유한다. 그래서 table의 내용은 참조가 넘기는 row가 아니라 정의와 class 인자만으로
정한다. Class 인자는 넘긴 row의 하한을 담지 않으므로, 같은 class 인자를 넘기는
참조들도 하한이 서로 다를 수 있기 때문이다. Table은 정의의 row를 class 인자로 읽은
class가 정확하면 그 class의 대상 하나를 두고, 상한이면 정의의 row의 하한부터 그
상한까지 class마다 대상 하나를 둔다. 참조가 넘기는 row는 정의의 row에 적힌 ability
instance를 그대로 가지므로, 참조된 값의 실제 class는 이 범위 안에 있다. Class `c`의
대상은 정의의 row의 tail인 class 변수의 인자를 람다 본문과 같이 `c`로 좁힌 class
인자가 고르는 인스턴스이며, 대상들을 본문으로 삼아 같은 규칙으로 convention entry를
채운다. 대상이 class 없는 tail 때문에 그 값의 row가 요구하는 것보다 강한 `Cps`
인스턴스이면, 더 약한 convention entry는
[값 delimiter](cps-effects.md#값-delimiter)로 worker를 부르는
[`func_ref` adapter](cps-effects.md#cps-변환의-호출-대상-형상)다.

**생성된 closure.** CPS legalization이 만드는 closure(continuation, handler arm,
delimiter region)는 source callable 값이 아니다. 그 closure type의 callable은 자기
convention entry의 signature 자체이며, closure는 자기를 만든 convention으로만
들어온다. 그래서 다른 convention entry는 모두 trap entry이고 adapter를 두지 않는다.
Adapter는 source 람다와 `func_ref`에만 있으며, legalization이 만들어
`closure.lambda`와 `closure.new`에 명시한다.

**Convention table.** Closure 값은 convention table 참조 하나와 capture로 이루어진다
([closure 저장 layout](ir.md#closure-저장-layout)). Convention table은 closure를 만드는
code 대상(람다, `func_table`)마다 정적으로 하나이고, `Direct`,
`EvidenceDirect`, `Cps` 순서로 convention entry 셋을 담는다. Convention이 곧 table
안의 위치이므로 위치를 따로 정하지 않는다. Table은 생성 뒤 바뀌지 않으며 managed
참조를 담지 않는다.

## 사용자 정의 어셈블리 형식

두 dialect의 `func`와 `lambda`만 custom parse/print를 요구하며, 두 dialect가 같은
형식을 쓴다. `tribute_control`의 두 operation은 convention을 쓰지 않는다.
`tribute_cps.func`는 `convention(direct)`, `convention(evidence_direct)`,
`convention(cps)` 가운데 하나를 반드시 쓰며 그 operation의 convention으로 round
trip한다. Callable type에는 convention을 쓰지 않는다. 출력기는 body에서 convention을
추론하지 않는다.

`tribute_control.lambda`는 본문 하나를 가진다. `tribute_cps.lambda`는
`captures [...]` 뒤에 `entries`와 본문 목록을 쓰며, 본문마다 convention keyword 뒤에
본문을 쓴다. 본문은 convention 순서이고 convention마다 많아야 하나다. Adapter와
trap entry는 쓰지 않는다. 본문이 없는 convention entry는 legalization이
[채우기](#convention-entry) 규칙으로 정한다.

```text
tribute_cps.func @f(%x: T) -> R convention(cps)
    attributes {visibility = "private"} {
  ...
}

tribute_cps.func @decl(%x: T) -> R convention(direct)

%f = tribute_control.lambda(%x: T) -> R
    captures [%captured] attributes {debug_name = "apply"} {
  ...
}

%g = tribute_cps.lambda(%k: !K) -> R row(| 'a1) captures [] entries {
  direct {
    ...
  }
  evidence_direct {
    ...
  }
}
```

`func` 형식은 `func.func`처럼 symbol, 분해된 logical parameter/result, 선택적
추가 attribute와 선택적 body를 출력한다. Body가 있으면 entry block label을
생략하고 signature parameter를 entry argument로 복원하며 declaration은 body가
없다. `lambda` 형식은 `closure.lambda`처럼 SSA result, 분해된 source
parameter/result, 명시적 `captures [...]`, 선택적 추가 attribute와 body 또는 entry
목록을 출력하고 entry label을 생략한다. `captures [...]`는 capture가 없어도
`captures []`로 항상 출력하며 canonical 순서는 captures, 선택적 attributes, body
또는 `entries`다.

Row binder는 위치로 이름을 붙여 `'a0`, `'a1`, …로 쓴다. `func`는 symbol 바로 뒤의
`[...]`에 binder를 순서대로 선언한다. 종류가 `param`이 아니면 이름 앞에 `ability_arg`,
`local`, `body` 중 하나를 쓰고, 합집합의 결과인 binder는 `'a0 = 'a1 | 'a2`처럼 source를
선언 순서로 적는다. Binder가 없으면 `[...]`를 생략한다. 이름은 위치에서 정해지므로
parser는 `i`번째 binder의 이름이 `'ai`가 아니면 거부한다. 이 표기가
`row_binders`와 `row_unions` 속성을 대신한다.

`lambda`는 operation 이름 바로 뒤, 매개변수 앞의 `[...]`에 자기가 양화하는 binder를
인스턴스화 순서로 적는다. `func`의 header와 달리 이것은 둘러싼 `func`의 `local`
binder 이름을 나열할 뿐이며, 종류나 합집합을 쓰지 않고 위치 이름 검사도 하지
않는다. 이 표기가 `quantified_rows` 속성을 대신한다.

```text
%step = tribute_control.lambda['a1](%m: core.i32) -> core.i32 row(| 'a1)
    captures [] {
  ...
}
%s = tribute_control.instantiate %step {row_args = [{tail = 0}]} : !Step0
```

Row는 convention 앞에 `row(@A, @B | 'a1)` 형식으로 쓴다. Instance 목록과 `|` 뒤의
tail binder이며, instance가 없으면 `row(| 'a1)`, tail이 없으면 `row(@A)`다. 닫힌 빈
row는 쓰지 않는다. `lambda`의 row tail은 둘러싼
`func`의 binder 이름을 쓴다. 이름은 위치에서 정해지므로 type alias 안의 tail도 같은
뜻이다.

```text
tribute_cps.func @map['a0](%xs: !List, %f: !F) -> !List row(| 'a0) convention(cps) {
  ...
}

tribute_cps.func @join['a0 = 'a1 | 'a2, 'a1, 'a2](%x: T) -> T row(| 'a0) convention(cps)

tribute_cps.func @main['a0]() -> core.nil row(| 'a0) convention(direct) {
  ...
}
```

간결한 형식은 `ability_row` 외의 non-reserved type attribute가 없는 source
signature에만 쓴다.
그런 attribute가 있으면 `func`/`lambda`는 generic assembly를 써서 complete
`type` attribute 또는 result type expression을 출력한다. Metadata의 소유자는 항상
그 `tribute_control.func_sig` type/alias 하나이며, 같은 alias를 참조하는 여러
operation이 operation-local override를 만들 수 없다. Alias가 없으면
generic type-bearing 위치에 attributed `func_sig` 전체를 inline으로 출력한다.
Operation attribute는 별개의 generic attribute dictionary에 남는다.
`func_ref`, `instantiate`, `call`, `call_indirect`, `return`은 generic assembly가
모든 정보를 손실 없이 표현하므로 custom format을 만들지 않는다.

## Named callable과 symbol

Named callable의 source definition은 body를 가진 Tribute callable이다.
`extern "intrinsic"`은 compiler-reserved directive이며, canonical qualified
source name이 intrinsic identity가 된다. 지원되는 identity와 완전한 logical signature는
mutation 전에 검증하며, unknown directive나 signature mismatch는 lowering input error다.
Compiler intrinsic identity는 패키지 `std`와 사용자 선언을 병합한 AST 전체에서
monomorphization 전에 검증한다. 미사용 generic 선언과 중첩 모듈의 선언도 포함하며, 지원하지 않는 모든
directive를 각 선언의 source span에서 진단한다. 등록된 compiler
intrinsic의 logical callable convention은 항상 `Direct`이다. Intrinsic lowering은
identity의 마지막 독자다. 직접 호출을 모두 낮춘 뒤 identity를 소비하고, 참조가 남지
않은 선언은 지운다. Generic specialization은 base
identity를 concrete declaration으로 transport할 수 있지만, mangled name을 parse하여
identity를 복구하지 않는다. Private runtime helper는 target stage에서만 physical
signature로 만들며
source-logical `adt.typeref`를 받을 수 없다. Ordinary bodyless `extern "C"`는 명시적
trusted/unsafe FFI boundary다. 완전한 signature에 managed semantic parameter나 result가
있어도 허용하지만, 사용자가 Tribute의 representation과 ownership contract를 지킨다는
책임을 진다. Native ABI adapter는 managed argument를 borrowed로, managed result를 fresh
owned transfer로 취급한다. 이것은 compiler intrinsic directive가 아니며, intrinsic
lowering은 계속 supported identity와 complete signature를 요구한다. `C` 이외의
bodyless declaration은 이 trusted FFI policy를 얻지 않는다. Textual attribute, symbol
spelling, 위치 또는 printed IR만으로 generic specialization identity를 복구하거나
승격하지 않는다.

Frontend는 해석된 canonical declaration과 specialization을 qualified symbol에
대응시킨다. 정의, 직접 호출, 함수 참조와 intrinsic declaration metadata는 같은
대응을 사용한다. Import alias나 prelude의 짧은 이름을 이 심볼의 선언 경로로
사용하지 않는다. 컴파일 단위의 정의는 모두 `core.module` 하나에 있으며, 소스 module은
IR의 module 구조가 되지 않는다. 안쪽 module은 부모나 형제의 정의를 가리킬 수 없으므로
소스 module 사이의 참조를 module 중첩으로 표현할 수 없기 때문이다. 그래서 정의의
`sym_name`은 소스 경로를 `::`로 이은 문자열이고 참조는 그 이름 하나를 component로
둔다. 이 문자열에서 소스 경로를 다시 나누어 복원하지 않으며, 구조화된 경로는
frontend가 가진다. 이 대응은 source lookup이나 외부 linkage 계약을 변경하지 않는다.

정의의 이름(식별자)과 링크 이름은 다르다. 정의는 `link_name` 문자열 속성으로 링크
이름을 따로 줄 수 있고, 없으면 `sym_name`이 링크 이름이다. 링크 이름은 symbol 방출과
runtime 바인딩 판정만 읽으며, 참조와 symbol table은 `sym_name`을 쓴다. C linkage의
이름 공간은 하나이므로 frontend는 `extern "C"` 선언에 선언한 이름을 `link_name`으로
둔다. 그래서 다른 소스 module에 있는 같은 이름의 C 선언은 IR에서 서로 다른 정의이고
링크 이름만 같다. 링크 이름이 같은 bodyless 선언은 하나의 외부 symbol을 가리키므로
signature가 같아야 하며, 다르면 target 검증이 거부한다.

Frontend 경계 verifier는 metadata 전체와 module의 callable graph를 mutation 전에
대조한다. Direct call은 qualified symbol을 유일하게 resolve하고 완전한 signature를
맞춰야 한다. Indirect call은 exact `tribute_control.func_sig` signature와 source-logical
callable producer를 요구한다. Return은 enclosing callable의 logical result와 일치해야
한다. Body가 없는 declaration도 body traversal 없이 같은 검사를 받는다.

## Callable 출처

Whole-IR verifier는 두 dialect의 `call_indirect`의 callee가 source signature를
선언에서 얻은 값인지 따라간다. 그 판정 규칙은 값을 만든 operation이, block 인자이면
그 block을 가진 operation이 자기 선언에서 정한다. 규칙은 operation 자신의
선언·속성이나 해소된 선언과의 일치로 근거를 세우고, 다른 값에 기대는 경우(lambda의
`instantiate`, callee의 `call_indirect` 결과, `handle`에 이르는 yield 등)에는 그
값들이 모두 출처를 갖추어야 한다. 규칙이 없는 operation의 결과와 block 인자는 출처가
아니다. 각 값은 그 값을 만든 operation, block 인자이면 그 block을 가진 operation
하나의 규칙으로 판정되므로, 판정은 거쳐 간 모든 값의 conjunction이다. `handle`의
handler에서 그 결과로 되돌아오는 순환은 새 값을 더하지 않으므로 순환 밖에서 이르는
값만 출처를 갖추면 된다. 이 출처 규칙은 handler를 구현하는 전략과 무관하다.

## Evidence 선택 속성

`evidence_plan`은 호출이 callee에게 넘길 evidence를 caller evidence에서 고르는
선택이며 [type-inference.md](type-inference.md#호출의-evidence-선택)가 정한 값을
typechecking 결과로 복사한다. `call`, `call_indirect`, `tail_call`,
`tail_call_indirect`, `resume`, `handle`이 가질 수 있다. `handle`의 선택은 body
evidence를 만들기 전에 적용하며 `mask`만 담는다.

```text
{evidence_plan = [{mask = @"<instance key>"}, {dup = ...}]}
```

- 원소는 key 하나짜리 dictionary다. Key `mask` 또는 `dup`이 연산을, 값이 exact
  ability instance([`ability.decl`](../crates/tribute-ir/src/dialect/ability.rs)의
  symbol 참조)를 나타내며 원소는 순서대로 적용한다.
  한 instance는 한 번만 나온다.
- Key `outer`는 그 instance의 가장 위 handler를 설치한 지점의 evidence로 바꾼다.
  Source-logical operation은 이 key를 쓰지 않는다. CPS legalization이 arm 본문에
  중첩된 handle body 안의 `resume`에만 붙이며, 중첩 한 겹에 한 원소다. 같은
  instance가 여러 번 나올 수 있다.
- Row가 여러 tail의 합집합인 callable에 관한 key는 셋이다.
  `{select = i}`는 둘러싼 callable의 `i`번째 tail의 evidence로 바꾸며 목록의 첫
  원소로만 온다. `{push = instance}`는 고른 tail evidence 위에 둘러싼 evidence가
  가진 그 instance의 가장 위 handler를 얹으며 `select` 뒤에만 온다.
  `{tails = [plan, ...]}`는 callee의 tail마다 선택 하나를 담고 목록의 마지막
  원소로만 온다. 각 선택은 같은 문법의 목록이고 `tails`를 다시 담지 않는다.
  `push`와 `tails`의 선택은 연산이 시작한 evidence(앞선 `outer`를 적용한 것)에서
  읽는다.

  ```text
  {evidence_plan = [{select = 0}, {push = @State}]}
  {evidence_plan = [{dup = @State}, {tails = [[], [{mask = @State}]]}]}
  ```

- 속성이 없으면 그대로 전달한다. 빈 목록은 쓰지 않는다.
- Verifier는 원소 형상과 위치, `handle`의 `mask` 전용 규칙, instance 중복을
  검사한다. `select`의 index는 둘러싼 정의의 [합집합](#tail의-합집합) source 수보다
  작아야 하고, `tails`의 선택 수는 callee 합집합의 source 수와 같아야 한다. 선택이
  row와 맞는지는 typechecking이 책임진다.
- CPS legalization은 선택을 바꾸지 않고 옮긴다. 호출과 `resume`의 선택은 그것이
  만든 evidence-taking `func.call`, `func.tail_call`, `func.call_indirect`,
  `func.tail_call_indirect`와 그 suffix의 `ability.suffix_frame`에, `handle`의 선택은
  `ability.handle`에 같은 이름의 속성으로 둔다. Evidence를 받지 않는 callee로 가는
  호출에는 옮기지 않는다. Frame 펼치기는 frame operation의 선택을 층을 다시 만드는
  transfer와 `ability.handle_dispatch`에 옮긴다.
  `resolve_evidence`가 이를 `effect.mask`/`effect.dup`/`effect.outer`와
  `effect.tail`/`effect.push`/`effect.with_tail`로 만든다
  ([cps-effects.md](cps-effects.md#row-directed-evidence)).

Callable operation의 physical lowering은
[cps-effects.md](cps-effects.md#pre-cps-callable-shape)에만 정의한다.

## 구조화된 continuation 불변 조건

Frontend output은 모든 실행 가능한 region 내부에서 strict ANF다. Strict child는
왼쪽에서 오른쪽으로 정확히 한 번 평가한다. 선택된 case/conditional arm, case
guard, short-circuit 오른쪽 항은 선택된 `scf.*` region 안에 남고 hoist하지
않는다. Handler body와 nested handle body는 독립적인 실행 region이다.
일반 structured control은 기존 `scf.*` dialect에 남으며
`tribute_control_to_cps`가 그 region과 suffix를 재귀적으로 변환한다.

Shared CPS conversion은 남은 operation과 enclosing region exit를 위한 명시적인
logical continuation으로 region을 lower한다:

1. Operation 위치의 continuation은 현재 block의 strict suffix를 포함한다.
2. Case, conditional, short-circuit operation에서는 각 branch가 먼저 해당
   structured operation의 merge에 도달한 뒤 enclosing suffix를 실행하는
   continuation을 받는다. 선택된 branch만 평가한다.
3. Handle body는 delimiter continuation을 받는다. Body가 정상 완료되면
   `completion`에 들어간 뒤 enclosing continuation을 실행한다.
4. `operation_kind = "fn"`인 perform은 continuation을 capture하지 않는다.
   Shared conversion이 tail path로 dispatch하며 자동으로 resume된 operation
   result는 일반적인 남은 block suffix로 흐른다.
5. Resumptive general handler의 resume token은 중단된 body continuation을
   가리킨다. `tribute_control.resume`이 이를 호출한 뒤 arm-local strict
   suffix를 계속 실행한다. Arm이 resume하지 않으면 yield가 일치하는 handle을
   직접 완료하고 중단된 suffix와 completion region을 건너뛴다. Source
   `op -> Never` arm은 token을 받지 않으며 이 non-resuming path만 취할 수 있다.
6. Nested handle은 자체 delimiter를 설치한다. Perform은 둘러싼 callable 또는
   handle body row의 명시 label에 묶인 handler가 처리하며, row tail로 들어온
   operation은 그 사이에 설치된 handler를 지나친다
   ([type-inference.md](type-inference.md#모듈-수준-함수의-관계)). Resume하면 perform과 해당
   handler 사이에서 선택된 모든 structured frame에 다시 진입한다. Resume하지
   않고 완료하면 그 frame을 포기한다.

CPS 변환 뒤 Cps callable은 `Evidence, ability.frame<R>, source args`를 받고
`core.never`로 끝난다. 생성된 completion과 exact resume도 Evidence와
`ability.frame<R>`를 명시적으로 받는다. `lower_continuation_frames`가 이 추상 frame을
nominal `ContinuationFrame<R>`로 펼치며, 펼친 resume은 동적 frame에서 handle 층의
dispatcher를 불변 값으로 다시 만든 뒤 suffix와 nested handle을 계속 실행한다. 세
dispatch 계층의 정확한 형상은 [cps-effects.md](cps-effects.md#dispatch-layers)를 따른다.

이 단일 region/suffix 규칙은 case arm과 guard, conditional, short-circuit
오른쪽 항, nested handle body와 arm, resume path, 그리고 이들을 감싸는 strict
work를 모두 다룬다. AST containment scan이나 construct-specific continuation
convention은 dialect contract에 포함되지 않는다.

## `ability` continuation frame 표면

CPS legalization과 frame 펼치기 사이의 계약이다. 펼치기의 의미와 pipeline 위치는
[cps-effects.md](cps-effects.md#abstract-continuation-frames)를 따른다. 표면은
`ability.frame<R>` type과 `ability.suffix_frame`, `ability.exit`, `ability.handle`,
`ability.perform`, `ability.abort`, `ability.delimit` operation이며, 각각의 형상은
[`ability` dialect 선언](../crates/tribute-ir/src/dialect/ability.rs)이 정한다.

- **`effect.delimit`:** Frame 펼치기가 `ability.delimit`을 같은 operand와 결과를
  가진 `effect.delimit`으로 낮추고, target ABI 경계가 그것을 소비한다.
  `effect.delimit`도 같은 관계를 선언하되 body의 frame은 답 타입을 가진 layout
  참조다. 물리화가 그 답 타입과 `core.never` 결과를 지우므로, target ABI 경계는
  물리화 직후 다른 검증 지점 없이 `effect.delimit`을 소비한다.
- **검증:** Operation verifier는 선언의 타입 관계, frame과 closure의 `R` 일치, arm 표와
  arm closure의 일치, closure의 calling convention을 검사한다. Terminator 위치와
  enclosing callable이 `Cps`인지는 whole-IR 검증이 확인한다. `ability.delimit`을
  제외한 operation은 `Cps` flow에만 온다.
- **소유권과 값 흐름:** 모든 operand는 일반 SSA use다. Frame 값은 불변이며 여러 번
  쓸 수 있다. Resumption의 affine 사용은 펼치기가 runtime 상태로 강제한다.
- **위치:** suffix와 frame operation은 그것을 만든 source operation의 span을, handle과
  perform은 각 source `handle`/`perform`의 span을 쓴다.

`lower_continuation_frames` 뒤에는 이 타입과 operation이 남지 않는다.

## `ability`와 `effect`

`ability.*` represents effect evidence and handler dispatch. Ability operations
are lowered through the effect pipeline; ability-related types may remain until
their target-specific representation is selected.

`effect.*` represents the target-independent ABI between high-level ability
semantics and backend-specific evidence/callable layouts. It carries semantic
inputs such as evidence, ability identity, operation name, payload,
continuation, and handler closures. It must not expose Marker field indices,
handler-table storage layout, closure field positions, or backend function
pointer representation.

## `closure`

`closure.*` represents closure allocation and projection. `closure.closure`
names the closure's callable type, which carries no convention; each
`closure.new` names one entry function per convention, and each closure
transfer names the convention it enters through
([convention entry](#convention-entry)). The closure type and the entries keep
their exact types through shared effect
lowering and target-ABI validation. Only after that validation may target
lowering choose the closure storage layouts, rewriting every type-bearing
surface coherently.
Until then a lowered storage pack keeps its exact closure type through an
unrealized cast, which becomes an upcast to the closure base layout when the
storage layout is selected; this is not semantic type equivalence. A closure
is one object holding its convention table reference and its captured values; the
[closure 저장 layout](ir.md#closure-저장-layout) section defines the operations,
layouts, and casts.

Storage finalization 이후 closure base layout에서 정확한 capture layout으로, 또는
erased reference에서 정확한 closure base layout으로 복원하는
`core.unrealized_conversion_cast`는 generic cleanup에서 no-op으로 소거하지 않는다.
Compiler가 구성한 closure layout의 전체 type identity로 이 경계를 선택하며, 이름이나
비슷한 필드 모양으로 추론하지 않는다. Target은 정확한 cast 결과 타입으로 transfer를
검증한 뒤 emission 전에 복원을 물리화한다. Wasm은 concrete `ref.cast`를 생성하고
native는 pointer 표현으로 변환한다. 이 규칙은 다른 struct나 nominal reference의
일반 변환 정책을 바꾸지 않으며, semantic callable 검증을 대체하지 않는다.

## `adt`

`adt.*` represents target-independent product, sum, array, reference, and
literal operations.

`adt.typeref`는 nominal managed reference type이다. 유효한 값은 이 type 자체로
managed이며 pointer provenance로 managed 여부를 다시 판정하지 않는다.
`adt.ref_null`은 정확히 지정한 `adt.typeref`의 null inhabitant이고 retain/release는
null에 대해 no-op이다. `adt.ref_cast`는 확인된 같은 nominal identity의 managed
reference 사이에서만 허용한다. `core.ptr`는 항상 unmanaged이므로 raw pointer,
allocator result, code address, Evidence, borrowed buffer와 이를 통과하는 cast chain은
`adt.typeref`가 될 수 없다. 이 규칙은 typed frontend 경계에서 reachable IR을 바꾸기
전에 검사한다.

Dynamic `tribute_rt.anyref`에서 `adt.typeref`로 복원하는 명시적인
`core.unrealized_conversion_cast`는 generic cleanup에서 no-op으로 소거하지
않는다. Nominal reference의 정확한 결과 타입은 target 변환까지 보존하며,
native는 pointer 표현으로 물리화하고 Wasm은 concrete `ref.cast`를 생성한다.
이 규칙은 일반 `adt.struct` 복원이나 nominal provenance 검증을 바꾸지 않는다.

### `adt` 배열

`adt.array<E>`는 원소 타입 `E`의 값을 순서대로 담는 배열 object에 대한 managed
참조다. `E`는 scalar, pointer, managed 참조 중 하나이며 원소 하나는 machine 값
하나다. Struct, variant, closure 같은 aggregate는 그 object에 대한 managed 참조로
담고 inline으로 펼치지 않는다.

`core.array<E>`는 이와 다른 unmanaged linear array다. 연속된 memory에 놓인 `E`
값의 나열일 뿐 object가 아니며, 소유권이나 회수 방식을 갖지 않는다. `adt` 배열
operation은 `core.array`를 받지 않으며, 어느 target도 `core.array`를 managed 참조로
다루지 않는다.

배열을 만드는 operation은 `adt.array_new`, `adt.array_of`, `adt.array_splice`
셋이며, 각각 결과 배열의 원소를 정하는 방법이 하나로 고정되어 있다. Operand 개수나
모양으로 operation의 뜻을 고르지 않는다. 각 operation의 결과와 전제는
[`adt` dialect 선언](../crates/tribute-ir/src/dialect/adt.rs)이 정한다. 세
operation의 결과 타입은 `type` 속성의 `adt.array<E>`이고, `initial`과
`elements`는 `E` 타입이다. `adt.array_splice`의 `source`는 결과와 같은 타입이다.

`adt`의 배열 operation은 target이 객체를 회수하는 방식을 가정하지 않는다. Native
RC가 배열 operation에 두는 ownership 규칙과 제자리 쓰기의 순환 제약은
[rc.md](rc.md#비순환-object-그래프)가 정한다.

`adt.array_get`과 `adt.array_set`의 index, `adt.array_splice`의 `start`와 `end`,
`adt.array_new`의 `length`, `adt.array_len`의 결과는
[`core.index`](ir.md#coreindex-주소-폭-정수)다. 배열의 원소 개수와 위치는 target의 주소
폭으로 센다. 고정 폭 정수에서 온 index는 `arith.index_cast`나
`arith.index_castui`로 명시적으로 바꾼다.

Index는 부호 없는 값으로 해석한다. Target은 각 operation이 선언한 범위 전제를
어긴 실행을 trap으로 끝내며, 범위 밖 원소를 읽거나 쓰거나 다른 값으로
대신하지 않는다. Target이 표현할 수 있는 최대 길이나 할당 크기를 넘는 길이도
trap이다.

## `list`

`list.*` represents the opaque canonical `List(a)` sequence contract. It uses
`list.empty`, `list.prepend`, `list.is_empty`, `list.head`, and `list.tail`.
These operations carry element/result types but no variant tags, node field
indices, allocation sizes, or target layout metadata. `list.prepend` is
semantically persistent: it returns a new sequence and does not mutate its tail.
Shared lowering may build a literal by first evaluating all elements left to
right and then applying `list.prepend` in reverse value order.
`list.head` and `list.tail` are internal observation operations with a
non-empty input precondition. Compiler-generated uses must establish
non-emptiness before executing either operation. A backend must trap if the
precondition is violated; it must not return a type-default head, a null tail,
or any other fallback value.
The public `List::prepend(value, tail)` wrapper of package `std` delegates to a
private registry-verified compiler intrinsic, whose calls lower to the same
`list.prepend` operation. A source-defined function merely spelled
`List::prepend` remains an ordinary call. The private intrinsic declaration is a
compiler/standard-library boundary, not an additional public symbol or a layout
contract.

List patterns lower to sequence observations. Exact-length patterns require an
empty remainder; prefix-rest patterns return the remainder as the same canonical
List type. A backend must eliminate `list.*` before its backend-ready boundary.

Source-logical List 패턴의 `list.head` 결과 타입은 `element_type`과 같아야 한다.
둘 다 전체 패턴의 검증된 `List(a)`에서 논리 타입으로 변환한 `a`를 사용한다.
이 계약은 매칭 검사와 성공한 arm의 바인딩에 동일하게 적용한다.
생성자 패턴 노드의 callable 메타데이터는 생성자 해석용으로 보존하며,
원소 값의 타입으로 사용하지 않는다.

List의 logical representation은 compiler-owned nominal identity로 선택한다.
동명 source ADT의 등록은 이 선택을 바꾸지 않는다. 현재 lowering이 사용하는
`tribute_rt.anyref`는 표현상의 선택이며 List의 semantic identity를 대신하지
않는다. Source nominal의 `adt.typeref`와 layout은 동일한 해석된 선언 및
specialization에 대응해야 한다.

## Nominal layout의 게시

Source-logical frontend는 nominal layout을 생성할 때 실제 `adt.struct` 또는
`adt.enum` 타입을 normalized nominal symbol의 IR type alias로 게시한다.
`adt.typeref`의 `name`, layout의 `name`(둘 다 문자열), 게시된 alias의 이름은
텍스트가 일치해야 한다.
생성 연산 없이 signature에만 등장하는 특수화도 같은 계약을 따르며, 필드에서
참조하는 compiler-generated nominal tuple layout도 게시한다. Frontend 내부
type map이나 type interner에만 존재하는 layout은 게시된 정의가 아니다.

<!-- markdownlint-disable-next-line MD033 -->
<a id="compiler-generated-tuple-layout"></a>

Compiler-generated tuple layout의 이름은 원소의 logical type에서 정해지는 함수이며,
원소가 callable type이면 그 [row](#ability-row)를 포함한다. 그래서 row만
다른 두 tuple은 다른 layout이고, [row 치환](#row-인자와-치환)은 원소를 치환한 layout을
같은 함수로 다시 만든다. 이 함수는 IR 계층이 소유하며 frontend와 pass가 함께 쓴다.
Row를 버리는 CPS legalization도 tuple layout을 원소를 변환한 뒤 같은 함수로 다시
만들고, 그 이름을 가리키는 typeref와 게시된 alias를 새 이름으로 바꾼다. 그래서 row만
다른 tuple layout은 legalization 뒤에 하나가 된다. Nominal layout의 필드 type과,
row 인자로 치환한 그 필드 type을 쓰는 `adt` 생성·접근 operation도 같은 tuple layout을
가리키게 된다.

특수화된 enum의 variant 필드 타입은 해당 인스턴스에 함께 치환된 constructor
스킴에서 가져온다. 원본 generic 스킴이나 생성·패턴 표현식으로 필드 표현을
추정하지 않는다. `adt.variant_new`의 operand materialization과
`adt.variant_get`의 결과 타입은 같은 게시된 layout을 따른다. 특수화하지 않은
generic layout의 erased 필드는 기존 boxing과 payload recovery를 유지한다.
Signature나 필드에서만 참조하는 nominal 인스턴스도 의존 layout을 준비하고
게시해야 한다. 재귀 참조는 같은 선언 및 타입 인자의 인스턴스를 공유한다.

CPS와 closure 변환은 게시된 alias와 layout 내부 타입을 함께 변환한다.
Native ownership 검사는 게시된 정의와 IR에서 도달하는 layout만 사용하며,
각 reachable typeref가 같은 이름의 유일한 layout으로 해석되어야 한다.
누락되거나 모호한 layout은 거부하고, interner 전체 검색이나 이름의 일부를
맞추는 방식으로 layout을 추정하지 않는다.

## `tribute.types`

The root `core.module` carries the package `std` type identities the compiler relies
on in one dictionary attribute, `tribute.types`. Each key is the canonical
declaration path of a type of package `std`, and each value is the exact `TypeRef`
produced from that declaration. The entry `std::String` is the `adt.enum` type
of `std::String`; runtime types such as `std::runtime::evidence::Marker`
(`new-plans/modules.md`, "Compiler 소유 runtime 모듈") are entries of the same
dictionary. Lowering consumes an entry directly; it must not rediscover a type
by type name, variant names, or field layout. This metadata is semantic compiler
state rather than a nominal-layout convention. The frontend preserves each
declaration's stable identity through type checking and compares that
identity, not its name, when materializing an entry. Type conversion converts
the entries like any other type-bearing attribute.

The current specialized textual printer for `core.module` does not serialize
arbitrary module attributes. Consequently, parsing printed IR conservatively
drops `tribute.types`. Every lowering that needs an entry fails closed: when
the entry is absent it reports a compiler error and never reconstructs the type
from type names or layout. A backend may still process byte constants, but must
reject `adt.string_const` when the `std::String` entry is absent, and a pass
that builds evidence types or lowers `effect.*` must reject the module when the
`std::runtime::evidence::Marker` entry is absent.

## Root entry 참조

실행 파일 빌드의 root 패키지가 진입점을 선언하면, frontend는 root `core.module`의
`tribute.entry` attribute에 그 함수의 `SymbolRef`를 담는다. 진입점은 언어 규칙이
정한다(root 모듈의 `main`). 이후 단계는 symbol 철자가 아니라 이 참조로 진입점을
찾는다.

- Class 특수화, `fn` arm 호출 검사, root bridge 합성처럼 bridge 전에 root worker를
  다루는 단계는 이 참조가 가리키는 정의를 root worker로 읽는다. 참조는 module
  수준의 symbol 참조이므로, 참조되지 않는 정의 제거는 이 정의를 root로 남긴다.
- 참조가 root module의 정의 하나로 해석되지 않으면 bridge 합성은 오류로 거부한다.
  참조가 없으면 진입점이 없는 module이며 bridge를 합성하지 않는다.
- Root bridge 합성은 참조를 소비한다. Root worker로 이름을 바꾸고 모듈 안의 참조를
  옮긴 뒤 attribute를 제거한다. 그 뒤의 진입점은 bridge가 만든 root `main` wrapper이고,
  target은 이 target ABI symbol로 진입점을 찾는다([출구 계약](abi-boundary.md#representationabi-경계)).
