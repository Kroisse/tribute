# Tribute 고수준 dialect

TrunkIR core 위에서 Tribute가 정의하는 고수준 dialect의 여러 operation에 걸친
규칙이다. TrunkIR 자체의 설계는 [ir.md](ir.md)를 따른다.

<!-- markdownlint-disable-next-line MD033 -->
<a id="direct-style-control"></a>

## 직접형 호출 대상과 제어

`tribute_control.*`은 typed frontend lowering과 실행 전략 사이의
target-independent 직접형 경계다. CPS 전략은 이 경계 뒤에서 convention에 따라
뜻이 달라지는 callable operation을
[`tribute_cps`](#convention과-class-특수화)로 바꾼다. Dialect identifier는
정확히 `tribute_control`이다.

이 dialect는 CPS legalization 전의 Tribute 고유 callable과 effect-control
의미를 함께 소유한다.
산술, ADT/list/tuple/record 구성과 structured selection은 기존 dialect에 남는다.
`func.*`, `closure.*`, `func.func_sig`는 이 경계에 나타나지 않는다.

논리 callable type은 `tribute_control.func_sig<(Params...) -> Result>`이다.
`Params`와 `Result`는 source-logical type이다. 이 type은 calling convention을
갖지 않는다([convention entry](#convention-entry)). `tribute_control`과
`tribute_cps`가 이 type을 함께 쓰며, convention은 `tribute_cps`의 정의, 람다의
convention entry, 간접 이전 operation에 둔다
([convention과 class 특수화](#convention과-class-특수화)). 저장 형식과 type
verifier가 검사하는 것은
[`FuncSig` 선언](../crates/tribute-ir/src/dialect/tribute_control/func_sig.rs)이
정한다.

Type은 한 번 intern되어 모든 phase가 공유하므로 type verifier는 component
type을 검사하지 않는다. 이 경계의 `tribute_control` validator가 검사한다.
Component는 어느 깊이에도 미해결 type, resume token, physical callable type을
담지 않는다. Physical callable type은 calling convention의 ABI로 들어가는
callable의 type이며 `PhysicalCallableType` type interface를 구현한다.
`func.func_sig`, `clif.func_sig`, `wasm.func_sig`, `closure.closure`가 여기에
속한다. `tribute_rt.anyref`와 `core.ptr`는 source-logical type이다.

Convention 값은 keyword 문자열 `"direct"`, `"evidence_direct"`, `"cps"` 가운데
하나다. 이 문자열이 convention의 유일한 저장 형식이다. 정수나 다른 값은 잘못된
convention이다. Convention은 [CPS 전략](cps-effects.md#row와-호출-규약)이
[ability row](#ability-row)에서 계산한 값이며 body에서 추론하지 않는다.

`tribute_cps` operation의 convention은 그 operation이 선언한 속성이다. Physical
operation의 convention은
[`ConventionBearing` interface](cps-effects.md#pre-cps-callable-shape)로만 읽고
쓴다. `closure.lambda`는 본문마다 convention을 자기 `conventions` 속성으로
선언한다. Legalization은 convention을 기존 `CallableAbi`와 physical
`func.func_sig`, `closure.closure`, `tribute.calling_convention` operation
attribute로 바꾼다. 이 type과 convention의 전체 규칙은
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
| `tribute_control.trap` | [trap intrinsic](modules.md#trap-intrinsic)의 직접 호출. 원인을 가지고 실행을 끝내는 terminator |

Class 특수화가 만드는 `tribute_cps` operation은 다음과 같다. `func`,
`call_indirect`, `tail_call_indirect`는 같은 이름의 `tribute_control`
operation에 convention 하나를 더한 것이다. `lambda`는 convention 하나 대신
convention entry마다의 본문 목록을 가지고, `func_ref`는 convention 대신
`func_table` 참조를 가지며, `func_table`은 그 참조가 가리키는 convention table
정의다.

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

`func.constant`, `func.unreachable`의 logical 복제는 없다. Named function
value는 `func_ref`가 표현한다. `func.constant`는 후속 physical closure lowering이
만든다. `func.unreachable`은 reject adapter 같은 compiler helper 안에서만
legalization 뒤에 사용한다. `func.trap`의 source-logical 짝은
[`tribute_control.trap`](#trap)이다.

Source [`become`](syntax.md#tail-call-become)은
`tribute_control.tail_call`/`tail_call_indirect`로 나타난다. 이 두 operation이
`func.tail_call`/`func.tail_call_indirect`의 source-logical 짝이다. 표시 없는
호출을 tail transfer로 바꾸는 것은 보장이 없는 최적화다
([cps-effects.md](cps-effects.md#source-proper-tail-call의-적법화)).
Legalization은 알려진 target에 `func.tail_call`, closure, continuation과
`done_k` target에 새 `func.tail_call_indirect`를 만들 수 있다.

이 dialect는 opaque type `tribute_control.resume_token<input, answer>`도
소유한다. 그 계약과, logical result가 `Never`인 general operation이 resumption을
만들지 않는다는 규칙은
[`resume_token`과 `perform`의 선언](../crates/tribute-ir/src/dialect/tribute_control.rs)이
정한다.

### Trap

`tribute_control.trap`은 [trap intrinsic](modules.md#trap-intrinsic)의 직접
호출이며, 원인(`cause`)을 가지고 실행을 끝낸다. Convention에 따라 뜻이 달라지지
않으므로 evidence와 continuation을 받지 않고, legalization은 어느 convention의
callable 안에서든 cause keyword를 `code`로 가진 `func.trap`으로 바꾼다. 두
operation의 계약은 각자의 선언
([`tribute_control.trap`](../crates/tribute-ir/src/dialect/tribute_control.rs),
[`func.trap`](../crates/trunk-ir/src/dialect/func.rs))이 정한다.

- **Terminator.** 두 operation 모두 [`CallableExit`](ir.md#선언적-operation-schema)다.
  Block의 마지막 operation이며 결과가 없다. Callable 본문을 끝내거나,
  `scf.yield` 대신 region 하나를 고르는 operation(`scf.if`,
  `scf.index_switch`)의 arm을 끝낸다. 그 operation이 tail position일 필요는
  없다. Operation의 결과는 `scf.yield`로 끝나는 arm에서만 오므로, trap은 자리의
  타입에 맞는 값을 만들지 않는다. Trap으로 끝난 arm을 가려내는 일은
  `CallableExit`와 `RegionBranch` 조회가 하며 operation 이름을 검사하지 않는다.
- **`handle`의 region.** `tribute_control.handle`의 body, completion, arm은
  `tribute_control.yield`로 끝나므로 trap이 직접 끝낼 수 없다. 그 안에서는 중첩된
  `scf` operation의 arm을 끝낸다.
- **도달할 수 있는 종료.** `func.unreachable`은 실행이 도달하지 않는 지점의
  표시이고, `func.trap`은 프로그램이 도달할 수 있으며 도달하면 반드시 trap하는
  정의된 종료다. 범위 검사 실패처럼 올바른 compiler가 만든 프로그램도 도달하는
  trap을 `func.unreachable`로 쓰지 않는다. 최적화는 `func.trap`에 이르는 경로를
  실행되지 않는 경로로 가정해 지우지 않는다.
- **원인의 전달.** Cause keyword는 `tribute_control.trap`의 `cause`에서
  `func.trap`의 `code`로, native에서는 `clif.trap`의 `code`로 그대로 내려간다.
  Wasm의 `unreachable` 명령은 operand가 없으므로 `func.trap`을 `wasm.unreachable`로
  낮출 때 원인은 사라진다. 두 target 모두 실행 중에 원인을 출력하지 않는다.
  Native의 기계어 trap도 원인을 구별하지 않으며, 원인은 target dialect까지의 IR을
  읽는 쪽에게만 보인다.

## Ability row

`tribute_control.func_sig`는 callable의 ability row를 `ability_row` type 속성으로
가진다. 속성의 저장 형식과 함수 형태의 `row(..)` 절은
[`FuncSig` 선언](../crates/tribute-ir/src/dialect/tribute_control/func_sig.rs)이
정한다. Row의 `instances`는
[`ability.decl`](../crates/tribute-ir/src/dialect/ability.rs)에 대한 symbol
참조이고, `tail`은 둘러싼 정의의 [row binder](#row-binder) index다.

```text
tribute_control.func_sig<(T) -> R row(@"<instance A>", @"<instance B>" | 'a1)>
```

- **의미:** Row는 [instance의 집합](type-inference.md#기본-규칙)이다. 명시
  label과 tail이 같은 instance를 담는 호출에서 어느 위치가 어느 handler를
  받는지는 type이 담지 않는다. Typechecking이 인스턴스화에서 계산한
  [evidence 선택](#evidence-선택-속성)이 담는다.
- **정체성:** signature의 정체성은 입력, 결과, row의 조합이다. Convention은
  signature에 들어가지 않는다. 같은 row도 인스턴스가 고정한 class에 따라 다른
  convention으로 불릴 수 있다([cps-effects.md](cps-effects.md#row와-호출-규약)).
- **범위:** row는 source-logical 경계 안에만 있다. CPS legalization은
  `func.func_sig`를 만들 때 row를 버리고
  [`instantiate`](../crates/tribute-ir/src/dialect/tribute_control.rs)를
  피연산자로 대체한다. Representation/ABI 경계의 출구는 row와 아래 속성들을
  거부한다.

### Row binder

두 dialect의 `func`는 자기 row binder를 `row_binders` 속성으로 선언한다. 네
종류 `param`, `ability_arg`, `local`, `body`의 뜻과 목록의 순서는
[`tribute_control.func` 선언](../crates/tribute-ir/src/dialect/tribute_control.rs)이
정한다. 양화한(∀로 묶은) binder는 `param`과 `ability_arg`이고, `local`은 값이
람다인 `let`의 지역 scheme이 양화한 변수로 그 람다가
[소유](#지역-람다의-양화와-인스턴스화)한다. 값이 람다가 아닌 binding은 그 지역
scheme의 변수를 인스턴스화하는 사용이 없고, 그 변수는 `body`다.

Binder 종류는 row의 구조만 기록하며 실행 전략을 담지 않는다. CPS 전략의 class
특수화는 tail의 class를 binder 종류에 따라 읽는다.

- `param`: 그 인스턴스가 고정한 class다.
- `ability_arg`: `Cps`다.
- `body`: 정의 안의 어느 callable type의 매개변수 타입에도 나타나지 않으면 요구
  없음이고, 나타나면 `Cps`다([tail 규칙](cps-effects.md#row와-호출-규약)).
  매개변수 타입 안의 [row 인스턴스](#nominal-값-타입) row 인자에 나타난 것도 매개변수 타입에
  나타난 것이다. 이 판정은 binder마다 정의 전체에서 한 번 하므로, 그 binder를
  읽는 모든 자리가 같은 class를 읽는다.
- `local`: 그것을 소유한 람다의 본문마다 다르며,
  [convention entry](#convention-entry)의 class 읽기가 정한다.

정의 안의 모든 tail은 이 목록의 index다. Signature, 중첩된 callable type, block
인자, operation 결과, tuple layout의 원소,
[row 인스턴스의 row 인자](#nominal-row-인자)가 모두 같다. Type은 binder를
도입하지 않는다. 예외는 row 매개변수를 가진 nominal layout 하나이며, 그 binder는
layout 안의 type에만 미친다.
[`ability.operation`](../crates/tribute-ir/src/dialect/ability.rs)의 type은
tail을 가질 수 없다.

### Nominal row 인자

[Row 매개변수](types.md#row-매개변수)를 가진 struct·enum의 layout
(`adt.struct`, `adt.enum`)은 자기 row binder를 type 속성 `row_params`로
선언하고, 그 선언의 값 type인 [row 인스턴스](#nominal-값-타입)는 row
매개변수마다 row 하나를 type 속성 `row_args`로 가진다. 속성의
저장 형식, textual form, 정체성 규칙은
[`adt` dialect 선언](../crates/tribute-ir/src/dialect/adt.rs)이 정한다. Layout의
binder는 layout 안의 type에만 미치고, `row_args`의 tail은 그 type을 쓴 정의의
[binder](#row-binder) index다.

Nominal layout은 row 인자와 상관없이 하나다.
[Tuple layout](#compiler-generated-tuple-layout)은 row를 layout 이름에 담고
둘러싼 정의의 binder를 쓴다.

```text
adt.struct<"Callback"['a0], (tribute_control.func_sig<(core.i64) -> core.i64 row(| 'a0)>)>
adt.struct<"Callback"[row(@"<State(Int)>")]>
adt.struct<"Callback"[row(| 'a2)]>
adt.struct<"Callback"[row()]>
```

- `adt` 생성·접근 operation이 보는 필드 type은 layout의 필드 type에서 layout
  binder를 그 operation이 다루는 값의 `row_args`로
  [치환](#row-인자와-치환)한 것이다. Frontend는 `struct_get`, `variant_get`
  결과와 `struct_new`, `variant_new` 피연산자의 type을 이 치환으로 쓴다.
- Row 인자는 다른 row와 같이 source-logical 경계 안에만 있다. CPS
  legalization은 `row_args`와 `row_params`를 버리고, 바뀐 layout을
  [nominal layout 변환](#nominal-layout-변환) 규칙으로 새 키로 만든다. Row
  인스턴스는 그 `def`의 새 키가 된다. Tuple layout은
  [새 이름](#compiler-generated-tuple-layout)으로 다시 만든다.
  Representation/ABI 경계의 출구는 `row_args`와 `row_params`를 거부한다.

### Tail의 합집합

Row tail이 여러 tail의 합집합으로 선언된 정의는 `row_unions` 속성으로 그
합집합을 적는다. 원소의 형식은
[`tribute_control.func` 선언](../crates/tribute-ir/src/dialect/tribute_control.rs)이
정하며, 각 합집합은 열린 source의 tail을 선언 순서로 담는다.

이 순서가 evidence tail 칸의 번호다. 정의 안의
[`{select = i}`](#evidence-선택-속성)는 자기 row tail을 결과로 하는 합집합의
`i`번째 source를 가리킨다. 호출의 `{tails = [...]}`는 callee 합집합의 source마다
선택 하나를 같은 순서로 담는다.

### 지역 람다의 양화와 인스턴스화

값이 람다인 `let`의 지역 scheme이 양화한 row 변수는 그 람다가 소유한다. 두
dialect의 `lambda`는 `quantified_rows` 속성에 자기가 양화하는 `local` binder의
index를 인스턴스화 순서로 적는다. 이 binder는 둘러싼 `func`의 binder이며,
람다가 새 binder를 만들지는 않는다. 모든 `local` binder는 정확히 한 람다가
소유하고, `quantified_rows`의 원소는 모두 `local` binder다. 람다의 entry
본문들은 같은 binder를 공유하며, entry마다 그 binder에 class를 하나씩 고정한다.
쓰이지 않는 람다는 양화할 사용이 없으므로 그 변수는 `body`다.

이런 람다의 모든 사용은
[`tribute_control.instantiate`](../crates/tribute-ir/src/dialect/tribute_control.rs)를
거친다. 호출뿐 아니라 인자로 넘기거나 다른 값에 담는 사용도 같다. 다른 지역
람다의 본문 안에 있는 사용도 그 자리의 `instantiate`이며, row 인자의 tail은
바깥 람다가 소유한 binder를 가리킬 수 있다. Binder 번호는 정의 하나에서
평평하므로 scope가 중첩되지 않는다. `instantiate`는 type의 row만 치환하며
convention이나 entry를 고르지 않는다. 한 람다 식은 `tribute_control.lambda`
하나이고, 사용처마다 따로 만들지 않는다. Class 특수화도 람다를 값 하나로 두며
[본문](#convention-entry)만 class마다 늘린다.

다른 `let`의 값으로 람다를 다시 묶는 별칭은 람다가 아닌 값이며, 별칭 자리에서
람다를 한 번 인스턴스화한다. 람다나 함수 참조가 아닌 `let`은 row 변수를
양화하지 않으므로([일반화](type-inference.md#generalization)), 그런 binding의
모든 사용은 값을 만든 쪽과 같은 row를 본다. 사용이 고정한 row는 값의 type에도
그대로 들어가고, 아무 사용도 고정하지 않은 row 변수만 `body` binder로 남는다.

별칭이나 계산 결과를 통한 간접 이전과 별칭 자리의 인스턴스화는 그 `body`
binder를 같은 [tail 규칙](cps-effects.md#row와-호출-규약)으로 읽어 같은 class를
고르며, 값은 그 entry를 가진다.

- 그 binder가 정의 안의 어느 매개변수 타입에도 나타나지 않으면 둘 다 요구
  없음으로 읽는다.
- 나타나면 둘 다 `Cps`로 읽는다. 그런 값의 매개변수 row tail이 그 `body`
  binder이면, 그 자리에 넘기는 람다는 매개변수의 callable type으로 만들고, 함수
  참조는 그 type의 `tribute_control.func_ref`로 다시 만든다.

이름 붙은 함수를 그대로 묶은 binding과 그런 binding을 다시 묶은 별칭은, 람다의
별칭과 달리 값을 묶지 않는다. 그 binding의 사용은 묶인 값을 읽지 않고 함수를
다시 참조한다. 그 `tribute_control.func_ref`는 타입 검사가 그 사용에 고른
인스턴스의 row를 넘긴다. 함수 참조 자리의 row 인자는 별칭마다 scheme을
인스턴스화한 row를 사용 쪽에서부터 거슬러 대입해 얻는다.

### Row 인자와 치환

Row-polymorphic 정의를 참조하는 `call`, `tail_call`, `func_ref`는 `row_args`로
callee의 양화된 binder마다 row 하나를 넘긴다. 각 row는 `ability_row`와 같은
형식이며 tail은 참조하는 쪽 정의의 binder index다. Callee에 양화된 binder가
없으면 속성을 생략한다.

참조의 callee symbol은 class 특수화 전에는 정의를, 뒤에는 class 인스턴스 하나를
가리키며, 그 convention은 인스턴스 정의의 것이다. Row 인자의 class가 인스턴스의
class와 맞는지는 [convention class](cps-effects.md#row-변수의-convention-class)
규칙이 정한다.

참조가 보는 callee signature는 callee의 type에 row 인자를 치환한 것이다. 치환은
type 안의 모든 `func_sig`(자기 자신 포함)와 row 인스턴스의 row 인자에서, tail이
양화된 binder `t`인 row `(I, t)`를 `(I ∪ args[t].instances, args[t].tail)`로
바꾸고 정렬한다. 다른 속성은 바꾸지 않는다. Layout 안의 type은 치환하지 않는다.
Tuple layout은 원소를 치환한 뒤 같은 이름 규칙으로 다시
만든다([layout 이름](#compiler-generated-tuple-layout)). `call_indirect`와
`tail_call_indirect`는 row 인자가 없다.

`instantiate`의 `row_args`는 대상 람다의 `quantified_rows` 순서를 따른다.
`i`번째 row 인자가 `quantified_rows`의 `i`번째 binder를 대체하며, 치환 규칙은
위와 같다.

### Convention과 class 특수화

Convention은 `tribute_cps` dialect의 operation만 가진다. Frontend는
`tribute_control`만 내보내며, 그 IR에는 convention이 없다. CPS 전략의
[class 특수화 pass](cps-effects.md#class-특수화-pass)는 class 인스턴스를 만들면서
convention에 따라 뜻이 달라지는 operation 다섯을 같은 이름의 `tribute_cps`
operation으로 바꾼다.

- `tribute_cps.func`는 자기 convention을 가진다. 인스턴스마다 하나다.
- `tribute_cps.lambda`는 class마다의 본문과,
  [convention entry](#convention-entry)마다 그 entry를 채우는 본문을 가진다.
- `tribute_cps.call_indirect`와 `tribute_cps.tail_call_indirect`는 callee를
  부르는 convention을 가진다.
- `tribute_cps.func_ref`는 `tribute_cps.func_table` 하나를 가리키고, 그 table이
  convention entry마다 그 entry로 들어갈 대상 인스턴스를 가진다.

나머지 `tribute_control` operation(`call`, `tail_call`, `return`, `instantiate`,
`perform`, `handle`, `resume`, `yield`)은 그대로 남아 `tribute_cps`
callable의 본문 안에서 쓰인다.

이 변환은 full conversion이다. Class 특수화 뒤에는 위 다섯 operation의
`tribute_control` 형태가 남지 않으며, legalization의 입력 검증은 남은 것을
거부한다.

두 dialect의 callable 값은 모두 `tribute_control.func_sig`를 type으로 쓰며, 이
type은 convention을 갖지 않는다. `tribute_cps`도 row와 logical type을 가진
source-logical IR이며 hidden parameter는 아직 없다.

Convention에 관한 규칙은 다음과 같다.

- `tribute_control.tail_call`의 callee 인스턴스와
  `tribute_cps.tail_call_indirect`의 convention은 감싼 callable(정의, 또는 그
  이전을 담은 entry)보다 강하지 않다.
- 정의의 convention과 간접 이전의 convention은
  [CPS 전략](cps-effects.md#row와-호출-규약)이 그 row와 인스턴스의 class에서
  계산한 값이다.

Type의 일치를 요구하는 규칙은 두 dialect에서 같고, convention을 비교하지 않는다.

### Convention entry

Callable 값과 그 type은 convention을 갖지 않는다. 호출 지점은 둘러싼 인스턴스의
class로 정해지는 자기 convention 하나만 정적으로 알며, callee 값이 어느 class에서
만들어졌는지 묻지 않는다. 그 차이는 값이 가진 convention entry가 흡수한다.

예를 들어 `let wrap = fn(k) { fn() { k() } }`를 순수한 `k`와
`EvidenceDirect`인 `k`로 쓰면, 두 사용은 모두 `Direct`를 고르고 `wrap`은
`Direct` 본문 하나를 가진다. 그 본문이 만드는 `fn() { k() }`는 상한
`EvidenceDirect`의 row를 가지므로 `Direct`와 `EvidenceDirect` 본문을 함께
가지며, 각 본문은 `k`를 자기 convention으로 부른다.

**Convention entry.** Callable 값은 convention마다 convention entry 하나를
가진다. Convention entry는 그 convention으로 들어왔을 때 실행할 물리
callable이며, 본문, adapter, trap entry 가운데 하나다. 물리 ABI(hidden
parameter)는 그 convention이 정하고, convention entry를 구현하는 물리 함수를
entry 함수라 한다.

간접 이전은 자기 convention의 convention entry로 들어가며, 실행 중에 entry를
고르지 않는다. 값마다 convention entry는 `Direct`, `EvidenceDirect`, `Cps` 셋을
넘지 않는다.

**Class 읽기.** Class 특수화는 row의 class를 정확한 class나 상한으로 읽는다.
상한 `c`로 읽은 row는 실행 중에 `c`보다 약하거나 같은 어느 class든 요구할 수
있다는 뜻이다. 상한 `Direct`는 정확한 `Direct`와 같다. Row의 **하한**은 row에
적힌 ability instance들의 requirement를 합친 class이며, row가 실행 중에 요구하는
class는 하한보다 약하지 않다.

- 둘러싼 인스턴스의 class 변수는 그 인스턴스가 고정한 정확한 class나 상한으로
  읽는다.
- 람다가 양화한 `local` binder는 아래 본문 `c` 안에서, class `c`를 고르는
  사용들이 그 binder에 넘기는 row의 class로 읽는다. 그 사용들이 모두 같은 정확한
  class를 넘기면 그 class이고, 그렇지 않으면 넘기는 class들을 합친 상한이다.
  람다 자신의 row에 나타나는지와 관계없다.
- Row의 class는 하한과 tail의 class를 합친 것이다. Tail이 상한이면 row의
  class도 상한이다. 다만 하한이 그 상한과 같으면 정확한 class다.

**본문.** 람다는 다음 class마다 본문 하나를 가지며, 다른 class의 본문은 없다.

- `instantiate`로만 쓰이는 `let` 람다는 사용이 고르는 class마다 본문을 둔다.
  사용의 class는 그 사용의 `instantiate`가 람다 자신의 row에 넘기는 row의
  class다.
- 그 밖의 람다는 자기 row의 class가 정확하면 그 class의 본문 하나를 둔다.
- 사용의 class나 자기 row의 class가 상한이면, 그 row의 하한부터 상한까지의
  class마다 본문을 둔다.

Class `c`의 본문은 위 읽기를 따르되, 자기 row의 tail이 상한이면 그 상한을
`c`로 좁혀 상한으로 읽는다.

**값은 나누지 않는다.** 같은 class를 고르는 사용들이 양화한 binder에 서로 다른
class를 넘겨도 람다는 값 하나다. 앞의 `wrap`이 그 예다. 상한으로 읽어도 결과는
바뀌지 않는다.

**Capture.** 람다의 capture 목록은 하나이며, 모든 본문이 실제 사용하는 외부
SSA 값과 일치한다. Capture는 closure 하나에 한 번만 담긴다.

**채우기.** Class `c`의 convention entry는 다음과 같다.

- 람다에 class `c`의 본문이 있으면 그 본문이다.
- 없고 `c`보다 약한 본문이 있으면, 그 가운데 가장 강한 본문을 부르는
  adapter다.
- 약한 본문도 없으면 trap entry다.

**Adapter.** `EvidenceDirect` entry의 adapter는 evidence를 받고 쓰지 않으며,
`Cps` entry의 adapter는 본문의 결과를 `Done<R>`로 넘긴다. Adapter는 본문을 한
번 부르고 끝나며 CPS 제어를 만들지 않는다.

Adapter는 본문을 이름으로 부르지 않는다. Environment 입력으로 받은 closure
자신의 convention table에서 본문 class의 entry를 `closure.entry`로 읽고, 같은
closure를 넘겨 부른다. Adapter는 람다나 capture에 매이지 않고 자기 entry
signature와 본문의 class로만 정해지며, target은 둘이 같은 convention entry들이
adapter 하나를 공유하게 할 수 있다.

**Trap entry.** Trap entry는 그 convention entry의 signature를 가지고
들어오자마자 trap하는 함수이며, target은 signature가 같은 trap entry들이 하나를
공유하게 한다.

Trap entry는 실행되지 않는다. Class는 row의 tail에서 계산하고, 값이 tail의
정체를 잃을 수 있는 자리(`ability_arg` tail, 매개변수 타입에 나타나는 `body`
tail)는 `Cps`로만 읽는다.

모든 간접 이전의 convention은 그 이전을 담은 인스턴스와 본문의 class 읽기로
callee type을 읽은 class(상한이면 상한)와 같아야 하며, 다르면 컴파일 오류다.

**함수 참조.** 함수 참조의 convention entry 대상은 module 수준의 `func_table`
정의 하나가 모아 가지며, `func_ref`는 그 table을 가리킨다. Table은 정의와
참조의 class 인자마다 하나이므로, 같은 정의를 같은 class 인자로 참조하는
`func_ref`들은 table을 공유한다.

Table의 내용은 참조가 넘기는 row가 아니라 정의와 class 인자만으로 정한다.
Table은 정의의 row를 class 인자로 읽은 class가 정확하면 그 class의 대상 하나를
두고, 상한이면 정의의 row의 하한부터 그 상한까지 class마다 대상 하나를 둔다.

Class `c`의 대상은 정의의 row의 tail인 class 변수의 인자를 람다 본문과 같이
`c`로 좁힌 class 인자가 고르는 인스턴스이며, 대상들을 본문으로 삼아 같은
규칙으로 convention entry를 채운다. 대상이 class 없는 tail 때문에 그 값의 row가
요구하는 것보다 강한 `Cps` 인스턴스이면, 더 약한 convention entry는
[값 delimiter](cps-effects.md#값-delimiter)로 worker를 부르는
[`func_ref` adapter](cps-effects.md#cps-변환의-호출-대상-형상)다.

**생성된 closure.** CPS legalization이 만드는 closure(continuation, handler
arm, delimiter region)는 source callable 값이 아니다. 그 closure type의
callable은 자기 convention entry의 signature 자체이며, closure는 자기를 만든
convention으로만 들어온다. 다른 convention entry는 모두 trap entry이고
adapter를 두지 않는다. Adapter는 source 람다와 `func_ref`에만 있으며,
legalization이 만들어 `closure.lambda`와 `closure.new`에 명시한다.

**Convention table.** Closure 값은 convention table 참조 하나와 capture로
이루어진다([closure 저장 layout](#closure-저장-layout)). Convention table은
closure를 만드는 code 대상(람다, `func_table`)마다 정적으로 하나이고, `Direct`,
`EvidenceDirect`, `Cps` 순서로 convention entry 셋을 담는다. Convention이 곧
table 안의 위치이므로 위치를 따로 정하지 않는다. Table은 생성 뒤 바뀌지 않으며
managed 참조를 담지 않는다.

## 사용자 정의 어셈블리 형식

두 dialect의 `func`와 `lambda`만 손으로 쓴 parse/print를 가지며, 두 dialect는
같은 형식을 쓴다. 나머지 operation은 선언적 형식을 쓴다. 각 형식은 해당
선언이 정한다: row binder header와 `row(..)` 절, `inferred`, `attributes {..}`,
body를 쓰는 `func` 형식은
[`tribute_control.func`](../crates/tribute-ir/src/dialect/tribute_control.rs)가,
양화 binder 목록과 `captures [..]`를 쓰는 `lambda` 형식은
`tribute_control.lambda`가, `convention(..)`과 `entries`는
[`tribute_cps.func`와 `tribute_cps.lambda`](../crates/tribute-ir/src/dialect/tribute_cps.rs)가,
row 절 자체는
[`FuncSig`](../crates/tribute-ir/src/dialect/tribute_control/func_sig.rs)가 정한다.

```text
tribute_cps.func @join['a0 = 'a1 | 'a2, 'a1, 'a2](%x: T) -> T row(| 'a0) convention(cps)

%step = tribute_control.lambda['a1](%m: core.i32) -> core.i32 row(| 'a1)
    captures [] {
  ...
}
%s = tribute_control.instantiate %step {row_args = [{tail = 0}]} : !Step0
```

Convention은 operation의 표기이며 callable type에는 쓰지 않는다. Type alias
안의 row tail은 inline으로 쓴 것과 같은 뜻이다.

간결한 형식은 `ability_row` 외의 non-reserved type attribute가 없는 source
signature에만 쓴다. 그런 attribute가 있으면 `func`/`lambda`는 generic assembly를
써서 complete `type` attribute 또는 result type expression을 출력한다.
Metadata의 소유자는 항상 그 `tribute_control.func_sig` type/alias 하나다. 같은
alias를 참조하는 여러 operation은 operation-local override를 만들 수 없다.
Alias가 없으면 generic type-bearing 위치에 attributed `func_sig` 전체를
inline으로 출력한다. Operation attribute는 별개의 generic attribute
dictionary에 남는다.

## Named callable과 symbol

Named callable의 source definition은 body를 가진 Tribute callable이다.

`extern "intrinsic"`은 compiler-reserved directive이며, canonical qualified
source name이 intrinsic identity가 된다. 등록된 compiler intrinsic의 logical
callable convention은 항상 `Direct`이다. Compiler intrinsic identity는 패키지
`std`와 사용자 선언을 병합한 AST 전체에서 monomorphization 전에 검증한다.
미사용 generic 선언과 중첩 모듈의 선언도 포함한다. 지원되는 identity와 완전한
logical signature는 mutation 전에 검증한다. Unknown directive나 signature
mismatch는 lowering input error이며, 지원하지 않는 모든 directive는 각 선언의
source span에서 진단한다. Intrinsic lowering은 identity의 마지막 독자다. 직접
호출을 모두 낮춘 뒤 identity를 소비하고, 참조가 남지 않은 선언은 지운다.

Generic specialization은 base identity를 concrete declaration으로
transport할 수 있다. Mangled name을 parse하여 identity를 복구하지 않는다.
Textual attribute, symbol spelling, 위치 또는 printed IR만으로 generic
specialization identity를 복구하거나 승격하지 않는다.

Private runtime helper는 target stage에서만 physical signature로 만든다.
Private runtime helper는 source-logical [nominal 값 타입](#nominal-값-타입)을 받을 수 없다.

Ordinary bodyless `extern "C"`는 명시적 trusted/unsafe FFI boundary다. 완전한
signature에 managed semantic parameter나 result가 있어도 허용하며, Tribute의
representation과 ownership contract를 지킬 책임은 사용자가 진다. Native ABI
adapter는 managed argument를 borrowed로, managed result를 fresh owned
transfer로 취급한다. `C` 이외의 bodyless declaration은 이 trusted FFI policy를
얻지 않는다. `extern "C"`는 compiler intrinsic directive가 아니다. Intrinsic
lowering은 계속 supported identity와 complete signature를 요구한다.

Frontend는 해석된 canonical declaration과 specialization을 qualified
symbol에 대응시킨다. 정의, 직접 호출, 함수 참조와 intrinsic declaration
metadata는 같은 대응을 사용한다. Import alias나 prelude의 짧은 이름은 이
symbol의 선언 경로로 쓰지 않는다. 이 대응은 source lookup이나 외부 linkage
계약을 바꾸지 않는다.

컴파일 단위의 정의는 모두 `core.module` 하나에 있으며, 소스 module은 IR의
module 구조가 되지 않는다. 정의의 `sym_name`은 소스 경로를 `::`로 이은
문자열이고, 참조는 그 이름 하나를 component로 둔다. 이 문자열을 다시 나누어
소스 경로를 복원하지 않으며, 구조화된 경로는 frontend가 가진다.

정의의 이름(식별자)과 링크 이름은 다르다. 정의는 `link_name` 문자열 속성으로
링크 이름을 따로 줄 수 있고, 없으면 `sym_name`이 링크 이름이다. 링크 이름은
symbol 방출과 runtime 바인딩 판정만 읽으며, 참조와 symbol table은 `sym_name`을
쓴다. Frontend는 `extern "C"` 선언에 선언한 이름을 `link_name`으로 둔다. 다른
소스 module에 있는 같은 이름의 C 선언은 IR에서 서로 다른 정의이고 링크 이름만
같다. 링크 이름이 같은 bodyless 선언은 signature가 같아야 하며, 다르면 target
검증이 거부한다.

Frontend 경계 verifier는 metadata 전체와 module의 callable graph를 mutation
전에 대조한다. Direct call은 qualified symbol을 유일하게 resolve하고 완전한
signature를 맞춰야 한다. Indirect call은 exact `tribute_control.func_sig`
signature와 source-logical callable producer를 요구한다. Return은 enclosing
callable의 logical result와 일치해야 한다. Body가 없는 declaration도 body
traversal 없이 같은 검사를 받는다.

## Callable 출처

Whole-IR verifier는 두 dialect의 `call_indirect`에서 callee가 source
signature를 선언에서 얻은 값인지 따라간다. 각 값의 판정 규칙은 그 값을 만든
operation이, block 인자이면 그 block을 가진 operation이 자기 선언에서 정한다.
규칙이 없는 operation의 결과와 block 인자는 출처가 아니다.

규칙은 operation 자신의 선언·속성이나 해소된 선언과의 일치로 근거를 세운다.
다른 값에 기대는 규칙(lambda의 `instantiate`, callee의 `call_indirect` 결과,
`handle`에 이르는 yield 등)에서는 그 값들이 모두 출처를 갖추어야 한다.
`handle`의 handler에서 그 결과로 되돌아오는 순환은 새 값을 더하지 않으므로
순환 밖에서 이르는 값만 출처를 갖추면 된다. 이 출처 규칙은 handler를 구현하는
전략과 무관하다.

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

일반 structured control은 기존 `scf.*` dialect에 남는다.
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
   Shared conversion이 tail path로 dispatch하며, 자동으로 resume된 operation
   result는 일반적인 남은 block suffix로 흐른다.
5. Resumptive general handler의 resume token은 중단된 body continuation을
   가리킨다. `tribute_control.resume`이 이를 호출한 뒤 arm-local strict
   suffix를 계속 실행한다. Arm이 resume하지 않으면 yield가 일치하는 handle을
   직접 완료하고 중단된 suffix와 completion region을 건너뛴다. Source
   `op -> Never` arm은 token을 받지 않으며 이 non-resuming path만 취할 수 있다.
6. Nested handle은 자체 delimiter를 설치한다. Perform은 둘러싼 callable 또는
   handle body row의 명시 label에 묶인 handler가 처리하며, row tail로 들어온
   operation은 그 사이에 설치된 handler를 지나친다
   ([type-inference.md](type-inference.md#모듈-수준-함수의-관계)).
   Resume하면 perform과 해당 handler 사이에서 선택된 모든 structured frame에
   다시 진입한다. Resume하지 않고 완료하면 그 frame을 포기한다.

CPS 변환 뒤 Cps callable은 `Evidence, ability.frame<R>, source args`를 받고
`core.never`로 끝난다. 생성된 completion과 exact resume도 Evidence와
`ability.frame<R>`를 명시적으로 받는다.

`lower_continuation_frames`가 이 추상 frame을 nominal `ContinuationFrame<R>`로
펼친다. 펼친 resume은 동적 frame에서 handle 층의 dispatcher를 불변 값으로 다시
만든 뒤 suffix와 nested handle을 계속 실행한다. 세 dispatch 계층의 정확한 형상은
[cps-effects.md](cps-effects.md#dispatch-layers)를 따른다.

이 단일 region/suffix 규칙은 case arm과 guard, conditional, short-circuit
오른쪽 항, nested handle body와 arm, resume path, 그리고 이들을 감싸는 strict
work를 모두 다룬다. AST containment scan이나 construct-specific continuation
convention은 dialect contract에 포함되지 않는다.

## `ability` continuation frame 표면

CPS legalization과 frame 펼치기 사이의 계약이다. 펼치기의 의미와 pipeline 위치는
[cps-effects.md](cps-effects.md#abstract-continuation-frames)를 따른다. 표면은
`ability.frame<R>` type과 `ability.suffix_frame`, `ability.exit`,
`ability.handle`, `ability.perform`, `ability.abort`, `ability.delimit`
operation이며, 각각의 형상은
[`ability` dialect 선언](../crates/tribute-ir/src/dialect/ability.rs)이 정한다.

- **`effect.delimit`:** Frame 펼치기가 `ability.delimit`을 같은 operand와 결과를
  가진 `effect.delimit`으로 낮추고, target ABI 경계가 물리화 직후 다른 검증
  지점 없이 그것을 소비한다. 두 operation의 계약은 각 선언이 정한다.
- **검증:** Operation verifier는 선언의 타입 관계, frame과 closure의 `R` 일치,
  arm 표와 arm closure의 일치, arm closure의 형상을 검사한다. Closure type은
  convention을 적지 않으므로 convention은 검사하지 않는다. Terminator 위치와
  enclosing callable이 `Cps`인지는 whole-IR 검증이 확인한다. `ability.delimit`을
  제외한 operation은 `Cps` flow에만 온다.
- **소유권과 값 흐름:** 모든 operand는 일반 SSA use다. Frame 값은 불변이며 여러
  번 쓸 수 있다. Resumption의 affine 사용은 펼치기가 runtime 상태로 강제한다.
- **위치:** suffix와 frame operation은 그것을 만든 source operation의 span을,
  handle과 perform은 각 source `handle`/`perform`의 span을 쓴다.

`lower_continuation_frames` 뒤에는 이 타입과 operation이 남지 않는다.

## `ability`와 `effect`

`ability.*`는 effect evidence와 handler dispatch를 표현한다. Ability operation은
effect pipeline을 통해 lower된다. Ability 관련 type은 target-specific 표현이
선택될 때까지 남을 수 있다.

`effect.*`는 high-level ability semantics와 backend-specific evidence/callable
layout 사이의 target-independent ABI를 표현한다. `effect.*`는 evidence, ability
identity, operation name, payload, continuation, handler closure 같은 semantic
input을 운반한다. `effect.*`는 Marker field index, handler-table storage layout,
closure field 위치, backend function pointer 표현을 노출하지 않는다.

## `closure`

`closure.*`는 closure allocation과 projection을 표현한다. Closure는 convention
table reference와 captured value를 가진 하나의 object다.

`closure.closure`는 closure의 callable type을 명명하며, 그 type은 convention을
갖지 않는다. 각 `closure.new`는 convention마다 entry function 하나를 명명한다. 각
closure transfer는 자신이 들어가는 convention을 명명한다
([convention entry](#convention-entry)).

Closure type과 entry는 shared effect lowering과 target-ABI validation을 거치는
동안 exact type을 유지한다. Target lowering은 그 validation 뒤에만 closure
storage layout을 선택할 수 있으며, 이때 모든 type-bearing surface를 일관되게
재작성한다. Operation, layout, cast는
[closure 저장 layout](#closure-저장-layout) 절이 정의한다.

그 전까지 lowered storage pack은 unrealized cast로 exact closure type을 유지한다.
이 cast는 storage layout을 선택할 때 closure base layout으로의 upcast가 된다.
이는 semantic type equivalence가 아니다.

Storage finalization 이후 closure base layout에서 정확한 capture layout으로, 또는
erased reference에서 정확한 closure base layout으로 복원하는
`core.unrealized_conversion_cast`는 generic cleanup에서 no-op으로 소거하지
않는다. Compiler가 구성한 closure layout의 전체 type identity로 이 경계를
선택하며, 이름이나 비슷한 필드 모양으로 추론하지 않는다. Target은 정확한 cast
결과 타입으로 transfer를 검증한다. 그 뒤 emission 전에 복원을 물리화한다. Wasm은
concrete `ref.cast`를 생성하고 native는 pointer 표현으로 변환한다. 이 규칙은 다른
struct나 nominal reference의 일반 변환 정책을 바꾸지 않으며, semantic callable
검증을 대체하지 않는다.

### Closure 저장 layout

Closure 값은 [convention table](#convention-entry) 참조와 capture한 값을 함께
담은 객체 하나다. 별도의 environment 객체를 두지 않으며, closure를 만드는 일은
할당 하나이고 capture를 읽는 일은 closure의 필드 접근 하나다. Closure의 의미적
타입은 capture와 무관하게 `closure.closure<S>` 하나이고, capture마다 다른
layout은 저장 표현에만 있다.

#### Shared IR 표현

- `closure.new`는 entry 함수 목록과 capture 값을 순서대로 받아 closure를 만든다.
  Entry 함수 목록은 [convention table](#convention-entry)의 순서를 따르는 셋이다.
  같은 목록을 가진 `closure.new`들은 convention table 하나를 공유한다.
  Capture 목록은 비어 있을 수 있다. Environment를 따로 받지 않는다.
- Entry 함수의 environment 입력은 그 closure 자신이다. 입력의 타입은
  `closure.new` 결과의 closure 타입이고, 위치는 그 entry의 바깥 convention이 정한
  environment index다. 이 입력의 ownership 계약은 다른 물리 입력과 같다
  ([출구에서 확정되는 물리 계약](abi-boundary.md#출구에서-확정되는-물리-계약)).
- Entry 함수는 environment 입력에서 capture를 `closure.unpack`으로만 읽는다.
  `closure.unpack %self`의 결과 타입 목록은 capture 타입 목록이며, entry 함수를
  참조하는 모든 `closure.new`의 operand 타입 목록과 정확히 같아야 한다. Whole-IR
  검증은 이 일치를 entry 함수와 그 참조에서 확인하며 struct 이름이나 필드 모양으로
  대신하지 않는다. Capture가 없는 entry 함수는 `closure.unpack`을 갖지 않는다.
- `closure.entry`는 closure의 convention table에서 convention 하나의 entry 함수
  참조를 읽는다. Closure 호출은 자기 convention의 entry를 읽어 호출하고
  closure 값 자체를 environment 입력으로 넘긴다. Environment를 따로 꺼내는
  projection은 없다.
- Lambda lifting, CPS continuation(`Done`, `Dispatch`, resume), `func_ref` entry,
  target 진입 bridge가 만드는 frame closure를 포함한 모든 closure 생성은
  `closure.new` 하나로 표현한다. Compiler가 만드는 closure도 같은 convention entry 규칙을
  따르며, 한 convention으로만 불리는 closure는 그 convention과 더 강한 convention의
  entry만 의미가 있다. 여러
  closure가 공유하는 가변 상태는 managed cell 참조를 capture해서 표현하며,
  closure의 필드는 생성 뒤 바뀌지 않는다.

#### Layout

Closure 저장 layout은 두 종류다.

- **Closure base**(`layout = "closure"`)는 convention table 필드 하나를 가진다. Closure
  값은 언제나 object에 담기며, 그 box `heap.box<base>`가 모든 closure 값의 저장
  타입이다. Closure를 저장하거나 전달하는 위치(signature, block 인자,
  `ContinuationFrame<R>`의 필드, evidence runtime의 `Dispatch`)는 이 타입을 쓴다. Capture가
  없는 closure는 이 layout으로 할당한다.
- **Closure capture layout**(`layout = "closure_captures"`)은 base의 필드 뒤에
  capture를 `closure.new`의 순서대로 두고, type 속성 `closure_base`로 그 base를
  가리킨다. Capture가 하나 이상인 closure는 이 layout으로 할당한다.
- `closure_base`는 closure 저장 계약이 소유하는 속성이다. `layout`처럼 canonical
  생성자만 붙이고 interning identity에 참여하며, 필드 타입을 바꾸는 타입 변환은
  이 속성 안의 base도 같은 변환으로 바꾼다. Capture layout의 앞부분 필드는 base의
  필드와 타입과 속성이 같다. Type verifier가 이 규칙을 `"closure_captures"` layout에
  적용한다. 이 속성은 저장 표현만 설명하며, Tribute 소스와 의미적 타입 검사에는
  subtyping이 없다.
- Capture layout의 box는 `heap.cast` 하나로 base의 box 자리에 놓인다(upcast). 이
  cast는 참조만 바꾸며 object를 바꾸지 않는다. Base의 box를 capture layout의 box로
  보려면 정확한 결과 타입을 가진 명시적 cast가 필요하며, target이 그 cast를 검증하고
  물리화한다. Target은 base와 capture layout의 관계를
  `closure_base`로만 정하고, 필드 모양이 prefix처럼 보인다는 사실에서 추론하지
  않는다.
- Capture layout의 identity는 base와 capture 필드 타입 목록이다. Base와 capture
  타입 목록이 같은 두 closure는 같은 layout을 쓴다. Base와 capture layout은 이름
  없는 [literal layout](#adtstruct-nominal-layout-type)이다.
- Convention table 필드의 타입은 경계 안에서 `tribute_rt.convention_table_ref`이고,
  target이 자기 정적 table 참조 표현으로 바꾼다. 이 필드는 managed 참조가 아니며
  해제하지 않는다. Target은 closure의 callable type `S`마다 base를
  정한다. 모든 `S`에 base 하나를 쓸 수도 있고 `S`마다 다른 base를 둘 수도 있다.
  `S`마다 base를 두고 entry를 typed 함수 참조로 표현하면, entry 함수의 environment
  입력이 다시 그 base이므로 base와 entry 함수 타입은 서로를 가리킨다. 이 재귀는
  base를 [식별 타입](ir.md#식별-타입)인 nominal layout으로 두고 그 키로 끊는다.
- Capture layout의 `closure_base`는 `closure.new` 결과 타입의 `S`에서 고른 base다.

#### Storage 확정

[Representation/ABI 경계](abi-boundary.md#representationabi-경계)의
closure operation lowering과
storage 확정은 다음 변환을 함께 적용한다.

- `closure.new`는 convention table 참조와 capture를 필드로 가진 `heap.struct_new`
  하나가 된다. Capture가 있으면 결과는 capture layout의 box에서 base의 box로의
  upcast이고, 없으면 base 할당 그대로다. Table 참조는 entry 함수 목록을 담은
  `tribute_rt.convention_table`이 만든다.
- `closure.entry`는 base의 convention table 필드를 `heap.struct_get`으로 읽고 그
  table에서 `tribute_rt.convention_entry`로 convention의 entry를 읽는 일이 된다. 간접
  호출은 closure 필드 하나와 table 원소 하나를 읽는다.
- 두 `tribute_rt` operation은 경계 출구를 지나며, target 단계가 convention table을 entry
  함수 목록마다 정적 data 하나로 한 번 만든다.
- `closure.unpack`은 environment 입력을 base의 box에서 정확한 capture layout의 box로
  바꾸는 `heap.cast` 하나와 capture 필드 읽기가 된다. 이 cast는 위의 복원 규칙을
  따르며, Wasm에서는 `ref.cast`, native에서는 pointer 그대로다.
- `closure.closure<S>`는 모든 type-bearing surface에서 `S`의 base의 box가 된다.
- Capture 필드의 타입은 capture 값의 `adt` 값 타입이다. [표현 선택](abi-boundary.md#표현-선택)이
  다른 layout의 필드처럼 그 저장 표현을 정한다.
- Closure 값이 일반 reference erasure 자리(boxed source value, effect payload 등)를
  지날 수 있지만, 거기서 closure로 돌아오는 cast는 exact base를 결과 타입으로
  명시한다.
  Closure를 저장하는 compiler 소유 layout의 필드는 erased reference가 아니라 base를
  쓴다.

## `adt`

`adt.*`는 target에 독립적인 product, sum, array 값과 literal operation을
나타낸다. `adt`는 값 의미론이다. `adt` 타입의 값은 object에 대한 참조가 아니라 값
자체이며, 제자리 쓰기, null, 참조 사이의 cast는 `adt`에 없다. 값이 object에 담기는지,
SSA 값으로 쪼개지는지, 다른 object 안에 inline으로 놓이는지는
[표현 선택](abi-boundary.md#표현-선택)이 정한다. Object와 그 참조는
[`heap`](#heap)이 나타낸다.

`core.ptr`는 항상 unmanaged다. Raw pointer, allocator result, code address,
Evidence, borrowed buffer와 이를 통과하는 cast chain은
[nominal 값 타입](#nominal-값-타입)이 될 수 없다. 이 규칙은 typed frontend 경계에서
reachable IR을 바꾸기 전에 검사한다.

값이 지워지는 자리(effect payload, 제네릭으로 지운 값 등)에서 nominal 값 타입과
`tribute_rt.anyref` 사이의 명시적인 `core.unrealized_conversion_cast`는 generic
cleanup에서 no-op으로 소거하지 않는다. 표현 선택은 이 cast를, 값을 object에 담는
일(또는 담긴 값을 꺼내는 일)과 `heap.box`와 `anyref` 사이의 명시적
`core.unrealized_conversion_cast`로 바꾼다. 이 cast는 참조의 타입만 바꾸므로 출구를
지날 수 있다. 출구 검증이 거부하는 것은 scalar와 `anyref` 사이의 cast뿐이다. 정확한
결과 타입은 target 변환까지 보존한다. Native는 이를 pointer 표현으로 물리화하고
Wasm은 concrete `ref.cast`를 생성한다.

### `adt.struct` nominal layout type

`adt.struct`는 struct layout이다. 필드 타입의 위치 목록을 가지며 전용 textual
문법을 쓴다. 필드 이름은 layout이 아니라 source 선언의
[nominal 선언](#nominal-선언)이 소유하며, 필드는 위치로만 가리킨다.

이름이 있는 layout은 nominal이고, 이름이 없는 layout은 literal이다(LLVM의
identified struct와 literal struct). Nominal layout은
[식별 타입](ir.md#식별-타입)이며 식별자는 이름이다. 이름은 식별자 여부와 상관없이
항상 따옴표 문자열로 쓴다. Struct 이름은 타입 이름공간의 이름이며 symbol table의
정의가 아니므로 `@`를 붙이지 않는다.

```text
adt.struct<"Point", (core.i32, core.i32)>
adt.struct<"Node", (core.i32 {k = "v"}, adt.struct<"Node">)>
adt.struct<"Empty", ()>
adt.struct<"geo::Point", (core.f64, core.f64)>
adt.struct<"Point">
adt.struct<"Callback"[row(@"<State(Int)>")]>
adt.struct<(core.ptr), {layout = "closure"}>
adt.struct<(core.ptr, core.i32),
  {layout = "closure_captures",
   closure_base = adt.struct<(core.ptr), {layout = "closure"}>}>
```

- Literal layout의 저장 표현은 다음과 같다. 필드 타입은 `params`에 선언 순서대로
  둔다. 필드 하나의 속성은 그 필드의
  [매개변수 속성](ir.md#타입-매개변수-속성)이며, textual form은 필드 타입 뒤에
  쓴다.
- Nominal layout의 키는 매개변수 없이 type 속성만 가진다. Struct 이름인 `name`
  (문자열), row binder 수인 `row_params`, continuation frame의 결과 타입 같은 그
  밖의 키 속성이다. [몸체](ir.md#식별-타입)는 같은 필드와 필드 속성을 가진 속성
  없는 literal `adt.struct`다. 필드 접근과 layout 계산은 몸체의 필드를 읽는다.
- Nominal layout은 source가 선언한 struct, compiler가 만든
  [tuple layout](#compiler-generated-tuple-layout), continuation frame처럼 값이
  nominal 타입으로 가리키는 layout이다. 그 값의 타입은
  [nominal 값 타입](#nominal-값-타입)이다.
- Literal layout은 nominal 타입으로 가리킬 일이 없는 compiler 저장 layout이다. Closure
  저장, list node, ability payload, completion cell 같은 layout이며, 정체는 필드
  타입과 type 속성뿐이다. 필드가 같은 두 literal layout은 같은 타입이다. Compiler
  소유 layout은 이름이 아니라 [`layout`](ir.md#runtime-layout-식별자) 속성으로
  식별한다.
- 필드가 없는 struct는 매개변수가 없는 literal이거나 그런 literal을 몸체로 가진
  nominal layout이다. 필드 목록을 담는 별도 type 속성은 없다. Type verifier는
  이름 있는 `adt.struct`의 매개변수와 이름 없는 `adt.struct`의 `row_params`를
  거부한다.
- Textual form은 키를 `"이름"`과 binder, `<…>` 안 마지막 원소인 나머지 키
  속성으로 쓰고, 몸체를 쓰는 자리에서는 이름 뒤에 몸체의 필드 목록을 쓴다.
  `adt.struct<"Point">`는 키만 쓴 것이다. Type 속성의 `name`, `row_params`,
  `param_attrs`는 전용 문법이 소유하며 type 속성 dictionary에는 쓰지 않는다.
  Literal layout은 필드 목록과 `layout` 같은 type 속성을 쓴다.
- Literal layout의 필드 타입은 매개변수이므로 일반 타입 순회와 변환이 그대로
  도달한다. Nominal layout의 필드는 몸체에 있으므로, 필드 타입을 바꾸는 변환은
  아래 규칙으로 새 키를 만든다. 필드 수를 유지하는 변환은 필드 속성을 위치 그대로
  둔다.
- 재귀 참조는 몸체가 자기 키를 담아 표현한다. 몸체는 정체에 들어가지 않으므로
  재귀는 [식별 타입](ir.md#식별-타입)의 키에서 끝난다.
- Row 매개변수를 가진 선언의 layout은 type 속성 `row_params`를 가지며, 필드 type의
  tail은 그 binder의 index다([Nominal row 인자](#nominal-row-인자)). Textual
  form은 binder를 이름 바로 뒤에 쓴다: `adt.struct<"Callback"['a0], (...)>`.
- 이름도 매개변수도 없이 `def`와 `row_args`만 가진 `adt.struct`는 row 인스턴스다.
  [Nominal 값 타입](#nominal-값-타입)에서 정한다.

#### Nominal layout 변환

몸체는 키마다 한 번만 묶이므로, 필드 타입을 바꾸는 변환은 바뀌는 layout마다 새
키를 만든다. `adt.struct`와 [`adt.enum`](#adtenum-nominal-layout-type)에 같은
규칙을 적용하며, `adt` dialect가 이 규칙을 하나의 helper로 소유하고 layout을
변환하는 pass가 함께 쓴다.

- 변환은 module과 게시된 alias가 닿는 nominal layout을 모두 본다. 키 속성이
  바뀌거나 몸체의 필드 타입이 바뀌는 layout, 그리고 키나 몸체가 그런 layout의 키나
  row 인스턴스를 담는 layout이 바뀐다. 나머지 layout은 키와 몸체를 그대로 둔다.
- 바뀌는 layout의 새 키는 변환한 키 속성이다. 키 속성이 그대로이면 새 키의 이름은
  옛 이름 뒤에 `$`와 변환의 tag를 붙인 것이다(`"Point$abi"`).
- 새 키를 모두 정한 뒤에 몸체를 변환해 묶고, 옛 layout을 가리키던 alias를 새 키로
  다시 게시한다.
- 이름에서 tag를 떼어 원래 이름을 되찾지 않는다. Source 이름이 필요한 단계는
  [nominal 선언](#nominal-선언)에서 읽는다.
- [Compiler-generated tuple layout](#compiler-generated-tuple-layout)의 이름은
  원소에서 정해지므로 tuple은 위 두 규칙의 예외다. 원소가 바뀐 tuple layout의 새
  키는 변환한 원소로 이름 함수가 다시 계산한 이름이며 tag를 붙이지 않는다.
  그래서 원소가 담은 바뀌는 layout의 새 키를 먼저 정한 뒤에 tuple의 새 키를
  정한다.

#### Nominal 수준과 structural 수준

Struct 이름은 IR에서 nominal layout을 해석하는 단계까지만 의미를 가진다.
표현 선택, ownership 계획, ABI 검증, frontend의 record pattern 해석이 여기에 속한다.
그 아래의 target lowering은 필드의 물리 표현과 순서만 사용한다. 이름이 필요한 runtime
동작(해제, 값 출력, variant 판별)은 [runtime 타입 descriptor](runtime-types.md)가
맡는다. Nominal layout을 마지막으로 해석하는 target 경계가 할당에 descriptor 번호를
새긴다.

Nominal layout은 두 수준을 거친다.

- 표현 선택부터 descriptor 번호가 정해질 때까지 object의 layout은
  [`heap.box<L>`](#heap)의 `L`인 `adt` layout이다. Descriptor는 nominal identity로
  정하므로 이 구간에서 `L`은 nominal layout이어야 한다. 재귀 layout은 식별 타입의
  키로 자기를 가리키고, enum layout의 box는 그 enum의 어느 variant object든 가리킨다.
  구조적인 저수준 struct로는 이 셋을 나타낼 수 없다.
- 번호를 할당에 새긴 뒤에는 이름이 필요 없다. 그 아래에서 object 접근은 이름 없는
  저수준 struct로 field의 물리 위치만 읽는다.

저수준 struct 타입이 갖는 identity는 runtime 동작이 달라지는 경우로 한정한다.
Layout과 runtime 동작이 같은 두 소스 타입은 같은 저수준 struct를 쓰고
descriptor로만 구별된다.

- Native에서 `heap.box<L>` 값은 native 타입 변환이 pointer로 바꾼다. 저수준 struct는
  값의 타입이 아니라 field 접근 operation의 layout 속성에만 남는다. Nominal layout을
  마지막으로 해석하는 경계에서 struct field 접근의 `adt` layout을 이름 없는
  `mem.struct<T...>`로 내린다. `mem.struct`는 필드 타입만 갖고 자연 정렬 memory
  layout을 뜻한다. 필드 타입은 target 표현이며, 해제 동작이 다른 managed
  참조(`tribute_rt.anyref`)와 unmanaged 포인터(`core.ptr`)는 크기가 같아도 구분한다.
  Inline으로 놓인 필드는 중첩된 `mem.struct`이며 그 offset도 같은 자연 정렬 규칙으로
  정한다. 할당이 해제하는 필드가 managed 참조이며, 이 판정은 ownership 계획의 것이고
  물리 타입에서 다시 유도하지 않는다. `mem.struct`의 field offset과 크기는 같은
  nominal layout에서 계산한 할당 크기와 일치한다.
- 할당은 descriptor를 nominal layout 타입으로 찾으므로 RC header를 새기는 단계가
  nominal layout의 마지막 사용처다. Field offset 계산은 타입 변환 없이
  `mem.struct`만 읽고, field 접근이 `clif.load`와 `clif.store`의 offset이 될 때
  `mem.struct`도 사라진다. Cranelift에는 aggregate 타입이 없다.
- Wasm에서 `heap.box<L>`은 GC 참조가 된다. Nominal layout을 마지막으로 읽는
  `heap` lowering 뒤에 `L`을 이름 없는 `wasm_gc.struct<T...>`로 바꾼다. 이 치환은
  함수 signature와 block 인자를 포함한 모듈 전체에 적용되므로 모든 위치가 같은
  타입을 가진다. Variant는 처음부터 자기 필드의 `wasm_gc.struct`로 낮춘다. 필드
  표현이 같은 struct와 variant는 같은 GC 타입이며, 첫 필드의 descriptor가 runtime
  identity를 맡는다. Inline으로 놓인 필드는 그 leaf들이 바깥 GC struct의 필드가
  된다. Compiler 소유 layout은 [`layout`](ir.md#runtime-layout-식별자)으로 식별한다.
- 저수준 struct는 재귀하지 않는다. 재귀 참조는 이미 native pointer나 Wasm 추상
  reference로 끊겨 있다.
- 두 target 모두 [`adt.enum`](#adtenum-nominal-layout-type)의 variant 객체를 자기
  필드만 가진 저수준 struct로 두고, 어느 variant인지는 값의
  [descriptor](runtime-types.md#variant-판별)로 판별한다. Native에서 필드는 payload의
  offset 0부터 자연 정렬로 놓이고, variant field 접근은 struct field 접근과 같이 그
  variant의 `mem.struct`를 읽는다. Variant 객체는 자기 필드의 크기로 할당하므로
  enum 타입의 값은 하나의 정적 할당 크기를 갖지 않는다. 그 값의 해제 크기는 값의
  descriptor가 정한다.

### `adt.enum` nominal layout type

`adt.enum`은 nominal enum layout이다. 이름과 variant의 위치 목록을 가지며 전용
textual 문법을 쓴다. Variant 이름과 필드 이름은 source 선언의
[nominal 선언](#nominal-선언)이 소유한다. 이름 표기 규칙은
`adt.struct`와 같다. 이름 있는 `adt.enum`은 `adt.struct`의 nominal layout처럼
이름을 식별자로 하는 [식별 타입](ir.md#식별-타입)이고, 이름 없는 `adt.enum`은 그
몸체로만 쓰는 literal이다.

```text
adt.enum<"Option", [(), (T)]>
adt.enum<"List", [(), (core.i32, adt.enum<"List">)]>
adt.enum<"geo::Shape", [(), (core.f64, core.f64 {k = 1})], {k = "x"}>
adt.enum<"Never", []>
adt.enum<"Option">
adt.enum<[(), (core.i32)]>
```

- Variant는 괄호로 감싼 필드 타입 목록이다. 필드가 없어도 괄호를 쓴다. 필드 타입
  뒤에 그 필드의 속성 dictionary가 뒤따를 수 있다.
- Literal `adt.enum`의 저장 표현은 다음과 같다. Variant 하나가 타입 매개변수
  하나이며 선언 순서대로 놓인다. Variant는 `adt.variant` 타입이고, 그 `params`가
  필드 타입이다. 필드의 속성은 그 필드의
  [매개변수 속성](ir.md#타입-매개변수-속성)이다. 필드가 같은 두 variant는 같은
  `adt.variant` 타입이며, variant는 위치로만 구별한다. 매개변수가 없으면
  variant가 없는 enum이다. Variant 목록을 담는 별도 type 속성은 없다.
- 이름 있는 `adt.enum`의 키와 몸체는 `adt.struct`의 nominal layout과 같다. 키는
  매개변수 없이 `name`, `row_params`와 그 밖의 키 속성을 가지며, 몸체는 속성 없는
  literal `adt.enum`이다. 이름 없는 `adt.enum`은 type 속성을 갖지 않는다. Type
  verifier가 이 규칙을 모든 interned `adt.enum`에 적용한다.
- Textual form은 `adt.struct`와 같이 키를 쓰고, 몸체를 쓰는 자리에서는 이름 뒤에
  variant 목록을 쓴다. Type 속성의 `name`과 `row_params`는 전용 문법이 소유하며
  속성 dictionary에는 쓰지 않는다. 나머지 키 속성은 `<…>` 안 마지막 원소로 둔다.
- 필드 타입을 바꾸는 변환은
  [nominal layout 변환](#nominal-layout-변환) 규칙을 따른다.
- Variant 매개변수는 [매개변수 속성](ir.md#타입-매개변수-속성)을 갖지 않는다. 전용
  문법에 그것을 쓸 자리가 없으므로 `adt.enum`의 `param_attrs`는 type verifier가
  거부한다. 필드의 속성은 그 variant(`adt.variant`)의 매개변수 속성이다.
- `adt.variant`는 literal `adt.enum`의 매개변수로만 쓴다. 값의 타입이나 연산의
  타입 속성이 되지 않으며, 연산은 이름 있는 enum 타입과 선언 순서의 variant
  위치(`tag`)로 variant를 가리킨다. 위치는 `adt.variant_tag`가 내는 값과 같다.
- 재귀 참조는 `adt.struct`처럼 몸체가 자기 키를 담아 표현한다.
- Row 매개변수를 가진 선언의 layout은 `adt.struct`와 같이 `row_params`를 가지며,
  binder를 이름 바로 뒤에 쓴다: `adt.enum<"Step"['a0], [...]>`.

### Nominal 값 타입

Nominal layout의 값의 타입은 layout을 가리키는 nominal 값 타입이다. 값이 있는
위치(operation 결과와 피연산자, block 인자, 함수 signature, 다른 layout의 필드)는 이
타입을 쓴다. Nominal 값 타입은 값이며 managed 참조가 아니다. 그 값이 object에
담기는지는 [표현 선택](abi-boundary.md#표현-선택)이 정하고, object에 담긴 값의
참조는 [`heap.box`](#heap) 타입이다. 표현 선택 뒤에는 nominal 값 타입을 가진 값이
없다.

```text
adt.struct<"Point">
adt.enum<"Shape">
adt.struct<"Callback"[row(@"<State(Int)>")]>
adt.enum<"Step"[row(| 'a0)]>
```

- Row 매개변수가 없는 layout의 값 타입은 layout의 키 자체다.
- Row 매개변수를 가진 layout의 값 타입은 그 layout의 row 인스턴스다. Row
  인스턴스는 이름과 매개변수 없이 type 속성 `def`(layout의 키)와 `row_args`(row
  매개변수마다 row 하나)를 가진, 같은 종류의 `adt.struct` 또는 `adt.enum`이다. 식별
  타입이 아니며 내용으로 정체가 정해진다. `row_args`의 수는 `def`의 `row_params`와
  같다. Row 매개변수를 가진 키를 그대로 값 타입으로 쓰지 않는다.
- `row_args`의 원소는 선언 순서의 row 매개변수마다 callable의 `ability_row`와 같은
  형식의 row 하나다. Tail은 그 타입을 쓴 정의의 binder를 가리키고, layout 안에서는
  그 layout의 binder를 가리킨다. Row 인자는 타입의 정체에 들어가므로, row 인자만 다른
  두 row 인스턴스는 한 layout의 다른 타입이다.
- Textual form은 키의 이름 뒤에 row 인자를 쓴다: `adt.struct<"Callback"[row(..)]>`.
  Binder 목록(`['a0]`)과 달리 원소가 `row(`로 시작하며, 빈 row도 `row()`로 생략하지
  않고 쓴다. 이 문법으로 쓸 수 없는 속성 값은 일반 형식으로 출력한다. Parser는
  이름과 row 인자의 수로 키를 만들어 `def`로 삼는다. Row 인스턴스는 이름을 alias로
  제안하지 않으며, alias 안의 tail은
  [다른 type alias](#사용자-정의-어셈블리-형식)에서처럼 그 alias를 쓴 정의의
  binder를 가리킨다.
- 값 타입이 가리키는 layout은 키 자체이거나 row 인스턴스의 `def`다. 필드와 variant의
  타입은 그 layout에서 읽고, row 인스턴스이면 layout binder를 `row_args`로
  치환한다([Nominal row 인자](#nominal-row-인자)).
- Nominal 값 타입은 이름으로 다른 layout을 찾지 않는다. Compiler가 아는 layout의
  값 타입(evidence marker 등)은 [`tribute.types`](#tributetypes)에서 키를 읽는다.

### Nominal 선언

Source가 선언한 struct와 enum은 각각 module 최상위의 선언 operation 하나를 가진다.
선언은 source 이름을 소유하며, 이름이 필요한 단계는 layout 타입이 아니라 선언에서
이름을 읽는다. LLVM의 debug metadata가 source 이름을 타입과 따로 두는 것과 같다.

```text
adt.struct_decl "Point" !Point ["x", "y"]
adt.enum_decl "Shape" !Shape ["Dot", "Rect"]
```

- `adt.struct_decl`은 struct의 source 이름과 필드 이름을, `adt.enum_decl`은
  enum의 source 이름과 variant 이름을 선언 순서대로 담는다. 이름 목록의 수는
  layout의 필드 수, variant 수와 같다. Variant의 필드 이름은 담지 않는다.
- Source 이름은 출력을 위한 표기다. 특수화되지 않은 선언은 식별 경로 그대로
  쓰고(`"std::io::ReadLineResult"`), 특수화된 선언은 layout 이름의 mangling 대신
  식별 경로에 타입 인자를 source 문법으로 붙여 쓴다(`"pkg::Option(Nat)"`).
- 선언은 이름 붙이는 layout을 type 속성 `type`으로 가리킨다. 선언과 layout의
  대응은 이 타입의 정체이며, layout의 `name` 문자열을 비교해 찾지 않는다. 필드와
  variant의 타입은 layout이 소유한다. 텍스트 형식은 source 이름, layout 타입,
  이름 목록 순서로 쓰며, 이름 목록은 문자열 list 속성의 일반 문법이다. 이름은
  layout 이름처럼 식별자 여부와 상관없이 항상 따옴표 문자열이다.
- Layout을 변환하는 pass는 operation 속성 안의 타입을 변환하듯 선언의 `type`도
  변환한다. 그래서 선언은 변환된 layout을 가리키며, 같은 layout을 `type`으로 쓰는
  operation과 일치한다. 변환이 layout 이름에 tag를 붙여도
  ([nominal layout 변환](#nominal-layout-변환)) 선언의 source 이름은 그대로다.
- 선언이 생긴 source 위치가 필요한 단계는 선언의 `tribute.definition` 속성을
  읽는다.
- Frontend는 nominal layout을 게시할 때 그 선언을 함께 내보낸다. 특수화된 enum은
  인스턴스마다 layout과 선언을 가진다. Compiler가 만든 layout(closure 저장,
  continuation frame, tuple layout 등)은 선언이 없다.
- Variant 연산은 이름이 아니라 위치로 variant를 가리킨다. Compiler가 이름으로 아는
  compiler 소유 variant(문자열 literal이 만드는 `String`의 `Leaf`, evidence의
  `Option`의 `Some`, `std::io::ReadLineResult`의 각 variant)를 만드는 pass는 그
  위치를 선언에서 이름으로 찾는다.
- Native RTTI 생성은 struct·enum의 이름, struct의 필드 이름, variant 이름을 선언에서
  읽는다([RTTI Table](rc.md#rtti-table)). 선언이 없는 layout의 레코드는 위치
  번호(`"0"`, `"1"`, ...)를 필드와 variant의 이름으로 쓰고, 레코드 이름은 layout의
  이름이다. 이름이 없는 literal layout이면 `layout` 속성 값, 그것도 없으면
  `"struct"`다. 선언은
  값의 의미를 바꾸지 않으며, `heap` operation을 내리는 target
  pass가 지운다.

### `adt` 배열

`adt.array<E>`는 원소 타입 `E`의 값을 순서대로 나열한 배열 값이며, 원소가 될 수
있는 타입은 [`adt` dialect 선언](../crates/tribute-ir/src/dialect/adt.rs)이 정한다.
배열의 길이는 값마다 다르므로 배열 값은 SSA 값으로 쪼개거나 다른 object 안에 놓을 수
없고, [표현 선택](abi-boundary.md#표현-선택)은 배열 값을 언제나
[`heap.box`](#heap)에 담는다. 원소의 표현도 표현 선택이 정한다. 원소를 inline으로
저장할 수 없는 동안 aggregate 원소는 box에 담긴 값이며, 배열 object는 그 box를
원소로 가진다.

`core.array<E>`는 이와 다른 unmanaged linear array다. 연속된 memory에 놓인 `E`
값의 나열일 뿐 object가 아니며, 소유권이나 회수 방식을 갖지 않는다. `adt` 배열
operation은 `core.array`를 받지 않으며, 어느 target도 `core.array`를 managed
참조로 다루지 않는다.

배열을 만드는 operation은 `adt.array_new`, `adt.array_of`, `adt.array_splice`
셋이며, 각각 결과 배열의 원소를 정하는 방법이 하나로 고정되어 있다. Operand
개수나 모양으로 operation의 뜻을 고르지 않는다. 각 operation의 결과와 전제는
그 선언이 정한다. 배열 값은 바뀌지 않으므로 `adt`에는 원소를 쓰는 operation이
없다. 원소를 바꾼 배열은 `adt.array_splice`로 새로 만든다.

`adt`의 배열 operation은 target이 객체를 회수하는 방식을 가정하지 않는다.
Native RC가 배열 object에 두는 ownership 규칙과 순환 제약은
[rc.md](rc.md#비순환-object-그래프)가 정한다.

`adt.array_get`의 index, `adt.array_splice`의 `start`와 `end`,
`adt.array_new`의 `length`, `adt.array_len`의 결과는
[`core.index`](ir.md#coreindex-주소-폭-정수)다. 배열의 원소 개수와 위치는
target의 주소 폭으로 센다. 고정 폭 정수에서 온 index는 `arith.index_cast`나
`arith.index_castui`로 명시적으로 바꾼다.

Index는 부호 없는 값으로 해석한다. Target은 각 operation이 선언한 범위 전제를
어긴 실행을 trap으로 끝내며, 범위 밖 원소를 읽거나 쓰거나 다른 값으로
대신하지 않는다. Target이 표현할 수 있는 최대 길이나 할당 크기를 넘는 길이도
trap이다. 같은 범위 계약은 배열 object를 다루는 `heap` 배열 operation에도 적용된다.

## `heap`

`heap.*`는 값을 담은 object와 그 참조를 나타낸다. `adt`가 값을 다룬다면 `heap`은
object를 다룬다. 두 target이 같은 dialect를 쓰며, target이 object를 회수하는
방식(native RC, WasmGC)을 가정하지 않는다. 각 operation의 계약은 `heap` dialect
선언이 정한다.

### `heap.box`

- `heap.box<L>`은 layout `L`인 object에 대한 managed 참조다. `L`은 `adt.struct`,
  `adt.enum`, `adt.array` layout이다. `heap.box`는 managed 참조라는 표지
  (`ManagedRefType`)를 가지며, layout 타입은 그 표지를 갖지 않고 descriptor가 기술하는
  layout이라는 표지(`DescriptorLayoutType`)만 가진다.
- Object의 [descriptor](runtime-types.md)는 `L`로 정한다. Enum layout의 box는 그
  enum의 어느 variant object든 가리키며, 어느 variant인지는 object의 descriptor가
  정한다.
- 모든 `heap.box<L>`은 null 값을 가진다. `heap.null`이 만들며, retain과 release는
  null에 대해 아무 일도 하지 않는다. 소스 값은 null이 아니다. Null은 compiler가 만든
  표현(native `List`의 빈 node, 아직 채우지 않은 completion cell 등)만 쓴다.
- `L`의 필드 타입은 표현 선택 전에는 `adt` 값 타입이고, 표현 선택 뒤에는 그 필드의
  저장 표현이다. Box에 담는 값의 필드는 `heap.box`, inline으로 놓는 값의 필드는 그
  layout, scalar는 scalar다. 필드 타입을 바꾸는 이 변환은
  [nominal layout 변환](#nominal-layout-변환)을 따른다.

### Operation

`heap` operation은 `adt` operation을 object 위에서 하는 것이다.

- 할당: `heap.struct_new`, `heap.variant_new`, `heap.array_new`,
  `heap.array_of`, `heap.array_splice`. 같은 이름의 `adt` operation이 만드는 값을 담은
  object를 할당하고 그 box를 낸다.
- 읽기: `heap.struct_get`, `heap.variant_is`, `heap.variant_tag`, `heap.variant_get`,
  `heap.array_get`, `heap.array_len`. Box를 빌려 object를 읽는다. Variant 판별과 선언
  위치는 object의 [descriptor](runtime-types.md#variant-판별)로 구한다. Inline으로
  놓인 필드를 읽으면 그 값의 leaf들이 나온다.
- 쓰기: `heap.struct_set`, `heap.array_set`. Object를 제자리에서 바꾼다. 소스 값은
  바뀌지 않으므로 compiler가 소유하는 cell만 쓰며, 순환 제약은
  [rc.md](rc.md#비순환-object-그래프)가 정한다.
- Literal: `heap.bytes_const`, `heap.string_const`. 같은 이름의 `adt` literal이 만드는
  값을 담은 object의 box를 낸다.
- 참조: `heap.null`, `heap.is_null`, `heap.cast`. `heap.cast`는 nominal identity가
  같다고 확립된 box 사이와, closure capture layout의 box와 closure base의 box 사이에
  쓴다. `heap.box`와 `tribute_rt.anyref` 사이의 이동은 `heap.cast`가 아니라 명시적인
  `core.unrealized_conversion_cast`이며, [`adt`](#adt)의 erasure cast 규칙을 따른다.

### 위치

- `heap`은 대부분 [표현 선택](abi-boundary.md#표현-선택)이 `adt`에서 만든다. 표현
  선택보다 앞에서 `heap`을 직접 만드는 곳은 참조 의미론이 필요한 compiler 소유 저장뿐이다.
  [값 delimiter](cps-effects.md#값-delimiter)의 completion cell, one-shot resumption의
  소비 표시, [closure 저장](#closure-저장-layout)이 그렇다. 이 구간에서 `adt`와
  `heap`은 함께 쓰인다.
- Representation/ABI 경계 출구 뒤에는 `adt` 타입을 가진 값과 `adt`의 값 operation이
  없다. `adt` layout은 `heap.box`의 매개변수, `heap` operation의 layout 속성,
  [nominal 선언](#nominal-선언) 안에만 남는다. Object 접근은 모두 `heap` operation이며
  target lowering이 이를 내린다.
- Literal operation(`adt.bytes_const`, `adt.string_const`)의 결과는 배열이나 enum이라
  언제나 boxed다. 표현 선택은 이를 같은 literal을 담은 object의 box를 내는
  `heap.bytes_const`, `heap.string_const`로 바꾸고, target의 상수 lowering이 이를 내린다.
  그 결과는 다른 할당처럼 unit 하나를 가진 object다.

## `list`

`list.*`는 opaque canonical `List(a)` sequence 계약을 나타내며
`list.empty`, `list.prepend`, `list.is_empty`, `list.head`, `list.tail`을
쓴다. 이 operation은 element/result type을 갖지만 variant tag, node field
index, allocation size, target layout metadata는 갖지 않는다.

각 operation의 계약은 [`list` dialect 선언](../crates/tribute-ir/src/dialect/list.rs)이
정한다. Shared lowering은 literal을 만들 때 모든 원소를 왼쪽에서 오른쪽으로
먼저 평가한 뒤 값의 역순으로 `list.prepend`를 적용할 수 있다.

Package `std`의 public `List::prepend(value, tail)` wrapper는 registry로
검증된 private compiler intrinsic에 위임하고, 그 호출은 같은 `list.prepend`로
lowering된다. `List::prepend`라는 철자만 가진 source 정의 함수는 일반 호출로
남는다. Private intrinsic 선언은 compiler와 standard library 사이의 경계이며,
추가 public symbol이나 layout 계약이 아니다.

List 패턴은 sequence 관찰로 lowering한다. Exact-length 패턴은 빈 remainder를
요구하고, prefix-rest 패턴은 remainder를 같은 canonical List 타입으로
반환한다. Backend는 backend-ready 경계 전에 `list.*`를 제거해야 한다.

Source-logical List 패턴의 `list.head` 결과 타입은 `element_type`과 같아야
한다. 둘 다 전체 패턴의 검증된 `List(a)`에서 논리 타입으로 변환한 `a`를
사용한다. 이 계약은 매칭 검사와 성공한 arm의 바인딩에 동일하게 적용한다.
생성자 패턴 노드의 callable 메타데이터는 생성자 해석용으로 보존하며, 원소
값의 타입으로 사용하지 않는다.

List의 logical representation은 compiler-owned nominal identity로 선택한다.
동명 source ADT의 등록은 이 선택을 바꾸지 않는다. 현재 lowering이 사용하는
`tribute_rt.anyref`는 표현상의 선택이며 List의 semantic identity를 대신하지
않는다. Source nominal의 값 타입과 layout은 동일한 해석된 선언 및
specialization에 대응해야 한다.

## Nominal layout의 게시

Source-logical frontend는 nominal layout마다 몸체 없는 키를 먼저 만들고, 필드
타입을 다 만든 뒤 그 키에 몸체를 묶는다. 그래서 필드는 재귀나 전방 참조인
layout도 그 키나 [row 인스턴스](#nominal-값-타입)로 가리킨다. 키는
normalized nominal symbol의 IR type alias로 게시하며, layout의 `name`(문자열)과
게시된 alias의 이름은 텍스트가 일치해야 한다. Layout이
[변환](#nominal-layout-변환)으로 tag 붙은 새 키를 가지면 alias가 새 키를 가리킨다.
생성 연산 없이 signature에만 등장하는 특수화도 같은 계약을 따르며, 필드에서
참조하는 compiler-generated nominal tuple layout도 게시한다. 몸체가 묶이지 않은
키는 정의가 아니다.

<!-- markdownlint-disable-next-line MD033 -->
<a id="compiler-generated-tuple-layout"></a>

Compiler-generated tuple layout의 이름은 원소의 logical type에서 정해지는
함수이며, 원소가 callable type이면 그 [row](#ability-row)를 포함한다. Row만
다른 두 tuple은 다른 layout이다. [row 치환](#row-인자와-치환)은 원소를 치환한
layout을 같은 함수로 다시 만든다. 이 함수는 IR 계층이 소유하며 frontend와
pass가 함께 쓴다.

Row를 버리는 CPS legalization은 tuple layout을 변환한 원소로 같은 함수로 다시
만든다([nominal layout 변환](#nominal-layout-변환)의 tuple 예외). Tuple의 값
타입과 게시된 alias도 새 이름으로 바꾼다. Row만 다른 tuple layout은
legalization 뒤에 하나가 된다. Nominal layout의 필드 type과, row 인자로 치환한
그 필드 type을 쓰는 `adt` 생성·접근 operation도 같은 tuple layout을 가리키게
된다.

특수화된 enum의 variant 필드 타입은 해당 인스턴스에 함께 치환된 constructor
스킴에서 가져온다. 원본 generic 스킴이나 생성·패턴 표현식으로 필드 표현을
추정하지 않는다. `adt.variant_new`의 operand materialization과
`adt.variant_get`의 결과 타입은 같은 게시된 layout을 따른다. 특수화하지 않은
generic layout의 erased 필드는 기존 boxing과 payload recovery를 유지한다.
Signature나 필드에서만 참조하는 nominal 인스턴스도 의존 layout을 준비하고
게시해야 한다. 재귀 참조는 같은 선언 및 타입 인자의 인스턴스를 공유한다.

CPS, closure, target ABI 변환은 layout의 필드 타입을
[nominal layout 변환](#nominal-layout-변환) 규칙으로 변환하고, 게시된
alias도 변환한 키로 바꾼다. Native ownership 검사는 IR에서 도달하는 layout만
사용하며, interner 전체 검색이나 이름의 일부를 맞추는 방식으로 layout을 추정하지
않는다.

## `tribute.types`

Root `core.module`은 compiler가 의존하는 package `std`의 type identity를 하나의
dictionary attribute `tribute.types`에 담는다. 각 key는 package `std`에 있는
type의 canonical 선언 경로이고, 각 value는 그 선언에서 만든 정확한
`TypeRef`다. `std::String` entry는 `std::String`의 `adt.enum` 타입이다.
`std::runtime::evidence::Marker` 같은 runtime type도 같은 dictionary의
entry다([Compiler 소유 runtime 모듈](modules.md#compiler-소유-runtime-모듈)).

이 metadata는 nominal-layout 관례가 아니라 semantic compiler 상태다.
Frontend는 type checking 내내 각 선언의 안정적인 identity를 보존하고, entry를
만들 때 이름이 아니라 그 identity를 비교한다. Lowering은 entry를 직접 소비하며
type 이름, variant 이름, field layout으로 type을 다시 찾지 않는다. Type 변환은
entry를 다른 type-bearing attribute와 같은 방식으로 변환한다.

출력한 IR을 파싱하면 `tribute.types`가 보수적으로 빠진다. Entry가 필요한
lowering은 모두 fail closed다. Entry가 없으면 compiler 오류를 보고하며, type
이름이나 layout에서 type을 재구성하지 않는다. Backend는 byte constant를 여전히
처리할 수 있지만, `std::String` entry가 없으면 `adt.string_const`를 거부해야
한다. Evidence type을 만들거나 `effect.*`를 lowering하는 pass는
`std::runtime::evidence::Marker` entry가 없으면 module을 거부해야 한다.

## Root entry 참조

실행 파일 빌드의 root 패키지가 진입점을 선언하면, frontend는 root
`core.module`의 `tribute.entry` attribute에 그 함수의 `SymbolRef`를 담는다.
진입점은 언어 규칙이 정한다(root 모듈의 `main`). 이후 단계는 symbol 철자가
아니라 이 참조로 진입점을 찾는다.

- Class 특수화, `fn` arm 호출 검사, root bridge 합성처럼 bridge 전에 root
  worker를 다루는 단계는 이 참조가 가리키는 정의를 root worker로 읽는다.
  참조되지 않는 정의 제거는 이 정의를 root로 남긴다.
- 참조가 root module의 정의 하나로 해석되지 않으면 bridge 합성은 오류로
  거부한다. 참조가 없으면 진입점이 없는 module이며 bridge를 합성하지 않는다.
- Root bridge 합성은 참조를 소비한다. Root worker로 이름을 바꾸고 모듈 안의
  참조를 옮긴 뒤 attribute를 제거한다. 그 뒤의 진입점은 bridge가 만든 root
  `main` wrapper이고, target은 이 target ABI symbol로 진입점을 찾는다
  ([출구 계약](abi-boundary.md#representationabi-경계)).
