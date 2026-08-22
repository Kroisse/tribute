# TrunkIR Design

TrunkIR is Tribute's central intermediate representation between typed source
programs and target-specific Wasm or native backends.

## Principles

- SSA-based, using block arguments instead of phi nodes.
- Dialect namespaced: operations are written as `<dialect>.<operation>`.
- Multi-level: high-level, mid-level, and low-level dialects may coexist.
- Structured control flow is preferred until a backend requires CFG lowering.
- Lowering boundaries are declared with `ConversionTarget`, not Rust phase
  types such as `Module<Phase>`.

## Dialect Layers

```text
Infrastructure
  core      module structure and conversion glue

High-level
  tribute   unresolved or source-level frontend constructs
  tribute_control
            Tribute-specific source-logical callables and direct-style control
  ability   evidence and handler dispatch semantics
  effect    target-independent effect ABI
  closure   closure construction and decomposition
  adt       structs, variants, arrays, references, literals
  list      opaque persistent-sequence construction and observation

Mid-level
  func      functions, calls, returns, function references
  scf       structured control flow and region yields
  arith     constants, arithmetic, comparisons, casts
  mem       low-level memory/data operations

Low-level
  wasm.*    Wasm and WasmGC-oriented operations
  clif.*    Cranelift-oriented operations
```

## Conversion Legality

`ConversionTarget` classifies operations as `Legal`, `Illegal`, or `Unknown`.
An unspecified operation is `Unknown`, not legal.

Partial conversion may leave unknown operations for later passes. Full
conversion boundaries, such as backend-ready native IR, must reject unknown
operations.

직접형 제어 pipeline은 다음과 같은 이름의 경계를 사용한다:

| 경계 | `ConversionTarget` mode | 필수 적법성 |
| ---- | ---- | ---- |
| `tribute-control-pre-cps` | frontend 적합성 검사에는 full, 변환 중에는 partial | `tribute_control.*`은 `core.module`, `core.never`와 일반 `core` 값 type, `scf`, `arith`, `adt`, `list`, `tribute_rt`, `tribute_io`와 공존할 수 있다. `func.*`, `closure.*`, `core.func`, 기존 `ability.*`, `effect.*`, legacy CPS 구성 operation은 illegal이다. |
| `tribute-control-post-cps` | shared CPS 변환 뒤 partial | `tribute_control` dialect의 모든 operation과 type이 illegal이다. Physical `func.*`, `closure.*`, `core.func`, 기존 `ability.*`, `effect.*`, 일반 dialect는 이후 pass를 위해 공존할 수 있다. |
| `tribute-backend-ready-native` | Tribute full 경계 뒤 generic Cranelift 경계 | `tribute_control`, `ability`, `effect`, `closure`, `list`, `tribute_io`, conversion cast가 없다. 명시적으로 열거한 native infrastructure와 `clif.*` operation만 남는다. |
| `tribute-backend-ready-wasm` | emission-ready full 경계 | `tribute_control`, `ability`, `effect`, `closure`, `list`, `tribute_io`, `wasm_gc`, `core.unrealized_conversion_cast`가 없다. 명시적으로 열거한 Wasm infrastructure와 `wasm.*` operation만 legal이며 unknown operation은 illegal이다. |

Post-CPS helper는
`ConversionTarget::new().illegal_dialect("tribute_control")`와 partial
verification을 조합한다. Full mode에서는
unknown operation이 legal하지 않으므로 frontend 적합성 target과 backend full
target이 legal dialect와 operation을 열거해야 한다. `ConversionTarget`은
operation 적법성만 검사하므로 Tribute whole-IR type walk가 pre-CPS의
`core.func`/`closure.closure`와 post-CPS의
`tribute_control.callable`/`resume_token`을 별도로 거부한다. 같은 walk는
`tribute-backend-ready-native`와 `tribute-backend-ready-wasm`에서도 필수이며
남은 두 `tribute_control` type을 거부한다. 함수·호출 signature, operand/result,
block argument, type attribute와 nested type parameter를 재귀적으로 검사하며,
operation 적법성 검사와 모두 통과해야 named boundary가 성립한다. Region 소유
operation을 recursively legal로 표시해서 nested illegal operation을 가려서는
안 된다. Generic `trunk-ir`의 Cranelift 경계는 Tribute에 독립적으로 유지하며,
그 전에 Tribute pipeline이 high-level dialect와 type을 거부한다.

최종 native/Wasm 경계는 `tribute.calling_convention = 2` (`Cps`)인 worker와 생성된
continuation/`done_k`/handler-dispatch의 result가 비어 있는지도 검사한다.
CPS control result 역할의 `anyref`, nominal `__tribute_cps_control` enum과
result-producing CPS dispatch는 illegal이다. Boxed source value, effect payload,
closure environment와 dispatch field의 일반 `anyref` 사용은 허용한다.

하나의 rewrite 도중에는 source `tribute_control.*`과 새
`ability.*`/`effect.*` 결과가 일시적으로 공존할 수 있다. 이는 partial
conversion 내부의 구현 상태이지 성공한 named boundary가 아니다. 성공한
`tribute-control-post-cps` verification은 남은 모든 `tribute_control.*`
operation을 source location과 함께 conversion failure로 보고한다.

## Validation Layers

TrunkIR validation is layered by responsibility. The compiler should use the
smallest layer that can state an invariant precisely, and should not introduce a
separate semantic-contract framework unless these layers leave a concrete gap.

| Layer | Responsibility | Failure timing | Examples |
| ---- | ---- | ---- | ---- |
| Operation verifier | Local invariants of one operation: operand/result counts, required attributes, attribute domains, region shape, and terminator requirements that can be checked without global analysis. | During parsing/building when possible, or during explicit operation validation checkpoints. | `arith.cmpf` accepts only supported predicates; an op with regions requires the expected region count and terminator form. |
| Conversion target | Dialect and type legality at a named lowering boundary. | Immediately after a pass or pass group claims a conversion boundary. Partial conversion rejects explicitly illegal operations; full conversion also rejects unknown operations. | Ability lowering leaves no `ability.perform`; backend-ready native IR contains only `clif.*` plus allowed infrastructure ops. |
| Pass-manager verifier | Whole-IR consistency after transformations, with the offending pass identified. | After each pass registered in a `PassManager` when a verifier hook is installed. | SSA use-chain consistency, value visibility across isolated regions, or other graph-wide invariants. |
| Operation interface | Shared behavior queried generically across dialects. | At the consumer that needs dialect-independent behavior. Interfaces should be introduced only for multiple concrete consumers or one generic transform. | `PureOps` for DCE removability; `IsolatedFromAboveOps` for nested pass-manager anchoring. |

Local operation verifiers must not depend on conversion state or pass ordering.
Conversion targets must not duplicate local semantic checks. Pass-manager
verifiers should remain responsible for graph-wide invariants that require
walking use-def chains, symbol tables, or nested region relationships. Operation
interfaces describe behavior, not validation phases.

## Core Invariants

- A `core.module` owns the top-level region for a compilation unit.
- `core.unrealized_conversion_cast` is temporary conversion glue and must not
  remain at backend-ready boundaries.
- Operation and type names are interned `Symbol`s. Qualified paths are stored as
  `::`-separated symbols.
- Nested regions use normal SSA visibility rules: values defined inside a
  nested region are not visible outside it unless yielded or otherwise modeled
  by the operation.

## High-Level Dialects

`tribute.*` represents source-level constructs that should disappear after
resolution, type checking, TDNR, and AST-to-IR lowering.

<!-- markdownlint-disable-next-line MD033 -->
<a id="direct-style-control"></a>

### 직접형 호출 대상과 제어

`tribute_control.*`은 typed frontend lowering과 shared CPS conversion 사이의
target-independent 직접형 경계다. Dialect identifier는 정확히
`tribute_control`이다. TrunkIR은 qualified operation을 하나의
`<dialect>.<operation>` 쌍으로 parse하므로 dialect identifier 안에 점을 넣는
등 separator를 하나 더 사용하는 표기는 invalid이다.

이 dialect는 CPS legalization 전의 Tribute 고유 callable과 effect-control
의미를 함께 소유한다. ANF는 input invariant이지 dialect의 정체성이 아니다.
산술, ADT/list/tuple/record 구성과 structured selection은 기존 dialect에 남지만,
`func.*`, `closure.*`, `core.func`는 physical 표현이므로 이 경계에 나타나지
않는다.

논리 callable type은
`tribute_control.callable(Result, Params...) {tribute.calling_convention = N}`이다.
`Result`와 `Params`는 source-logical type이며 기존 code `Direct = 0`,
`EvidenceDirect = 1`, `Cps = 2`를 그대로 사용한다. 이 metadata는 typechecking
결과를 복사한 것으로 body에서 추론하지 않는다. Legalization은 이를 기존
`CallableAbi`와 physical `core.func`, `closure.closure` 및
`tribute.calling_convention` operation attribute로 바꾼다. Type verifier는
result 하나와 parameter 0개 이상, convention domain, 모든 component type의
resolution을 검사하며 physical `core.func`나 `closure.closure`를 component로
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
| `tribute_control.return` | `func` 또는 `lambda` body의 logical result 반환 |
| `tribute_control.perform` | source `fn` 또는 general `op` 하나를 semantic kind를 보존한 직접형으로 호출 |
| `tribute_control.handle` | 직접형 computation, completion arm, handler table의 경계를 설정 |
| `tribute_control.handler` | handle 안의 `fn` 또는 general `op` handler arm을 기술 |
| `tribute_control.resume` | resumptive general handler arm에 바인딩된 affine resumption을 소비 |
| `tribute_control.yield` | 실행 가능한 `tribute_control` region을 logical value로 종료 |

`func.tail_call`, `func.tail_call_indirect`, `func.constant`, `func.unreachable`의
logical 복제는 없다. Tail 형상은 legalization 결과이고 named function value는
`func_ref`가 표현한다. Legalization은 알려진 target에 `func.tail_call`, closure,
continuation과 `done_k` target에 새 `func.tail_call_indirect`를 만들 수 있다.
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

### 사용자 정의 어셈블리 형식

`tribute_control.func`와 `tribute_control.lambda`만 custom parse/print를 요구한다.
간결한 convention 문법은 `convention(direct)`,
`convention(evidence_direct)`, `convention(cps)`이며 각각 callable type의
`tribute.calling_convention = 0`, `1`, `2`로 round trip한다. Parser는 signature와
keyword로 `tribute_control.callable`을 만들고 convention을 type에만 저장한다.
출력기도 type attribute만 읽으며 body에서 추론하거나 operation에 중복
attribute를 쓰지 않는다. 추가 operation attribute에
`tribute.calling_convention`을 다시 적으면 parser 또는 verifier가 거부한다.

```text
tribute_control.func @f(%x: T) -> R convention(cps)
    attributes {visibility = @private} {
  ...
}

tribute_control.func @decl(%x: T) -> R convention(direct)

%f = tribute_control.lambda(%x: T) -> R convention(cps)
    captures [%captured] attributes {debug_name = "apply"} {
  ...
}

%g = tribute_control.lambda() -> R convention(direct)
    captures [] {
  ...
}
```

`func` 형식은 `func.func`처럼 symbol, 분해된 logical parameter/result, 선택적
추가 attribute와 선택적 body를 출력한다. Body가 있으면 entry block label을
생략하고 signature parameter를 entry argument로 복원하며 declaration은 body가
없다. `lambda` 형식은 `closure.lambda`처럼 SSA result, 분해된 source
parameter/result, 필수 convention, 명시적 `captures [...]`, 선택적 추가
attribute와 body를 출력하고 entry label을 생략한다. `captures [...]`는 capture가
없어도 `captures []`로 항상 출력하며 canonical 순서는 convention, captures,
선택적 attributes, body다. `func_ref`, `call`, `call_indirect`, `return`은
generic assembly가 모든 정보를 손실 없이 표현하므로 custom format을 만들지
않는다.

#### `tribute_control.func`

```text
tribute_control.func {sym_name = @id, type = !Callable} (%x: T) { ... }
```

- **형상:** 피연산자와 결과는 없다. `sym_name: Symbol`과
  `type: tribute_control.callable(Result, Params...)`가 필수다. 선언은 region이
  없고 정의는 source parameter만 block argument로 받는 single-block `body`
  하나이며 `tribute_control.return`으로 끝난다. Foreign ABI 같은 비제어
  attribute는 보존한다.
- **의미:** source named function의 logical signature와 typechecking이 선택한
  convention을 정의하며 hidden evidence, environment, `done_k`를 포함하지 않는다.
- **검증:** local verifier는 attribute, region, block argument, return type을
  callable type과 맞춘다. Whole-IR verifier는 symbol uniqueness를 검사한다.
- **소유권과 값 흐름:** body는 isolated-from-above다. Source local과 parameter만
  block argument로 들어온다.
- **위치:** source function 또는 extern declaration 전체 span이다.

Named callable의 origin은 symbol이나 `abi` 문자열과 별개인 typed frontend
metadata다. Source 정의는 body를 가진 Tribute callable이고, compiler intrinsic은
canonical registry가 부여한 semantic identity와 완전한 logical signature를 함께
가진다. Private runtime helper는 target stage에서만 physical signature로 만들며
source-logical `adt.typeref`를 받을 수 없다. 아직 별도 user FFI 계약이 없는 bodyless
declaration은 managed semantic parameter나 result를 사용할 수 없다. Textual attribute,
symbol spelling, 위치 또는 printed IR만으로 origin을 복구하거나 승격하지 않는다.

Frontend 경계 verifier는 metadata 전체와 module의 callable graph를 mutation 전에
대조한다. Direct call은 module-local symbol을 유일하게 resolve하고 완전한 signature를
맞춰야 한다. Indirect call은 exact `tribute_control.callable` signature와 source-logical
callable producer를 요구한다. Return은 enclosing callable의 logical result와 일치해야
한다. Body가 없는 declaration도 body traversal 없이 같은 검사를 받는다.

#### `tribute_control.lambda`

```text
%f = tribute_control.lambda [%capture0, ...] : !Callable { ... }
```

- **형상:** 피연산자는 typechecking된 capture를 source 순서로 나열한다. 결과는
  `tribute_control.callable` 하나이고 필수 attribute는 없다. Single-block body는
  source parameter만 block argument로 받으며 `tribute_control.return`으로 끝난다.
- **의미:** source lambda를 만들며 body는 lexical capture와 parameter를 사용한다.
- **검증:** local verifier는 result signature, block argument, return type을
  맞춘다. Whole-IR verifier는 body의 외부 SSA reference와 capture 집합이 정확히
  일치하는지 검사한다.
- **소유권과 값 흐름:** capture는 일반 SSA use이며 다른 hidden operand는 없다.
- **위치:** source lambda 전체 span이다.

#### `tribute_control.func_ref`

```text
%f = tribute_control.func_ref {func_ref = @id} : !Callable
```

- **형상:** 피연산자와 region은 없고 terminator가 아니다. `func_ref: Symbol`이
  필수이며 결과는 `tribute_control.callable` 하나다.
- **의미:** named function을 first-class source value로 만든다. Source가 named
  function을 higher-order value로 사용할 수 있으므로 필요하다. Result는 대상과
  같은 source signature이며 convention은 대상 worker와 같거나 더 강할 수 있다.
- **검증:** local verifier는 attribute와 result 형상을 검사한다. Whole-IR
  verifier는 symbol resolution, source signature 일치와 convention 순서를
  검사한다.
- **소유권과 값 흐름:** 결과는 일반 SSA value다.
- **위치:** named function을 값으로 사용한 source reference span이다.

#### `tribute_control.call`

```text
%result = tribute_control.call %arg0, ... {callee = @f} : Result
```

- **형상:** declaration 순서의 source argument, source-logical result 하나,
  `callee: Symbol`을 가지며 region이 없는 non-terminator다.
- **의미:** named source callable을 직접 호출한다.
- **검증:** local verifier는 attribute와 resolved operand/result를 검사한다.
  Whole-IR verifier는 callee `tribute_control.func`의 arity/type을 맞춘다.
- **소유권과 값 흐름:** argument/result는 일반 SSA value이고 hidden operand는
  없다.
- **위치:** callee와 argument를 포함한 source call span이다.

#### `tribute_control.call_indirect`

```text
%result = tribute_control.call_indirect %callee, %arg0, ... : Result
```

- **형상:** `tribute_control.callable` callee, source argument, callee type의
  source-logical result 하나를 가지며 attribute/region이 없는 non-terminator다.
- **의미:** lambda, `func_ref`, parameter 또는 capture로 얻은 callable을 호출한다.
- **검증:** local verifier는 callee signature와 argument/result type을 맞춘다.
- **소유권과 값 흐름:** callee와 argument는 일반 SSA use이고 environment,
  evidence, `done_k`는 없다.
- **위치:** callee와 argument를 포함한 source indirect-call span이다.

#### `tribute_control.return`

```text
tribute_control.return %value
```

- **형상:** source-logical result 하나를 받고 결과/attribute/region은 없다.
  `tribute_control.func` 또는 `lambda` body의 terminator이며 다른 위치에서는
  invalid이다.
- **의미:** enclosing callable의 logical result를 반환한다.
- **지역 검증:** enclosing callable result와 operand type을 맞춘다.
- **소유권과 값 흐름:** 일반 SSA value를 소비한다.
- **위치:** source return 또는 implicit body-result expression span이다.

Callable operation의 physical lowering은
[cps-effects.md](cps-effects.md#pre-cps-callable-shape)에만 정의한다.

#### `tribute_control.perform`

```text
%result = tribute_control.perform %arg0, ... {
  ability_ref = !State,
  op_name = @get,
  operation_kind = @op
} : ResultType
```

- **피연산자:** declaration 순서로 놓인, 이미 평가된 source argument 0개 이상이다.
  값은 logical type을 유지하며 tuple packing과 erasure는 conversion이 담당한다.
- **결과:** logical operation result 하나만 만든다. Source `Never` result는
  `core.never`이며 physical `Never` control carrier를 선택하지 않는다.
- **속성:** `ability_ref: Type`, `op_name: Symbol`,
  `operation_kind: Symbol`이 필수다. `operation_kind`는 정확히 `fn` 또는 `op`이며
  typecheck된 operation declaration에서 복사한다. 이는 body나 use site에서
  추론하는 lowering hint가 아니라 source-semantic metadata다. 모든 source
  ability invocation이 이 operation을 사용하며 shared conversion은 kind를
  재분류하지 않는다.
- **영역과 block argument:** 없다.
- **종결자:** 아니다. 직접형 IR에서 이 operation은 terminator가 아니다.
- **의미:** 선언된 kind로 source operation을 호출한다.
  `operation_kind = @fn`이면 선택된 handler result가 직접형 평가를 자동으로
  resume한다. Shared conversion은 continuation을 capture하지 않고 기존 tail
  dispatch 경로를 사용한다. `operation_kind = @op`이면 일치하는 handler가
  resume할 수 있으며, 이 경우 operation 뒤에서 실행을 계속하고 `%result`가
  결과가 된다. Resume하지 않으면 선택된 general handler가 일치하는 handle을
  완료하고 겉으로 보이는 suffix는 평가하지 않는다. Source `op -> Never`는
  resume할 수 없으므로 logical resumption을 만들거나 겉으로 보이는 suffix를
  capture하지 않는다.

- **지역 검증:** 세 attribute를 요구하고 `operation_kind` domain을
  검사하며, result가 정확히 하나이고 region은 없으며 operand/result type이
  inference variable이 아니라 resolve되었는지 확인한다. Symbol-aware frontend
  적합성 검사는 `ability_ref`와 `op_name`이 resolve한 operation declaration의
  `fn`/`op` kind, parameter type, result type이 attribute, operand, result와
  일치하는지도 확인한다. 어떤 verifier도 control flow, handler, result type,
  calling convention에서 kind를 추론해서는 안 된다.
  `tribute_control.handler`와 containing-handle verifier는 handler entry에 아래
  계약의 kind별 형상을 적용한다.
- **소유권과 값 흐름:** operand는 일반 SSA use다. 이 operation은
  source-visible continuation value를 만들지 않으며 continuation 구성은 shared
  CPS conversion이 소유한다. `@fn`에는 continuation 없는 `ability.call`/tail
  dispatch를, `@op`에는 suffix continuation을 받는 `ability.perform`/CPS
  dispatch를 만든다. `op -> Never`에는 대신 기존 ability/effect ABI가 요구하는
  실제 zero-capture reject continuation을 공급하며 그 body는
  `func.unreachable`이다. 이 continuation은 source suffix를 capture하지 않는다.
  Null, in-band sentinel, 임의의 `anyref`는 continuation이 아니다. ABI adapter의
  전체 규칙은
  [cps-effects.md](cps-effects.md#direct-style-control-boundary)를 따른다.
- **위치:** 가능하면 qualified callee와 argument를 모두 포함하는
  ability-operation call의 source location이다.

#### `tribute_control.handle`

```text
%answer = tribute_control.handle : AnswerType
  body {
    ...
    tribute_control.yield %body_value
  }
  completion(%completed: BodyType) {
    ...
    tribute_control.yield %answer_value
  }
  handlers {
    tribute_control.handler ... { ... }
    ...
  }
```

- **피연산자:** 없다. 실행 가능한 region이 capture하는 값은 일반 enclosing SSA
  visibility를 사용한다.
- **결과:** logical handle result 하나만 만든다.
- **속성:** 필수 attribute는 없다. Dynamic prompt/owner tag와 physical tail-call
  표현은 이 operation의 의미 attribute가 아니라 legalization 계약이다.
- **영역:** 고정된 `body`, `completion`, `handlers` 순서로 정확히 세 개다.
- **Block argument:** `body`는 argument가 없는 block 하나다. `completion`은
  argument가 정확히 하나인 block 하나이며, 그 type은 `body`가 yield한 value의
  type과 같다. `handlers`는 argument가 없는 block 하나이며
  `tribute_control.handler` entry만 포함한다.
- **종결자:** `body`와 `completion`은 `tribute_control.yield`로 끝난다.
  `handlers` block은 선언적 table이며 terminator가 없다.
- **의미:** `body`가 정상 완료되면 `completion`을 정확히 한 번 평가하고 그 값을
  반환한다. Resume하지 않고 완료된 general handler arm은 arm value를 handle
  result로 반환하고 `completion`을 건너뛴다. Tail-resumptive `fn` arm은 중단된
  computation에 자동으로 공급되는 operation result를 반환한다.
- **지역 검증:** 고정된 region 개수, single-block 형상, block-argument
  개수, terminator, yield type equality를 강제한다. 또한 `handlers`의 모든
  direct child가 유일한 `(ability_ref, op_name)`
  `tribute_control.handler`인지 확인한다. Result type은 completion yield type과
  모든 general handler의 answer type과 같아야 한다.
- **소유권과 값 흐름:** 이 operation은 body가 resumptive general
  operation을 수행할 때 생기는 delimited resumption capability를 소유한다.
  해당 capability는 resumptive general handler entry의 `resume_token` block
  argument로만 노출된다. 값은 `tribute_control.yield`를 통해서만 실행 가능한
  region 밖으로 나간다.
- **위치:** 전체 source `handle` expression이다. Region/block location은 각각
  대응하는 body, completion arm, handler-list span을 사용한다.

Frontend는 항상 completion region을 materialize한다. Source에 `do` arm이 없으면
이 region은 body result에 대한 identity operation이다. Source 의미를 바꾸지
않으면서 conversion의 optional structural case를 없앤다.

#### `tribute_control.handler`

```text
tribute_control.handler {
  ability_ref = !State,
  op_name = @get,
  kind = @op,
  operation_result_type = ResultType
} (%arg0: Arg0Type, ..., %resume:
    tribute_control.resume_token<ResultType, AnswerType>) {
  ...
  tribute_control.yield %answer
}
```

- **피연산자와 결과:** 없다. Surrounding `tribute_control.handle`이 소유하는
  declarative entry다.
- **속성:** `ability_ref: Type`, `op_name: Symbol`, `kind: Symbol`,
  `operation_result_type: Type`이 필수다. `kind`는 정확히 `fn` 또는 `op`이다.
- **영역:** block 하나를 가진 실행 가능한 `body` region 하나만 있다.
- **Block argument:** source operation argument가 declaration 순서와 logical
  type으로 먼저 나온다. Resumptive `op` entry는 마지막에
  `resume_token<operation_result_type, handle-result-type>` argument 하나를
  갖는다. `operation_result_type`이 source `Never`인 `op`에는 token이 없고
  `fn` entry에도 token이 없다.
- **종결자:** body는 `tribute_control.yield`로 끝난다. `fn`에서는 yield
  type이 `operation_result_type`과 같고 자동으로 resume한다. `op`에서는 yield
  type이 token의 `answer` type과 같으며, arm이 `resume`을 통해 control을
  넘기지 않고 완료되면 enclosing handle result가 된다.
  `operation_result_type`이 source `Never`인 `op`에는 token이 없고 yield type은
  enclosing `tribute_control.handle` result type과 같아야 한다.
- **지역 검증:** 필수 attribute와 domain, block 하나, 마지막
  terminator, token 위치/parameter, 위 yield 규칙을 확인한다.
  `operation_result_type`이 `Never`인 general handler는 token argument가 없어야
  하며 nested region을 포함한 body 어디에도 `tribute_control.resume`이 없어야
  한다. ContinuationFrame placement, uniqueness와 `op -> Never` yield를 포함한 enclosing
  handle result와의 equality는 containing handle의 local verifier가 확인하고,
  converter도 rewrite 전에 같은 equality와 resume 부재를 검사해 위반을
  conversion failure로 보고한다. Symbol-aware frontend 적합성 검사는 참조된
  declaration의 `kind`, argument type,
  `operation_result_type`도 동일한지 확인하며, 어떤 verifier도 body 형상으로
  general `op`을 재분류하지 않는다.
- **소유권과 값 흐름:** 마지막 token argument가 있으면 affine이다.
  사용하지 않아 continuation을 drop하거나, `tribute_control.resume`까지 하나의
  static ownership path를 가질 수 있다. Closure가 이를 capture하면 그 static
  path가 closure로 이전된다. Copy, store, return, yield 또는 다른 방식의
  escape는 invalid이다. Static SSA validation은 capture한 closure가 dynamic하게
  한 번만 호출되는지 증명할 수 없으므로 lowered resumption은 runtime에서도
  one-shot consumption을 강제하고 두 번째 호출을 거부하거나 trap해야 한다.
  `fn` arm과 `op -> Never` arm에는 continuation capability가 없다.
- **위치:** operation header를 포함한 source handler arm이다.

#### `tribute_control.resume`

```text
%answer = tribute_control.resume %resume, %value : AnswerType
```

- **피연산자:** 정확히 두 개다. `%resume`은
  `resume_token<InputType, AnswerType>`이고 `%value`는 `InputType`이다.
- **결과:** `AnswerType` 하나만 만든다.
- **속성과 영역:** 없다.
- **종결자:** 아니다. `resume` 뒤의 strict work는 enclosing region에
  명시적으로 남으며 resumed computation이 반환한 뒤에만 실행된다.
- **의미:** lexical하게 가장 가까운 enclosing general handler의 one-shot
  resumption을 소비하고, 중단된 `perform`에 `%value`를 공급하며, resumed
  computation이 handle boundary에 도달했을 때 얻는 logical result를 반환한다.
  `fn` 또는 `op -> Never` arm에서는 invalid이다.
- **지역 검증:** operand/result arity와
  `resume_token<InputType, AnswerType>`이 요구하는 세 type equality를 강제한다.
- **소유권과 값 흐름:** token을 소비한다. Explicit closure capture를 통해
  token이 이 operation에 도달할 수 있지만 handler block argument에서 시작하는
  single static use-def path를 유지해야 한다. Affine-use validation은 capture와
  nested region을 따라가므로 whole-IR check다. Capture 때문에 반복적인 dynamic
  invocation이 가능하면 converted continuation의 runtime one-shot state가 최종
  enforcement boundary다.
- **위치:** source `resume` expression이다.

#### `tribute_control.yield`

```text
tribute_control.yield %value
```

- **피연산자:** logical value 하나만 받는다.
- **결과, 속성, 영역:** 없다.
- **종결자:** `handle` body, completion region, handler body의 terminator다.
  다른 위치에서는 invalid이다.
- **지역 검증:** 자체 형상을 강제한다. Owning operation이 placement와
  yield type을 검사한다.
- **소유권과 값 흐름:** 일반 logical value를 owning structured
  operation으로 전달한다. `resume_token`은 절대 yield할 수 없다.
- **위치:** region result를 만드는 source expression이다. 합성한 identity
  completion에는 owning `handle` location을 사용한다.

#### 구조화된 continuation 불변 조건

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
4. `operation_kind = @fn`인 perform은 continuation을 capture하지 않는다.
   Shared conversion이 tail path로 dispatch하며 자동으로 resume된 operation
   result는 일반적인 남은 block suffix로 흐른다.
5. Resumptive general handler의 resume token은 중단된 body continuation을
   가리킨다. `tribute_control.resume`이 이를 호출한 뒤 arm-local strict
   suffix를 계속 실행한다. Arm이 resume하지 않으면 yield가 일치하는 handle을
   직접 완료하고 중단된 suffix와 completion region을 건너뛴다. Source
   `op -> Never` arm은 token을 받지 않으며 이 non-resuming path만 취할 수 있다.
6. Nested handle은 자체 delimiter를 설치한다. Perform은 dynamic하게 설치된
   handler 중 가장 가까운 일치 handler가 처리한다. Resume하면 perform과 해당
   handler 사이에서 선택된 모든 structured frame에 다시 진입한다. Resume하지
   않고 완료하면 그 frame을 포기한다.

CPS 변환 뒤 Cps callable은 `Evidence, ContinuationFrame<R>, source args`를 받고
`core.never`로 끝난다. 생성된 completion과 exact resume도 Evidence와
ContinuationFrame을 명시적으로 받는다. Resume은 동적 ContinuationFrame에서 불변 어휘적 dispatcher를
재구성한 뒤 suffix와 nested handle을 계속 실행한다. 세 dispatch 계층의 정확한
형상은 [cps-effects.md](cps-effects.md#dispatch-layers)를 따른다.

이 단일 region/suffix 규칙은 case arm과 guard, conditional, short-circuit
오른쪽 항, nested handle body와 arm, resume path, 그리고 이들을 감싸는 strict
work를 모두 다룬다. AST containment scan이나 construct-specific continuation
convention은 dialect contract에 포함되지 않는다.

`ability.*` represents effect evidence and handler dispatch. Ability operations
are lowered through the effect pipeline; ability-related types may remain until
their target-specific representation is selected.

`effect.*` represents the target-independent ABI between high-level ability
semantics and backend-specific evidence/callable layouts. It carries semantic
inputs such as evidence, ability identity, operation name, payload,
continuation, and handler closures. It must not expose Marker field indices,
handler-table storage layout, closure field positions, or backend function
pointer representation.

`closure.*` represents closure allocation and projection. Closures lower
differently per backend: Wasm uses function references plus GC structures, while
native uses function pointers plus heap environments.

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
The public `List::prepend(value, tail)` prelude wrapper delegates to a private
ABI-marked compiler intrinsic, whose calls lower to the same `list.prepend`
operation. A source-defined function merely spelled `List::prepend` remains an
ordinary call. The private intrinsic ABI is a compiler/prelude boundary, not an
additional public symbol or a layout contract.

List patterns lower to sequence observations. Exact-length patterns require an
empty remainder; prefix-rest patterns return the remainder as the same canonical
List type. A backend must eliminate `list.*` before its backend-ready boundary.

The root `core.module` carries Tribute-specific well-known type identities as
`TypeRef` attributes. In particular, `tribute.type.string` is the exact
`adt.enum` type produced from the prelude `String` declaration. String-literal
lowering must consume this identity directly; it must not rediscover `String`
by type name, variant names, or field layout. This metadata is semantic compiler
state rather than a nominal-layout convention. The frontend preserves the
prelude declaration's stable identity through type checking and compares that
identity, not its name, when materializing this `TypeRef`.

The current specialized textual printer for `core.module` does not serialize
arbitrary module attributes. Consequently, parsing printed IR conservatively
drops well-known type metadata. A backend may still process byte constants, but
must reject `adt.string_const` when `tribute.type.string` is absent rather
than scanning types for a plausible replacement. A future textual format may
make these attributes round-trip explicitly.

## Mid-Level Dialects

`func.*` represents function definitions, direct calls, indirect calls, function
references, returns, tail calls, and unreachable control flow.
`func.tail_call_indirect`는 callable operand와 argument를 받고 result가 없는
terminator다. Shared verifier는 callee `core.func`와 enclosing caller의 result가
모두 `core.never`인지 검사한다. Native/Wasm signature lowering은 이를 empty result
vector로 바꾸고 각각 indirect return-call로 낮춘다.

`scf.*` represents structured control flow, including pattern/case regions and
region yields. Loop-like forms may be introduced by optimization passes such as
tail-call lowering.

`arith.*` represents constants, integer and floating arithmetic, comparisons,
bit operations, and numeric conversions.

`mem.*` represents low-level data, load, and store operations for runtime or FFI
support.

## Low-Level Dialects

`wasm.*` is the Wasm backend dialect. It models Wasm control flow, calls,
numeric operations, memory operations, and WasmGC constructs.

`wasm_gc.*` is a typed intermediate dialect for WasmGC lowering. Its operations
carry semantic heap types as mandatory `TypeRef` attributes and must not infer
nominal identity from an erased operand such as `anyref`. A module-wide type
layout pass assigns binary type-section indices and fully converts these
operations to `wasm.*` operations, whose required integer attributes correspond
to WebAssembly instruction immediates. This pass runs once, after unrealized
conversion casts have been materialized, because materialization may introduce
additional typed GC operations.

```text
wasm_gc.struct_get { type = !String$Leaf, field_idx = 0 }
  -- module GC type layout -->
wasm.struct_get { type_idx = 9, field_idx = 0 }
```

Builtin layouts follow the same rule. Lowering refers to canonical semantic
types such as `core.bytes` or the marker/evidence ADT types; the layout pass
maps those identities to reserved indices. The Wasm emitter accepts no residual
`wasm_gc.*` operations and does not infer missing indices. Function-signature
indices used by `call_indirect` are a separate concern and are not GC heap-type
identities.

`clif.*` is the native backend dialect. It models Cranelift-style functions,
calls, arithmetic, CFG control flow, memory access, stack slots, symbol
addresses, and numeric conversions.

Backend-ready full conversion targets must explicitly list which infrastructure
operations are still allowed next to the backend dialect.

## Pipeline Shape

```text
source
  -> parse / AST
  -> name resolution
  -> type checking
  -> TDNR
  -> AST-to-IR (tribute_control callable/control + ordinary value IR)
  -> shared CPS legalization
  -> shared lowering and optimization
  -> Wasm or native lowering
  -> backend-ready full conversion target
  -> emit
```

Important stage invariants:

| Stage | Required invariant |
| ---- | ---- |
| Resolution | Names, constructors, and variable references are resolved |
| Type check | Type variables and effect rows are solved |
| TDNR | Method-style calls are converted to resolved calls |
| AST-to-IR | `tribute_control.callable`과 callable/control operation, valid SSA use chain |
| Shared CPS legalization | 전체 callable graph가 physical `CallableAbi`와 direct/indirect tail transfer를 사용하며 `tribute_control` operation/type이 남지 않음 |
| Shared lowering | 명시된 경계에서 high-level ability dispatch operation이 제거됨 |
| Effect ABI | `effect.*` operations preserve dispatch semantics without backend layout details |
| Backend lowering | Backend-ready 검증이 성공하고 `effect.*`, CPS control carrier, trampoline이 남지 않음 |

## Type Model

Primitive scalar, pointer, reference, bytes, array, tuple, function, and nil
types are represented in TrunkIR. Library data types such as `Option`, `Result`,
and `Text` lower through ADT and runtime/library conventions. `List` is an
opaque nominal builtin whose shared construction and sequence-view observations
use `list.*`; target-specific passes choose and eliminate its private
representation.

## Open Questions

- Final closure environment representation for each backend.
- Reusable conversion targets for effect ABI lowering boundaries.
- Debug/source-map representation.
