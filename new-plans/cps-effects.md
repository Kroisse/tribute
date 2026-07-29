# 직접형 제어와 CPS 효과 처리 파이프라인

이 문서는 직접형 IR을 CPS로 바꾸는 conversion source of truth다. Operation
구조와 verifier 계약은 [ir.md](ir.md#direct-style-control), 계층별 소유권과
마이그레이션 순서는 [implementation.md](implementation.md#직접형-제어-소유권)를
따른다.

핵심 전략은 **tail-call CPS + evidence-based handler dispatch**이다.
WasmGC yield bubbling, `YieldResult` 중심 trampoline, `cont.*` dialect 직접
lowering은 현재 경로가 아니다.

<!-- markdownlint-disable-next-line MD033 -->
<a id="direct-style-control-boundary"></a>

## 직접형 호출 대상과 제어 경계

모든 ability operation invocation은 typechecking이 확정한
`operation_kind = @fn | @op`를 가진 `tribute_control.perform`이다. 일반 callable도
`tribute_control.func`, `lambda`, `func_ref`, `call`, `call_indirect`, `return`을
사용한다. Operation kind와 callable convention은 typechecking 결과이며
frontend나 conversion이 body 형상에서 추론하거나 재분류하지 않는다.

```text
Direct < EvidenceDirect < Cps
```

Effect row, convention 순서, 실행 region 내부의 ANF invariant는 바뀌지 않는다.

<!-- markdownlint-disable-next-line MD033 -->
<a id="pre-cps-callable-shape"></a>

### CPS 변환 전 호출 대상 형상

입력 형상은 [ir.md](ir.md#direct-style-control)의 operation/type 계약을 따른다.
모든 callable은 exact `tribute.calling_convention` code 0/1/2를 가진 logical
type과 source parameter/result만 사용하며 evidence, environment, `done_k`는
아직 없다.

Issue #823은 closure extraction 전에 전체 callable graph를 하나의 단위로
변환한다:

| Convention | #823 이후 physical parameter | 현재 physical result |
| ---- | ---- | ---- |
| `Direct` | 소스 parameter | 소스 result |
| `EvidenceDirect` | evidence 뒤 소스 parameter | 소스 result |
| `Cps` | evidence, `done_k`, source parameter 순서 | 현재 opaque compatibility control result |

Parameter 순서는 기존 `CallableAbi` 순서다. `Cps`의 `done_k`는 source result를
받고 worker와 같은 opaque control-result type을 반환한다. 현재 physical type은
compatibility `anyref`를 유지할 수 있지만, 이는 영구 logical result나 #774의
향후 `Never`/`Step` 정책이 아니다.

Conversion은 먼저 모든 logical callable type, definition, lambda, `func_ref`,
direct/indirect call, return의 대응 관계를 검증하고 physical symbol과 type을
계획한 뒤 함께 rewrite한다:

- `tribute_control.func`는 `func.func`와 physical `core.func`를 만든다.
- `tribute_control.lambda`는 physical `closure.lambda`와
  `closure.closure<core.func<...>>`를 만든다. 이 단계의 signature에는
  `CallableAbi` hidden parameter가 있지만 environment는 없으며, 기존 closure
  lowering이 environment를 interpose한다.
- `tribute_control.func_ref`는 빈 environment의 `closure.new`와 env-bearing
  adapter `func.func`를 만든다. Adapter는 result callable의 같거나 더 강한
  convention을 사용하고 필요한 hidden operand와 source argument를 대상 physical
  worker에 전달하므로 named function도 다른 closure와 같은 `call_indirect` ABI를
  갖는다. 기존 closure lowering은 이 `closure.new`를 `func.constant`와 physical
  closure struct로 바꾼다.
- `tribute_control.call`과 `call_indirect`는 각각 physical `func.call`과
  `func.call_indirect`가 된다. 현재 evidence와 `Cps` suffix continuation을
  `CallableAbi` 순서로 삽입하고, indirect call의 environment는 후속 closure
  lowering이 삽입한다.
- `tribute_control.return`은 `Direct`/`EvidenceDirect`에서 `func.return`이 된다.
  `Cps`에서는 `done_k(value)`를 호출하고 compatibility control result를
  `func.return`한다.

기존 `CallableAbi::interpose_environment`에 따라 extracted lambda와 `func_ref`
adapter의 최종 parameter 순서는 `Direct`에서 `environment, source...`,
`EvidenceDirect`에서 `evidence, environment, source...`, `Cps`에서
`evidence, environment, done_k, source...`다. `call_indirect`도 같은 순서로
environment를 삽입한다.

생성한 physical definition, lambda, adapter, direct/indirect call에는 logical
type의 convention을 기존 `tribute.calling_convention` attribute로 복사한다.
Metadata/type/symbol 불일치는 conversion failure이며 hidden operand를 추측하지
않는다. 현재 TrunkIR의 `TypeConverter`, function signature conversion, dialect
builder를 재사용하되 `tribute_control` 전용 graph pattern은 #823이 구현한다.

같은 #823 legalization에서 각 `tribute_control.perform`을 typecheck된
`operation_kind`에 따라 변환한다. `@fn`은 suffix를 capture하지 않고 기존
`ability.call`/tail-dispatch 경로를 만들며, `@op`은 기존
`ability.perform`/CPS 경로를 만든다. 이 결정은 `CallableAbi`에 encode하지 않으며
pass가 operation kind를 추론하거나 변경해서는 안 된다.

Root `main` delimiter와 external ABI 조합 책임은
[implementation.md](implementation.md#직접형-제어-소유권)을 따른다.

### 논리적 CPS 적법화

Shared conversion은 실행 가능한 region 하나를
`convert_region(region, exit_k)`로 lower한다. `exit_k`는 region result를 위한
logical continuation이며 선택된 backend carrier가 아니다. Conversion은
operation을 왼쪽에서 오른쪽으로 소비한다:

- 일반 value operation은 기존 dialect에 남고 그 result는 남은 suffix로 흐른다.
- `tribute_control.call` 또는 `call_indirect`의 convention이 `Cps`이면 현재
  suffix continuation을 공급한다. `Direct`와 `EvidenceDirect` call result는
  일반 suffix로 흐른다.
- `tribute_control.perform`은 검증된 `operation_kind`만으로 분기한다. `@fn`이면
  suffix를 capture하지 않고 기존 `ability.call`/`effect.dispatch_tail` 경로를
  만들며 반환된 operation result가 일반 suffix로 흐른다. `@op`이면 현재
  suffix와 일치하는 dynamic handle boundary까지 선택된 모든 enclosing
  structured exit를 capture한 뒤 기존 `ability.perform`/`effect.dispatch_cps`
  경로를 만든다. Operand packing은 이 lower 경계에서만 수행한다.
  `op -> Never`에는 suffix를 capture하지 않고 기존 필수 ABI operand에 실제
  zero-capture reject continuation을 공급한다.
- `tribute_control.yield value`는 region의 `exit_k(value)`를 호출한다.

`scf.if`, case lowering, guarded arm, short-circuit selection에서는 각 실행 branch를
독립적으로 변환한다. 각 branch의 exit continuation은 먼저 structured merge에
도달한 뒤 enclosing suffix를 실행한다. Condition, scrutinee, 앞선 strict value는
source 순서로 한 번만 평가하며 선택된 branch, guard, 오른쪽 항만 실행한다.

`tribute_control.handle`은 delimiter를 만든다:

1. Normal body yield는 completion region에 정확히 한 번 들어가며 completion
   yield는 enclosing `exit_k`를 계속 실행한다.
2. Dynamic하게 설치된 handler 중 가장 가까운 일치 handler가 operation
   argument를 받는다. `fn` arm은 operation result를 yield하고 자동으로 resume한다.
3. Resumptive general `op` arm은 affine resumption을 받는다.
   `tribute_control.resume`은 수행된 operation result를 공급하고 capture한 body
   continuation을 실행한 뒤, 반환된 handle answer로 arm-local suffix를 실행한다.
   Source `op -> Never` arm은 resumption을 받지 않고 `resume`을 포함할 수 없다.
4. Resume하지 않고 yield하는 general arm은 해당 handle을 직접 완료한다. 포기된
   performed-computation suffix와 completion region을 건너뛴다.
5. Nested handle은 같은 delimiter를 재귀적으로 만든다. Resumed path는 perform과
   handler 사이에서 capture된 모든 case/conditional/short-circuit/nested-handle
   frame에 다시 진입한다. Non-resumed path는 해당 frame을 포기한다.

Converter는 region, block suffix, `tribute_control.yield`, 검증된
`operation_kind`, callable convention metadata에서 모든 continuation을
도출해야 한다. Typed AST를 다시 scan하거나 AST containment를 검사하거나
operation kind를 추론하거나 case, guard, short-circuit, nested handle 전용 규칙을
추가해서는 안 된다.

Affine `resume_token`의 type/placement는 operation-local verifier가, use-def와
closure-capture를 지나는 single static ownership path는 whole-IR verifier가
검사한다. Static SSA 검사는 capture된 closure의 반복 호출을 막지 못하므로
conversion은 runtime consumed state를 만들고 두 번째 호출을 continuation
재진입 전에 거부하거나 trap한다. Token은 post-CPS IR에 남지 않는다.

Canonical `core.never`인 source `op -> Never`는 token과 source suffix
continuation을 만들지 않는다. 기존 ABI에는 capture가 없고 body가
`func.unreachable`인 typed reject continuation을 전달한다. Null, in-band
sentinel, 임의의 `anyref`는 대체할 수 없으며 호출되면 trap한다.

### 적법화 경계

`tribute-control-pre-cps` named boundary는 검증된 `tribute_control.*`과 일반
value/structured dialect만 허용한다. Frontend 적합성 검사는 `func.*`,
`closure.*`, `core.func`, `closure.closure`, 기존 `ability.*`, `effect.*`,
legacy CPS-dispatch operation을 모두 거부한다. Issue #823의 partial rewrite
도중에는 logical/physical operation이 일시적으로 공존할 수 있지만 이 상태는
named pre-CPS boundary가 아니다.

성공한 shared conversion은 defining rule이
`illegal_dialect("tribute_control")`인 partial
`tribute-control-post-cps` target과 Tribute type walk를 검증한다. 남은
`tribute_control` operation 또는 `callable`/`resume_token` type은 source
location에서 conversion failure가 된다. 이 경계에는 일관된 physical
`func.*`/`closure.*`/`core.func` graph와 lowered ability/effect operation만
남는다. Native와 Wasm의 backend-ready Tribute boundary는 남은
`tribute_control.*`, `ability.*`, `effect.*`를 각각 독립적으로 거부한다.

논리적 CPS 함수는 source result를 직접 반환하지 않는다. 완료 값은 `done_k`의
인자로 전달되고 함수와 continuation의 control result는 `Never`다. 아래 예시의
`anyref` result와 `func.return %result`는 현재 구현이 true tail call 대신
continuation chain의 결과를 되돌려 보내기 위해 사용하는 compatibility carrier다.
향후 control lowering은 이를 true tail call의 `Never` 또는 trampoline의 `Step`으로
대체할 수 있다.

## 현재 비공개 CPS 완료 전달체 (#815)

Until #774 supplies a general logical/backend control-result separation, the
narrow #815 compatibility protocol at a handle boundary is:

```text
__tribute_cps_control = Normal(anyref) | Escape(owner_tag: i32, payload: anyref)
```

This is neither a source type nor an ABI convention. Its physical ABI remains
`anyref`; only lowering code that has constructed the carrier, or received it
at a boundary documented to produce it, may use `adt.variant_is`,
`adt.variant_cast`, or `adt.variant_get`. In particular, arbitrary source
values, public ADTs, and in-band sentinels must never be tested as this carrier.
`owner_tag` is the existing runtime-unique prompt tag allocated for each
dynamic handler installation. It is an integer token, not a syntactic tag, a
source value, a closure/reference identity, or a newly allocated owner object.

The body of a handle receives a `Normal`-producing done continuation. Raw CPS
producers are adapted to `Normal` only when that exact private continuation is
passed to them. A general `op` handler arm that completes without resuming
returns `Escape(owner_tag, value)`, where the tag is read from the evidence
marker that selected that handler. `Normal(value)` alone continues through the
performed computation's normal continuation and then the handle's `do` clause.

The protocol composes through every nested handle answer:

- `Normal(value)` runs that handle's `do` clause and then continues normally.
- `Escape(owner, value)` at a non-owner handle is forwarded unchanged. It does
  not run that handle's `do` clause, any source continuation, or a resumed
  handler arm's strict suffix.
- `Escape(owner, value)` at the matching dynamic owner completes that handle
  with `value`, bypassing its `do` clause. Its parent receives this as an
  ordinary normal completion.

Consequently, a `HandleAnswer` remains an opaque, proven private carrier while
it can cross a handle delimiter. The matching dynamic owner consumes an
`Escape` before the final source cast; `Normal` has already run the selected
handle's `do` clause. The final source boundary therefore receives the resolved
logical value and performs no carrier probe or tag-specific unwrap. This
preserves #817's compositional `lower_value`/`lower_comp` API: no AST
containment scan, special case/short-circuit mode, duplicate raw lowering path,
or source effect-row or worker-convention change is introduced.

The shared evidence ABI threads the dynamic tag through `effect.extend`,
`ability.handle_dispatch`, and the general handler-dispatch closure. Native
and Wasm obtain the same tag from the selected evidence marker before invoking
that closure. Shared lowering compares integer owner tags; it does not require
target-independent reference equality or an owner allocation. 이는 하나의
private compatibility representation일 뿐 logical
`tribute_control` semantic contract도, #774의 general backend carrier-selection
정책도 아니다.

현재 compatibility representation에서 캡처 없는 identity `done_k`의 함수 본문은
컴파일 단위 전체에서 동일하다. AST-to-IR lowering은 이 내부 함수 정의를 compilation
root에 한 번만 만들고 모든 사용 지점에서 같은 함수 심볼을 참조한다. 다만 각 사용
지점의 null environment와 `closure.new`는 SSA 영역 가시성을 지키기 위해 해당 영역에
각각 생성한다. 독립적으로 codegen되는 compilation unit은 자체 정의를 가지며, 향후
separate compilation이 도입되면 backend의 link-once 정책으로 합칠 수 있다.
이 deduplication은 독립적으로 끌 수 있어야 하며, 동일한 frontend IR
경계에서 enabled/disabled snapshot과 native 실행 결과를 비교한다.

## 핵심 설계

### `fn` operation: direct dispatch

직접형 입력은 위 규칙의 `operation_kind = @fn` perform이며, #823은
continuation을 만들지 않고 기존 `ability.call` 경로로 내린다:

```text
%result = ability.call %arg
  { ability_ref = @Logger, op_name = @log }
```

Shared lowering converts it to a target-independent effect ABI operation:

```text
%payload = cast %arg to anyref
%result = effect.dispatch_tail %ev, %payload
  { ability_ref = @Logger, op_name = @log }
```

Native lowering then lowers that ABI operation to the current evidence lookup
and indirect-call representation:

```text
%marker = ability.evidence_lookup %ev { ability_ref = @Logger }
%tr_dispatch = adt.struct_get %marker, MarkerField::TrDispatchFn
%fn = adt.struct_get %tr_dispatch, 0
%env = adt.struct_get %tr_dispatch, 1
%op_idx = arith.const <hash(Logger, log)>
%result = func.call_indirect %fn(%ev, %env, %op_idx, %arg_anyref)
```

즉 `fn` operation은 CPS 변환, continuation allocation, resume dispatch를
우회한다.

### `op` operation: tail-call CPS dispatch

직접형 입력은 위 규칙의 `operation_kind = @op` perform이며, #823은 block
suffix에서 continuation closure를 구성해 `ability.perform`으로 내린다.

```text
%result = ability.perform %continuation, %arg
  { ability_ref = @State, op_name = @get }
```

Shared lowering converts it to a target-independent effect ABI operation:

```text
%payload = cast %arg to anyref
%cont = cast %continuation to anyref
%result = effect.dispatch_cps %ev, %cont, %payload
  { ability_ref = @State, op_name = @get }
```

Native lowering then finds the `handler_dispatch` closure in evidence and
tail-calls it:

```text
%marker = ability.evidence_lookup %ev { ability_ref = @State }
%handler = adt.struct_get %marker, MarkerField::HandlerDispatch
%owner_tag = adt.struct_get %marker, MarkerField::PromptTag
%fn = adt.struct_get %handler, 0
%env = adt.struct_get %handler, 1
%op_idx = arith.const <hash(State, get)>
%result = func.call_indirect %fn(
  %ev, %env, %continuation_anyref, %owner_tag, %op_idx, %arg_anyref)
func.return %result
```

Effect point 이후의 코드는 이미 `%continuation` closure 안에 있으므로,
`ability.perform` 이후의 같은 function-body ops는 dead code가 된다.

이 lowering은 source kind를 재분류하지 않는다. 일반 `op` handler가 실제로
항상 tail-resumptive인지 분석하여 tail path로 최적화하는 작업은 표준 `@op`
semantic lowering 이후의 별도 IR optimization이다.

### 현재 frontend 값/계산 경계

다음 `ast_to_ir` normalization과 `lower_value`/`lower_comp` 분리는 현재
implementation baseline이다. Compositional 동작은 직접형 region conversion의
근거이며 migration 도중에도 정확해야 한다. 그러나 frontend-owned
normalization과 continuation 구성은 영구 phase boundary가 아니다. #824 뒤에는
`tribute-front`가 검증된 직접형 region을 emit하고 #826 뒤에는 대체된 CPS-only
frontend machinery를 제거한다.

`ast_to_ir`는 lowering 직전에 작은 typed-AST A-normalization을 한 번 수행한다.
이것은 새 HIR, source language phase, 또는 source effect 의미가 아니다. 기존
`Expr<TypedRef>` / `Stmt<TypedRef>`만을 반환하는 target-independent administrative
layer이며, CPS computation entry에서 strict child를 source 순서대로 fresh typed
local에 bind한다. 따라서 block CPS lowering은 nested-call 문법 탐색이 아니라
정상형 block만 소비한다.

정규화는 CPS child만이 아니라 같은 strict sequence의 모든 non-atomic child를
atomize한다. call의 computed callee와 arguments, constructor/tuple/list element,
record spread/field는 source 순서로 처리한다. short-circuit RHS, case guard/arm,
lambda body, handle body와 handler arm은 독립 region 안에서 재귀 정규화되며
바깥으로 hoist하지 않는다.

semantic lowering API는 다음 두 경계를 유지한다.

```text
lower_value(expr)       -> source ValueRef
lower_comp(expr, k)     -> 현재 logical control result
```

`lower_value`는 source value를 만드는 직접 평가만 한다. 잠재적으로 `Cps`인
expression을 받으면 raw lowering하지 않고 lowering invariant 위반으로 거부한다.
lambda construction은 lambda body 또는 호출의 latent effect와 관계없이 이 경로에
남는다. `Direct`,
`EvidenceDirect`, `Cps`의 선택은 source effect row와 function-level convention에
의해서만 결정되며, 이 경계가 source row 의미를 바꾸지 않는다.

Worker convention selection starts from the concrete-effect prescan. Before
lowering bodies, `ast_to_ir` monotonically promotes a `Direct` or
`EvidenceDirect` definition to `Cps` when `evaluation_control_class(body)` is
`Cps`, then repeats over all definitions until no convention changes. It never
demotes a worker. This lets a promoted named callee and recursive call graph
propagate Cps requirements while leaving a pure unannotated worker `Direct`.
For example, an `Option::map`-style worker that invokes an open-effect callback
is promoted to `Cps` even though its concrete declared effect row is empty.
Lambda construction remains a direct expression. Its worker convention is
already selected from the inferred lambda function type, whose open effect row
conservatively selects `Cps`, so definition promotion does not need a separate
lambda-specific pass.

### Root `main` delimiter

Root `main` is the one target-independent CPS delimiter. Its valid source
residual-effect contract remains the existing pure-or-`Io` entry contract; a
residual general effect is still diagnosed before backend lowering. The
concrete-effect prescan therefore keeps the top-level root `main` worker at
its `Direct` or `EvidenceDirect` seed even when evaluating its body must use
`Cps` solely because it calls a generic/open callback worker. Nested-module
functions named `main` are ordinary workers and are not this delimiter.

When that root body is a computation, frontend lowering supplies the shared
identity `done_k`, lowers with `lower_comp`, and converts the completed opaque
compatibility result to the root source return type at this one sanctioned
entry boundary before `func.return`. A Direct root creates empty evidence
through the existing evidence placeholder path; an EvidenceDirect root uses
its ABI evidence argument. This closes an implementation-level open callback
convention without changing source effects or asking native/Wasm backends to
accept a `Cps` `main`. Backend-ready conversion continues to reject genuine
residual `effect.*` operations.

The source entry contract returns `Nil`, so the delimiter materializes that
`Nil` after the completed control chain and deliberately discards its opaque
completion payload; it never exposes the carrier as a source value.

`lower_comp`는 normalized top-level CPS producer (`op`, Cps named/local/computed
call, structured case/short-circuit/resume)를 만났을 때 그 결과를 받는
continuation으로 나머지 normalized block을 lowering한다. 결과는 compatibility
`anyref` carrier를 감싼 opaque control result이며 `func.return`, `scf.yield`, CPS
call/perform 같은 control sink에서만 소비한다. source cast, constructor, 또는
source call argument로 사용할 수 없다.

Convention metadata가 없는 synthetic compatibility continuation은
creation-time evidence를 closure environment에 capture하지 않는다. Closure lowering은
그 continuation의 가장 가까운 physical `func.func` entry 첫 argument가 evidence
type일 때에만, 새 lifted entry의 injected evidence argument로 그 정확한 SSA value를
remap한다. Explicit capture가 같은 value를 이미 소유하면 capture 의미를 우선한다.
따라서 nested synthetic continuation도 invocation-time ABI evidence를 다음 lifted
function으로 전달하며, 임의의 evidence-typed 외부 value를 type만으로 치환하지 않는다.

Strict subexpression은 normalization에서 source 순서대로 atomized된 뒤 같은
computation 안에서 이어진다.

```text
consume(effectful(), pure_arg)

let __cps_tmp0 = effectful()
let __cps_tmp1 = pure_arg
consume(__cps_tmp0, __cps_tmp1)
```

callee와 argument, tuple/constructor/record 요소, unary/binary operand, case
scrutinee와 guard는 이 규칙을 따른다. 선택적 위치는 독립 evaluation region을
만들고 그 region의 continuation을 공유한다.

- `&&`/`||`의 RHS는 선택된 `scf.if` region 안에서만 lowering한다.
- case arm과 guard는 scrutinee가 선택한 arm region 안에서만 lowering한다.
- `handle`은 source-value 위치에서는 `Direct`로 보이지만, lowering domain은
  context-sensitive다. `Ambient` 또는 `HandleAnswer` parent에서는
  `lower_handle_comp`가 private carrier를 유지해 nested non-owner `Escape`를
  전달한다. 오직 source/ambient entry가 `lower_handle_source`를 사용하며,
  dynamic owner가 control을 소비한 뒤 logical value를 cast한다. Raw
  Direct/EvidenceDirect entry는 handle 전체를 정확히 한 번 normalize하고,
  이미 normalized CPS parent는 다시 normalize하지 않는다.
- `resume`은 computation producer다. resumed `Normal`만 arm의 strict suffix로
  들어간다. suffix가 현재 owner의 `Escape`를 돌려 주면 enclosing `do`가 볼
  `Normal`로 retag하고, foreign `Escape`는 suffix, `do`, source continuation을
  건너뛰어 그대로 전달한다. resume does not cast a `HandleAnswer` to a source
  result.
- `handle` body와 handler arm은 설치된 evidence/handler boundary 안에서만
  lowering한다.
- case guard는 pattern match 뒤 arm-local strict evaluation으로 lowering한다.
  handler arm의 `resume`은 arm-local computation의 continuation을 사용한다.

따라서 실행되지 않은 branch는 eager하게 실행되지 않고, CPS producer가 handler
boundary 밖의 continuation을 캡처하지 않는다. 이 compositional path가 ad-hoc
nested-call lifting과 반복 containment scan을 대체한다. 현재 AST에는 unary
expression variant가 없으므로 unary strict-child normalization은 적용 대상이
아니다.

`#816`은 현재 `anyref` compatibility carrier를 보존한다. #815는 위의
owner-tagged private escape protocol만 추가한다. General logical/backend
control-result 선택은 계속 #774의 작업이다. 직접형 `tribute_control` contract는
위에서 확정했으며 구현과 migration은 Issues #822부터 #826까지가 담당한다.

Rust lowering은 sealed answer domain marker로 이 경계를 표현한다. `Ambient`는
함수/바깥 CPS chain의 control answer, `HandleAnswer`는 handle delimiter 안의
delimited answer, `TailResume`는 `fn` handler arm의 좁은 `ability.yield` endpoint다.
`ContinuationRef<D>`와 `ControlResultRef<D>`는 marker가 다른 carrier를 섞을 수
없게 한다. source value conversion is only after the dynamic owner has consumed
the private carrier at a source boundary; resume remains entirely in the
private answer domains.

### `handle`: evidence extension + handler closures

`handle` lowering은 두 종류의 dispatch closure를 만든다.

- `handler_dispatch`: `(k, owner_tag, op_idx, value) -> anyref`
  - general `op` handlers용
  - `owner_tag` is the selected Marker's dynamic prompt owner; `resume`은
    continuation closure 호출로 lowering된다.
- `tr_dispatch_fn`: `(op_idx, value) -> anyref`
  - `fn` handlers용
  - continuation 없이 handler 결과가 inline result가 된다.

`resolve_evidence`는 handler boundary에서 새 marker를 만들어 evidence를
확장한다.

Shared evidence resolution represents handler installation with the same effect
ABI instead of constructing the concrete Marker layout directly:

```text
%ev2 = effect.extend %ev, %prompt_tag, %tr_dispatch_fn, %handler_dispatch
  { ability_ref = @State }
```

Backends lower `effect.extend` to their own evidence representation. The native
backend maps it to the current `__tribute_evidence_extend` ABI.

```text
struct Marker {
    ability_id: i32,
    prompt_tag: i32,
    tr_dispatch_fn: ptr,
    handler_dispatch: ptr,
}
```

Evidence는 ability id 기준으로 정렬된 marker 배열이며, handler 설치 시
새 evidence 값을 만든다.

Marker layout과 evidence runtime ABI는 `tribute-ir`의
`ability::MarkerField`와 `ability::evidence_abi`가 컴파일러 내부의 단일
정의다. 필드 순서는 다음과 같고 모든 shared pass와 backend lowering은 이
순서를 직접 숫자로 복제하지 않는다.

| Field | Index | Type | Meaning |
| --- | ---: | --- | --- |
| `ability_id` | 0 | `i32` | stable ability key for sorted evidence lookup |
| `prompt_tag` | 1 | `i32` | prompt installed for the active handler |
| `tr_dispatch_fn` | 2 | `ptr` | tail-resumptive dispatch closure or null |
| `handler_dispatch` | 3 | `ptr` | full CPS dispatch closure or null |

WasmGC uses the same field order and shared field identifiers, but its concrete
GC marker type stores the dispatch closures as `anyref` closure references
instead of native `ptr` values. Wasm effect ABI lowering therefore expands
`effect.dispatch_tail` and `effect.dispatch_cps` into evidence lookup,
`wasm.struct_get` of the selected marker closure, closure table-index/env
decomposition, and `wasm.call_indirect`.

Empty evidence is represented in high-level IR as an empty `core.array(Marker)`
or null evidence placeholder, and backend lowering turns that into the target
runtime representation. Native lowering maps it to `__tribute_evidence_empty()`.
When a handler for the same `ability_id` is nested inside an outer handler,
evidence extension replaces the existing marker so lookup resolves to the
nearest handler.

Native runtime ABI:

```text
__tribute_evidence_empty() -> ptr
__tribute_evidence_lookup(ev: ptr, ability_id: i32) -> i32
__tribute_evidence_extend(
    ev: ptr,
    ability_id: i32,
    prompt_tag: i32,
    tr_dispatch_fn: ptr,
    handler_dispatch: ptr,
) -> ptr
__tribute_evidence_lookup_tr(ev: ptr, ability_id: i32) -> ptr
__tribute_evidence_lookup_handler(ev: ptr, ability_id: i32) -> ptr
```

### `ability.handle_dispatch`

현재 구현에서 `ability.handle_dispatch`는 runtime dispatch loop가 아니다.
Effect 발생 시점에서 이미 handler closure로 tail-call되므로,
`lower_handle_dispatch`는 body result에 `done` handler를 적용하는 역할만 한다.

<!-- markdownlint-disable-next-line MD033 -->
<a id="current-shared-middle-end-pipeline"></a>

## 현재 공통 middle-end 파이프라인

Issues #823-#825를 구현하기 전까지 현재 구현된 shared pipeline은 다음과 같다:

```text
ast_to_ir
→ lower_closure_lambda
→ intrinsic_to_arith
→ closure_lower
→ lower_ability_perform
→ convert_tail_resumptive
→ resolve_evidence
→ lower_handle_dispatch
→ effect ABI verification
→ backend-specific lowering
```

향후 effect specialization과 handler inlining은 `optimizations.md`에 정의한 동일한
validation contract를 사용한다.

현재 baseline에서 `ast_to_ir`는 evidence parameter와 compatibility CPS
representation을 갖는 effectful function과 closure를 만든다. 이는 현재 구현의
설명이지 영구 ownership boundary가 아니다. 계획한 target은 기존 ability/effect
pass 앞에 `tribute_control` CPS legalization을 삽입한다. #824 뒤에는 frontend가
continuation을 구성하지 않고 #826이 대체된 path를 제거한다. Shared lowering은
계속 같은 `effect.*` ABI operation을 emit하며 backend가 이를 evidence runtime
call, closure decomposition, target-specific indirect call로 lower한다.

## Effect ABI Boundary

The `effect` dialect is the target-independent boundary between language
semantics and concrete runtime layout.

Initial operations:

- `effect.extend(evidence, prompt_tag, tr_dispatch_fn, handler_dispatch)
  { ability_ref } -> evidence`
- `effect.dispatch_tail(evidence, payload) { ability_ref, op_name } -> result`
- `effect.dispatch_cps(evidence, continuation, payload)
  { ability_ref, op_name } -> result`

Rules:

- `ability.perform` and `ability.call` are illegal after the shared
  ability-dispatch lowering boundary.
- `effect.*` operations may remain after shared lowering and before
  backend-specific effect ABI lowering.
- Backend-ready conversion targets must reject residual `effect.*` operations.
- Shared passes must not inspect Marker field numbers, handler-table storage
  layout, closure field positions, or backend function-pointer representation.
- Payload value는 shared lowering에서 이미 single value로 pack한다. 직접형 frontend는
  source-logical `tribute_control.perform` operand를 유지하며 dispatch 전략에
  맞춰 pack하지 않는다. 누락된 payload는 `effect.*`에 도달하기 전에
  target-independent null/empty value로 명시적으로 표현한다.

## Backend Implications

### Native

Native target은 현재 주 개발 경로다. Evidence runtime은 `tribute-runtime`의
`__tribute_evidence_*` C ABI 함수로 제공되고, native effect ABI lowering은
`effect.*`를 marker lookup helper, runtime evidence extension, closure
decomposition, and indirect calls로 변환한다.

### WasmGC

WasmGC도 같은 shared middle-end를 사용한다. `wasm/evidence_to_wasm`은
`effect.extend`를 marker construction + `__tribute_evidence_extend` helper
call로 낮추고, `effect.dispatch_tail` / `effect.dispatch_cps`는
`__tribute_evidence_lookup`, marker closure field access, closure
table-index/env unpacking, and `wasm.call_indirect`로 낮춘다.

현재 WasmGC backend에는 이전 yield bubbling/trampoline 설계의 builtin 타입
(`Step`, `Continuation`, `ResumeWrapper`)이 남아 있다. 이 타입들은 active effect
ABI의 의미론적 기준이 아니며, WasmGC backend 우선순위를 올리기 전에 실제 필요
여부를 정리해야 한다.

## 폐기된 접근

다음 접근은 현재 구현 기준의 active path가 아니다.

- WasmGC yield bubbling
- Koka-style `YieldResult { Done, Shift }`를 effectful return type으로 전파
- `cont_to_yield_bubbling` pass
- `cont.*` dialect를 libmprompt 또는 stack switching으로 직접 lowering

관련 과거 설계는 git history에서 확인할 수 있지만, 새 구현 작업의 기준으로
사용하지 않는다.
