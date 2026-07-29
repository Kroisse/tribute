# Direct control and CPS effect-handling pipeline

This document defines the logical direct-style/CPS boundary and records the
current compatibility implementation used while M2 migrates to it.

핵심 전략은 **tail-call CPS + evidence-based handler dispatch**이다.
WasmGC yield bubbling, `YieldResult` 중심 trampoline, `cont.*` dialect 직접
lowering은 현재 경로가 아니다.

## M2 direct-style boundary

The target pipeline is:

```text
typed AST
  -> verified direct-style tribute_control IR
  -> shared CPS legalization in tribute-passes
  -> ability.* / effect.* / closure.* IR
  -> target-specific lowering
```

This boundary is specified by #821 but is not implemented merely because this
document describes it. #822 adds the dialect, #823 adds conversion over
synthetic IR, #824 migrates frontend emission, #825 makes legalization
mandatory in both backend pipelines, and #826 removes the superseded
frontend-owned normalization and continuation construction.

`tribute_control` contains only `perform`, `handle`, `handler`, `resume`, and
`yield`, plus the opaque affine `resume_token<input, answer>` type. Their full
structural contracts are in [ir.md](ir.md#direct-style-control). Ordinary
calls and data operations stay in their existing dialects. There is no
effectful invocation operation: existing direct/indirect call operations and
their callable convention metadata already distinguish `Direct`,
`EvidenceDirect`, and `Cps`. Every ability operation invocation, whether
declared `fn` or `op`, is `tribute_control.perform`. Its required
`operation_kind = @fn | @op` metadata preserves the kind established by source
typechecking.

Source effects and worker conventions do not change at this boundary:

```text
Direct < EvidenceDirect < Cps
```

Effect rows continue to contain ability identities rather than operation
identities. Closed empty, fn-only, general-op, and open-row convention
selection retain their existing meanings. ANF is a representation invariant
inside each executable region, not a source phase and not the identity of the
dialect. The operation kind and callable convention are distinct facts:
typechecking never infers `fn` from a tail-resumptive-looking `op` body, and
the frontend does not select a dispatch representation.

### Pre-CPS callable shape

The `tribute-control-pre-cps` boundary uses source-logical callable signatures
for every convention:

- `func.func` entry block arguments are exactly the source parameters, and its
  result is the source result.
- `closure.lambda` block arguments and its `closure<func<...>>` result type use
  the same source-logical parameters and result.
- `func.call` and `func.call_indirect` pass only source arguments and produce
  the source result.
- Definitions, lambdas, function/closure types, and call sites carry the
  existing `CallingConvention` metadata. They do not encode the convention by
  pre-inserting operands.

Consequently, neither `EvidenceDirect` nor `Cps` has a hidden evidence
parameter at this boundary, and `Cps` has no frontend-created `done_k`.
The typing/convention analysis has already established convention metadata,
which `tribute-front` preserves without constructing its physical ABI.

At this same boundary, `tribute-front` emits every ability invocation as
`tribute_control.perform` and copies the typechecked declaration kind into
`operation_kind`. It does not emit `ability.call`, `ability.perform`, or
`effect.dispatch_*`, infer kind from the handler body, or choose tail versus
CPS dispatch.

Issue #823 signature-converts definitions and every matching direct/indirect call
site together, before closure extraction:

| Convention | Physical parameters after #823 | Physical result during M2 |
| ---- | ---- | ---- |
| `Direct` | source parameters | source result |
| `EvidenceDirect` | evidence, then source parameters | source result |
| `Cps` | evidence, `done_k`, then source parameters | current opaque compatibility control result |

The parameter order is the existing `CallableAbi` order. For `Cps`, `done_k`
accepts the source result and returns the same opaque control-result type as
the worker. During M2 that physical type may remain the current compatibility
`anyref`; this table does not make it a permanent logical result or select the
future `Never`/`Step` policy owned by #774.

The conversion uses TrunkIR's `TypeConverter` and signature-conversion support
to rewrite `func.func` entry arguments, `core.func` types, closure types,
`closure.lambda` entries, and direct and indirect call sites consistently.
It threads the current evidence value into `EvidenceDirect` and `Cps` calls.
It converts a `Cps` body `func.return value` into
`return done_k(value)` in the compatibility representation; direct-style
control points and structured-region yields are converted with the same body
continuation. A call whose metadata and callable type disagree is a conversion
failure, not a reason to guess hidden operands.

In the same #823 legalization, each `tribute_control.perform` is converted
from its typechecked `operation_kind`: `@fn` produces the existing
`ability.call`/tail-dispatch path without capturing the suffix, while `@op`
produces the existing `ability.perform`/CPS path. This decision is not encoded
in `CallableAbi`, and the pass must not infer or change the operation kind.

This joint signature conversion is why a separate
`tribute_control.invoke` operation is unnecessary. Callable convention
metadata selects the conversion, while existing call and closure operations
continue to represent invocation and construction.

Root source `main` is the one composition-owned delimiter. Its external worker
remains `Direct` for a pure entry or `EvidenceDirect` for an `Io` entry; it is
never exposed to a backend as `Cps`. If evaluating its body requires a `Cps`
chain only because of a generic/open callback, #823 closes that body with a
compiler-generated terminal continuation while retaining the external
signature. A Direct root obtains the existing empty-evidence value for the
internal chain; an EvidenceDirect root uses its newly inserted evidence
argument. The root orchestration/conversion layer, not `tribute-front`, owns
this delimiter and the final source-result return.

### Logical CPS legalization

The shared conversion lowers one executable region as
`convert_region(region, exit_k)`. `exit_k` is a logical continuation for the
region result; it is not a selected backend carrier. The conversion consumes
operations from left to right:

- An ordinary direct operation stays in its existing dialect and its result
  flows to the remaining suffix.
- A `Cps` `func.call` or `func.call_indirect` uses its existing convention
  metadata and the jointly converted callable type. Conversion supplies
  evidence and a continuation for the suffix using `CallableAbi` order.
- `tribute_control.perform` branches only on its verified `operation_kind`.
  For `@fn`, conversion does not capture the suffix and produces the existing
  `ability.call`/`effect.dispatch_tail` path; the returned operation result
  flows through the ordinary suffix. For `@op`, it captures the current suffix
  and all selected enclosing structured exits up to the matching dynamic
  handle boundary, then produces the existing
  `ability.perform`/`effect.dispatch_cps` path. Operands are packed only at
  that lower boundary. For `op -> Never`, conversion captures no suffix and
  supplies a real zero-capture reject continuation to the existing required
  ABI operand.
- `tribute_control.yield value` invokes the region's `exit_k(value)`.

At `scf.if`, case lowering, a guarded arm, or short-circuit selection, each
executable branch is converted independently with an exit continuation that
first reaches the structured merge and then runs the enclosing suffix. The
condition, scrutinee, and preceding strict values are evaluated once in source
order. Only the selected branch, guard, or right-hand side executes.

`tribute_control.handle` creates a delimiter:

1. Normal body yield enters the completion region exactly once; completion
   yield continues through the enclosing `exit_k`.
2. The nearest dynamically installed matching handler receives the operation
   arguments. A `fn` arm yields the operation result and resumes
   automatically.
3. A resumptive general `op` arm receives an affine resumption. Its
   `tribute_control.resume` supplies the performed operation's result, runs
   the captured body continuation, and then runs the arm-local suffix with the
   returned handle answer. A source `op -> Never` arm receives no resumption
   and cannot contain `resume`.
4. A general arm that yields without resuming completes that handle directly.
   It bypasses the abandoned performed-computation suffix and the completion
   region.
5. A nested handle introduces the same delimiter recursively. A resumed path
   re-enters every selected case/conditional/short-circuit/nested-handle frame
   captured between the perform and its handler. A non-resumed path abandons
   those frames.

The converter must derive all continuations from regions, block suffixes,
`tribute_control.yield`, verified `operation_kind`, and callable convention
metadata. It must not rescan the typed AST, inspect AST containment, infer an
operation kind, or add construct-specific rules for case, guard, short-circuit,
or nested handles.

The affine `resume_token` is a logical ownership capability. It may be unused
or consumed once. Source `op -> Never`, identified by canonical `core.never`
rather than an erased `anyref` placeholder, creates no token. Closure capture
transfers the token's static ownership path; copying, storing, returning, or
yielding it is invalid. Operation-local verification checks token/result types
and rejects a token or nested `resume` in a `Never` handler. Whole-IR
validation follows use-def and closure-capture edges to enforce one static
ownership path and lexical association with one resumptive general handler.

Static SSA validation does not prove dynamic one-shot behavior after capture:
the same closure value may be invoked repeatedly. Conversion must therefore
lower a captured resumption with runtime consumed state, and a second
invocation must be rejected or trap before re-entering the continuation. This
is the existing scoped-resumption runtime responsibility, not a backend
carrier-selection decision. Conversion consumes the direct token and does not
expose it in post-CPS IR.

The `op -> Never` reject continuation is a compiler-generated closure with the
ordinary continuation ABI, no captures, and a `func.unreachable` body. It may
be shared within one compilation unit. `ability.perform` receives that typed
closure, and `effect.dispatch_cps` receives its ordinary closure-to-`anyref`
conversion. Null, an in-band sentinel, and arbitrary `anyref` are invalid
substitutes. Invocation traps because a `Never` handler is forbidden to
resume; the adapter is an ABI guard, not a source path or backend carrier
choice.

### Legalization boundaries

The named `tribute-control-pre-cps` boundary accepts verified
`tribute_control.*` beside ordinary high-level and mid-level dialects.
Frontend conformance makes all existing `ability.*`, `effect.*`, and legacy
CPS-dispatch operations illegal. During a partial #823 rewrite, source and
lowered operations may coexist transiently; that transient state is not the
named pre-CPS boundary.

Successful shared conversion verifies the partial
`tribute-control-post-cps` target, whose defining rule is
`illegal_dialect("tribute_control")`. Any residual direct-control operation is
a conversion failure at its source location. Later shared passes may still
consume `ability.*`, `effect.*`, and `closure.*`. Native and Wasm
backend-ready Tribute boundaries independently reject residual
`tribute_control.*`, `ability.*`, and `effect.*`; they do not rely only on a
previous pass having run.

논리적 CPS 함수는 source result를 직접 반환하지 않는다. 완료 값은 `done_k`의
인자로 전달되고 함수와 continuation의 control result는 `Never`다. 아래 예시의
`anyref` result와 `func.return %result`는 현재 구현이 true tail call 대신
continuation chain의 결과를 되돌려 보내기 위해 사용하는 compatibility carrier다.
향후 control lowering은 이를 true tail call의 `Never` 또는 trampoline의 `Step`으로
대체할 수 있다.

## Current private CPS completion carrier (#815)

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
target-independent reference equality or an owner allocation. This is a
single private compatibility representation, not the logical
`tribute_control` semantic contract and not #774's general backend
carrier-selection policy.

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

`fn`으로 선언된 ability operation은 tail-resumptive임을 선언부에서 보장한다.
Typechecking은 이 source kind를 보존하고 frontend는 모든 ability invocation과
마찬가지로 direct-style `perform`을 만든다:

```text
%result = tribute_control.perform %arg {
  ability_ref = @Logger,
  op_name = @log,
  operation_kind = @fn
}
```

Issue #823 semantic legalization이 `operation_kind = @fn`을 보고 continuation을
만들지 않은 채 기존 `ability.call` 경로로 내린다:

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

`op`으로 선언된 general operation도 frontend에서는 source-logical 결과를 갖는
direct-style `perform`이다:

```text
%result = tribute_control.perform {
  ability_ref = @State,
  op_name = @get,
  operation_kind = @op
}
```

Issue #823 semantic legalization이 `operation_kind = @op`을 보고 block suffix에서
명시적인 continuation closure를 구성한 뒤 `ability.perform`으로 내린다.

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

### Current pre-M2 frontend value/computation boundary

The following `ast_to_ir` normalization and `lower_value`/`lower_comp` split is
the current implementation baseline. Its compositional behavior is evidence
for the M2 region conversion and must remain correct during migration, but
frontend-owned normalization and continuation construction are not the
permanent phase boundary. After #824, `tribute-front` emits verified
direct-style regions; after #826, superseded CPS-only frontend machinery is
removed.

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

`#816`은 현재 `anyref` compatibility carrier를 보존한다. #815 adds only the
owner-tagged private escape protocol above. General logical/backend
control-result selection remains #774 work. The direct-style
`tribute_control` contract is fixed above; implementation and migration remain
owned by Issues #822 through #826.

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

## Current shared middle-end pipeline

Until #823-#825 land, the implemented shared pipeline remains:

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

Future effect specialization and handler inlining use the same validation
contract defined in `optimizations.md`.

In this current baseline, `ast_to_ir` creates effectful functions and closures
with evidence parameters and the compatibility CPS representation. This is
descriptive, not the permanent ownership boundary. The M2 target inserts
`tribute_control` CPS legalization before the existing ability/effect passes;
after #824 the frontend does not construct continuations, and #826 removes the
superseded path. Shared lowering still emits the same `effect.*` ABI
operations, which backends lower into evidence runtime calls, closure
decomposition, and target-specific indirect calls.

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
- Payload values are already packed into a single value by shared lowering.
  The M2 frontend retains source-logical `tribute_control.perform` operands
  and does not pack them for a dispatch strategy. Missing payloads are
  represented explicitly by a target-independent null/empty value before
  reaching `effect.*`.

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
