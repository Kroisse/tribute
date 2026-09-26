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
| `tribute-control-pre-cps` | frontend 적합성 검사에는 full, 변환 중에는 partial | `tribute_control.*`은 `core.module`, `core.never`와 일반 `core` 값 type, `scf`, `arith`, `adt`, `list`, `tribute_rt`, `tribute_io`와 공존할 수 있다. `func.*`, `closure.*`, `func.func_sig`, `ability.*`, `effect.*`는 illegal이다. |
| `tribute-control-post-cps` | shared CPS 변환 뒤 partial | `tribute_control` dialect의 모든 operation과 type이 illegal이다. Shared `func.*`, `closure.*`, `func.func_sig`, `ability.*`, `effect.*`, 일반 dialect는 이후 pass를 위해 공존할 수 있다. |
| `tribute-backend-ready-native` | 목표 최종 코드 생성 계약 | Native 최종 코드 생성 경계는 남아 있는 논리적 제어 표현을 거부하고 대상 연산/타입 및 물리적 함수 호출 규약을 검증해야 한다. |
| `tribute-backend-ready-wasm` | 목표 최종 코드 생성 계약 | Wasm 최종 코드 생성 경계는 같은 계약을 강제해야 한다. `verify_wasm_backend_ready`는 별개의 부분 검증이다. |

Post-CPS helper는
`ConversionTarget::new().illegal_dialect("tribute_control")`와 partial
verification을 조합한다. Full mode에서는
unknown operation이 legal하지 않으므로 frontend 적합성 target과 backend full
target이 legal dialect와 operation을 열거해야 한다. `ConversionTarget`은
operation 적법성만 검사하므로 Tribute whole-IR type walk가 pre-CPS의
`func.func_sig`/`closure.closure`와 post-CPS의
`tribute_control.func_sig`/`resume_token`을 별도로 거부한다. 최종 코드 생성의
안전성은 최종 Native/Wasm 코드 생성 경계의 목표 계약이다. 이 계약은 보존된
메타데이터를 실제로 생성되는 값 및 함수 시그니처의 입출력 항목과 구분해야 한다.
이 경계를 만족한다고 선언하는 컴파일 경로는 해당 검증을 코드 생성 전에 수행해야
한다. 대상별 코드 생성기는 자체 입력 검증을 담당한다. Wasm의
`verify_wasm_backend_ready`는 `ability.*`와 `effect.*` 제거를 확인하는 중간 단계의
부분 검증이며, 그 통과만으로 최종 경계의 모든 조건이 충족되지는 않는다. 논리 Unit은
`[core.nil]`이고, 논리 CPS는
`[core.never]`이며, 최종 물리적 CPS 결과 목록은 `[]`이다.
Region 소유 operation을
recursively legal로 표시해서 nested illegal operation을
가려서는 안 된다.

하나의 rewrite 도중에는 source `tribute_control.*`과 새
`ability.*`/`effect.*` 결과가 일시적으로 공존할 수 있다. 이는 partial
conversion 내부의 구현 상태이지 성공한 named boundary가 아니다. 성공한
`tribute-control-post-cps` verification은 남은 모든 `tribute_control.*`
operation을 source location과 함께 conversion failure로 보고한다.

## Validation Layers

TrunkIR validation is layered by responsibility. The compiler should use the
smallest layer that can state an invariant precisely.

| Layer | Responsibility | Failure timing | Examples |
| ---- | ---- | ---- | ---- |
| Operation verifier | Local invariants of one operation: operand/result/successor counts, operand/result/type-attribute type constraints and their equality or projection relations, required attributes, attribute domains, region shape, and terminator requirements that can be checked without global analysis. | At explicit operation validation checkpoints. Parsers, raw builders, and intermediate rewrites may temporarily violate schema constraints; only checks outside the schema, such as custom assembly parsing, may also fail at parse time. | `arith.cmpf` accepts only supported predicates; an op with regions requires the expected region count and terminator form. |
| Conversion target | Dialect and type legality at a named lowering boundary. | Immediately after a pass or pass group claims a conversion boundary. Partial conversion rejects explicitly illegal operations; full conversion also rejects unknown operations. | Ability lowering leaves no `ability.perform`; backend-ready native IR contains only `clif.*` plus allowed infrastructure ops. |
| Pass-manager verifier | Whole-IR consistency after transformations, with the offending pass identified. | After each pass registered in a `PassManager` when a verifier hook is installed. | SSA use-chain consistency, value visibility across isolated regions, or other graph-wide invariants. |
| Operation interface | Shared behavior queried generically across dialects. | At the consumer that needs dialect-independent behavior. Interfaces should be introduced only for multiple concrete consumers or one generic transform. | `PureOps` for DCE removability; `IsolatedFromAboveOps` for nested pass-manager anchoring. |

Local operation verifiers must not depend on conversion state or pass ordering.
Conversion targets must not duplicate local semantic checks. Pass-manager
verifiers should remain responsible for graph-wide invariants that require
walking use-def chains, symbol tables, or nested region relationships. Operation
interfaces describe behavior, not validation phases.

### 선언적 operation schema

Operation verifier의 로컬 계약은 `#[dialect]` operation 정의에서 선언적으로
기술한다. 모든 operation은 이 문법으로 정의하며, 타입 제약이 없는 entity는 `_`로
선언한다. 이 schema는 operation verifier layer를 생성하는 표현이며,
같은 정의에서 검증 코드, builder, 정적 schema descriptor를 만든다. Assembly
format과 선언적 rewrite 도구는 operation 정의를 중복하지 않고 이 descriptor를
소비한다.

정의 문법은 Rust 함수 시그니처와 trait bound 모델을 따른다.

- 파라미터 wrapper가 종류를 정한다. `Value<C>`는 operand 하나,
  `Variadic<C>`는 같은 제약을 만족하는 0개 이상의 operand, `Values<L>`는
  타입 목록 `L`과 개수·순서·타입이 정확히 일치하는 operand 목록이다.
  `Attr<K>`는 attribute이고, `Option<Attr<K>>`는 선택 attribute다. 가변
  operand 구간은 operand 중 마지막 하나만 허용한다.
- 결과는 `-> Value<C>`(accessor `result`) 또는 `-> Variadic<C>` /
  `-> Values<L>`(accessor `results`)로 선언한다. 결과가 0개 또는 1개인
  operation은 `-> Option<Value<C>>`로 선언한다. 결과가 없는 operation과
  `core.nil` 결과 하나를 가진 operation은 서로 다르다.
- Region은 선언 순서대로 저장된다. 외부 함수 선언의 body처럼 없을 수 있는
  region은 선택 region으로 선언하며, 마지막 region만 선택일 수 있다.
- 이름 있는 제네릭 파라미터는 논리적 타입 변수다. 같은 변수가 여러 위치에
  나타나면 정확한 타입 동일성을 요구한다. `impl B`는 위치마다 독립인 익명
  변수이고, `_`는 무제약이다. 타입 변수는 생성된 Rust 코드의 제네릭이나
  monomorphization을 뜻하지 않는다.
- Bound는 교집합이다. Exact bound는 하나의 dialect 타입과 그 wrapper의 내부
  invariant를 요구하며, 한 변수에 서로 다른 exact bound를 둘 수 없다.
  Category bound는 `IntegerLike`, `BoolLike`, `FloatLike`처럼 타입 범주를
  요구하며 여러 개를 함께 둘 수 있다. 이 범주들은 `core` 스칼라 타입만 보는
  닫힌 판별이다. 다른 dialect의 타입까지 포함해야 하는 실제 소비자가 생기기
  전에는 dialect 등록형 범주를 두지 않는다.
- 결과 위치에 `impl` 없이 직접 쓴 bound는 그 bound가 가리키는 하나의 고정
  타입이다(`-> Value<core::I32>`). 하나의 타입을 가리키지 않는 bound를 이렇게
  쓰면 컴파일 시점에 거부한다.
- 파생 타입은 투영으로 참조한다. `S::Type`은 변수 자체를 타입 attribute
  값으로 쓰는 투영이다. Bound는 제공하는 투영의 이름과 종류(단일 타입 또는
  타입 목록)를 선언한다. 예를 들어 signature 타입은 `Inputs`/`Results`
  목록을, 파라미터형 dialect 타입은 선언된 파라미터를 제공한다. 둘 이상의
  bound가 같은 이름을 제공하면 `<S as B>::X`로 명시해야 한다.
- Schema로 표현할 수 없는 로컬 조건은 operation별 사용자 verifier가 맡는다.
  사용자 verifier는 생성된 검사를 통과한 operation에서만 실행된다.

생성된 검증은 다음 순서로 진행하며, 앞 단계가 실패하면 그 조건에 의존하는 뒤
단계를 실행하지 않는다. Accessor는 이 순서를 거치지 않은 malformed operation에서
panic하는 대신 진단을 남길 수 있어야 한다.

1. Operand/result 개수와 필수 attribute 존재.
2. 개별 타입 제약과 typed attribute의 내부 유효성.
3. 타입 변수 바인딩과 동일성.
4. 투영과 타입 목록 관계.
5. 사용자 정의 로컬 verifier.

진단은 operation, 위치, 필드 이름과 index, 기대 제약, 실제 타입을 포함한다.
정의 자체의 오류(미선언 변수, 모호하거나 존재하지 않는 투영, 종류 불일치,
exact bound 충돌)는 컴파일 시점에 거부한다. 다른 crate의 dialect 타입을
참조하는 경우에도 bound가 내보내는 const descriptor로 같은 검사를 수행한다.

선언적 제약은 verifier checkpoint에서만 강제한다. Parser, raw operation
builder, operation clone, 결과 타입 재지정은 제약을 우회할 수 있으며, rewrite
중간 상태가 일시적으로 제약을 위반하는 것도 허용한다. Pass 경계는 checkpoint다.
각 pass가 끝나면 IR은 선언된 제약을 다시 만족해야 한다. 값의 타입만 먼저 바꾸는
부분 변환은 `core.unrealized_conversion_cast`로 use가 선언한 타입을 유지하며,
target 타입 변환이 양쪽 타입을 같게 만든 뒤에 cast를 해소한다. 표현이 같다는
이유만으로 타입이 다른 값을 cast 없이 대입하지 않는다.

Cast 처리는 materialization과 reconciliation으로 나뉜다. Target 타입 변환 전의
materialization은 boxing처럼 실제 operation이 필요한 cast만 물리화하고, 타입만
바꾸는 cast는 남긴다. Target 타입 변환은 op 결과, block 인자, cast 결과를
포함한 모든 값의 타입을 변환한다. Target materialization은 boxing, unboxing,
reference cast처럼 실제 representation을 바꾸는 operation만 만들며, 타입이 다른
값을 그대로 넘기지 않는다. 표현이 같은 두 타입은 target 타입 변환 뒤 같은
타입이 되므로 그 사이의 cast는 동일 타입 cast로 남는다. Reconciliation은 type
converter 없이 동일 타입 cast, 원래 타입으로 돌아오는 cast chain, 사용되지 않는
cast만 제거하므로 어느 지점에서 실행해도 의미를 바꾸지 않는다. Reconciliation 뒤에 남은 cast는 변환 버그이며
target emission 경계의 legality 검사가 거부한다.

생성된 builder는 entity 종류별로 입력을 묶는다. Operand 전체를 선언 순서대로
받는 것으로 시작하고, attribute는 이름별로 받는다. 추론할 수 없는 결과 타입,
region, successor는 각각 한 묶음으로 받는다. 묶음 안의 순서는 선언 순서다.
Builder는 결과 타입이 고정 타입, 단일 operand나 필수 attribute로 바인딩된
변수, 또는 그 변수의 투영으로 유일하게 결정될 때만 결과 타입을 추론한다.
입력 타입을 검사하거나 cast를 삽입하지 않으며, 필수 입력의 누락은 프로그래밍
오류로 취급한다. 심볼 해석, 소유 callable, conversion 경계, ownership처럼 한
operation 밖의 정보가 필요한 조건은 schema가 아니라 기존 whole-IR verifier와
conversion target이 소유한다.

Operation schema는 해당 operation의 모든 유효한 등장에서 성립해야 하는 조건만
담는다. 특정 target의 physical 대입 호환성이나 pipeline 단계에 따라 달라지는
타입 합법성은 schema 제약이 아니다.

Control-flow and SSA forwarding queries use three target-independent operation
interfaces. `Branch` describes raw block successors and the operands forwarded
to each successor block's arguments. `RegionBranch` describes possible transfers
from an operation entry or one of its nested-region terminators to a nested
region or back to the parent operation. `RegionBranchTerminator` describes the
values a terminator forwards along such a region transfer. Region branch points
and successors are explicit values: a point is either `Parent` or a concrete
nested-region terminator, and a successor is either a concrete region or
`Parent`.

`CallableExit` is a separate dialect-registered operation interface for an
operation that ends execution of the current callable and therefore cannot
continue after its enclosing structured region. It is not a generic
terminator marker: `scf.yield`, `scf.break`, and `scf.continue` transfer within
structured control flow and do not satisfy it. Dialects register only their
verified callable exits, including ordinary `func.return`, proper tail
transfers, and unreachable control flow. The typed model and its dynamic query
are fallible. An unregistered operation, malformed registered operation, or
query error is never evidence that a region is terminal.

Structured-to-CFG lowering may use `CallableExit` only after preserving its
own structural rules: the region must have one block, the exit must be that
block's final operation, and a nested structured operation must expose every
possible entry successor through `RegionBranch`. A `Parent` successor, a
missing mapping, or an incomplete query leaves a reachable continuation and
must retain the merge path. The same conservative query is shared with Wasm
structured lowering; it does not imply arbitrary multi-block or loop CFG
termination analysis.

이 보수적 판정은 target-independent `StructuredControlAnalysis`가 소유한다.
분석은 `Analysis` 계약에 따라 IR을 읽기만 하며, 자식에서 부모 순서로 각
region과 SCF operation의 증명 결과를 저장한다. 부모 판정은 저장된 자식 결과를
조회하므로 중첩 subtree를 다시 탐색하지 않는다. 증명되지 않은 terminal 성질은
정상적인 분석 결과이며, interface 오류도 terminal 증거로 사용하지 않는다.
Wasm switch의 지원 타입이나 target conversion 진단은 분석의 책임이 아니다.
중첩 `scf.if`와 `scf.loop`는 결과가 없거나 미사용 `core.never` 결과 하나만
가지고, 모든 진입 successor region이 terminal이면 enclosing region의 종료를
증명한다. Loop 본문 자체가 종료되는 경우만 포함하며 `continue` 순환은 증명하지
않는다.

Native와 Wasm lowering은 phase 범위의 `AnalysisCache`에서 분석을 조회하고,
각 target의 검증과 변환 결정을 도출한다. Mutation 전에 캐시를 무효화하며,
rewrite는 원래 operation에 대한 변환 전 결정만 소비한다. 보존성을 증명하지
않은 다른 pass를 가로질러 이 결정을 재사용하지 않는다.

`IrContext`는 주소와 무관한 고유 identity와 단조 증가하는 전체 IR revision을
가진다. Operation, value/use-list, block, region, type·path interner 및 전역
metadata의 관측 가능한 변경은 첫 쓰기 전에 revision을 올린다. 직접 mutable
reference를 제공하는 경로는 실제 변경 여부를 알 수 없으므로 반환 전에
보수적으로 올린다. 부분 변경 뒤 pass가 실패해도 revision은 되돌리지 않는다.
진단 버퍼는 IR 상태가 아니며 분석의 입력으로 사용하지 않는다.

`AnalysisCache`는 조회마다 context identity와 revision을 대조한다. 어느
하나라도 달라지면 기존 결과와 분석 간 의존성 기록을 모두 폐기한 뒤 현재
context에 결합한다. 따라서 변경이 없는 IR의 반복 조회는 같은 `Arc`를
재사용하지만, 변경된 IR을 읽는 조회는 이전 결과를 반환하지 않는다. 다른
`IrContext`에 같은 캐시를 넘겨도 새 context에서 다시 계산한다. 명시적인
분석 무효화는 revision이 그대로일 때 관련 분석과 그 의존 분석만 제거한다.
Pass의 보존 선언은 revision 검사가 거부한 결과를 재사용하게 할 수 없다.
이미 반환한 `Arc`는 이전 IR에 대한 snapshot으로 남으며, 현재 사실이 필요한
소비자는 캐시를 다시 조회한다. 조회 결과나 실패한 분석 계산은 변경 전후에
부분적으로 캐시되지 않는다.

Native ownership planning의 policy-neutral 입력은 `scf_to_cf` 이후,
`func_to_clif` 이전 경계에서 fallible 분석이 소유한다. Module 범위 분석은
module body, `func.func` 목록, 중복 없는 function 정의, 검증된 managed
nominal layout을 담는다. Function 범위 분석은 그 module 분석에 의존하며,
검증된 flat CFG, type erasure 이전의 managed 값, typed 계약에서 유도한
exact managed alias root, 검증된 managed projection-owner 관계, liveness가
소비하는 policy-neutral block use/definition 입력을 담는다. Alias root는
`core.ptr`이나 물리적 형태에서 provenance를 추론하지 않는다. 분석은 IR을
읽기만 하며, 잘못된 CFG, alias, nominal layout 또는 projection 계약을
fail-closed로 거부하고 실패를 캐시하지 않는다.

이 사실은 `NativeOwnershipPlanOptions`와 무관하게 동일하다. Borrow elision,
entry ownership 같은 정책 선택은 사실을 소비하는 planner가 적용하며 사실의
identity나 계산에 참여하지 않는다. 따라서 phase 범위 캐시는 정책-중립 사실만
재사용하며, planner는 호출마다 그 사실 위에 정책 결정을 새로 적용한다.

Native managed liveness는 정의된 `func.func`를 대상으로 하는 별도의 의존
분석이다. 분석 결과는 같은 function ownership facts를 캐시에서 조회하고,
보수적 managed liveness와 검증된 managed projection borrow의 owner 수명을
연장하는 liveness를 질의 view로 제공한다. 두 view는 각각 block별 `defs`,
`live_in`, `live_out`을 제공하고 기존 역순 고정점 계산을 사용한다. 각 view의
고정점은 최초 질의 때 계산하여 결과 안에 보관하므로 사용하지 않는 view는
계산하지 않는다. 런타임 정책 값은 분석 캐시의 identity에 참여하지 않는다.
선행 facts 조회 실패는 원래 분석 오류를 그대로 전파하며 liveness 결과를
캐시하지 않는다. `elide_proven_field_borrows`만 view를 선택하고
`elide_proven_borrowed_parameters`는 선택에 관여하지 않는다. Action planner는
선택된 결과를 소비한다.

`wasm.if`, `wasm.block`, `wasm.loop`의 typed builder는 명시적인 결과 타입 목록을
받는다. 빈 목록은 SSA 결과가 없는 제어 연산이며 `core.nil` 결과 하나와 다르다.
SCF lowering이 지원하는 기존 단일 값 결과의 개수와 타입 변환은 유지한다.
미사용 결과를 제거하는 rewrite는 새 결과가 비어 있고 기존 결과에 use가 없음을
검사하는 명시적 경로를 사용한다. 일반 operation 교체는 결과를 1:1로 대응시킨다.

These interfaces are queried through TrunkIR's operation registry and remain
object-safe. Their dyn-facing methods return named concrete small collections;
they do not use RPITIT, GATs, caller-visible tuples, or dialect-specific
downcasts. The collections describe transfers from the existing operand,
result, block, region, and successor lists rather than storing a second copy of
those IR facts.

Shared interface verification is fail-closed for every registered mapping. A
block edge must report every raw successor exactly once, remain in the source
region, and forward exactly one type-compatible value per successor block
argument. A region edge must remain within the owning operation's nested region
tree, use a valid nested terminator branch point, and forward exactly one
type-compatible value per successor region entry argument or parent operation
result. A registered region terminator without a complete owning
`RegionBranch` mapping is invalid. Consumers that require a complete control-flow
view must likewise reject operations for which the applicable interface is not
registered; interfaces describe possible transfers and do not materialize a
flat CFG.

`scf.switch` keeps its existing textual container region of `scf.case` and
`scf.default` wrapper operations. That container is declarative rather than an
executable successor. The switch's `RegionBranch` successors are the case and
default body regions derived from those direct wrappers (or `Parent` for the
implicit unmatched path when no default exists). The wrappers also expose
their own parent/body boundary, so generic nested-region walkers never need to
decode the switch's container shape themselves.

### Callable 본문 구조

선택적 본문을 소유하는 callable operation은 region이 없으면 선언이고, region이
하나이며 entry block이 있으면 정의다. Region은 있지만 entry block이 없거나 region이
둘 이상이면 malformed이며, 무참조 여부와 관계없이 거부한다. Entry 이후의 block
수와 내용은 별도 CFG 및 operation 검증이 담당한다.

이 분류는 dialect 등록이나 시그니처, symbol, 바인딩에 의존하지 않는 공통 구조
질의가 담당한다. 각 consumer는 자신이 처리하는 callable을 선택하고, 구조 오류에
dialect와 함수 identity를 붙여 진단한다. Ownership planning과 target emission은
같은 구조 판정을 사용한다.

Bodyless 선언 자체는 malformed가 아니다. 시그니처와 외부 바인딩 검증 및 emission
처분은 각 target이 소유한다. 이 처분은 대상 symbol 삭제 pass나 generic DCE
reachability와 구분되며 IR을 변경하지 않는다. 구체적인 바인딩 규칙은
[native 계약](cranelift-backend.md#bodyless-선언의-바인딩)과
[Wasm 계약](wasm-backend.md#bodyless-선언의-처분)을 따른다.

Symbol 사용 질의는 operation-local이다. Container operation이 직접 symbol 속성을
갖지 않는다는 사실은 자식 region의 사용을 부정하지 않는다. Symbol 사용을 수집하는
consumer는 nested region을 재귀적으로 순회해야 하며, "사용 없음"을 자식 순회 생략의
근거로 삼아서는 안 된다.

## Core Invariants

- A `core.module` owns the top-level region for a compilation unit.
- `core.unrealized_conversion_cast` is temporary conversion glue and must not
  remain at target emission boundaries. 두 타입이 모두 `tribute_control.func_sig`이면
  calling convention을 바꿀 수 없으며, pre-CPS whole-IR 검증에서 이를 거부한다.
  이 cast는 callable adapter를 생성하지 않는다. Named callable의 convention 강화는
  declaration provenance가 있는 `tribute_control.func_ref`로 표현하고 shared
  legalization에서 실제 adapter를 생성해야 한다.
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
`func.*`, `closure.*`, `func.func_sig`는 physical 표현이므로 이 경계에 나타나지
않는다.

논리 callable type은
`tribute_control.func_sig<(Params...) -> Result> {tribute.calling_convention = N}`이다.
It uses flat `[inputs..., results...]` storage with mandatory `num_inputs` and
`num_results` u32 delimiters; its source-owned validated API requires exactly
one result. `Result`와 `Params`는 source-logical type이며 기존 code `Direct = 0`,
`EvidenceDirect = 1`, `Cps = 2`를 그대로 사용한다. 이 metadata는 typechecking
결과를 복사한 것으로 body에서 추론하지 않는다. Legalization은 이를 기존
`CallableAbi`와 physical `func.func_sig`, `closure.closure` 및
`tribute.calling_convention` operation attribute로 바꾼다. Type verifier는
result 하나와 parameter 0개 이상, convention domain, 모든 component type의
resolution을 검사하며 physical `func.func_sig`나 `closure.closure`를 component로
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
keyword로 `tribute_control.func_sig`을 만들고 convention을 type에만 저장한다.
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
선택적 attributes, body다.

간결한 형식은 non-reserved type attribute가 없는 source signature에만 쓴다.
그런 attribute가 있으면 `func`/`lambda`는 generic assembly를 써서 complete
`type` attribute 또는 result type expression을 출력한다. Metadata의 소유자는 항상
그 `tribute_control.func_sig` type/alias 하나이며, 같은 alias를 참조하는 여러
operation이 operation-local override를 만들 수 없다. Alias가 없으면
generic type-bearing 위치에 attributed `func_sig` 전체를 inline으로 출력한다.
Operation attribute는 별개의 generic attribute dictionary에 남는다.
`func_ref`, `call`, `call_indirect`, `return`은 generic assembly가 모든 정보를
손실 없이 표현하므로 custom format을 만들지 않는다.

#### `tribute_control.func`

```text
tribute_control.func {sym_name = @id, type = !Callable} (%x: T) { ... }
```

- **형상:** 피연산자와 결과는 없다. `sym_name: Symbol`과
  `type: tribute_control.func_sig<(Params...) -> Result>`가 필수다. 선언은 region이
  없고 정의는 source parameter만 block argument로 받는 single-block `body`
  하나이며 `tribute_control.return`으로 끝난다. Foreign ABI 같은 비제어
  attribute는 보존한다.
- **의미:** source named function의 logical signature와 typechecking이 선택한
  convention을 정의하며 hidden evidence, environment, `ContinuationFrame<R>`를 포함하지 않는다.
- **검증:** local verifier는 attribute, region, block argument, return type을
  callable type과 맞춘다. Whole-IR verifier는 symbol uniqueness를 검사한다.
- **소유권과 값 흐름:** body는 isolated-from-above다. Source local과 parameter만
  block argument로 들어온다.
- **위치:** source function 또는 extern declaration 전체 span이다.

Named callable의 source definition은 body를 가진 Tribute callable이다.
`extern "intrinsic"`은 compiler-reserved directive이며, canonical qualified
source name이 intrinsic identity가 된다. 지원되는 identity와 완전한 logical signature는
mutation 전에 검증하며, unknown directive나 signature mismatch는 lowering input error다.
Compiler intrinsic identity는 prelude와 사용자 선언을 병합한 AST 전체에서
monomorphization 전에 검증한다. 미사용 generic 선언과 중첩 모듈의 선언도 포함하며, 지원하지 않는 모든
directive를 각 선언의 source span에서 진단한다. 등록된 compiler
intrinsic의 logical callable convention은 항상 `Direct`이다. Generic specialization은 base
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

Frontend는 해석된 canonical declaration과 specialization을 module-local symbol에
대응시킨다. 정의, 직접 호출, 함수 참조와 intrinsic declaration metadata는 같은
대응을 사용한다. Import alias나 prelude의 짧은 이름을 이 심볼의 선언 경로로
사용하지 않는다. 소유 symbol table 안에서 심볼은 유일해야 하며, 이 대응은
source lookup이나 외부 linkage 계약을 변경하지 않는다.

Frontend 경계 verifier는 metadata 전체와 module의 callable graph를 mutation 전에
대조한다. Direct call은 module-local symbol을 유일하게 resolve하고 완전한 signature를
맞춰야 한다. Indirect call은 exact `tribute_control.func_sig` signature와 source-logical
callable producer를 요구한다. Return은 enclosing callable의 logical result와 일치해야
한다. Body가 없는 declaration도 body traversal 없이 같은 검사를 받는다.

#### `tribute_control.lambda`

```text
%f = tribute_control.lambda [%capture0, ...] : !Callable { ... }
```

- **형상:** 피연산자는 typechecking된 capture를 source 순서로 나열한다. 결과는
  `tribute_control.func_sig` 하나이고 필수 attribute는 없다. Single-block body는
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
  필수이며 결과는 `tribute_control.func_sig` 하나다.
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

- **형상:** `tribute_control.func_sig` callee, source argument, callee type의
  source-logical result 하나를 가지며 attribute/region이 없는 non-terminator다.
- **의미:** lambda, `func_ref`, parameter 또는 capture로 얻은 callable을 호출한다.
- **검증:** local verifier는 callee signature와 argument/result type을 맞춘다.
- **소유권과 값 흐름:** callee와 argument는 일반 SSA use이고 environment,
  evidence, `ContinuationFrame<R>`는 없다.
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

`closure.*` represents closure allocation and projection. Convention-proven
`closure.closure` keeps its exact callable type through shared effect lowering
and target-ABI validation. Only after that validation may target lowering choose
the canonical closure storage layout, rewriting every type-bearing surface
coherently and removing transient storage-pack provenance; that provenance is
not semantic type equivalence. Closures lower differently per backend: Wasm
uses function references plus GC structures, while native uses function
pointers plus heap environments.

Storage finalization 이후 erased `tribute_rt.anyref`에서 정확한 canonical closure
storage로 복원하는 `core.unrealized_conversion_cast`는 generic cleanup에서
no-op으로 소거하지 않는다. Compiler가 구성한 canonical storage의 전체 type
identity로 이 경계를 선택하며, 이름이나 비슷한 필드 모양으로 추론하지 않는다.
Target은 정확한 cast 결과 타입으로 transfer를 검증한 뒤 emission 전에 복원을
물리화한다. Wasm은 concrete `ref.cast`를 생성하고 native는 pointer 표현으로
변환한다. 이 규칙은 다른 struct나 nominal reference의 일반 변환 정책을 바꾸지
않으며, semantic callable 검증을 대체하지 않는다.

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
registry-verified compiler intrinsic, whose calls lower to the same
`list.prepend` operation. A source-defined function merely spelled
`List::prepend` remains an ordinary call. The private intrinsic declaration is a
compiler/prelude boundary, not an additional public symbol or a layout contract.

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

Source-logical frontend는 nominal layout을 생성할 때 실제 `adt.struct` 또는
`adt.enum` 타입을 normalized nominal symbol의 IR type alias로 게시한다.
`adt.typeref`의 `name`, layout의 `name`, 게시된 alias의 이름은 일치해야 한다.
생성 연산 없이 signature에만 등장하는 특수화도 같은 계약을 따르며, 필드에서
참조하는 compiler-generated nominal tuple layout도 게시한다. Frontend 내부
type map이나 type interner에만 존재하는 layout은 게시된 정의가 아니다.

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
than scanning types for a plausible replacement.

## Mid-Level Dialects

`func.*` represents function definitions, direct calls, indirect calls, function
references, returns, tail calls, and unreachable control flow.
`func.call_indirect`와 `func.tail_call_indirect`는 필수 `signature: TypeRef`
attribute로 exact `func.func_sig` contract를 가지며, callee가 erased pointer나 table
index로 바뀐 뒤에도 이 contract를 보존한다. 인자 목록과 `func.call_indirect`의 결과
목록은 signature의 입력·결과 목록과 정확히 일치해야 한다. Callee 타입이
`func.func_sig`이거나 `closure.closure`에 담긴 `func.func_sig`이면 signature와 같아야
한다. `tribute.calling_convention`이 붙은 indirect transfer는 verifier가 convention과도
대조한다. Physical operand type, symbol, ABI string 또는 storage shape로 signature를
재구성하지 않는다.
Backend indirect-call operation도 runtime-queried `IndirectCallLike` operation
interface를 통해 같은 contract를 보존한다. 각 dialect는 attribute spelling과
accessor를 소유한다 (`func`와 `wasm`은 필수 `signature`,
`clif`는 필수 `sig`).
Generic consumer는 다른 dialect의 attribute key 대신 interface를 query한다.
Interface를 구현하지 않는 operation에는 exact signature가 없으며, erased callee나
result operand로 이를 추론해서는 안 된다.
`func.tail_call_indirect`는 callable operand와 argument를 받고 result가 없는
terminator다. Shared verifier checks complete caller/callee result-list agreement;
logical CPS callables both retain `[core.never]` despite the resultless transfer
operation. 대상 ABI 변환은 검증된 CPS callable의 결과 목록을 원자적으로
`[]`로 바꾸며 실제 Unit `[core.nil]`은 보존한다.

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
conversion casts have been converted and reconciled, because materialization
may introduce additional typed GC operations.

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
| AST-to-IR | `tribute_control.func_sig`과 callable/control operation, valid SSA use chain |
| Shared CPS legalization | 전체 callable graph가 physical `CallableAbi`와 direct/indirect tail transfer를 사용하며 `tribute_control` operation/type이 남지 않음 |
| Shared lowering | 명시된 경계에서 high-level ability dispatch operation이 제거됨 |
| Effect ABI | `effect.*` operations preserve dispatch semantics without backend layout details |
| Native ownership | Typed managed value의 native raw handoff는 `tribute_rt.into_raw`가 one ownership unit을 소비할 때만 허용되며 `core.ptr`는 항상 unmanaged |
| Backend lowering | Backend-ready 검증이 성공하고 `effect.*`, `tribute_rt.into_raw`, CPS control carrier, trampoline이 남지 않음 |

## Type Model

Primitive scalar, pointer, reference, bytes, array, tuple, function, and nil
types are represented in TrunkIR. Library data types such as `Option`, `Result`,
and `Text` lower through ADT and runtime/library conventions. `List` is an
opaque nominal builtin whose shared construction and sequence-view observations
use `list.*`; target-specific passes choose and eliminate its private
representation.

### `func.func_sig` function type

`func.func_sig` stores inputs and results as separate logical lists. It keeps a
flat interned parameter vector, and its canonical textual form follows MLIR
FunctionType syntax:

```text
func.func_sig<() -> ()>
func.func_sig<(core.i32) -> core.i64>
func.func_sig<(Evidence, Frame, core.i32) -> core.never>
```

The input list is always parenthesized. An empty result list is printed as
`()`, while the single supported result is printed without parentheses.
TrunkIR recognizes a parenthesized multi-result list so it can diagnose it, but
currently rejects more than one result.

The interned representation is:

```text
TypeData {
  dialect: func,
  name: func_sig,
  params: [input_0, ..., input_n, result_0?],
  attrs: {
    num_inputs: n,
    num_results: 0 | 1,
    ...other_type_attributes
  }
}
```

Both counts are mandatory `u32` delimiters and participate in type identity.
They do not identify an ABI. Their sum must equal `params.len()`, and
`num_results` must not exceed one. The generic recursive type walk continues to
visit the entire flat `params` vector. The canonical printer hides only these
two reserved attributes and preserves every other type attribute; textual type
attribute dictionaries may not specify either reserved key.

함수 타입의 textual spelling은 `func.func_sig<(inputs...) -> result>`만 사용한다.
Reader는 `core.func(Return, Params...)`를 거부한다. Production code는 검증된
`func::func_sig` API로 함수 타입을 만들며, 직접 `TypeData`를 만드는 코드는
malformed-type verifier test에 한정한다.

The three result states have distinct meanings:

```text
logical Unit function: results = [core.nil]
logical CPS function:  results = [core.never]
physical CPS function: results = []
```

The physical proper-tail ABI is proven only by the combination of
`tribute.calling_convention = Cps` and exact `results = []`. The stored counts
are not an ABI marker, and general resultless IR need not be CPS.

Shared `func.call` and `func.call_indirect` support zero or one SSA result.
Calls match complete input/result lists; returns match the enclosing function's
result list. Proper-tail operations themselves remain resultless, including
when transferring to a logical `[core.never]` callable, and require matching
caller/callee result lists. Indirect calls require an exact signature, which
must agree with a typed callee.

Custom `func.func` assembly omits the arrow for zero results and prints `-> T`
for one result. Absent arrow means zero IR results, never implicit Unit.
Generic operation syntax preserves its explicit `type` attribute; custom
assembly retains an explicit type when needed to preserve type attributes and
validates its agreement with the decomposed signature. Shared conversion maps
every input/result and nested type attribute, preserving cardinality, and
checks entry argument arity before changing the signature or entry arguments.
Normal `validate_all` combines local operation checks with contextual shared
function-contract checks. Local checks enforce zero/one ordinary-call results,
resultless terminators, and exact typed indirect operand/result agreement.
Contextual checks verify returns, caller/callee tail result lists, and direct
calls whose declarations are available. An undeclared runtime callee has no
inferred signature; it does not exempt other known contracts from validation.
Duplicate, malformed, or non-callable declarations are errors.

Callable region ownership comes from a dialect-registered operation interface,
not unqualified operation names. `func.func` exposes its explicit function type;
`closure.lambda` exposes the function type inside its result closure type. The
nearest registered owner is authoritative, including when its signature is
malformed. Shared validation does not infer a lambda signature from an attribute
or an outer function, and does not select a physical CPS ABI.

### `wasm.func_sig` Wasm 호출 계약

`wasm.func_sig`는 Wasm이 소유하는 함수 호출 계약이다. 공통 및 소스 시그니처와
마찬가지로 입력 우선의 평탄한 `[inputs..., results...]` 벡터와 필수 `u32` 속성
`num_inputs`·`num_results`를 사용하며, 결과는 0개 이상을 허용한다. 두 개수의
합은 벡터 길이와 같아야 하고, 두 속성은 타입 동일성에 참여한다. 이 속성은 저장
경계이지 ABI나 호출 규약의 증거가 아니다. 예약되지 않은 타입 속성도 타입
동일성에 포함되며 파싱·출력·별칭·재귀 변환에서 보존된다.

Wasm 함수와 가져오기 선언, 직접·간접 호출, 반환, 타입 섹션 수집, 검증 및 코드
생성은 이 타입을 사용한다. 모든 간접 호출은 exact `wasm.func_sig`를
명시해야 하며, 누락된 시그니처는 lowering과 emission에서 오류다. Operand, result,
상위 함수의 return, `type_idx` 또는 타입 정보가 지워진 테이블 인덱스에서
시그니처를 재구성하지 않는다. 빈 결과 목록만으로
CPS를 판정하지 않는다. 논리적 Unit 함수, 논리적 CPS 함수, 물리적 CPS 함수의
결과 구분은 [공통 `func.func_sig` 계약](#funcfunc_sig-function-type)을 따른다.

### `clif.func_sig` 네이티브 호출 계약

`clif.func_sig`는 네이티브가 소유하는 함수 호출 계약이다. 입력 우선의 평탄한
`[inputs..., results...]` 벡터와 필수 `u32` 속성 `num_inputs`·`num_results`를
사용하고 결과는 0개 이상을 허용한다. 두 개수의 합은 벡터 길이와 같아야 하며,
두 속성과 예약되지 않은 타입 속성은 모두 타입 동일성에 참여한다. count는 저장
경계일 뿐 ABI 또는 calling convention의 증거가 아니다. parser, printer, alias는
예약되지 않은 속성을 보존하며, 호출 계약 내부의 재귀 타입 변환도 중첩 메타데이터를
그 소유 타입에 보존한다.

네이티브 함수와 선언, 직접·간접 호출, 반환, proper tail transfer 및 Cranelift 코드
생성은 이 타입을 사용한다. 간접 호출의 `sig`는 정확한 `clif.func_sig`이어야 하며,
타입이 지워진 함수 포인터, 심볼, ABI 문자열 또는 저장 형태에서 재구성하지 않는다.
결과가 비었다는 사실만으로 CPS를 판정하지 않는다. 논리 Unit `[core.nil]`, 논리
CPS `[core.never]`, 물리 CPS `[]`의 구분은
[공통 `func.func_sig` 계약](#funcfunc_sig-function-type)을 따른다.

네이티브 최종 호출 계약의 각 operand와 result slot은 `clif.func_sig`의 같은 순서
slot과 정확히 같은 TrunkIR type이어야 한다. semantic reference SSA 값은 native
lowering이 그 producer 또는 block argument를 `core.ptr`로 명시적으로 낮춘 뒤에만
`core.ptr` slot을 채울 수 있다. 검증과 emission은 dialect 이름, type attribute, ABI
문자열, symbol, erased representation에서 pointer 동치를 추론하지 않는다. 이 규칙은
`core.nil`의 정해진 zero-width projection과 별개이며, 다른 contract type 사이의
호환성 규칙을 만들지 않는다.
