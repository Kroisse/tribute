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
  `Attr<K>`는 attribute이고, `Option<Attr<K>>`는 선택 attribute다.
  가변 operand 구간은 operand 중 마지막 하나만 허용한다.
- Attribute 종류 `K`는 bound와 마찬가지로 Rust 타입이다. 종류는 schema가
  검사하는 값 영역, accessor가 돌려주는 값, builder가 받는 값을 스스로
  정의하며, 정의 문법은 종류의 이름을 해석하지 않는다. 새 종류는 그 타입을
  정의하는 것으로 추가된다. `[K]`는 모든 원소가 종류 `K`인 목록 종류이고,
  원소 종류는 이름 있는 종류여야 한다. `Dict<V>`는 모든 값이 종류 `V`인
  dictionary 종류다. `_`는 모든 attribute 값을 받는다.
  문자열 종류만 예외로, 정의 문법이 알아보고 pool handle을 돌려주는 accessor를
  하나 더 만든다.
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
타입이 되므로 그 사이의 cast는 동일 타입 cast로 남는다.

공용 materialization과 type converter는 subtype 관계를 다루지 않는다. WasmGC처럼
서브타입 참조를 상위 타입 자리에 그대로 받는 target은 target 변환 단계에서
backend의 물리적 할당 가능성 규칙에 따라 서브타입에서 상위 타입으로 가는 cast를
지우고 source 값을 그대로 쓴다(upcast elision). 표현이 같다는 사실만으로는
subtype이 아니다.

Reconciliation은 type converter 없이 동일 타입 cast, 원래 타입으로 돌아오는 cast
chain, 사용되지 않는 cast만 제거하므로 어느 지점에서 실행해도 의미를 바꾸지
않는다. Reconciliation 뒤에 남은 cast는 변환 버그이며 target emission 경계의
legality 검사가 거부한다.

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

`AnalysisCache`는 `IrContext`와 별개의 값으로 남으며, pass와 분석 소비자는
캐시를 명시적인 인자로 받는다. `IrContext`를 캐시와 함께 소유하는 별도
session이나 editor 계층은 두지 않는다. `IrContext`의 필드는 비공개이고 IR
변경은 그 메서드로만 일어나므로 `IrContext` 자체가 변경을 관찰하는 editor
역할을 한다. 캐시를 `IrContext` 안에 두지 않는 이유는 두 가지이다. 분석
계산은 `&IrContext`를 읽는 동안 캐시에 써야 하고, 공유 읽기만 받는 검증기가
캐시를 조회하려면 내부 가변성이 필요해져 `IrContext`의 공유 읽기 안전성과
충돌한다.

컴파일 pipeline의 각 phase는 캐시 하나를 소유하며, 그 phase의 pass manager,
pass, pass 검증기와 pass 바깥의 분석 소비자가 이를 공유한다. 캐시 인자를 받지
않는 단독 진입점은 자체 캐시를 새로 만들며, phase 안의 호출은 캐시를 받는
진입점을 사용한다. Pass manager의 검증기는
입력을 한 번 검사한 뒤 revision을 바꾼 pass 다음에만 실행한다. IR을 바꾸지
않은 pass는 불변 조건을 깨뜨릴 수 없기 때문이다. 검증기는 IR을 바꾸지 않으므로
검증 중 계산된 분석은 이후 pass를 위해 캐시에 남는다.

변경 중에 분석 결과를 다루는 소비자는 다음 세 방식 중 하나를 따른다.

- **보유 후 재구축**: 결과의 `Arc`를 쥔 채 rewrite하고, IR을 바꾼 반복마다
  분석을 무효화한 뒤 다음 반복에서 다시 조회한다. 보유한 결과가 이전 IR의
  snapshot이라는 사실은 그 결과를 읽는 rewrite가 감수하며 문서화한다.
- **결정 추출 후 무효화**: 변경 전에 원래 operation에 대한 결정을 모두
  추출하고 분석을 무효화한 뒤, rewrite는 추출한 결정만 소비한다.
- **보존한 계획의 재검증**: 읽기 전용 단계에서 만든 계획을 보존하되, 변경
  단계는 사용 전에 계획이 가리키는 operation identity와 layout을 현재 IR에
  대조한다.

분석은 기본적으로 context 전체에 의존한다. `AnalysisContext::ir()`는 임의의
읽기를 허용하므로, 분석 대상 operation이나 그 operation의
`IsolatedFromAbove` 성질만으로는 읽기 범위가 그 subtree에 국한된다는 것이
증명되지 않는다. 다른 함수의 변경 뒤에도 한 함수의 결과를 재사용하는 범위
재사용은 다음 조건을 모두 만족할 때만 허용한다.

- 분석이 대상 subtree와 선언한 선행 분석만 읽는다고 명시적으로 선언한다.
  선언하지 않은 분석은 context 전체 의존으로 남는다.
- `IrContext`가 변경 전에 영향받는 `IsolatedFromAbove` 조상을 기록한다. 이동은
  이전 부모와 새 부모 모두에, operand 변경과 RAUW는 관련 정의·사용자·use-list
  모두에 영향을 준다. Detached 사용자를 만드는 것도 값의 use를 바꾼다. 범위를
  알 수 없는 변경, 직접 mutable reference 경로와 전역 metadata 변경은 context
  전체 무효화로 되돌아간다.
- 선행 분석이 무효화되면 그에 의존하는 분석도 무효화된다. Module 집계 분석은
  포함한 함수 어느 하나가 바뀌어도 무효화되며, 다시 계산한 결과가 이전과
  같으면 그 의존 분석을 유지할 수 있다(early cutoff). 의존 분석의 캐시 hit은
  반환 전에 선행 분석의 신선도를 확인한다.

범위 재사용을 쓰는 소비자가 생기기 전까지 위 조건은 계약으로만 존재하며,
모든 분석은 context 전체 revision 검사를 따른다. Pass의 보존 선언은 범위 검사가
거부한 결과도 재사용하게 할 수 없다.

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
- `core.nil`의 유일한 값은 속성 없는 `core.nil_value`가 만든다. `arith.const`는
  nil을 만들지 않는다.
- Operation과 type의 이름은 `Symbol`이다. Symbol table 정의의 qualified path는
  component마다 `Symbol` 하나인 `SymbolPath`로 저장한다.
- `Symbol`은 8바이트 atom이며 세 형태 중 하나다. 7바이트 이하의 이름은 값 안에
  직접 담는다. IR 기반 계층이 선언한 dialect, operation, type, 속성 이름은 빌드
  때 만든 static 집합의 색인이다. 그 밖의 이름은 프로세스 전역 동적 집합의
  참조 카운트 항목이며, 마지막 `Symbol`이 사라지면 집합에서 제거된다. Static
  집합에 없는 이름도 유효한 `Symbol`이다.
- `Symbol`의 동등성은 값 비교이고 hash는 atom에 미리 계산된 값을 쓴다. 순서는
  텍스트 순서이므로 `Symbol`을 key로 정렬한 자료구조는 이름 순서로 순회한다.
  Hash 순회 순서는 출력에 영향을 주어서는 안 된다. `Symbol`은 `Copy`가 아니다.
  이름을 읽기만 하는 곳은 `&Symbol`을 받는다.
- 함수 symbol 참조(`callee`, `func_ref`, target dialect의 직접 호출과 주소 참조)는
  항상 root module 기준 qualified path이다. 정의는 자기 module 안의 `sym_name`을
  가지며, 정의의 qualified path는 root module을 제외한 중첩 `core.module` 이름들과
  `sym_name`을 차례로 component로 둔 것이다. 이름 없는 `core.module`은 경로에
  기여하지 않는다. 참조는 참조하는 operation이 속한 module을 기준으로 해석하거나
  바깥 module로 찾아 올라가지 않는다. Qualified path는 module tree 전체에서 유일해야
  하며, 중복 정의는 모호한 참조가 아니라 IR 오류다.
- `SymbolPath`의 component는 이름 하나를 쓰인 그대로 담는다. 이름이 `::`를
  포함해도 component 하나이며, 같은 텍스트를 중첩 module로 풀어 쓴 path와는 다른
  path다. Textual form은 component마다 `@`를 붙여 `::`로 잇는다
  (`@left::@helper`). Identifier가 아닌 component는 따옴표로 감싸므로, `::`를 포함한
  이름 하나는 `@"left::helper"`로 쓴다. Path의 순서는 component를 차례로 비교한
  순서다.
- 함수 참조를 모으는 분석(call graph, global DCE)은 참조하는 operation이나 속성의
  이름을 나열하지 않고 operation 속성의 모든 symbol 참조를 모은다. 직접 호출
  operation은 `CallLike` interface로 callee를 게시하며, 그 callee 참조만 호출이다.
  그 밖의 참조는 모두 주소 참조이고, 대상 함수는 값으로 escape한 것으로 본다. 함수
  정의 밖에 있는 참조는 reachability root다. Symbol 참조는 operation 속성에만 두며,
  type 속성과 block 인자 속성에는 두지 않는다.
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
`tribute_control.func_sig<(Params...) -> Result, {tribute.calling_convention = N}>`이다.
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
    attributes {visibility = "private"} {
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
tribute_control.func {sym_name = "id", type = !Callable} (%x: T) { ... }
```

- **형상:** 피연산자와 결과는 없다. `sym_name: String`과
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
사용하지 않는다. Qualified symbol은 module tree 전체에서 유일해야 하며, 이 대응은
source lookup이나 외부 linkage 계약을 변경하지 않는다.

Frontend 경계 verifier는 metadata 전체와 module의 callable graph를 mutation 전에
대조한다. Direct call은 qualified symbol을 유일하게 resolve하고 완전한 signature를
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
  `callee: Symbol`, 선택적 [`evidence_plan`](#evidence-선택-속성)을 가지며 region이
  없는 non-terminator다.
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
  source-logical result 하나와 선택적 [`evidence_plan`](#evidence-선택-속성)을 가지며
  region이 없는 non-terminator다.
- **의미:** lambda, `func_ref`, parameter 또는 capture로 얻은 callable을 호출한다.
- **검증:** local verifier는 callee signature와 argument/result type을 맞춘다.
- **소유권과 값 흐름:** callee와 argument는 일반 SSA use이고 environment,
  evidence, `ContinuationFrame<R>`는 없다.
- **위치:** callee와 argument를 포함한 source indirect-call span이다.

#### Evidence 선택 속성

`evidence_plan`은 호출이 callee에게 넘길 evidence를 caller evidence에서 고르는
선택이며 [type-inference.md](type-inference.md#호출의-evidence-선택)가 정한 값을
typechecking 결과로 복사한다. `call`, `call_indirect`, `resume`, `handle`이 가질
수 있다. `handle`의 선택은 body evidence를 만들기 전에 적용하며 `mask`만 담는다.

```text
{evidence_plan = [{mask = core.ability_ref<{name = "State", ...}>}, {dup = ...}]}
```

- 원소는 key 하나짜리 dictionary다. Key `mask` 또는 `dup`이 연산을, 값이 exact
  ability instance(`core.ability_ref` type)를 나타내며 원소는 순서대로 적용한다.
  한 instance는 한 번만 나온다.
- Key `outer`는 그 instance의 가장 위 handler를 설치한 지점의 evidence로 바꾼다.
  Source-logical operation은 이 key를 쓰지 않는다. CPS legalization이 arm 본문에
  중첩된 handle body 안의 `resume`에만 붙이며, 중첩 한 겹에 한 원소다. 같은
  instance가 여러 번 나올 수 있다.
- 속성이 없으면 그대로 전달한다. 빈 목록은 쓰지 않는다.
- Verifier는 원소 형상, `handle`의 `mask` 전용 규칙, instance 중복만 검사한다.
  Row 정보는 IR에 없으므로 선택의 옳고 그름은 typechecking이 책임진다.
- CPS legalization은 선택을 바꾸지 않고 옮긴다. 호출과 `resume`의 선택은 그것이
  만든 evidence-taking `func.call`, `func.tail_call`, `func.call_indirect`,
  `func.tail_call_indirect`에, `handle`의 선택은 `ability.handle_dispatch`에 같은
  이름의 속성으로 둔다. Evidence를 받지 않는 callee로 가는 호출에는 옮기지 않는다.
  `resolve_evidence`가 이를 `effect.mask`/`effect.dup`/`effect.outer`로 만든다
  ([cps-effects.md](cps-effects.md#row-directed-evidence)).

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
  op_name = "get",
  operation_kind = "op"
} : ResultType
```

- **피연산자:** declaration 순서로 놓인, 이미 평가된 source argument 0개 이상이다.
  값은 logical type을 유지하며 tuple packing과 erasure는 conversion이 담당한다.
- **결과:** logical operation result 하나만 만든다. Source `Never` result는
  `core.never`이며 physical `Never` control carrier를 선택하지 않는다.
- **속성:** `ability_ref: Type`, `op_name: String`,
  `operation_kind: String`이 필수다. `operation_kind`는 정확히 `fn` 또는 `op`이며
  typecheck된 operation declaration에서 복사한다. 이는 body나 use site에서
  추론하는 lowering hint가 아니라 source-semantic metadata다. 모든 source
  ability invocation이 이 operation을 사용하며 shared conversion은 kind를
  재분류하지 않는다.
- **영역과 block argument:** 없다.
- **종결자:** 아니다. 직접형 IR에서 이 operation은 terminator가 아니다.
- **의미:** 선언된 kind로 source operation을 호출한다.
  `operation_kind = "fn"`이면 선택된 handler result가 직접형 평가를 자동으로
  resume한다. Shared conversion은 continuation을 capture하지 않고 기존 tail
  dispatch 경로를 사용한다. `operation_kind = "op"`이면 일치하는 handler가
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
  CPS conversion이 소유한다. `"fn"`에는 continuation 없는 `ability.call`/tail
  dispatch를, `"op"`에는 suffix continuation을 받는 `ability.perform`/CPS
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
- **속성:** 필수 attribute는 없다. 선택적 [`evidence_plan`](#evidence-선택-속성)은
  body evidence를 만들기 전에 바깥 evidence에서 가릴 handler를 정한다. Dynamic
  prompt/owner tag와 physical tail-call 표현은 이 operation의 의미 attribute가
  아니라 legalization 계약이다.
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
  op_name = "get",
  kind = "op",
  operation_result_type = ResultType
} (%arg0: Arg0Type, ..., %resume:
    tribute_control.resume_token<ResultType, AnswerType>) {
  ...
  tribute_control.yield %answer
}
```

- **피연산자와 결과:** 없다. Surrounding `tribute_control.handle`이 소유하는
  declarative entry다.
- **속성:** `ability_ref: Type`, `op_name: String`, `kind: String`,
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
- **속성과 영역:** 선택적 [`evidence_plan`](#evidence-선택-속성)만 가지며 region은 없다.
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

CPS 변환 뒤 Cps callable은 `Evidence, ContinuationFrame<R>, source args`를 받고
`core.never`로 끝난다. 생성된 completion과 exact resume도 Evidence와
ContinuationFrame을 명시적으로 받는다. Resume은 동적 ContinuationFrame에서 handle
층의 dispatcher를 불변 값으로 다시 만든 뒤 suffix와 nested handle을 계속 실행한다. 세 dispatch 계층의 정확한
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
coherently. Until then a lowered storage pack keeps its exact closure type
through an unrealized cast, which becomes an identity and is removed when the
storage layout is selected; this is not semantic type equivalence. Closures
lower differently per backend: Wasm uses function references plus GC
structures, while native uses function pointers plus heap environments.

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
`adt.typeref`의 `name`, layout의 `name`(둘 다 문자열), 게시된 alias의 이름은
텍스트가 일치해야 한다.
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
`arith.const`는 정수와 부동소수 값만 만든다. 정수 결과는 `Int` 속성을 받고,
`core.i1`은 `Bool` 속성도 받는다. 부동소수 결과는 `FloatBits` 속성을 받는다.
정수 타입 `core.i{N}`은 부호가 없다(signless). 부호는 타입이 아니라 각 operation이
정한다. 부호에 따라 결과가 달라지는 operation은 모두 부호 있는 쪽과 없는 쪽을 명시한
쌍으로 둔다: `divsi`/`divui`, `remsi`/`remui`, `shr`/`shru`, `cmpi` predicate,
그리고 아래 변환이다. 부호와 무관한 operation(`addi`, `subi`, `muli`, 비트 연산)은
하나만 둔다.

| 변환 | Operation | 조건 |
| --- | --- | --- |
| 정수 넓히기 | `extsi` / `extui` | 결과 폭이 입력보다 넓다 |
| 정수 좁히기 | `trunci` | 결과 폭이 입력보다 좁다 |
| 정수 → 부동소수 | `sitofp` / `uitofp` | |
| 부동소수 → 정수 | `fptosi` / `fptoui` | 범위 밖 값과 NaN은 trap한다 |
| 부동소수 폭 | `extf` / `truncf` | 넓히기 / 좁히기 |

`arith`는 포인터를 다루지 않는다. 정수는 포인터가 되지 않으며, 포인터와 정수 사이의
변환 operation은 없다. 포인터는 할당, runtime, `mem.data`에서 나오고 `mem.ptr_add`로만
파생되므로 모든 포인터의 출처를 추적할 수 있다. `arith.const`는 `core.ptr` 값을 만들지
않는다.

`mem.*` represents low-level data, load, and store operations for runtime or FFI
support. 주소는 모두 `core.ptr`다. 유일한 포인터 상수는 `mem.null`이 만드는 null이다.
Null은 출처가 없고 역참조할 수 없으므로 출처 추적의 예외가 아니다. `mem.data`는
`core.ptr`를 만들고, `mem.load`와
`mem.store`는 `core.ptr` 주소에서 machine scalar(정수, 부동소수, `core.ptr`) 하나를
읽고 쓴다. Managed reference나 aggregate는 scalar가 아니다. Managed reference의
payload를 제자리에서 읽는 lowering은 그 reference를 `core.ptr`로 보는 cast를 명시하고,
runtime이 넘겨준 pointer를 managed reference로 받는 lowering은 `core.ptr`로 읽은 뒤
cast를 명시한다.
주소 계산은 `mem.ptr_add(base: core.ptr, offset: 정수) -> core.ptr` 하나로 표현한다.
`offset`은 byte 단위다. Schema는 정수 범주만 강제하며, `offset`을 target의 pointer
폭에 맞추는 것은 생산자의 책임이다. 원소 크기 배율은 적용하지 않으므로, 필요하면 `arith`
곱셈으로 명시한다. 결과는 `base`의 provenance를 유지한다. `mem.load`와 `mem.store`는
즉시값 `offset`만 받고 동적 offset을 받지 않는다. 동적 주소는 `mem.ptr_add`로
만든다. 구조적 index로 주소를 유도하는 GEP식 연산은 두지 않는다.

## Low-Level Dialects

`wasm.*` is the Wasm backend dialect. It models Wasm control flow, calls,
numeric operations, memory operations, and WasmGC constructs.

`wasm_gc.*` is a typed intermediate dialect for WasmGC lowering. Its operations
carry heap types as mandatory `TypeRef` attributes and must not infer a type
from an erased operand such as `anyref`. A user struct or variant is the
structural `wasm_gc.struct<T...>`, whose parameters are its physical field
types; equal field lists are one GC type. A module-wide type
layout pass assigns binary type-section indices and fully converts these
operations to `wasm.*` operations, whose required integer attributes correspond
to WebAssembly instruction immediates. This pass runs once, after unrealized
conversion casts have been converted and reconciled, because materialization
may introduce additional typed GC operations.

```text
wasm_gc.struct_get { type = wasm_gc.struct<core.i32, !Bytes>, field_idx = 1 }
  -- module GC type layout -->
wasm.struct_get { type_idx = 9, field_idx = 1 }
```

Builtin layouts follow the same rule. Lowering refers to canonical semantic
types such as `core.bytes` or the marker/evidence ADT types; the layout pass
maps those identities to reserved indices. The Wasm emitter accepts no residual
`wasm_gc.*` operations and does not infer missing indices. Function-signature
indices used by `call_indirect` are a separate concern and are not GC heap-type
identities.

`clif.*` is the native backend dialect. It models Cranelift-style functions,
read-only data objects, calls, arithmetic, CFG control flow, memory access,
stack slots, symbol addresses, and numeric conversions.

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
| Native RTTI | Ownership planning이 선언한 `tribute_rtti.layout`이 RC header lowering까지 모든 할당 layout을 정확히 한 번씩 선언하며, RC header lowering 뒤에는 남지 않음 |
| Representation/ABI 경계 | 아래 [경계 계약](#representationabi-경계)의 출구 적법성을 만족함 |
| Backend lowering | Backend-ready 검증이 성공하고 `effect.*`, `tribute_rt.into_raw`, CPS control carrier, trampoline이 남지 않음 |

<!-- markdownlint-disable-next-line MD033 -->
<a id="representationabi-경계"></a>

### Representation/ABI 경계

Representation/ABI 경계는 상위 제어 의미를 물리적 호출·저장·runtime 계약으로
소비하는 target별 단계다. 이 단계는 shared middle-end가 끝난 뒤 시작하고 target
dialect lowering(`func_to_clif`, `func_to_wasm` 등) 앞에서 끝난다. 경계 이후의 모든
pass — native ownership/RTTI 계획, target dialect lowering, backend 검증기, emitter —
는 출구가 명시한 물리 계약만 소비한다.

#### 단계 구성

경계는 target별로 여러 pass로 이루어질 수 있으며 새 dialect, crate 또는 모든 target에
같은 layout을 요구하지 않는다. 경계 안에서 다음을 순서대로 완결한다.

1. Exact callable/dispatch/frame 계약 검증과 CPS signature 물리화
2. Root/export 및 target 진입점 bridge 합성
3. Closure operation lowering
4. Target evidence lowering: `effect.*` dispatch, evidence 조회·확장, fresh prompt,
   one-shot 검사를 runtime 호출과 ordinary/proper-tail transfer로 바꾼다. 이
   lowering은 공유 value/control dialect 수준에서 수행하고 target dialect operation을
   직접 만들지 않는다.
5. Closure storage layout 확정
6. 경계 출구 검증

지원되는 FFI/intrinsic bridge의 의미도 이 단계 안에서 완결한다.

#### 출구에서 확정되는 물리 계약

- **Exact signature:** 모든 정의·선언·직접 호출·간접 호출은 exact `func.func_sig`를
  가진다. Environment, evidence, continuation 인자는 이미 signature의 순서 있는 입력이며
  별도 slot 속성으로 위치를 기록하지 않는다. 타입이 있는 `func.constant`의 결과는
  참조 대상 함수의 signature와 정확히 같다. Closure의 호출자에게 보이는 signature처럼
  environment를 뺀 타입은 경계 안에서만 쓰며, 함수 참조의 타입으로 경계를 넘지 않는다.
- **기계 호출 규약:** 기계 호출 규약은 physical `func.func_sig`의 `call_conv` type
  속성이 소유한다. 경계는 CPS signature를 물리화할 때 target과 무관하게 그 signature에
  `call_conv = "tail"`을 일괄 부여하며, 속성이 없으면 platform 규약이다. `call_conv`는
  type identity에 참여하고 함수 정의, 직접 호출의 피호출자, 간접 호출 signature가 모두
  같은 signature에서 읽는다. 기계 규약을 구별하는 target은 이 값을 target signature로
  옮기고, 기계 규약이 없는 target은 이를 무시하고 target signature에서 버린다.
- **제어 이전의 세 가지 구별:** 빈 결과 목록은 machine stack에 결과가 없다는 뜻일
  뿐이다. Proper tail transfer는 `func.tail_call`/`func.tail_call_indirect` operation
  자체로 표현한다. 기계 수준 noreturn은 `func.unreachable` 같은 명시적 terminator로
  표현한다. 어느 하나에서 다른 것을 추론하지 않는다.
- **매개변수 ownership:** RC target에서 managed 매개변수의 entry mode
  (`borrowed`/`retained`/`consumed`,
  [rc.md](rc.md#proper-tail-ownership-transfer)) 중 callable 계약이 요구하는
  것은 exact physical signature의 일부로 표현한다. 그래서 직접 정의와
  exact indirect signature 모두에서 같은 계약을 읽을 수 있다. 인코딩은 signature의
  [타입 매개변수 속성](#타입-매개변수-속성) `tribute.ownership = "consumed"`이며,
  경계가 물리 CPS callable의 모든 입력에 붙인다. 표시가 없는 managed 매개변수는
  retained 계약을 가진다.
- **Closure/frame 저장:** Compiler가 소유하는 runtime layout은 명시적
  [runtime layout 식별자](#runtime-layout-식별자)로 구별한다. Struct 이름, field
  모양, `arrayref` 같은 erased heap 형상을 provenance로 쓰지 않는다.
- **진입점:** 출구의 root `main`은 bridge 합성이 만든 wrapper다. Hidden 매개변수가
  없고 결과는 `Nil`이며 platform 규약을 따른다. Body는 root worker 호출 하나로 끝나고,
  모듈 안에서 이 `main`을 참조하는 곳은 없다. 원래 source `main`은 모든 calling
  convention에서 root worker가 되고, 모듈 안의 참조도 worker로 옮겨 간다. 초기
  evidence와 CPS root frame처럼 source calling convention에 따라 달라지는 부분은
  bridge 합성이 소비한다. Target은 이 wrapper를 그 자리에서 target 진입점으로 바꾼다.
  이때 runtime 초기화, 종료 코드, sanitizer 초기화처럼 platform 고유 작업만 더하고,
  별도 진입 함수를 만들거나 `main`의 이름을 바꾸지 않는다.
- **Runtime helper 바인딩:** 출구에서 참조가 남은 bodyless 선언은 명시적 바인딩
  의도를 가진다. `abi = "C"` 선언은 C linkage이므로 선언의 symbol 이름이 곧 바인딩
  정체성이다. 이는 symbol 철자에서 의미를 추론하는 것이 아니라 FFI 선언이 명시한
  링크 이름이다. 각 target은 자신이 충족하는 C 바인딩을 명시적으로 선언한다. Native는
  runtime library와 링커가 C 이름을 해석하므로 모든 C 선언을 충족한다고 본다. Wasm은
  target이 구현을 제공하는 runtime helper 목록만 충족하며, 그 목록은 구현과 같은
  곳에서 정의한다. 참조되는 C 선언을 target이 충족하지 못하면 emission이 아니라 경계
  출구 검증이 거부한다. 최종 import 등록과 미참조 선언의 처분은 target emission이
  정한다.

#### 출구 적법성

출구에 남으면 안 되는 것:

- `tribute_control.*`, `ability.*`, `effect.*`, `closure.*` operation과
  `closure.closure` type
- Callable 결과로서의 `core.never`. 논리 CPS 결과는 물리 결과 목록 `[]`로 바뀐다.
- 의미적 호출 규약과 제어 metadata: `tribute.calling_convention`,
  `tribute.root_source_result`, `tribute.cps_continuation_frame_result`,
  `tribute.closure_environment_index`
- 물리 계약으로 옮기지 않은 채 남은 handler/resume/prompt 정체성과 effect row

출구 이후에도 보존하는 것:

- 기계 호출 규약, exact signature, 외부 바인딩(`abi`)
- Typed managed layout, 명시적 layout 식별자, ownership/RTTI 입력(매개변수 속성
  `tribute.ownership` 포함)과 `tribute.type.string` 같은 타입 식별 metadata
- Location과 `tribute.definition.*` 같은 실행에 관여하지 않는 source/debug 정보

경계 이후 pass는 금지된 metadata를 조회하거나 다시 만들지 않는다. 이름, 포인터 형태,
arity, 빈 결과 목록에서 소실된 의미를 복원하지 않는다. 의미적 분류를 이름만 바꾼
물리 속성으로 복제하는 것도 허용하지 않는다.

출구 검증기는 operation, callable type, alias, 중첩 type 속성, block 인자와 그
속성을 재귀적으로 검사한다. 속성 값 안의 dictionary key도 깊이와 관계없이 같은
규칙으로 분류한다. 금지 목록뿐 아니라 보존 목록에도 없는 언어 전용
`tribute.*` 속성은 분류되지 않은 metadata로 보고한다. 새 언어 전용 속성은 금지 또는
보존 중 하나로 분류된 뒤에만 출구를 넘을 수 있다. 입력과 결과 타입이 같은
unrealized cast, 대상 함수의 signature와 다른 타입의 `func.constant`도 위반으로
보고한다. 매개변수 ownership 계약도 검사한다. `call_conv = "tail"`인 signature의
입력에 `tribute.ownership = "consumed"`가 없거나, `tribute.ownership` 값이
`"consumed"`가 아니면 위반이다. Root `main`이 있으면 그것이 매개변수가 없고 결과가
`Nil`이며 platform 규약을 따르는 정의인지도 검사한다. 위반이 하나라도 있으면
target dialect lowering에 들어가기 전에 컴파일이 실패한다. 이 검증은 build 구성과 관계없이 항상 수행한다.

#### Unrealized cast

공유 converter로 알 수 있는 materialization(boxing 등)은 출구 전에 끝난다.
출구에는 target type 변환을 기다리는 `core.unrealized_conversion_cast`가 남을 수
있다. 이런 cast의 적법성은 target 타입에 따라 정해지므로 target type 변환이 결과
타입을 변환하고 필요한 representation 변경을 materialize한다. 그 끝에서 converter
없는 `reconcile_unrealized_casts`가 cast를 닫으며, 그 뒤 남은 cast는 target emission
경계가 거부한다.

#### 직렬화

경계 출력은 textual IR로 출력한 뒤 새 `IrContext`에서 파싱해도 frontend나 control
보조 테이블 없이 target pipeline을 계속 진행할 수 있어야 한다. 출구 계약에 필요한
정보는 모두 IR의 operation, type, attribute로 표현한다.

## Type Model

Primitive scalar, pointer, reference, bytes, array, tuple, function, and nil
types are represented in TrunkIR. Library data types such as `Option`, `Result`,
and `Text` lower through ADT and runtime/library conventions. `List` is an
opaque nominal builtin whose shared construction and sequence-view observations
use `list.*`; target-specific passes choose and eliminate its private
representation.

### 타입 textual form

타입은 스스로 끝을 표시한다(self-delimiting). 타입이 소유하는 매개변수와 속성은
모두 qualified name 바로 뒤의 `<…>` 안에 쓰고, 짝이 맞는 `>`에서 타입이 끝난다.
그 뒤에 오는 `{…}`는 타입에 속하지 않는다. Operation 속성, region, 또는 바깥
타입에서 이 타입이 매개변수일 때의 매개변수 속성이다.

```text
core.i32
core.tuple<core.i32, core.ptr {k = "v"}>
core.ref<core.i32, {nullable = true}>
adt.typeref<{name = "Point"}>
func.func_sig<(core.i32 {tribute.ownership = "consumed"}) -> core.i64, {call_conv = "tail"}>
```

- 매개변수도 속성도 없는 타입은 이름만 쓴다.
- 일반 형태 `dialect.name<원소, ...>`의 원소는 매개변수이거나 type 속성
  dictionary다. `{`로 시작하는 원소가 type 속성이며, 마지막에 한 번만 올 수 있다.
  매개변수 바로 뒤의 dictionary는 그 매개변수의
  [매개변수 속성](#타입-매개변수-속성)이다.
- `<` 바로 다음이 `(`이면 함수 형태다:
  `<(inputs...) -> results[, {type 속성}]>`. 입력과 결과도 각각 매개변수 속성을
  가질 수 있다. 결과 목록 규칙은
  [`func.func_sig`](#funcfunc_sig-function-type)를 따른다.
- 이 규칙은 일반 형태와 전용 문법에 똑같이 적용된다. 전용 문법을 가진 타입도
  모든 내용을 `<…>` 안에 둔다. 타입 안의 `(…)`는 함수 입력·결과처럼 위치가 있는
  목록에만 쓴다.
- 전용 문법은 그 타입을 정의하는 dialect가 type assembly format으로 등록해
  소유한다. 전용 형식은 일반 형식과 같은 저장 표현으로 읽히며, 전용 형식으로
  표현할 수 없는 타입(예: 검증에 실패한 타입)은 일반 형식으로 출력한다. 일반
  형식은 항상 읽을 수 있다.
- 저장 표현 전용인 예약 type 속성은 textual dictionary에 쓰지 않는다. 매개변수
  속성 list(`param_attrs`)와 함수 타입의 count가 여기에 해당하며, reader는 이를
  거부한다.

### Attribute 값

Operation, block 인자와 type의 속성 값은 다음 domain을 가진다: `unit`, bool,
정수, 부동소수점, 문자열, bytes, symbol 참조, type, location, list, dictionary.

Symbol 참조는 symbol table의 정의를 qualified path로 가리킨다(`callee = @foo`). 참조가
아닌 이름 값은 문자열이다. 비교 조건(`predicate`, `cond`), trap code, wasm value·heap
type 이름, import의 module·name처럼 정해진 짧은 이름(atom)이 여기에 해당하며
`predicate = "slt"`로 쓴다. Ability 이름(`core.ability_ref`의 `name`)과 operation
이름(`op_name`), operation kind(`"fn"`, `"op"`), compiler intrinsic identity
(`tribute.compiler_intrinsic`)도 symbol table의 정의가 아니므로 문자열이다. 기계 호출
규약(`call_conv = "tail"`), 매개변수 ownership 계약(`tribute.ownership = "consumed"`),
block 인자의 binding 이름(`bind_name`)도 같은 이유로 문자열이다.

정의의 이름(`sym_name`)도 문자열이다. 정의는 symbol table에 자기 이름을 등록할 뿐
다른 정의를 가리키지 않으므로 참조가 아니다(MLIR의 `sym_name`도 `StringAttr`다).
`func.func @foo`처럼 정의 operation의 전용 문법은 이름 앞에 `@`를 붙이지만, 저장되는
값은 문자열이다. 그래서 속성 값으로 쓰인 `@`는 언제나 참조를 뜻하고, 속성을 훑어
참조를 일반적으로 찾을 수 있다.

문자열 값은 그 속성을 가진 `IrContext`의 문자열 pool에 uniquing되며, 속성은 pool
handle만 담는다(MLIR에서 `StringAttr`가 `MLIRContext`에 uniquing되는 것과 같다).
같은 context 안에서 같은 내용의 문자열은 같은 handle을 가지므로 속성의 identity와
hash는 내용으로 정해진다. 문자열 값을 만들고 읽는 일은 context를 거친다. Handle은
다른 arena 참조(type, operation 등)와 마찬가지로 그것을 만든 context 안에서만 유효하다.
Context를 복제하면 pool도 복사되므로, 복제 전에 만든 handle은 양쪽에서 유효하고 복제
뒤에 만든 handle은 만든 쪽에서만 유효하다. Pool은 context와 함께 해제된다.

Dictionary(`Attribute::Dict`)는 symbol key에서 속성 값으로의 map이다. Key는 정렬된
순서로 보관·출력되며, identity와 hash는 삽입 순서가 아니라 key-value 내용으로
정해진다. Textual form은 값 위치의 `{key = value, ...}`이고, 빈 dictionary는 `{}`다.
값 위치에서는 `{`가 region body를 시작하지 않으므로 type 속성 dictionary나 operation
body와 모호하지 않다. Reader는 한 dictionary 안의 중복 key를 거부한다.

List와 dictionary는 임의로 중첩된다. 속성 값 안의 type은 type walk의 일부다. 속성에
담긴 type을 변환하거나 검사하는 pass는 list와 dictionary 안까지 모든 type에 도달해야
하며, 일부 variant만 따라가고 나머지를 그대로 통과시키지 않는다. 공용 순회는
`Attribute::visit_types`, `Attribute::map_types`, `Attribute::try_map_types`가 소유한다.

속성 값 안의 symbol 참조도 같은 방식으로 찾는다. 정의를 가리키는 참조를 모으는 pass는
속성 이름을 나열하지 않고 `Attribute::visit_symbol_refs`로 list와 dictionary 안까지
모든 참조에 도달한다.

### 타입 매개변수 속성

`TypeData.params`는 타입 인스턴스를 결정하는 매개변수 값이다(MLIR의 type
parameter와 같은 용법). 함수 signature에서는 input과 result 타입이 매개변수다.
매개변수 하나에 붙는 속성은 예약 type 속성 `param_attrs`에 둔다.

- 값은 `params`와 같은 순서, 같은 길이의 dictionary list다. 속성이 없는 매개변수는
  `{}`를 가진다.
- 모든 dictionary가 비어 있으면 key를 생략한다. 이것이 유일한 canonical 형태이므로
  매개변수 속성이 없는 타입은 하나의 identity를 가진다. 속성이 다른 두 타입은 서로
  다른 identity를 가진다.
- Textual form에서는 각 매개변수 바로 뒤에 dictionary로 쓴다:
  `core.tuple<core.i32, core.ptr {k = "v"}>`. Printer는 `param_attrs` key를
  출력하지 않고 reader는 이 key를 거부하므로, 매개변수 속성의 textual form은 이것
  하나뿐이다. 빈 dictionary(`core.i32 {}`)는 속성이 없는 것으로 읽는다.
- Type verifier는 모든 interned type에 같은 규칙을 적용한다. List가 아닌 값,
  dictionary가 아닌 원소, 길이 불일치, 모두 빈 list는 오류다.
- TrunkIR은 key의 의미를 해석하지 않는다. 의미는 key를 정의하는 dialect나 언어
  계층이 소유한다.
- 속성은 그것이 설명하는 매개변수를 따라간다. 매개변수를 끼우거나 빼며 타입을
  다시 만드는 pass는 이 list도 같은 위치에서 고친다. 새로 끼운 매개변수는 끼우는
  계층의 계약이 정한 속성을 가지며, 계약이 없으면 `{}`다. 예를 들어 경계 안에서
  물리 CPS callable에 끼운 환경은 다른 입력과 같은 ownership 계약을 가진다. 빠지거나
  다른 의미의 값으로 대체된 매개변수의 속성은 버린다. 예를 들어
  CPS 규약은 source result를 논리 `core.never`로 대체하고 물리 signature에서
  없애므로 그 result의 속성도 사라진다. 같은 개수를 유지하는 변환은 위치를 그대로
  두고, 속성 안의 type만 변환한다.
- 한 callable에서 파생한 signature들, 예를 들어 closure 환경을 끼운 정의, 함수
  참조 adapter, 호출 지점이 기대하는 물리 signature는 모두 같은 규칙으로 유도한다.
  그래야 정확한 타입 동일성 비교가 어긋나지 않는다.

### Runtime layout 식별자

Compiler가 소유하는 runtime 저장 layout은 예약 type 속성 `layout`으로 식별한다.
값은 layout 종류를 나타내는 문자열(atom)이다. Symbol table의 정의를 가리키지
않으므로 symbol이 아니다.

| `layout` | 붙는 타입 | 뜻 |
| --- | --- | --- |
| `"closure"` | canonical closure `adt.struct` | 함수 참조와 environment로 이루어진 closure 저장 |
| `"evidence_marker"` | evidence marker `adt.struct` | 한 handler의 ability id, prompt, dispatch closure, 가린 marker |
| `"evidence"` | evidence `core.array` | ability id 순으로 정렬된 가장 위 marker 배열 |
| `"bytes"` | Wasm bytes `adt.struct` | backing 배열, 시작 offset, 길이로 이루어진 `Bytes` 저장 |
| `"bytes_data"` | Wasm bytes backing `core.array<core.i8>` | `Bytes`가 가리키는 byte 배열 |
| `"boxed_f64"` | Wasm boxed float `adt.struct` | uniform 참조 자리에 놓이는 `f64` 필드 하나짜리 `Float` 저장 |
| `"described"` | Wasm `Described` `adt.struct` | descriptor 필드 하나로 이루어진 사용자 struct와 variant의 공통 supertype |

- 속성은 저장 layout만 나타낸다. 의미 분류를 physical 이름으로 복제하지 않으며,
  같은 의미의 값이라도 저장 layout이 다르면 이 속성으로 구별하지 않는다.
- 일반 type 속성처럼 interning identity에 참여한다. `layout`이 없는 같은 모양의
  타입과는 다른 타입이다. Textual form은 일반 type 속성과 같다:
  `adt.struct<_closure(func_ptr: core.ptr, env: core.ptr), {layout = "closure"}>`.
- 속성은 그 layout을 만드는 compiler의 canonical 생성자만 붙인다. Frontend와
  소스에서 온 타입은 이 속성을 갖지 않는다. 그래서 사용자 타입이 같은 이름이나
  모양을 가져도 compiler layout으로 취급되지 않는다.
- Target이 layout의 field 표현을 바꾸는 경우(예: native closure layout 적응)에도
  같은 `layout` 값을 유지한다. Target type 변환은 식별자를 가진 타입을 식별자가
  없는 erased reference(예: Wasm `arrayref`)로 바꾸지 않는다.
- Representation/ABI 경계 이후의 pass는 compiler 소유 layout을 이 속성으로만
  판별한다. Struct 이름, field 모양, element 타입, erased reference 타입으로
  판별하지 않는다. TrunkIR은 값의 의미를 해석하지 않으며, 의미는 이 속성을 정의하는
  언어 계층과 그 layout을 구현하는 target이 소유한다.

### `adt.struct` nominal layout type

`adt.struct`는 이름 있는 nominal struct layout이다. 이름과 필드 이름을 항상 가지며
전용 textual 문법을 쓴다. 이름은 식별자(`[A-Za-z_][A-Za-z0-9_]*`)면 그대로 쓰고,
아니면 따옴표 문자열로 쓴다(`adt.struct<"Nested::Closure"(…)>`, `"0": core.i32`).
Struct 이름과 필드 이름은 타입 이름공간의 이름이며 symbol table의 정의가
아니므로 `@`를 붙이지 않는다.

```text
adt.struct<Point(x: core.i32, y: core.i32)>
adt.struct<Node(value: core.i32 {k = "v"}, next: adt.typeref<{name = "Node"}>)>
adt.struct<_closure(func_ptr: core.ptr, env: core.ptr), {layout = "closure"}>
adt.struct<Empty()>
```

- 저장 표현은 다음과 같다. 필드 타입은 `params`에 선언 순서대로 둔다. 필드 이름은
  각 필드의 [매개변수 속성](#타입-매개변수-속성) `name`(문자열)이고, struct 이름은
  type 속성 `name`(문자열)이다. 필드 하나의 다른 속성은 같은 매개변수 속성
  dictionary에 함께 둔다.
- 이름과 모든 필드 이름은 필수이고, 필드 이름은 struct 안에서 겹치지 않는다.
  매개변수가 없으면 필드가 없는 struct다. 필드 목록을 담는 별도 type 속성은 없다.
  Type verifier가 이 규칙을 모든 interned `adt.struct`에 적용한다.
- 필드 속성 안의 `name`과 type 속성의 `name`, `param_attrs`는 전용 문법이
  소유한다. Textual form의 필드 속성과 type 속성 dictionary에는 쓰지 않는다.
  `layout` 같은 나머지 type 속성은 `<…>` 안 마지막 원소로 둔다.
- 필드 타입이 매개변수이므로 일반 타입 순회와 변환은 필드 타입에 그대로 도달한다.
  필드 수를 유지하는 변환은 필드 이름과 속성을 위치 그대로 둔다.
- 재귀 참조는 같은 이름의 `adt.typeref`로 표현한다. `adt.struct`는 자기 자신을
  매개변수로 갖지 않는다.

#### Nominal 수준과 structural 수준

Struct 이름과 필드 이름은 IR에서 nominal layout을 해석하는 단계까지만 의미를
가진다. `adt.typeref`를 같은 이름의 layout으로 해석하는 단계, ownership 계획, ABI
검증, frontend의 record pattern 해석이 여기에 속한다. 그 아래의 target lowering은
필드의 물리 표현과 순서만 사용한다. 이름이 필요한 runtime 동작(해제, 값 출력,
variant 판별)은 [runtime 타입 descriptor](runtime-types.md)가 맡는다. Nominal
layout을 마지막으로 해석하는 target 경계가 할당에 descriptor 번호를 새긴다.

저수준 struct 타입이 갖는 identity는 runtime 동작이 달라지는 경우로 한정한다.
Layout과 runtime 동작이 같은 두 소스 타입은 같은 저수준 struct를 쓰고
descriptor로만 구별된다.

- Native는 nominal layout을 마지막으로 해석하는 경계에서 struct field 접근의
  `adt.struct`를 이름 없는 `mem.struct<T...>`로 내린다. `mem.struct`는 필드 타입만
  갖고 자연 정렬 memory layout을 뜻한다. 필드 타입은 target 표현이며, 해제 동작이
  다른 managed 참조(`tribute_rt.anyref`)와 unmanaged 포인터(`core.ptr`)는 크기가
  같아도 구분한다. 할당이 해제하는 필드가 managed 참조이며, 이 판정은 ownership
  계획의 것이고 물리 타입에서 다시 유도하지 않는다. `mem.struct`의 field offset과
  크기는 같은 nominal layout에서 계산한 할당 크기와 일치한다.
- 할당은 descriptor를 nominal layout 타입으로 찾으므로 RC header를 새기는 단계가
  nominal `adt.struct`의 마지막 사용처다. Field offset 계산은 타입 변환 없이
  `mem.struct`만 읽고, field 접근이 `clif.load`와 `clif.store`의 offset이 될 때
  `mem.struct`도 사라진다. Cranelift에는 aggregate 타입이 없다.
- Wasm은 nominal layout을 마지막으로 읽는 `adt_to_wasm` 뒤에 `adt.struct`를 이름
  없는 `wasm_gc.struct<T...>`로 바꾼다. 이 치환은 함수 signature와 block 인자를
  포함한 모듈 전체에 적용되므로 모든 위치가 같은 타입을 가진다. Variant는 처음부터
  자기 필드의 `wasm_gc.struct`로 낮춘다. 필드 표현이 같은 struct와 variant는 같은
  GC 타입이며, 첫 필드의 descriptor가 runtime identity를 맡는다. Compiler 소유 layout은
  [`layout`](#runtime-layout-식별자)으로 식별한다.
- 저수준 struct는 재귀하지 않는다. 재귀 참조는 이미 native pointer나 Wasm 추상
  reference로 끊겨 있다.
- 두 target 모두 [`adt.enum`](#adtenum-nominal-layout-type)의 variant 객체를 자기
  필드만 가진 저수준 struct로 두고, 어느 variant인지는 값의
  [descriptor](runtime-types.md#variant-판별)로 판별한다. Native에서 필드는 payload의
  offset 0부터 자연 정렬로 놓이고, variant field 접근은 struct field 접근과 같이 그
  variant의 `mem.struct`를 읽는다. 한 enum의 variant는 모두 가장 큰 variant의
  payload 크기로 할당하므로, enum 타입의 값은 variant와 무관하게 하나의 정적 할당
  크기를 가진다.

### `adt.enum` nominal layout type

`adt.enum`은 이름 있는 nominal enum layout이다. 이름과 variant 목록을 가지며 전용
textual 문법을 쓴다. 이름 표기 규칙은 `adt.struct`와 같다.

```text
adt.enum<Option { None(), Some(value: T) }>
adt.enum<List { Empty(), Cons(core.i32, adt.typeref<{name = "List"}>) }>
adt.enum<"geo::Shape" { Dot(), Rect(core.f64, core.f64 {k = 1}) }, {layout = "x"}>
adt.enum<Never {}>
```

- Variant는 이름과 괄호로 감싼 필드 목록이다. 필드가 없어도 괄호를 쓴다. 필드는
  `name: type` 또는 이름 없이 `type`이며, 어느 쪽이든 그 필드의 다른 속성
  dictionary가 뒤따를 수 있다.
- 저장 표현은 다음과 같다. Variant 하나가 `adt.enum`의 타입 매개변수 하나이며 선언
  순서대로 놓인다. Variant는 `adt.variant` 타입이고, 그 `params`가 필드 타입, type
  속성 `name`(문자열)이 variant 이름이다. 필드 이름이 있으면 그 필드의
  [매개변수 속성](#타입-매개변수-속성) `name`(문자열)이다. Enum 이름은 `adt.enum`의
  type 속성 `name`(문자열)이다.
- Enum 이름과 모든 variant 이름은 필수이고, variant 이름은 enum 안에서 겹치지
  않는다. 매개변수가 없으면 variant가 없는 enum이다. Variant 목록을 담는 별도 type
  속성은 없다. Type verifier가 이 규칙을 모든 interned `adt.enum`에 적용한다.
- Variant와 필드의 `name`, enum type 속성의 `name`은 전용 문법이 소유한다. Textual
  form의 속성 dictionary에는 쓰지 않는다. 나머지 type 속성은 `<…>` 안 마지막
  원소로 둔다.
- Variant 매개변수는 [매개변수 속성](#타입-매개변수-속성)을 갖지 않는다. 전용 문법에
  그것을 쓸 자리가 없으므로 `adt.enum`의 `param_attrs`는 type verifier가 거부한다.
  필드의 속성은 그 variant(`adt.variant`)의 매개변수 속성이다.
- `adt.variant`는 `adt.enum`의 매개변수로만 쓴다. 값의 타입이나 연산의 타입 속성이
  되지 않으며, 연산은 enum 타입과 variant 이름(`tag`)으로 variant를 가리킨다.
- Variant 필드 타입이 매개변수의 매개변수이므로 일반 타입 순회와 변환은 variant
  필드 타입에 그대로 도달한다.
- 재귀 참조는 같은 이름의 `adt.typeref`로 표현한다.

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
Following the [self-delimiting type form](#타입-textual-form), each input or
result may carry its parameter attributes, and the remaining type attributes
come last inside the brackets:

```text
func.func_sig<(core.i32 {tribute.ownership = "consumed"}) -> (), {call_conv = "tail"}>
```

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
Production code는 검증된
`func::func_sig` API로 함수 타입을 만들며, 직접 `TypeData`를 만드는 코드는
malformed-type verifier test에 한정한다.

The three result states have distinct meanings:

```text
logical Unit function: results = [core.nil]
logical CPS function:  results = [core.never]
physical CPS function: results = []
```

The stored counts are not an ABI marker, and general resultless IR need not be
CPS. Before the [representation/ABI boundary](#representationabi-경계), the
physical CPS callable is identified by its semantic calling convention together
with `results = []`. After the boundary, the semantic convention is gone: a
proper tail transfer is the `func.tail_call`/`func.tail_call_indirect` operation
itself, and the machine calling convention is the signature's `call_conv`
attribute.

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
every input/result and nested type attribute, including types inside
[parameter attributes](#타입-매개변수-속성), preserving cardinality, and
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
동일성에 포함되며 파싱·출력·별칭·재귀 변환에서 보존된다. 다만 공통
`func.func_sig`를 `wasm.func_sig`로 변환할 때는 입력과 결과 타입만 옮기고
예약되지 않은 속성은 [타입 매개변수 속성](#타입-매개변수-속성)을 포함해 모두 버린다. Wasm 함수 타입은 구조적이라 바이너리에는
매개변수와 결과 타입만 남으므로, 공통 signature metadata는 Wasm에서 의미가 없고
동일한 Wasm 타입을 갈라놓을 뿐이다.

Wasm 함수와 가져오기 선언, 직접·간접 호출, 반환, 타입 섹션 수집, 검증 및 코드
생성은 이 타입을 사용한다. 모든 간접 호출은 exact `wasm.func_sig`를
명시해야 하며, 누락된 시그니처는 lowering과 emission에서 오류다. Operand, result,
상위 함수의 return, `type_idx` 또는 타입 정보가 지워진 테이블 인덱스에서
시그니처를 재구성하지 않는다. 빈 결과 목록만으로
CPS를 판정하지 않는다. 논리적 Unit 함수, 논리적 CPS 함수, 물리적 CPS 함수의
결과 구분은 [공통 `func.func_sig` 계약](#funcfunc_sig-function-type)을 따른다.
Wasm에는 별도의 기계 호출 규약이 없으므로 `call_conv`도 위 규칙에 따라 버린다.
Proper tail transfer는 `wasm.return_call`과
`wasm.return_call_indirect` operation으로만 표현하며, 그 검증은 피호출자와 둘러싼
함수의 결과 목록 호환만 본다.

### `clif.func_sig` 네이티브 호출 계약

`clif.func_sig`는 네이티브가 소유하는 함수 호출 계약이다. 입력 우선의 평탄한
`[inputs..., results...]` 벡터와 필수 `u32` 속성 `num_inputs`·`num_results`를
사용하고 결과는 0개 이상을 허용한다. 두 개수의 합은 벡터 길이와 같아야 하며,
두 속성과 예약되지 않은 타입 속성은 모두 타입 동일성에 참여한다. count는 저장
경계일 뿐 ABI 또는 calling convention의 증거가 아니다. parser, printer, alias는
예약되지 않은 속성을 보존하며, 호출 계약 내부의 재귀 타입 변환도 중첩 메타데이터를
그 소유 타입에 보존한다. `func.func_sig`에서 변환할 때 타입 매개변수 속성은 위치를
유지한 채 옮기고, Cranelift signature 번역은 이를 읽지 않는다.

네이티브 함수와 선언, 직접·간접 호출, 반환, proper tail transfer 및 Cranelift 코드
생성은 이 타입을 사용한다. 간접 호출의 `sig`는 정확한 `clif.func_sig`이어야 하며,
타입이 지워진 함수 포인터, 심볼, ABI 문자열 또는 저장 형태에서 재구성하지 않는다.
결과가 비었다는 사실만으로 CPS를 판정하지 않는다. 논리 Unit `[core.nil]`, 논리
CPS `[core.never]`, 물리 CPS `[]`의 구분은
[공통 `func.func_sig` 계약](#funcfunc_sig-function-type)을 따른다.
Cranelift 호출 규약은 `clif.func_sig`의 `call_conv` 속성이 정하며, native
lowering은 `func.func_sig`의 `call_conv`를 그대로 옮긴다. 속성이 없으면 platform
규약이다. Emitter는 operation이나 함수의 다른 속성에서 호출 규약을 고르지 않는다.

네이티브 최종 호출 계약의 각 operand와 result slot은 `clif.func_sig`의 같은 순서
slot과 정확히 같은 TrunkIR type이어야 한다. semantic reference SSA 값은 native
lowering이 그 producer 또는 block argument를 `core.ptr`로 명시적으로 낮춘 뒤에만
`core.ptr` slot을 채울 수 있다. 검증과 emission은 dialect 이름, type attribute, ABI
문자열, symbol, erased representation에서 pointer 동치를 추론하지 않는다. 이 규칙은
`core.nil`의 정해진 zero-width projection과 별개이며, 다른 contract type 사이의
호환성 규칙을 만들지 않는다.
