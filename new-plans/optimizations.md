# Tribute Optimizations

이 문서는 source-logical compiler pipeline의 최적화 경계와 검증 계약을 정의한다.
Callable ABI, operation kind, ownership 의미는 최적화 설정에 따라 달라지지 않는다.

## Legalization과 최적화의 경계

Frontend는 typechecked AST를 source-logical `tribute_control` IR로 내린다.
Monomorphization은 이 경계 전에 checked function instance와 nominal metadata를
재작성한다. 자세한 계약은 [generics.md](generics.md)를 따른다.

Shared `tribute_control_to_cps`는 callable graph 전체를 검증한 뒤 CPS를 적법화한다.
`fn` operation은 선언이 보장한 tail-resumptive 계약에 따라 `ability.call`로,
`op` operation은 `ability.perform`으로 변환한다. Handler 본문의 마지막 호출이나
resume 횟수로 operation kind를 재분류하지 않는다. Continuation 구성, 명시적인
evidence 전달과 target ABI 물리화는 필수 lowering이며 선택적인 최적화가 아니다.

정확한 결과 타입, calling convention과 hidden parameter placement는
[cps-effects.md](cps-effects.md)를 따른다. `anyref` boxing은 일반적인 값 표현이며,
특수화나 최적화가 source 결과를 control carrier로 바꾸는 근거가 되지 않는다.

## 적용 위치

| 경계 | 변환 | 보존해야 하는 계약 |
| ---- | ---- | ----------------- |
| Frontend preparation | 함수·nominal 타입 monomorphization | 해석된 선언 identity, checked instance와 치환된 semantic metadata |
| Shared CPS 이후, target closure storage 이전 | 일반 함수 inlining | exact callable ABI, 명시적 evidence·ContinuationFrame, proper-tail control flow |
| Target cleanup | global DCE, canonicalization, local DCE, 실제 operation이 필요한 conversion cast materialization | side effect, reachable transfer와 target type legality |
| Native typed ownership planning | proven borrowed parameter·field temporary elision | managed layout, entry ownership와 사용·탈출 증명 |
| Native RC lowering 이전 | paired retain/release elimination | alias barrier, 각 reference의 수명과 소유권 |

일반 inlining은 지원되는 single-block, `cf` 없는 함수만 변환한다. 효과 호출을
인라인해도 operation declaration이나 handler의 의미를 다시 추론하지 않는다.
선택적인 native RC 정책과 materialization 순서는 [rc.md](rc.md)를 따른다.

## 선택 가능한 native 정책

`OptimizationOptions`는 active native pipeline이 소비하는 정책만 노출한다.
`production()`과 `baseline()`은 같은 frontend와 shared legalization을 실행한다.
Baseline은 다음 native 최적화만 끈다.

| 정책 | Baseline | Production |
| ---- | -------- | ---------- |
| `PairedRcEliminationPolicy` | `Disabled` | `Enabled` |
| `BorrowedParameterPolicy` | `Preserve` | `ElideProvenBorrowed` |
| `TemporaryBorrowPolicy` | `Preserve` | `ElideProvenFieldBorrows` |

Lowering 옵션은 stage별 immutable value로 전달한다. 재사용 가능한 compiler helper를
생성하는 최적화가 필요하다면 그 mutable cache는 생성 단계가 소유한다. 소비자가 없는
옵션이나 서로 다른 lowering 경로를 선택하는 compatibility switch는 두지 않는다.

## Optimization validation contract

선택적인 최적화는 production에 켜기 전에 다음 조건을 만족해야 한다.

1. 같은 source fixture를 최적화 비활성/활성 상태로 컴파일하고 실행하여 observable
   result가 같음을 확인한다.
2. Named pipeline boundary에서 focused before/after IR을 검사한다. Lowering 내부
   정책은 같은 경계의 두 결과를 비교하고, IR pass는 변환 직전·직후를 비교한다.
3. 제거 대상 operation 수와 반드시 남아야 하는 semantic operation을 구조적으로
   확인한다.
4. 증명이 불완전하면 IR을 유지하는 negative fixture를 둔다.
5. Ability와 RC 검증은 공통 실행 fixture/helper를 사용한다.

비교 중에는 검사할 정책 하나만 바꾼다. 다른 최적화 정책, target과 sanitizer 설정은
동일하게 유지한다. 실행 시간이나 메모리 개선은 재현 가능한 측정으로 보고하며,
예상 수치를 검증된 성능 결과처럼 문서화하지 않는다.

Production composition과 옵션은 [`src/pipeline.rs`](../src/pipeline.rs), native
conformance 검증은 [`tests/optimization_conformance.rs`](../tests/optimization_conformance.rs)에
있다. 필수 legalization 자체의 동등성은 active pipeline·handler execution·target
검증으로 확인한다.

## 추가 최적화의 계약

다음 최적화는 현재 pipeline의 별도 pass나 지원 주장이 아니다. 도입할 때에도
source-logical 경계와 위 검증 조건을 유지해야 한다.

- **Effect specialization와 handler inlining:** concrete effect row만으로 handler
  identity나 구현을 확정하지 않는다. 명시적인 dispatch와 closure provenance가
  뒷받침할 때만 evidence lookup 또는 간접 호출을 제거한다. Open row는 선언된
  callable convention을 유지한다.
- **Continuation allocation elision:** one-shot capability의 수명과 모든 capture의
  ownership을 증명한 경우에만 heap allocation을 대체한다. Scope 밖으로 탈출하거나
  증명이 불완전한 continuation은 원래 저장소 계약을 유지한다.
- **Partial evaluation:** compile-time 입력이 확정되고 effect·trap·평가 순서가
  보존되는 계산만 미리 수행한다.
- **Profile-guided optimization:** profile은 후보를 고르는 근거이며 의미 보존
  증명을 대체하지 않는다.

## References

- [Generalized Evidence Passing (Koka)](https://www.microsoft.com/en-us/research/publication/generalized-evidence-passing-for-effect-handlers-or-efficient-compilation-of-effect-handlers-to-c/)
- [GHC Specialisation](https://wiki.haskell.org/Inlining_and_Specialisation)
- [MLton Whole-Program Optimization](http://mlton.org/WholeProgramOptimization)
