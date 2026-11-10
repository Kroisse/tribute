# Frontend 단계와 증분 분석

이 문서는 frontend의 단계 경계와 증분 분석 계약을 정한다. 구문 분석, 이름 해석,
타입 검사, 특수화, 논리 IR 생성이 각각 무엇을 소유하는지, 그리고 Salsa query가
어떤 입력을 읽고 편집 뒤에 무엇을 재사용하는지를 다룬다. Frontend 이후의 pass
순서는 [lowering.md](lowering.md#pipeline-order)가 정한다.

## 표준 라이브러리 입력

패키지 `std`의 well-known type은 typechecking 결과의 별도 metadata로 보존한다.
`WellKnownTypes`는 `std::String`의 semantic type과 stable declaration identity를
frontend 경계까지 전달한다. AST-to-IR은 declaration identity로 exact TrunkIR
`TypeRef`를 구해 root module의 [`tribute.types`](tribute-dialects.md#tributetypes)
attribute에 기록한다.

표준 라이브러리 source(패키지 `std`의 루트 모듈)는 compilation 입력이다. 이름
해석은 패키지마다 자기 루트에서 하며, 사용자 패키지의 해석은 의존 패키지 `std`의
해석 결과를 의존성 표로 참조한다. 타입 검사도 패키지마다 하며, 사용자 패키지의
검사는 의존 패키지 `std`의 검사 결과를 같은 의존성 표로 조회한다. 한 패키지의 타입
검사 결과는 그 패키지가 선언한 항목만 담고 의존 패키지의 선언을 복사하지 않는다.
`std`의 해석과 타입 검사 결과는 사용자 source에 의존하지 않으므로, 사용자 source를
고쳐도 다시 계산하지 않는다. 테스트는 표준
라이브러리 source 뒤에 덧붙인 source를 패키지 `std`의 일부로 compile해, 사용자 패키지가 가리킬 수 없는 패키지 내부 항목
(`std::runtime`)을 사용자 프로그램이 부를 수 있는 함수로 감싸 시험한다.

## 프론트엔드 단계 경계와 정보 소유권

프론트엔드의 각 단계는 자신이 확정한 의미 정보와 원본 위치를 다음 단계에
함께 전달한다. 다음 단계는 표현식 모양, 출력 이름, 물리 표현을 이용하여 이미
확정된 타입이나 선언 identity를 다시 추론하지 않는다.

| 경계 | 소유하는 정보와 책임 |
| --- | --- |
| 구문 분석 | unresolved AST, 원본 위치, 구문 진단 |
| 이름 해석 | 선언과 참조의 identity, lexical binding, 선언 테이블 |
| 타입 검사 | 자기 패키지의 typed 선언 테이블, type schemes, 표현식 타입, 선택된 callable과 operation 인스턴스 |
| 특수화 | 구체 인스턴스를 더한 선언 테이블, 함께 치환된 메타데이터, compiler intrinsic identity, 원본 위치 대응 |
| 논리 IR 생성 | source-logical callable/control IR, 검증에 필요한 선언 메타데이터 |

모듈 계층은 구문 분석과 이름 해석에서만 의미를 가진다. Scope, `use`,
companion 모듈과 visibility는 이 계층을 따라 해석한다. 이름 해석은 그 결과를
선언 테이블로 내보낸다. 테이블의 각 항목은 선언 하나와 그 선언을 둘러싼 모듈의
path를 가지며, 모듈 path는 선언 identity의 path에서 마지막 component를 뺀 것과
같다. 이후 단계는 모듈 트리를 다시 걸어 path를 재구성하지 않고 테이블 항목에서
읽는다. Scope에 따라 달라지는 타입 조회도 항목의 모듈 path를 기준으로 한다.

테이블의 순서는 명시적이다. 원본 선언은 소스 순서를 따르고, 특수화가 만든
인스턴스는 정해진 정렬 순서로 뒤에 덧붙는다. 해시 순서는 테이블 순서나 IR 선언
순서에 영향을 주지 않는다. 특수화는 인스턴스를 그 path의 모듈에 속한 새 항목으로
더하며, 모듈 path를 표현하기 위한 합성 모듈 선언을 만들지 않는다.

타입 검사 결과와 특수화 결과는 서로 다른 단계의 계약이다. Lowering 입력은
특수화 결과에서 구성하며, AST 복제와 대응 메타데이터의 복제·치환을 하나의
작업으로 취급한다. 선언 정보와 표현식·참조별 정보를 함께 전달하여 함수
인스턴스, constructor scheme, handler/perform/lambda 정보, coverage 증명,
nominal identity와 intrinsic provenance가 누락되지 않게 한다. 이 계약은
동일한 정보를 여러 결과 객체에 중복 저장하도록 요구하지 않는다. 특수화 결과에는
구체 함수 정의만 있으며, generic 함수 template과 그 메타데이터는 제외한다
([Generics — 특수화 결과](generics.md#특수화-결과)).

Lowering 입력을 만드는 경로는 특수화 결과에서 출발하는 하나뿐이며, 타입 검사
결과는 lowering 입력이 될 수 없다. 테스트도 같은 경로를 거친다.

패키지의 특수화는 그 패키지의 checked 결과와 그것이 의존하는 패키지들의 checked
결과에만 의존하는 tracked 결과이며, 다른 패키지의 특수화 결과에는 의존하지 않는다.
특수화는 그 패키지의 선언 테이블을 대상으로 한다. 의존 패키지의 선언
메타데이터(function·constructor scheme, ability 정의, well-known identity, compiler
intrinsic identity)는 조회로 받고, 의존 패키지 generic의 본문과 표현식·참조별
메타데이터는 인스턴스를 복제할 template으로만 읽는다. 수집은 그 패키지의 본문과 그
특수화가 만든 복제본에서 출발하며, 거기서 도달하는 인스턴스는 타입 인자가 의존
패키지의 타입뿐이어도 모두 그 패키지의 특수화가 만든다. 다형적 재귀 검사와 도달하는
인스턴스 검증도 그 패키지의 함수와 거기서 도달하는 인스턴스를 대상으로 한다. 따라서
같은 인스턴스를 여러 패키지의 특수화가 각자 만들 수 있다. 인스턴스의
이름·본문·메타데이터는 그 generic의 checked 본문과 타입 인자만으로 정해지므로 어느
패키지가 만들었든 같다. 한 패키지를 편집하면 그 패키지와 그것에 의존하는 패키지의
특수화만 다시 계산하며, 의존 패키지가 어떤 인스턴스를 만드는지는 그것에 의존하는
패키지의 특수화 결과에 영향을 주지 않는다.

이름 해석과 타입 검사, 특수화는 패키지마다 하지만 이후 단계는 빌드 전체의 하나의
module에서 실행하므로([Linking — Compilation Unit](linking.md#compilation-unit)),
빌드는 패키지들의 특수화 결과를 하나의 특수화 결과로 병합한다. 패키지들은 의존
순서(위상 순서)를 따라 각 패키지를 그것이 의존하는 패키지들 뒤에, 컴파일하는
패키지를 마지막에 둔다. 의존 관계가 순서를 정하지 않는 패키지 사이에도 순서는
빌드마다 같게 정해진다. 병합한 테이블은 이 순서로 각 패키지의 원본 선언을 둔 뒤 각
패키지의 인스턴스를 두며, 각 패키지의 인스턴스는 그 특수화가 정한 정렬 순서를
따른다. 둘 이상의 결과에 같은 인스턴스가 있으면 순서상 처음 것 하나만 남긴다. 함수
인스턴스는 path로, nominal 인스턴스는 그 선언의 노드로 같은지 정한다. Nominal
인스턴스의 노드는 타입 인자가 정하며, 같은 타입 인자를 다르게 표기하면 path가 같아도
노드가 다르다. 함수 인스턴스도 그렇게 표기가 다르면 path는 같고 복제한 노드는
다르므로, 남기지 않은 사본의 노드에만 있는 메타데이터는 그 사본과 함께 버린다. 선언
메타데이터와 표현식·참조별 메타데이터도 같은 순서로 합치며, 같은 key의 값은 위 성질에
따라 같다. 병합한 결과는 모든 패키지의 본문을 한 번에 특수화한
결과와 원본 선언의 순서, 인스턴스, 메타데이터가 같다. 의존 패키지의 결과에 이미 있는
인스턴스를 다시 만들지 않는 것은 최적화로만 허용하며, 그때도 병합한 결과는 같아야
한다. Backend separate compilation에서 인스턴스 소유권을 정하는 조건은
[Linking — Separate Compilation의 요구 조건](linking.md#separate-compilation의-요구-조건)을
따른다.

Compiler intrinsic 선언의 identity는 특수화 결과가 함께 가진다. Runtime item
검사와 병합된 source 목록은 root crate의 준비 단계가 lowering에 공급하는 환경이다.
Runtime item은 의존 패키지 `std`에만 있으므로 그 검사는 `std`의 checked 결과에만
의존한다.

Nominal 선언 인덱스와 치환된 constructor 스킴은 frontend 준비 단계가 소유한다.
Lowering은 준비에 성공한 AST와 대응 스킴을 함께 소비한다. 인스턴스 수집,
스킴 재사용, variant 연결과 확장 한도의 상세 계약은
[Generics — Nominal 타입 수집과 재작성](generics.md#nominal-타입-수집과-재작성)을 따른다.

Method 후보 선택과 scheme instantiation은 타입 체커가 소유한다. 선택된 호출은
문법과 무관하게 함수 호출의 검사 규칙 하나를 거친다. 후보가 하나로 좁혀지지 않은
호출은 제약 해결 과정에서 선택을 확정한다. `x.f(y)`와 `f(x, y)`는 이름 해석
뒤로 같은 호출 노드이고, callee는 그 이름이 가리킬 수 있는 함수들이다. 선택된
함수는 그 callee를 대신하며, 선택된 callee와 인스턴스를 소비할 뿐 source
annotation에서 타입을 재구성하지 않는다. 해결되지 않은 참조는 원본 위치의 진단으로 남는다.

타입 체커 내부에서 선언 환경, 표현식 검사, 제약 해결, 치환·일반화, 결과
확정, 진단 출력은 구분된 책임이다. 제약 해결 상태와 해결 순서는 하나의
solver가 소유한다. Equality, coercion, common-result join과 ability-row
관계의 의미를 모듈 분리 때문에 바꾸지 않는다. 함수 binder, 지역 binder,
추론 변수의 치환 정책은 구분하며, 타입 순회에는 effect 인자와 유지되는
row 관계도 포함한다.

진단은 제약의 source origin을 유지한 구조화된 오류에서 생성한다. 검사 중
재방문에 필요한 기록과 다음 단계로 내보낼 결과는 수명이 다르며, 결과를
확정할 때 AST와 메타데이터에 같은 치환·일반화를 적용한다. 미해결 solver
변수를 downstream 의미 정보로 내보내지 않는다.

CLI, LSP와 테스트는 같은 분석 의미를 사용하며 source와 외부 선언 환경을
명시적으로 전달한다. 패키지 `std` 로딩과 타깃·공유 패스 조합은 root crate의
책임이다. 오류가 있는 분석 결과를 조회하는 것과 IR 생성을 허용하는 것은
구분한다. LSP는 부분 결과를 사용할 수 있지만 컴파일 경로는 frontend 오류가
있으면 lowering을 시작하지 않는다.

이 공통 분석은 `tribute-front`가 소유하며, source를 분석할 환경으로 source가
의존하는 패키지들의 의존성 표를 인자로 받는다. 라이브러리 패키지는 이름, root
모듈 source, 의존성 표를 가진 입력이다.
의존성 표는 다른 라이브러리 패키지를 이름으로 가리키며, `std`가 아닌 모든
라이브러리 패키지의 표는 `std`를 포함한다. 패키지 `std`도 그런 라이브러리 패키지
하나이며 의존성 표가 비어 있다. 패키지의 환경 구성, 이름 해석과 타입 검사는 그
패키지를 key로 하는 query이므로, 한 패키지는 한 번만 검사되고 그 결과를 그
패키지에 의존하는 모든 패키지의 분석이 공유한다. 실행 파일 루트 패키지에 의존하는
패키지는 없으므로, 루트는 입력을 따로 두지 않고 source와 의존 패키지 목록으로
분석한다. 외부 선언 없이 source만 분석할 때는 root 모듈이 비어 있어 compiler
소유 이름만 선언하는 패키지 `std` 하나를 의존성 표로 쓴다.

이름 해석이 이름으로 찾는 패키지는 의존성 표에 있는 패키지뿐이다. 다만 re-export가
가리키는 대상은 그 re-export를 둔 패키지의 의존 패키지에 있을 수 있으므로, 경로를
그 대상에서 이어갈 때는 전이적으로 의존하는 패키지의 환경도 읽는다. 타입 검사는 의존성 표의 패키지와 그
패키지들이 전이적으로 의존하는 모든 패키지의 export를 읽는다. 의존 패키지의
export에 있는 타입이 직접 의존하지 않는 패키지의 선언을 가리킬 수 있기 때문이다.
의존성 표에 순환이 있으면 패키지 오류다. Root crate는 패키지 입력을 만들고, 같은
분석 결과에서 lowering을 시작할지 결정한다.

정의가 지켜야 하는 source 규칙은 타입 검사가 정의마다 진단한다. 그 정의가
인스턴스화되는지와 상관없이, 도달하지 않는 generic 함수 본문에서도 같은 진단이
나온다. 리터럴 패턴이 자기 타입의 `==`를 요구하는 규칙, ability operation 호출의
인자 개수가 그 예다. 특수화와 IR lowering은 이 규칙들이 이미 지켜졌다고 보고,
위반을 만나면 source 진단이 아니라 compiler 내부 불일치로 취급한다. Lowering이
내는 진단은 타깃 표현의 구현 한계뿐이며, 이 한계는 실제로 생성되는 코드에만
적용된다.

진단만 내는 컴파일은 frontend, 논리 IR 생성, class 특수화를 거쳐
[`fn` arm 호출 검사](cps-effects.md#fn-arm-호출-검사)에서 멈춘다. Source 규칙
위반은 모두 이 지점까지 드러나므로, 이 컴파일은 frontend 진단과 `fn` arm 호출
진단을 모두 보고한다. 이 컴파일은 도달 가능성으로 함수를 거르지 않아, 도달할 수
없는 함수도 class 특수화를 거친다. `fn` arm 호출 검사는 자기 규칙대로 root에서
닿는 인스턴스만 검사한다. 그 뒤 단계(CPS 적법화, evidence와 handler lowering,
representation/ABI 경계)의 실패는 source 진단이 아니라 compiler 내부 불일치이며,
산출물을 만드는 컴파일이 보고한다.

## 패키지 source 트리와 파일 입력

패키지의 AST는 root 파일 하나에서 출발한 source 트리 전체다. 본문 없는 `mod`
선언마다 [모듈 선언](modules.md#모듈-선언)이 정한 파일을 읽어 그 선언의 본문으로
넣으며, 결과는 같은 패키지를 인라인 모듈로 쓴 AST와 같은 모양이다. 이름 해석과
타입 검사는 파일 구성을 모른 채 패키지 하나를 검사한다. 트리를 만드는 query는
`tribute-front`가 소유하며 root 파일과 노드 번호의 기준 모듈 이름을 key로 한다.
라이브러리 패키지 입력과 매니페스트 없이 컴파일하는 source 파일이 모두 이 query를
거친다.

- 파일 하나의 구문 분석은 그 파일의 source를 key로 하는 query다. 트리 query는 각
  파일의 구문 분석 결과를 모아 본문을 채우고 span map을 합치기만 하므로, 파일
  하나를 편집하면 그 파일의 구문 분석과 트리 query만 다시 실행되고 다른 파일은
  다시 구문 분석하지 않는다.
- 파일은 URI로 찾는 Salsa 입력이다. 경로마다 입력 하나를 처음 요청받을 때 만들며,
  그 입력의 값은 파일이 있으면 source, 없으면 없음이다. 없는 파일도 입력으로
  남으므로 트리 query가 찾았으나 없던 경로는 그 query의 의존성이 되고, 나중에 그
  파일이 생기면 트리가 무효화된다. 경로별 입력은 지우지 않으며, 파일이 사라지면
  값을 없음으로 바꾼다. 한 경로에는 입력이 하나뿐이어야 하므로, 같은 경로의
  편집기 버퍼와 디스크 파일은 같은 입력을 갱신한다. 경로는 `file:` URI로 쓰며,
  URI path가 그대로 허용하지 않는 바이트(`%` 포함)는 percent-encoding한다. 파일을
  읽거나 진단에 경로를 보일 때는 decode한 경로를 쓴다.
- Query는 database 종류에 의존하지 않도록 `&dyn salsa::Database`만 받는다. 그래서
  경로별 입력을 만드는 provider는 database마다 하나인 singleton 입력이 가지며,
  provider가 파일을 어디서 읽을지(디스크, 메모리)는 database를 만드는 쪽이 정한다.
  Provider가 없는 database에서는 모든 경로에 파일이 없다. 매니페스트처럼 source가
  아닌 파일도 같은 경로별 입력으로 읽으며, 그 입력에서는 텍스트만 읽는다.
- 트리 query는 본문 없는 `mod`의 파일이 없을 때, 본문 있는 `mod`와 같은 이름의
  파일이 있을 때, 한 파일에 트리가 두 번 닿을 때 그 `mod` 선언에 오류를 내고 그
  모듈의 본문을 비운다. 그래서 이름 해석이 받는 패키지 AST에는 본문 없는 모듈이
  없다.

**매니페스트와 패키지 입력.** 패키지 그래프도 `tribute-front`의 query가 경로별
파일 입력에서 만든다([매니페스트](modules.md#매니페스트-tributetoml)).

- 매니페스트 하나의 해석은 그 파일 입력을 key로 하는 query이며, TOML 구문 오류와
  매니페스트 자신의 오류를 그 매니페스트의 위치로 보고한다.
- 의존성 항목을 확인하는 query는 각 항목의 오류를 그 항목의 위치로 보고하고, 적재할
  수 있는 의존 패키지의 매니페스트 목록만 돌려준다. 패키지 이름도 위치 없이 따로
  읽는다. 라이브러리 패키지 입력과 의존성 표 입력은 매니페스트 파일 입력과 암묵적
  의존성 표(`std`)를 key로 하는 query가 이 두 결과로만 만든다. 그래서 위치만 바꾸는
  매니페스트 편집은 패키지 입력을 다시 만들지 않으며, 의존성이 바뀐 매니페스트는 그
  패키지와 그 패키지에 의존하는 패키지의 입력만 다시 만든다.
- 순환은 의존성을 따라가기 전에 매니페스트 해석 결과만으로 확인한다. 의존 패키지가
  의존하는 쪽 매니페스트에 닿는 항목은 따라가지 않으므로, 패키지 입력을 만드는
  query는 서로를 순환 호출하지 않는다.
- 같은 디렉토리를 가리키는 의존성은 같은 매니페스트 경로가 되므로 같은 query 결과,
  즉 같은 패키지 입력을 받는다.
- 실행 파일 루트의 의존성 표는 frontend가 source 파일의 경로로 정하지 않고, 그
  source를 루트로 고른 도구가 정한다. Database마다 하나인 singleton 입력이 루트
  source마다 그 패키지의 매니페스트 파일 입력을 가지며, 등록된 루트는 그
  매니페스트의 의존성 표로 분석한다. CLI는 패키지 디렉토리를 컴파일할 때 그
  `src/main.trb`를 등록하고 source 파일 입력은 등록하지 않으므로, 등록되지 않은
  루트는 암묵적 의존성 표로 분석한다. Language server는 같은 입력에서 경로로 찾기를
  켜며, 등록되지 않은 루트는 경로가 정하는 패키지의 의존성 표로 분석한다. 이 찾기는
  매니페스트 후보 경로의 입력을 읽는 query이므로, 매니페스트가 생기거나 사라지면
  다시 계산된다([Language Server](lsp.md#문서가-속한-패키지)).

**빌드 입력 파일.** 빌드가 읽는 파일 목록은 파일 시스템을 훑어 만들지 않고, 위
query가 경로별 입력에 물어본 경로에서 만든다. 목록에는 아직 없는 경로도 들어간다.
그 경로에 파일이 생기면 빌드 결과가 바뀌기 때문이다. 그래서 CLI는 이 목록을
출력하고, language server는 이 목록을 감시한다([Language Server](lsp.md#입력과-무효화)).

- Source 트리는 자신이 물어본 경로를 모두 가진다. Root 파일, 본문 없는 `mod`의
  파일, 그리고 본문 있는 `mod`와 같은 위치의 파일 경로다. 마지막 경로는 파일이
  있으면 오류이므로, 없어도 트리의 입력이다.
- 매니페스트 없는 root source의 입력은 그 source 트리가 물어본 경로다.
- 패키지 디렉토리의 입력은 매니페스트, `src/lib.trb`와 `src/main.trb`의 경로,
  그리고 있는 각 루트 파일의 source 트리가 물어본 경로다. 의존성을 포함하면 각
  `[dependencies]` 항목의 매니페스트 경로를 그 파일이 있든 없든 더하고, 있는
  매니페스트마다 그 패키지의 매니페스트, `src/lib.trb` 경로, 라이브러리 source
  트리가 물어본 경로를 같은 방식으로 더한다. 의존 패키지의 `src/main.trb`는
  빌드가 읽지 않으므로 넣지 않는다. 매니페스트는 순환을 보고만 하므로, 의존성은
  이미 방문한 매니페스트를 다시 따라가지 않는다.
- 패키지 하나의 입력은 그 매니페스트 파일 입력과 암묵적 의존성 표를 key로 하는
  query이고, 의존성을 포함한 목록은 패키지마다 이 query의 결과를 합친다. 그래서
  파일 내용만 바뀌어 각 패키지의 목록이 같으면 합친 목록은 다시 계산하지 않는다.
- 목록은 `file:` URI를 decode한 파일 경로이며, 경로 순으로 정렬하고 중복이 없다.
  컴파일러가 내장한 `std`의 prelude는 파일이 아니므로 들어가지 않는다.
- CLI의 `tribute files <입력>`은 `compile`과 같은 입력(패키지 디렉토리나 source
  파일)의 목록을 한 줄에 경로 하나씩 출력한다. 기본으로는 있는 파일만 출력해
  `grep`이나 `find` 같은 도구에 넘길 수 있게 하고, `--candidates`는 아직 없는
  경로도 출력해 외부 감시 도구가 쓸 수 있게 한다. `--deps`는 의존성을 포함한다.

**Source 위치.** 트리는 패키지의 각 파일을 그 파일의 parse 결과와 함께, 패키지
안 module path 순으로 담는다. Root 파일의 경로는 빈 경로이다. 파일 모듈의 파일은
그 모듈의 경로를 module path로 lowering하므로, 그 item은 같은 item을 inline으로
선언했을 때와 같은 소유 item 식별자를 갖고 다른 파일의 item과 겹치지 않는다.
경로 순서에서 한 모듈과 그 아래 모듈의 파일은 연속 구간이므로, 하위 트리의
파일은 이진 탐색으로 찾는다.

Span은 바이트 위치와 그 위치가 속한 source id를 함께 가진다. 파일의 source id는
root와 파일 모듈 모두 그 파일 URI의 hash로, `NodeId`의 source hash와 같은 값이다.
Source id 0은 어느 파일에도 속하지 않는 위치이다. 파일의 구문 진단도 같은 source
id를 가진다. 트리의
span map은 파일들의 span을 하나로 합치지 않는다. 노드의 `NodeId`가 가진 source
hash로 그 파일을 찾아 그 파일의 상대 span과 layout에서 위치를 읽는다. 따라서 한
파일의 item을 옮기면 그 파일의 parse 결과만 바뀌고, 트리의 패키지 AST가 같으면
그 AST와 다른 파일의 위치만 읽는 분석은 다시 실행되지 않는다. 진단을 출력하는 쪽은 source id로
진단의 파일을 찾는다. IR location은 같은 source hash에서 파일 경로를 정한다.

## 함수 단위 type check 재사용

Parse와 이름 해석은 모듈 단위 query이다. Type check는 모듈 수준 함수마다
따로 실행되며, 편집 뒤에는 입력이 바뀐 함수만 다시 검사한다.

- **함수의 입력:** 이름 해석은 모듈 수준 함수마다 그 함수의 identity(`FuncDefId`)로
  식별되는 Salsa 값을 만들고, 그 값이 함수의 resolved 선언을 담는다. 함수 검사
  query는 이 선언과 모듈의 선언 환경만 읽는다. 선언 환경은 모듈의 타입·함수
  시그니처·생성자·ability 선언과 의존 package의 export로 만든 별도 query의
  결과이며, 함수 본문은 담지 않는다.
- **재검사 조건:** 함수 본문만 바뀐 편집은 그 함수의 resolved 선언만 바꾸므로 그
  함수만 다시 검사한다. 시그니처나 타입 선언이 바뀌면 선언 환경이 바뀌어 모든
  함수를 다시 검사한다.
- **위치:** 이름 해석과 type check는 span map을 읽지 않는다. 그 진단은 위치를
  `NodeId`로 가진 채 Salsa accumulator로 보고된다. 위치는 query 밖에서 진단을
  읽는 쪽이 정한다. 그 source의 package와 의존 package의 span map을 함께 보고,
  노드의 source hash로 파일을 찾는다. 따라서 코드를 옮기기만 한 편집은 진단이
  있어도 이름 해석과 함수 검사 결과를 재사용하고, 진단 위치는 현재 source를
  따른다.
- **모듈 결과:** 모듈 query는 각 함수의 typed 선언과 노드 메타데이터를 선언 순서로
  합친다. Specialization과 lowering은 이 모듈 결과를 입력으로 받으며 함수 단위로
  재사용하지 않는다.

## 편집 위치와 무관한 frontend identity

Frontend가 AST와 분석 결과에 담는 identity는 source 위치에 의존하지 않는다. 한
item을 편집해도 다른 item의 resolved·typed 선언은 바뀌지 않으며, 이것이 결과를
item 단위로 재사용하기 위한 전제이다.

- **AST 노드:** `NodeId`는 노드를 소유한 item과 그 item 안에서 astgen이 매긴
  순번으로 정해진다. Item은 module path, 종류, 이름, 같은 module에서 그 item보다
  앞선 같은 종류·이름의 item 수로 식별한다. `fn`, 타입, ability, `mod`, `use`
  선언이 item이며, `use` 선언은 이름 대신 선언 텍스트로 식별한다. 이름이 없는
  선언도 빈 이름의 item이다. Item에 속하지 않는 노드(root module 자신)는 그
  module이 소유한다. 바뀌지 않은 item은 다른 item이
  추가·삭제되거나 위치가 옮겨져도 같은 `NodeId`를 갖는다. 예외는 편집이 그 item보다
  앞선 같은 종류·이름의 item 수를 바꿀 때뿐이다.
- **지역 변수:** `LocalId`는 module 수준 함수 하나 안에서만 유일하며, 각 함수가
  자기 번호를 매긴다.
- **선언:** 함수와 타입 선언의 identity는 package를 포함한 path로 정해진다.
  타입 선언 규칙은 `types.md`의 Nominal Identity를 따른다.
- **위치:** Source 위치는 `NodeId`에서 절대 span으로 가는 별도 span map에만 있고,
  AST와 분석 결과는 위치를 담지 않는다. 진단처럼 위치가 필요한 출력만 span map을
  읽는다. Parse 결과는 각 노드의 span을 소유 item 시작점 기준의 상대
  위치로 담고, 각 item의 시작점과 root module의 span은 별도의 layout에
  담는다. Root module은 파일 전체를 감싸므로 길이가 바뀌는 편집마다 layout도
  바뀐다. 노드의 span은 자기 item 밖의 item을 덮지 않는다. 그래서 inline
  `mod` 선언의 span은 이름까지의 header이고, body의 범위는 syntax tree에서
  얻는다. Span map은 둘을 더해 절대 span을 만든다. 따라서 item을 바꾸지 않고
  옮기면 layout만 바뀐다.
- **Item별 parse 결과:** Parse는 파일 전체를 한 번에 lowering하되, 결과를 item
  하나당 tracked struct 하나로 낸다. 그 identity는 item의 owner key이고,
  선언과 상대 span은 tracked field이다. Inline module의 item은 그 module
  item의 자식으로 두고, module item의 선언은 body를 담지 않는다. 따라서 한
  item의 본문을 편집해도 다른 item의 field는 바뀌지 않고, item 목록은 item이
  추가·삭제되거나 순서가 바뀔 때만 바뀐다. Module 전체 AST와 span map은
  item들로부터 조립하며, 이를 읽는 쪽은 어느 item이 바뀌어도 다시 실행된다.
  Item 단위로 결과를 재사용하려면 그 결과가 내보내는 진단도 소유 item 기준의
  상대 위치로 표현해, 결과를 다시 계산하지 않고도 절대 위치로 옮길 수 있어야 한다.
