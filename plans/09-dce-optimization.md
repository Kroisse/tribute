# Dead Code Elimination (DCE) Pass for TrunkIR

## Overview
trunk-ir에 dead code elimination 최적화 pass를 구현합니다. 결과가 사용되지 않고 부작용이 없는 operation을 제거합니다.

## Components

1. **OpInterface 시스템** (`op_interface.rs`) - Operation 속성 조회를 위한 인터페이스
2. **DCE Pass** (`transforms/dce.rs`) - 실제 최적화 구현

## Location
- `crates/trunk-ir/src/op_interface.rs` - OpInterface 시스템
- `crates/trunk-ir/src/transforms/dce.rs` - DCE pass

trunk-ir에 배치합니다. 언어 독립적이고 재사용 가능한 최적화입니다.

---

## Part 1: OpInterface System

`type_interface.rs`와 유사한 패턴으로 operation 속성 조회 시스템을 구현합니다.

### Design Principle (MLIR 방식)

**등록 안 함 = Unknown (제거 불가) - Conservative by default**

Pure operation만 등록합니다. 등록되지 않은 operation은 side effect가 있을 수 있다고 가정하여 DCE에서 제거하지 않습니다.

- 안전성 우선: 등록 누락 시 최적화만 안 됨 (버그 아님)
- Side-effecting op 누락으로 인한 잘못된 코드 생성 방지

### Pure Trait + Registry
```rust
/// Marker trait for pure operations (no side effects, safe to remove if unused)
pub trait Pure: DialectOp<'_> {}

/// Registration for runtime lookup
pub struct PureOpRegistration {
    pub dialect: &'static str,
    pub op_name: &'static str,
}

inventory::collect!(PureOpRegistration);

// Pure operation 등록 (trait impl과 함께)
impl Pure for arith::Add<'_> {}
inventory::submit! { PureOps::register("arith", "add") }

impl Pure for arith::Const<'_> {}
inventory::submit! { PureOps::register("arith", "const") }
// ...
```

### Query API
```rust
impl PureOps {
    /// Operation이 pure인지 확인 (등록된 경우만 true)
    pub fn is_pure(db: &dyn Database, op: &Operation<'_>) -> bool;

    /// DCE로 제거 가능한지 확인 (= is_pure)
    pub fn is_removable(db: &dyn Database, op: &Operation<'_>) -> bool;
}
```

### Pure Operations (등록 필요)

| Dialect | Operations |
|---------|-----------|
| `arith` | `const`, `add`, `sub`, `mul`, `div`, `rem`, `neg`, `cmp_*`, `and`, `or`, `xor`, `shl`, `shr`, `shru`, `cast`, `trunc`, `extend`, `convert` |
| `adt` | `struct_new`, `struct_get`, `variant_new`, `variant_tag`, `variant_get`, `array_new`, `array_get`, `array_len`, `ref_null`, `ref_is_null`, `ref_cast`, `string_const`, `bytes_const` |
| `func` | `constant` |
| `closure` | `new`, `func`, `env` |
| `mem` | `data`, `load` |
| `list` | `new`, `get`, `len`, `view_front`, `view_back`, `set`, `push_front`, `push_back`, `concat`, `slice` |

Side-effecting operations (func.call, mem.store, scf.yield 등)은 등록 불필요 - 기본값이 "제거 불가".

---

## Part 2: DCE Pass

### Algorithm

**Backward Liveness Analysis + Sweep:**

1. **Mark Phase**: 루트 값(terminator, side-effecting op의 operand)에서 시작하여 use-def 체인을 따라 liveness 전파
2. **Sweep Phase**: live로 마킹되지 않은 operation 제거
3. **Fixpoint**: 변경이 없을 때까지 반복

### DCE에서 OpInterface 사용

```rust
fn is_dead(&self, op: &Operation<'db>) -> bool {
    // Pure로 등록되지 않은 operation은 제거 불가
    if !PureOps::is_pure(self.db, op) {
        return false;
    }

    // 결과가 없는 operation은 유지 (side-effect일 수 있음)
    if op.results(self.db).is_empty() {
        return false;
    }

    // 모든 결과가 live하지 않으면 dead
    (0..op.results(self.db).len())
        .all(|i| !self.live_values.contains(&op.result(self.db, i)))
}
```

## API Design

```rust
// crates/trunk-ir/src/transforms/dce.rs

/// DCE 설정
#[derive(Debug, Clone, Default)]
pub struct DceConfig {
    pub max_iterations: usize,  // 기본값: 100
    pub recursive: bool,        // 중첩 region 처리 여부, 기본값: true
}

/// DCE 결과
#[derive(Debug)]
pub struct DceResult<'db> {
    pub module: Module<'db>,
    pub removed_count: usize,
    pub iterations: usize,
    pub reached_fixpoint: bool,
}

/// 메인 API
pub fn eliminate_dead_code<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
) -> DceResult<'db>;

pub fn eliminate_dead_code_with_config<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
    config: DceConfig,
) -> DceResult<'db>;

/// Operation이 제거 가능한지 확인 (부작용 없고 terminator 아님)
pub fn is_removable<'db>(db: &'db dyn salsa::Database, op: &Operation<'db>) -> bool;
```

## Implementation Steps

### Step 1: OpInterface 시스템 생성
- [ ] `crates/trunk-ir/src/op_interface.rs` 생성
- [ ] `Pure` marker trait 정의
- [ ] `PureOpRegistration` 구조체 정의
- [ ] `inventory::collect!` 설정
- [ ] `PureOps` 구조체와 `is_pure()`, `is_removable()` 메서드 구현
- [ ] `crates/trunk-ir/src/lib.rs`에 `pub mod op_interface;` 추가

### Step 2: Pure operation 등록 (side-effecting은 등록 불필요)
- [ ] `dialect/arith.rs` - 모든 연산 (`const`, `add`, `sub`, `mul`, ...)
- [ ] `dialect/adt.rs` - getter/constructor (`struct_new`, `struct_get`, `variant_*`, `array_new`, `array_get`, ...)
- [ ] `dialect/func.rs` - `constant`만
- [ ] `dialect/closure.rs` - `new`, `func`, `env`
- [ ] `dialect/mem.rs` - `data`, `load`
- [ ] `dialect/list.rs` - 모든 연산

### Step 3: transforms 모듈 생성
- [ ] `crates/trunk-ir/src/transforms/mod.rs` 생성
- [ ] `crates/trunk-ir/src/transforms/dce.rs` 생성
- [ ] `crates/trunk-ir/src/lib.rs`에 `pub mod transforms;` 추가

### Step 4: Liveness 분석
- [ ] `DcePass` 구조체 정의
- [ ] `compute_live_values()` - 루트에서 liveness 전파
- [ ] `collect_root_values()` - terminator/side-effecting op의 operand 수집

### Step 5: Sweep 구현
- [ ] `sweep_module()` - 모듈 전체 순회
- [ ] `sweep_region()` - region 순회
- [ ] `sweep_block()` - block 순회, dead op 필터링
- [ ] `is_dead()` - operation의 모든 결과가 dead인지 확인

### Step 6: Fixpoint 루프
- [ ] `run()` - 변경 없을 때까지 mark-sweep 반복
- [ ] `DceResult` 반환

### Step 7: 테스트
- [ ] 미사용 상수 제거 테스트
- [ ] 사용 중인 값 유지 테스트
- [ ] Side-effecting op 유지 테스트
- [ ] 중첩 region 테스트
- [ ] Fixpoint 도달 테스트

## Files to Create/Modify

| File | Action |
|------|--------|
| `crates/trunk-ir/src/op_interface.rs` | **생성** - Pure trait + registry |
| `crates/trunk-ir/src/transforms/mod.rs` | **생성** |
| `crates/trunk-ir/src/transforms/dce.rs` | **생성** - DCE 구현 |
| `crates/trunk-ir/src/lib.rs` | `pub mod op_interface;`, `pub mod transforms;` 추가 |
| `crates/trunk-ir/src/dialect/arith.rs` | Pure ops 등록 (모든 연산) |
| `crates/trunk-ir/src/dialect/adt.rs` | Pure ops 등록 (getter/constructor) |
| `crates/trunk-ir/src/dialect/func.rs` | Pure ops 등록 (`constant`만) |
| `crates/trunk-ir/src/dialect/closure.rs` | Pure ops 등록 (모든 연산) |
| `crates/trunk-ir/src/dialect/mem.rs` | Pure ops 등록 (`data`, `load`) |
| `crates/trunk-ir/src/dialect/list.rs` | Pure ops 등록 (모든 연산) |

Side-effecting dialects (ability, cont, scf 등)은 수정 불필요 - 기본값이 "제거 불가".

## Future Enhancements (이번 구현 범위 외)

1. **Effect-aware DCE**: 함수 정의의 effect 정보를 분석하여 pure 함수 호출도 제거
2. **Interprocedural DCE**: 미사용 함수 정의 제거
3. **Debug info**: 제거된 operation 추적
4. **추가 OpInterface**: `IsTerminator`, `HasNoSideEffects` 등 세분화된 인터페이스
