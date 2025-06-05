# Tribute GC Design Strategy

## Phase 1: Simple Mark-and-Sweep (MVP)
- 간단한 구현으로 시작
- MLIR에서 GC 런타임 함수 호출
- 학습과 프로토타이핑에 적합

## Phase 2: Reference Counting + Cycle Detection
- 대부분의 경우 즉시 해제
- 주기적으로 순환 참조 검출
- 예측 가능한 성능

## Phase 3: Generational GC (장기)
- 고성능 요구사항 충족
- 복잡한 애플리케이션 지원

## Boxed Value 구조체 설계:

```c
typedef enum {
    TYPE_NUMBER,
    TYPE_STRING, 
    TYPE_BOOLEAN,
    TYPE_FUNCTION,
    TYPE_LIST,
    TYPE_NIL
} tribute_type_t;

typedef struct tribute_boxed {
    tribute_type_t type;
    uint32_t ref_count;  // RC용
    uint8_t gc_mark;     // Mark-sweep용
    union {
        int64_t number;
        struct {
            char* data;
            size_t length;
        } string;
        struct tribute_boxed* list_head;
        void* function_ptr;
    } value;
} tribute_boxed_t;
```

## MLIR Runtime 함수들:

```mlir
// 할당
func.func @tribute_alloc(%size: i64) -> !llvm.ptr<i8>

// 참조 카운트
func.func @tribute_retain(!llvm.ptr<i8>) -> !llvm.ptr<i8>
func.func @tribute_release(!llvm.ptr<i8>)

// GC 트리거
func.func @tribute_gc_collect()

// Boxing/Unboxing
func.func @tribute_box_number(i64) -> !llvm.ptr<i8>
func.func @tribute_unbox_number(!llvm.ptr<i8>) -> i64
```