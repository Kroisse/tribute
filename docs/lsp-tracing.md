# LSP Tracing Integration

Tribute LSP 서버는 `tracing` 크레이트와 통합되어 있어, 에디터의 LSP 로그 패널에서 디버그 로그를 확인할 수 있습니다.

## 사용 방법

### 1. 코드에서 tracing 매크로 사용

```rust
use tracing::{debug, info, warn, error};

// 간단한 로그
tracing::info!("LSP server starting");

// 구조화된 로그 (필드 포함)
tracing::debug!(
    line = position.line,
    character = position.character,
    "Hover request"
);

// Debug 포맷팅 (?는 Debug, %는 Display)
tracing::debug!(uri = ?uri, "Document opened");
tracing::debug!(type_str = %type_str, "Found type");
```

### 2. 로그 레벨 설정

`--log` CLI 옵션으로 로그 레벨을 제어할 수 있습니다:

```bash
# 모든 DEBUG 로그 표시
tribute --log debug serve

# Tribute 관련 로그만 표시
tribute --log tribute=debug serve

# 특정 모듈만 표시
tribute --log tribute::lsp=trace serve

# 여러 설정 조합
tribute --log "tribute=debug,salsa=warn" serve

# 기본값은 warn
tribute serve  # --log warn과 동일
```

참고: `--log` 옵션은 전역 옵션이므로 모든 서브커맨드에서 사용할 수 있습니다:

```bash
tribute --log debug compile file.trb
tribute --log trace debug file.trb
```

### 3. 에디터에서 로그 확인

#### VS Code
1. 출력(Output) 패널 열기: `Cmd+Shift+U` (macOS) 또는 `Ctrl+Shift+U` (Windows/Linux)
2. 드롭다운에서 "Tribute Language Server" 선택

#### Zed
1. 커맨드 팔레트 열기: `Cmd+Shift+P`
2. "lsp: open log" 검색 및 실행

#### Neovim (nvim-lspconfig)
```vim
:LspLog
```

## 구현 상세

### 아키텍처

```
tracing 매크로 호출
    ↓
LspLayer (커스텀 tracing_subscriber::Layer)
    ↓
crossbeam_channel (connection.sender)
    ↓
LSP window/logMessage notification
    ↓
에디터 로그 패널
```

`LspLayer`는 tracing 이벤트를 받아서 LSP의 `window/logMessage` notification으로 변환합니다.
`crossbeam-channel`을 사용하여 메시지를 LSP connection으로 전송합니다 (lsp-server가 이미 사용 중).

### 로그 레벨 매핑

| tracing Level | LSP MessageType |
|---------------|-----------------|
| ERROR         | ERROR           |
| WARN          | WARNING         |
| INFO          | INFO            |
| DEBUG/TRACE   | LOG             |

## 예제

### 기본 사용
```rust
tracing::info!("Server started");
tracing::debug!("Processing request");
tracing::warn!("Deprecated feature used");
tracing::error!("Failed to compile");
```

### 구조화된 로깅
```rust
// 필드와 함께 로그
tracing::debug!(
    file = ?uri,
    line = position.line,
    "Type information requested"
);

// 계산된 값 포함
tracing::debug!(
    symbol_count = symbols.len(),
    "Found symbols"
);
```

### 성능 측정
```rust
use tracing::instrument;

#[instrument(skip(db))]
fn compile_module(db: &dyn Database, uri: &Uri) -> Result<Module> {
    // 함수 호출이 자동으로 로그됨
    // 진입/종료 시간도 기록 가능
}
```

## 추가 정보

### 의존성

- `tracing`: 구조화된 로깅 프레임워크
- `tracing-subscriber`: subscriber 구현 및 레이어 시스템
- `crossbeam-channel`: 스레드 간 메시지 전송 (lsp-server가 이미 사용 중)

### 성능 고려사항

- 로그 메시지는 비동기적으로 전송되므로 메인 로직을 차단하지 않습니다
- `RUST_LOG` 환경 변수로 런타임에 로그 레벨 조정 가능
- 프로덕션에서는 `RUST_LOG=info` 또는 `warn` 권장
