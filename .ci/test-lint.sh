#!/bin/sh

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TEMP_DIR"' EXIT HUP INT TERM

mkdir "$TEMP_DIR/bin"

cat >"$TEMP_DIR/bin/cargo" <<'EOF'
#!/bin/sh

case "$1" in
    fmt)
        exit 0
        ;;
    clippy)
        line=1
        while [ "$line" -le 100 ]; do
            echo "clippy output $line"
            line=$((line + 1))
        done
        echo "CLIPPY_FAILURE_MARKER"
        exit 1
        ;;
    *)
        exit 0
        ;;
esac
EOF

cat >"$TEMP_DIR/bin/npx" <<EOF
#!/bin/sh
touch "$TEMP_DIR/npx-ran"
exit 0
EOF

chmod +x "$TEMP_DIR/bin/cargo" "$TEMP_DIR/bin/npx"

if PATH="$TEMP_DIR/bin:$PATH" "$SCRIPT_DIR/lint.sh" >"$TEMP_DIR/output" 2>&1; then
    echo "lint.sh unexpectedly succeeded" >&2
    exit 1
else
    status=$?
fi

if [ "$status" -ne 2 ]; then
    echo "lint.sh returned $status instead of 2" >&2
    exit 1
fi

if ! grep -q "CLIPPY_FAILURE_MARKER" "$TEMP_DIR/output"; then
    echo "lint.sh did not print the tail of clippy output" >&2
    exit 1
fi

if grep -q "clippy output 1$" "$TEMP_DIR/output"; then
    echo "lint.sh printed unbounded clippy output" >&2
    exit 1
fi

if [ -e "$TEMP_DIR/npx-ran" ]; then
    echo "lint.sh continued after clippy failed" >&2
    exit 1
fi
