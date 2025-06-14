#!/bin/bash

# Performance comparison script for Tribute interpreter vs compiler
# Usage: ./scripts/performance_comparison.sh

set -e

echo "🚀 Tribute Performance Comparison: Interpreter vs Compiler"
echo "========================================================="

# Build the project
echo "📦 Building Tribute..."
cargo build --release > /dev/null 2>&1

# Test files
TEST_FILES=(
    "lang-examples/basic.trb"
    "lang-examples/functions.trb"
    "lang-examples/pattern_matching.trb"
    "lang-examples/performance_test.trb"
    "lang-examples/calc.trb"
)

echo ""
echo "📊 Performance Results:"
echo "======================="

for file in "${TEST_FILES[@]}"; do
    if [ -f "$file" ]; then
        filename=$(basename "$file" .trb)
        echo ""
        echo "🎯 Testing: $filename"
        echo "-------------------"
        
        # Test interpreter
        echo -n "📝 Interpreter: "
        INTERPRETER_TIME=$( { time cargo run --release --bin trbi "$file" > /dev/null 2>&1; } 2>&1 | grep real | awk '{print $2}')
        echo "$INTERPRETER_TIME"
        
        # Test compiler (if compilation succeeds)
        echo -n "⚡ Compiler: "
        if cargo run --release --bin trbc -- --compile "$file" -o "/tmp/tribute_test_$filename" > /dev/null 2>&1; then
            COMPILER_TIME=$( { time cargo run --release --bin trbc -- --compile "$file" -o "/tmp/tribute_test_$filename" > /dev/null 2>&1; } 2>&1 | grep real | awk '{print $2}')
            echo "$COMPILER_TIME"
            
            # Check object file size
            object_file="/tmp/tribute_test_$filename.o"
            if [ -f "$object_file" ]; then
                SIZE=$(stat -f%z "$object_file" 2>/dev/null || stat -c%s "$object_file" 2>/dev/null | numfmt --to=iec)
                echo "📁 Object size: $SIZE"
                rm -f "$object_file"
            fi
        else
            echo "❌ Compilation failed"
        fi
    fi
done

echo ""
echo "🎉 Performance comparison completed!"
echo ""
echo "📋 Summary:"
echo "- Interpreter: Direct HIR evaluation"
echo "- Compiler: Cranelift → object file generation"
echo "- Object files are optimized machine code"
echo ""
echo "💡 Tips:"
echo "- Run 'cargo run --bin trbc -- --test' for comprehensive testing"
echo "- Run 'cargo bench -p tribute-runtime' for detailed TrString benchmarks"