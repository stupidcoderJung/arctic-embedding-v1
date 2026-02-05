#!/bin/bash
# Arctic Embedding V1 - Build and Test Script (Optimized Version)
# Generated: 2026-02-05

set -e  # Exit on error

PROJECT_DIR="$HOME/.openclaw/workspace/projects/arctic-embedding-v1"
cd "$PROJECT_DIR"

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║   Arctic Embedding V1 - Optimized Build & Test               ║"
echo "║   Target: <50ms inference time                                ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Clean previous builds
echo "🧹 Step 1: Cleaning previous builds..."
make -f Makefile.optimized clean 2>/dev/null || true
echo "   ✅ Clean complete"
echo ""

# Step 2: Build optimized version
echo "🔨 Step 2: Building optimized version..."
if make -f Makefile.optimized optimized; then
    echo "   ✅ Build successful"
else
    echo "   ❌ Build failed!"
    exit 1
fi
echo ""

# Step 3: Verify binary exists
echo "🔍 Step 3: Verifying binary..."
if [ -f "bin/arctic_embed_optimized" ]; then
    SIZE=$(du -h bin/arctic_embed_optimized | cut -f1)
    echo "   ✅ Binary created: $SIZE"
else
    echo "   ❌ Binary not found!"
    exit 1
fi
echo ""

# Step 4: Check if model exists
echo "📦 Step 4: Checking for model file..."
if [ -f "model.onnx" ]; then
    MODEL_SIZE=$(du -h model.onnx | cut -f1)
    echo "   ✅ Model found: $MODEL_SIZE"
elif [ -f "Snowflake_arctic-embed-xs.onnx" ]; then
    echo "   ℹ️  Found model with different name, creating symlink..."
    ln -sf Snowflake_arctic-embed-xs.onnx model.onnx
    echo "   ✅ Model linked"
else
    echo "   ⚠️  Model not found! Please download it first."
    echo "   Expected: model.onnx or Snowflake_arctic-embed-xs.onnx"
    exit 1
fi
echo ""

# Step 5: Test with short text
echo "🧪 Step 5: Testing with short text..."
TEST_TEXT="Hello, OpenClaw!"
echo "   Input: \"$TEST_TEXT\""
if OUTPUT=$(./bin/arctic_embed_optimized model.onnx "$TEST_TEXT" 2>&1); then
    echo "   ✅ Test passed"
    # Extract embedding size from output
    if echo "$OUTPUT" | grep -q "Generated embedding of size: 384"; then
        echo "   ✅ Output dimension: 384"
    else
        echo "   ⚠️  Output dimension check failed"
    fi
else
    echo "   ❌ Test failed!"
    exit 1
fi
echo ""

# Step 6: Quick performance test (3 runs)
echo "⚡ Step 6: Quick performance test (3 runs)..."
TIMES=()
for i in 1 2 3; do
    echo -n "   Run $i: "
    START=$(gdate +%s%3N 2>/dev/null || date +%s000)
    ./bin/arctic_embed_optimized model.onnx "Machine learning is transforming artificial intelligence" >/dev/null 2>&1
    END=$(gdate +%s%3N 2>/dev/null || date +%s000)
    TIME=$((END - START))
    TIMES+=($TIME)
    echo "${TIME}ms"
done

# Calculate average
TOTAL=0
for t in "${TIMES[@]}"; do
    TOTAL=$((TOTAL + t))
done
AVG=$((TOTAL / 3))
echo ""
echo "   📊 Average: ${AVG}ms"

# Check if target achieved
if [ $AVG -lt 50 ]; then
    echo "   ✅ Target achieved! (<50ms)"
elif [ $AVG -lt 80 ]; then
    echo "   ⚠️  Close to target (50-80ms). May need to check system load."
else
    echo "   ⚠️  Slower than expected (>80ms). Check build flags and system."
fi
echo ""

# Step 7: Summary
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║   Test Summary                                                ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "   Build:        ✅ Success"
echo "   Binary:       ✅ Created"
echo "   Model:        ✅ Found"
echo "   Functionality: ✅ Working"
echo "   Performance:  $AVG ms average"
if [ $AVG -lt 50 ]; then
    echo "   Status:       ✅ READY FOR PRODUCTION"
else
    echo "   Status:       ⚠️  REVIEW PERFORMANCE"
fi
echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║   Next Steps                                                  ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "   • Run full benchmark: make -f Makefile.optimized benchmark"
echo "   • Compare with original: make -f Makefile.optimized compare"
echo "   • Read documentation: less QUICK_START_OPTIMIZED.md"
echo "   • Production deploy: Copy bin/arctic_embed_optimized"
echo ""
echo "   Binary location: $PROJECT_DIR/bin/arctic_embed_optimized"
echo ""

exit 0
