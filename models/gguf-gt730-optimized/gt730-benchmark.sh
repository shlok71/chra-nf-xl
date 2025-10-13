#!/bin/bash
# GT 730 Performance Benchmark Script

MODEL_DIR="./models/gguf-gt730-optimized"
TEST_PROMPT="The quick brown fox jumps over the lazy dog. Explain this phrase."
RESULTS_FILE="$MODEL_DIR/benchmark-results.txt"

echo "🔬 GT 730 Performance Benchmark" | tee "$RESULTS_FILE"
echo "================================" | tee -a "$RESULTS_FILE"
echo "Date: $(date)" | tee -a "$RESULTS_FILE"
echo "GPU: NVIDIA GT 730" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

# Test each quantization level
for level in Q2_K Q3_K_S Q3_K_M Q4_K_S; do
    MODEL_FILE="$MODEL_DIR/chra-nf-xl-technical-${level}.gguf"
    
    if [ -f "$MODEL_FILE" ]; then
        echo "" | tee -a "$RESULTS_FILE"
        echo "📊 Testing $level..." | tee -a "$RESULTS_FILE"
        echo "📁 File: $MODEL_FILE" | tee -a "$RESULTS_FILE"
        
        # Get file size
        FILE_SIZE=$(du -h "$MODEL_FILE" | cut -f1)
        echo "💾 Size: $FILE_SIZE" | tee -a "$RESULTS_FILE"
        
        # Measure inference time
        START_TIME=$(date +%s.%N)
        
        if [ -f "llama.cpp/build/bin/main" ]; then
            OUTPUT=$(./llama.cpp/build/bin/main \
                --model "$MODEL_FILE" \
                --prompt "$TEST_PROMPT" \
                --threads 4 \
                --ctx-size 512 \
                --batch-size 256 \
                --temp 0.7 \
                --top-p 0.9 \
                --repeat-penalty 1.1 \
                --n-predict 50 \
                --quiet 2>&1)
        else
            echo "❌ llama.cpp main executable not found" | tee -a "$RESULTS_FILE"
            continue
        fi
        
        END_TIME=$(date +%s.%N)
        ELAPSED_TIME=$(echo "$END_TIME - $START_TIME" | bc)
        
        echo "⏱️  Inference Time: ${ELAPSED_TIME}s" | tee -a "$RESULTS_FILE"
        
        # Extract VRAM usage if available
        if command -v nvidia-smi &> /dev/null; then
            VRAM_USAGE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
            echo "🎮 VRAM Usage: ${VRAM_USAGE}MB" | tee -a "$RESULTS_FILE"
        fi
        
        # Calculate tokens per second (rough estimate)
        if [[ $ELAPSED_TIME =~ ^[0-9]+\.?[0-9]*$ ]]; then
            TOKENS_PER_SEC=$(echo "scale=2; 50 / $ELAPSED_TIME" | bc)
            echo "🚀 Tokens/sec: $TOKENS_PER_SEC" | tee -a "$RESULTS_FILE"
        fi
        
        echo "✅ $level benchmark completed" | tee -a "$RESULTS_FILE"
    else
        echo "" | tee -a "$RESULTS_FILE"
        echo "❌ $level file not found: $MODEL_FILE" | tee -a "$RESULTS_FILE"
    fi
done

echo "" | tee -a "$RESULTS_FILE"
echo "🎉 Benchmark completed!" | tee -a "$RESULTS_FILE"
echo "📈 Results saved to: $RESULTS_FILE" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"
echo "💡 Recommendations:" | tee -a "$RESULTS_FILE"
echo "   🎮 GT 730 1GB/2GB: Use Q2_K for best performance" | tee -a "$RESULTS_FILE"
echo "   🎮 GT 730 4GB: Use Q3_K_S for balance of quality and speed" | tee -a "$RESULTS_FILE"
echo "   🔧 If running out of VRAM, try reducing context size to 256" | tee -a "$RESULTS_FILE"
