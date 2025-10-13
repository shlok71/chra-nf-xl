#!/bin/bash
# GGUF Conversion Script for GT 730 (Linux/macOS)

MODEL_NAME="chra-nf-xl-technical"
BASE_MODEL="./models/chra-nf-xl-base"
OUTPUT_DIR="./models/gguf-gt730-optimized"

echo "🚀 Converting to GGUF format for GT 730..."
echo "📊 Model: $MODEL_NAME"
echo "📁 Base Model: $BASE_MODEL"
echo "📁 Output Directory: $OUTPUT_DIR"

# Check if llama.cpp is available
if [ ! -d "llama.cpp" ]; then
    echo "📥 Installing llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp.git
    cd llama.cpp
    
    # Try CMake build
    if command -v cmake &> /dev/null; then
        echo "🔨 Building with CMake..."
        mkdir -p build
        cd build
        cmake .. -DLLAMA_CUBLAS=ON
        cmake --build . --config Release
        cd ../..
    else
        echo "⚠️  CMake not found, please install CMake"
        exit 1
    fi
    
    cd ..
    echo "✅ llama.cpp installed"
fi

# Convert base model to GGUF
echo "🔄 Converting base model to GGUF format..."
python llama.cpp/convert.py \
    --outfile "$OUTPUT_DIR/${MODEL_NAME}-base.gguf" \
    --outtype f16 \
    "$BASE_MODEL"

# Quantize for GT 730
echo "⚡ Quantizing for GT 730 optimization..."

quantization_levels=("Q2_K" "Q3_K_S" "Q3_K_M" "Q4_K_S")

for level in "${quantization_levels[@]}"; do
    echo "🎯 Creating $level quantization..."
    
    ./llama.cpp/build/bin/quantize "$OUTPUT_DIR/${MODEL_NAME}-base.gguf" "$OUTPUT_DIR/${MODEL_NAME}-${level}.gguf" ${level}
    
    if [ $? -eq 0 ]; then
        echo "✅ $level completed"
        
        # Show file size
        if [ -f "$OUTPUT_DIR/${MODEL_NAME}-${level}.gguf" ]; then
            SIZE=$(du -h "$OUTPUT_DIR/${MODEL_NAME}-${level}.gguf" | cut -f1)
            echo "📊 File size: $SIZE"
        fi
    else
        echo "❌ $level failed"
    fi
    
    echo ""
done

echo "🎉 GGUF conversion completed!"
echo "📊 Generated files:"
ls -lh "$OUTPUT_DIR/${MODEL_NAME}"*.gguf

echo ""
echo "💡 GT 730 Recommendations:"
echo "   🎮 GT 730 1GB/2GB: Use Q2_K"
echo "   🎮 GT 730 4GB: Use Q3_K_S"
echo ""
echo "🚀 To test inference:"
echo "   ./gt730-inference.sh $OUTPUT_DIR/${MODEL_NAME}-Q2_K.gguf \"Your prompt here\""
