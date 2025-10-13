#!/bin/bash
# GT 730 Optimized Inference Script

MODEL_PATH="$1"
PROMPT="$2"

if [ -z "$MODEL_PATH" ] || [ -z "$PROMPT" ]; then
    echo "Usage: $0 <model_path> <prompt>"
    echo "Example: $0 ./models/gguf-gt730-optimized/chra-nf-xl-technical-Q2_K.gguf 'Hello, how are you?'"
    exit 1
fi

echo "🚀 GT 730 Optimized Inference Starting..."
echo "📊 Model: $MODEL_PATH"
echo "💬 Prompt: $PROMPT"

# Check if model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model file not found: $MODEL_PATH"
    exit 1
fi

# GT 730 optimized settings
THREADS=4
CTX_SIZE=512
BATCH_SIZE=256
TEMP=0.7
TOP_P=0.9
REPEAT_PENALTY=1.1

echo "⚙️  Settings:"
echo "   🧵 Threads: $THREADS"
echo "   📝 Context Size: $CTX_SIZE"
echo "   📦 Batch Size: $BATCH_SIZE"
echo "   🌡️  Temperature: $TEMP"
echo "   🎯 Top-P: $TOP_P"

# Run inference
if [ -f "llama.cpp/build/bin/main" ]; then
    ./llama.cpp/build/bin/main \
        --model "$MODEL_PATH" \
        --prompt "$PROMPT" \
        --threads "$THREADS" \
        --ctx-size "$CTX_SIZE" \
        --batch-size "$BATCH_SIZE" \
        --temp "$TEMP" \
        --top-p "$TOP_P" \
        --repeat-penalty "$REPEAT_PENALTY" \
        --color \
        --interactive
elif [ -f "llama.cpp/main" ]; then
    ./llama.cpp/main \
        --model "$MODEL_PATH" \
        --prompt "$PROMPT" \
        --threads "$THREADS" \
        --ctx-size "$CTX_SIZE" \
        --batch-size "$BATCH_SIZE" \
        --temp "$TEMP" \
        --top-p "$TOP_P" \
        --repeat-penalty "$REPEAT_PENALTY" \
        --color \
        --interactive
else
    echo "❌ llama.cpp main executable not found"
    echo "Please run the conversion script first: ./convert-to-gguf.sh"
    exit 1
fi

echo "✅ Inference completed!"
