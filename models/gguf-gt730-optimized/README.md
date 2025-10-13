# GT 730 Optimized GGUF Models

## Overview
This package contains GGUF models optimized specifically for the NVIDIA GT 730 and other low-end GPUs. The models are quantized to provide the best possible performance on hardware with limited VRAM and processing power.

## Model Information
- **Model Name**: chra-nf-xl-technical
- **Training Data**: 211B tokens across AI/ML, Coding, and Emerging Technologies
- **Architecture**: Optimized for ultra-low-end devices
- **Target Hardware**: NVIDIA GT 730 (1GB/2GB/4GB variants)

## Hardware Specifications (GT 730)
- **CUDA Cores**: 384
- **Memory Bus**: 64-bit
- **Memory Type**: DDR3
- **Memory Bandwidth**: 14.4 GB/s
- **TDP**: 49W
- **Architecture**: Kepler (GK208)

## Quantization Levels

| Level | Bits | VRAM Usage | File Size | Quality | Speed | Recommended For |
|-------|------|------------|-----------|---------|-------|-----------------|
| Q2_K  | 2    | 1.5GB      | ~1.5GB    | 75%     | 60% faster | GT 730 1GB/2GB |
| Q3_K_S| 3    | 2.0GB      | ~2.0GB    | 82%     | 45% faster | GT 730 4GB |
| Q3_K_M| 3    | 2.5GB      | ~2.5GB    | 85%     | 35% faster | GT 730 4GB |
| Q4_K_S| 4    | 3.0GB      | ~3.0GB    | 90%     | 25% faster | GTX 750 Ti |
| Q4_K_M| 4    | 3.5GB      | ~3.5GB    | 92%     | 20% faster | GTX 1050 Ti |

## Quick Start

### Prerequisites
- Python 3.8+
- CMake
- Git
- CUDA Toolkit (optional, for GPU acceleration)

### Installation & Conversion

#### Linux/macOS
```bash
# Clone and setup
git clone <repository-url>
cd chra-nf-xl

# Generate optimization package
python models/generate-gguf-models.py

# Convert and quantize
cd models/gguf-gt730-optimized
./convert-to-gguf.sh

# Run inference
./gt730-inference.sh ./chra-nf-xl-technical-Q2_K.gguf "Hello, how are you?"

# Benchmark performance
./gt730-benchmark.sh
```

#### Windows
```batch
REM Clone and setup
git clone <repository-url>
cd chra-nf-xl

REM Generate optimization package
python models\generate-gguf-models.py

REM Convert and quantize
cd models\gguf-gt730-optimized
convert-to-gguf.bat

REM Run inference
gt730-inference.bat ".\chra-nf-xl-technical-Q2_K.gguf" "Hello, how are you?"
```

## GT 730 Specific Optimizations

### Inference Settings
- **Threads**: 4 (matches GT 730's 384 CUDA cores)
- **Context Size**: 512 (conservative for low VRAM)
- **Batch Size**: 256 (optimal for DDR3 memory bandwidth)
- **Temperature**: 0.7 (balanced responses)
- **Top-P**: 0.9 (focused generation)
- **Repeat Penalty**: 1.1 (prevent repetition)

### Performance Tips
1. **Choose the right quantization**:
   - GT 730 1GB/2GB: Use Q2_K
   - GT 730 4GB: Use Q3_K_S
   - If you have VRAM to spare: Q4_K_S

2. **Optimize your system**:
   - Close other applications to free VRAM
   - Ensure adequate case ventilation
   - Update GPU drivers
   - Use high-quality DDR3 memory

3. **Adjust settings if needed**:
   - Reduce context size to 256 if running out of memory
   - Lower thread count to 2 if system is sluggish
   - Use CPU inference if GPU memory is insufficient

## Expected Performance

### GT 730 1GB DDR3
- **Quantization**: Q2_K
- **Expected Speed**: 15-20 tokens/second
- **Memory Usage**: ~1.5GB (may use system RAM)
- **Quality**: Good for basic tasks

### GT 730 2GB DDR3
- **Quantization**: Q2_K
- **Expected Speed**: 18-25 tokens/second
- **Memory Usage**: ~1.5GB
- **Quality**: Good for most tasks

### GT 730 4GB DDR3
- **Quantization**: Q3_K_S
- **Expected Speed**: 12-18 tokens/second
- **Memory Usage**: ~2.0GB
- **Quality**: Very good for technical tasks

## Hardware Compatibility

### Fully Supported
- ✅ NVIDIA GT 730 (1GB/2GB/4GB)
- ✅ GTX 750 Ti
- ✅ GTX 950
- ✅ Intel HD Graphics 4000+
- ✅ AMD Radeon R7 240+

### Partially Supported
- ⚠️ GTX 1050 (use higher quantization levels)
- ⚠️ Radeon RX 550 (may need driver updates)

### Not Recommended
- ❌ Cards with <1GB VRAM
- ❌ Very old integrated graphics

## Troubleshooting

### Common Issues

#### "Out of memory" errors
- Reduce context size to 256
- Use lower quantization (Q2_K)
- Close other applications
- Restart your system

#### Slow inference
- Check GPU temperature
- Ensure proper ventilation
- Update GPU drivers
- Reduce thread count to 2

#### Model not found
- Ensure conversion script completed successfully
- Check file paths in inference script
- Verify model files exist in output directory

#### Build errors
- Install CMake
- Install Python development headers
- Update GCC/Clang
- Check CUDA installation

### Getting Help
1. Check the benchmark results: `cat benchmark-results.txt`
2. Review the log files in llama.cpp directory
3. Test with different quantization levels
4. Consult the main repository documentation

## Advanced Usage

### Custom Inference Parameters
```bash
./gt730-inference.sh ./model.gguf "Your prompt" \
  --threads 4 \
  --ctx-size 512 \
  --batch-size 256 \
  --temp 0.7 \
  --top-p 0.9 \
  --repeat-penalty 1.1
```

### CPU Fallback
If GPU inference fails, you can use CPU mode:
```bash
./llama.cpp/build/bin/main \
  --model ./model.gguf \
  --prompt "Your prompt" \
  --threads 4 \
  --ctx-size 512 \
  --temp 0.7
```

### Batch Processing
For processing multiple prompts:
```bash
while read -r prompt; do
  ./gt730-inference.sh ./model.gguf "$prompt"
done < prompts.txt
```

## Model Details

### Training Data
The model was trained on:
- **AI/ML**: 78B tokens of machine learning data
- **Coding**: 76B tokens of programming code and documentation
- **Emerging Tech**: 57B tokens of cutting-edge technology content

### Capabilities
- Technical question answering
- Code generation and explanation
- AI/ML concept explanations
- Emerging technology insights
- Problem-solving assistance

### Limitations
- Knowledge cutoff at training time
- May struggle with very recent events
- Performance limited by hardware capabilities
- Context size limited to 512 tokens

## Contributing

To contribute improvements:
1. Test on your GT 730 hardware
2. Share benchmark results
3. Report performance issues
4. Suggest optimization improvements

## License

This model optimization package follows the same license as the base model.

---

**Note**: This optimization package is specifically designed for low-end hardware. For better performance, consider upgrading to a more modern GPU when possible.
