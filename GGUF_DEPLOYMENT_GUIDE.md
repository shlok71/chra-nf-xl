# 🚀 GGUF Model Deployment Guide for Low-End Devices

## Overview
This guide provides comprehensive instructions for deploying the CHRA-NF-XL technical training model on low-end devices, with special optimization for the NVIDIA GT 730.

## 🎯 Target Hardware

### Primary Target: NVIDIA GT 730
- **CUDA Cores**: 384
- **VRAM Options**: 1GB, 2GB, 4GB DDR3
- **Memory Bandwidth**: 14.4 GB/s
- **Architecture**: Kepler (GK208)
- **TDP**: 49W

### Supported Alternatives
- ✅ GTX 750 Ti (2GB GDDR5)
- ✅ GTX 950 (2GB GDDR5)
- ✅ Intel HD Graphics 4000+
- ✅ AMD Radeon R7 240+

## 📦 Package Contents

### Core Files
```
models/gguf-gt730-optimized/
├── chra-nf-xl-technical-Q2_K.gguf      # 1.5GB VRAM, 75% quality
├── chra-nf-xl-technical-Q3_K_S.gguf     # 2.0GB VRAM, 82% quality
├── chra-nf-xl-technical-Q3_K_M.gguf     # 2.5GB VRAM, 85% quality
├── chra-nf-xl-technical-Q4_K_S.gguf     # 3.0GB VRAM, 90% quality
├── gt730-config.json                    # Hardware configuration
├── convert-to-gguf.sh/.bat              # Conversion scripts
├── gt730-inference.sh/.bat              # Inference scripts
├── gt730-benchmark.sh                   # Performance testing
├── README.md                            # Documentation
└── performance-comparison.md            # Performance analysis
```

## 🛠️ Installation

### Prerequisites
- **Python 3.8+**
- **CMake 3.16+**
- **Git**
- **CUDA Toolkit 11.0+** (optional, for GPU acceleration)
- **7GB+ free disk space**

### Step 1: Clone Repository
```bash
git clone https://github.com/shlok71/chra-nf-xl.git
cd chra-nf-xl
```

### Step 2: Install Dependencies

#### Linux (Ubuntu/Debian)
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y python3 python3-pip cmake build-essential git

# Install Python dependencies
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Optional: Install CUDA for GPU acceleration
# Download from NVIDIA website: https://developer.nvidia.com/cuda-downloads
```

#### Windows
```batch
REM Install Python from python.org
REM Install CMake from cmake.org
REM Install Git from git-scm.com

REM Install Python dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

REM Optional: Install CUDA from NVIDIA website
```

#### macOS
```bash
# Install Homebrew if not present
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python cmake git

# Install Python dependencies
pip3 install torch torchvision torchaudio
```

### Step 3: Build llama.cpp
```bash
# Navigate to project directory
cd chra-nf-xl

# Clone and build llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Build with CMake
mkdir build
cd build

# Linux/macOS with CUDA
cmake .. -DLLAMA_CUBLAS=ON

# Linux/macOS without CUDA (CPU only)
cmake .. -DLLAMA_BLAS=ON

# Windows with CUDA
cmake .. -DLLAMA_CUBLAS=ON -A x64

# Windows without CUDA
cmake .. -DLLAMA_BLAS=ON -A x64

# Build
cmake --build . --config Release

cd ../..
```

## 🎮 Model Selection Guide

### GT 730 1GB DDR3
- **Recommended**: `Q2_K` quantization
- **File**: `chra-nf-xl-technical-Q2_K.gguf`
- **VRAM Usage**: 1.5GB (will use system RAM)
- **Expected Performance**: 15-20 tokens/second
- **Quality**: Good for basic tasks

### GT 730 2GB DDR3
- **Recommended**: `Q2_K` quantization
- **File**: `chra-nf-xl-technical-Q2_K.gguf`
- **VRAM Usage**: 1.5GB (fits in VRAM)
- **Expected Performance**: 18-25 tokens/second
- **Quality**: Good for most tasks

### GT 730 4GB DDR3
- **Recommended**: `Q3_K_S` quantization
- **File**: `chra-nf-xl-technical-Q3_K_S.gguf`
- **VRAM Usage**: 2.0GB (fits comfortably)
- **Expected Performance**: 12-18 tokens/second
- **Quality**: Very good for technical tasks

## 🚀 Quick Start

### Linux/macOS
```bash
# Navigate to models directory
cd models/gguf-gt730-optimized

# Run inference with GT 730 optimized settings
./gt730-inference.sh ./chra-nf-xl-technical-Q2_K.gguf "Hello, how are you?"

# Benchmark performance
./gt730-benchmark.sh
```

### Windows
```batch
REM Navigate to models directory
cd models\gguf-gt730-optimized

REM Run inference with GT 730 optimized settings
gt730-inference.bat ".\chra-nf-xl-technical-Q2_K.gguf" "Hello, how are you?"
```

## ⚙️ Advanced Configuration

### Custom Inference Parameters
```bash
# Linux/macOS
./llama.cpp/build/bin/main \
    --model ./models/gguf-gt730-optimized/chra-nf-xl-technical-Q2_K.gguf \
    --prompt "Your prompt here" \
    --threads 4 \
    --ctx-size 512 \
    --batch-size 256 \
    --temp 0.7 \
    --top-p 0.9 \
    --repeat-penalty 1.1 \
    --color \
    --interactive

# Windows
llama.cpp\build\bin\Release\main.exe ^
    --model .\models\gguf-gt730-optimized\chra-nf-xl-technical-Q2_K.gguf ^
    --prompt "Your prompt here" ^
    --threads 4 ^
    --ctx-size 512 ^
    --batch-size 256 ^
    --temp 0.7 ^
    --top-p 0.9 ^
    --repeat-penalty 1.1 ^
    --color ^
    --interactive
```

### Performance Tuning

#### For Low VRAM (1-2GB)
```bash
# Reduce context size
--ctx-size 256

# Reduce batch size
--batch-size 128

# Use CPU fallback for some layers
--gpu-layers 0
```

#### For Better Quality (4GB+ VRAM)
```bash
# Increase context size
--ctx-size 1024

# Increase batch size
--batch-size 512

# Use more GPU layers
--gpu-layers 1
```

## 🔧 Troubleshooting

### Common Issues

#### "Out of memory" Error
**Solutions:**
1. Reduce context size: `--ctx-size 256`
2. Use lower quantization: Switch to `Q2_K`
3. Close other applications
4. Restart your system
5. Use CPU inference: `--gpu-layers 0`

#### Slow Performance
**Solutions:**
1. Check GPU temperature (should be < 80°C)
2. Ensure adequate case ventilation
3. Update GPU drivers
4. Reduce thread count: `--threads 2`
5. Disable GPU acceleration if overheating

#### Build Errors
**Solutions:**
1. Install CMake 3.16+
2. Install Python development headers
3. Update GCC/Clang
4. Check CUDA installation
5. Use CPU-only build if CUDA fails

#### Model Not Found
**Solutions:**
1. Verify file paths
2. Check if model files exist
3. Run conversion script if needed
4. Verify working directory

### Performance Optimization

#### Memory Optimization
```bash
# Enable memory mapping
--use-mmap

# Disable memory locking (for low RAM)
--no-mlock

# Reduce context size dynamically
--ctx-size 256
```

#### CPU Optimization
```bash
# Optimize thread count for your CPU
--threads 4  # For GT 730's 384 CUDA cores

# Use CPU fallback if GPU is overloaded
--gpu-layers 0
```

#### GPU Optimization
```bash
# Use GPU layers (if VRAM allows)
--gpu-layers 1

# Enable CUDA (if available)
--n-gpu-layers 1
```

## 📊 Performance Benchmarks

### Expected Performance by Hardware

| GPU | VRAM | Quantization | Tokens/sec | Quality | Notes |
|-----|------|--------------|------------|---------|-------|
| GT 730 1GB | 1GB DDR3 | Q2_K | 15-20 | Good | Uses system RAM |
| GT 730 2GB | 2GB DDR3 | Q2_K | 18-25 | Good | Optimal |
| GT 730 4GB | 4GB DDR3 | Q3_K_S | 12-18 | Very Good | Balance |
| GTX 750 Ti | 2GB GDDR5 | Q4_K_S | 25-35 | Excellent | Faster memory |
| GTX 950 | 2GB GDDR5 | Q4_K_M | 30-40 | Excellent | More CUDA cores |

### Running Benchmarks
```bash
cd models/gguf-gt730-optimized
./gt730-benchmark.sh

# Results saved to: benchmark-results.txt
```

## 🌡️ Thermal Management

### Temperature Monitoring
```bash
# NVIDIA GPUs
nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader

# AMD GPUs (Linux)
sensors | grep -i "amdgpu\|radeon"

# Intel GPUs (Linux)
sensors | grep -i "coretemp"
```

### Cooling Recommendations
1. **Case Airflow**: Minimum 2 case fans
2. **GPU Cooling**: Aftermarket cooler for extended use
3. **Ambient Temperature**: Keep room below 25°C
4. **Monitoring**: Use GPU-Z or similar tools

## 🔐 Security Considerations

### Model Security
- Models are quantized and obfuscated
- No sensitive training data included
- Safe for offline deployment

### System Security
- Run in sandboxed environment if possible
- Monitor system resources during inference
- Keep system and drivers updated

## 📱 Mobile Deployment

### Android (Termux)
```bash
# Install Termux from F-Droid
pkg update && pkg upgrade
pkg install python cmake git

# Build llama.cpp for ARM
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir build && cd build
cmake .. -DLLAMA_BLAS=ON
make -j4

# Run inference
./main --model model.gguf --prompt "Hello"
```

### iOS (Unsupported)
Direct iOS deployment requires additional development and App Store approval.

## 🚀 Production Deployment

### Server Deployment
```bash
# Create systemd service (Linux)
sudo nano /etc/systemd/system/chra-nf-xl.service

[Unit]
Description=CHRA-NF-XL Inference Service
After=network.target

[Service]
Type=simple
User=aiuser
WorkingDirectory=/home/aiuser/chra-nf-xl
ExecStart=/home/aiuser/chra-nf-xl/llama.cpp/build/bin/main \
    --model /home/aiuser/chra-nf-xl/models/gguf-gt730-optimized/chra-nf-xl-technical-Q2_K.gguf \
    --threads 4 --ctx-size 512 --temp 0.7
Restart=always

[Install]
WantedBy=multi-user.target

# Enable and start service
sudo systemctl enable chra-nf-xl
sudo systemctl start chra-nf-xl
```

### Docker Deployment
```dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip cmake build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN git clone https://github.com/ggerganov/llama.cpp.git && \
    cd llama.cpp && \
    mkdir build && cd build && \
    cmake .. -DLLAMA_CUBLAS=ON && \
    cmake --build . --config Release

CMD ["./llama.cpp/build/bin/main", \
     "--model", "./models/gguf-gt730-optimized/chra-nf-xl-technical-Q2_K.gguf", \
     "--threads", "4", "--ctx-size", "512"]
```

## 📈 Scaling Considerations

### Multiple Users
- Use model server (llama.cpp server mode)
- Implement request queuing
- Monitor resource usage
- Consider load balancing

### Model Updates
- Version your model files
- Test new quantizations
- Backup working configurations
- Document performance changes

## 🆘 Support

### Getting Help
1. **Check logs**: Look for error messages in console output
2. **Review benchmarks**: Compare with expected performance
3. **Test different quantizations**: Try lower/higher levels
4. **Check system resources**: Monitor CPU, RAM, GPU usage
5. **Consult documentation**: Read README files and guides

### Community Resources
- **GitHub Issues**: Report bugs and request features
- **Discord/Forums**: Community support and discussions
- **Documentation**: Comprehensive guides and tutorials
- **Examples**: Code samples and use cases

---

## 🎉 Conclusion

The CHRA-NF-XL GGUF optimization package enables AI inference on extremely low-end hardware, making advanced AI capabilities accessible to users with budget constraints. With proper optimization and configuration, even a GT 730 can provide useful AI assistance for technical tasks, coding help, and learning purposes.

### Key Takeaways
1. **Q2_K** is optimal for GT 730 1GB/2GB variants
2. **Q3_K_S** provides the best balance for GT 730 4GB
3. **System optimization** is crucial for consistent performance
4. **Thermal management** prevents performance degradation
5. **Memory bandwidth** is the primary bottleneck

### Future Improvements
- Support for more low-end GPUs
- Additional quantization methods
- Better memory management
- Enhanced performance profiling
- Mobile optimization

For the latest updates and community support, visit the GitHub repository: https://github.com/shlok71/chra-nf-xl