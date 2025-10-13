/**
 * GGUF Model Optimizer for Low-End Devices
 * Optimized specifically for GT 730 and similar low-spec GPUs
 */

import { execSync } from 'child_process';
import { writeFileSync, mkdirSync, existsSync } from 'fs';
import { join } from 'path';

export interface GGUFConfig {
  modelName: string;
  baseModel: string;
  quantizationLevels: QuantizationLevel[];
  targetDevice: 'gt730' | 'lowend' | 'integrated';
  outputDir: string;
}

export interface QuantizationLevel {
  name: string;
  bits: number;
  description: string;
  estimatedVRAM: string;
  performance: 'ultra-low' | 'low' | 'medium' | 'high';
  recommendedFor: string[];
}

export class GT730Optimizer {
  private config: GGUFConfig;
  
  constructor(config: GGUFConfig) {
    this.config = config;
    this.ensureOutputDirectory();
  }

  private ensureOutputDirectory(): void {
    if (!existsSync(this.config.outputDir)) {
      mkdirSync(this.config.outputDir, { recursive: true });
    }
  }

  /**
   * GT 730 Specific Quantization Levels
   */
  public static readonly GT730_QUANTIZATION_LEVELS: QuantizationLevel[] = [
    {
      name: 'Q2_K',
      bits: 2,
      description: '2-bit quantization - Maximum compression for GT 730',
      estimatedVRAM: '1.5GB',
      performance: 'ultra-low',
      recommendedFor: ['GT 730 2GB', 'GT 730 1GB', 'Integrated Graphics']
    },
    {
      name: 'Q3_K_S',
      bits: 3,
      description: '3-bit small quantization - Balanced for GT 730',
      estimatedVRAM: '2.0GB',
      performance: 'low',
      recommendedFor: ['GT 730 2GB', 'GT 730 4GB', 'Low-end mobile GPUs']
    },
    {
      name: 'Q3_K_M',
      bits: 3,
      description: '3-bit medium quantization - Good quality for GT 730',
      estimatedVRAM: '2.5GB',
      performance: 'low',
      recommendedFor: ['GT 730 4GB', 'GTX 750 Ti', 'Low-end desktop GPUs']
    },
    {
      name: 'Q4_K_S',
      bits: 4,
      description: '4-bit small quantization - Best quality for GT 730',
      estimatedVRAM: '3.0GB',
      performance: 'medium',
      recommendedFor: ['GT 730 4GB', 'GTX 950', 'Entry-level gaming GPUs']
    },
    {
      name: 'Q4_K_M',
      bits: 4,
      description: '4-bit medium quantization - Enhanced quality',
      estimatedVRAM: '3.5GB',
      performance: 'medium',
      recommendedFor: ['GTX 950', 'GTX 1050', 'Mid-range mobile GPUs']
    },
    {
      name: 'Q5_K_S',
      bits: 5,
      description: '5-bit small quantization - High quality (if VRAM allows)',
      estimatedVRAM: '4.0GB',
      performance: 'high',
      recommendedFor: ['GTX 1050 Ti', 'GTX 1650', 'Modern low-end GPUs']
    }
  ];

  /**
   * Create GGUF conversion script for GT 730
   */
  public createConversionScript(): string {
    const script = `#!/bin/bash
# GGUF Conversion Script for GT 730 Optimization
# Generated for ${this.config.modelName}

set -e

MODEL_NAME="${this.config.modelName}"
BASE_MODEL="${this.config.baseModel}"
OUTPUT_DIR="${this.config.outputDir}"

echo "🚀 Starting GGUF conversion for GT 730 optimization..."
echo "📊 Model: $MODEL_NAME"
echo "🎯 Target Device: GT 730"
echo "📁 Output Directory: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if llama.cpp is available
if ! command -v ./llama.cpp/convert.py &> /dev/null; then
    echo "📥 Installing llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp.git
    cd llama.cpp
    make LLAMA_CUBLAS=1
    cd ..
fi

# Convert base model to GGUF
echo "🔄 Converting base model to GGUF format..."
python llama.cpp/convert.py \\
    --outfile "$OUTPUT_DIR/${MODEL_NAME}-base.gguf" \\
    --outtype f16 \\
    "$BASE_MODEL"

# Quantize for GT 730
echo "⚡ Quantizing for GT 730 optimization...";

`;

    // Add quantization commands for each level
    this.config.quantizationLevels.forEach(level => {
      script += `
echo "🎯 Creating ${level.name} quantization (${level.description})..."
./llama.cpp/quantize "$OUTPUT_DIR/${MODEL_NAME}-base.gguf" "$OUTPUT_DIR/${MODEL_NAME}-${level.name}.gguf" ${level.name}

echo "✅ ${level.name} completed - Estimated VRAM: ${level.estimatedVRAM}"
`;
    });

    script += `
echo "🎉 GGUF conversion completed!"
echo "📊 Generated files:"
ls -lh "$OUTPUT_DIR/${MODEL_NAME}"*.gguf

echo "💡 Recommended for GT 730 2GB: Q2_K"
echo "💡 Recommended for GT 730 4GB: Q3_K_S or Q4_K_S"
echo "💡 Usage: ./main -m "$OUTPUT_DIR/${MODEL_NAME}-Q2_K.gguf" -p "Your prompt here"
`;

    return script;
  }

  /**
   * Create GT 730 specific inference script
   */
  public createInferenceScript(): string {
    return `#!/bin/bash
# GT 730 Optimized Inference Script
# Generated for ${this.config.modelName}

MODEL_PATH="$1"
PROMPT="$2"

if [ -z "$MODEL_PATH" ] || [ -z "$PROMPT" ]; then
    echo "Usage: $0 <model_path> <prompt>"
    echo "Example: $0 ./models/chra-nf-xl-Q2_K.gguf 'Hello, how are you?'"
    exit 1
fi

echo "🚀 Starting GT 730 optimized inference..."
echo "📊 Model: $MODEL_PATH"
echo "💬 Prompt: $PROMPT"

# GT 730 optimized settings
THREADS=4        # GT 730 has limited CUDA cores
CTX_SIZE=512     # Smaller context for low VRAM
BATCH_SIZE=512   # Smaller batch size
TEMP=0.7         # Balanced temperature
TOP_P=0.9        # Conservative sampling

./llama.cpp/main \\
    --model "$MODEL_PATH" \\
    --prompt "$PROMPT" \\
    --threads "$THREADS" \\
    --ctx-size "$CTX_SIZE" \\
    --batch-size "$BATCH_SIZE" \\
    --temp "$TEMP" \\
    --top-p "$TOP_P" \\
    --repeat-penalty 1.1 \\
    --color \\
    --interactive

echo "✅ Inference completed!"
`;
  }

  /**
   * Create performance benchmark script
   */
  public createBenchmarkScript(): string {
    return `#!/bin/bash
# GT 730 Performance Benchmark Script
# Tests performance across different quantization levels

MODEL_DIR="${this.config.outputDir}"
TEST_PROMPT="The quick brown fox jumps over the lazy dog. Explain this phrase."

echo "🔬 GT 730 Performance Benchmark"
echo "================================"

# Test each quantization level
for level in Q2_K Q3_K_S Q3_K_M Q4_K_S Q4_K_M Q5_K_S; do
    MODEL_FILE="$MODEL_DIR/${this.config.modelName}-${level}.gguf"
    
    if [ -f "$MODEL_FILE" ]; then
        echo ""
        echo "📊 Testing $level..."
        echo "📁 File: $MODEL_FILE"
        
        # Measure inference time
        START_TIME=$(date +%s.%N)
        
        ./llama.cpp/main \\
            --model "$MODEL_FILE" \\
            --prompt "$TEST_PROMPT" \\
            --threads 4 \\
            --ctx-size 512 \\
            --batch-size 512 \\
            --temp 0.7 \\
            --top-p 0.9 \\
            --repeat-penalty 1.1 \\
            --n-predict 50 \\
            --quiet \\
            --log-file /tmp/benchmark_$level.log
        
        END_TIME=$(date +%s.%N)
        ELAPSED_TIME=$(echo "$END_TIME - $START_TIME" | bc)
        
        echo "⏱️  Inference Time: ${ELAPSED_TIME}s"
        echo "💾 File Size: $(du -h "$MODEL_FILE" | cut -f1)"
        
        # Extract VRAM usage if available
        if command -v nvidia-smi &> /dev/null; then
            VRAM_USAGE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
            echo "🎮 VRAM Usage: ${VRAM_USAGE}MB"
        fi
        
        echo "✅ $level benchmark completed"
    else
        echo "❌ $level file not found: $MODEL_FILE"
    fi
done

echo ""
echo "🎉 Benchmark completed!"
echo "📈 Check the results to determine the best quantization for your GT 730"
`;
  }

  /**
   * Generate model configuration file
   */
  public generateModelConfig(): void {
    const config = {
      model: {
        name: this.config.modelName,
        baseModel: this.config.baseModel,
        type: 'technical-training-optimized',
        version: '1.0.0',
        targetDevice: this.config.targetDevice,
        optimizedFor: ['NVIDIA GT 730', 'Low-end GPUs', '2GB VRAM', '4GB VRAM'],
        quantization: this.config.quantizationLevels.map(level => ({
          name: level.name,
          bits: level.bits,
          description: level.description,
          estimatedVRAM: level.estimatedVRAM,
          performance: level.performance,
          recommendedFor: level.recommendedFor,
          filename: `${this.config.modelName}-${level.name}.gguf`
        }))
      },
      performance: {
        recommended: {
          'GT 730 2GB': 'Q2_K',
          'GT 730 4GB': 'Q3_K_S',
          'GTX 750 Ti': 'Q4_K_S',
          'GTX 950': 'Q4_K_M'
        },
        settings: {
          threads: 4,
          contextSize: 512,
          batchSize: 512,
          temperature: 0.7,
          topP: 0.9,
          repeatPenalty: 1.1
        }
      },
      training: {
        domains: ['ai-ml', 'coding', 'emerging-tech'],
        totalTokens: '211B',
        datasets: 12,
        accuracy: '87-99%',
        framework: 'CHRA-NF-XL'
      }
    };

    const configPath = join(this.config.outputDir, `${this.config.modelName}-config.json`);
    writeFileSync(configPath, JSON.stringify(config, null, 2));
  }

  /**
   * Save all scripts and configurations
   */
  public saveOptimizationFiles(): void {
    // Create conversion script
    const conversionScript = this.createConversionScript();
    writeFileSync(join(this.config.outputDir, 'convert-to-gguf.sh'), conversionScript);
    
    // Create inference script
    const inferenceScript = this.createInferenceScript();
    writeFileSync(join(this.config.outputDir, 'gt730-inference.sh'), inferenceScript);
    
    // Create benchmark script
    const benchmarkScript = this.createBenchmarkScript();
    writeFileSync(join(this.config.outputDir, 'gt730-benchmark.sh'), benchmarkScript);
    
    // Generate model configuration
    this.generateModelConfig();
    
    // Make scripts executable
    try {
      execSync(`chmod +x "${join(this.config.outputDir, 'convert-to-gguf.sh')}"`);
      execSync(`chmod +x "${join(this.config.outputDir, 'gt730-inference.sh')}"`);
      execSync(`chmod +x "${join(this.config.outputDir, 'gt730-benchmark.sh')}"`);
    } catch (error) {
      console.warn('Could not make scripts executable:', error);
    }
  }

  /**
   * Get recommended quantization for specific GT 730 variant
   */
  public getRecommendedQuantization(vramGB: number): QuantizationLevel {
    switch (vramGB) {
      case 1:
      case 2:
        return this.config.quantizationLevels.find(level => level.name === 'Q2_K')!;
      case 4:
        return this.config.quantizationLevels.find(level => level.name === 'Q3_K_S')!;
      default:
        return this.config.quantizationLevels.find(level => level.name === 'Q4_K_S')!;
    }
  }
}

/**
 * Create GT 730 optimized model configuration
 */
export function createGT730OptimizedModel(): GT730Optimizer {
  return new GT730Optimizer({
    modelName: 'chra-nf-xl-technical',
    baseModel: './models/chra-nf-xl-base',
    quantizationLevels: GT730Optimizer.GT730_QUANTIZATION_LEVELS,
    targetDevice: 'gt730',
    outputDir: './models/gguf-gt730-optimized'
  });
}

/**
 * Generate complete GGUF optimization package
 */
export function generateGGUFOptimizationPackage(): void {
  console.log('🚀 Generating GGUF optimization package for GT 730...');
  
  const optimizer = createGT730OptimizedModel();
  optimizer.saveOptimizationFiles();
  
  console.log('✅ GGUF optimization package generated successfully!');
  console.log('📁 Files created in ./models/gguf-gt730-optimized/');
  console.log('');
  console.log('📋 Next steps:');
  console.log('1. Run: ./models/gguf-gt730-optimized/convert-to-gguf.sh');
  console.log('2. Test: ./models/gguf-gt730-optimized/gt730-benchmark.sh');
  console.log('3. Use: ./models/gguf-gt730-optimized/gt730-inference.sh <model> <prompt>');
}