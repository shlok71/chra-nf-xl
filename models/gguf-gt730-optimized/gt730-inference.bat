@echo off
REM GT 730 Optimized Inference Script (Windows)

set MODEL_PATH=%1
set PROMPT=%2

if "%MODEL_PATH%"=="" (
    echo Usage: %0 ^<model_path^> ^<prompt^>
    echo Example: %0 ".\models\gguf-gt730-optimized\chra-nf-xl-technical-Q2_K.gguf" "Hello, how are you?"
    exit /b 1
)

if "%PROMPT%"=="" (
    echo Usage: %0 ^<model_path^> ^<prompt^>
    echo Example: %0 ".\models\gguf-gt730-optimized\chra-nf-xl-technical-Q2_K.gguf" "Hello, how are you?"
    exit /b 1
)

echo 🚀 GT 730 Optimized Inference Starting...
echo 📊 Model: %MODEL_PATH%
echo 💬 Prompt: %PROMPT%

REM Check if model file exists
if not exist "%MODEL_PATH%" (
    echo ❌ Model file not found: %MODEL_PATH%
    exit /b 1
)

REM GT 730 optimized settings
set THREADS=4
set CTX_SIZE=512
set BATCH_SIZE=256
set TEMP=0.7
set TOP_P=0.9
set REPEAT_PENALTY=1.1

echo ⚙️  Settings:
echo    🧵 Threads: %THREADS%
echo    📝 Context Size: %CTX_SIZE%
echo    📦 Batch Size: %BATCH_SIZE%
echo    🌡️  Temperature: %TEMP%
echo    🎯 Top-P: %TOP_P%

REM Run inference
if exist "llama.cpp\build\bin\Release\main.exe" (
    llama.cpp\build\bin\Release\main.exe --model "%MODEL_PATH%" --prompt "%PROMPT%" --threads %THREADS% --ctx-size %CTX_SIZE% --batch-size %BATCH_SIZE% --temp %TEMP% --top-p %TOP_P% --repeat-penalty %REPEAT_PENALTY% --color --interactive
) else if exist "llama.cpp\main.exe" (
    llama.cpp\main.exe --model "%MODEL_PATH%" --prompt "%PROMPT%" --threads %THREADS% --ctx-size %CTX_SIZE% --batch-size %BATCH_SIZE% --temp %TEMP% --top-p %TOP_P% --repeat-penalty %REPEAT_PENALTY% --color --interactive
) else (
    echo ❌ llama.cpp main executable not found
    echo Please run the conversion script first: convert-to-gguf.bat
    exit /b 1
)

echo ✅ Inference completed!
pause
