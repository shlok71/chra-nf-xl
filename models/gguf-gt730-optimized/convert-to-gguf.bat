@echo off
REM GGUF Conversion Script for GT 730 (Windows)

set MODEL_NAME=chra-nf-xl-technical
set BASE_MODEL=./models/chra-nf-xl-base
set OUTPUT_DIR=./models/gguf-gt730-optimized

echo 🚀 Converting to GGUF format for GT 730...
echo 📊 Model: %MODEL_NAME%
echo 📁 Base Model: %BASE_MODEL%
echo 📁 Output Directory: %OUTPUT_DIR%

REM Check if llama.cpp is available
if not exist "llama.cpp" (
    echo 📥 Installing llama.cpp...
    git clone https://github.com/ggerganov/llama.cpp.git
    cd llama.cpp
    
    REM Try CMake build
    where cmake >nul 2>nul
    if %errorlevel% equ 0 (
        echo 🔨 Building with CMake...
        mkdir build
        cd build
        cmake .. -DLLAMA_CUBLAS=ON
        cmake --build . --config Release
        cd ..\..
    ) else (
        echo ⚠️  CMake not found, please install CMake
        exit /b 1
    )
    
    cd ..
    echo ✅ llama.cpp installed
)

REM Convert base model to GGUF
echo 🔄 Converting base model to GGUF format...
python llama.cpp\convert.py --outfile "%OUTPUT_DIR%\%MODEL_NAME%-base.gguf" --outtype f16 "%BASE_MODEL%"

REM Quantize for GT 730
echo ⚡ Quantizing for GT 730 optimization...

llama.cpp\build\bin\Release\quantize.exe "%OUTPUT_DIR%\%MODEL_NAME%-base.gguf" "%OUTPUT_DIR%\%MODEL_NAME%-Q2_K.gguf" Q2_K
if %errorlevel% equ 0 echo ✅ Q2_K completed

llama.cpp\build\bin\Release\quantize.exe "%OUTPUT_DIR%\%MODEL_NAME%-base.gguf" "%OUTPUT_DIR%\%MODEL_NAME%-Q3_K_S.gguf" Q3_K_S
if %errorlevel% equ 0 echo ✅ Q3_K_S completed

llama.cpp\build\bin\Release\quantize.exe "%OUTPUT_DIR%\%MODEL_NAME%-base.gguf" "%OUTPUT_DIR%\%MODEL_NAME%-Q3_K_M.gguf" Q3_K_M
if %errorlevel% equ 0 echo ✅ Q3_K_M completed

llama.cpp\build\bin\Release\quantize.exe "%OUTPUT_DIR%\%MODEL_NAME%-base.gguf" "%OUTPUT_DIR%\%MODEL_NAME%-Q4_K_S.gguf" Q4_K_S
if %errorlevel% equ 0 echo ✅ Q4_K_S completed

echo 🎉 GGUF conversion completed!
echo 📊 Generated files:
dir "%OUTPUT_DIR%\%MODEL_NAME%*.gguf"

echo.
echo 💡 GT 730 Recommendations:
echo    🎮 GT 730 1GB/2GB: Use Q2_K
echo    🎮 GT 730 4GB: Use Q3_K_S
echo.
echo 🚀 To test inference:
echo    gt730-inference.bat "%OUTPUT_DIR%\%MODEL_NAME%-Q2_K.gguf" "Your prompt here"
pause
