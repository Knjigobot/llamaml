@echo off
cd /d "%~dp0"
echo [Llamaml] Building Pure OxCaml GGML & LLaMA.cpp Engine...
dune build @all
if %errorlevel% neq 0 (
    echo [Error] Dune build failed.
    exit /b %errorlevel%
)
echo [Llamaml] Running automated formal verification and unit test suite...
dune runtest
if %errorlevel% neq 0 (
    echo [Error] Tests failed.
    exit /b %errorlevel%
)
echo [Llamaml] Build and verification completed successfully.
