@echo off
REM Windows convenience wrapper. Delegates to the cross-platform launcher.
if "%USE_CUDA%"=="" set USE_CUDA=0
if "%MODELS_ROOT%"=="" set MODELS_ROOT=models
if "%ASR_API_CONFIG%"=="" set ASR_API_CONFIG=config.json
python run.py
