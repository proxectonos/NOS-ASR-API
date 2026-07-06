#!/usr/bin/env bash
# Linux/macOS/WSL convenience wrapper. Delegates to the cross-platform launcher.
export USE_CUDA="${USE_CUDA:-0}"  # 1 to enable GPU inference
export MODELS_ROOT="${MODELS_ROOT:-models}"
export ASR_API_CONFIG="${ASR_API_CONFIG:-config.json}"
exec python run.py
