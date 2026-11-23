#!/usr/bin/env bash
# TEST-ONLY stand-in for the cloud model: a 27B model served by vLLM on the same VM as the local 9B model.
# Run this ON THE VM, in its own tmux window, alongside scripts/azure/serve_9b.sh:
#
#   GPU=1 bash scripts/azure/serve_9b.sh             # local model, Qwen/Qwen3.5-9B on GPU 1, port 8000
#   bash scripts/azure/serve_cloud_27b.sh            # "cloud" model, Qwen/Qwen3.8-27B on GPU 0, port 8003, BF16
#   QUANT=fp8 bash scripts/azure/serve_cloud_27b.sh  # if GPU 0 has less than ~70 GiB free (needs ~40 GiB)
#   bash scripts/azure/serve_cloud_27b.sh check      # free memory per GPU, then exit
#
# It is a thin wrapper around serve_9b.sh with different defaults, and its own API key file, so the harness keeps
# the two keys in separate variables (CODEBLUE_LOCAL_API_KEY and CODEBLUE_CLOUD_API_KEY). The harness treats
# this server exactly like a real cloud API: every request to it goes to cloud_log.jsonl and is scanned.
# Any variable serve_9b.sh understands (MODEL, PORT, GPU, GPU_GB, TP, MAX_LEN, EXTRA_ARGS, ...) can be overridden.
set -euo pipefail
export MODEL=${MODEL:-Qwen/Qwen3.8-27B}
export PORT=${PORT:-8003}
export GPU=${GPU:-0}
export KEY_FILE=${KEY_FILE:-$HOME/.codeblue/cloud_api_key}
export LOG_TAG=${LOG_TAG:-codeblue_cloud}
# 27B dense: BF16 weights are ~56 GB, FP8 ~28 GB, plus KV cache and CUDA graphs at MAX_LEN=32768.
if [ -z "${GPU_GB:-}" ]; then
  if [ -n "${QUANT:-}" ]; then export GPU_GB=40; else export GPU_GB=70; fi
fi
exec bash "$(dirname "$0")/serve_9b.sh" "$@"
