#!/usr/bin/env bash
# Serve the Code Blue LOCAL model (default Qwen/Qwen3.5-9B, the size EC2-Bench uses) with vLLM on a GPU VM.
# Run this ON THE VM, inside tmux so it outlives your SSH session:
#
#   tmux new -s codeblue
#   bash scripts/azure/serve_9b.sh                    # Qwen/Qwen3.5-9B, BF16, port 8000
#   QUANT=fp8 bash scripts/azure/serve_9b.sh          # ~10 GB of weights instead of ~19 GB (shared GPUs)
#   MODEL=google/gemma-4-26b-it GPU_GB=70 bash scripts/azure/serve_9b.sh   # the bigger local model arm
#   bash scripts/azure/serve_9b.sh check              # free memory per GPU, then exit
#
# Security: the server binds to 127.0.0.1 only and requires an API key (vLLM reads VLLM_API_KEY from the
# environment, so the key never shows up in `ps` for other users of a shared VM). Reach it from your laptop with
# scripts/azure/tunnel.sh. Never open port 8000 in the Azure network security group.
#
# Memory: on a shared VM other people's servers already use most of each GPU. GPU_GB caps this server's share
# (vLLM gets --gpu-memory-utilization = GPU_GB / total); the script refuses to start if less than GPU_GB is free.
# Pattern borrowed from sep-for-improvement (gpu.max_gpu_gb) and dso_agentdojov2 (local_victim/serve_vllm.sh).
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3.5-9B}
SERVED_NAME=${SERVED_NAME:-$MODEL}
PORT=${PORT:-8000}
VENV=${VENV:-$HOME/venvs/vllm}
GPU=${GPU:-auto}                 # auto = the GPU with the most free memory; or 0, 1, "0,1"
TP=${TP:-1}                      # 2 splits the model over two GPUs (set GPU="0,1")
QUANT=${QUANT:-}                 # empty = BF16; fp8 = on-the-fly FP8 weights (about half the memory)
MAX_LEN=${MAX_LEN:-32768}        # Code Blue episodes are short; 32k keeps the KV cache small
MAX_SEQS=${MAX_SEQS:-16}
KEY_FILE=${KEY_FILE:-$HOME/.codeblue/vllm_api_key}
LOG_DIR=${LOG_DIR:-$HOME/vllm_logs}
LOG_TAG=${LOG_TAG:-codeblue_server}   # file name prefix of the server log in LOG_DIR

if [ -z "${GPU_GB:-}" ]; then
  if [ -n "$QUANT" ]; then GPU_GB=16; else GPU_GB=26; fi   # weights + KV cache + CUDA graphs, with headroom
fi

free_report() { nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv,noheader,nounits; }

if [ "${1:-}" = "check" ]; then
  echo "index, name, free MiB, total MiB"; free_report
  echo "this config needs GPU_GB=$GPU_GB GiB free on $TP GPU(s)"
  exit 0
fi

if [ ! -x "$VENV/bin/vllm" ]; then
  echo "vLLM not found in $VENV. Install it first: docs/AZURE.md, step 1." >&2
  exit 1
fi

# API key: from VLLM_API_KEY, else from KEY_FILE (created on first run, readable only by you).
if [ -z "${VLLM_API_KEY:-}" ]; then
  if [ ! -s "$KEY_FILE" ]; then
    mkdir -p "$(dirname "$KEY_FILE")"; chmod 700 "$(dirname "$KEY_FILE")"
    (umask 077; python3 -c 'import secrets; print(secrets.token_urlsafe(32))' >"$KEY_FILE")
    echo "created a new API key in $KEY_FILE (the harness reads it from the api_key_env variable in its config)"
  fi
  VLLM_API_KEY=$(cat "$KEY_FILE")
fi
export VLLM_API_KEY

# Pick GPU(s) and check there is room.
mapfile -t ROWS < <(free_report)
if [ "$GPU" = "auto" ]; then
  GPU=$(printf '%s\n' "${ROWS[@]}" | sort -t, -k3 -n -r | head -n "$TP" | cut -d, -f1 | tr -d ' ' | sort -n | paste -sd, -)
fi
TOTAL_MIB=0
for g in ${GPU//,/ }; do
  row=$(printf '%s\n' "${ROWS[@]}" | awk -F', *' -v g="$g" '$1==g')
  free=$(echo "$row" | awk -F', *' '{print $3}'); total=$(echo "$row" | awk -F', *' '{print $4}')
  if [ -z "$free" ]; then echo "no GPU $g on this machine" >&2; exit 1; fi
  if [ "$free" -lt $((GPU_GB * 1024)) ]; then
    echo "GPU $g has $((free / 1024)) GiB free, this server needs GPU_GB=$GPU_GB. Try QUANT=fp8, a smaller GPU_GB," >&2
    echo "TP=2 GPU=0,1, or wait for other jobs to shrink ('bash $0 check' shows free memory)." >&2
    exit 1
  fi
  TOTAL_MIB=$total
done
UTIL=$(python3 -c "print(round(min(0.95, $GPU_GB * 1024 / $TOTAL_MIB), 3))")

mkdir -p "$LOG_DIR"
INFO="$LOG_DIR/${LOG_TAG}_$(date -u +%Y%m%dT%H%M%SZ).txt"
{
  echo "model=$MODEL served_as=$SERVED_NAME port=$PORT gpu=$GPU tp=$TP quant=${QUANT:-bf16} gpu_gb=$GPU_GB util=$UTIL max_len=$MAX_LEN max_seqs=$MAX_SEQS extra=${EXTRA_ARGS:-}"
  "$VENV/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} | tee "$INFO"
echo "keep $INFO with the run's results (it records exactly what served the local model)"

THINKING_ARGS=()
case "$MODEL" in
  # Qwen3.5 thinks by default; the harness wants short JSON actions, so thinking is off unless THINKING=1.
  # Requests can still turn it on with chat_template_kwargs.
  *Qwen3*) [ "${THINKING:-0}" = "1" ] || THINKING_ARGS=(--default-chat-template-kwargs '{"enable_thinking": false}')
           THINKING_ARGS+=(--reasoning-parser qwen3) ;;
esac

CUDA_VISIBLE_DEVICES=$GPU "$VENV/bin/vllm" serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --dtype bfloat16 ${QUANT:+--quantization "$QUANT"} \
  --max-model-len "$MAX_LEN" --max-num-seqs "$MAX_SEQS" \
  --gpu-memory-utilization "$UTIL" \
  --generation-config vllm \
  "${THINKING_ARGS[@]}" \
  ${EXTRA_ARGS:-} 2>&1 | tee -a "$INFO"
