# Serving the 9B local model on an Azure GPU VM

This connects the harness to a local model (default `Qwen/Qwen3.5-9B`, the size EC2-Bench uses) served by vLLM on
an Azure GPU VM. It reuses what already works in the user's other two projects:

* **sep-for-improvement** serves Qwen3.5 models with vLLM on the shared H100 VM, with no sudo (uv + a venv), inside
  tmux, and caps its share of each GPU because other people's vLLM servers fill most of it (`gpu.max_gpu_gb`).
* **dso_agentdojov2** (`local_victim/`) serves models with `vllm serve` on `127.0.0.1` only, records the server
  versions next to the results, and reaches the server from another machine with an SSH tunnel instead of an open
  port. Clients just get `--base-url http://localhost:8000/v1`.

Code Blue does the same: `scripts/azure/serve_9b.sh` on the VM, `scripts/azure/tunnel.sh` on your laptop, and
`configs/azure-9b.toml` points `[local]` at `http://localhost:8000/v1`.

```
 laptop (harness)                                   Azure GPU VM
 python -m codeblue run ──► localhost:8000 ══ SSH ══► 127.0.0.1:8000  vllm serve Qwen/Qwen3.5-9B  (API key)
        │  [local] traffic ─► runs/<run>/local_log.jsonl
        └─ [cloud] traffic ─► CloudGateway ─► runs/<run>/cloud_log.jsonl ─► cloud API (or the mock)
```

**Where the VM sits in the study.** The VM stands in for "the company's own hardware". Everything sent to it is
logged to `local_log.jsonl` and is never scanned as exposure; only `cloud_log.jsonl` is. Two guards keep that true:
the harness refuses a `[local]` `base_url` that is not loopback or a private IP (so local traffic can't silently go
over the public internet), and refuses `[local]` and `[cloud]` with the same `base_url`. Set `allow_remote = true` in
`[local]` only if you deliberately use some other secured endpoint.

**Security.** The server listens on `127.0.0.1` only and needs an API key, which vLLM reads from `VLLM_API_KEY` (so
other users on a shared VM can't see it in `ps`). Do not add an inbound rule for port 8000 in the VM's network
security group; SSH (22) is the only port you need, and Azure Bastion or `az ssh vm` work too.

## 1. One-time setup on the VM (no sudo)

If you already followed `local_victim/README.md` in dso_agentdojov2, `~/venvs/vllm` exists and you can skip the vLLM
install; Qwen3.5 needs vLLM 0.17 or newer.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
uv venv --python 3.12 ~/venvs/vllm
uv pip install --python ~/venvs/vllm/bin/python vllm --torch-backend=auto
~/venvs/vllm/bin/python -c "import torch, vllm; print(vllm.__version__, torch.version.cuda, torch.cuda.device_count())"

git clone https://github.com/Ethandupro/code-blue-privately.git ~/code-blue-privately
df -h ~        # the 9B weights are ~19 GB; export HF_HOME=/big/disk/hf if home is small
```

## 2. Start the server on the VM

```bash
cd ~/code-blue-privately
bash scripts/azure/serve_9b.sh check          # free memory on each GPU
tmux new -s codeblue                          # the server must outlive your SSH session
bash scripts/azure/serve_9b.sh                # Qwen/Qwen3.5-9B, BF16, on the GPU with the most free memory
```

The first start creates an API key in `~/.codeblue/vllm_api_key` (only you can read it) and downloads the weights.
It is ready when the log shows `Application startup complete`. Detach with `Ctrl-b d`. The script writes the model,
vLLM, PyTorch and GPU versions to `~/vllm_logs/codeblue_server_<time>.txt`; keep that file with the run's results.

Memory on the shared VM (other users' servers took most of each GPU last time):

| setting | needs free on one GPU | when |
|---|---|---|
| default (BF16, `GPU_GB=26`) | 26 GiB | the GPU has room |
| `QUANT=fp8` (`GPU_GB=16`) | 16 GiB | tight GPU; answers shift slightly, so serve every condition you compare the same way |
| `TP=2 GPU=0,1 GPU_GB=14` | 14 GiB on each of two GPUs | neither GPU has 26 GiB |
| `MAX_LEN=16384 MAX_SEQS=8` | less KV cache | vLLM says there is no memory for cache blocks |

Other options: `PORT`, `MODEL` (e.g. `google/gemma-4-26b-it GPU_GB=70` for the bigger local arm), `THINKING=1`,
`EXTRA_ARGS` (passed to `vllm serve`; on this VM the no-nvcc workaround is
`VLLM_ATTENTION_BACKEND=FLASH_ATTN EXTRA_ARGS=--enforce-eager`). Qwen3.5 checkpoints include a vision encoder; if
memory is tight you can try `EXTRA_ARGS='--limit-mm-per-prompt {"image":0,"video":0}'`.

## 3. Connect from your laptop

```bash
export CODEBLUE_LOCAL_API_KEY=$(ssh <vm> cat ~/.codeblue/vllm_api_key)
bash scripts/azure/tunnel.sh <vm>             # in its own terminal; Ctrl-C closes it
python -m codeblue ping --config configs/azure-9b.toml
python -m codeblue run  --config configs/azure-9b.toml
```

`<vm>` is whatever you normally `ssh` to (an alias from `~/.ssh/config`, or `azureuser@<public-ip>`). `ping` lists the
served models, checks the config's `model` is one of them, and times a two-word reply. If local port 8000 is busy,
use `LOCAL_PORT=8001` for the tunnel and change `base_url` to match.

**Alternative: run the harness on the VM itself.** The harness has no dependencies beyond Python 3.11, so you can
skip the tunnel: in a second tmux window on the VM, `export CODEBLUE_LOCAL_API_KEY=$(cat ~/.codeblue/vllm_api_key)`,
then the same `ping` and `run` commands. Results land in `~/code-blue-privately/runs/`; copy them back with `scp -r`.

## 4. Adding the real cloud model

`configs/azure-9b.toml` keeps the mock cloud so a first run costs nothing. For real rescues, replace its `[cloud]`
with the block in `configs/vllm-cloud.toml` and export `CODEBLUE_CLOUD_API_KEY`. The cloud key and the local key are
different variables, the two models are logged to different files, and only the cloud log is scanned.

(Parked, not part of the main study.) For harnesses we don't control (EC2-Bench, TheAgentCompany), point their *local* model at the tunnel
(`http://localhost:8000/v1`, key `CODEBLUE_LOCAL_API_KEY`) and only their *cloud* model at the logging proxy
(`python -m codeblue proxy ...`, see `optional/README.md`).

## 5. Test setup: a 27B "cloud" model on the same VM

To test real (non-mock) rescues without paying for a cloud API, serve a bigger model on a second GPU of the same VM
and point `[cloud]` at it. The default is the local 9B model on GPU 1 (port 8000) and `Qwen/Qwen3.8-27B` (dense
27B, official Qwen release, August 2026) on GPU 0 (port 8003). The harness treats port 8003 like any cloud API: every
request goes to `cloud_log.jsonl` and is scanned. This checks the full pipeline with a model that can actually reason;
the numbers are **not** results about frontier cloud models.

```bash
bash scripts/azure/serve_cloud_27b.sh check       # free memory on each GPU
tmux new -s codeblue                              # window 0: local model
GPU=1 bash scripts/azure/serve_9b.sh              # Ctrl-b c for a new window, then:
bash scripts/azure/serve_cloud_27b.sh             # window 1: "cloud" model, BF16, needs ~70 GiB free on GPU 0
QUANT=fp8 bash scripts/azure/serve_cloud_27b.sh   #   ...or FP8 if GPU 0 has less (needs ~40 GiB)
```

`serve_cloud_27b.sh` is `serve_9b.sh` with other defaults (`MODEL`, `PORT=8003`, `GPU=0`, `GPU_GB=70` or 40 with
FP8), its own key in `~/.codeblue/cloud_api_key`, and its own log, `~/vllm_logs/codeblue_cloud_<time>.txt`. Any
`serve_9b.sh` variable can be overridden. `TP=2` is not an option here because GPU 1 already hosts the 9B model, so
if GPU 0 is short of memory use `QUANT=fp8` or a smaller `MAX_LEN`. Qwen3.8 is newer than Qwen3.5: if vLLM says the
architecture is not supported, upgrade it (`uv pip install --python ~/venvs/vllm/bin/python -U vllm --torch-backend=auto`).
The checkpoint has a vision encoder; `EXTRA_ARGS='--limit-mm-per-prompt {"image":0,"video":0}'` saves a little memory.

Then, in a third window (or on your laptop with a second tunnel, `REMOTE_PORT=8003 LOCAL_PORT=8003 bash
scripts/azure/tunnel.sh <vm>`):

```bash
export CODEBLUE_LOCAL_API_KEY=$(cat ~/.codeblue/vllm_api_key)
export CODEBLUE_CLOUD_API_KEY=$(cat ~/.codeblue/cloud_api_key)
python -m codeblue ping --config configs/azure-9b-local-27b-cloud.toml           # checks [local]
python -m codeblue ping --cloud --config configs/azure-9b-local-27b-cloud.toml   # checks [cloud]
python -m codeblue run  --config configs/azure-9b-local-27b-cloud.toml
```

## Troubleshooting

* `cannot reach http://localhost:8000/v1`: the tunnel is closed, or the server is still loading. On the VM,
  `curl -s -H "Authorization: Bearer $(cat ~/.codeblue/vllm_api_key)" localhost:8000/v1/models`.
* `HTTP 401`: the laptop's `CODEBLUE_LOCAL_API_KEY` doesn't match `~/.codeblue/vllm_api_key` on the VM.
* `does not list 'Qwen/Qwen3.5-9B'`: the server was started with another `MODEL` or `SERVED_NAME`; make the
  config's `model` match.
* `GPU n has X GiB free`: see the memory table above. `bash scripts/azure/serve_9b.sh check` shows what is free now.
* `Engine core initialization failed` is only the last line; the cause is earlier in the log:
  `grep -n -iE "error|memory|cuda|not supported" ~/vllm_logs/codeblue_server_*.txt | tail -40`.
