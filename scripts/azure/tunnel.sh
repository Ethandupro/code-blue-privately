#!/usr/bin/env bash
# Forward the VM's vLLM port to this machine over SSH, so the harness talks to http://localhost:8000/v1 and the
# server never needs a public port. Run on your LAPTOP (or wherever you run `python -m codeblue`).
#
#   bash scripts/azure/tunnel.sh azureuser@<vm-ip-or-ssh-alias>     # blocks; Ctrl-C closes the tunnel
#   LOCAL_PORT=8001 bash scripts/azure/tunnel.sh my-vm             # if 8000 is taken locally
#
# Uses your normal SSH key / ~/.ssh/config (an `az ssh vm` config works too). Keep-alives stop idle drops.
set -euo pipefail
HOST=${1:?usage: tunnel.sh <ssh host>}
REMOTE_PORT=${REMOTE_PORT:-8000}
LOCAL_PORT=${LOCAL_PORT:-8000}
echo "localhost:$LOCAL_PORT -> $HOST:127.0.0.1:$REMOTE_PORT (Ctrl-C to close)"
echo "then: python -m codeblue ping --config configs/azure-9b.toml"
exec ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=4 \
  -L "127.0.0.1:$LOCAL_PORT:127.0.0.1:$REMOTE_PORT" "$HOST"
