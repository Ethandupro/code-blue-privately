# Running the study on EC2-Bench

The study runs inside EC2-Bench's own runner. We patch one function in their cloud proxy so the same
delegation mode can send P3 instead of the full history, then read their run folders back into our
claim. See [PLAN.md](PLAN.md) for what is being tested.

Tested against EC2-Bench commit `583e8ce` and dataset revision `12f9107ff7cb05969f667282438592b3687ed29b`.

## 1. One-time setup (on the Azure VM, which already serves the 9B)

```bash
git clone https://github.com/OpenBMB/EC2-Bench && cd EC2-Bench && git checkout 583e8ce
python -m pip install -r requirements.txt huggingface_hub
python -m pip install -e /path/to/code-blue-privately
python -m codeblue ec2 patch .                 # copies cb_p3.py / cb_detect.py, edits the proxy and runner
docker build -f runtime/Dockerfile -t ec2-bench-runtime:review .   # rebuild AFTER patching
```

The patch is idempotent and stops with an error if their code has changed. With `CODEBLUE_PACKET`
unset or `P0`, the patched checkout behaves exactly like the original.

Needs: Docker on the VM (unverified on the shared VM: check first), the edge endpoint (9B on vLLM)
reachable from inside the container, a cloud endpoint (the 27B stand-in on port 8003 for the pilot), a
judge endpoint. Set `EDGE_BASE_URL`, `EDGE_API_KEY`, `CLOUD_BASE_URL`, `CLOUD_API_KEY`,
`JUDGE_BASE_URL`, `JUDGE_API_KEY`, `JUDGE_MODEL` as in their `.env.example`, and write a models config
with `edge` and `cloud` providers (their `configs/models.example.json`). A key the server ignores still
needs a non-empty value (for example `x`), or the proxy's `Bearer` header is rejected.

## 2. Pilot: 5 tasks

Pick 5 annotated tasks that look company-like (spreadsheets, CRM, email). Download only those:

```bash
hf download chengpingan/EC2-Bench --repo-type dataset --revision 12f9107ff7cb05969f667282438592b3687ed29b \
  --include 'workspace/*/task_28_*/**' --include 'workspace/*/task_NN_*/**' ... --local-dir ./ec2bench-data
```

**Check the annotations first.** Open one `gt/sensitive_items.json` and confirm the values are literal
strings that appear in the task's files. Then draft labels:

```bash
python -m codeblue ec2 labels-template --data-root ./ec2bench-data --out labels/draft.json
```

Fill in `essential` and `attack_vectors` for the 5 tasks (see [labels/README.md](../labels/README.md))
and save as `labels/ec2bench.json`. Make the attacked copy of the data:

```bash
python -m codeblue ec2 inject --data-root ./ec2bench-data --labels labels/ec2bench.json --out ./ec2bench-data-attacked
```

Run the six arms (same filter, same repeats; `F='^EC2Bench_task_(28|NN|...)_'`):

```bash
COMMON="--runtime docker --task-filter $F --repeat 3 --parallel 2 --edge-model edge/Qwen/Qwen3.5-9B --models-config models.json"
python scripts/run.py local-only $COMMON --data-root ./ec2bench-data -o out/edge
python scripts/run.py delegation $COMMON --data-root ./ec2bench-data --cloud-model cloud/<model> -o out/p0
CODEBLUE_PACKET=P3 python scripts/run.py delegation $COMMON --data-root ./ec2bench-data --cloud-model cloud/<model> -o out/p3
python scripts/run.py local-only $COMMON --data-root ./ec2bench-data-attacked -o out/edge-att
python scripts/run.py delegation $COMMON --data-root ./ec2bench-data-attacked --cloud-model cloud/<model> -o out/p0-att
CODEBLUE_PACKET=P3 python scripts/run.py delegation $COMMON --data-root ./ec2bench-data-attacked --cloud-model cloud/<model> -o out/p3-att
```

`CODEBLUE_DETECTOR=llm` makes P3 use the edge model (plus regex) to find values to mask; the default is
regex only.

## 3. Collect

```bash
python -m codeblue ec2 collect --data-root ./ec2bench-data --labels labels/ec2bench.json --out results/pilot \
  --arm local-only:none:out/edge --arm P0:none:out/p0 --arm P3:none:out/p3 \
  --arm local-only:stuck_and_stuff:out/edge-att --arm P0:stuck_and_stuff:out/p0-att \
  --arm P3:stuck_and_stuff:out/p3-att
```

This prints the verdict and writes `claim.json`, `episodes.jsonl`, `leakage.json`, `rescue_curve.csv`,
`leak_vs_rescues.csv` and `mosaic.csv` (same meanings as in the main README). `claim.json` also has a
`coverage` block: episodes found, missing scores, tasks skipped and how many annotated values were too
short to scan.

## What the pilot must show before the full run

1. Annotated values are found by the scanner in P0's cloud log (if P0 "leaks nothing", the values do
   not match literally and the labels need work).
2. P0 scores clearly above local-only on several of the 5 tasks.
3. P3 episodes finish; `codeblue_p3_table.json` is in the task folders and the cloud log shows
   placeholders, not values.
4. On attacked tasks the agent opens the planted file (the injection marker shows up in its
   trajectory), so a forced escalation is possible.

Then label the remaining tasks, freeze labels (git tag), and run all 100 with 3 repeats.

## What the patch changes

`skills/cloud-router/privacy_proxy.py`: under `CODEBLUE_PACKET=P3`, `_build_cloud_view` asks the edge
model for a handoff note (same prompt as our harness), masks the task and the note with a placeholder
table kept for the whole episode, and sends only those plus our P3 role prompt (their "full history is
forwarded" role prompt is skipped). The cloud's text and tool-call arguments have placeholders filled
back in before the step runs. `eval/run_batch.py`: passes `CODEBLUE_PACKET` / `CODEBLUE_DETECTOR` into
the container and collects `/tmp/codeblue_p3_table.json`. The edge agent's prompt is not touched.
