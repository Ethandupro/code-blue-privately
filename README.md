# Code Blue, Privately

**The question.** A company runs a small AI agent on its own hardware so its confidential files stay
in-house. When the agent gets stuck, it asks a big cloud model to take over for a couple of steps. To
do that it usually sends the cloud model its whole history, which contains everything it has read.
How much confidential data does that hand over, and can a simple change to what is sent stop most of
it without making the help useless?

**The claim we test (H1).** Sending a **placeholder-masked handoff note (P3)** instead of the agent's
**full history (P0)** cuts the share of confidential items that reach the cloud by **at least 75%**,
while keeping **at least 80%** of the rescue's benefit.

**How.** The same tasks run three ways: local agent alone, rescued with P0, rescued with P3. Every
cloud-bound request is logged before it is sent and scanned for secrets that were planted and
labelled in advance. Two numbers decide the claim: *leakage cut* and *gain kept*. Everything is in
one page: **[docs/PLAN.md](docs/PLAN.md)**. A diagram: [docs/system_flow.html](docs/system_flow.html).

This is a measurement harness, not a new privacy tool. Everything runs offline with mock models;
real models plug in through any OpenAI-compatible endpoint.

## Quick start (no GPU, no API key)

```bash
python -m pip install -e ".[dev]"     # Python 3.11+, no runtime dependencies
pytest -q
python -m codeblue run --config configs/mock.toml
```

The run ends with the claim check, for example:

```
CLAIM  P3 vs P0 under the selfcheck trigger  (3 tasks)
  local-only  score 0.000   secrets leaked 0.0%   rescues 0   blocked 0
  P0          score 1.000   secrets leaked 44.4%   rescues 1   blocked 0
  P3          score 1.000   secrets leaked 2.8%   rescues 1.67   blocked 0
  leakage cut  94%  95% CI [88%, 100%]  need >= 75%  -> supported
  gain kept    100%  95% CI [100%, 100%]  need >= 80%  -> supported
  note: only 3 tasks; the plan needs at least 40 for a real result
```

**These are mock numbers.** The mock cloud can only use what its packet shows, so they only prove
the plumbing works. They say nothing about real models.

Each run writes `runs/<timestamp>/`:

| file | what it holds |
|---|---|
| `claim.json` | the two numbers, their 95% intervals and the verdict |
| `summary.csv` | per condition: score, rescues, blocked trigger firings, cloud calls, characters sent, secrets leaked |
| `leakage.json` | per episode: which secrets leaked, how, at which cloud call, essential vs bystander |
| `episodes.jsonl` | every step of every episode: who acted, rescues, blocked firings, score |
| `cloud_log.jsonl` | every cloud-bound request (logged **before** sending) and its reply |
| `local_log.jsonl` | local-model traffic, kept apart and never scanned |

`python -m codeblue claim runs/<a> runs/<b> ...` pools several runs, one per data room.

## How it fits together

```
  data room ◄── local agent ── stuck? (self-check) ──► packet: P0 or P3 ──► CloudGateway ──► cloud model
  (tools)            ▲                                  │ P3's placeholder        │
                     └── cloud's action (placeholders ◄─┘ table stays local)      └─► cloud_log.jsonl ──► leak scanner
                         filled back in locally)  ◄────────────────────────────────
```

* **Rescue loop** (`codeblue/loop.py`). The local agent acts until the self-check says it is stuck.
  The cloud then takes `takeover_steps` (2) steps, each from a fresh packet, and hands back.
  **Nothing caps the number of rescues** except the task's step budget (12), which is the same in
  every condition; rescues are measured, not limited. `max_rescues` and `cooldown` exist only for a
  sensitivity run, and when set, the trigger is still asked every step and each firing they block is
  recorded (`blocked_firings`), so a cap can never hide how often the agent wanted help.
* **Trigger** (`codeblue/triggers.py`). AgentCollab-style self-check: after each step the local
  model answers PROGRESSING or STUCK, backed by stall signals (same action twice, two errors in a
  row). `never` gives the local-only condition.
* **Packets** (`codeblue/packets.py`):

  | packet | the cloud sees |
  |---|---|
  | P0 full history | the local agent's whole context, verbatim (what EC2-Bench measures) |
  | P3 placeholder note | the goal plus a handoff note written by the local model, with detected names, amounts, dates and codes replaced by consistent placeholders (`<NAME_1>`, `<MONEY_2>`). The table stays local and persists across rescues; placeholders in the cloud's reply are filled back in before the action runs |
  | P1 note, unmasked | diagnostic only: P3 without the masking |

* **Detector** for P3 (`codeblue/detect.py`): `regex` (money, percentages, dates, emails, account
  numbers, codes, large numbers, multi-word names, plus the company's own glossary). `oracle` uses
  the secret labels and is an upper bound only, never a headline result.
* **Leak scanner** (`codeblue/scan.py`, `python -m codeblue scan <log>`). Only requests are scanned,
  never replies. No AI judge: exact match, letters-and-digits match, near-miss canaries, numbers in
  other formats (2.84m = 2,841,000), keyword paraphrases, and composite secrets. Every secret is
  counted as essential (the task needs it) or bystander.
* **Claim check** (`codeblue/claim.py`): leakage cut and gain kept with a paired bootstrap over tasks.

## The data rooms

`data_rooms/harbourline/` is a fictional Singapore freight company with 8 files, 12 labelled secrets
and 3 tasks: 4 **canaries** (exact strings such as a project codename and a bank account), 7
**facts** that can be paraphrased (a valuation, a layoff count, a customer margin...), and 1
**composite** (two harmless facts that together reveal a secret deal). Each task names its
**essential** secrets; every other secret is a **bystander** for that task. The regex detector
deliberately misses "62 roles" in the HR file, so P3 still leaks the layoff fact on task 3: the kind
of gap the study should catch, and why P3's leakage is not zero.

The plan needs 4 rooms × 10 tasks, frozen before the final runs. New rooms use the same layout; see
the room's README.

## Using a real local model (including a 9B one)

The local model is just config. For the 9B model on a laptop or a single GPU through Ollama:

```bash
ollama pull qwen3.5:9b
python -m codeblue run --config configs/ollama-9b.toml
```

**On an Azure GPU VM** (or any GPU box you can SSH into): serve the model there with vLLM and reach it
through an SSH tunnel, so it never needs a public port. Full steps, memory settings for a shared GPU,
and troubleshooting are in [docs/AZURE.md](docs/AZURE.md).

```bash
bash scripts/azure/serve_9b.sh                                      # on the VM, inside tmux
export CODEBLUE_LOCAL_API_KEY=$(ssh <vm> cat ~/.codeblue/vllm_api_key)  # on your laptop
bash scripts/azure/tunnel.sh <vm>                                   # leave running
python -m codeblue ping --config configs/azure-9b.toml              # checks the endpoint, key and model name
python -m codeblue run  --config configs/azure-9b.toml
```

The harness refuses a `[local]` `base_url` that is not loopback or a private IP, and refuses `[local]`
and `[cloud]` with the same endpoint, so local traffic stays in `local_log.jsonl` and never mixes
with the cloud log.

To test real rescues for free, `scripts/azure/serve_cloud_27b.sh` serves a 27B model on a second GPU of the same VM
(port 8003) as a stand-in cloud, and `configs/azure-9b-local-27b-cloud.toml` points `[cloud]` at it; `codeblue ping
--cloud` checks that endpoint. See [docs/AZURE.md](docs/AZURE.md), section 5.

Change `model` to any tag Ollama or vLLM serves. `configs/vllm-cloud.toml` is the template for the
final runs (the 9B on vLLM, any OpenAI-compatible cloud API with the key in an environment
variable). For Qwen3-style models you can turn off thinking with
`extra_body = { chat_template_kwargs = { enable_thinking = false } }`; `<think>` blocks are
stripped either way.

## Not done yet

* Three more data rooms and seven more Harbourline tasks (40 tasks in all).
* The pilot with real models (9B local, 27B stand-in cloud), paused while GPU 1 is busy.
* Charts for the write-up.

Parked ideas (other packets, triggers, the injection attack, other benchmarks) are listed in
[optional/README.md](optional/README.md) with the reason each was cut.

Everything in `data_rooms/` is invented. Do not put real company data in this repository.
