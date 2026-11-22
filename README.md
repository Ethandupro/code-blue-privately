# Code Blue, Privately

When a small agent running on a company's own hardware gets stuck, it can call a frontier cloud
model for a short rescue (1 to 3 steps), then carry on locally. This repository measures **how
much confidential business data reaches the cloud model during those rescues**, and **which
rescue packet format keeps the rescue working while sending the least**.

It is a measurement harness, not a new sanitiser. Everything runs offline with mock models;
real models plug in through any OpenAI-compatible endpoint.

## Quick start (no GPU, no API key)

```bash
python -m pip install -e ".[dev]"     # Python 3.11+, no runtime dependencies
pytest -q                             # 31 tests
python -m codeblue run --config configs/mock.toml
```

The mock run prints one row per condition (local-only, P0 to P4 under the self-check and random
triggers, and cloud-only), and writes `runs/<timestamp>/` with:

| file | what it holds |
|---|---|
| `cloud_log.jsonl` | every cloud-bound request (logged **before** sending) and its reply |
| `local_log.jsonl` | local-model traffic (acting, self-checks, briefs, detection), kept apart |
| `episodes.jsonl` | every step of every episode, who took it, score, rescues |
| `leakage.json` | per episode: which secrets leaked, how, at which cloud call, essential vs bystander, cumulative curve |
| `summary.csv` | per condition: success, rescues, cloud calls, characters sent, essential and bystander leaks |

The mock numbers only show the plumbing works. The mock cloud can only use what its packet shows,
so the pattern is the expected one (P0 succeeds and leaks, P3 succeeds with far less leakage,
P2 breaks answers, P4 cannot help), but nothing here is a result about real models.

## How it fits together

```
           local model (vLLM / Ollama, configurable)
                 │  act, self-check, brief, detect  ──►  local_log.jsonl
                 ▼
  data room ◄── agent loop ── trigger fires? ──► packet builder (P0..P4) ──► CloudGateway ──► cloud model
  (tools)        ▲                                   │ placeholder table                │
                 └──── cloud action (placeholders ◄──┘ stays on device)                 └─► cloud_log.jsonl ──► scan
                       re-inserted locally) ◄────────────────────────────────────────────────
```

* **Rescue loop** (`codeblue/loop.py`). The local agent acts until the trigger fires. The cloud
  then takes over for `takeover_steps` (default 2) steps, each built from a fresh packet, and hands
  back. Rescues are capped (`max_rescues`, default 3) with a `cooldown` between them.
* **Triggers** (`codeblue/triggers.py`). `selfcheck` is AgentCollab-style: after each step the
  local model answers PROGRESSING or STUCK, backed by stall signals (same action twice, two errors
  in a row). `random` is the control; set its rate with `codeblue matched-rate runs/<run>/episodes.jsonl`
  so both arms call the cloud equally often. `never` is local-only, `always` is cloud-only.
* **Packets** (`codeblue/packets.py`):

  | packet | the cloud sees |
  |---|---|
  | P0 full history | the local agent's whole context, verbatim (the EC2-Bench-style baseline) |
  | P1 brief | the goal plus a handoff note written by the local model |
  | P2 blanked | full history with detected spans replaced by `[REDACTED]` |
  | P3 placeholders | full history with consistent typed placeholders (`<NAME_1>`, `<MONEY_2>`). The table stays local, persists across rescues in an episode, and placeholders in the cloud's reply are filled back in before the action runs |
  | P4 structure only | task category, tool names and an action skeleton; no goal, no content |

  New packets (for example a PAAC-style sanitise-and-distil baseline, P3b) subclass `PacketBuilder`
  and go in `BUILDERS`.
* **Detectors** for P2 and P3 (`codeblue/detect.py`): `regex` (money, percentages, dates, emails,
  account numbers, codes, large numbers, multi-word names, plus the organisation's own glossary),
  `llm` (asks the local model, falls back to regex), and `oracle` (uses the secret labels; an upper
  bound only, never a headline result).
* **Cloud gateway and proxy**. In-harness calls go through `CloudGateway`
  (`codeblue/gateway.py`). For harnesses we don't control (OpenHands in TheAgentCompany,
  EC2-Bench's runtime), run the HTTP logging proxy and point only their cloud base URL at it:

  ```bash
  export CODEBLUE_CLOUD_API_KEY=...
  python -m codeblue proxy --upstream https://openrouter.ai/api/v1 --api-key-env CODEBLUE_CLOUD_API_KEY \
      --log runs/proxy/cloud_log.jsonl --port 8787
  # harness cloud base URL -> http://127.0.0.1:8787/v1 ; send X-CodeBlue-Episode / X-CodeBlue-Task headers if possible
  ```

  Keep simulated colleagues, evaluators and the local model pointed elsewhere, so the cloud log
  contains only what the rescued agent sent. Streaming requests are refused.
* **Leak scanner** (`codeblue/scan.py`, `python -m codeblue scan <log>`). Only requests are
  scanned, never replies. Methods: `exact`, `squash` (letters and digits only), `edit`
  (near-miss canaries), `numeric` (2.84m = 2,841,000), `keywords` (paraphrases), and `composite`
  (all components leaked in the same episode, possibly across calls). Every secret is counted as
  essential or bystander for the task.

## The example data room

`data_rooms/harbourline/` is a fictional Singapore freight company with 8 files, 12 labelled
secrets and 3 tasks:

* 4 **canaries** (exact strings: project codename, deal code, email, bank account),
* 7 **facts** that can be paraphrased (valuation, CFO trip, layoffs, a customer margin, ...),
* 1 **composite** (CFO due-diligence trip to Johor + "Kestrel is headquartered in Johor" lets a
  reader infer the secret acquisition target, though neither fact says it).

Each task names its **essential** secrets (needed to answer); every other secret is a
**bystander** for that task. The regex detector deliberately misses "62 roles" in the HR file, so
P3 still leaks the layoff fact on task 3. That is the kind of gap the study should measure. See
the room's README for the file format; new rooms (the plan is 12 companies, about 100 tasks) use
the same layout.

## Using a real local model (including a 9B one)

The local model is just config. For a 9B model on a laptop or a single GPU through Ollama:

```bash
ollama pull qwen3.5:9b
python -m codeblue run --config configs/ollama-9b.toml
```

Change `model` to any tag Ollama or vLLM serves. `configs/vllm-cloud.toml` is the template for a
real run (Gemma 4 26B on vLLM, any OpenAI-compatible cloud API with the key in an environment
variable). For Qwen3-style models you can turn off thinking with
`extra_body = { chat_template_kwargs = { enable_thinking = false } }`; `<think>` blocks are
stripped either way. A smaller model will usually trigger more rescues, which means more cloud
calls and more exposure, so the local model size is itself a variable worth reporting.

## Not done yet

* Adapters for external benchmarks (EC2-Bench, TheAgentCompany, EnterpriseOps-Gym). The proxy is
  the bridge; each benchmark still needs a mapping from its sensitive data to `secrets.json` labels.
* The P3b PAAC-style baseline packet.
* More data rooms and tasks, and an independent check that the mock pattern holds with real models.

Everything in `data_rooms/` is invented. Do not put real company data in this repository.
