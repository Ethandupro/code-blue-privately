# Parked: ideas that do not serve the main claim

The main study tests one claim (see [docs/PLAN.md](../docs/PLAN.md)): P3 placeholder notes versus P0
full history, on our own data rooms. Everything below was part of the earlier, larger plan. It is
parked, not deleted: the code still runs, and any of these can become a follow-up once the main
result exists.

| idea | where the code is | why it was cut |
|---|---|---|
| P1 brief without masking | `codeblue/packets.py` (P3 is built on it) | Kept in code. Run it only if someone asks whether the cut comes from summarising or from masking. |
| P2 blanked brief (`[REDACTED]`) | `codeblue/optional/packets_extra.py` | A second method to compare. Not needed to test whether P3 works. |
| P4 structure only | same file | Cannot help by design; it only shows the obvious floor. |
| P2h / P3h (masking applied to the full history) | same file | Ablations; cost as much as P0 per call. |
| P3b, a PAAC-style sanitise-and-distil baseline | not built | A second baseline from prior work; good follow-up. |
| Random trigger at a matched rate, `codeblue matched-rate` | `codeblue/optional/triggers_extra.py` | Tests the trigger, not the packet. |
| Cloud-only (`always` trigger) | same file | Tells us the ceiling, not whether P3 works. |
| Privacy-aware and early-warning triggers | not built | A separate question (H3). |
| Capped rescues (`max_rescues`, `cooldown`) | `codeblue/loop.py`, off by default | The main study measures rescues instead of limiting them. Set them for a sensitivity run; blocked firings are logged. |
| Single-purpose injection variants (`stuck_claim`, `fake_error`, `loop_bait`, `handoff_stuffing`) | `codeblue/attacks.py` | The main study uses one combined attack, `stuck_and_stuff`, as part 3 of the claim. The variants stay for working out which half of the attack did the damage. |
| Self-check defences (spotlighting, an actions-only view) and a two-stage attack screen | PR #3 (branch `claude/project-thread-8v381w`), not merged | Testing defences is a second question. The attack itself is now in the main study. |
| Attacker model that infers secrets from what leaked; composite-secret inference | not built (composite secrets are still scanned) | Needs a second model and its own validation. |
| Accumulation curves across rescues | `leakage.json` still records `cumulative_curve` | Reported as a side figure if useful, not a claim. |
| Other benchmarks: EC2-Bench, TheAgentCompany, EnterpriseOps-Gym | HTTP logging proxy in `codeblue/optional/proxy_server.py` (`codeblue proxy`) | Each needs an adapter and secret labels. Stretch goal: re-run P0 vs P3 on EC2-Bench's organisational tasks. |
| A second local model (e.g. Gemma 4 26B) | config only | One model keeps the claim clean; model size is future work. |

The old full grid still runs offline:

```bash
python -m codeblue run --config configs/optional/mock-full-grid.toml
```
