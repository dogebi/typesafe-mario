# Running typesafe-mario on Laya instead of TypeSafe

`typesafe-mario` asks three typed judgments per decision (`Choice` for the next
controller macro, `Noul` for "should a jump start now", `Score` for immediate
danger) and sends them to TypeSafe's hosted Jev. Laya answers the identical
request shape, so the integration is a second `Policy` plus two adjustments that
were measured rather than assumed.

```bash
# one-time
uv venv --python 3.13 && uv pip install -e ".[mario,dev]"

# Laya must be up (the tetris project's sidecar, 127.0.0.1:8091)
cd ~/Downloads/dev/jev-tetris && ./play-local.sh status     # jev + laya sidecars

cd ~/Downloads/dev/typesafe-mario
uv run typesafe-mario play --policy laya                    # live dashboard
uv run typesafe-mario play --policy laya --display none --max-decisions 200 --seed 7
uv run typesafe-mario play --policy laya --laya-stall-break  # + scripted stall recovery
python bench_laya.py 123,124,125 60                         # the four configs below
```

| Flag | Meaning |
| --- | --- |
| `--policy laya` | local Laya over `/v1/systemone` (no key, no network) |
| `--laya-url` | sidecar base URL (`LAYA_URL`, default `http://127.0.0.1:8091`) |
| `--laya-prior-t` | temperature of the code-side prior mixed with Laya's probabilities; `0` = Laya alone (default `LAYA_PRIOR_T` or `0.4`) |
| `--laya-stall-break` | on a stall, bypass Laya with a scripted back-off/runway/jump cycle (see below); forced decisions are logged as `source=stall_break` |
| `--laya-stall-threshold` | stalled decisions before the recovery takes over (default `LAYA_STALL_THRESHOLD` or `2`) |

## What had to change

1. **Instructions fit 256 tokens.** Laya reads `instructions + options` inside a
   256-token budget and the whole sequence in 1024, so the TypeSafe instruction
   block (~330 tokens of prose by itself) cannot be sent as-is. `laya.py` carries
   a compact block saying the same thing in ~90 tokens.
2. **The state must survive to the end.** Laya puts the state *after* the
   options, so whatever overflows the window is dropped — and that would be the
   observations. `_compact_state` sends the fields the judgments read
   (`player`, `trajectory`, `hazard`, `terrain`, `reaction_timing`,
   `recent_control`, `episode`) and drops the rest.

A third change is a *prior*, not a rewrite: Laya's own distribution was the
problem (below), so the code-derived facts enter as
`log softmax(score / t) + log p_laya`, exactly the mechanism the tetris port uses.
Laya can still overrule the prior; when it does, the count is recorded.

A fourth is the **stall recovery** (`--laya-stall-break`). The prior keeps Mario
alive but parks him at the first pipe, and no prompt wording moves Laya off it
(see "The first pipe"), so the recovery is scripted in code: when
`episode.stalled_frames` reaches the threshold, Laya is bypassed for a cycle of
back-off → runway → running jump (`STALL_BREAK_CYCLE` in `laya.py`). Laya is
still queried on those decisions so its `Noul`/`Score` judgments stay in the
log, but the action is the scripted one.

Because a scripted action must never be reported as a model answer, every
decision now carries `source` (`laya` / `stall_break` / `fallback` /
`heuristic` / `typesafe`) and `laya_choice` (what Laya itself said), and both
land in the run log. Before this, the log recorded only the action, so a prior
re-rank and a model pick were indistinguishable.

## Measured (M1 Pro 16 GB, 3 seeds × 60 decisions, `bench_laya.py`)

| Configuration | max x | decisions | deaths | forced | actions | p50 |
| --- | --- | --- | --- | --- | --- | --- |
| Laya alone (`--laya-prior-t 0`) | 290 | 15 | **3/3** | 0 | `right_run_jump` ×15 | 279-470 ms |
| Laya alone + stall-break | 290 | 15 | **3/3** | 0 | `right_run_jump` ×15 | 251-336 ms |
| Laya + prior (`0.4`) | **594** | 60 (no death) | **0/3** | 0 | `right_run_jump` ×49, `right_run` ×11 | 264-387 ms |
| Laya + prior + stall-break | **664** | 45 | **3/3** | 4 | `right_run_jump` ×30, `right_run` ×13, `left` ×2 | 318-396 ms |

All four rows are identical across seeds 123/124/125 to the decision, because
Laya is deterministic and the emulator is seeded. Two readings matter:

- **The stall-break is inert without the prior.** Laya alone dies at x=290 in 15
  decisions, before `stalled_frames` can reach the threshold, so the breaker
  never fires (`forced=0`). It is a fix for a plateau, not for dying early.
- **It trades safety for progress.** With the prior, the pipe at x=594 holds for
  the whole run and Mario never dies (0/3) — but he also never passes it. With
  the breaker he clears it (x=664) and then dies 3/3 on the goomba beyond,
  because Laya answers `right_run_jump` into the enemy exactly as before. The
  pipe was sheltering him. Passing the pipe is therefore *not* yet a win on the
  survival metric; enemy handling is the next gap.

Laya alone is degenerate on this task: it answered `right_run_jump` on every
decision, `jump_needed` = 1.00 on every decision, `danger` ≈ 1.95 of 2 (max) on
every decision, and died at the same x as the offline heuristic. Its own
`Noul`/`Score` verdicts are saturated, which is why one typed judgment is not
enough signal here.

## The first pipe (x=594), measured

A 200-decision run with the prior holds x=594 with `episode.stalled_frames` at
168. The state at the plateau is unambiguous — `terrain.obstacle_ahead=true`,
`obstacle_distance_tiles=1`, `obstacle_height_tiles=3`, `clear_forward_tiles=0`,
and `player.horizontal_speed_px_per_frame=0` — and Laya answers
`right_run_jump` with p=0.9976 and `left` with p=0.0000 for 119 consecutive
decisions. A standstill jump cannot clear a 3-tile pipe; it needs runway.

The prompt already asks it to notice this ("if `episode.stalled_frames` rises and
`recent_control.outcome` is blocked, the last action failed: change it"), and
that trigger never fires: `stalled_frames` does rise (0 → 118), but
`recent_control.outcome` stays `advanced`, because the 20-frame window still
contains the approach run (`progress_gained_pixels=160`). The remedy is
therefore disabled by its own precondition, which is why the recovery had to be
scripted in code rather than requested in the prompt.

**Live mode plays much better than `--display none`.** The dashboard (`play
--policy laya`, seed 123) reached x=1744 in 31 decisions before dying, decisions
every 260-780 ms, p50 365 ms — against x=594 / 60 decisions in headless
fast-forward. The difference is real time: the emulator advances during the ~0.3 s
the decision takes, so each held action covers ~26 frames instead of the bare 8
the headless loop steps, and `reaction_timing` carries the measured delay. Do not
read headless distances as the bot's skill.

Latency budget: each decision is three sequential requests (choice, noul, score)
with a ~750-token prompt → ~257-365 ms p50 on the local multilingual checkpoint.

If the sidecar is not running, `LayaPolicy` raises
`RuntimeError: Laya is not reachable at ... (Connection refused)` with the command
to start it. Note that this shell exports a proxy: the policy explicitly bypasses
it for the loopback sidecar (urllib would otherwise route 127.0.0.1 through the
proxy and get a 502).
