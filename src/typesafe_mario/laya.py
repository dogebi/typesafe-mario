"""Laya as the decider: the same three typed judgments, answered by a local model.

`TypeSafePolicy` talks to TypeSafe's hosted Jev. This policy speaks the identical
`/v1/systemone` shape to a Laya sidecar on this machine (see the tetris project's
`ai/laya_mlx_server.py`), so no API key and no network are involved.

Two things had to change for Laya, both measured rather than assumed:

- **Instructions fit in 256 tokens.** Laya reads `instructions + options` in one
  256-token budget and the whole sequence (state last) in 1024, so the TypeSafe
  instruction block — about 330 tokens of prose on its own — cannot be sent
  as-is. The compact block below says the same thing in ~90 tokens.
- **The state must survive to the end of the sequence.** Laya appends the state
  after the options, so a state that overflows the 1024-token window is the part
  that gets dropped. `_compact_state` keeps the fields the judgments actually
  read and drops the rest.

Like the tetris port, the choice is not taken on trust: every decision records
`source` on the `Decision` and in the run log, so a model answer ("laya"), a
code-side re-rank, a scripted stall recovery ("stall_break") and a fallback
("fallback") are never passed off as each other. `--laya-stall-break` enables
the scripted recovery described at `STALL_BREAK_CYCLE`.
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from collections.abc import Collection, Mapping, Sequence
from typing import Any

from .actions import Action
from .policy import Decision
from .state import MarioSnapshot

DEFAULT_URL = os.environ.get("LAYA_URL", "http://127.0.0.1:8091")

# Short option texts: the choice is scored on these, so they carry the meaning
# that matters ("hold a running jump now") without spending 20 tokens each.
ACTION_TEXT: dict[Action, str] = {
    Action.NOOP: "release the controls, keep momentum",
    Action.RIGHT: "walk right, no jump",
    Action.RIGHT_JUMP: "hold a forward jump",
    Action.RIGHT_RUN: "run right, terrain clear",
    Action.RIGHT_RUN_JUMP: "running jump, hold it while rising",
    Action.JUMP: "jump nearly in place",
    Action.LEFT: "move left to evade or recover",
}

INSTRUCTIONS = (
    "Super Mario Bros. Pick the controller macro to hold for the next 8 frames. "
    "Goal: reach the flag without dying.\n"
    "A trusted obstacle or gap within 3 tiles needs a forward jump. While "
    "airborne trust terrain.last_grounded_preview over current geometry. If "
    "hazard.jump_must_start_this_decision is true, take a running jump now. "
    "Keep forward speed while trajectory.crossing_known_gap. If "
    "episode.stalled_frames rises and recent_control.outcome is blocked, the "
    "last action failed: change it. Reply with the option id."
)

JUMP_INSTRUCTIONS = (
    "Should a forward jump start now, or stay held if already rising? Yes if "
    "hazard.jump_must_start_this_decision is true, if a trusted obstacle or gap "
    "is within 3 tiles, if projected contact is imminent, or if a jump over a "
    "known gap is still rising."
)

DANGER_INSTRUCTIONS = "How dangerous is Mario's immediate situation?"

DANGER_LEVELS = [
    "safe open movement",
    "obstacle or enemy soon",
    "immediate collision, fall or enemy threat",
]

# Laya alone was measured on this task to answer `right_run_jump` on 15 of 15
# decisions (and "jump now" with probability 1.00 on all of them), then die at
# the same x as the offline heuristic. That is the failure the tetris port
# documented: a near-degenerate distribution over the options. So the
# code-derived facts enter as a *prior* — the pick maximises
# log softmax(score / PRIOR_T) + log p_laya — which Laya can still overrule
# when it is confident. PRIOR_T=0 turns the prior off (Laya alone).
PRIOR_T = float(os.environ.get("LAYA_PRIOR_T", "0.4"))

# A stall is the one failure Laya cannot be talked out of. Measured on the real
# game: at x=594 (first pipe) the state says obstacle_ahead at 1 tile, height 3,
# clear_forward_tiles=0 and Mario's speed is 0, yet Laya answers right_run_jump
# with p=0.9976 and left with p=0.0000 for 119 consecutive decisions, because a
# standstill jump cannot clear a 3-tile pipe -- it needs runway. The instruction
# trigger ("stalled_frames rising and recent_control.outcome blocked") never
# fires either: the 20-frame window still contains the approach run, so the
# outcome stays "advanced". So the recovery is scripted in code: back off, build
# runway, then commit to the running jump. Off by default; --laya-stall-break
# enables it and every forced decision is logged with source="stall_break".
STALL_BREAK_CYCLE: tuple[Action, ...] = (
    Action.LEFT,
    Action.LEFT,
    Action.RIGHT_RUN,
    Action.RIGHT_RUN,
    Action.RIGHT_RUN,
    Action.RIGHT_RUN_JUMP,
)
STALL_BREAK_THRESHOLD = int(os.environ.get("LAYA_STALL_THRESHOLD", "2"))


def stall_break_action(
    stalled_steps: int,
    threshold: int,
    cycle_index: int,
    allowed: Collection[Action],
) -> Action | None:
    """The scripted recovery action, or None to leave the decision to Laya.

    `cycle_index` is how many forced decisions have already been spent in the
    current stall, so the cycle advances one step per decision.
    """
    if threshold <= 0 or stalled_steps < threshold:
        return None
    candidate = STALL_BREAK_CYCLE[cycle_index % len(STALL_BREAK_CYCLE)]
    return candidate if candidate in allowed else None


def _action_scores(state: Mapping[str, Any]) -> dict[Action, float]:
    """Deterministic ranking from the same facts the instructions describe."""
    hazard = state.get("hazard", {})
    terrain = state.get("terrain", {})
    trajectory = state.get("trajectory", {})
    grounded = state.get("player", {}).get("grounded", True)

    jump_now = bool(hazard.get("jump_must_start_this_decision")) or bool(
        hazard.get("contact_within_reaction_horizon")
    )
    blocked = bool(terrain.get("obstacle_ahead")) or bool(terrain.get("gap_ahead"))
    crossing = bool(trajectory.get("crossing_known_gap"))

    scores = dict.fromkeys(Action, 0.0)
    if jump_now or blocked:
        scores[Action.RIGHT_RUN_JUMP] = 3.0
        scores[Action.RIGHT_JUMP] = 2.0
    elif crossing or not grounded:
        scores[Action.RIGHT_RUN_JUMP] = 2.5
        scores[Action.RIGHT_JUMP] = 1.5
    else:
        scores[Action.RIGHT_RUN] = 3.0
        scores[Action.RIGHT] = 2.0
    scores[Action.NOOP] -= 1.0
    scores[Action.LEFT] -= 1.0
    scores[Action.JUMP] -= 1.0
    return scores


def _combine_with_prior(
    probabilities: Mapping[str, float], state: Mapping[str, Any], prior_t: float
) -> dict[str, float]:
    """log softmax(score / prior_t) + log p_laya, normalised over the options."""
    if prior_t <= 0 or not probabilities:
        return dict(probabilities)
    scores = _action_scores(state)
    top = max(scores.values())
    weights = {}
    for name, probability in probabilities.items():
        try:
            score = scores[Action(name)]
        except ValueError:  # an action we did not rank: leave Laya's mass alone
            score = top
        weights[name] = math.exp((score - top) / prior_t) * max(1e-6, float(probability))
    total = sum(weights.values())
    if total <= 0:
        return dict(probabilities)
    return {name: value / total for name, value in weights.items()}


# The fields each judgment reads, in the compact order they are sent.
_STATE_KEYS = (
    "objective",
    "player",
    "trajectory",
    "hazard",
    "terrain",
    "reaction_timing",
    "recent_control",
    "episode",
)
_PLAYER_KEYS = (
    "x",
    "y",
    "horizontal_speed_px_per_frame",
    "grounded",
    "jump_phase",
    "powerup_status",
)
_TRAJECTORY_KEYS = ("airborne_frames", "crossing_known_gap", "gap_width_at_commit_tiles")


def _round(value: Any) -> Any:
    """One decimal is plenty for a decision, and saves tokens in the prompt."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    return round(float(value), 1)


def _compact_state(state: Mapping[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {"objective": state.get("objective")}
    player = state.get("player", {})
    compact["player"] = {k: _round(player.get(k)) for k in _PLAYER_KEYS if k in player}
    trajectory = state.get("trajectory", {})
    compact["trajectory"] = {k: _round(trajectory.get(k)) for k in _TRAJECTORY_KEYS}
    # hazard and terrain are already small structured dictionaries; keep them whole
    compact["hazard"] = state.get("hazard", {})
    compact["terrain"] = state.get("terrain", {})
    compact["reaction_timing"] = state.get("reaction_timing", {})
    compact["recent_control"] = state.get("recent_control", {})
    compact["episode"] = state.get("episode", {})
    return compact


class LayaPolicy:
    """Local Laya over HTTP; the Policy protocol, minus the API key."""

    def __init__(
        self,
        url: str = DEFAULT_URL,
        timeout_s: float = 20.0,
        prior_t: float = PRIOR_T,
        stall_break: bool = False,
        stall_threshold: int = STALL_BREAK_THRESHOLD,
    ) -> None:
        self.url = url.rstrip("/") + "/v1/systemone"
        self.timeout_s = timeout_s
        self.prior_t = prior_t
        self.stall_break = stall_break
        self.stall_threshold = stall_threshold
        self.calls = 0
        self.latencies_ms: list[float] = []
        self.overruled = 0  # decisions where the prior beat Laya's own top pick
        self.stall_overrides = 0  # decisions taken by the scripted stall recovery
        self._stall_index = 0  # steps already spent in the current stall
        # The sidecar is on loopback, but this shell may export http_proxy, and
        # urllib would then send even 127.0.0.1 through it (a proxy answers 502
        # for the loopback port instead of connecting). Never proxy these calls.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def close(self) -> None:  # mirrors TypeSafePolicy
        return None

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=self.timeout_s) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:  # the sidecar answered, but with an error
            detail = error.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(
                f"Laya returned HTTP {error.code} for {self.url}: {detail}"
            ) from error
        except urllib.error.URLError as error:  # nothing listening, or it died mid-run
            raise RuntimeError(
                f"Laya is not reachable at {self.url} ({error.reason}). Start the sidecar "
                f"(cd ~/Downloads/dev/jev-tetris && ./play-local.sh status) or pass "
                f"--laya-url."
            ) from error

    def choose(self, snapshot: MarioSnapshot, actions: Sequence[Action]) -> Decision:
        criteria = {action.value: ACTION_TEXT[action] for action in actions}
        questions = {
            "next_action": {
                "type": "choice",
                "instructions": INSTRUCTIONS,
                "criteria": criteria,
            },
            "jump_needed": {"type": "noul", "instructions": JUMP_INSTRUCTIONS},
            "danger": {
                "type": "score",
                "instructions": DANGER_INSTRUCTIONS,
                "criteria": DANGER_LEVELS,
            },
        }
        compact_state = _compact_state(snapshot.to_state())
        payload = {"state": compact_state, "questions": questions}

        started = time.perf_counter()
        answers = self._post(payload)["answers"]
        latency_ms = (time.perf_counter() - started) * 1000
        self.calls += 1
        self.latencies_ms.append(latency_ms)

        choice = answers.get("next_action", {})
        probabilities = {
            str(key): float(value) for key, value in (choice.get("probabilities") or {}).items()
        }
        laya_choice = str(choice.get("choice") or "")
        allowed = {action.value for action in actions}

        # Advance the scripted recovery cycle, if the run is stalled.
        stall = (
            stall_break_action(
                snapshot.stalled_steps,
                self.stall_threshold,
                self._stall_index,
                {action for action in actions},
            )
            if self.stall_break
            else None
        )
        self._stall_index = self._stall_index + 1 if stall is not None else 0

        combined = _combine_with_prior(probabilities, compact_state, self.prior_t)
        if combined:
            probabilities = combined
        if self.prior_t > 0 and combined:
            picked = max(combined, key=lambda name: combined[name])
            if picked != laya_choice:
                self.overruled += 1
        else:
            # Prior off means "Laya alone", so the model's stated choice wins over
            # the argmax of its own distribution -- the two can disagree when the
            # distribution is flat, and letting dict order decide would put a
            # non-Laya pick behind source="laya".
            picked = laya_choice or (
                max(combined, key=lambda name: combined[name]) if combined else ""
            )

        jump_needed = _maybe_float(answers.get("jump_needed", {}).get("noul"))
        danger = _maybe_float(answers.get("danger", {}).get("score"))

        if stall is not None:
            # A scripted decision, not a model answer: no probabilities, no
            # confidence, and `laya_choice` still records what Laya said.
            self.stall_overrides += 1
            return Decision(
                action=stall,
                confidence=0.0,
                probabilities={},
                latency_ms=latency_ms,
                jump_needed_probability=jump_needed,
                danger_score=danger,
                source="stall_break",
                laya_choice=laya_choice or None,
            )

        source = "laya"
        if picked not in allowed:  # no usable answer: fall back, and say so
            picked = (
                Action.RIGHT_RUN.value if Action.RIGHT_RUN.value in allowed else actions[0].value
            )
            probabilities = {}
            confidence = 0.0
            source = "fallback"
        else:
            confidence = float(choice.get("confidence") or 0.0)
            if probabilities:
                confidence = float(probabilities[picked])

        return Decision(
            action=Action(picked),
            confidence=confidence,
            probabilities=probabilities,
            latency_ms=latency_ms,
            jump_needed_probability=jump_needed,
            danger_score=danger,
            source=source,
            laya_choice=laya_choice or None,
        )


def _maybe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
