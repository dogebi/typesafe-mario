from __future__ import annotations

import dataclasses
import unittest
from typing import Any

from typesafe_mario.actions import Action
from typesafe_mario.laya import (
    STALL_BREAK_CYCLE,
    LayaPolicy,
    _combine_with_prior,
    stall_break_action,
)
from typesafe_mario.state import MarioSnapshot, MarioStateParser

ALL_ACTIONS = tuple(Action)


def make_snapshot(stalled: int = 0) -> MarioSnapshot:
    """A parsed snapshot with `stalled_steps` forced to a known value."""
    parser = MarioStateParser()
    info: dict[str, object] = {
        "world": 1,
        "stage": 1,
        "area": 1,
        "x_pos": 100,
        "y_pos": 80,
        "y_pixel": 80,
        "progress": 100,
        "time": 390,
        "life": 2,
        "status": "small",
    }
    return dataclasses.replace(parser.parse(info), stalled_steps=stalled)


class StubPolicy(LayaPolicy):
    """LayaPolicy with the sidecar replaced by a canned answer."""

    def __init__(
        self, choice: str, probabilities: dict[str, float] | None = None, **kwargs: Any
    ) -> None:
        super().__init__(url="http://127.0.0.1:1", **kwargs)
        self.choice = choice
        self.probabilities = probabilities or {a.value: 1.0 / len(ALL_ACTIONS) for a in ALL_ACTIONS}
        self.sent: list[dict] = []

    def _post(self, payload: dict) -> dict:
        self.sent.append(payload)
        return {
            "answers": {
                "next_action": {
                    "choice": self.choice,
                    "probabilities": self.probabilities,
                    "confidence": 0.42,
                },
                "jump_needed": {"noul": 0.9},
                "danger": {"score": 1.5},
            }
        }


class StallBreakActionTests(unittest.TestCase):
    def test_disabled_threshold_never_fires(self) -> None:
        self.assertIsNone(stall_break_action(50, 0, 0, set(ALL_ACTIONS)))
        self.assertIsNone(stall_break_action(-1, 0, 0, set(ALL_ACTIONS)))

    def test_below_threshold_leaves_decision_to_laya(self) -> None:
        self.assertIsNone(stall_break_action(1, 2, 0, set(ALL_ACTIONS)))

    def test_cycle_advances_one_step_per_decision(self) -> None:
        actions = [
            stall_break_action(2, 2, index, set(ALL_ACTIONS))
            for index in range(len(STALL_BREAK_CYCLE))
        ]
        self.assertEqual(actions, list(STALL_BREAK_CYCLE))

    def test_cycle_wraps_around(self) -> None:
        self.assertEqual(
            stall_break_action(9, 2, len(STALL_BREAK_CYCLE), set(ALL_ACTIONS)),
            STALL_BREAK_CYCLE[0],
        )

    def test_unavailable_action_falls_back_to_laya(self) -> None:
        self.assertIsNone(stall_break_action(5, 2, 0, {Action.RIGHT_RUN}))


class LayaPolicyProvenanceTests(unittest.TestCase):
    def test_plain_decision_is_labeled_laya(self) -> None:
        policy = StubPolicy("right_run", prior_t=0.0)
        decision = policy.choose(make_snapshot(stalled=0), ALL_ACTIONS)

        self.assertEqual(decision.action, Action.RIGHT_RUN)
        self.assertEqual(decision.source, "laya")
        self.assertEqual(decision.laya_choice, "right_run")
        self.assertTrue(decision.probabilities)
        self.assertEqual(decision.jump_needed_probability, 0.9)
        self.assertEqual(decision.danger_score, 1.5)

    def test_prior_re_ranks_without_losing_layas_answer(self) -> None:
        # The prior ranks blocked terrain as right_run_jump; Laya said left.
        state = {"player": {"grounded": True}, "terrain": {"obstacle_ahead": True}}
        probabilities = {"left": 0.6, "right_run_jump": 0.4}
        combined = _combine_with_prior(probabilities, state, 0.4)

        self.assertEqual(max(combined, key=lambda name: combined[name]), "right_run_jump")

        policy = StubPolicy("left", probabilities=probabilities, prior_t=0.4)
        decision = policy.choose(make_snapshot(0), ALL_ACTIONS)

        self.assertEqual(decision.source, "laya")
        self.assertEqual(decision.laya_choice, "left")

    def test_unknown_choice_is_labeled_fallback(self) -> None:
        policy = StubPolicy("moonwalk", probabilities={"moonwalk": 1.0}, prior_t=0.0)
        decision = policy.choose(make_snapshot(0), ALL_ACTIONS)

        self.assertEqual(decision.action, Action.RIGHT_RUN)
        self.assertEqual(decision.source, "fallback")
        self.assertEqual(decision.confidence, 0.0)
        self.assertEqual(decision.probabilities, {})

    def test_prior_off_honours_laya_over_its_own_argmax(self) -> None:
        # A flat distribution must not let dict order decide behind source="laya".
        policy = StubPolicy("right_jump", probabilities={"noop": 0.5, "right_jump": 0.5}, prior_t=0.0)
        decision = policy.choose(make_snapshot(0), ALL_ACTIONS)

        self.assertEqual(decision.action, Action.RIGHT_JUMP)
        self.assertEqual(decision.source, "laya")
        self.assertEqual(decision.laya_choice, "right_jump")

    def test_stall_break_overrides_laya_and_is_labeled(self) -> None:
        policy = StubPolicy("right_run_jump", prior_t=0.0, stall_break=True, stall_threshold=2)
        decision = policy.choose(make_snapshot(stalled=5), ALL_ACTIONS)

        self.assertEqual(decision.action, Action.LEFT)
        self.assertEqual(decision.source, "stall_break")
        self.assertEqual(decision.confidence, 0.0)
        self.assertEqual(decision.probabilities, {})
        # The model's own answer is still recorded, and it is still asked.
        self.assertEqual(decision.laya_choice, "right_run_jump")
        self.assertEqual(policy.calls, 1)
        self.assertEqual(policy.stall_overrides, 1)

    def test_stall_break_off_by_default(self) -> None:
        policy = StubPolicy("right_run_jump", prior_t=0.0)
        decision = policy.choose(make_snapshot(stalled=99), ALL_ACTIONS)

        self.assertEqual(decision.action, Action.RIGHT_RUN_JUMP)
        self.assertEqual(decision.source, "laya")

    def test_stall_cycle_advances_then_resets(self) -> None:
        policy = StubPolicy("right_run_jump", prior_t=0.0, stall_break=True, stall_threshold=2)
        picked = [policy.choose(make_snapshot(stalled=4), ALL_ACTIONS).action for _ in range(3)]
        self.assertEqual(picked, [Action.LEFT, Action.LEFT, Action.RIGHT_RUN])

        # A decision with no stall clears the cycle, so the next stall restarts it.
        policy.choose(make_snapshot(stalled=0), ALL_ACTIONS)
        self.assertEqual(policy.choose(make_snapshot(stalled=4), ALL_ACTIONS).action, Action.LEFT)


if __name__ == "__main__":
    unittest.main()
