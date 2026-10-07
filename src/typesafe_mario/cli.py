from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .laya import DEFAULT_URL as LAYA_DEFAULT_URL
from .laya import PRIOR_T, STALL_BREAK_THRESHOLD, LayaPolicy
from .policy import HeuristicPolicy, TypeSafePolicy
from .runner import run_episode
from .state import MarioStateParser


def _demo_ram() -> bytearray:
    ram = bytearray(0x0800)
    # Mario position and one goomba ahead.
    ram[0x006D] = 0
    ram[0x0086] = 172
    ram[0x000F] = 1
    ram[0x0016] = 0x06
    ram[0x006E] = 0
    ram[0x0087] = 214
    ram[0x00CF] = 79
    # Fill the tile-map row immediately below Mario with solid ground.
    ground_row = (79 + 16 - 32) // 16
    for column in range(16):
        ram[0x0500 + ground_row * 16 + column] = 1
    return ram


def state_demo() -> int:
    info = {
        "world": 1,
        "stage": 1,
        "area": 1,
        "x_pos": 172,
        "y_pos": 79,
        "y_pixel": 79,
        "left_x_pos": 60,
        "progress": 172,
        "progress_max": 172,
        "status": "small",
        "player_state": 8,
        "life": 2,
        "coins": 0,
        "score": 200,
        "time": 387,
        "death": False,
        "clear": False,
    }
    snapshot = MarioStateParser().parse(info, _demo_ram(), previous_action="right")
    print(json.dumps(snapshot.to_state(), indent=2))
    print("\n--- TEXT VIEW ---\n")
    print(snapshot.to_text())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="typesafe-mario")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("state-demo", help="Show structured and text state without API calls")

    play = subparsers.add_parser("play", help="Run a Mario episode")
    play.add_argument("--env", default="SuperMarioBros-1-1-v0")
    play.add_argument(
        "--frames-per-decision",
        type=int,
        default=8,
        help="Minimum macro duration in emulator frames",
    )
    play.add_argument("--max-decisions", type=int, default=2000)
    play.add_argument("--seed", type=int, default=123)
    play.add_argument("--policy", choices=("typesafe", "laya", "heuristic"), default="typesafe")
    play.add_argument(
        "--laya-url",
        default=LAYA_DEFAULT_URL,
        help="Laya sidecar base URL (ai/laya_mlx_server.py in the tetris project)",
    )
    play.add_argument(
        "--laya-prior-t",
        type=float,
        default=None,
        help="Temperature of the code-side action prior mixed with Laya's "
        "probabilities; 0 disables it and leaves Laya alone (default LAYA_PRIOR_T or 0.4)",
    )
    play.add_argument(
        "--laya-stall-break",
        action="store_true",
        help="On a stall, bypass Laya with a scripted back-off/runway/jump cycle "
        "(runway is required for tall pipes; Laya holds a standstill jump instead). "
        "Forced decisions are logged with source=stall_break.",
    )
    play.add_argument(
        "--laya-stall-threshold",
        type=int,
        default=STALL_BREAK_THRESHOLD,
        help="Stalled decisions before --laya-stall-break takes over "
        f"(default LAYA_STALL_THRESHOLD or {STALL_BREAK_THRESHOLD})",
    )
    play.add_argument(
        "--display",
        choices=("dashboard", "game", "none"),
        default="dashboard",
        help="Combined telemetry dashboard, plain game window, or headless mode",
    )
    play.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    play.add_argument(
        "--screenshot",
        type=Path,
        help="Save the first populated dashboard frame as a PNG",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "state-demo":
        return state_demo()
    if args.command == "play":
        if args.policy == "typesafe":
            policy = TypeSafePolicy()
        elif args.policy == "laya":
            policy = LayaPolicy(
                args.laya_url,
                prior_t=args.laya_prior_t if args.laya_prior_t is not None else PRIOR_T,
                stall_break=args.laya_stall_break,
                stall_threshold=args.laya_stall_threshold,
            )
        else:
            policy = HeuristicPolicy()
        log_path = run_episode(
            env_id=args.env,
            policy=policy,
            frames_per_decision=args.frames_per_decision,
            max_decisions=args.max_decisions,
            seed=args.seed,
            artifacts_dir=args.artifacts_dir,
            display=args.display,
            screenshot_path=args.screenshot,
        )
        print(f"Run log: {log_path.resolve()}")
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
