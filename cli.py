
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .env import AuditEnv
from .policies import (
    MarkAllPolicy,
    RandomInvestigator,
    ReconciliationPolicy,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run the offline Audit RL simulator"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--policy",
        choices=["mark-all", "random", "reconcile"],
        default="random",
    )
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--interacting", action="store_true")
    parser.add_argument("--output", default="trajectories.jsonl")
    args = parser.parse_args()

    cfg = {
        "clean": args.clean,
        "interacting": args.interacting,
    }

    env = AuditEnv(reveal_reward=True)
    observation, reset_info = env.reset(args.seed, cfg)

    policies = {
        "mark-all": MarkAllPolicy(),
        "random": RandomInvestigator(args.seed),
        "reconcile": ReconciliationPolicy(),
    }

    policy = policies[args.policy]
    episode_id = observation.get("manifest", {}).get(
        "episode_id",
        f"ep-{args.seed:08d}",
    )

    trajectory = []

    for n in range(100):
        action = policy.act(observation)

        (
            new_observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        trajectory.append({
            "episode_id": episode_id,
            "step": n + 1,
            "action": action,
            "observation": new_observation,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        })

        observation = new_observation

        if terminated or truncated:
            break

    Path(args.output).write_text(
        "".join(
            json.dumps(item, ensure_ascii=False) + "\n"
            for item in trajectory
        ),
        encoding="utf-8",
    )

    print(json.dumps({
        "policy": args.policy,
        "steps": len(trajectory),
        "last_reward": trajectory[-1]["reward"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
