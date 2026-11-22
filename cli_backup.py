from __future__ import annotations

import argparse
import json
from pathlib import Path

from .env import AuditEnv
from .policies import OraclePolicy, MarkAllPolicy, RandomInvestigator, ReconciliationPolicy


def main():
    parser = argparse.ArgumentParser(description="Run the offline Audit RL simulator")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--policy", choices=["oracle", "mark-all", "random", "reconcile"], default="oracle")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--interacting", action="store_true")
    parser.add_argument("--output", default="trajectories.jsonl")
    args = parser.parse_args()

    cfg = {"clean": args.clean, "interacting": args.interacting}
    env = AuditEnv(reveal_reward=True)
    obs, _ = env.reset(args.seed, cfg)
    policies = {
        "oracle": OraclePolicy(), "mark-all": MarkAllPolicy(),
        "random": RandomInvestigator(args.seed), "reconcile": ReconciliationPolicy()
    }
    policy = policies[args.policy]
    trajectory = []
    for n in range(100):
        action = policy.act(env)
        new_obs, reward, terminated, truncated, info = env.step(action)
        trajectory.append({
            "episode_id": env.public["manifest.json"]["episode_id"],
            "step": n + 1, "action": action, "observation": new_obs,
            "reward": reward, "terminated": terminated, "truncated": truncated,
        })
        if terminated or truncated:
            break
    Path(args.output).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in trajectory), encoding="utf-8")
    print(json.dumps({"steps": len(trajectory), "last_reward": trajectory[-1]["reward"],
                      "closed_metrics": env.evaluator_result()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
