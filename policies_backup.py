"""Small baseline policies and offline runner helpers."""

from __future__ import annotations

import random


class OraclePolicy:
    """Upper-bound policy for local evaluation only."""

    def act(self, env):
        truth = env.export_closed()["truth"]
        findings = []

        for issue in truth["issues"]:
            evidence = issue["acceptable_evidence_sets"][0]
            findings.append({
                "type": issue["type"],
                "affected_ids": issue["affected_ids"],
                "claimed_amount_minor": issue["expected_amount_minor"],
                "currency": issue["currency"],
                "evidence_ids": evidence,
                "explanation": "Matched the disclosed synthetic evidence set.",
            })

        return {"tool": "submit", "findings": findings}


class MarkAllPolicy:
    """Adversarial policy: accuse every invoice."""

    def act(self, env):
        findings = []

        for inv in env.tables["invoices"][:30]:
            findings.append({
                "type": "duplicate_invoice",
                "affected_ids": [inv["invoice_id"]],
                "claimed_amount_minor": int(inv["total_minor"]),
                "currency": inv["currency"],
                "evidence_ids": [inv["doc_id"]],
                "explanation": "Flagged by broad heuristic.",
            })

        return {"tool": "submit", "findings": findings}


class RandomInvestigator:
    """Randomly inspects records, then submits an empty report."""

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def act(self, env):
        if env.steps >= 8:
            return {"tool": "submit", "findings": []}

        choices = ["ledger", "invoices", "payments", "fx"]
        table = self.rng.choice(choices)

        return {
            "tool": "list_records",
            "table": table,
            "limit": 10,
        }


class ReconciliationPolicy:
    """Conservative baseline: inspect tables, then submit silence."""

    def act(self, env):
        if env.steps < 4:
            table = ["invoices", "payments", "ledger", "fx"][env.steps]
            return {
                "tool": "list_records",
                "table": table,
                "limit": 50,
            }

        return {"tool": "submit", "findings": []}


def run_policy(env, policy, seed=0, max_steps=100):
    obs, info = env.reset(seed=seed)
    trajectory = []

    for step_no in range(max_steps):
        action = policy.act(env)

        next_obs, reward, terminated, truncated, step_info = env.step(action)

        trajectory.append({
            "episode_id": (
                obs["manifest"]["episode_id"]
                if "manifest" in obs
                else f"ep-{seed:08d}"
            ),
            "step": step_no + 1,
            "action": action,
            "observation": next_obs,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        })

        obs = next_obs

        if terminated or truncated:
            break

    return trajectory