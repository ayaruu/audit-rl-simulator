
"""Baseline policies for the Audit RL simulator.

Normal policies interact only through environment actions and observations.
OraclePolicy is reserved for evaluator-side upper-bound checks.
"""

from __future__ import annotations

import json
import random


class OraclePolicy:
    """Evaluator-only upper-bound policy. Never use in normal agent runs."""

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
                "explanation": "Evaluator-only reference finding.",
            })

        return {"tool": "submit", "findings": findings}


class RandomInvestigator:
    """Inspects random public tables, then submits an empty report."""

    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.calls = 0

    def act(self, observation):
        if self.calls >= 8:
            return {"tool": "submit", "findings": []}

        self.calls += 1
        table = self.rng.choice(
            ["ledger", "invoices", "payments", "fx"]
        )
        return {
            "tool": "list_records",
            "table": table,
            "limit": 10,
        }


class MarkAllPolicy:
    """Adversarial baseline: flags the first 30 invoices."""

    def __init__(self):
        self.waiting = False

    def act(self, observation):
        if not self.waiting:
            self.waiting = True
            return {
                "tool": "list_records",
                "table": "invoices",
                "limit": 30,
            }

        result = observation.get("result", {})
        invoices = result.get("records", [])
        findings = []

        for inv in invoices[:30]:
            findings.append({
                "type": "duplicate_invoice",
                "affected_ids": [inv["invoice_id"]],
                "claimed_amount_minor": int(inv["total_minor"]),
                "currency": inv["currency"],
                "evidence_ids": [inv["doc_id"]],
                "explanation": "Flagged by broad heuristic.",
            })

        return {"tool": "submit", "findings": findings}


class ReconciliationPolicy:
    """Rule-based baseline that reads records through list_records."""

    def __init__(self):
        self.table_names = ["invoices", "ledger", "payments"]
        self.table_index = 0
        self.cursor = None
        self.waiting = False
        self.data = {
            "invoices": [],
            "ledger": [],
            "payments": [],
        }
        self.manifest = {}

    def _consume_previous_page(self, observation):
        result = observation.get("result", {})

        if not isinstance(result, dict) or "records" not in result:
            # If a table request failed, move on rather than getting stuck.
            self.table_index += 1
            self.cursor = None
            return

        table = self.table_names[self.table_index]
        self.data[table].extend(result.get("records", []))

        next_cursor = result.get("next_cursor")
        if next_cursor is None:
            self.table_index += 1
            self.cursor = None
        else:
            self.cursor = next_cursor

    def _build_findings(self):
        invoices = self.data["invoices"]
        ledger = self.data["ledger"]
        payments = self.data["payments"]
        findings = []

        # 1. Look for duplicate invoices.
        groups = {}

        for inv in invoices:
            signature = (
                inv.get("entity_id"),
                inv.get("issued_at"),
                inv.get("currency"),
                inv.get("total_minor"),
                json.dumps(
                    inv.get("line_items", []),
                    sort_keys=True,
                ),
            )
            groups.setdefault(signature, []).append(inv)

        for group in groups.values():
            if len(group) < 2:
                continue

            first = group[0]

            for duplicate in group[1:]:
                duplicate_rows = [
                    row for row in ledger
                    if row.get("invoice_id") == duplicate.get("invoice_id")
                ]

                if not duplicate_rows:
                    continue

                findings.append({
                    "type": "duplicate_invoice",
                    "affected_ids": [
                        first["invoice_id"],
                        duplicate["invoice_id"],
                    ],
                    "claimed_amount_minor": int(first["total_minor"]),
                    "currency": first["currency"],
                    "evidence_ids": [
                        first["doc_id"],
                        duplicate["doc_id"],
                        duplicate_rows[0]["row_id"],
                    ],
                    "explanation": (
                        "Invoices have matching supplier, date, currency, "
                        "amount and line items."
                    ),
                })

        # 2. Compare invoice currency with accounts-payable ledger currency.
        for inv in invoices:
            matching_rows = [
                row for row in ledger
                if row.get("invoice_id") == inv.get("invoice_id")
                and row.get("account") == "Accounts Payable"
            ]

            for row in matching_rows:
                if row.get("currency") != inv.get("currency"):
                    findings.append({
                        "type": "wrong_fx",
                        "affected_ids": [
                            inv["invoice_id"],
                            row["row_id"],
                        ],
                        "claimed_amount_minor": int(inv["total_minor"]),
                        "currency": inv["currency"],
                        "evidence_ids": [
                            inv["doc_id"],
                            row["row_id"],
                        ],
                        "explanation": (
                            "Invoice currency differs from the related "
                            "accounts-payable ledger currency."
                        ),
                    })

        # 3. Check accounts-payable posting dates against the period.
        period_start = self.manifest.get("period_start", "")
        period_end = self.manifest.get("period_end", "")

        for inv in invoices:
            matching_rows = [
                row for row in ledger
                if row.get("invoice_id") == inv.get("invoice_id")
                and row.get("account") == "Accounts Payable"
            ]

            for row in matching_rows:
                row_date = str(row.get("timestamp", ""))[:10]

                if row_date and period_start and period_end:
                    if not (period_start <= row_date <= period_end):
                        findings.append({
                            "type": "period_cutoff",
                            "affected_ids": [
                                row["row_id"],
                                inv["invoice_id"],
                            ],
                            "claimed_amount_minor": int(inv["total_minor"]),
                            "currency": inv["currency"],
                            "evidence_ids": [
                                inv["doc_id"],
                                row["row_id"],
                            ],
                            "explanation": (
                                "Accounts-payable posting date is outside "
                                "the reporting period."
                            ),
                        })

        # 4. Look for payments without invoice allocation.
        for payment in payments:
            if payment.get("invoice_refs"):
                continue

            payment_date = str(payment.get("timestamp", ""))[:10]
            amount = int(payment.get("amount_minor", 0))
            currency = payment.get("currency")
            candidates = []

            for inv in invoices:
                if inv.get("currency") != currency:
                    continue

                invoice_date = str(inv.get("issued_at", ""))[:10]
                due_date = str(inv.get("due_at", ""))[:10]
                invoice_amount = int(inv.get("total_minor", 0))

                if not (invoice_date <= payment_date <= due_date):
                    continue

                if amount not in (
                    invoice_amount,
                    invoice_amount // 2,
                ):
                    continue

                candidates.append(inv)

            if len(candidates) == 1:
                inv = candidates[0]
                findings.append({
                    "type": "payment_allocation",
                    "affected_ids": [
                        payment["payment_id"],
                        inv["invoice_id"],
                    ],
                    "claimed_amount_minor": amount,
                    "currency": currency,
                    "evidence_ids": [
                        payment["doc_id"],
                        inv["doc_id"],
                    ],
                    "explanation": (
                        "Payment has no invoice allocation and matches "
                        "one plausible invoice by date, amount and currency."
                    ),
                })

        return findings[:30]

    def act(self, observation):
        # Save the public manifest from the initial observation.
        if not self.manifest:
            self.manifest = observation.get("manifest", {})

        # Process the page returned by the previous tool action.
        if self.waiting:
            self._consume_previous_page(observation)
            self.waiting = False

        # Request the next page of the current table.
        if self.table_index < len(self.table_names):
            table = self.table_names[self.table_index]
            action = {
                "tool": "list_records",
                "table": table,
                "limit": 50,
            }

            if self.cursor is not None:
                action["cursor"] = self.cursor

            self.waiting = True
            return action

        # All required public records have been collected.
        return {
            "tool": "submit",
            "findings": self._build_findings(),
        }


def run_policy(env, policy, seed=0, max_steps=100):
    """Run an observation-based policy and return its trajectory."""

    observation, info = env.reset(seed=seed)
    manifest = observation.get("manifest", {})
    episode_id = manifest.get("episode_id", f"ep-{seed:08d}")
    trajectory = []

    for step_no in range(max_steps):
        action = policy.act(observation)
        next_observation, reward, terminated, truncated, step_info = (
            env.step(action)
        )

        trajectory.append({
            "episode_id": episode_id,
            "step": step_no + 1,
            "action": action,
            "observation": next_observation,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        })

        observation = next_observation

        if terminated or truncated:
            break

    return trajectory
