"""JSON-compatible, Gymnasium-style environment without Gymnasium dependency."""

from __future__ import annotations

import ast
import csv
import io
import json
import operator
import time
from decimal import Decimal, InvalidOperation
from typing import Any

from .evaluator import evaluate_submission
from .generator import generate_episode


class SafeCalculator:
    OPS = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.USub: operator.neg, ast.UAdd: operator.pos,
    }

    @classmethod
    def calculate(cls, expression: str) -> str:
        if not isinstance(expression, str) or len(expression) > 200:
            raise ValueError("invalid expression")
        tree = ast.parse(expression, mode="eval")
        def visit(node):
            if isinstance(node, ast.Expression):
                return visit(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return Decimal(str(node.value))
            if isinstance(node, ast.BinOp) and type(node.op) in cls.OPS:
                return cls.OPS[type(node.op)](visit(node.left), visit(node.right))
            if isinstance(node, ast.UnaryOp) and type(node.op) in cls.OPS:
                return cls.OPS[type(node.op)](visit(node.operand))
            raise ValueError("only decimal arithmetic is allowed")
        try:
            result = visit(tree)
        except (InvalidOperation, ZeroDivisionError, ArithmeticError) as exc:
            raise ValueError("calculation error") from exc
        if not result.is_finite():
            raise ValueError("non-finite result")
        return str(result)


class AuditEnv:
    TABLES = {"ledger", "invoices", "payments", "fx"}
    MAX_STEPS = 100
    MAX_PAGE = 50
    TIME_LIMIT_SECONDS = 60

    def __init__(self, reveal_reward: bool = True):
        self.reveal_reward = bool(reveal_reward)
        self.reset(0)

    @staticmethod
    def _read_csv(text: str) -> list[dict]:
        return list(csv.DictReader(io.StringIO(text)))

    @staticmethod
    def _read_jsonl(text: str) -> list[dict]:
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    def reset(self, seed: int = 0, config: dict | None = None):
        episode = generate_episode(seed, config)
        self.public = episode["public"]
        self._truth = episode["closed"]["truth.json"]
        self._provenance = episode["closed"]["provenance.json"]
        self.tables = {
            "ledger": self._read_csv(self.public["ledger.csv"]),
            "invoices": self._read_jsonl(self.public["invoices.jsonl"]),
            "payments": self._read_jsonl(self.public["payments.jsonl"]),
            "fx": self._read_csv(self.public["fx.csv"]),
        }
        self.documents = {
            x["doc_id"]: x for x in self.tables["invoices"] + self.tables["payments"]
        }
        self.steps = 0
        self.started = time.monotonic()
        self.done = False
        self.last_submission = None
        obs = {
            "task": "Inspect synthetic company records and submit evidence-backed audit findings.",
            "manifest": self.public["manifest.json"],
            "policy": self.public["policy.json"],
            "remaining_steps": self.MAX_STEPS,
        }
        return obs, {"version": "0.1.0"}

    def _remaining(self):
        return max(0, self.MAX_STEPS - self.steps)

    def _finish_truncated(self):
        self.done = True
        return {"result": None, "remaining_steps": 0}, 0.0, False, True, {}

    def step(self, action: dict):
        if self.done:
            return {"error": "episode already finished", "remaining_steps": 0}, 0.0, True, False, {}
        if time.monotonic() - self.started > self.TIME_LIMIT_SECONDS:
            return self._finish_truncated()
        if self.steps >= self.MAX_STEPS:
            return self._finish_truncated()

        self.steps += 1
        result = None
        try:
            if not isinstance(action, dict):
                raise ValueError("action must be an object")
            tool = action.get("tool")
            if tool == "list_records":
                table = action.get("table")
                if table not in self.TABLES:
                    raise ValueError("unknown table")
                limit = int(action.get("limit", 20))
                if limit < 1 or limit > self.MAX_PAGE:
                    raise ValueError("limit must be between 1 and 50")
                cursor = action.get("cursor")
                offset = 0 if cursor in (None, "") else int(cursor)
                if offset < 0 or offset > len(self.tables[table]):
                    raise ValueError("invalid cursor")
                rows = self.tables[table][offset:offset+limit]
                next_offset = offset + len(rows)
                result = {"records": rows, "next_cursor": str(next_offset) if next_offset < len(self.tables[table]) else None}
            elif tool == "get_record":
                table, record_id = action.get("table"), str(action.get("id", ""))
                if table not in self.TABLES:
                    raise ValueError("unknown table")
                key = {"ledger": "row_id", "invoices": "invoice_id",
                       "payments": "payment_id", "fx": "date"}[table]
                matches = [r for r in self.tables[table] if str(r.get(key)) == record_id]
                if not matches:
                    raise ValueError("record not found")
                result = {"records": matches[:10]}
            elif tool == "get_document":
                doc_id = str(action.get("doc_id", ""))
                if doc_id not in self.documents:
                    raise ValueError("document not found")
                result = {"document": self.documents[doc_id]}
            elif tool == "calculator":
                result = {"value": SafeCalculator.calculate(action.get("expression", ""))}
            elif tool == "submit":
                payload = action.get("findings")
                submission = {"episode_id": self.public["manifest.json"]["episode_id"],
                              "findings": payload}
                if not isinstance(payload, list):
                    raise ValueError("findings must be a list")
                judged = evaluate_submission(submission, self._truth, self.steps)
                self.last_submission = judged
                self.done = True
                reward = judged["reward"] if self.reveal_reward else 0.0
                info = {"submission_status": "accepted" if judged["valid"] else "invalid"}
                if self.reveal_reward:
                    info["metrics"] = {k: judged[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1")}
                return {"result": "submitted", "remaining_steps": self._remaining()}, reward, True, False, info
            else:
                raise ValueError("unknown action")
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            result = {"error": str(exc)[:160]}

        if time.monotonic() - self.started > self.TIME_LIMIT_SECONDS:
            return self._finish_truncated()
        if self.steps >= self.MAX_STEPS:
            self.done = True
            return {"result": None, "remaining_steps": 0}, 0.0, False, True, {}
        return {"result": result, "remaining_steps": self._remaining()}, 0.0, False, False, {}

    def evaluator_result(self):
        """Evaluator-only helper. Do not expose this method to an agent process."""
        return self.last_submission

    def export_closed(self):
        """For offline dataset/evaluator tooling only; never call from agent adapters."""
        return {"truth": self._truth, "provenance": self._provenance}

    def export_public(self):
        return self.public
