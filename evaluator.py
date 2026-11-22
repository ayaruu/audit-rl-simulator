"""Closed-side finding matcher and terminal reward."""

from __future__ import annotations

from typing import Any

TOLERANCE_MINOR = 1
MAX_FINDINGS = 30
MAX_EVIDENCE_IDS = 10
MAX_EXPLANATION_LENGTH = 1000


def _canon(values: Any) -> tuple[str, ...]:
    if not isinstance(values, list):
        return ()
    return tuple(sorted({str(value) for value in values}))


def match_findings(findings: list[dict], truth: list[dict]) -> dict:
    """Match submitted findings to ground-truth issues one-to-one."""

    used_truth: set[int] = set()
    tp = 0
    fp = 0
    matches = []

    for finding_index, finding in enumerate(findings):
        matched = None

        for truth_index, issue in enumerate(truth):
            if truth_index in used_truth:
                continue

            if finding.get("type") != issue.get("type"):
                continue

            if _canon(finding.get("affected_ids")) != _canon(
                issue.get("affected_ids")
            ):
                continue

            if str(finding.get("currency", "")).upper() != str(
                issue.get("currency", "")
            ).upper():
                continue

            try:
                amount_ok = abs(
                    int(finding.get("claimed_amount_minor"))
                    - int(issue.get("expected_amount_minor"))
                ) <= int(issue.get("tolerance_minor", TOLERANCE_MINOR))
            except (TypeError, ValueError, OverflowError):
                amount_ok = False

            if not amount_ok:
                continue

            supplied = set(_canon(finding.get("evidence_ids")))
            acceptable = issue.get("acceptable_evidence_sets", [])

            evidence_ok = (
                isinstance(acceptable, list)
                and any(
                    isinstance(group, list)
                    and set(_canon(group)).issubset(supplied)
                    for group in acceptable
                )
            )

            if not evidence_ok:
                continue

            matched = truth_index
            break

        if matched is None:
            fp += 1
        else:
            tp += 1
            used_truth.add(matched)
            matches.append(
                {
                    "finding_index": finding_index,
                    "issue_index": matched,
                }
            )

    fn = len(truth) - tp

    if tp + fp == 0 and tp + fn == 0:
        precision = recall = f1 = 1.0
    else:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matches": matches,
    }


def terminal_reward(f1: float, inspection_steps: int) -> float:
    raw = 2.0 * float(f1) - 1.0 - 0.002 * int(inspection_steps)
    return max(-1.0, min(1.0, raw))


def evaluate_submission(
    payload: dict,
    truth_doc: dict,
    inspection_steps: int,
) -> dict:
    """Validate a submission and calculate its terminal reward."""

    if not isinstance(payload, dict):
        return {
            "valid": False,
            "reward": -1.0,
            "error": "submission must be an object",
        }

    findings = payload.get("findings")

    if not isinstance(findings, list):
        return {
            "valid": False,
            "reward": -1.0,
            "error": "findings must be a list",
        }

    if len(findings) > MAX_FINDINGS:
        return {
            "valid": False,
            "reward": -1.0,
            "error": "too many findings",
        }

    for item in findings:
        if not isinstance(item, dict):
            return {
                "valid": False,
                "reward": -1.0,
                "error": "each finding must be an object",
            }

        required = {
            "type",
            "affected_ids",
            "claimed_amount_minor",
            "currency",
            "evidence_ids",
            "explanation",
        }

        if not required.issubset(item):
            return {
                "valid": False,
                "reward": -1.0,
                "error": "finding missing required fields",
            }

        if not isinstance(item["affected_ids"], list):
            return {
                "valid": False,
                "reward": -1.0,
                "error": "affected_ids must be a list",
            }

        if not isinstance(item["evidence_ids"], list):
            return {
                "valid": False,
                "reward": -1.0,
                "error": "evidence_ids must be a list",
            }

        if len(item["evidence_ids"]) > MAX_EVIDENCE_IDS:
            return {
                "valid": False,
                "reward": -1.0,
                "error": "too many evidence IDs",
            }

        if not isinstance(item["explanation"], str):
            return {
                "valid": False,
                "reward": -1.0,
                "error": "explanation must be a string",
            }

        if len(item["explanation"]) > MAX_EXPLANATION_LENGTH:
            return {
                "valid": False,
                "reward": -1.0,
                "error": "explanation too long",
            }

    if payload.get("episode_id") != truth_doc.get("episode_id"):
        return {
            "valid": False,
            "reward": -1.0,
            "error": "episode_id mismatch",
        }

    metrics = match_findings(findings, truth_doc.get("issues", []))

    return {
        "valid": True,
        **metrics,
        "reward": terminal_reward(metrics["f1"], inspection_steps),
    }