"""Deterministic synthetic episode generator.

All records are fictional. Generation happens before the agent receives the
episode. The returned truth/provenance objects must remain evaluator-only.
"""

from __future__ import annotations

import csv
import io
import json
import random
from datetime import datetime, timedelta, timezone
from typing import Any

COMPANIES = ["Northstar Components", "Blue Cedar Trading", "Orion Paperworks",
             "Steppe Logistics", "Amberline Foods", "Caspian Office Supply"]
NAMES = ["Aster", "Boreal", "Cinder", "Delta", "Elm", "Fable", "Garnet", "Harbor"]
CURRENCIES = ["USD", "EUR", "KZT"]
MINOR_UNITS = {"USD": 2, "EUR": 2, "KZT": 2}


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows)


def generate_episode(seed: int, config: dict | None = None) -> dict:
    """Return public files plus evaluator-only truth/provenance.

    config: issue_types, clean, interacting, period_start, period_end,
            ledger_rows (200..2000), document_count target (20..200).
    """
    cfg = dict(config or {})
    rng = random.Random(int(seed))
    episode_id = f"ep-{int(seed):08d}"
    base = str(cfg.get("base_currency", "USD")).upper()
    if base not in CURRENCIES:
        base = "USD"
    start = datetime.fromisoformat(cfg.get("period_start", "2026-01-01T00:00:00+00:00"))
    end = datetime.fromisoformat(cfg.get("period_end", "2026-01-31T23:59:59+00:00"))
    if start.tzinfo is None: start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None: end = end.replace(tzinfo=timezone.utc)
    tz_name = str(cfg.get("timezone", "UTC"))

    issue_types = list(cfg.get("issue_types", [
        "duplicate_invoice", "payment_allocation", "wrong_fx", "period_cutoff"
    ]))
    if cfg.get("clean", False):
        issue_types = []
    if cfg.get("interacting", False):
        issue_types = ["payment_allocation", "wrong_fx", "period_cutoff"]
    issue_types = [x for x in issue_types if x in {
        "duplicate_invoice", "payment_allocation", "wrong_fx", "period_cutoff"
    }][:4]

    target_rows = max(200, min(2000, int(cfg.get("ledger_rows", 220))))
    company = rng.choice(COMPANIES)
    entities = [f"ENT-{rng.randrange(100000,999999)}" for _ in range(8)]
    ledger: list[dict] = []
    invoices: list[dict] = []
    payments: list[dict] = []
    fx: list[dict] = []
    issues: list[dict] = []
    provenance: list[dict] = []
    docs: dict[str, dict] = {}

    # Daily FX table; decimal strings only.
    rates = {"USD": "1.000000", "EUR": "1.080000", "KZT": "0.002100"}
    for offset in range(-5, 45):
        day = (start + timedelta(days=offset)).date().isoformat()
        for cur in CURRENCIES:
            fx.append({"date": day, "source_currency": cur, "target_currency": base,
                       "rate": rates[cur] if cur != base else "1.000000"})

    def add_ledger(account, debit, credit, currency, entity, invoice_id=None, payment_id=None, timestamp=None):
        row_id = f"LED-{rng.randrange(10**8,10**9)}"
        ledger.append({
            "row_id": row_id, "timestamp": timestamp or _iso(start + timedelta(
                seconds=rng.randrange(max(1, int((end-start).total_seconds()) + 1)))),
            "account": account, "debit_minor": int(debit), "credit_minor": int(credit),
            "currency": currency, "entity_id": entity, "invoice_id": invoice_id,
            "payment_id": payment_id,
        })
        return row_id

    # Coherent invoices and matched payments. Split payments are deliberately common.
    n_invoices = max(18, min(70, target_rows // 8))
    for i in range(n_invoices):
        inv_id = f"INV-{rng.randrange(10**7,10**8)}"
        entity = rng.choice(entities)
        cur = rng.choice(CURRENCIES)
        amount = rng.randrange(15000, 900000)
        issued = start + timedelta(days=rng.randrange(max(1, (end-start).days + 1)))
        inv = {
            "doc_id": f"DOC-I-{i:04d}", "invoice_id": inv_id, "entity_id": entity,
            "issued_at": _iso(issued), "due_at": _iso(issued + timedelta(days=30)),
            "currency": cur, "total_minor": amount,
            "line_items": [{"description": rng.choice(["Consulting", "Materials", "Freight", "Services"]),
                            "quantity": 1, "unit_price_minor": amount}],
        }
        invoices.append(inv); docs[inv["doc_id"]] = inv
        add_ledger("Operating Expense", amount, 0, cur, entity, invoice_id=inv_id)
        add_ledger("Accounts Payable", 0, amount, cur, entity, invoice_id=inv_id)

        # Usually pay full amount; sometimes split into two legitimate payments.
        parts = [amount] if rng.random() < 0.65 else [amount // 2, amount - amount // 2]
        for p, part in enumerate(parts):
            pay_id = f"PAY-{rng.randrange(10**7,10**8)}"
            pay = {
                "doc_id": f"DOC-P-{i:04d}-{p}", "payment_id": pay_id,
                "timestamp": _iso(issued + timedelta(days=rng.randrange(1, 25))),
                "currency": cur, "amount_minor": part, "invoice_refs": [inv_id],
            }
            payments.append(pay); docs[pay["doc_id"]] = pay
            add_ledger("Accounts Payable", part, 0, cur, entity, invoice_id=inv_id,
                       payment_id=pay_id)
            add_ledger("Cash", 0, part, cur, entity, invoice_id=inv_id,
                       payment_id=pay_id)

    # Pad to requested ledger scale with balanced, harmless bookkeeping entries.
    while len(ledger) < target_rows:
        cur = rng.choice(CURRENCIES); entity = rng.choice(entities)
        amt = rng.randrange(1000, 50000)
        ts = _iso(start + timedelta(seconds=rng.randrange(max(1, int((end-start).total_seconds()) + 1))))
        add_ledger("Miscellaneous Expense", amt, 0, cur, entity, timestamp=ts)
        add_ledger("Accrued Liabilities", 0, amt, cur, entity, timestamp=ts)

    # Inject independently configured issues. Keep exact evidence references.
    selected_invoices = rng.sample(invoices, min(len(issue_types), len(invoices)))
    for issue_type, inv in zip(issue_types, selected_invoices):
        inv_id, cur, amount, entity = inv["invoice_id"], inv["currency"], inv["total_minor"], inv["entity_id"]
        if issue_type == "duplicate_invoice":
            duplicate_id = f"{inv_id}-COPY"
            dup = dict(inv)
            dup.update({"doc_id": f"DOC-DUP-{rng.randrange(1000,9999)}", "invoice_id": duplicate_id})
            invoices.append(dup); docs[dup["doc_id"]] = dup
            # Duplicate ledger expense posting is the financial effect.
            row = add_ledger("Operating Expense", amount, 0, cur, entity,
                             invoice_id=duplicate_id)
            issue = {
                "issue_id": f"ISS-{len(issues)+1}", "type": issue_type,
                "affected_ids": [inv_id, duplicate_id], "expected_amount_minor": amount,
                "currency": cur,
                "acceptable_evidence_sets": [[inv["doc_id"], dup["doc_id"], row]],
                "tolerance_minor": 1,
            }
            provenance.append({"issue_id": issue["issue_id"], "operation": "duplicate_invoice_and_posting",
                               "source_invoice": inv_id, "new_invoice": duplicate_id, "ledger_row": row})
        elif issue_type == "payment_allocation":
            pay = next((p for p in payments if inv_id in p["invoice_refs"]), None)
            if pay is None: continue
            pay["invoice_refs"] = []
            issue = {
                "issue_id": f"ISS-{len(issues)+1}", "type": issue_type,
                "affected_ids": [pay["payment_id"], inv_id],
                "expected_amount_minor": int(pay["amount_minor"]), "currency": cur,
                "acceptable_evidence_sets": [[pay["doc_id"], inv["doc_id"]]],
                "tolerance_minor": 1,
            }
            provenance.append({"issue_id": issue["issue_id"], "operation": "remove_payment_allocation",
                               "payment_id": pay["payment_id"], "invoice_id": inv_id})
        elif issue_type == "wrong_fx":
            # A foreign-currency invoice with a deliberately wrong ledger currency label.
            foreign = next((x for x in CURRENCIES if x != base), None)
            if foreign is None: continue
            inv["currency"] = foreign
            row = next((r for r in ledger if r["invoice_id"] == inv_id and r["account"] == "Accounts Payable"), None)
            if row is None: continue
            row["currency"] = base
            issue = {
                "issue_id": f"ISS-{len(issues)+1}", "type": issue_type,
                "affected_ids": [inv_id, row["row_id"]], "expected_amount_minor": amount,
                "currency": foreign,
                "acceptable_evidence_sets": [[inv["doc_id"], row["row_id"]]],
                "tolerance_minor": 1,
            }
            provenance.append({"issue_id": issue["issue_id"], "operation": "change_ledger_currency_without_fx",
                               "invoice_id": inv_id, "ledger_row": row["row_id"]})
        else:  # period cutoff
            row = next((r for r in ledger if r["invoice_id"] == inv_id and r["account"] == "Accounts Payable"), None)
            if row is None: continue
            outside = end + timedelta(days=1, hours=2)
            row["timestamp"] = _iso(outside)
            issue = {
                "issue_id": f"ISS-{len(issues)+1}", "type": issue_type,
                "affected_ids": [row["row_id"], inv_id], "expected_amount_minor": amount,
                "currency": cur,
                "acceptable_evidence_sets": [[inv["doc_id"], row["row_id"]]],
                "tolerance_minor": 1,
            }
            provenance.append({"issue_id": issue["issue_id"], "operation": "move_posting_outside_period",
                               "invoice_id": inv_id, "ledger_row": row["row_id"]})
        issues.append(issue)

    # Explicit benign decoys: split payments, repeated amounts, and refunds.
    if payments:
        pay = rng.choice(payments)
        refund_id = f"PAY-REF-{rng.randrange(10000,99999)}"
        refund = {"doc_id": f"DOC-REF-{refund_id}", "payment_id": refund_id,
                  "timestamp": pay["timestamp"], "currency": pay["currency"],
                  "amount_minor": max(1, pay["amount_minor"] // 10), "invoice_refs": pay["invoice_refs"]}
        payments.append(refund); docs[refund["doc_id"]] = refund

    rng.shuffle(ledger); rng.shuffle(invoices); rng.shuffle(payments)
    manifest = {
        "schema_version": "1.0", "episode_id": episode_id, "company_name": company,
        "base_currency": base, "period_start": _iso(start), "period_end": _iso(end),
        "timezone": tz_name, "ledger_count": len(ledger),
        "document_count": len(invoices) + len(payments),
        "rounding": {"mode": "ROUND_HALF_UP", "minor_units": MINOR_UNITS},
    }
    policy = {
        "fx_date_selection": "invoice_issue_date",
        "period_boundaries": "inclusive ISO-8601 timestamps",
        "rounding": "ROUND_HALF_UP to currency minor unit",
        "base_currency": base,
    }
    ledger_io = io.StringIO()
    fields = ["row_id", "timestamp", "account", "debit_minor", "credit_minor",
              "currency", "entity_id", "invoice_id", "payment_id"]
    writer = csv.DictWriter(ledger_io, fieldnames=fields)
    writer.writeheader(); writer.writerows(ledger)

    public = {
        "manifest.json": manifest,
        "ledger.csv": ledger_io.getvalue(),
        "invoices.jsonl": _jsonl(invoices),
        "payments.jsonl": _jsonl(payments),
        "fx.csv": _csv_text(fx, ["date", "source_currency", "target_currency", "rate"]),
        "policy.json": policy,
    }
    closed = {
        "truth.json": {"episode_id": episode_id, "issues": issues},
        "provenance.json": {"episode_id": episode_id, "changes": provenance},
    }
    return {"public": public, "closed": closed}


def _csv_text(rows: list[dict], fields: list[str]) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader(); writer.writerows(rows)
    return out.getvalue()
