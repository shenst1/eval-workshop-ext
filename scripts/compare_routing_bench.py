#!/usr/bin/env python3
"""Compare escalate-only agent routing vs Jev (per-call and batch).

Runs routers on the same JSONL dataset and prints accuracy, wall-clock,
tokens, USD cost, and API call counts. Costs use ``helpdesk.pricing``.

Usage:
    uv run python scripts/compare_routing_bench.py \
        --dataset data/live_workshop_routing.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from typesafe_sdk import AsyncTypeSafeClient

from agent_platform.observability import (
    CostTracker,
    MultiProcessor,
    Span,
    ToolSpanData,
    TracingProcessor,
)
from agent_platform.runner import collect
from helpdesk.app import create_app, default_model
from helpdesk.jev_router import route_batch_with_jev, route_with_jev
from helpdesk.pricing import estimate_cost, format_rates_footer, resolve_rate


@dataclass
class MethodTotals:
    """Accumulated metrics for one routing method."""

    name: str
    model: str
    correct: int = 0
    total: int = 0
    seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    api_calls: int = 0
    departments: list[str] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def cost_usd(self) -> float:
        return estimate_cost(self.model, self.input_tokens, self.output_tokens)


class ToolCallCollector(TracingProcessor):
    """Records tool call names and parsed JSON args from tool spans."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def on_span_start(self, span: Span) -> None:
        return

    def on_span_end(self, span: Span) -> None:
        if not isinstance(span.span_data, ToolSpanData):
            return
        args: dict = {}
        raw = span.span_data.input
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    args = parsed
            except json.JSONDecodeError:
                args = {}
        self.calls.append({"name": span.span_data.name, "args": args})

    def clear(self) -> None:
        self.calls.clear()

    def department(self) -> str | None:
        for call in self.calls:
            if call["name"] == "escalate_to_department":
                dept = call["args"].get("department")
                if isinstance(dept, str):
                    return dept
        return None


def load_examples(path: Path) -> list[dict]:
    examples = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def _print_compare(label: str, slow: MethodTotals, fast: MethodTotals) -> None:
    if fast.seconds > 0:
        print(f"Speedup ({label}): {slow.seconds / fast.seconds:.2f}x")
    cost_delta = slow.cost_usd - fast.cost_usd
    print(f"Cost delta ({label}): ${cost_delta:.6f}")
    if fast.cost_usd > 0:
        print(
            f"Cost multiplier ({label}): "
            f"{slow.cost_usd / fast.cost_usd:.2f}x"
        )


def print_table(
    agent: MethodTotals,
    jev: MethodTotals,
    jev_batch: MethodTotals,
) -> None:
    rows = [agent, jev, jev_batch]
    headers = (
        "Method",
        "Accuracy",
        "Total s",
        "Avg s",
        "Calls",
        "In tok",
        "Out tok",
        "Cost USD",
    )
    data = []
    for row in rows:
        avg = row.seconds / row.total if row.total else 0.0
        data.append(
            [
                row.name,
                f"{row.correct}/{row.total} ({row.accuracy:.0%})",
                f"{row.seconds:.2f}",
                f"{avg:.2f}",
                str(row.api_calls),
                str(row.input_tokens),
                str(row.output_tokens),
                f"${row.cost_usd:.6f}",
            ]
        )

    widths = [len(h) for h in headers]
    for row in data:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(cols: list[str]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cols))

    print(fmt(list(headers)))
    print(fmt(["-" * w for w in widths]))
    for row in data:
        print(fmt(row))

    print()
    _print_compare("agent / jev-batch", agent, jev_batch)
    print()
    _print_compare("jev-per-call / jev-batch", jev, jev_batch)
    print()
    print(f"Agent model:     {agent.model}")
    print(f"Jev model:       {jev.model}")
    print(f"Jev batch model: {jev_batch.model}")
    print()
    print(format_rates_footer([agent.model, jev.model, jev_batch.model]))


async def run_agent_pass(
    examples: list[dict],
    model: str,
) -> MethodTotals:
    totals = MethodTotals(name="Agent", model=model)
    cost = CostTracker()
    tools = ToolCallCollector()
    client = create_app(
        config="baseline.yaml",
        model=model,
        processor=MultiProcessor([cost, tools]),
    )

    for ex in examples:
        request = ex.get("input", {}).get("request", "")
        expected = (ex.get("output") or {}).get("department")
        tools.clear()
        before_in = cost.total_input_tokens
        before_out = cost.total_output_tokens

        t0 = time.perf_counter()
        await collect(client.send(request))
        elapsed = time.perf_counter() - t0

        # No escalate tool call → treat as department "None" (matches dataset labels).
        actual = tools.department() or "None"
        totals.seconds += elapsed
        totals.total += 1
        totals.api_calls += 1
        totals.input_tokens += cost.total_input_tokens - before_in
        totals.output_tokens += cost.total_output_tokens - before_out
        totals.departments.append(actual)
        if expected is not None and actual == expected:
            totals.correct += 1
        print(
            f"  [agent] {ex.get('id', '?')}: "
            f"{actual!r} (expected {expected!r}) {elapsed:.2f}s"
        )

    return totals


async def run_jev_pass(examples: list[dict]) -> MethodTotals:
    totals = MethodTotals(name="Jev (per call)", model="jev-latest")
    async with AsyncTypeSafeClient() as ts_client:
        for ex in examples:
            request = ex.get("input", {}).get("request", "")
            expected = (ex.get("output") or {}).get("department")

            t0 = time.perf_counter()
            result = await route_with_jev(
                request, client=ts_client, escalate=False
            )
            elapsed = time.perf_counter() - t0

            if result.model:
                totals.model = result.model
                resolve_rate(result.model)

            totals.seconds += elapsed
            totals.total += 1
            totals.api_calls += 1
            totals.input_tokens += result.input_tokens
            totals.output_tokens += result.output_tokens
            totals.departments.append(result.department)
            if expected is not None and result.department == expected:
                totals.correct += 1
            print(
                f"  [jev]   {ex.get('id', '?')}: "
                f"{result.department!r} (expected {expected!r}) {elapsed:.2f}s "
                f"tok={result.input_tokens}/{result.output_tokens}"
            )

    return totals


async def run_jev_batch_pass(examples: list[dict]) -> MethodTotals:
    tickets: dict[str, str] = {}
    expected_by_id: dict[str, str | None] = {}
    for ex in examples:
        ex_id = str(ex.get("id", f"ex_{len(tickets)}"))
        tickets[ex_id] = ex.get("input", {}).get("request", "")
        expected_by_id[ex_id] = (ex.get("output") or {}).get("department")

    totals = MethodTotals(name="Jev (batch)", model="jev-latest")
    print(f"  Jev batch: 1 system_one call for {len(tickets)} examples")

    async with AsyncTypeSafeClient() as ts_client:
        t0 = time.perf_counter()
        batch = await route_batch_with_jev(tickets, client=ts_client)
        elapsed = time.perf_counter() - t0

    if batch.model:
        totals.model = batch.model
        resolve_rate(batch.model)

    totals.seconds = elapsed
    totals.api_calls = batch.api_calls
    totals.input_tokens = batch.input_tokens
    totals.output_tokens = batch.output_tokens
    totals.total = len(tickets)

    for ex_id, request in tickets.items():
        result = batch.results[ex_id]
        expected = expected_by_id[ex_id]
        totals.departments.append(result.department)
        if expected is not None and result.department == expected:
            totals.correct += 1
        print(
            f"  [batch] {ex_id}: "
            f"{result.department!r} (expected {expected!r})"
        )

    print(
        f"  [batch] wall={elapsed:.2f}s "
        f"tok={batch.input_tokens}/{batch.output_tokens} "
        f"calls={batch.api_calls}"
    )
    return totals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare agent vs Jev routing (speed + cost)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/live_workshop_routing.jsonl"),
        help="JSONL dataset with input.request and output.department",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Agent model override (default: helpdesk default_model())",
    )
    return parser.parse_args()


async def main() -> int:
    load_dotenv(override=True)
    args = parse_args()
    examples = load_examples(args.dataset)
    if not examples:
        print(f"No examples in {args.dataset}")
        return 1

    model = args.model or default_model()
    resolve_rate(model)

    print(f"Dataset: {args.dataset} ({len(examples)} examples)")
    print(f"Agent model: {model}")
    print()
    print("=== Agent (baseline escalate-only) ===")
    agent = await run_agent_pass(examples, model)
    print()
    print("=== Jev (per call) ===")
    jev = await run_jev_pass(examples)
    print()
    print("=== Jev (batch) ===")
    jev_batch = await run_jev_batch_pass(examples)
    print()
    print_table(agent, jev, jev_batch)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
