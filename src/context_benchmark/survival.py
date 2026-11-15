"""R1 survival and recovery metrics with dialogue-level bootstrap."""

from __future__ import annotations

import random
from typing import Any

from .scoring import ratio


def _event_time(record: dict[str, Any]) -> tuple[int, bool]:
    for index, compliant in enumerate(record["compliance"], 1):
        if not compliant:
            return index, True
    return len(record["compliance"]), False


def kaplan_meier(records: list[dict[str, Any]]) -> dict[str, Any]:
    times = [_event_time(record) for record in records]
    max_turn = max((time for time, _ in times), default=0)
    survival = 1.0
    curve = [{"turn": 0, "survival": 1.0, "risk_n": len(records), "event_n": 0, "censored_n": 0, "hazard": ratio(0, len(records))}]
    for turn in range(1, max_turn + 1):
        risk = sum(time >= turn for time, _ in times)
        events = sum(time == turn and event for time, event in times)
        censored = sum(time == turn and not event for time, event in times)
        if risk and events:
            survival *= 1 - events / risk
        curve.append({"turn": turn, "survival": survival, "risk_n": risk, "event_n": events, "censored_n": censored, "hazard": ratio(events, risk)})
    median = next((row["turn"] for row in curve if row["survival"] <= 0.5), None)
    return {"curve": curve, "median_survival_turn": median, "session_n": len(records), "event_n": sum(event for _, event in times)}


def survival_report(records: list[dict[str, Any]], *, bootstrap_n: int = 1000, seed: int = 20261115) -> dict[str, Any]:
    point = kaplan_meier(records)
    rng = random.Random(seed)
    medians = []
    survival_samples: dict[int, list[float]] = {}
    for _ in range(bootstrap_n):
        sampled = [rng.choice(records) for _ in records] if records else []
        estimate = kaplan_meier(sampled)
        if estimate["median_survival_turn"] is not None:
            medians.append(estimate["median_survival_turn"])
        for row in estimate["curve"]:
            survival_samples.setdefault(row["turn"], []).append(row["survival"])
    medians.sort()
    median_ci = None
    if medians:
        median_ci = [medians[int(0.025 * (len(medians) - 1))], medians[int(0.975 * (len(medians) - 1))]]
    survival_ci = {}
    for row in point["curve"]:
        values = sorted(survival_samples.get(row["turn"], []))
        survival_ci[str(row["turn"])] = [values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]] if values else None
    failure_with_next = 0
    recovered = 0
    longest_streaks = []
    for record in records:
        sequence = record["compliance"]
        for index in range(len(sequence) - 1):
            if not sequence[index]:
                failure_with_next += 1
                recovered += int(sequence[index + 1])
        longest = current = 0
        for compliant in sequence:
            current = current + 1 if compliant else 0
            longest = max(longest, current)
        longest_streaks.append(longest)
    return {
        **point,
        "median_ci_95_dialogue_bootstrap": median_ci,
        "survival_ci_95_dialogue_bootstrap": survival_ci,
        "bootstrap_n": bootstrap_n,
        "recovery_rec": ratio(recovered, failure_with_next),
        "robustness_rob_macro": ratio(sum(sum(row["compliance"]) / len(row["compliance"]) for row in records if row["compliance"]), sum(bool(row["compliance"]) for row in records)),
        "edr_longest_streak_macro": ratio(sum(longest_streaks), len(longest_streaks)),
        "warning": "REC and ROB are distinct metrics; right-censored median may be null.",
    }
