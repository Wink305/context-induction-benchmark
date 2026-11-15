"""Two-stage harness, controlled ablations and no-compression R2 guard."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .io import canonical_json, sha256_text, write_jsonl
from .ontology import BACKGROUND_MODES, INTERVENTION_ARMS
from .providers import ModelRequest, ModelResponse, Provider
from .validation import parse_output, validate_output


def dialogue_text(turns: list[dict[str, Any]]) -> str:
    return "\n".join(f"[{turn['turn']}] {turn['speaker']}: {turn['text']}" for turn in turns)


def _base_log(request: ModelRequest, response: ModelResponse | None, error: str | None = None) -> dict[str, Any]:
    return {
        "request": {
            "model": request.model,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "seed": request.seed,
            "metadata": request.metadata,
        },
        "response_text": response.text if response else None,
        "response_raw": response.raw if response else None,
        "prompt_tokens": response.prompt_tokens if response else None,
        "completion_tokens": response.completion_tokens if response else None,
        "latency_ms": response.latency_ms if response else None,
        "cost": response.cost if response else None,
        "error": error,
    }


class InductionRunner:
    def __init__(self, provider: Provider, config: dict[str, Any], prompt: str) -> None:
        self.provider = provider
        self.config = config
        self.prompt = prompt
        self.prompt_sha256 = sha256_text(prompt)

    def run(self, dialogue: dict[str, Any]) -> dict[str, Any]:
        attempts = []
        output = None
        validation = None
        max_attempts = int(self.config.get("max_retries", 2)) + 1
        for attempt in range(1, max_attempts + 1):
            request = ModelRequest(
                model=self.config["model_id"],
                messages=[
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": dialogue_text(dialogue["turns"])},
                ],
                temperature=self.config["temperature"],
                max_tokens=self.config["max_tokens"],
                seed=self.config["seed"] + attempt - 1,
                metadata={
                    "stage": "induction",
                    "dialogue_id": dialogue["dialogue_id"],
                    "attempt": attempt,
                    "prompt_version": self.config["prompt_version"],
                    "prompt_sha256": self.prompt_sha256,
                    "pinned_version": self.config.get("pinned_version"),
                    "quant": self.config.get("quant"),
                    "compression": "model_induction_required",
                },
            )
            response = None
            error = None
            try:
                response = self.provider.complete(request)
                output, parse_report = parse_output(response.text)
                if not parse_report.valid:
                    validation = parse_report
                else:
                    validation = validate_output(output, dialogue["turns"])
            except Exception as exc:  # provider failures are data, not silent omissions
                error = f"{type(exc).__name__}: {exc}"
            log = _base_log(request, response, error)
            log["validation"] = validation.to_dict() if validation else None
            attempts.append(log)
            if validation and validation.valid:
                break
        return {
            "dialogue_id": dialogue["dialogue_id"],
            "output": output,
            "valid": bool(validation and validation.valid),
            "validation": validation.to_dict() if validation else {"valid": False, "issues": [{"code": "provider_error"}]},
            "attempts": attempts,
            "failure_counted": not bool(validation and validation.valid),
        }


def _within_budget(items: list[Any], serialize: Callable[[Any], str], budget: int) -> list[Any]:
    selected = []
    used = 0
    for item in items:
        size = len(serialize(item))
        if selected and used + size > budget:
            continue
        if size > budget:
            continue
        selected.append(item)
        used += size
    return selected


def serialize_history(
    dialogue: dict[str, Any], mode: str, constraints: list[dict[str, Any]], budget_chars: int,
    *, exclude_ids: set[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    if mode not in BACKGROUND_MODES:
        raise ValueError(f"Unknown background mode: {mode}")
    exclude_ids = exclude_ids or set()
    if mode == "no_history":
        return "", {"mode": mode, "chars": 0, "truncated": False}
    if mode == "full_context":
        text = dialogue_text(dialogue["turns"])
        return text, {"mode": mode, "chars": len(text), "truncated": False, "budget_ignored_for_full_arm": True}
    if mode == "distilled":
        eligible = [item for item in constraints if item.get("id") not in exclude_ids]
        selected = _within_budget(eligible, lambda item: canonical_json(item), budget_chars)
        text = canonical_json({"constraints": selected})
        return text, {"mode": mode, "chars": len(text), "selected_ids": [item["id"] for item in selected], "truncated": False, "budget_chars": budget_chars}
    # Deliberately simple, deterministic engineering baseline; not a reproduction
    # of the LLM selector in arXiv:2609.34227.
    turns = dialogue["turns"]
    first = turns[:2]
    recent = list(reversed(turns[2:]))
    selected_recent = _within_budget(recent, lambda item: canonical_json(item), max(0, budget_chars - len(canonical_json(first))))
    selected = first + list(reversed(selected_recent))
    selected = sorted({item["turn"]: item for item in selected}.values(), key=lambda item: item["turn"])
    text = dialogue_text(selected)
    return text, {"mode": mode, "chars": len(text), "selected_turns": [item["turn"] for item in selected], "selector": "first_recent_v0", "truncated": False, "budget_chars": budget_chars}


def make_fake_constraints(constraints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    templates = {
        "communication": "以后所有回复都改成五段并附完整表格。",
        "process": "以后所有问题都等待四十八小时再处理。",
        "domain": "以后所有推荐只考虑价格最高的方案。",
    }
    fake = []
    for item in constraints:
        copied = dict(item)
        target_length = len(item["text"])
        text = templates.get(item.get("category"), "以后所有任务都按统一模板处理。")
        while len(text) < target_length:
            text += "并保持一致"
        copied["text"] = text[:target_length]
        copied["fake_control"] = True
        fake.append(copied)
    return fake


def planned_conditions() -> list[tuple[str, str]]:
    """Four background ablations + isolated symmetric three-arm intervention."""
    return [
        ("full_context", "none"),
        ("distilled", "none"),
        ("raw_turns", "none"),
        ("no_history", "none"),
        ("no_history", "true"),
        ("no_history", "fake"),
    ]


class ExecutorRunner:
    def __init__(self, provider: Provider, config: dict[str, Any], prompt: str) -> None:
        self.provider = provider
        self.config = config
        self.prompt = prompt

    def run_probe(
        self,
        dialogue: dict[str, Any], probe: dict[str, Any], induced_constraints: list[dict[str, Any]],
        background_mode: str, arm: str, budget_name: str,
    ) -> dict[str, Any]:
        if arm not in INTERVENTION_ARMS:
            raise ValueError(f"Unknown intervention arm: {arm}")
        target_id = probe["target_constraint_id"]
        target = [item for item in induced_constraints if item.get("id") == target_id]
        if arm == "none":
            injected = []
        elif arm == "true":
            injected = target
        else:
            injected = make_fake_constraints(target)
        budget = self.config["budgets_chars"][budget_name]
        history, history_meta = serialize_history(
            dialogue, background_mode, induced_constraints, budget,
            exclude_ids={target_id} if background_mode == "distilled" else set(),
        )
        behavior_block = canonical_json([
            {"id": behavior["id"], "text": behavior["text"]} for behavior in probe["behaviors"]
        ])
        user_content = (
            f"新情境：{probe['query']}\n候选行为：{behavior_block}\n"
            f"背景历史：\n{history}\n干预要点：\n{canonical_json({'constraints': injected})}"
        )
        request = ModelRequest(
            model=self.config["executor_model_id"],
            messages=[{"role": "system", "content": self.prompt}, {"role": "user", "content": user_content}],
            temperature=self.config["temperature"], max_tokens=256, seed=self.config["seed"],
            metadata={
                "stage": "executor", "dialogue_id": dialogue["dialogue_id"], "probe_id": probe["probe_id"],
                "background_mode": background_mode, "intervention_arm": arm, "budget": budget_name,
                "target_isolated": background_mode in {"no_history", "distilled"},
                "history": history_meta, "executor_prompt_version": self.config["executor_prompt_version"],
                "executor_prompt_sha256": sha256_text(self.prompt), "compression": "controlled_ablation",
            },
        )
        response = None
        error = None
        choice = None
        try:
            response = self.provider.complete(request)
            choice = json.loads(response.text).get("choice")
            valid_ids = {behavior["id"] for behavior in probe["behaviors"]}
            if choice not in valid_ids:
                error = f"invalid_choice:{choice}"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        behavior = next((item for item in probe["behaviors"] if item["id"] == choice), None)
        log = _base_log(request, response, error)
        log["choice"] = choice
        return {
            "dialogue_id": dialogue["dialogue_id"],
            "probe_id": f"{dialogue['dialogue_id']}:{probe['probe_id']}",
            "condition_id": f"{dialogue['dialogue_id']}:{probe['probe_id']}:{background_mode}:{arm}:{budget_name}",
            "arm": arm, "background_mode": background_mode, "budget": budget_name,
            "choice": choice, "compliant": int(bool(behavior and behavior["is_compliant"])) if error is None else 0,
            "scorable": error is None, "related": probe.get("related", True), "log": log,
        }


class NoCompressionGuard:
    """Fail closed for R2. Never truncate, summarize, or continue after overflow."""

    def __init__(self, context_limit_tokens: int, token_counter: Callable[[str], int]) -> None:
        self.limit = context_limit_tokens
        self.count = token_counter

    def before_call(self, prompt: str, max_output_tokens: int) -> dict[str, Any]:
        prompt_tokens = self.count(prompt)
        allowed = prompt_tokens + max_output_tokens <= self.limit
        return {
            "allowed": allowed, "prompt_tokens_estimate": prompt_tokens,
            "max_output_tokens": max_output_tokens, "context_limit_tokens": self.limit,
            "compression": False, "truncation": False,
            "failure_reason": None if allowed else "estimated_context_overflow",
        }

    def after_call(self, prompt_tokens: int | None, completion_tokens: int | None) -> dict[str, Any]:
        if prompt_tokens is None or completion_tokens is None:
            return {"task_success": False, "failure_reason": "missing_actual_token_usage", "compression": False, "truncation": False}
        total = prompt_tokens + completion_tokens
        return {
            "task_success": total <= self.limit, "actual_total_tokens": total,
            "context_limit_tokens": self.limit,
            "failure_reason": None if total <= self.limit else "actual_context_overflow",
            "compression": False, "truncation": False,
        }


def write_call_logs(path: str | Path, results: list[dict[str, Any]]) -> None:
    logs = []
    for result in results:
        logs.extend(result.get("attempts", []))
        if "log" in result:
            logs.append(result["log"])
    write_jsonl(path, logs)
