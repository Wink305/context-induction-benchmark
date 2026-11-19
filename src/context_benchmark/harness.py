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

INJECTED_FIELDS = (
    "id", "type", "text", "evidence_turns", "valid_from", "valid_to",
    "scope", "superseded_by", "category",
)


def public_injected(item: dict[str, Any]) -> dict[str, Any]:
    """Keep only schema fields — never leak arm metadata to the executor.

    The fake arm used to carry ``fake_control: true`` which was serialised into
    the prompt, letting any model identify the control arm from metadata alone.
    Both true and fake arms must be indistinguishable except for their content.
    """
    return {key: item[key] for key in INJECTED_FIELDS if key in item}


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


def _within_budget(
    items: list[Any], serialize: Callable[[Any], str], budget: int
) -> tuple[list[Any], list[Any]]:
    """Return (selected, dropped). Every item, including the first, obeys budget."""
    selected: list[Any] = []
    dropped: list[Any] = []
    used = 0
    for item in items:
        size = len(serialize(item))
        if used + size > budget:
            dropped.append(item)
            continue
        selected.append(item)
        used += size
    return selected, dropped


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
        selected, dropped = _within_budget(eligible, lambda item: canonical_json(item), budget_chars)
        text = canonical_json({"constraints": selected})
        return text, {
            "mode": mode, "chars": len(text),
            "selected_ids": [item["id"] for item in selected],
            "dropped_ids": [item["id"] for item in dropped],
            "truncated": bool(dropped), "budget_chars": budget_chars,
        }
    # Deliberately simple, deterministic engineering baseline; not a reproduction
    # of the LLM selector in arXiv:2609.34227.
    turns = dialogue["turns"]
    first = turns[:2]
    recent = list(reversed(turns[2:]))
    selected_recent, dropped_recent = _within_budget(
        recent, lambda item: canonical_json(item), max(0, budget_chars - len(canonical_json(first)))
    )
    selected = first + list(reversed(selected_recent))
    selected = sorted({item["turn"]: item for item in selected}.values(), key=lambda item: item["turn"])
    text = dialogue_text(selected)
    return text, {
        "mode": mode, "chars": len(text),
        "selected_turns": [item["turn"] for item in selected],
        "dropped_turns": [item["turn"] for item in dropped_recent],
        "selector": "first_recent_v0", "truncated": bool(dropped_recent), "budget_chars": budget_chars,
    }


def make_fake_constraints(constraints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build length-aligned, plausible fake constraints for the control arm.

    No metadata field identifies these as fake: the injected JSON must be
    indistinguishable from the true arm except for its content. Length is
    matched so a model cannot infer the arm from token count.
    """
    templates = {
        "communication": "以后所有回复都改成五段并附完整表格。",
        "process": "以后所有问题都等待四十八小时再处理。",
        "domain": "以后所有推荐只考虑价格最高的方案。",
    }
    fake = []
    for item in constraints:
        copied = public_injected(item)
        target_length = len(item["text"])
        text = templates.get(item.get("category"), "以后所有任务都按统一模板处理。")
        while len(text) < target_length:
            text += "并保持一致"
        copied["text"] = text[:target_length]
        fake.append(copied)
    return fake


def planned_conditions() -> list[tuple[str, str]]:
    """Four background ablations, each with the full three-arm intervention.

    The old design ran true/fake arms only under ``no_history``, confounding
    the intervention with the strongest ablation. Both target-isolating
    backgrounds (``distilled`` and ``no_history``) now carry all three arms so
    the injection effect can be compared across compression strengths.
    """
    return [
        (background, arm)
        for background in BACKGROUND_MODES
        for arm in INTERVENTION_ARMS
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
        condition_id = f"{dialogue['dialogue_id']}:{probe['probe_id']}:{background_mode}:{arm}:{budget_name}"
        if arm in {"true", "fake"} and not target:
            # Without the induced target there is nothing to inject; counting
            # this as a compliant/not-compliant choice would be a fake score.
            return {
                "dialogue_id": dialogue["dialogue_id"],
                "probe_id": f"{dialogue['dialogue_id']}:{probe['probe_id']}",
                "condition_id": condition_id,
                "arm": arm, "background_mode": background_mode, "budget": budget_name,
                "choice": None, "compliant": 0, "scorable": False,
                "related": probe.get("related"),
                "error": "target_missing_from_induction",
                "log": {"error": "target_missing_from_induction", "target_constraint_id": target_id},
            }
        if arm == "none":
            injected = []
        elif arm == "true":
            injected = [public_injected(item) for item in target]
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
                model=self.config.get("executor_model_id") or self.config["model_id"],
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
            "condition_id": condition_id,
            "arm": arm, "background_mode": background_mode, "budget": budget_name,
            "choice": choice, "compliant": int(bool(behavior and behavior["is_compliant"])) if error is None else 0,
            "scorable": error is None, "related": probe.get("related"), "log": log,
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


def char_token_counter(text: str) -> int:
    """Deterministic fallback token estimate (chars/4). Not a real tokenizer.

    Dimension F must declare its counting method; a real run replaces this with
    the model's tokenizer. The estimate is used only for the pre-call guard —
    the authoritative pass/fail uses actual provider usage in after_call.
    """
    return max(1, len(text) // 4)


def run_no_compression(
    provider: Provider,
    config: dict[str, Any],
    task: dict[str, Any],
    prompt: str,
    *,
    runner: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Dimension F: feed the full dialogue with no compression whatsoever.

    Uses NoCompressionGuard before and after the call. Any truncation, missing
    usage, or context overflow fails the task rather than being silently
    compacted. This is the "bare model in a fixed window" baseline.
    """
    limit = int(config.get("r2_context_limit_tokens", 65536))
    guard = NoCompressionGuard(limit, char_token_counter)
    full_text = dialogue_text(task["turns"])
    user_content = f"对话全文：\n{full_text}\n\n请归纳其中的用户约束。只输出 JSON。"
    prompt_tokens_est = guard.count(prompt + user_content)
    before = guard.before_call(prompt + user_content, int(config.get("max_tokens", 4096)))
    record: dict[str, Any] = {
        "dialogue_id": task["dialogue_id"],
        "guard_before": before,
        "estimated_prompt_tokens": prompt_tokens_est,
        "compression": False,
        "truncation": False,
        "context_limit_tokens": limit,
    }
    if not before["allowed"]:
        record.update({"task_success": False, "failure_reason": before["failure_reason"], "scorable": False})
        return record
    request = ModelRequest(
        model=config["model_id"],
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=config.get("temperature", 0.0),
        max_tokens=config.get("max_tokens", 4096),
        seed=config.get("seed", 0),
        metadata={"stage": "r2_no_compression", "dialogue_id": task["dialogue_id"], "compression": "none"},
    )
    response = None
    error = None
    try:
        response = provider.complete(request)
    except Exception as exc:  # provider failure is a failed task, not a skipped row
        error = f"{type(exc).__name__}: {exc}"
    record["log"] = _base_log(request, response, error)
    if error or response is None:
        record.update({"task_success": False, "failure_reason": error, "scorable": False})
        return record
    after = guard.after_call(response.prompt_tokens, response.completion_tokens)
    record["guard_after"] = after
    record["actual_tokens"] = {
        "prompt": response.prompt_tokens,
        "completion": response.completion_tokens,
    }
    record["latency_ms"] = response.latency_ms
    record["cost"] = response.cost
    output, parse_report = parse_output(response.text)
    validation = validate_output(output, task["turns"]) if parse_report.valid else parse_report
    record.update({
        "task_success": bool(after["task_success"] and validation.valid),
        "failure_reason": after["failure_reason"] or (None if validation.valid else "invalid_output"),
        "scorable": True,
        "valid_output": bool(validation.valid),
    })
    return record


def run_survival_trace(
    provider: Provider,
    config: dict[str, Any],
    dialogue: dict[str, Any],
    truth_row: dict[str, Any],
    prompt: str,
) -> dict[str, Any]:
    """Dimension E: produce a per-turn compliance sequence for survival analysis.

    For each user turn that carries a constraint, build a probe at that point
    in time and ask the executor whether the correct behaviour is followed.
    The resulting boolean sequence is the input to `survival_report` (first
    violation = event). Every probe is scorable; failures are recorded as a
    violation, not dropped.
    """
    constraints = truth_row.get("candidate_truth", [])
    compliance: list[bool] = []
    probes: list[dict[str, Any]] = []
    for unit in constraints:
        first_turn = min(unit["evidence_turns"])
        # Two candidate behaviours: one follows the constraint, one ignores it.
        behaviors = [
            {"id": "follow", "text": unit["text"], "is_compliant": True},
            {"id": "ignore", "text": f"忽略规则：{unit['text']}", "is_compliant": False},
        ]
        probe = {"probe_id": f"{unit['id']}@{first_turn}", "target_constraint_id": unit["id"],
                 "query": f"到第 {first_turn} 轮时，应遵循哪条用户规则？", "behaviors": behaviors,
                 "related": True}
        history = dialogue_text([t for t in dialogue["turns"] if t["turn"] <= first_turn])
        user_content = (
            f"新情境：{probe['query']}\n"
            f"候选行为：{canonical_json([{'id': b['id'], 'text': b['text']} for b in behaviors])}\n"
            f"背景历史：\n{history}\n干预要点：{canonical_json({'constraints': [public_injected(unit)]})}"
        )
        request = ModelRequest(
            model=config.get("executor_model_id") or config["model_id"],
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user_content}],
            temperature=config.get("temperature", 0.0), max_tokens=64, seed=config.get("seed", 0),
            metadata={"stage": "e_survival_trace", "dialogue_id": dialogue["dialogue_id"],
                      "constraint_id": unit["id"], "turn": first_turn},
        )
        response = None
        error = None
        choice = None
        try:
            response = provider.complete(request)
            choice = json.loads(response.text).get("choice")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        behavior = next((b for b in behaviors if b["id"] == choice), None)
        ok = bool(behavior and behavior["is_compliant"])
        compliance.append(ok)
        probes.append({
            "constraint_id": unit["id"], "turn": first_turn, "choice": choice,
            "compliant": ok, "error": error,
        })
    record = {"dialogue_id": dialogue["dialogue_id"], "compliance": compliance, "probes": probes}
    if not compliance:
        record["warning"] = "no constraints in this dialogue; empty compliance sequence"
    return record


def write_call_logs(path: str | Path, results: list[dict[str, Any]]) -> None:
    logs = []
    for result in results:
        logs.extend(result.get("attempts", []))
        if "log" in result:
            logs.append(result["log"])
    write_jsonl(path, logs)
