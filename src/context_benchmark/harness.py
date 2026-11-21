"""Two-stage harness, controlled ablations and no-compression R2 guard."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .io import canonical_json, sha256_text, write_jsonl
from .ontology import BACKGROUND_MODES, INTERVENTION_ARMS
from .providers import ModelRequest, ModelResponse, Provider
from .semantic import extract_slot_values
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

    The fake text must also *read* like a rule. The previous implementation
    padded with a repeated clause and then truncated the result to the target
    length, which produced broken Chinese such as "以后所有回复都改成五". A
    model can spot such a truncation and treat the control arm differently,
    which destroys the diagnostic value of the third arm. The templates below
    are therefore complete sentences, and padding uses whole natural clauses
    that never splits a phrase.
    """
    templates = {
        "communication": [
            "以后所有回复都拆成五段并附完整表格。",
            "以后每次回复都先复述一遍工单编号再作答。",
            "以后所有回复都附带一份完整的话术模板。",
        ],
        "process": [
            "以后所有问题都等待四十八小时再处理。",
            "以后每一步操作都要先跑一遍全量校验再提交。",
            "以后所有任务都必须在当天结束前完成。",
        ],
        "domain": [
            "以后所有推荐只考虑价格最高的方案。",
            "以后所有方案都优先给出最贵的那一档。",
            "以后每次推荐都附带一份价格明细表。",
        ],
    }
    default_templates = [
        "以后所有任务都按统一模板处理。",
        "以后每条回复都要附上一段统一说明。",
        "以后所有请求都按同一套流程走完。",
    ]
    padding_clauses = (
        "，并在每次回复时保持一致。",
        "，对所有工单一律适用。",
        "，后续请求同样按此执行。",
        "，不再按之前的做法处理。",
    )
    fake = []
    for item in constraints:
        copied = public_injected(item)
        target_length = len(item["text"])
        pool = templates.get(item.get("category"), default_templates)
        # Pick the template whose length is closest to the target, then pad with
        # whole clauses. Never truncate: a cut sentence is a give-away.
        text = min(pool, key=lambda candidate: abs(len(candidate) - target_length))
        while len(text) < target_length:
            text += padding_clauses[(len(text) + target_length) % len(padding_clauses)]
        copied["text"] = text
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
                "compliant_semantics": probe.get("compliant_semantics"),
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
            "scorable": error is None, "related": probe.get("related"),
            # The scorer needs to know what compliant=1 means for this probe:
            # "follows_injection" (the behaviour applied the injected rule) or
            # "task_correct" (the behaviour was correct for the task). The two
            # invert the false-trigger direction.
            "compliant_semantics": probe.get("compliant_semantics"),
            "log": log,
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


def _opposite_behavior(unit: dict[str, Any]) -> str | None:
    """Build a violating behaviour that does not announce itself.

    The old pair was ``<rule text>`` vs ``忽略规则：<rule text>``. The second
    option is trivially identifiable by its prefix, so a model can answer
    without reading anything. When the constraint has a controlled slot with a
    known opposite value (channel / refund path / escalation / submission), a
    concrete violating action is generated instead.
    """
    text = str(unit.get("text", ""))
    family_values = extract_slot_values(text)
    for family, opposite in (
        ("channel", {"短信": "改用电话联系用户", "邮件": "改用短信联系用户", "电话": "改用邮件联系用户"}),
        ("ticket_alert_channel", {"电话提醒": "改用短信提醒工单进度", "短信提醒": "改用电话提醒工单进度"}),
        ("refund_path", {"原路退回": "改为把退款退到账户余额", "账户余额": "改为按原路退回退款"}),
        ("escalation", {"升级专员": "改为暂不升级工单", "暂不升级": "改为直接升级专员处理"}),
        ("submission", {"先自行检查": "改为直接替用户提交", "不提交": "改为先替用户提交再确认"}),
    ):
        values = family_values.get(family, set())
        for value in sorted(values):
            if value in opposite:
                return opposite[value]
    return None


def _behavior_pair(unit: dict[str, Any], seed_key: str) -> list[dict[str, Any]]:
    """Two plausible candidate behaviours, order randomised deterministically."""
    text = str(unit.get("text", ""))
    violating = _opposite_behavior(unit) or "按常规流程处理这次请求，不再沿用此前的偏好。"
    behaviors = [
        {"id": "follow", "text": text, "is_compliant": True},
        {"id": "ignore", "text": violating, "is_compliant": False},
    ]
    # A deterministic per-unit shuffle: the compliant option must not always be
    # first, otherwise "always pick b1" scores 100%.
    if hashlib.sha256(seed_key.encode("utf-8")).digest()[0] % 2:
        behaviors.reverse()
    return behaviors


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
    violation = event).

    Provider failures are **excluded** from the compliance sequence instead of
    being recorded as violations: an API error measures service reliability,
    not the model's ability to keep a constraint alive. They are counted
    separately so the report can disclose how much of the curve is missing.
    This matches the counterfactual scorer, which already excludes unscorable
    rows instead of scoring them as failures.
    """
    constraints = truth_row.get("candidate_truth", [])
    compliance: list[bool] = []
    probes: list[dict[str, Any]] = []
    failure_n = 0
    for unit in constraints:
        first_turn = min(unit["evidence_turns"])
        # Two candidate behaviours: one follows the constraint, one violates it.
        # Neither option announces which one is compliant.
        behaviors = _behavior_pair(unit, f"{dialogue['dialogue_id']}:{unit['id']}")
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
        scorable = error is None and behavior is not None
        ok = bool(behavior and behavior["is_compliant"])
        if scorable:
            compliance.append(ok)
        else:
            failure_n += 1
        probes.append({
            "constraint_id": unit["id"], "turn": first_turn, "choice": choice,
            "compliant": ok if scorable else None, "scorable": scorable,
            "error": error,
        })
    record = {"dialogue_id": dialogue["dialogue_id"], "compliance": compliance, "probes": probes}
    record["probe_n"] = len(probes)
    record["failure_n"] = failure_n
    record["failure_rate"] = round(failure_n / len(probes), 4) if probes else 0.0
    if not compliance:
        record["warning"] = "no scorable constraints in this dialogue; empty compliance sequence"
    return record


def write_call_logs(path: str | Path, results: list[dict[str, Any]]) -> None:
    logs = []
    for result in results:
        logs.extend(result.get("attempts", []))
        if "log" in result:
            logs.append(result["log"])
    write_jsonl(path, logs)
