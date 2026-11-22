"""Machine-readable six-axis dimension registry.

v0.4 re-organises the scoreboard into one primary dimension plus five
supporting/attribution axes. This is a *structural* re-registration, not a
claim that three new constructs were invented: D is split out of A because
section 5.5 of the design record forbids folding must-include and
must-exclude into one accuracy number, and E/F are the former R1/R2 axes
promoted to named dimensions while remaining outside the ranking.

No weighted total is produced. A/B/C/D are reported as a profile; E/F are
attribution-only and never folded into that profile.

Metric contract
---------------
``metrics`` lists only names a scorer in this repository actually produces
(``IMPLEMENTED_METRICS``); ``promised_metrics`` lists the ones the design
document promises that nothing produces yet, and ``metrics_state`` says which
of the two applies. The earlier registry mixed both in one list, which made
``hit_rate``, ``FAR``, ``live_constraint_accuracy``, ``hierarchy_S_U_T``,
``task_completion`` and ``quality_decay`` look implemented. They are not, and
``tests/test_dimensions.py`` now fails if a registered metric has no
implementation behind it.
"""

from __future__ import annotations

from typing import Any

DIMENSION_SET_VERSION = "six-axis-v1"
PRIMARY_DIMENSION = "A"

# ---------------------------------------------------------------------------
# Implemented metric names.
#
# Every name below must exist as a key in the output of the module it points
# at. ``tests/test_dimensions.py`` asserts that, so a metric can no longer be
# registered without an implementation behind it — the earlier registry listed
# ``hit_rate``, ``FAR``, ``live_constraint_accuracy``, ``hierarchy_S_U_T``,
# ``task_completion`` and ``quality_decay``, none of which any scorer ever
# produced. A dotted name means "this key under that key".
#
# Keys are repository-relative paths so the test can read the source and prove
# the metric is really there.
# ---------------------------------------------------------------------------
IMPLEMENTED_METRICS: dict[str, tuple[str, ...]] = {
    "src/context_benchmark/scoring.py": (
        "must_include", "must_exclude_absence", "fama_macro",
        "per_type.precision", "per_type.recall", "per_type.f1",
        "classification.precision", "classification.recall", "classification.f1",
        "superseded_type_accuracy", "expired_reuse_rate", "supersession_link_accuracy",
        "true_delta_mean", "fake_delta_mean", "negative_transfer_rate", "false_trigger_rate",
        "fmr", "empty_output_accuracy", "unmatched_prediction_rate", "invalid_output_rate",
    ),
    "src/context_benchmark/survival.py": (
        "median_survival_turn", "median_ci_95_dialogue_bootstrap",
        "recovery_rec", "robustness_rob_macro",
    ),
    "src/context_benchmark/harness.py": (
        "task_success", "actual_tokens", "latency_ms", "cost",
    ),
}


def implemented_metric_names() -> set[str]:
    """Every metric name that a scorer in this repository actually produces."""
    return {name for names in IMPLEMENTED_METRICS.values() for name in names}

DIMENSIONS: list[dict[str, Any]] = [
    {
        "id": "A",
        "name": "约束判别精度",
        "name_en": "Constraint discrimination",
        "role": "primary",
        "rankable": True,
        "orthogonality_required": True,
        "metrics": ["must_include", "must_exclude_absence", "per_type.precision", "per_type.recall", "per_type.f1"],
        "promised_metrics": [],
        "metrics_state": "implemented",
        "gold": "gold v1.0 (human); candidate_truth is engineering-only",
        "gate": "停止条件 1：S/O 类 Cohen's κ ≥ 0.60，否则项目不成立",
        "design_section": "§3.1",
        "note": "唯一主维度：项目成立与否由它决定，人工 κ 挂在这里。",
    },
    {
        "id": "B",
        "name": "联想命中与迁移安全性",
        "name_en": "Association and transfer safety",
        "role": "supporting",
        "rankable": True,
        "orthogonality_required": True,
        "metrics": ["true_delta_mean", "fake_delta_mean", "false_trigger_rate", "negative_transfer_rate"],
        "promised_metrics": ["hit_rate", "FAR"],
        "metrics_state": "partial",
        "gold": "行为锚定 + 真/无/伪三臂；不需要「关键要点」的文本 gold",
        "gate": "P1 四组消融 + 伪要点臂增益不显著，否则构念作废",
        "design_section": "§3.2",
        "note": "与 A 共线性必须由 P1 相关矩阵检验（r < 0.8）。",
    },
    {
        "id": "C",
        "name": "冲突与时效判断",
        "name_en": "Conflict and staleness",
        "role": "supporting",
        "rankable": True,
        "orthogonality_required": True,
        "metrics": ["superseded_type_accuracy", "expired_reuse_rate", "supersession_link_accuracy"],
        "promised_metrics": ["live_constraint_accuracy", "hierarchy_S_U_T"],
        "metrics_state": "partial",
        "gold": "显式 memory trace，程序可判，不依赖 judge",
        "gate": "难度梯度必须避免地板效应（单跳→多跳→多实体）",
        "design_section": "§3.3",
        "note": "层级冲突（平台>用户>一次性>工具）P1 出题、P2 扩题。",
    },
    {
        "id": "D",
        "name": "幻觉抵抗与空输出校准",
        "name_en": "Hallucination resistance and abstention calibration",
        "role": "supporting",
        "rankable": True,
        "orthogonality_required": True,
        "metrics": ["fmr", "empty_output_accuracy", "unmatched_prediction_rate", "invalid_output_rate"],
        "promised_metrics": [],
        "metrics_state": "implemented",
        "gold": "干扰段伪约束 + 空段；与 A 的分母不同，必须独立报",
        "gate": "§5.5 分母披露；must-include / must-exclude 不得合成单一准确率",
        "design_section": "§3.4 (v0.4)",
        "note": "从原维度 A 拆出。分母与失败模式不同，单列才能定位「凭空捏造」和「不敢留空」。",
    },
    {
        "id": "E",
        "name": "长程衰减生存曲线",
        "name_en": "Long-range decay survival (former R1)",
        "role": "attribution",
        "rankable": False,
        "orthogonality_required": False,
        "metrics": ["median_survival_turn", "median_ci_95_dialogue_bootstrap", "recovery_rec", "robustness_rob_macro"],
        "promised_metrics": [],
        "metrics_state": "implemented",
        "gold": "首次违反 / 首次遗忘为事件，Kaplan–Meier",
        "gate": "与 F 分开报；压缩策略须固定并声明",
        "design_section": "§4.1",
        "note": "原 R1。归因轴，不进排名、不进总分。",
    },
    {
        "id": "F",
        "name": "无压缩上下文耐用度基线",
        "name_en": "No-compression durability baseline (former R2)",
        "role": "attribution",
        "rankable": False,
        "orthogonality_required": False,
        "metrics": ["task_success", "actual_tokens", "latency_ms", "cost"],
        "promised_metrics": ["quality_decay"],
        "metrics_state": "partial",
        "gold": "固定窗口、禁止压缩/摘要/截断、超限即失败",
        "gate": "token 必须与 latency / cost 同报，否则结论误导",
        "design_section": "§4.2",
        "note": "原 R2。测模型本身而非 Agent 压缩策略，绝不并入排名。",
    },
]

# Reserved for P2; not part of the six-axis set until real redacted logs and
# a compliance review exist. Kept here so contributors cannot silently add it.
RESERVED_DIMENSIONS = [
    {"id": "G", "name": "用户身份 / 会话边界", "since": "P2", "status": "reserved_not_active"},
]


def dimension_ids() -> list[str]:
    return [item["id"] for item in DIMENSIONS]


def rankable_ids() -> list[str]:
    return [item["id"] for item in DIMENSIONS if item["rankable"]]


def attribution_ids() -> list[str]:
    return [item["id"] for item in DIMENSIONS if item["role"] == "attribution"]


def get_dimension(identifier: str) -> dict[str, Any] | None:
    return next((item for item in DIMENSIONS if item["id"] == identifier), None)


def validate_dimension_set(ids: list[str], *, scores_declared: bool = False) -> list[str]:
    """Return issue strings. Keeps contributor submissions honest about axes.

    Structural validity (known ids, no duplicates, primary present, at least
    one rankable axis) is enforced unconditionally. ``scores_declared`` only
    adds the requirement that a *non-empty* report exists — it is never a
    switch that lets an invalid axis set through when false.
    """
    issues: list[str] = []
    known = dimension_ids()
    if len(set(ids)) != len(ids):
        issues.append("duplicate_dimension_id")
    for identifier in ids:
        if identifier not in known:
            issues.append(f"unknown_dimension:{identifier}")
    # E/F are attribution-only and may be reported on their own, as long as no
    # score is declared. The moment a score is declared, the ranked report must
    # include the primary axis; that is the rule that stops attribution axes
    # being smuggled into the leaderboard.
    if scores_declared:
        if not ids:
            issues.append("scores_declared_but_no_dimension")
        else:
            if PRIMARY_DIMENSION not in ids:
                issues.append("primary_dimension_A_missing_from_ranked_report")
            if not any(get_dimension(i)["rankable"] for i in ids if i in known):
                issues.append("no_rankable_dimension_declared")
    return issues


def registry() -> dict[str, Any]:
    return {
        "dimension_set_version": DIMENSION_SET_VERSION,
        "primary_dimension": PRIMARY_DIMENSION,
        "dimension_n": len(DIMENSIONS),
        "rankable_dimensions": rankable_ids(),
        "attribution_dimensions": attribution_ids(),
        "weighted_total": False,
        "dimensions": DIMENSIONS,
        "reserved": RESERVED_DIMENSIONS,
    }
