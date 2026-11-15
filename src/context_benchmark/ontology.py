"""Single source of truth for labels and experimental conditions."""

TYPE_TO_CODE = {
    "standing": "S",
    "one_off": "O",
    "implicit": "I",
    "superseded": "T",
}
CODE_TO_TYPE = {value: key for key, value in TYPE_TO_CODE.items()}
CATEGORIES = {"communication", "process", "domain"}
SCOPES = {"global", "session", "task"}
RULES = tuple(f"R{i}" for i in range(1, 9))
BACKGROUND_MODES = ("full_context", "distilled", "raw_turns", "no_history")
INTERVENTION_ARMS = ("true", "none", "fake")
BUDGETS = ("tight", "medium", "generous")
ACTIVE_TYPES = {"standing", "implicit"}
