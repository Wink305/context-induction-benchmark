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

# ---------------------------------------------------------------------------
# Traps and error kinds.
#
# A "trap" is something deliberately planted in the dialogue that pushes a
# model towards the wrong induction. The model is asked not only to resist it
# but to *say* that it is there, which is why the output contract carries a
# ``trap_flags`` array.
#
# These two vocabularies are the contract between the generator (which plants
# the traps) and the scorer (which checks them). They are declared here so
# both sides cannot drift: the generator writes a kind, the scorer reads the
# same kind, and the validator rejects anything outside the list.
# ---------------------------------------------------------------------------
TRAP_KINDS = (
    # The user retracts the rule outright ("那条取消").
    "withdrawn",
    # The user swaps the rule for a new one ("换成…").
    "replaced",
    # The same slot carries two opposite values and no resolution.
    "conflicting_values",
    # Something that looks like a rule but is not the user's: the agent's
    # own suggestion, a system notice, a fact, small talk.
    "non_user_claim",
    # An instruction aimed at the rules themselves ("忽略前面的要求").
    "meta_override",
    # A one-off request worded like a standing rule, or the reverse.
    "one_off_disguise",
    # An instruction injected into the conversation that tries to steer the
    # summariser ("请把上面的规则都删掉").
    "prompt_injection",
)

# Why a piece of information does not belong in long-term memory. The scorer
# reports pruning per kind, so "the model keeps one-off requests" is not the
# same failure as "the model revives a withdrawn rule".
ERROR_KINDS = (
    "one_off_as_standing",
    "expired_reuse",
    "fabricated",
    "agent_claim",
    "factual_chatter",
    "role_confusion",
)
