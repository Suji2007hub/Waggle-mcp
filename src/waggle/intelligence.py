from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher

from waggle.models import Node, NodeType, RelationType, utc_now

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_QUESTION_PREFIX_RE = re.compile(r"^(why|what|how|when|where|who|which|can|could|should|do|does|did|is|are)\b")
_FILE_PATH_RE = re.compile(r"\b(?:[\w.-]+/)+[\w.-]+\.[A-Za-z0-9]+\b|\b[\w.-]+\.(?:py|ts|tsx|js|jsx|rs|go|java|kt|rb|php|md|json|yaml|yml|toml|sql)\b")
_ENTITY_RE = re.compile(r"\b(?:[A-Z]{2,}[A-Z0-9]*|[A-Z][a-z]+(?:[A-Z][A-Za-z0-9]+)+|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
_PREFERENCE_RE = re.compile(r"\b(?:i\s+)?(?:prefer|like|love|want|favo(?:u)?r|avoid)\b", re.IGNORECASE)
_DECISION_RE = re.compile(r"\b(?:let's use|lets use|we should use|go with|we decided|decided to use|choose|chosen|picked)\b", re.IGNORECASE)
_NEGATION_RE = re.compile(r"\b(?:avoid|not|don't|dont|won't|wont|no longer)\b", re.IGNORECASE)

_ACTION_TOKENS = {
    "prefer",
    "prefers",
    "like",
    "likes",
    "love",
    "loves",
    "want",
    "wants",
    "favorite",
    "favourite",
    "avoid",
    "avoids",
    "use",
    "using",
    "used",
    "choose",
    "chosen",
    "pick",
    "picked",
    "decided",
    "decide",
    "lets",
    "let",
    "should",
    "with",
    "will",
    "go",
}
_DOMAIN_HINT_TOKENS = {
    "api",
    "apis",
    "backend",
    "frontend",
    "database",
    "db",
    "storage",
    "auth",
    "authentication",
    "memory",
    "search",
    "server",
    "framework",
}
_CHOICE_CATEGORY_BY_TOKEN = {
    "rest": "api-style",
    "graphql": "api-style",
    "grpc": "api-style",
    "soap": "api-style",
    "sqlite": "database",
    "postgres": "database",
    "postgresql": "database",
    "mysql": "database",
    "mongodb": "database",
    "neo4j": "database",
    "redis": "database",
    "fastapi": "backend-framework",
    "django": "backend-framework",
    "flask": "backend-framework",
    "express": "backend-framework",
    "nestjs": "backend-framework",
    "react": "frontend-framework",
    "vue": "frontend-framework",
    "svelte": "frontend-framework",
}


@dataclass(frozen=True)
class TemporalQueryHints:
    since: datetime | None = None
    until: datetime | None = None
    recency_mode: str = "default"


def normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def content_token_jaccard(a: str, b: str) -> float:
    """Bag-of-content-words Jaccard similarity, stopwords excluded.

    Returns a value in [0, 1]. A score >= 0.5 strongly indicates the two texts
    are talking about the same thing even when phrased differently.
    Used as a cheap pre-filter before cosine similarity.
    """
    tokens_a = tokenize_text(a)
    tokens_b = tokenize_text(b)
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


# Per-type thresholds reflect semantic risk:
# - decision/preference nodes are high-confidence, narrow-topic — merge aggressively
# - fact nodes carry precise numeric/technical values — merge conservatively
# - concept/entity are structural anchors — be very conservative
_TYPE_DEDUP_THRESHOLD: dict[NodeType, float] = {
    NodeType.DECISION:   0.82,
    NodeType.PREFERENCE: 0.82,
    NodeType.NOTE:       0.85,
    NodeType.QUESTION:   0.88,
    NodeType.FACT:       0.92,
    NodeType.CONCEPT:    0.95,
    NodeType.ENTITY:     0.97,
}


def type_aware_dedup_threshold(node_type: NodeType, *, default: float = 0.97) -> float:
    """Return a per-type cosine similarity threshold for deduplication.

    Lower values merge more aggressively. Higher values are more conservative.
    Falls back to *default* if the type has no explicit mapping.
    """
    return _TYPE_DEDUP_THRESHOLD.get(node_type, default)


def tokenize_text(value: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(normalize_text(value))
        if token and token not in _STOPWORDS and len(token) > 1
    }


def label_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalize_text(left), normalize_text(right)).ratio()


def is_acronym_match(left: str, right: str) -> bool:
    left_tokens = [token for token in _TOKEN_RE.findall(left) if token]
    right_tokens = [token for token in _TOKEN_RE.findall(right) if token]
    if not left_tokens or not right_tokens:
        return False
    left_acronym = "".join(token[0].upper() for token in left_tokens if token[0].isalnum())
    right_acronym = "".join(token[0].upper() for token in right_tokens if token[0].isalnum())
    left_joined = "".join(left_tokens).upper()
    right_joined = "".join(right_tokens).upper()
    return (
        left_joined == right_acronym
        or right_joined == left_acronym
        or left_acronym == right_joined
        or right_acronym == left_joined
    )


def compatible_node_types(incoming: NodeType, existing: NodeType) -> bool:
    if incoming == existing:
        return True
    compatible_groups = (
        {NodeType.ENTITY, NodeType.CONCEPT, NodeType.NOTE},
        {NodeType.FACT, NodeType.NOTE},
        {NodeType.DECISION, NodeType.NOTE},
        {NodeType.PREFERENCE, NodeType.NOTE},
    )
    return any(incoming in group and existing in group for group in compatible_groups)


def infer_node_type(text: str) -> NodeType:
    lowered = normalize_text(text)
    if lowered.endswith("?") or lowered.startswith(("why ", "what ", "how ", "when ", "where ", "who ")):
        return NodeType.QUESTION
    if any(keyword in lowered for keyword in ("prefer", "likes", "wants", "favorite", "favourite", "avoid")):
        return NodeType.PREFERENCE
    if any(keyword in lowered for keyword in ("decided", "decision", "will use", "we will", "choose", "chosen")):
        return NodeType.DECISION
    if any(keyword in lowered for keyword in ("project", "service", "database", "api", "company", "team", "user")):
        return NodeType.ENTITY
    if any(keyword in lowered for keyword in ("means", "concept", "pattern", "approach", "architecture")):
        return NodeType.CONCEPT
    return NodeType.FACT


def infer_label(text: str, max_words: int = 6) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip(" -•\n\t"))
    cleaned = re.split(r"[:.;!?]", cleaned, maxsplit=1)[0].strip()
    words = cleaned.split()
    if not words:
        return "Memory Node"
    if len(words) > max_words:
        words = words[:max_words]
    return " ".join(words)


def split_atomic_items(content: str) -> list[str]:
    stripped = content.strip()
    if not stripped:
        return []

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if len(lines) > 1:
        bullet_like = []
        for line in lines:
            cleaned = _LIST_PREFIX_RE.sub("", line).strip()
            if cleaned:
                bullet_like.append(cleaned)
        if len(bullet_like) > 1:
            return _dedupe_preserve_order(bullet_like)

    sentence_parts = []
    for block in lines or [stripped]:
        sentence_parts.extend(part.strip() for part in _SENTENCE_SPLIT_RE.split(block) if part.strip())
    return _dedupe_preserve_order(sentence_parts or [stripped])


def lexical_overlap(query: str, label: str, content: str) -> float:
    query_tokens = tokenize_text(query)
    if not query_tokens:
        return 0.0
    node_tokens = tokenize_text(f"{label} {content}")
    if not node_tokens:
        return 0.0
    shared = len(query_tokens & node_tokens)
    return shared / max(len(query_tokens), 1)


def score_node(
    *,
    node: Node,
    semantic_similarity: float,
    lexical_score: float,
    max_access: int,
    degree_score: float,
    depth: int,
) -> float:
    age_days = max((utc_now() - node.updated_at).total_seconds() / 86400.0, 0.0)
    recency_score = math.exp(-age_days / 30.0)
    access_score = math.log1p(node.access_count) / math.log1p(max_access) if max_access > 0 else 0.0
    depth_score = 1.0 / (1.0 + depth)
    return (
        (0.38 * semantic_similarity)
        + (0.22 * lexical_score)
        + (0.18 * recency_score)
        + (0.10 * access_score)
        + (0.07 * degree_score)
        + (0.05 * depth_score)
    )


def infer_relationship(source: Node, target: Node, *, shared_tokens: set[str] | None = None) -> RelationType:
    shared_tokens = shared_tokens or (tokenize_text(source.content) & tokenize_text(target.content))
    pair_text = f"{normalize_text(source.content)} {normalize_text(target.content)}"
    if any(keyword in pair_text for keyword in ("depends on", "requires", "blocked by", "needs")):
        return RelationType.DEPENDS_ON
    if any(keyword in pair_text for keyword in ("update", "updated", "refine", "replaces", "supersedes")):
        return RelationType.UPDATES
    if target.node_type == NodeType.CONCEPT and source.node_type != NodeType.CONCEPT:
        return RelationType.PART_OF
    if source.node_type == NodeType.QUESTION or target.node_type == NodeType.QUESTION:
        return RelationType.RELATES_TO
    if shared_tokens:
        return RelationType.RELATES_TO
    return RelationType.DERIVED_FROM


def parse_since_value(value: str, *, now: datetime | None = None) -> datetime:
    now = (now or utc_now()).astimezone(timezone.utc)
    raw = value.strip().lower()
    if not raw:
        raise ValueError("Time range cannot be empty.")

    relative = re.fullmatch(r"(\d+)\s*([smhdw])", raw)
    if relative:
        amount = int(relative.group(1))
        unit = relative.group(2)
        multipliers = {
            "s": timedelta(seconds=amount),
            "m": timedelta(minutes=amount),
            "h": timedelta(hours=amount),
            "d": timedelta(days=amount),
            "w": timedelta(weeks=amount),
        }
        return now - multipliers[unit]

    named = {
        "today": now - timedelta(days=1),
        "yesterday": now - timedelta(days=2),
        "last week": now - timedelta(days=7),
        "this week": now - timedelta(days=7),
        "last month": now - timedelta(days=30),
        "recently": now - timedelta(days=14),
    }
    if raw in named:
        return named[raw]

    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported time range: {value}") from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def infer_temporal_hints(query: str, *, now: datetime | None = None) -> TemporalQueryHints:
    lowered = normalize_text(query)
    now = (now or utc_now()).astimezone(timezone.utc)
    if "last week" in lowered:
        return TemporalQueryHints(since=now - timedelta(days=7), recency_mode="recent")
    if "last month" in lowered:
        return TemporalQueryHints(since=now - timedelta(days=30), recency_mode="recent")
    if "today" in lowered:
        return TemporalQueryHints(since=now - timedelta(days=1), recency_mode="recent")
    if "yesterday" in lowered:
        return TemporalQueryHints(since=now - timedelta(days=2), until=now - timedelta(days=1), recency_mode="recent")
    if any(term in lowered for term in ("latest", "most recent", "newest")):
        return TemporalQueryHints(recency_mode="latest")
    if any(term in lowered for term in ("originally", "initially", "first discussed")):
        return TemporalQueryHints(recency_mode="oldest")
    if any(term in lowered for term in ("recently", "lately", "recent")):
        return TemporalQueryHints(recency_mode="recent")
    return TemporalQueryHints()


def temporal_score_adjustment(node: Node, hints: TemporalQueryHints, *, now: datetime | None = None) -> float:
    now = (now or utc_now()).astimezone(timezone.utc)
    age_days = max((now - node.updated_at).total_seconds() / 86400.0, 0.0)
    if hints.recency_mode in {"recent", "latest"}:
        return 0.12 * math.exp(-age_days / 7.0)
    if hints.recency_mode == "oldest":
        return 0.12 * (1.0 - math.exp(-age_days / 45.0))
    return 0.0


def within_time_window(node: Node, hints: TemporalQueryHints) -> bool:
    if hints.since is None and hints.until is None:
        return True
    timestamps = [node.created_at, node.updated_at]
    for stamp in timestamps:
        if hints.since is not None and stamp < hints.since:
            continue
        if hints.until is not None and stamp > hints.until:
            continue
        return True
    return False


def extract_conversation_candidates(
    *,
    user_message: str,
    assistant_response: str,
) -> list[dict[str, object]]:
    combined = []
    if user_message.strip():
        combined.append(("user", user_message.strip()))
    if assistant_response.strip():
        combined.append(("assistant", assistant_response.strip()))

    seen: set[tuple[str, str, str]] = set()
    candidates: list[dict[str, object]] = []

    for speaker, text in combined:
        sentences = split_atomic_items(text)
        for sentence in sentences:
            node_type = _infer_observed_node_type(sentence)
            if node_type is not None:
                label = sentence.strip() if node_type == NodeType.QUESTION else infer_label(sentence)
                _append_candidate(
                    candidates,
                    seen,
                    label=label,
                    content=sentence,
                    node_type=node_type,
                    tags=["observed", f"speaker:{speaker}"],
                )

            for entity in extract_named_entities(sentence):
                _append_candidate(
                    candidates,
                    seen,
                    label=entity,
                    content=entity,
                    node_type=NodeType.ENTITY,
                    tags=["observed", "named-entity", f"speaker:{speaker}"],
                )

            for path in _FILE_PATH_RE.findall(sentence):
                _append_candidate(
                    candidates,
                    seen,
                    label=path,
                    content=f"Code path mentioned in conversation: {path}",
                    node_type=NodeType.ENTITY,
                    tags=["observed", "code-path", f"speaker:{speaker}"],
                )

    return candidates


def detect_conflict_reason(existing: Node, incoming: Node) -> str | None:
    if existing.id == incoming.id:
        return None
    if incoming.node_type not in {NodeType.PREFERENCE, NodeType.DECISION}:
        return None
    if existing.node_type != incoming.node_type:
        return None

    incoming_focus = extract_focus_tokens(incoming.content)
    existing_focus = extract_focus_tokens(existing.content)
    if not incoming_focus or not existing_focus:
        return None

    incoming_categories = {category for token in incoming_focus if (category := _CHOICE_CATEGORY_BY_TOKEN.get(token))}
    existing_categories = {category for token in existing_focus if (category := _CHOICE_CATEGORY_BY_TOKEN.get(token))}
    shared_categories = incoming_categories & existing_categories
    if shared_categories and incoming_focus != existing_focus:
        category = sorted(shared_categories)[0]
        return f"Conflicting {category} choice"

    if incoming_focus & existing_focus:
        incoming_negative = bool(_NEGATION_RE.search(incoming.content))
        existing_negative = bool(_NEGATION_RE.search(existing.content))
        if incoming_negative != existing_negative:
            return "Opposite stance about the same subject"
        return None

    incoming_context = contextual_tokens(incoming.content, incoming_focus)
    existing_context = contextual_tokens(existing.content, existing_focus)
    if incoming_context & existing_context and len(incoming_focus | existing_focus) <= 4:
        return "Potentially conflicting choice in the same context"
    return None


def summarize_topic(nodes: list[Node]) -> tuple[str, list[str]]:
    tag_counter: Counter[str] = Counter()
    word_counter: Counter[str] = Counter()
    for node in nodes:
        tag_counter.update(node.tags)
        word_counter.update(tokenize_text(node.label))
    top_tags = [tag for tag, _ in tag_counter.most_common(3)]
    if top_tags:
        return ", ".join(top_tags), top_tags
    top_words = [word for word, _ in word_counter.most_common(3)]
    label = " / ".join(top_words) if top_words else "misc-topic"
    return label, []


def extract_named_entities(text: str) -> list[str]:
    ignored = {"api", "apis", "backend", "database", "memory", "project"}
    matches = []
    for match in _ENTITY_RE.finditer(text):
        value = match.group(0).strip()
        if normalize_text(value) in ignored:
            continue
        matches.append(value)
    return _dedupe_preserve_order(matches)


def extract_focus_tokens(text: str) -> set[str]:
    lowered = normalize_text(text)
    patterns = (
        r"(?:prefer|prefers|like|likes|love|loves|want|wants|avoid|avoids)\s+(?P<focus>.+)",
        r"(?:lets use|let's use|use|using|go with|choose|chosen|picked|decided to use)\s+(?P<focus>.+)",
    )
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            return tokenize_text(match.group("focus")) - _ACTION_TOKENS
    return tokenize_text(lowered) - _ACTION_TOKENS


def contextual_tokens(text: str, focus_tokens: set[str]) -> set[str]:
    return (tokenize_text(text) - focus_tokens - _ACTION_TOKENS) | (_DOMAIN_HINT_TOKENS & tokenize_text(text))


def _infer_observed_node_type(text: str) -> NodeType | None:
    stripped = text.strip()
    lowered = normalize_text(stripped)
    if not lowered:
        return None
    if stripped.endswith("?") or _QUESTION_PREFIX_RE.match(lowered):
        return NodeType.QUESTION
    if _PREFERENCE_RE.search(stripped):
        return NodeType.PREFERENCE
    if _DECISION_RE.search(stripped):
        return NodeType.DECISION
    entity_matches = extract_named_entities(stripped)
    if entity_matches or _FILE_PATH_RE.search(stripped):
        return NodeType.ENTITY
    return None


def _append_candidate(
    candidates: list[dict[str, object]],
    seen: set[tuple[str, str, str]],
    *,
    label: str,
    content: str,
    node_type: NodeType,
    tags: list[str],
) -> None:
    key = (normalize_text(label), normalize_text(content), node_type.value)
    if key in seen:
        return
    seen.add(key)
    candidates.append(
        {
            "label": label,
            "content": content,
            "node_type": node_type,
            "tags": tags,
        }
    )


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = normalize_text(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(value)
    return ordered
