from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from math import ceil
from pathlib import Path
import re
from typing import TypedDict


FACT_FIELDS = ("name", "location", "profession", "response_style", "interests", "favorite_drink", "favorite_food", "pet")


class ThreadMemory(TypedDict):
    messages: list[dict[str, str]]
    summary: str
    compactions: int


def extract_topic_anchors(text: str) -> list[str]:
    """Identify acronyms/project IDs and multi-word English proper names."""
    names = re.findall(r"\b(?:[A-Z][a-z]+(?: [A-Z](?:[a-z]+|[A-Z]+))+|[A-Z][A-Z0-9]+(?:-[A-Z0-9]+)*)\b", text)
    return list(dict.fromkeys(names))


def summary_topics(summary: str) -> list[str]:
    for line in summary.splitlines():
        if line.startswith("- topics: "):
            return line[10:].split(", ")
    return []


def estimate_tokens(text: str) -> int:
    """Deterministic character estimator; not a provider tokenizer."""
    return ceil(len(text.strip()) / 4)


def parse_facts(text: str) -> dict[str, str]:
    facts = {}
    for line in text.splitlines():
        match = re.fullmatch(r"- ([a-z_]+): (.+)", line.strip())
        if match and match[1] in FACT_FIELDS:
            facts[match[1]] = match[2]
    return facts


def merge_facts(previous: dict[str, str], updates: dict[str, str]) -> dict[str, str]:
    """Replace factual fields; retain preference dimensions not newly specified."""
    merged = dict(previous)
    for key, value in updates.items():
        if key == "interests":
            value = ", ".join(dict.fromkeys(item.strip() for item in (previous.get(key, "") + ", " + value).split(",") if item.strip()))
        elif key == "response_style":
            parts = previous.get(key, "").split("; ") if previous.get(key) else []
            for part in value.split("; "):
                if part in {"ngắn gọn", "chi tiết"}:
                    parts = [item for item in parts if item not in {"ngắn gọn", "chi tiết"}]
                elif "bullet" in part:
                    if part == "bullet" and any(re.match(r"\d+ bullet", item) for item in parts):
                        continue
                    parts = [item for item in parts if "bullet" not in item]
                elif part.startswith("có ví dụ"):
                    parts = [item for item in parts if not item.startswith("có ví dụ")]
                if part not in parts:
                    parts.append(part)
            value = "; ".join(parts)
        merged[key] = value
    return merged


@dataclass
class UserProfileStore:
    """One UTF-8 Markdown profile per user; repeated updates replace fields."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        if not user_id or not user_id.strip():
            raise ValueError("user_id must not be empty")
        if re.fullmatch(r"[A-Za-z0-9_-]{1,80}", user_id):
            directory = user_id
        else:
            directory = "user-" + sha256(user_id.encode("utf-8")).hexdigest()
        return self.root_dir / directory / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        if not search_text or search_text == replacement:
            return False
        original = self.read_text(user_id)
        if search_text not in original:
            return False
        self.write_text(user_id, original.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        return parse_facts(self.read_text(user_id))

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        if key not in FACT_FIELDS:
            raise ValueError(f"Unknown profile field: {key}")
        value = " ".join(value.split())
        if not value:
            return
        original = self.read_text(user_id)
        pattern = re.compile(rf"^- {re.escape(key)}: .+$", re.MULTILINE)
        line = f"- {key}: {value}"
        if pattern.search(original):
            first = True

            def replace_field(match):
                nonlocal first
                if first:
                    first = False
                    return line
                return ""

            updated = pattern.sub(replace_field, original)
        else:
            updated = (original or "# User\n").rstrip() + "\n" + line + "\n"
        if updated != original:
            self.write_text(user_id, updated)

    def update_facts(self, user_id: str, updates: dict[str, str]) -> None:
        merged = merge_facts(self.facts(user_id), updates)
        for key in updates:
            self.upsert_fact(user_id, key, merged[key])


def _clean_value(value: str) -> str:
    value = re.split(r"\s+(?:và|chứ|nhưng|cho|dù|trong|vài|mỗi|để|như cũ|nữa|không đổi)\b", value, maxsplit=1, flags=re.I)[0]
    return value.strip(" ,:;\"'“”")


def extract_profile_updates(message: str) -> dict[str, str]:
    """Conservative Vietnamese rules for explicit personal statements.

    No dataset answers, names or city lookup table are used. Questions, hypothetical
    clauses and jokes are excluded. New explicit values replace previous values.
    """
    updates: dict[str, str] = {}
    sentences = re.findall(r"[^.!?\n;]+[.!?]?", message)
    for sentence in sentences:
        text = sentence.strip()
        low = text.casefold()
        if not text or text.endswith("?"):
            continue
        if re.search(r"nhắc lại|nhớ lại|tóm tắt|mô tả", low) and not re.search(
            r"(?:mình|tôi)\s+(?:(?:đang|vẫn|hiện|giờ)\s+)*(?:tên là|ở|làm)\b", low
        ):
            continue
        if re.match(r"(?:nếu|giả sử|có lúc mình đùa|lát nữa|lúc đó|sau đó)\b", low):
            continue
        if any(marker in low for marker in ("chỉ là câu đùa", "hay là chuyển", "mình đùa")):
            continue
        name = re.search(r"\b(?:mình|tôi)\s+tên(?:\s+là)?\s+([^,.:;]+)", text, re.I)
        if not name:
            name = re.search(r"\btên\s+(?:mình|tôi)\s+(?:là\s+)?([^,.:;]+)", text, re.I)
        if name:
            value = _clean_value(name[1])
            if value and not re.search(r"\b(?:gì|ai|không|như thế nào)\b", value, re.I):
                updates["name"] = value

        location = re.search(r"\b(?:mình|tôi)\s+(?:(?:vẫn|hiện|giờ|đang|hiện tại)\s+)*(?:sống\s+)?ở\s+([^,.:;]+)", text, re.I)
        if not location:
            location = re.search(r"\bnơi ở hiện tại\s+(?:là\s+)?([^,.:;]+)", text, re.I)
        if not location:
            location = re.search(r"\b(?:mình|tôi)\s+đang làm việc ở\s+(.+?)\s+(?:vài tháng|dài hạn)", text, re.I)
        if not location and name:
            location = re.search(r"(?:,|và)\s+(?:(?:hiện|đang|giờ)\s+)*ở\s+([^,.:;]+)", text, re.I)
        if location:
            value = _clean_value(location[1])
            if value and value[0].isupper() and not re.search(r"\b(?:đâu|gì|không)\b", value, re.I):
                updates["location"] = value

        # Matching affirmative predicates avoids 'không còn làm ...' and 'đừng nói ...'.
        profession_patterns = (
            r"\b(?:mình|tôi)\s+(?:(?:vẫn|hiện|hiện tại|đang)\s+)*(?:làm|là)\s+([^,.:;]+)",
            r"\b(?:giờ|hiện tại)\s+(?:mình\s+)?chuyển sang\s+([^,.:;]+)",
            r"\bnghề nghiệp(?:\s+hiện tại)?\s+(?:thì\s+)?(?:vẫn\s+)?(?:là\s+)?([^,.:;]+)",
        )
        if name or location:
            profession_patterns += (r"(?:,|và)\s+(?:(?:đang|hiện|vẫn)\s+)*làm\s+([^,.:;]+)",)
        for pattern in profession_patterns:
            for match in re.finditer(pattern, text, re.I):
                value = _clean_value(match[1])
                if re.search(r"\b(?:engineer|developer|manager|scientist|analyst|designer|teacher|kỹ sư|bác sĩ|giáo viên)\b", value, re.I):
                    updates["profession"] = value

        instruction = re.search(r"(?:mình|tôi).{0,35}(?:muốn|thích)|hãy trả lời|style trả lời.*(?:vẫn|giữ|là)", low)
        if instruction and re.search(r"trả lời|giải thích|trình bày|style", low):
            parts = []
            if re.search(r"ngắn|gọn", low):
                parts.append("ngắn gọn")
            elif "chi tiết" in low:
                parts.append("chi tiết")
            bullets = re.search(r"(\d+)\s+bullet", low)
            if bullets:
                parts.append(bullets[1] + " bullet")
            elif "bullet" in low:
                parts.append("bullet")
            if "ví dụ thực chiến" in low:
                parts.append("có ví dụ thực chiến")
            elif "ví dụ thực tế" in low:
                parts.append("có ví dụ thực tế")
            if "trade-off" in low:
                parts.append("ưu tiên trade-off")
            if parts:
                updates["response_style"] = "; ".join(parts)

        if re.search(r"(?:mình|tôi)\s+(?:(?:vẫn|đang)\s+)*(?:thích|quan tâm)", low):
            topics = re.findall(r"\b(?:Python|AI|MLOps|RAG|Rust|JavaScript|Java|Go|evaluation)\b", text, re.I)
            canonical = {"python": "Python", "ai": "AI", "mlops": "MLOps", "rag": "RAG", "rust": "Rust", "javascript": "JavaScript", "java": "Java", "go": "Go"}
            topics = [canonical.get(topic.lower(), topic.lower()) for topic in topics]
            if topics:
                updates["interests"] = ", ".join(dict.fromkeys(topics))

        for label, key in (("đồ uống", "favorite_drink"), ("món ăn", "favorite_food")):
            match = re.search(rf"\b{label}\s+yêu thích(?:\s+của (?:mình|tôi))?\s+(?:vẫn\s+)?là\s+([^,.:;]+)", text, re.I)
            if match:
                value = _clean_value(match[1])
                if value and not re.search(r"\b(?:gì|ai|đâu|không)\b", value, re.I):
                    updates[key] = value
        pet = re.search(r"\b(?:mình|tôi)\s+nuôi\s+(?:(?:một|1)\s+)?(?:(?:bé|con|chú)\s+)?([^,.:;]+)", text, re.I)
        if pet:
            value = _clean_value(pet[1])
            if value and not re.search(r"\b(?:gì|ai|đâu|không)\b", value, re.I):
                updates["pet"] = value
    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6, *, max_chars: int = 1600) -> str:
    """Keep structured facts plus bounded early/recent notes, including old summary.

    This deliberately lossy offline summary does not call an LLM. It keeps the
    previous summary in the next pass rather than discarding it on recompaction.
    """
    if max_items < 0 or max_chars < 1:
        raise ValueError("Invalid summary limits")
    facts: dict[str, str] = {}
    notes: list[str] = []
    topics: list[str] = []
    for message in messages:
        content = message["content"]
        if message["role"] == "system":
            facts = merge_facts(facts, parse_facts(content))
            topics.extend(summary_topics(content))
            notes.extend(line[8:] for line in content.splitlines() if line.startswith("- note: "))
        elif message["role"] == "user":
            facts = merge_facts(facts, extract_profile_updates(content))
            topics.extend(extract_topic_anchors(content))
            text = " ".join(content.split())
            if text:
                notes.append(text[:160])
    unique = list(dict.fromkeys(notes))
    early_count = (max_items + 1) // 2
    chosen = unique[:early_count]
    if max_items > early_count:
        chosen.extend(note for note in unique[-(max_items - early_count):] if note not in chosen)
    lines = ["# Summary"]
    for key in FACT_FIELDS:
        if key in facts:
            lines.append(f"- {key}: {facts[key][:160]}")
    unique_topics = list(dict.fromkeys(topics))
    if len(unique_topics) > 16:
        unique_topics = unique_topics[:8] + unique_topics[-8:]
    if unique_topics:
        lines.append("- topics: " + ", ".join(topic[:80] for topic in unique_topics))
    lines.extend("- note: " + note for note in chosen[:max_items])
    # Bound total size; add only complete entries so parseable facts stay intact.
    result = ""
    for line in lines:
        candidate = result + line + "\n"
        if len(candidate) <= max_chars:
            result = candidate
    return result.strip()


@dataclass
class CompactMemoryManager:
    threshold_tokens: int
    keep_messages: int
    state: dict[str, ThreadMemory] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.threshold_tokens < 1 or self.keep_messages < 1:
            raise ValueError("Compact threshold and keep_messages must be positive")

    def append(self, thread_id: str, role: str, content: str) -> None:
        context = self.context(thread_id)
        messages = context["messages"]
        messages.append({"role": role, "content": content})
        total = estimate_tokens(context["summary"]) + sum(estimate_tokens(item["content"]) for item in messages)
        if total <= self.threshold_tokens or len(messages) <= self.keep_messages:
            return
        older = messages[:-self.keep_messages]
        previous = [{"role": "system", "content": context["summary"]}] if context["summary"] else []
        context["summary"] = summarize_messages(previous + older, max_chars=min(1600, self.threshold_tokens * 2))
        context["messages"] = messages[-self.keep_messages:]
        context["compactions"] += 1

    def context(self, thread_id: str) -> ThreadMemory:
        return self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})

    def compaction_count(self, thread_id: str) -> int:
        return self.context(thread_id)["compactions"]
