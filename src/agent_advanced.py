from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates, extract_topic_anchors, merge_facts, parse_facts, summary_topics
from model_provider import build_chat_model
from agent_baseline import (
    SYSTEM_PROMPT, LiveContext, build_live_agent, facts_from_messages,
    message_text, respond_from_facts,
)


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Thread-local recent messages/summary plus a disk-backed user profile."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(self.config.compact_threshold_tokens, self.config.compact_keep_messages)
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.thread_owners: dict[str, str] = {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if not user_id.strip() or not thread_id.strip():
            raise ValueError("user_id and thread_id must not be empty")
        if thread_id in self.thread_owners and self.thread_owners[thread_id] != user_id:
            raise ValueError("A thread cannot belong to multiple users")
        self.thread_owners[thread_id] = user_id
        if self.langchain_agent is None:
            return self._reply_offline(user_id, thread_id, message)
        self.profile_store.update_facts(user_id, extract_profile_updates(message))
        before_output, before_prompt = self.token_usage(thread_id), self.prompt_token_usage(thread_id)
        result = self.langchain_agent.invoke({"messages": [{"role": "user", "content": message}]},
                                             config={"configurable": {"thread_id": thread_id}},
                                             context=LiveContext(user_id, thread_id, str(self.profile_store.path_for(user_id))))
        answer = message_text(result["messages"][-1])
        self.compact_memory.state[thread_id] = {
            "messages": [{"role": "user" if item.type == "human" else "assistant", "content": message_text(item)}
                         for item in result["messages"]],
            "summary": result.get("summary", ""), "compactions": result.get("compactions", 0),
        }
        return {"answer": answer, "agent_tokens": self.token_usage(thread_id) - before_output,
                "prompt_tokens": self.prompt_token_usage(thread_id) - before_prompt, "mode": "live"}

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        self.profile_store.update_facts(user_id, extract_profile_updates(message))
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        answer = self._offline_response(user_id, thread_id, message)
        output_tokens = estimate_tokens(answer)
        self.compact_memory.append(thread_id, "assistant", answer)
        self._record_live_usage(thread_id, output_tokens, prompt_tokens)
        return {"answer": answer, "agent_tokens": output_tokens, "prompt_tokens": prompt_tokens, "mode": "offline"}

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        context = self.compact_memory.context(thread_id)
        return (estimate_tokens(SYSTEM_PROMPT) + estimate_tokens(self.profile_store.read_text(user_id)) +
                estimate_tokens(context["summary"]) + sum(estimate_tokens(item["content"]) for item in context["messages"]))

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        context = self.compact_memory.context(thread_id)
        facts = merge_facts(parse_facts(context["summary"]), facts_from_messages(context["messages"]))
        # The disk profile is authoritative for corrected facts; a summary can be older.
        facts = merge_facts(facts, self.profile_store.facts(user_id))
        topics = summary_topics(context["summary"])
        topics.extend(topic for item in context["messages"][:-1] if item["role"] == "user" for topic in extract_topic_anchors(item["content"]))
        return respond_from_facts(message, facts, topics)

    def _record_live_usage(self, thread_id: str, output: int, prompt: int) -> None:
        self.thread_tokens[thread_id] = self.token_usage(thread_id) + output
        self.thread_prompt_tokens[thread_id] = self.prompt_token_usage(thread_id) + prompt

    def _maybe_build_langchain_agent(self):
        if self.force_offline or self.config.offline:
            return None
        return build_live_agent(build_chat_model(self.config.model), self._record_live_usage,
                                profile_store=self.profile_store, threshold_tokens=self.config.compact_threshold_tokens,
                                keep_messages=self.config.compact_keep_messages)
