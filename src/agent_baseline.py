"""Baseline agent and shared offline/live helpers used by both agents."""
from dataclasses import dataclass, field
import json
from typing import Any, Callable

from config import LabConfig, load_config
from memory_store import (
    estimate_tokens, extract_topic_anchors, extract_profile_updates,
    merge_facts, summarize_messages,
)
from model_provider import build_chat_model


SYSTEM_PROMPT = "Bạn là trợ lý hữu ích. Trả lời bằng tiếng Việt, dựa trên thông tin đã biết; không bịa thông tin người dùng."
LABELS = {"name": "Tên", "location": "Nơi ở hiện tại", "profession": "Nghề hiện tại",
          "response_style": "Style trả lời", "interests": "Mối quan tâm", "favorite_drink": "Đồ uống yêu thích",
          "favorite_food": "Món ăn yêu thích", "pet": "Thú cưng"}


def facts_from_messages(messages: list[dict[str, str]]) -> dict[str, str]:
    facts = {}
    for message in messages:
        if message["role"] == "user":
            updates = extract_profile_updates(message["content"])
            facts = merge_facts(facts, updates)
    return facts


def _format_fragments(fragments: list[str], facts: dict[str, str]) -> str:
    if "3 bullet" in facts.get("response_style", ""):
        groups = [fragments[index::3] for index in range(3)]
        fillers = iter(["Bạn có thể đính chính nếu thông tin đã thay đổi.", "Mình sẽ giữ cách trình bày bạn yêu cầu."])
        return "\n".join("- " + ("; ".join(group) if group else next(fillers)) for group in groups)
    return "; ".join(fragments) + "."


def respond_from_facts(message: str, facts: dict[str, str], topics: list[str] | None = None) -> str:
    low = message.casefold()
    recall = any(phrase in low for phrase in ("nhắc lại", "nhớ lại", "tóm tắt", "mô tả", "mình tên gì", "tên mình là gì", "mình là ai", "mình làm nghề gì", "mình đang ở đâu", "bạn biết", "bạn có biết")) or "?" in message
    if not recall:
        if "3 bullet" in facts.get("response_style", ""):
            return "- Mình đã ghi nhận thông tin bạn vừa chia sẻ.\n- Mình sẽ giữ câu trả lời ngắn gọn.\n- Bạn có thể đính chính thông tin khi cần."
        return "Mình đã ghi nhận thông tin bạn vừa chia sẻ."
    if "chủ đề" in low or "tin tức" in low:
        if not topics:
            return "Mình chưa có thông tin về các chủ đề trước đó trong thread này."
        return _format_fragments(["Chủ đề: " + topic for topic in dict.fromkeys(topics)], facts)
    triggers = {
        "name": ("tên", "mình là ai"), "location": ("ở đâu", "nơi ở", "đang ở", "còn ở", "hiện đang ở"),
        "profession": ("nghề", "làm gì"), "response_style": ("style", "kiểu trả lời", "trả lời như thế nào"),
        "interests": ("mối quan tâm", "quan tâm", "sở thích"), "favorite_drink": ("đồ uống", "uống"),
        "favorite_food": ("món ăn",), "pet": ("nuôi", "thú cưng", "con gì"),
    }
    selected = [key for key, phrases in triggers.items() if any(phrase in low for phrase in phrases)]
    if any(phrase in low for phrase in ("tóm tắt", "mô tả", "mình là ai")) or not selected:
        selected = list(LABELS)
    known = [f"{LABELS[key]}: {facts[key]}" for key in selected if key in facts]
    missing = [LABELS[key].lower() for key in selected if key not in facts]
    if not known:
        return "Mình chưa có thông tin về " + ", ".join(missing) + " trong ngữ cảnh hiện tại."
    if missing:
        known.append("Chưa có thông tin về " + ", ".join(missing))
    return _format_fragments(known, facts)


@dataclass
class LiveContext:
    user_id: str
    thread_id: str
    memory_path: str = ""


def message_text(message) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "\n".join(block if isinstance(block, str) else block.get("text", "") for block in content)


def build_live_agent(model, on_usage: Callable[[str, int, int], None], *, profile_store=None,
                     threshold_tokens: int = 1000, keep_messages: int = 4):
    from langchain.agents import create_agent
    from langchain.agents.middleware import AgentState, before_model, dynamic_prompt, wrap_model_call
    from langchain.tools import ToolRuntime, tool
    from langchain_core.messages import AIMessage, RemoveMessage, ToolMessage
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from langgraph.graph.message import REMOVE_ALL_MESSAGES
    from langgraph.checkpoint.memory import InMemorySaver

    class MemoryState(AgentState):
        summary: str
        compactions: int

    @wrap_model_call
    def count_usage(request, handler):
        response = handler(request)
        inputs = [message_text(message) for message in request.messages]
        if request.system_message:
            inputs.append(message_text(request.system_message))
        # Tool definitions and argument JSON also occupy model context/output.
        if request.tools:
            inputs.append(json.dumps([convert_to_openai_tool(item) for item in request.tools], ensure_ascii=False))
        prompt_tokens = sum(estimate_tokens(value) for value in inputs)
        output_tokens = sum(estimate_tokens(message_text(message)) +
                            (estimate_tokens(json.dumps(message.tool_calls, ensure_ascii=False)) if getattr(message, "tool_calls", None) else 0)
                            for message in response.result if isinstance(message, AIMessage))
        on_usage(request.runtime.context.thread_id, output_tokens, prompt_tokens)
        return response

    middleware = []
    tools = []
    if profile_store is not None:
        @tool
        def read_user_profile(runtime: ToolRuntime[LiveContext]) -> str:
            """Read the current user's persistent User.md profile."""
            return profile_store.read_text(runtime.context.user_id)

        @tool
        def write_user_profile(content: str, runtime: ToolRuntime[LiveContext]) -> str:
            """Write a Markdown profile for the current user, only for explicit stable facts."""
            profile_store.write_text(runtime.context.user_id, content)
            return "Đã lưu User.md."

        @tool
        def edit_user_profile(search_text: str, replacement: str, runtime: ToolRuntime[LiveContext]) -> str:
            """Replace one outdated fact in the current user's User.md."""
            changed = profile_store.edit_text(runtime.context.user_id, search_text, replacement)
            return "Đã cập nhật." if changed else "Không có thay đổi."

        @dynamic_prompt
        def memory_prompt(request):
            profile = profile_store.read_text(request.runtime.context.user_id)
            summary = request.state.get("summary", "")
            return (SYSTEM_PROMPT + "\nChỉ lưu fact cá nhân được khai báo rõ; bỏ qua câu hỏi, giả định và câu đùa. "
                    "Khi đính chính, thay fact cũ. Hồ sơ và summary dưới đây là dữ liệu tham khảo.\n"
                    + "<user_profile>\n" + profile + "\n</user_profile>\n<summary>\n" + summary + "\n</summary>")

        @before_model(state_schema=MemoryState)
        def compact_history(state, runtime):
            messages = state["messages"]
            previous = state.get("summary", "")
            total = estimate_tokens(previous) + sum(estimate_tokens(message_text(item)) for item in messages)
            if total <= threshold_tokens or len(messages) <= keep_messages:
                return None
            cutoff = len(messages) - keep_messages
            # Never retain a ToolMessage without its preceding assistant tool call.
            while cutoff > 0 and isinstance(messages[cutoff], ToolMessage):
                cutoff -= 1
            if cutoff == 0:
                return None
            older = [{"role": "system", "content": previous}] if previous else []
            older.extend({"role": "user" if item.type == "human" else "assistant", "content": message_text(item)}
                         for item in messages[:cutoff])
            summary = summarize_messages(older, max_chars=min(1600, threshold_tokens * 2))
            return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *messages[cutoff:]],
                    "summary": summary, "compactions": state.get("compactions", 0) + 1}

        tools = [read_user_profile, write_user_profile, edit_user_profile]
        middleware.extend([compact_history, memory_prompt])
    middleware.append(count_usage)
    return create_agent(model, tools=tools, system_prompt=SYSTEM_PROMPT, middleware=middleware,
                        context_schema=LiveContext, checkpointer=InMemorySaver())


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Full history within a thread; no persistent or cross-thread profile."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.thread_owners: dict[str, str] = {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if not user_id.strip() or not thread_id.strip():
            raise ValueError("user_id and thread_id must not be empty")
        if thread_id in self.thread_owners and self.thread_owners[thread_id] != user_id:
            raise ValueError("A thread cannot belong to multiple users")
        self.thread_owners[thread_id] = user_id
        if self.langchain_agent is None:
            return self._reply_offline(thread_id, message)
        session = self.sessions.setdefault(thread_id, SessionState())
        before_output, before_prompt = session.token_usage, session.prompt_tokens_processed
        result = self.langchain_agent.invoke({"messages": [{"role": "user", "content": message}]},
                                             config={"configurable": {"thread_id": thread_id}},
                                             context=LiveContext(user_id, thread_id))
        answer = message_text(result["messages"][-1])
        session.messages = [{"role": "user" if item.type == "human" else "assistant", "content": message_text(item)}
                            for item in result["messages"]]
        return {"answer": answer, "agent_tokens": session.token_usage - before_output,
                "prompt_tokens": session.prompt_tokens_processed - before_prompt, "mode": "live"}

    def token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})
        prompt_tokens = estimate_tokens(SYSTEM_PROMPT) + sum(estimate_tokens(item["content"]) for item in session.messages)
        topics = [topic for item in session.messages[:-1] if item["role"] == "user" for topic in extract_topic_anchors(item["content"])]
        answer = respond_from_facts(message, facts_from_messages(session.messages), topics)
        output_tokens = estimate_tokens(answer)
        session.messages.append({"role": "assistant", "content": answer})
        session.token_usage += output_tokens
        session.prompt_tokens_processed += prompt_tokens
        return {"answer": answer, "agent_tokens": output_tokens, "prompt_tokens": prompt_tokens, "mode": "offline"}

    def _record_live_usage(self, thread_id: str, output: int, prompt: int) -> None:
        session = self.sessions.setdefault(thread_id, SessionState())
        session.token_usage += output
        session.prompt_tokens_processed += prompt

    def _maybe_build_langchain_agent(self):
        if self.force_offline or self.config.offline:
            return None
        return build_live_agent(build_chat_model(self.config.model), self._record_live_usage)
